#!/usr/bin/env python
"""
prepare_data.py
===============

Converts VinDr-Mammo DICOMs to training-ready float32 .npy arrays and
produces a master split CSV for use with train_baseline.py.

Pipeline
--------
1. Read breast-level_annotations.csv; map BI-RADS 1/2 → 0, BI-RADS 4/5 → 1,
   drop BI-RADS 3 images entirely.
2. Merge manufacturer info from vindr_metadata.csv (per image_id).
3. Honour the official train/test split column; further divide training studies
   into train/val (90/10 by default) at the study_id level, stratified jointly
   by (class label × manufacturer). Falls back to label-only stratification if
   any (label, manufacturer) stratum is too small to split.
4. For each image: load DICOM → apply RescaleSlope/Intercept → invert
   MONOCHROME1 → resize to 384 × 384 → percentile-normalise to [0, 1] →
   save as float32 .npy.
5. Write master_splits.csv: image_id, study_id, split, label, manufacturer,
   density, laterality, npy_path.

Usage
-----
    python prepare_data.py \
        --annotations-csv <vindr_root>/breast-level_annotations.csv \
        --metadata-csv vindr_metadata.csv \
        --images-dir <vindr_root>/images/ \
        --output-dir ./processed_npy/ \
        --output-csv master_splits.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Default paths — update if your layout differs
# ---------------------------------------------------------------------------
VINDR_ROOT = Path(
    r"D:\Mammo\vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0"
)
HERE = Path(__file__).resolve().parent


# BI-RADS string → binary label (None = excluded)
BIRADS_LABEL_MAP: dict[str, int | None] = {
    "BI-RADS 1": 0,
    "BI-RADS 2": 0,
    "BI-RADS 3": None,
    "BI-RADS 4": 1,
    "BI-RADS 5": 1,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare VinDr-Mammo DICOMs for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--annotations-csv", type=Path,
        default=VINDR_ROOT / "breast-level_annotations.csv",
        help="breast-level_annotations.csv from VinDr-Mammo root",
    )
    p.add_argument(
        "--metadata-csv", type=Path,
        default=HERE / "vindr_metadata.csv",
        help="vindr_metadata.csv with image_id → Manufacturer mapping",
    )
    p.add_argument(
        "--images-dir", type=Path,
        default=VINDR_ROOT / "images",
        help="Root images dir containing <study_id>/<image_id>.dicom",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=HERE / "processed_npy",
        help="Directory where .npy files are saved",
    )
    p.add_argument(
        "--output-csv", type=Path,
        default=HERE / "master_splits.csv",
        help="Output master CSV with split assignments and metadata",
    )
    p.add_argument(
        "--val-fraction", type=float, default=0.10,
        help="Fraction of training study_ids reserved for validation",
    )
    p.add_argument(
        "--target-size", type=int, default=384,
        help="Output image size (square)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading and labelling
# ---------------------------------------------------------------------------

def load_annotations(csv_path: Path) -> pd.DataFrame:
    """
    Read breast-level_annotations.csv and apply binary labelling.

    Returns a DataFrame with an added 'label' column (int). Rows with
    BI-RADS 3 (label = None) are dropped.
    """
    df = pd.read_csv(csv_path)
    n_before = len(df)
    df["label"] = df["breast_birads"].map(BIRADS_LABEL_MAP)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    n_dropped = n_before - len(df)
    print(f"  Annotations: {n_before} rows loaded, "
          f"{n_dropped} BI-RADS 3 dropped → {len(df)} remaining")
    return df


def merge_manufacturer(df: pd.DataFrame, metadata_csv: Path) -> pd.DataFrame:
    """Join manufacturer info from vindr_metadata.csv onto the annotation frame."""
    meta = (
        pd.read_csv(metadata_csv)[["image_id", "Manufacturer"]]
        .rename(columns={"Manufacturer": "manufacturer"})
    )
    df = df.merge(meta, on="image_id", how="left")
    df["manufacturer"] = df["manufacturer"].fillna("Unknown")
    return df


# ---------------------------------------------------------------------------
# Train / val split (study-level, stratified)
# ---------------------------------------------------------------------------

def _try_stratified_split(
    study_info: pd.DataFrame,
    stratum_col: str,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Attempt a StratifiedShuffleSplit on study_info using stratum_col.

    Returns (train_study_ids, val_study_ids) or raises ValueError if
    the stratum has too few samples.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    idx_all = np.arange(len(study_info))
    strata = study_info[stratum_col].values
    train_idx, val_idx = next(sss.split(idx_all, strata))
    train_ids = study_info.iloc[train_idx]["study_id"].tolist()
    val_ids = study_info.iloc[val_idx]["study_id"].tolist()
    return train_ids, val_ids


def make_train_val_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """
    Assign val/train sub-splits to the training portion of df.

    Splits at study_id level to prevent data leakage. Stratifies jointly by
    (label × manufacturer); falls back to label-only if any stratum is a
    singleton; falls back to random if that still fails.
    """
    train_mask = df["split"] == "training"
    train_df = df[train_mask]

    # One row per study: label = 1 if any image in the study is positive
    study_info = (
        train_df.groupby("study_id")
        .agg(label=("label", "max"), manufacturer=("manufacturer", "first"))
        .reset_index()
    )
    study_info["stratum_joint"] = (
        study_info["label"].astype(str) + "_" + study_info["manufacturer"]
    )
    study_info["stratum_label"] = study_info["label"].astype(str)

    # Try joint stratification first, fall back progressively
    val_study_ids: set[str] = set()
    for stratum_col, desc in [
        ("stratum_joint",  "label × manufacturer"),
        ("stratum_label",  "label only"),
    ]:
        min_count = study_info[stratum_col].value_counts().min()
        if min_count < 2:
            print(f"  Skipping {desc} stratification (smallest stratum = {min_count})")
            continue
        try:
            _, val_ids = _try_stratified_split(study_info, stratum_col, val_fraction, seed)
            val_study_ids = set(val_ids)
            print(f"  Split strategy: {desc}")
            break
        except ValueError as e:
            print(f"  {desc} split failed ({e}); trying fallback")

    if not val_study_ids:
        # Last resort: plain random split
        rng = np.random.default_rng(seed)
        all_ids = study_info["study_id"].tolist()
        n_val = max(1, int(len(all_ids) * val_fraction))
        val_study_ids = set(rng.choice(all_ids, size=n_val, replace=False).tolist())
        print("  Split strategy: random (all fallbacks failed)")

    n_val = len(val_study_ids)
    n_train = len(study_info) - n_val
    print(f"  Train studies: {n_train}, Val studies: {n_val}")

    df = df.copy()
    df.loc[train_mask & df["study_id"].isin(val_study_ids), "split"] = "val"
    df.loc[train_mask & ~df["study_id"].isin(val_study_ids), "split"] = "train"
    return df


# ---------------------------------------------------------------------------
# DICOM preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess_dicom(dicom_path: Path, target_size: int = 384) -> np.ndarray:
    """
    Load a DICOM and return a float32 [0, 1] array of shape (target_size, target_size).

    Steps:
    - Apply RescaleSlope / RescaleIntercept if present.
    - Invert MONOCHROME1 images (bright = tissue convention).
    - Resize to target_size × target_size (bilinear, anti-aliasing).
    - Clip and scale to [0, 1] using 0.5th / 99.5th percentiles.
    """
    ds = pydicom.dcmread(str(dicom_path))
    image = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    if getattr(ds, "PhotometricInterpretation", "").strip().upper() == "MONOCHROME1":
        image = np.max(image) - image

    image = resize(
        image,
        (target_size, target_size),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    p_low = float(np.percentile(image, 0.5))
    p_high = float(np.percentile(image, 99.5))
    if p_high > p_low:
        image = np.clip(image, p_low, p_high)
        image = (image - p_low) / (p_high - p_low)
    else:
        image = np.zeros_like(image)

    return image.astype(np.float32)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_distribution(df: pd.DataFrame) -> None:
    """Print class, manufacturer, and density distributions per split."""
    print()
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if sub.empty:
            continue
        n_pos = (sub["label"] == 1).sum()
        n_neg = (sub["label"] == 0).sum()
        print(f"  [{split.upper()}]  {len(sub)} images  "
              f"(pos={n_pos}, neg={n_neg}, "
              f"pos_rate={n_pos / max(1, len(sub)):.1%})")
        mfr_counts = sub["manufacturer"].value_counts().to_dict()
        print(f"    Manufacturer : {mfr_counts}")
        if "breast_density" in sub.columns:
            den_counts = sub["breast_density"].value_counts().to_dict()
            print(f"    Density      : {den_counts}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Build split table ---
    print("Loading annotations ...")
    df = load_annotations(args.annotations_csv)

    print("Merging manufacturer metadata ...")
    df = merge_manufacturer(df, args.metadata_csv)

    print("Building train / val / test splits ...")
    df = make_train_val_split(df, val_fraction=args.val_fraction, seed=args.seed)

    print_distribution(df)

    # --- Process DICOMs ---
    print(f"Processing {len(df)} DICOMs → {args.output_dir}")
    records: list[dict] = []
    errors: list[dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), unit="img"):
        image_id: str = row["image_id"]
        study_id: str = row["study_id"]
        dicom_path = args.images_dir / study_id / f"{image_id}.dicom"
        npy_path = args.output_dir / f"{image_id}.npy"

        try:
            image = load_and_preprocess_dicom(dicom_path, target_size=args.target_size)
            np.save(str(npy_path), image)
            records.append({
                "image_id":     image_id,
                "study_id":     study_id,
                "split":        row["split"],
                "label":        int(row["label"]),
                "manufacturer": row["manufacturer"],
                "density":      row.get("breast_density", ""),
                "laterality":   row.get("laterality", ""),
                "npy_path":     str(npy_path),
            })
        except Exception as exc:
            errors.append({"image_id": image_id, "error": str(exc)})
            tqdm.write(f"  ERROR {image_id}: {exc}")

    # --- Save master CSV ---
    master_df = pd.DataFrame(records)
    master_df.to_csv(args.output_csv, index=False)

    print(f"\nMaster CSV saved : {args.output_csv}  ({len(master_df)} rows)")
    if errors:
        err_path = args.output_csv.with_name(args.output_csv.stem + "_errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"Errors ({len(errors)}) logged to : {err_path}")

    print_distribution(master_df)
    print("Done.")


if __name__ == "__main__":
    main()
