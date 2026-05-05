#!/usr/bin/env python
"""
train_baseline.py
=================

ResNet-50 baseline classifier for VinDr-Mammo (binary: BI-RADS 1/2 vs 4/5).
Raw PyTorch, no Lightning.

Usage
-----
    python train_baseline.py \
        --data-csv master_splits.csv \
        --data-dir ./processed_npy/ \
        --output-dir ./output/resnet50/ \
        --epochs 50 \
        --batch-size 32 \
        --lr 1e-4 \
        --seed 42

Training produces:
  - best_model.pth          — checkpoint with highest validation AUC
  - training_log.csv        — per-epoch train_loss / val_loss / val_auc
  - test_predictions.csv    — image-level probabilities on the test set
  - roc_curve.png
  - reliability_diagram.png
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler
from torchvision import models, transforms
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MammoDataset(Dataset):
    """
    Loads float32 .npy mammogram patches and returns (image_tensor, label).

    Images are stored as (H, W) float32 arrays in [0, 1]. They are converted
    to (3, H, W) tensors by repeating the single channel three times so that
    pretrained ImageNet ResNet weights can be used.

    Training augmentation:
    - RandomHorizontalFlip (p = 0.5)
    - RandomAffine (degrees = 10, scale = (0.9, 1.1))
    - ColorJitter (brightness = 0.2, contrast = 0.2, p = 0.5)
      Applied via uint8 PIL conversion to match mammo-net behaviour.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        is_train: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.is_train = is_train

        self._affine = transforms.RandomAffine(degrees=10, scale=(0.9, 1.1))
        self._color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def get_labels(self) -> list[int]:
        """Return all labels; required by ImbalancedDatasetSampler."""
        return self.df["label"].tolist()

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        npy_path = self.data_dir / f"{row['image_id']}.npy"
        image = np.load(str(npy_path))  # (H, W) float32 [0, 1]

        if self.is_train:
            image = self._augment(image)

        # (H, W) → (3, H, W)
        tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Apply training augmentations; returns float32 [0, 1] array."""
        # Horizontal flip
        if random.random() < 0.5:
            image = np.fliplr(image).copy()

        # Affine via PIL (uint8 for clean interpolation)
        pil_img = Image.fromarray((image * 255).astype(np.uint8), mode="L")
        pil_img = self._affine(pil_img)

        # ColorJitter with p = 0.5; uint8 → jitter → float (mammo-net trick)
        if random.random() < 0.5:
            pil_img = self._color_jitter(pil_img)

        return np.array(pil_img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(device: torch.device) -> nn.Module:
    """
    ResNet-50 pretrained on ImageNet with the final layer replaced by a
    single-logit output for binary classification.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 1)
    return model.to(device)


# ---------------------------------------------------------------------------
# Training / validation helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch; returns mean loss."""
    model.train()
    total_loss = 0.0
    n = 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(images)
        n += len(images)

    return total_loss / n


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one validation epoch; returns (mean_loss, AUC)."""
    model.eval()
    total_loss = 0.0
    all_labels: list[float] = []
    all_probs: list[float] = []
    n = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val  ", leave=False):
            images = images.to(device)
            labels_gpu = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels_gpu)

            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            total_loss += loss.item() * len(images)
            n += len(images)
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.tolist())

    val_loss = total_loss / n
    # Guard against single-class batches (can happen on tiny val sets)
    if len(set(all_labels)) < 2:
        val_auc = float("nan")
    else:
        val_auc = roc_auc_score(all_labels, all_probs)

    return val_loss, val_auc


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) with equal-width bins.

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def find_youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the probability threshold maximising the Youden index (TPR − FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    return float(thresholds[np.argmax(youden)])


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    df: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Run inference on a DataLoader and compute all test metrics.

    Returns (metrics_dict, predictions_df).
    predictions_df columns: image_id, study_id, true_label,
                             predicted_prob, manufacturer, density.
    """
    model.eval()
    all_labels: list[float] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  test "):
            logits = model(images.to(device))
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    auc = roc_auc_score(y_true, y_prob)
    threshold = find_youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    # Sensitivity = TPR; specificity = TNR
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    metrics = {
        "auc":         auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "ece":         compute_ece(y_prob, y_true),
        "threshold":   threshold,
    }

    pred_df = df.copy()
    pred_df["true_label"]     = y_true
    pred_df["predicted_prob"] = y_prob
    keep = ["image_id", "study_id", "true_label", "predicted_prob",
            "manufacturer", "density"]
    pred_df = pred_df[[c for c in keep if c in pred_df.columns]]
    return metrics, pred_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc: float,
    save_path: Path,
) -> None:
    """Save an ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test Set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ece: float,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    """Save a reliability diagram showing calibration quality."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_conf: list[float] = []
    frac_pos: list[float] = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        mean_conf.append(float(y_prob[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(
        mean_conf, frac_pos,
        width=1.0 / n_bins * 0.9,
        align="center",
        alpha=0.7,
        label=f"ECE = {ece:.3f}",
    )
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Reliability Diagram (Test Set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train ResNet-50 baseline on VinDr-Mammo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-csv",    type=Path, required=True,
                   help="master_splits.csv from prepare_data.py")
    p.add_argument("--data-dir",    type=Path, required=True,
                   help="Directory containing <image_id>.npy files")
    p.add_argument("--output-dir",  type=Path, default=Path("./output/resnet50"))
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight-decay",type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=7,
                   help="Early stopping patience (epochs without val AUC improvement)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--num-workers", type=int,   default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load split CSV ---
    df_all = pd.read_csv(args.data_csv)
    df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
    df_val   = df_all[df_all["split"] == "val"].reset_index(drop=True)
    df_test  = df_all[df_all["split"] == "test"].reset_index(drop=True)
    print(f"Split sizes — train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")

    # --- Datasets ---
    ds_train = MammoDataset(df_train, args.data_dir, is_train=True)
    ds_val   = MammoDataset(df_val,   args.data_dir, is_train=False)
    ds_test  = MammoDataset(df_test,  args.data_dir, is_train=False)

    loader_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        sampler=ImbalancedDatasetSampler(ds_train),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    loader_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- Model, loss, optimizer, scheduler ---
    model     = build_model(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # --- Training loop ---
    checkpoint_path = args.output_dir / "best_model.pth"
    best_val_auc   = -1.0
    patience_count = 0
    log_rows: list[dict] = []

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(model, loader_train, criterion, optimizer, device)
        val_loss, val_auc = validate_epoch(model, loader_val, criterion, device)
        scheduler.step()

        auc_str = f"{val_auc:.4f}" if not math.isnan(val_auc) else "nan"
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={auc_str}")

        log_rows.append({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_auc":    val_auc,
        })

        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_count = 0
            torch.save(
                {
                    "epoch":               epoch + 1,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc":             val_auc,
                },
                checkpoint_path,
            )
            print(f"  ✓ New best val AUC: {best_val_auc:.4f} — checkpoint saved")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\nEarly stopping triggered (no improvement for {args.patience} epochs)")
                break

    # Save training log
    log_csv = args.output_dir / "training_log.csv"
    pd.DataFrame(log_rows).to_csv(log_csv, index=False)
    print(f"\nTraining log saved: {log_csv}")

    # --- Test evaluation ---
    print("\nLoading best checkpoint for test evaluation ...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics, pred_df = evaluate(model, loader_test, device, df_test)

    pred_csv = args.output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    # Plots
    y_true = pred_df["true_label"].values
    y_prob = pred_df["predicted_prob"].values
    plot_roc_curve(
        y_true, y_prob, metrics["auc"],
        args.output_dir / "roc_curve.png",
    )
    plot_reliability_diagram(
        y_true, y_prob, metrics["ece"],
        args.output_dir / "reliability_diagram.png",
    )

    # Print metrics
    print("\n" + "=" * 50)
    print("  TEST SET RESULTS")
    print("=" * 50)
    print(f"  AUC-ROC       : {metrics['auc']:.4f}")
    print(f"  Sensitivity   : {metrics['sensitivity']:.4f}  (Youden threshold = {metrics['threshold']:.3f})")
    print(f"  Specificity   : {metrics['specificity']:.4f}")
    print(f"  F1 Score      : {metrics['f1']:.4f}")
    print(f"  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  ECE (10 bins) : {metrics['ece']:.4f}")
    print("=" * 50)
    print(f"\n  Predictions   : {pred_csv}")
    print(f"  ROC curve     : {args.output_dir / 'roc_curve.png'}")
    print(f"  Reliability   : {args.output_dir / 'reliability_diagram.png'}")
    print(f"  Best checkpoint: {checkpoint_path}  (val AUC = {best_val_auc:.4f})")


if __name__ == "__main__":
    main()
