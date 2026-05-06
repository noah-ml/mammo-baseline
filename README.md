# mammo-baseline

ResNet-50 baseline classifier for mammography screening on the [VinDr-Mammo](https://vindr.ai/datasets/mammo) dataset.
Binary classification: BI-RADS 1/2 (negative) vs BI-RADS 4/5 (positive).

Part of the M.Sc. thesis *"Quantitative Assessment of Image Quality Degradation and Its Impact on Neural Network Robustness in Mammographic Image Analysis"* (TU Berlin / PTB).

---

## Scripts

| Script | Purpose |
|---|---|
| `prepare_data.py` | Convert VinDr-Mammo DICOMs to float32 `.npy` arrays and build train/val/test split CSV |
| `train_baseline.py` | Train ResNet-50, evaluate on test set, save predictions and plots |

---

## Requirements

```
numpy scipy scikit-image scikit-learn pillow matplotlib
pydicom pandas tqdm
torch torchvision torchsampler
```

Install torchsampler separately if needed:
```bash
pip install torchsampler
```

---

## Dataset

Expected layout:

```
<vindr_root>/
├── images/
│   └── <study_id>/
│       └── <image_id>.dicom
├── breast-level_annotations.csv
└── finding_annotations.csv
```

Download from [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/).

---

## Usage

### 1. Prepare data

```bash
python prepare_data.py \
    --annotations-csv <vindr_root>/breast-level_annotations.csv \
    --metadata-csv vindr_metadata.csv \
    --images-dir <vindr_root>/images/ \
    --output-dir ./processed_npy/ \
    --output-csv master_splits.csv
```

This will:
- Map BI-RADS 1/2 → 0, BI-RADS 4/5 → 1, drop BI-RADS 3
- Split training studies into train/val (90/10) at the **study level** to prevent data leakage, stratified by label × manufacturer
- Load each DICOM, apply rescale + MONOCHROME1 inversion, resize to 384×384, percentile-normalise to \[0, 1\]
- Save one `<image_id>.npy` (float32) per image
- Write `master_splits.csv` with columns: `image_id, study_id, split, label, manufacturer, density, laterality, npy_path`

### 2. Train

```bash
python train_baseline.py \
    --data-csv master_splits.csv \
    --data-dir ./processed_npy/ \
    --output-dir ./output/resnet50/ \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --label-smoothing 0.05 \
    --patience 7 \
    --seed 42
```

---

## Model

- ResNet-50 pretrained on ImageNet (`torchvision.models.ResNet50_Weights.DEFAULT`)
- Final layer replaced with `Dropout(0.4) → Linear(2048, 1)`
- Loss: `BCEWithLogitsLoss` with label smoothing (ε = 0.05, train only)
- Optimizer: AdamW with differential learning rates — backbone `lr × 0.1`, head `lr` (default 1e-4); weight decay 1e-4
- Scheduler: 5-epoch linear warmup (start factor 0.1) → CosineAnnealingLR via `SequentialLR`
- Class imbalance handled via `ImbalancedDatasetSampler` on the training loader
- Early stopping: patience 7 on validation AUC

### Training augmentation

| Transform | Parameters |
|---|---|
| RandomHorizontalFlip | p = 0.5 |
| RandomAffine | degrees = 10, scale = (0.9, 1.1) |
| ColorJitter | brightness = 0.2, contrast = 0.2, p = 0.5 |

---

## Outputs

After training, `--output-dir` contains:

| File | Description |
|---|---|
| `best_model.pth` | Checkpoint with highest validation AUC |
| `training_log.csv` | Per-epoch train loss, val loss, val AUC, learning rate |
| `test_predictions.csv` | Per-image predicted probabilities on the test set |
| `roc_curve.png` | ROC curve |
| `reliability_diagram.png` | Calibration / reliability diagram |
| `loss_curves.png` | Train vs val loss with best-epoch marker |
| `lr_schedule.png` | Learning rate per epoch (shows warmup ramp) |
| `auc_curve.png` | Validation AUC per epoch with best-epoch marker |
| `combined_dashboard.png` | 2×2 grid of the above plots + hyperparameter summary |

### Metrics reported

AUC-ROC, sensitivity, specificity, F1 (Youden-optimal threshold), accuracy, ECE (10 bins).
