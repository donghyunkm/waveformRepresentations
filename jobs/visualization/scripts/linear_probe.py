"""Linear probe: logistic regression on frozen pretrained embeddings.

Extracts mean-pooled embeddings from balanced subsamples of train/val/test,
fits logistic regression on train, and saves prediction tensors as .pt dicts
in predictions/ matching the downstream probe output format.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_JEPA_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "jepa" / "scripts"
if str(_JEPA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_JEPA_SCRIPTS))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, ForecastingDataset
from physiojepa.data_preprocessing import zarr_record_name
from physiojepa.jepa import JEPASimpleLightning
from physiojepa.patchtst import PatchTFTSimpleLightning

from pipeline_common import load_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
NUM_WORKERS = 8
RANDOM_STATE = 42

JEPA_CHECKPOINT = (
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/"
    "jepa_native_paper/2026-08-04-native-jepa-paper-1gpu-debug-v1/"
    "best-val-epoch=13-loss=0.21508.ckpt"
)
PATCHTST_CHECKPOINT = (
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/"
    "patchtst_self_supervised_paper/2026-08-05-patchtst-paper-1gpu-v1/"
    "best-val-epoch=03-loss=0.00329.ckpt"
)

PREDICTIONS_DIR = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/predictions"
)

SUBJECT_SPLIT_PATH = "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/manifests/hypotension_subject_split_fixed_v1.csv"
OUTCOME_DF_PATH = "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/labels/hypotension_labels_mimic_all_events_rolling5min.csv.gz"
SAMPLE_CACHE_DIR = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/fcn_hypotension_paper")
DATASET_FILENAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"

CHANNELS = ["ABP", "II", "PLETH"]
FREQUENCY = 125
SAMPLE_SEQ_LEN_SECONDS = 1800
FORECAST_WINDOW_SEC = [300]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_split_dataset(split: str) -> ForecastingDataset:
    """Load a dataset split using the fixed subject split."""
    subject_split = pd.read_csv(SUBJECT_SPLIT_PATH)
    subjects = set(
        subject_split.loc[subject_split["split"] == split, "subject_id"].astype(str)
    )
    outcomes = pd.read_csv(OUTCOME_DF_PATH)
    if "subject_id" not in outcomes.columns:
        outcomes["subject_id"] = outcomes["file_path"].map(
            lambda p: zarr_record_name(p).split("-", 1)[0]
        )
    outcomes = outcomes.loc[outcomes["subject_id"].astype(str).isin(subjects)].copy()
    outcomes["Time Stamp (seconds)"] = outcomes["Time Stamp (seconds)"].round()

    cache_path = SAMPLE_CACHE_DIR / f"{DATASET_FILENAME}-{split}_samples.csv.gz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Sample cache missing: {cache_path}")
    samples = pd.read_csv(cache_path)

    return ForecastingDataset(
        channels=CHANNELS,
        forecast_window_sec=FORECAST_WINDOW_SEC,
        outcome_df=outcomes,
        outcome_df_outcome_col="hypotension_label",
        file_col="file_path",
        y_date_column="date",
        outcome_df_seconds_since_column="Time Stamp (seconds)",
        outcome_df_duration_column="event_length",
        sample_df=samples,
        sample_seq_len_sec=SAMPLE_SEQ_LEN_SECONDS,
        frequency=FREQUENCY,
        butterworth_filters=None,
        median_filter_kernel_size=None,
        clip_interpolations=CLIP_INTERPOLATE_RANGES,
        constant_nan_tolerance=0.2,
        require_all_channels=True,
        infer_forecast_windows=False,
        normalize_signals=True,
    )


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Extract mean-pooled embeddings."""
    model.eval()
    model.to(device)
    all_embeddings = []
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(device)
        out = model(batch_x)
        if isinstance(out, tuple):
            z = out[0]
        else:
            z = out
        z_pooled = z.mean(dim=(1, 3))
        all_embeddings.append(z_pooled.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load train, val, and test datasets
    print("Loading train dataset...")
    train_dataset = load_split_dataset("train")
    print("Loading val dataset...")
    val_dataset = load_split_dataset("val")
    print("Loading test dataset...")
    test_dataset = load_split_dataset("test")

    label_col = f"outcome_val_{FORECAST_WINDOW_SEC[0]}sec"
    train_labels = train_dataset.sample_df[label_col].values
    val_labels = val_dataset.sample_df[label_col].values
    test_labels = test_dataset.sample_df[label_col].values

    # Subsample train: balanced 10K pos + 10K neg
    rng = np.random.default_rng(RANDOM_STATE)
    MAX_TRAIN = 20000
    pos_idx = np.where(train_labels == 1)[0]
    neg_idx = np.where(train_labels == 0)[0]
    n_pos = min(len(pos_idx), MAX_TRAIN // 2)
    n_neg = min(len(neg_idx), MAX_TRAIN // 2)
    train_indices = np.concatenate([
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=n_neg, replace=False),
    ])
    rng.shuffle(train_indices)
    print(f"Train subsample: {n_pos} pos + {n_neg} neg = {len(train_indices)} total")

    # Subsample val: balanced, capped at 10K
    val_pos_idx = np.where(val_labels == 1)[0]
    val_neg_idx = np.where(val_labels == 0)[0]
    n_val_pos = min(len(val_pos_idx), 5000)
    n_val_neg = min(len(val_neg_idx), 5000)
    val_indices = np.concatenate([
        rng.choice(val_pos_idx, size=n_val_pos, replace=False),
        rng.choice(val_neg_idx, size=n_val_neg, replace=False),
    ])
    rng.shuffle(val_indices)
    print(f"Val subsample: {n_val_pos} pos + {n_val_neg} neg = {len(val_indices)} total")

    # Subsample test: balanced, capped at 10K
    test_pos_idx = np.where(test_labels == 1)[0]
    test_neg_idx = np.where(test_labels == 0)[0]
    n_test_pos = min(len(test_pos_idx), 5000)
    n_test_neg = min(len(test_neg_idx), 5000)
    test_indices = np.concatenate([
        rng.choice(test_pos_idx, size=n_test_pos, replace=False),
        rng.choice(test_neg_idx, size=n_test_neg, replace=False),
    ])
    rng.shuffle(test_indices)
    print(f"Test subsample: {n_test_pos} pos + {n_test_neg} neg = {len(test_indices)} total")

    train_subset = Subset(train_dataset, train_indices.tolist())
    val_subset = Subset(val_dataset, val_indices.tolist())
    test_subset = Subset(test_dataset, test_indices.tolist())

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    train_y = train_labels[train_indices]
    val_y = val_labels[val_indices]
    test_y = test_labels[test_indices]

    models_config = [
        ("jepa", "JEPA", JEPA_CHECKPOINT, JEPASimpleLightning),
        ("patchtst", "PatchTST", PATCHTST_CHECKPOINT, PatchTFTSimpleLightning),
    ]

    for key, name, ckpt_path, model_class in models_config:
        print(f"\n--- {name} ---")
        model = model_class.load_from_checkpoint(ckpt_path, map_location="cpu")

        print(f"  Extracting train embeddings...")
        train_emb = extract_embeddings(model, train_loader, device)
        print(f"  Extracting val embeddings...")
        val_emb = extract_embeddings(model, val_loader, device)
        print(f"  Extracting test embeddings...")
        test_emb = extract_embeddings(model, test_loader, device)

        del model
        torch.cuda.empty_cache()

        # Standardize
        scaler = StandardScaler()
        train_emb_scaled = scaler.fit_transform(train_emb)
        val_emb_scaled = scaler.transform(val_emb)
        test_emb_scaled = scaler.transform(test_emb)

        # Fit logistic regression
        print(f"  Fitting logistic regression...")
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        clf.fit(train_emb_scaled, train_y)

        # Predict probabilities
        val_probs = clf.predict_proba(val_emb_scaled)[:, 1]
        test_probs = clf.predict_proba(test_emb_scaled)[:, 1]

        val_auroc = roc_auc_score(val_y, val_probs)
        val_ap = average_precision_score(val_y, val_probs)
        test_auroc = roc_auc_score(test_y, test_probs)
        test_ap = average_precision_score(test_y, test_probs)

        print(f"  Val  AUROC: {val_auroc:.4f}  AP: {val_ap:.4f}")
        print(f"  Test AUROC: {test_auroc:.4f}  AP: {test_ap:.4f}")

        # Save predictions in same format as downstream probes
        pred_dir = PREDICTIONS_DIR / f"{key}_linear_probe"
        pred_dir.mkdir(parents=True, exist_ok=True)
        output_path = pred_dir / f"{key}_linear_probe_fixed_v1.pt"
        predictions = {
            "val_preds": torch.tensor(val_probs, dtype=torch.float32),
            "val_targets": torch.tensor(val_y, dtype=torch.long),
            "test_preds": torch.tensor(test_probs, dtype=torch.float32),
            "test_targets": torch.tensor(test_y, dtype=torch.long),
        }
        tmp_path = output_path.with_suffix(".partial")
        torch.save(predictions, tmp_path)
        os.replace(tmp_path, output_path)
        print(f"  Saved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
