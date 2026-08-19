"""Per-patient trajectory plots in embedding space.

For each encoder, selects the longest negative (no-hypotension) ICU stays,
which have continuous ~60s sampling. Extracts all 30-min window embeddings
in chronological order, projects them into a shared UMAP space (fit on a
larger background set), and plots each patient as a separate subplot with
points colored by time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_JEPA_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "jepa" / "scripts"
if str(_JEPA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_JEPA_SCRIPTS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from torch.utils.data import DataLoader, Subset

from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, ForecastingDataset
from physiojepa.data_preprocessing import zarr_record_name
from physiojepa.jepa import JEPASimpleLightning
from physiojepa.patchtst import PatchTFTSimpleLightning

from pipeline_common import load_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_BACKGROUND = 5000  # balanced background samples to fit UMAP
N_STAYS = 8  # 8 longest negative (continuously sampled) stays
MAX_WINDOWS_PER_STAY = 200
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

OUTPUT_DIR = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/figures/embedding_viz"
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


def load_test_dataset() -> ForecastingDataset:
    """Load the held-out test set using the fixed subject split."""
    subject_split = pd.read_csv(SUBJECT_SPLIT_PATH)
    test_subjects = set(
        subject_split.loc[subject_split["split"] == "test", "subject_id"].astype(str)
    )
    outcomes = pd.read_csv(OUTCOME_DF_PATH)
    if "subject_id" not in outcomes.columns:
        outcomes["subject_id"] = outcomes["file_path"].map(
            lambda p: zarr_record_name(p).split("-", 1)[0]
        )
    outcomes = outcomes.loc[outcomes["subject_id"].astype(str).isin(test_subjects)].copy()
    outcomes["Time Stamp (seconds)"] = outcomes["Time Stamp (seconds)"].round()

    cache_path = SAMPLE_CACHE_DIR / f"{DATASET_FILENAME}-test_samples.csv.gz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Test sample cache missing: {cache_path}")
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


def balanced_subset_indices(
    dataset: ForecastingDataset, n_total: int, seed: int
) -> list[int]:
    """Return indices for a balanced subset (50/50 positive/negative)."""
    label_col = f"outcome_val_{FORECAST_WINDOW_SEC[0]}sec"
    labels = dataset.sample_df[label_col].values
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    rng = np.random.default_rng(seed)
    n_per_class = n_total // 2
    n_pos = min(n_per_class, len(pos_idx))
    n_neg = min(n_per_class, len(neg_idx))

    selected_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    selected_neg = rng.choice(neg_idx, size=n_neg, replace=False)

    indices = np.concatenate([selected_pos, selected_neg])
    rng.shuffle(indices)
    return indices.tolist()


@torch.no_grad()
def extract_embeddings_only(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run forward pass and collect mean-pooled embeddings."""
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_test_dataset()
    sample_df = test_dataset.sample_df.copy()
    label_col = f"outcome_val_{FORECAST_WINDOW_SEC[0]}sec"

    # Background indices for UMAP fitting
    bg_indices = balanced_subset_indices(test_dataset, N_BACKGROUND, RANDOM_STATE)
    bg_subset = Subset(test_dataset, bg_indices)
    bg_loader = DataLoader(
        bg_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    print(f"  Background size: {len(bg_subset)} samples")

    # Find longest negative stays (these have continuous ~60s sampling)
    stay_info = sample_df.groupby("file_path").agg(
        n_samples=("start_idx", "count"),
        has_positive=(label_col, "max"),
    )
    neg_stays = (
        stay_info[stay_info["has_positive"] == 0]
        .nlargest(N_STAYS, "n_samples")
        .index.tolist()
    )
    print(f"  Selected {len(neg_stays)} negative stays (continuously sampled)")

    # Build per-stay sample indices sorted by time
    stay_indices = {}
    for stay_path in neg_stays:
        stay_mask = sample_df["file_path"] == stay_path
        stay_rows = sample_df[stay_mask].sort_values("start_idx")
        indices = stay_rows.index.tolist()
        if len(indices) > MAX_WINDOWS_PER_STAY:
            step = len(indices) // MAX_WINDOWS_PER_STAY
            indices = indices[::step][:MAX_WINDOWS_PER_STAY]
        stay_indices[stay_path] = indices

    # Process each model
    models_config = [
        ("JEPA (epoch 13)", "jepa", JEPA_CHECKPOINT, JEPASimpleLightning),
        ("PatchTST (epoch 3)", "patchtst", PATCHTST_CHECKPOINT, PatchTFTSimpleLightning),
    ]

    for model_name, model_tag, ckpt_path, model_class in models_config:
        print(f"\n--- {model_name} ---")
        model = model_class.load_from_checkpoint(ckpt_path, map_location="cpu")

        # Extract background embeddings
        print(f"  Extracting background embeddings...")
        bg_embeddings = extract_embeddings_only(model, bg_loader, device)
        print(f"  Background embeddings: {bg_embeddings.shape}")

        # Fit UMAP on background
        print(f"  Fitting UMAP on {len(bg_embeddings)} background samples...")
        reducer = umap.UMAP(
            n_components=2, n_neighbors=30, min_dist=0.3,
            metric="cosine", random_state=RANDOM_STATE,
        )
        reducer.fit(bg_embeddings)
        bg_coords = reducer.transform(bg_embeddings)

        # Extract per-stay embeddings and transform
        stay_coords = {}
        for stay_path in neg_stays:
            indices = stay_indices[stay_path]
            subset = Subset(test_dataset, indices)
            loader = DataLoader(
                subset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=4, pin_memory=True,
            )
            stay_emb = extract_embeddings_only(model, loader, device)
            stay_coords[stay_path] = reducer.transform(stay_emb)

        del model
        torch.cuda.empty_cache()

        # Plot: 2 rows × 4 columns of continuously-sampled stays
        n_cols = 4
        n_rows = (len(neg_stays) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for idx, stay_path in enumerate(neg_stays):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            coords = stay_coords[stay_path]
            n_pts = len(coords)

            # Light background
            ax.scatter(
                bg_coords[:, 0], bg_coords[:, 1],
                c="lightgray", s=1, alpha=0.15, rasterized=True,
            )

            # Patient points colored by time
            ax.scatter(
                coords[:, 0], coords[:, 1],
                c=np.linspace(0, 1, n_pts), cmap="plasma",
                s=20, alpha=0.85, rasterized=True, edgecolors="none",
            )

            ax.set_xticks([])
            ax.set_yticks([])
            stay_id = Path(stay_path).stem[:20]
            ax.set_title(f"{stay_id} ({n_pts} win)", fontsize=9)

        # Hide unused subplots
        for idx in range(len(neg_stays), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        # Shared colorbar — placed outside the subplot area
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(0, 1)),
            cax=cbar_ax,
        )
        cbar.set_label("Time (early → late in stay)")

        fig.suptitle(
            f"{model_name} — Per-Patient Trajectories (No Hypotension, Continuous Sampling)\n"
            f"(UMAP fit on {len(bg_embeddings)} background test samples)",
            fontsize=12, y=0.98,
        )
        out_path = OUTPUT_DIR / f"trajectories_{model_tag}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
