"""UMAP and t-SNE scatter plots of pretrained encoder embeddings.

Extracts mean-pooled embeddings from a balanced subset of the held-out test set
and produces:
  - Individual UMAP/t-SNE plots per encoder colored by hypotension label
  - A side-by-side 2×2 comparison figure
  - Raw embeddings saved as .npz
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
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, ForecastingDataset
from physiojepa.data_preprocessing import zarr_record_name
from physiojepa.jepa import JEPASimpleLightning
from physiojepa.patchtst import PatchTFTSimpleLightning

from pipeline_common import load_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SAMPLES = 5000
N_DRAWS = 4  # number of independent balanced subsets
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
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run forward pass and collect mean-pooled embeddings + labels."""
    model.eval()
    model.to(device)

    all_embeddings = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        out = model(batch_x)
        if isinstance(out, tuple):
            z = out[0]
        else:
            z = out
        z_pooled = z.mean(dim=(1, 3))
        all_embeddings.append(z_pooled.cpu().numpy())
        all_labels.append(batch_y.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0).squeeze()
    return embeddings, labels


def plot_projection(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    method_name: str,
):
    """Create and save a scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    neg_mask = labels == 0
    pos_mask = labels == 1

    ax.scatter(
        coords_2d[neg_mask, 0], coords_2d[neg_mask, 1],
        c="#4ECDC4", s=8, alpha=0.4,
        label=f"No hypotension (n={neg_mask.sum()})", rasterized=True,
    )
    ax.scatter(
        coords_2d[pos_mask, 0], coords_2d[pos_mask, 1],
        c="#FF6B6B", s=8, alpha=0.6,
        label=f"Hypotension (n={pos_mask.sum()})", rasterized=True,
    )

    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.set_title(title)
    ax.legend(loc="upper right", markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


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

    models_config = [
        ("JEPA (epoch 13)", JEPA_CHECKPOINT, JEPASimpleLightning),
        ("PatchTST (epoch 3)", PATCHTST_CHECKPOINT, PatchTFTSimpleLightning),
    ]

    for draw_idx in range(N_DRAWS):
        seed = RANDOM_STATE + draw_idx
        print(f"\n{'='*60}")
        print(f"Draw {draw_idx + 1}/{N_DRAWS} (seed={seed})")
        print(f"{'='*60}")

        indices = balanced_subset_indices(test_dataset, N_SAMPLES, seed)
        subset = Subset(test_dataset, indices)
        loader = DataLoader(
            subset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )
        print(f"  Subset size: {len(subset)} samples")

        all_embeddings = {}

        for name, ckpt_path, model_class in models_config:
            print(f"\n  Loading {name} from {Path(ckpt_path).name}...")
            model = model_class.load_from_checkpoint(ckpt_path, map_location="cpu")

            print(f"    Extracting embeddings...")
            embeddings, labels = extract_embeddings(model, loader, device)
            all_embeddings[name] = (embeddings, labels)
            print(f"    Shape: {embeddings.shape}, pos={labels.sum():.0f}, neg={(1-labels).sum():.0f}")

            # Save raw embeddings (only for first draw)
            if draw_idx == 0:
                np.savez_compressed(
                    OUTPUT_DIR / f"embeddings_{name.split()[0].lower()}.npz",
                    embeddings=embeddings, labels=labels,
                )

            del model
            torch.cuda.empty_cache()

        # Compute projections
        suffix = f"_draw{draw_idx + 1}"
        print(f"\n  --- Computing projections (draw {draw_idx + 1}) ---")
        for name, (embeddings, labels) in all_embeddings.items():
            model_tag = name.split()[0].lower()

            # UMAP
            print(f"    UMAP for {name}...")
            reducer = umap.UMAP(
                n_components=2, n_neighbors=30, min_dist=0.3,
                metric="cosine", random_state=seed,
            )
            coords_umap = reducer.fit_transform(embeddings)
            plot_projection(
                coords_umap, labels,
                f"{name} — UMAP (draw {draw_idx + 1}, n={len(labels)})",
                OUTPUT_DIR / f"umap_{model_tag}{suffix}.png", "UMAP",
            )

            # t-SNE
            print(f"    t-SNE for {name}...")
            tsne = TSNE(
                n_components=2, perplexity=30, learning_rate="auto",
                init="pca", random_state=seed, n_jobs=-1,
            )
            coords_tsne = tsne.fit_transform(embeddings)
            plot_projection(
                coords_tsne, labels,
                f"{name} — t-SNE (draw {draw_idx + 1}, n={len(labels)})",
                OUTPUT_DIR / f"tsne_{model_tag}{suffix}.png", "t-SNE",
            )

        # Comparison figure for this draw
        print(f"\n  Creating comparison figure (draw {draw_idx + 1})...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        for col, (name, (embeddings, labels)) in enumerate(all_embeddings.items()):
            neg_mask = labels == 0
            pos_mask = labels == 1

            # UMAP (top row)
            reducer = umap.UMAP(
                n_components=2, n_neighbors=30, min_dist=0.3,
                metric="cosine", random_state=seed,
            )
            coords = reducer.fit_transform(embeddings)
            ax = axes[0, col]
            ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1], c="#4ECDC4", s=6, alpha=0.4, label="No hypotension", rasterized=True)
            ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1], c="#FF6B6B", s=6, alpha=0.6, label="Hypotension", rasterized=True)
            ax.set_title(f"{name} — UMAP")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc="upper right", markerscale=3, fontsize=9)

            # t-SNE (bottom row)
            tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=seed, n_jobs=-1)
            coords = tsne.fit_transform(embeddings)
            ax = axes[1, col]
            ax.scatter(coords[neg_mask, 0], coords[neg_mask, 1], c="#4ECDC4", s=6, alpha=0.4, label="No hypotension", rasterized=True)
            ax.scatter(coords[pos_mask, 0], coords[pos_mask, 1], c="#FF6B6B", s=6, alpha=0.6, label="Hypotension", rasterized=True)
            ax.set_title(f"{name} — t-SNE")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc="upper right", markerscale=3, fontsize=9)

        fig.suptitle(f"Pretrained Encoder Embeddings — Test Set (draw {draw_idx + 1}, balanced)", fontsize=14, y=0.98)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"comparison_umap_tsne{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUTPUT_DIR / f'comparison_umap_tsne{suffix}.png'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
