"""Fit Feature PCA and Temporal PCA on frozen JEPA encoder embeddings.

This script extracts embeddings from the pretrained native JEPA encoder on
training data and fits two PCA transforms:

1. **Feature PCA** — projects d_model (512) → n_components that capture ≥90%
   of pooled-token variance.  All tokens across windows and channels are pooled
   into shape [N*3*1800, 512], subsampled to 500K rows, and sklearn PCA is fit.

2. **Temporal PCA** — projects num_patch (1800) → n_components that achieve
   ≥90% median per-window reconstruction variance explained.  Temporal vectors
   are reshaped to [N*3*512, 1800], subsampled to 500K rows, and PCA is fit.
   Per-window explained variance is evaluated on 1000 held-out windows at
   various n_components to find the smallest k where median >= 0.90.

Outputs (saved as .npz files):
    /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/pca/feature_pca.npz
    /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/pca/temporal_pca.npz

Each .npz contains:
    - components: principal axes (n_components, d)
    - mean: centering vector (d,)
    - n_components: number of components fit
    - explained_variance_ratio: per-component variance ratios
    - n_components_90: smallest k for ≥90% variance criterion

Usage (from /gpfs/home/dk5565/PhysioJEPA/jobs/jepa/):
    python scripts/fit_pca.py
    python scripts/fit_pca.py --n_windows 10000 --batch_size 128
    python scripts/fit_pca.py --config configs/train_hypotension_fixed.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, ForecastingDataset
from physiojepa.data_preprocessing import zarr_record_name
from physiojepa.jepa import JEPASimpleLightning

from pipeline_common import load_config, require_output_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/pca"
)
MAX_SUBSAMPLE = 500_000  # maximum rows for PCA fitting
TEMPORAL_EVAL_WINDOWS = 1000  # windows for per-window variance evaluation
VARIANCE_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Data loading (mirrors train_hypotension_fixed.py)
# ---------------------------------------------------------------------------


def _load_train_samples(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load outcomes and the fixed-split training sample cache."""
    subject_split = pd.read_csv(config["paths"]["subject_split_path"])
    train_subjects = set(
        subject_split.loc[
            subject_split["split"] == "train", "subject_id"
        ].astype(str)
    )

    outcomes = pd.read_csv(config["paths"]["outcome_df_path"])
    if "subject_id" not in outcomes.columns:
        outcomes["subject_id"] = outcomes["file_path"].map(
            lambda path: zarr_record_name(path).split("-", 1)[0]
        )
    outcomes = outcomes.loc[
        outcomes["subject_id"].astype(str).isin(train_subjects)
    ].copy()
    outcomes["Time Stamp (seconds)"] = outcomes["Time Stamp (seconds)"].round()

    cache_dir = Path(config["paths"]["sample_cache_dir"])
    dataset_name = config["paths"]["dataset_filename"]
    cache_path = cache_dir / f"{dataset_name}-train_samples.csv.gz"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Required fixed-split training cache is missing: {cache_path}"
        )
    samples = pd.read_csv(cache_path)
    if "subject_id" not in samples.columns:
        samples["subject_id"] = samples["file_path"].map(
            lambda path: zarr_record_name(path).split("-", 1)[0]
        )

    # Remap container paths to local override directory if set
    containers_override = os.environ.get("PHYSIOJEPA_CONTAINERS_OVERRIDE")
    if containers_override:
        samples["file_path"] = samples["file_path"].map(
            lambda p: os.path.join(containers_override, os.path.basename(p))
        )

    return outcomes, samples.reset_index(drop=True)


def _build_dataset(
    config: dict, outcomes: pd.DataFrame, samples: pd.DataFrame
) -> ForecastingDataset:
    """Create a ForecastingDataset from config and cached frames."""
    dataset_cfg = config["dataset"]
    return ForecastingDataset(
        channels=dataset_cfg["channels"],
        forecast_window_sec=dataset_cfg["forecast_window_sec"],
        outcome_df=outcomes,
        outcome_df_outcome_col=dataset_cfg["y_outcome"],
        file_col="file_path",
        y_date_column="date",
        outcome_df_seconds_since_column="Time Stamp (seconds)",
        outcome_df_duration_column="event_length",
        sample_df=samples,
        sample_seq_len_sec=dataset_cfg["sample_seq_len_seconds"],
        frequency=dataset_cfg["frequency"],
        butterworth_filters=None,
        median_filter_kernel_size=None,
        clip_interpolations=CLIP_INTERPOLATE_RANGES,
        constant_nan_tolerance=dataset_cfg["constant_nan_tolerance"],
        require_all_channels=dataset_cfg["require_all_channels"],
        infer_forecast_windows=dataset_cfg["infer_forecast_windows"],
        normalize_signals=dataset_cfg["normalize_signals"],
    )


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_embeddings(
    encoder: torch.nn.Module,
    dataloader: DataLoader,
    n_windows: int,
    device: torch.device,
) -> np.ndarray:
    """Extract encoder embeddings for up to n_windows samples.

    Returns:
        embeddings: numpy array of shape [N, c_in=3, d_model=512, num_patch=1800]
    """
    encoder.eval()
    all_embeddings = []
    total_collected = 0

    for batch in dataloader:
        if total_collected >= n_windows:
            break

        # batch is (x, y, ...) or just x depending on dataset
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch

        x = x.to(device)
        # Encoder forward: input [bs, c_in, seq_len] -> output [bs, c_in, d_model, num_patch]
        emb = encoder(x)
        batch_size = emb.shape[0]

        # Only take what we need to reach n_windows
        remaining = n_windows - total_collected
        if batch_size > remaining:
            emb = emb[:remaining]

        all_embeddings.append(emb.cpu().numpy())
        total_collected += emb.shape[0]

        if total_collected % 500 == 0 or total_collected >= n_windows:
            print(f"  Extracted {total_collected}/{n_windows} windows", flush=True)

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Final embeddings shape: {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# Feature PCA: [N*3*1800, 512] -> find k for 90% variance
# ---------------------------------------------------------------------------


def fit_feature_pca(
    embeddings: np.ndarray, rng: np.random.Generator
) -> tuple[PCA, int]:
    """Fit PCA on pooled token features (d_model dimension).

    Args:
        embeddings: shape [N, 3, 512, 1800]
        rng: numpy random generator for reproducible subsampling

    Returns:
        (fitted PCA object, n_components for 90% variance)
    """
    N, c_in, d_model, num_patch = embeddings.shape
    print(f"\n{'='*60}")
    print("Feature PCA: projecting d_model={d_model} dimension")
    print(f"{'='*60}")

    # Pool all tokens: [N, 3, 512, 1800] -> [N*3*1800, 512]
    # Rearrange: move num_patch to combine with batch
    pooled = embeddings.transpose(0, 1, 3, 2)  # [N, 3, 1800, 512]
    pooled = pooled.reshape(-1, d_model)  # [N*3*1800, 512]
    print(f"  Pooled tokens shape: {pooled.shape}")

    # Subsample to MAX_SUBSAMPLE rows for tractable PCA
    n_total = pooled.shape[0]
    if n_total > MAX_SUBSAMPLE:
        indices = rng.choice(n_total, size=MAX_SUBSAMPLE, replace=False)
        pooled = pooled[indices]
        print(f"  Subsampled to {MAX_SUBSAMPLE} rows")

    # Remove any rows with NaN/inf
    valid_mask = np.isfinite(pooled).all(axis=1)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  Removing {n_invalid} rows with NaN/inf values")
        pooled = pooled[valid_mask]

    # Fit full PCA
    print(f"  Fitting PCA on {pooled.shape[0]} samples x {pooled.shape[1]} features...")
    pca = PCA(n_components=min(pooled.shape))
    pca.fit(pooled)

    # Find n_components for 90% cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = int(np.searchsorted(cumulative_variance, VARIANCE_THRESHOLD) + 1)

    print(f"\n  Results:")
    print(f"    Total components: {pca.n_components_}")
    print(f"    n_components for {VARIANCE_THRESHOLD*100:.0f}% variance: {n_components_90}")
    print(f"    Variance at k={n_components_90}: {cumulative_variance[n_components_90-1]:.4f}")
    print(f"    Top-10 explained variance ratios: {pca.explained_variance_ratio_[:10]}")

    return pca, n_components_90


# ---------------------------------------------------------------------------
# Temporal PCA: [N*3*512, 1800] -> find k for 90% median per-window variance
# ---------------------------------------------------------------------------


def _per_window_variance_explained(
    embeddings_eval: np.ndarray, pca: PCA, k: int
) -> np.ndarray:
    """Compute per-window explained variance at k components.

    Args:
        embeddings_eval: shape [N_eval, 3, 512, 1800] — evaluation windows
        pca: fitted temporal PCA
        k: number of components to use

    Returns:
        Array of per-window variance explained ratios, shape [N_eval]
    """
    N, c_in, d_model, num_patch = embeddings_eval.shape

    # Reshape to temporal vectors: [N, 3, 512, 1800] -> [N*3*512, 1800]
    temporal = embeddings_eval.reshape(N * c_in * d_model, num_patch)

    # Center with PCA mean
    centered = temporal - pca.mean_

    # Project to k components and reconstruct
    components_k = pca.components_[:k]  # [k, 1800]
    projected = centered @ components_k.T  # [N*3*512, k]
    reconstructed = projected @ components_k  # [N*3*512, 1800]

    # Compute per-window variance explained
    # Reshape back: [N, 3*512, 1800]
    centered_windows = centered.reshape(N, c_in * d_model, num_patch)
    reconstructed_windows = reconstructed.reshape(N, c_in * d_model, num_patch)

    # Total variance per window = sum of squared centered values
    total_var = (centered_windows ** 2).sum(axis=(1, 2))  # [N]
    # Residual variance = sum of squared residuals
    residual_var = ((centered_windows - reconstructed_windows) ** 2).sum(axis=(1, 2))

    # Avoid division by zero
    valid = total_var > 1e-10
    var_explained = np.zeros(N)
    var_explained[valid] = 1.0 - residual_var[valid] / total_var[valid]

    return var_explained


def fit_temporal_pca(
    embeddings: np.ndarray, rng: np.random.Generator
) -> tuple[PCA, int]:
    """Fit PCA on temporal dimension (num_patch) and find k for 90% median variance.

    Args:
        embeddings: shape [N, 3, 512, 1800]
        rng: numpy random generator

    Returns:
        (fitted PCA object, n_components for 90% median per-window variance)
    """
    N, c_in, d_model, num_patch = embeddings.shape
    print(f"\n{'='*60}")
    print(f"Temporal PCA: projecting num_patch={num_patch} dimension")
    print(f"{'='*60}")

    # Reshape to temporal vectors: [N, 3, 512, 1800] -> [N*3*512, 1800]
    temporal = embeddings.reshape(N * c_in * d_model, num_patch)
    print(f"  Temporal vectors shape: {temporal.shape}")

    # Subsample to MAX_SUBSAMPLE rows for tractable PCA fitting
    n_total = temporal.shape[0]
    if n_total > MAX_SUBSAMPLE:
        indices = rng.choice(n_total, size=MAX_SUBSAMPLE, replace=False)
        temporal = temporal[indices]
        print(f"  Subsampled to {MAX_SUBSAMPLE} rows for fitting")

    # Remove any rows with NaN/inf
    valid_mask = np.isfinite(temporal).all(axis=1)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  Removing {n_invalid} rows with NaN/inf values")
        temporal = temporal[valid_mask]

    # Fit full PCA on temporal dimension
    n_components_fit = min(temporal.shape)
    print(f"  Fitting PCA on {temporal.shape[0]} samples x {temporal.shape[1]} features...")
    pca = PCA(n_components=n_components_fit)
    pca.fit(temporal)

    # --- Evaluate per-window explained variance to find n_components_90 ---
    # Use a subset of windows for evaluation
    n_eval = min(TEMPORAL_EVAL_WINDOWS, N)
    eval_indices = rng.choice(N, size=n_eval, replace=False)
    embeddings_eval = embeddings[eval_indices]
    print(f"\n  Evaluating per-window variance on {n_eval} windows...")

    # Search for smallest k where median per-window variance >= 90%
    # Use binary search over possible k values for efficiency
    cumulative_global = np.cumsum(pca.explained_variance_ratio_)
    # Start search around where global cumulative hits threshold
    k_global_90 = int(np.searchsorted(cumulative_global, VARIANCE_THRESHOLD) + 1)
    print(f"  Global cumulative 90% at k={k_global_90}")

    # Evaluate a range of k values around the expected threshold
    # Search from small to large — find the first k where median >= 0.90
    k_candidates = sorted(set(
        list(range(max(1, k_global_90 - 50), min(k_global_90 + 100, pca.n_components_) + 1, 5))
        + list(range(max(1, k_global_90 - 10), min(k_global_90 + 20, pca.n_components_) + 1))
        + [k_global_90]
    ))

    n_components_90 = pca.n_components_  # fallback
    best_median = 0.0
    print(f"  Searching k in [{k_candidates[0]}, {k_candidates[-1]}]...")

    for k in k_candidates:
        var_explained = _per_window_variance_explained(embeddings_eval, pca, k)
        median_ve = float(np.median(var_explained))
        if k == k_global_90 or k == k_candidates[0] or k == k_candidates[-1]:
            print(f"    k={k:4d}: median VE = {median_ve:.4f}, "
                  f"mean = {var_explained.mean():.4f}, "
                  f"min = {var_explained.min():.4f}")
        if median_ve >= VARIANCE_THRESHOLD and k < n_components_90:
            n_components_90 = k
            best_median = median_ve

    # If we found the threshold, do a fine-grained search for exact boundary
    if n_components_90 < pca.n_components_:
        fine_start = max(1, n_components_90 - 5)
        fine_end = n_components_90 + 1
        for k in range(fine_start, fine_end):
            var_explained = _per_window_variance_explained(embeddings_eval, pca, k)
            median_ve = float(np.median(var_explained))
            if median_ve >= VARIANCE_THRESHOLD:
                n_components_90 = k
                best_median = median_ve
                break

    print(f"\n  Results:")
    print(f"    Total components fit: {pca.n_components_}")
    print(f"    Global cumulative 90% at: k={k_global_90}")
    print(f"    Median per-window 90% at: k={n_components_90} "
          f"(median VE = {best_median:.4f})")

    # Print per-channel breakdown for the chosen k
    print(f"\n  Per-channel breakdown at k={n_components_90}:")
    for ch_idx in range(c_in):
        # Evaluate single-channel temporal vectors
        ch_embeddings = embeddings_eval[:, ch_idx:ch_idx+1, :, :]  # [N, 1, 512, 1800]
        ch_temporal = ch_embeddings.reshape(n_eval * d_model, num_patch)
        ch_centered = ch_temporal - pca.mean_
        components_k = pca.components_[:n_components_90]
        ch_projected = ch_centered @ components_k.T
        ch_reconstructed = ch_projected @ components_k
        ch_centered_w = ch_centered.reshape(n_eval, d_model, num_patch)
        ch_recon_w = ch_reconstructed.reshape(n_eval, d_model, num_patch)
        total_var = (ch_centered_w ** 2).sum(axis=(1, 2))
        residual_var = ((ch_centered_w - ch_recon_w) ** 2).sum(axis=(1, 2))
        valid = total_var > 1e-10
        ch_ve = np.zeros(n_eval)
        ch_ve[valid] = 1.0 - residual_var[valid] / total_var[valid]
        ch_names = ["ABP", "ECG_II", "PLETH"]
        ch_name = ch_names[ch_idx] if ch_idx < len(ch_names) else f"ch{ch_idx}"
        print(f"    {ch_name}: median VE = {np.median(ch_ve):.4f}, "
              f"mean = {ch_ve.mean():.4f}")

    return pca, n_components_90


# ---------------------------------------------------------------------------
# Save PCA to .npz
# ---------------------------------------------------------------------------


def save_pca(pca: PCA, n_components_90: int, output_path: Path, label: str) -> None:
    """Save PCA transform as .npz file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        n_components=np.int32(pca.n_components_),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        n_components_90=np.int32(n_components_90),
    )
    print(f"\n  Saved {label} to {output_path}")
    print(f"    File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Feature PCA and Temporal PCA on JEPA encoder embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_hypotension_fixed.yaml",
        help="Path to YAML config (relative to working directory or absolute).",
    )
    parser.add_argument(
        "--n_windows",
        type=int,
        default=5000,
        help="Number of training windows to extract embeddings from.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for PCA .npz files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Seed everything
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("=" * 60)
    print("JEPA PCA Fitting Script")
    print("=" * 60)
    print(f"  Config:      {args.config}")
    print(f"  n_windows:   {args.n_windows}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  seed:        {args.seed}")
    print(f"  output_dir:  {args.output_dir}")

    # Load configuration
    os.environ.setdefault("PHYSIOJEPA_CONFIG", args.config)
    config, config_path = load_config(args.config)
    print(f"  Loaded config from: {config_path}")

    # Load encoder
    pretrained_path = Path(config["paths"]["pretrained_encoder_path"]).resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f"JEPA checkpoint not found: {pretrained_path}"
        )
    print(f"\n  Loading encoder from: {pretrained_path}")
    encoder_lightning = JEPASimpleLightning.load_from_checkpoint(
        str(pretrained_path), map_location="cpu"
    )
    # Use the full Lightning module for forward (handles reshape + permute)
    # Output shape: [bs, c_in=3, d_model=512, num_patch=1800]
    encoder_lightning.eval()
    encoder_lightning.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_lightning = encoder_lightning.to(device)
    print(f"  Encoder on device: {device}")
    print(f"  d_model={encoder_lightning.d_model}, "
          f"num_patch={encoder_lightning.num_patch}, "
          f"c_in={encoder_lightning.c_in}")

    # Load training data (same pattern as train_hypotension_fixed.py)
    print("\n  Loading training sample cache...")
    outcomes, samples = _load_train_samples(config)

    # Subsample windows if we have more than needed
    if len(samples) > args.n_windows:
        samples = samples.sample(
            n=args.n_windows, random_state=args.seed
        ).sort_values(["file_path", "start_idx"]).reset_index(drop=True)
        print(f"  Subsampled to {args.n_windows} windows from training cache")
    else:
        print(f"  Using all {len(samples)} available training windows")

    dataset = _build_dataset(config, outcomes, samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Extract embeddings
    print(f"\n  Extracting embeddings from {len(samples)} windows...")
    embeddings = extract_embeddings(
        encoder=encoder_lightning,
        dataloader=dataloader,
        n_windows=args.n_windows,
        device=device,
    )
    # Expected shape: [N, c_in=3, d_model=512, num_patch=1800]
    N, c_in, d_model, num_patch = embeddings.shape
    print(f"\n  Embeddings: N={N}, c_in={c_in}, d_model={d_model}, num_patch={num_patch}")

    # Free GPU memory
    encoder_lightning = encoder_lightning.cpu()
    torch.cuda.empty_cache()

    # --- Fit Feature PCA ---
    feature_pca, feature_n90 = fit_feature_pca(embeddings, rng)

    # --- Fit Temporal PCA ---
    temporal_pca, temporal_n90 = fit_temporal_pca(embeddings, rng)

    # --- Save results ---
    output_dir = Path(args.output_dir)
    require_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_pca(feature_pca, feature_n90, output_dir / "feature_pca.npz", "Feature PCA")
    save_pca(temporal_pca, temporal_n90, output_dir / "temporal_pca.npz", "Temporal PCA")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Windows processed:          {N}")
    print(f"  Encoder d_model:            {d_model}")
    print(f"  Encoder num_patch:          {num_patch}")
    print(f"  Channels:                   {c_in}")
    print(f"")
    print(f"  Feature PCA (d_model={d_model}):")
    print(f"    Total components:         {feature_pca.n_components_}")
    print(f"    n_components (90% var):   {feature_n90}")
    cumvar_feat = np.cumsum(feature_pca.explained_variance_ratio_)
    print(f"    Cumulative var at k={feature_n90}: {cumvar_feat[feature_n90-1]:.4f}")
    print(f"")
    print(f"  Temporal PCA (num_patch={num_patch}):")
    print(f"    Total components:         {temporal_pca.n_components_}")
    print(f"    n_components (90% med VE): {temporal_n90}")
    cumvar_temp = np.cumsum(temporal_pca.explained_variance_ratio_)
    print(f"    Global cumulative at k={temporal_n90}: {cumvar_temp[temporal_n90-1]:.4f}")
    print(f"")
    print(f"  Output files:")
    print(f"    {output_dir / 'feature_pca.npz'}")
    print(f"    {output_dir / 'temporal_pca.npz'}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
