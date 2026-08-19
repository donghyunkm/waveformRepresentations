"""Temporal PCA analysis of JEPA token embeddings.

Analyzes the temporal dimension (1800 patches) to determine how many temporal
modes/basis functions are needed to explain the variation in the embedding
sequence across time.

Two approaches:
1. Global: Pool all windows together, find shared temporal patterns
2. Per-window: Find how temporally complex each individual window is

Usage:
    python pca_temporal.py [--embeddings_path PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


DEFAULT_EMBEDDINGS = (
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/"
    "patient_token_sequences/token_embeddings_20patients_50windows_seed42.npz"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings_path", type=str, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--n_subsample", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading embeddings...")
    data = np.load(args.embeddings_path, allow_pickle=True)
    emb = data["embeddings"]  # [n_windows, 3, 512, 1800]
    n_windows, n_channels, d_model, n_patches = emb.shape
    print(f"  Shape: {emb.shape}")

    channel_names = ["ABP", "II", "PLETH"]

    # =================================================================
    # Approach 1: Global Temporal PCA
    # =================================================================
    print("\n=== Approach 1: Global Temporal PCA ===")
    print("Each sample = one (window, channel, feature_dim) as a 1800-dim vector")
    print("Question: How many temporal basis functions explain cross-window variation?\n")

    # Reshape to [n_windows * n_channels * d_model, 1800]
    temporal_vectors = emb.reshape(-1, n_patches).astype(np.float32)
    print(f"  Total temporal vectors: {temporal_vectors.shape[0]:,} x {n_patches}")

    idx = rng.choice(temporal_vectors.shape[0], size=args.n_subsample, replace=False)
    temporal_sub = temporal_vectors[idx]
    print(f"  Subsampled to {args.n_subsample:,}")

    pca_t = PCA(n_components=min(500, args.n_subsample))
    pca_t.fit(temporal_sub)
    cumvar = np.cumsum(pca_t.explained_variance_ratio_)

    print("\n  Variance thresholds:")
    for threshold in [0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
        n_dims = np.searchsorted(cumvar, threshold) + 1
        print(f"    {threshold*100:.0f}%: {n_dims} temporal modes")

    print(f"\n  Top 10 components:")
    for i in range(10):
        print(f"    TC{i+1:3d}: {pca_t.explained_variance_ratio_[i]*100:.2f}% "
              f"(cumulative: {cumvar[i]*100:.2f}%)")

    print("\n  Per channel (90% threshold):")
    for ch_idx, ch_name in enumerate(channel_names):
        ch_temporal = emb[:, ch_idx, :, :].reshape(-1, n_patches).astype(np.float32)
        idx_ch = rng.choice(ch_temporal.shape[0], size=min(args.n_subsample, ch_temporal.shape[0]), replace=False)
        pca_ch = PCA(n_components=min(500, len(idx_ch)))
        pca_ch.fit(ch_temporal[idx_ch])
        cumvar_ch = np.cumsum(pca_ch.explained_variance_ratio_)
        n90 = np.searchsorted(cumvar_ch, 0.90) + 1
        n95 = np.searchsorted(cumvar_ch, 0.95) + 1
        print(f"    {ch_name}: 90% -> {n90} modes, 95% -> {n95} modes")

    # =================================================================
    # Approach 2: Per-Window Temporal PCA
    # =================================================================
    print("\n\n=== Approach 2: Per-Window Temporal PCA ===")
    print("For each (window, channel): [512, 1800] matrix -> PCA on 1800-dim")
    print("Question: How many temporal modes does each individual window use?\n")

    dims_90_all = []
    dims_95_all = []
    dims_50_all = []
    per_channel_dims = {ch: [] for ch in channel_names}

    for w in range(n_windows):
        for ch_idx, ch_name in enumerate(channel_names):
            window_mat = emb[w, ch_idx, :, :].astype(np.float32)  # [512, 1800]
            pca_w = PCA(n_components=min(512, 1800))
            pca_w.fit(window_mat)
            cumvar_w = np.cumsum(pca_w.explained_variance_ratio_)
            d90 = np.searchsorted(cumvar_w, 0.90) + 1
            d95 = np.searchsorted(cumvar_w, 0.95) + 1
            d50 = np.searchsorted(cumvar_w, 0.50) + 1
            dims_90_all.append(d90)
            dims_95_all.append(d95)
            dims_50_all.append(d50)
            per_channel_dims[ch_name].append(d90)

    dims_90_all = np.array(dims_90_all)
    dims_95_all = np.array(dims_95_all)
    dims_50_all = np.array(dims_50_all)

    print(f"  Across all windows and channels ({len(dims_90_all)} matrices):")
    print(f"    50% variance: median={np.median(dims_50_all):.0f}, "
          f"mean={np.mean(dims_50_all):.1f}, "
          f"IQR=[{np.percentile(dims_50_all,25):.0f}, {np.percentile(dims_50_all,75):.0f}]")
    print(f"    90% variance: median={np.median(dims_90_all):.0f}, "
          f"mean={np.mean(dims_90_all):.1f}, "
          f"IQR=[{np.percentile(dims_90_all,25):.0f}, {np.percentile(dims_90_all,75):.0f}]")
    print(f"    95% variance: median={np.median(dims_95_all):.0f}, "
          f"mean={np.mean(dims_95_all):.1f}, "
          f"IQR=[{np.percentile(dims_95_all,25):.0f}, {np.percentile(dims_95_all,75):.0f}]")

    print(f"\n  Per channel (90% variance):")
    for ch_name in channel_names:
        d = np.array(per_channel_dims[ch_name])
        print(f"    {ch_name}: median={np.median(d):.0f}, mean={np.mean(d):.1f}, "
              f"IQR=[{np.percentile(d,25):.0f}, {np.percentile(d,75):.0f}]")


if __name__ == "__main__":
    main()
