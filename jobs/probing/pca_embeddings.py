"""PCA analysis of full token-sequence embeddings.

Analyzes the intrinsic dimensionality of the JEPA encoder's 512-dim token
representations by computing how many PCA components are needed to explain
various variance thresholds.

Usage:
    python pca_embeddings.py [--embeddings_path PATH] [--max_tokens 500000]
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
    parser.add_argument("--max_tokens", type=int, default=500_000,
                        help="Max tokens to subsample for PCA fit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading embeddings...")
    data = np.load(args.embeddings_path)
    emb = data["embeddings"]  # [n_windows, n_channels, d_model, n_patches]
    print(f"  Shape: {emb.shape}")

    n_windows, n_channels, d_model, n_patches = emb.shape
    channel_names = ["ABP", "II", "PLETH"]

    # --- All channels combined ---
    print("\n=== All channels combined ===")
    emb_flat = emb.transpose(0, 1, 3, 2).reshape(-1, d_model).astype(np.float32)
    print(f"  Flattened: {emb_flat.shape[0]:,} tokens x {d_model} dims")

    if emb_flat.shape[0] > args.max_tokens:
        idx = rng.choice(emb_flat.shape[0], size=args.max_tokens, replace=False)
        emb_flat = emb_flat[idx]
        print(f"  Subsampled to {args.max_tokens:,} tokens")

    print("  Running PCA...")
    pca = PCA(n_components=min(512, emb_flat.shape[0]))
    pca.fit(emb_flat)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    print("\n  Variance thresholds:")
    for threshold in [0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]:
        n_dims = np.searchsorted(cumvar, threshold) + 1
        print(f"    {threshold*100:.0f}%: {n_dims} dimensions")

    print(f"\n  Top 20 components:")
    for i in range(20):
        print(f"    PC{i+1:3d}: {pca.explained_variance_ratio_[i]*100:.2f}% "
              f"(cumulative: {cumvar[i]*100:.2f}%)")

    # --- Per-channel ---
    print("\n=== Per-channel PCA ===")
    for ch_idx, ch_name in enumerate(channel_names):
        ch_emb = emb[:, ch_idx, :, :].transpose(0, 2, 1).reshape(-1, d_model).astype(np.float32)
        if ch_emb.shape[0] > args.max_tokens:
            idx = rng.choice(ch_emb.shape[0], size=args.max_tokens, replace=False)
            ch_emb = ch_emb[idx]

        pca_ch = PCA(n_components=min(512, ch_emb.shape[0]))
        pca_ch.fit(ch_emb)
        cumvar_ch = np.cumsum(pca_ch.explained_variance_ratio_)

        print(f"\n  {ch_name}:")
        for threshold in [0.50, 0.75, 0.90, 0.95, 0.99]:
            n_dims = np.searchsorted(cumvar_ch, threshold) + 1
            print(f"    {threshold*100:.0f}%: {n_dims} dims")

    # --- Per-window PCA (variance across patches within a window) ---
    print("\n=== Per-window intrinsic dimensionality (median across windows) ===")
    dims_90 = []
    for w in range(min(n_windows, 200)):  # sample up to 200 windows
        w_emb = emb[w].transpose(0, 2, 1).reshape(-1, d_model).astype(np.float32)
        pca_w = PCA(n_components=min(d_model, w_emb.shape[0]))
        pca_w.fit(w_emb)
        cumvar_w = np.cumsum(pca_w.explained_variance_ratio_)
        dims_90.append(np.searchsorted(cumvar_w, 0.90) + 1)

    dims_90 = np.array(dims_90)
    print(f"  90% variance per window: median={np.median(dims_90):.0f}, "
          f"mean={np.mean(dims_90):.1f}, min={dims_90.min()}, max={dims_90.max()}")


if __name__ == "__main__":
    main()
