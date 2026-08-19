"""Intrinsic dimension estimation of JEPA token embeddings.

Computes PCA participation ratio, Two-NN (Facco et al. 2017), and
Levina-Bickel MLE (2005) estimates at the token level, per channel,
and at the window level (mean-pooled).

Usage:
    python intrinsic_dimension.py [--embeddings_path PATH] [--n_subsample 50000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


DEFAULT_EMBEDDINGS = (
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/"
    "patient_token_sequences/token_embeddings_20patients_50windows_seed42.npz"
)


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """PR = (sum lambda_i)^2 / sum(lambda_i^2)"""
    eigs = eigenvalues[eigenvalues > 0]
    return (eigs.sum() ** 2) / (eigs ** 2).sum()


def two_nn_dimension(X: np.ndarray, n_samples: int | None = None,
                     rng: np.random.Generator | None = None) -> float:
    """Two-NN intrinsic dimension estimator (Facco et al. 2017).

    Uses the ratio mu = r2/r1 of 2nd to 1st NN distances.
    ID = N / sum(log(mu_i))
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if n_samples and X.shape[0] > n_samples:
        idx = rng.choice(X.shape[0], size=n_samples, replace=False)
        X = X[idx]
    nn = NearestNeighbors(n_neighbors=3, algorithm="auto", n_jobs=-1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    # distances[:,0] = 0 (self), distances[:,1] = r1, distances[:,2] = r2
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    valid = r1 > 0
    mu = r2[valid] / r1[valid]
    n = valid.sum()
    id_estimate = n / np.sum(np.log(mu))
    return id_estimate


def levina_bickel_mle(X: np.ndarray, k: int = 10, n_samples: int | None = None,
                      rng: np.random.Generator | None = None) -> dict:
    """Levina-Bickel MLE intrinsic dimension estimator (2005).

    Estimates local dimension at each point from k-NN distances:
      m_k(x) = 1/(k-1) * sum_{j=1}^{k-1} log(T_k(x) / T_j(x))
      d_hat(x) = 1 / m_k(x)

    Returns dict with global mean, median, std, and IQR.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if n_samples and X.shape[0] > n_samples:
        idx = rng.choice(X.shape[0], size=n_samples, replace=False)
        X = X[idx]

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    T = distances[:, 1:]  # [n, k]
    valid = T[:, 0] > 0
    T = T[valid]

    T_k = T[:, k - 1:k]  # [n, 1]
    log_ratios = np.log(T_k / T[:, :k - 1])  # [n, k-1]
    m_k = log_ratios.mean(axis=1)
    d_local = 1.0 / m_k

    reasonable = (d_local > 0) & (d_local < 512) & np.isfinite(d_local)
    d_local_clean = d_local[reasonable]

    return {
        "global_mean": d_local_clean.mean(),
        "global_median": np.median(d_local_clean),
        "std": d_local_clean.std(),
        "q25": np.percentile(d_local_clean, 25),
        "q75": np.percentile(d_local_clean, 75),
        "n_valid": len(d_local_clean),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings_path", type=str, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--n_subsample", type=int, default=50_000,
                        help="Tokens to subsample for PCA fit")
    parser.add_argument("--n_2nn", type=int, default=20_000,
                        help="Points for Two-NN estimation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading embeddings...")
    data = np.load(args.embeddings_path, allow_pickle=True)
    emb = data["embeddings"]  # [n_windows, 3, 512, 1800]
    n_windows, n_channels, d_model, n_patches = emb.shape
    print(f"  Shape: {emb.shape}")

    channel_names = ["ABP", "II", "PLETH"]

    # === All tokens combined ===
    print("\n=== All tokens (all channels combined) ===")
    all_tokens = emb.transpose(0, 1, 3, 2).reshape(-1, d_model).astype(np.float32)
    print(f"  Total tokens: {all_tokens.shape[0]:,} x {d_model}")

    idx = rng.choice(all_tokens.shape[0], size=args.n_subsample, replace=False)
    tokens_sub = all_tokens[idx]

    pca = PCA(n_components=min(512, args.n_subsample))
    pca.fit(tokens_sub)
    pr = participation_ratio(pca.explained_variance_)
    print(f"  PCA participation ratio: {pr:.1f}")

    id_2nn = two_nn_dimension(tokens_sub, n_samples=args.n_2nn, rng=rng)
    print(f"  Two-NN intrinsic dimension: {id_2nn:.1f}")

    print("  Levina-Bickel MLE:")
    for k in [5, 10, 20, 50]:
        result = levina_bickel_mle(tokens_sub, k=k, rng=rng)
        print(f"    k={k:2d}: mean={result['global_mean']:.1f}, "
              f"median={result['global_median']:.1f}, "
              f"IQR=[{result['q25']:.1f}, {result['q75']:.1f}]")

    # === Per channel ===
    print("\n=== Per channel ===")
    for ch_idx, ch_name in enumerate(channel_names):
        ch_tokens = emb[:, ch_idx, :, :].transpose(0, 2, 1).reshape(-1, d_model).astype(np.float32)
        idx_ch = rng.choice(ch_tokens.shape[0], size=args.n_subsample, replace=False)
        ch_sub = ch_tokens[idx_ch]

        pca_ch = PCA(n_components=min(512, args.n_subsample))
        pca_ch.fit(ch_sub)
        pr_ch = participation_ratio(pca_ch.explained_variance_)

        id_2nn_ch = two_nn_dimension(ch_sub, n_samples=args.n_2nn, rng=rng)
        lb_ch = levina_bickel_mle(ch_sub, k=10, rng=rng)
        print(f"  {ch_name}: PR={pr_ch:.1f}, Two-NN={id_2nn_ch:.1f}, "
              f"LB(k=10): mean={lb_ch['global_mean']:.1f}, median={lb_ch['global_median']:.1f}")

    # === Window-level (mean-pooled across patches) ===
    print("\n=== Window-level (mean-pooled across patches) ===")
    window_emb = emb.mean(axis=3).reshape(n_windows, -1).astype(np.float32)  # [1000, 1536]
    print(f"  Window vectors: {window_emb.shape}")

    pca_win = PCA(n_components=min(window_emb.shape))
    pca_win.fit(window_emb)
    pr_win = participation_ratio(pca_win.explained_variance_)
    id_2nn_win = two_nn_dimension(window_emb, rng=rng)
    lb_win = levina_bickel_mle(window_emb, k=10, rng=rng)
    print(f"  PR={pr_win:.1f}, Two-NN={id_2nn_win:.1f}, "
          f"LB(k=10): mean={lb_win['global_mean']:.1f}, median={lb_win['global_median']:.1f}")

    # Per-channel mean-pooled
    print("\n=== Window-level per channel (mean-pooled, 512-dim) ===")
    for ch_idx, ch_name in enumerate(channel_names):
        ch_win = emb[:, ch_idx, :, :].mean(axis=2).astype(np.float32)  # [1000, 512]
        pca_chw = PCA(n_components=min(ch_win.shape))
        pca_chw.fit(ch_win)
        pr_chw = participation_ratio(pca_chw.explained_variance_)
        id_2nn_chw = two_nn_dimension(ch_win, rng=rng)
        lb_chw = levina_bickel_mle(ch_win, k=10, rng=rng)
        print(f"  {ch_name}: PR={pr_chw:.1f}, Two-NN={id_2nn_chw:.1f}, "
              f"LB(k=10): mean={lb_chw['global_mean']:.1f}, median={lb_chw['global_median']:.1f}")


if __name__ == "__main__":
    main()
