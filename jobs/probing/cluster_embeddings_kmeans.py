"""K-Means clustering of JEPA token embeddings with temporal structure preserved.

Applies PCA on the 512-dim token space (reducing each token from 512 → 121 dims),
preserving the full [n_patches, d_pca] temporal sequence per window. Then
flattens each window to a single vector [n_channels * n_patches * d_pca] for
K-Means clustering.

This retains temporal structure in the representation — two windows with the
same mean but different temporal patterns will cluster differently.

Usage:
    python cluster_embeddings_kmeans.py [--n_pca 121] [--k_min 2] [--k_max 15]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    homogeneity_completeness_v_measure,
)
from sklearn.preprocessing import StandardScaler


DEFAULT_EMBEDDINGS = (
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/"
    "patient_token_sequences/token_embeddings_20patients_50windows_seed42.npz"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings_path", type=str, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--n_pca", type=int, default=121,
                        help="Number of PCA components for 512-dim reduction")
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal_subsample", type=int, default=20,
                        help="Subsample patches from 1800 to this many (evenly spaced) "
                             "to keep clustering tractable. 0 = no subsampling.")
    args = parser.parse_args()

    # --- Load embeddings ---
    print("Loading embeddings...")
    data = np.load(args.embeddings_path, allow_pickle=True)
    emb = data["embeddings"]  # [n_windows, 3, 512, 1800]
    hemo_clusters = data["hemo_cluster"]
    subject_ids = data["subject_id"]
    hypo_labels = data["hypotension_label"]

    n_windows, n_channels, d_model, n_patches = emb.shape
    print(f"  Shape: {emb.shape}")
    print(f"  Hemo cluster distribution: {np.bincount(hemo_clusters[hemo_clusters >= 0].astype(int), minlength=7)}")
    print(f"  Unmatched (cluster=-1): {(hemo_clusters == -1).sum()}")

    # --- Temporal subsampling ---
    if args.temporal_subsample > 0 and args.temporal_subsample < n_patches:
        patch_indices = np.linspace(0, n_patches - 1, args.temporal_subsample, dtype=int)
        emb = emb[:, :, :, patch_indices]
        n_patches_sub = args.temporal_subsample
        print(f"\n  Temporal subsample: {n_patches} → {n_patches_sub} patches")
    else:
        n_patches_sub = n_patches
        print(f"\n  No temporal subsampling ({n_patches} patches)")

    # --- PCA on the 512-dim token space ---
    # Flatten all tokens across windows/channels/patches to fit PCA
    print(f"\nFitting PCA on 512-dim token space → {args.n_pca} dims...")
    all_tokens = emb.transpose(0, 1, 3, 2).reshape(-1, d_model).astype(np.float32)
    # all_tokens: [n_windows * n_channels * n_patches_sub, 512]
    print(f"  Total tokens: {all_tokens.shape[0]:,}")

    # Subsample tokens for PCA fit (500k is plenty)
    rng = np.random.default_rng(args.seed)
    max_fit = 500_000
    if all_tokens.shape[0] > max_fit:
        fit_idx = rng.choice(all_tokens.shape[0], size=max_fit, replace=False)
        pca = PCA(n_components=args.n_pca, random_state=args.seed)
        pca.fit(all_tokens[fit_idx])
    else:
        pca = PCA(n_components=args.n_pca, random_state=args.seed)
        pca.fit(all_tokens)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"  Variance explained by {args.n_pca} components: {cumvar[-1]*100:.1f}%")

    # Transform all tokens
    all_tokens_pca = pca.transform(all_tokens)
    # Reshape back: [n_windows, n_channels, n_patches_sub, n_pca]
    emb_pca = all_tokens_pca.reshape(n_windows, n_channels, n_patches_sub, args.n_pca)

    # --- Flatten to window-level vectors preserving temporal structure ---
    # [n_windows, n_channels * n_patches_sub * n_pca]
    window_vectors = emb_pca.reshape(n_windows, -1)
    print(f"\n  Window vector dimensionality: {window_vectors.shape[1]:,} "
          f"({n_channels} ch × {n_patches_sub} patches × {args.n_pca} pca)")

    # Standardize
    scaler = StandardScaler()
    window_final = scaler.fit_transform(window_vectors)

    # --- K-Means sweep ---
    print(f"\nK-Means sweep (k={args.k_min}–{args.k_max})...")
    print(f"{'k':>3} {'Inertia':>12} {'Silhouette':>11} {'ARI vs Hemo':>12} {'NMI vs Hemo':>12}")
    print("-" * 55)

    results = []
    valid_hemo_mask = hemo_clusters >= 0

    for k in range(args.k_min, args.k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed, max_iter=300)
        labels = km.fit_predict(window_final)

        sil = silhouette_score(window_final, labels)
        inertia = km.inertia_

        # Compare with hemo clusters (only valid ones)
        if valid_hemo_mask.sum() > 0:
            ari = adjusted_rand_score(hemo_clusters[valid_hemo_mask], labels[valid_hemo_mask])
            nmi = normalized_mutual_info_score(hemo_clusters[valid_hemo_mask], labels[valid_hemo_mask])
        else:
            ari = nmi = float("nan")

        results.append({
            "k": k, "inertia": inertia, "silhouette": sil,
            "ari_hemo": ari, "nmi_hemo": nmi, "labels": labels,
        })
        print(f"{k:3d} {inertia:12.0f} {sil:11.4f} {ari:12.4f} {nmi:12.4f}")

    # --- Detailed analysis at k=7 (matching hemo clusters) ---
    print("\n\n=== Detailed analysis at k=7 ===")
    k7 = next(r for r in results if r["k"] == 7)
    labels_7 = k7["labels"]

    # Homogeneity / completeness / v-measure vs hemo
    if valid_hemo_mask.sum() > 0:
        h, c, v = homogeneity_completeness_v_measure(
            hemo_clusters[valid_hemo_mask], labels_7[valid_hemo_mask]
        )
        print(f"  vs Hemo clusters: ARI={k7['ari_hemo']:.4f}, NMI={k7['nmi_hemo']:.4f}")
        print(f"  Homogeneity={h:.4f}, Completeness={c:.4f}, V-measure={v:.4f}")

    # Cluster vs patient: are clusters patient-specific?
    h_pat, c_pat, v_pat = homogeneity_completeness_v_measure(subject_ids, labels_7)
    print(f"\n  vs Patient ID: Homogeneity={h_pat:.4f}, Completeness={c_pat:.4f}")
    print(f"  (High homogeneity = clusters are patient-specific)")

    # Cluster vs hypotension
    ari_hypo = adjusted_rand_score(hypo_labels, labels_7)
    nmi_hypo = normalized_mutual_info_score(hypo_labels, labels_7)
    print(f"\n  vs Hypotension label: ARI={ari_hypo:.4f}, NMI={nmi_hypo:.4f}")

    # Hypotension prevalence per cluster
    print(f"\n  Hypotension prevalence per cluster:")
    for c_id in range(7):
        mask = labels_7 == c_id
        if mask.sum() > 0:
            prev = hypo_labels[mask].mean()
            print(f"    Cluster {c_id}: {mask.sum():4d} windows, "
                  f"hypo prevalence={prev*100:.1f}%")

    # --- Find optimal k ---
    best_sil = max(results, key=lambda r: r["silhouette"])
    print(f"\n\n=== Summary ===")
    print(f"  Best silhouette: k={best_sil['k']} (score={best_sil['silhouette']:.4f})")
    print(f"  k=7 silhouette: {k7['silhouette']:.4f}")
    print(f"  k=7 ARI vs hemo: {k7['ari_hemo']:.4f}")


if __name__ == "__main__":
    main()
