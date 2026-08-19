"""K-Means homogeneity analysis sweep across different k values.

Measures how K-Means clusters align with patient identity, hemodynamic state,
and hypotension labels as a function of cluster count.

Usage:
    python cluster_homogeneity_sweep.py

Expects embeddings at:
    /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/
    patient_token_sequences/token_embeddings_20patients_50windows_seed42.npz
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    homogeneity_completeness_v_measure,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    # Load embeddings
    emb_path = (
        "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/"
        "patient_token_sequences/token_embeddings_20patients_50windows_seed42.npz"
    )
    data = np.load(emb_path)
    embeddings = data["embeddings"].astype(np.float32)  # [1000, 3, 512, 1800]
    subject_ids = data["subject_id"]
    hemo_clusters = data["hemo_cluster"]
    hypo_labels = data["hypotension_label"]

    print(f"Embeddings: {embeddings.shape}")
    print(f"Unique patients: {len(np.unique(subject_ids))}")
    print(f"Hemo cluster distribution: {np.unique(hemo_clusters, return_counts=True)}")
    print(f"Hypotension prevalence: {hypo_labels.mean():.3f}")
    print()

    # Temporal subsample: 1800 -> 20 evenly spaced patches
    n_patches_sub = 20
    patch_indices = np.linspace(0, 1799, n_patches_sub, dtype=int)
    emb_sub = embeddings[:, :, :, patch_indices]  # [1000, 3, 512, 20]

    # PCA on feature dim (512 -> 121)
    n_windows = emb_sub.shape[0]
    tokens_flat = emb_sub.transpose(0, 1, 3, 2).reshape(-1, 512)  # [60000, 512]
    pca = PCA(n_components=121, random_state=42)
    tokens_pca = pca.fit_transform(tokens_flat)  # [60000, 121]
    tokens_pca = tokens_pca.reshape(n_windows, 3, n_patches_sub, 121)

    # Flatten per window: [1000, 3*20*121] = [1000, 7260]
    X = tokens_pca.reshape(n_windows, -1)
    X = StandardScaler().fit_transform(X)

    # Map subject_ids to integer labels
    unique_subjects = np.unique(subject_ids)
    subject_int = np.array(
        [np.where(unique_subjects == s)[0][0] for s in subject_ids]
    )

    # K-Means sweep
    k_values = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50]

    header = (
        f"{'k':>3} | {'Sil':>6} | "
        f"{'H(pat)':>7} {'C(pat)':>7} {'V(pat)':>7} | "
        f"{'H(hemo)':>8} {'C(hemo)':>8} {'V(hemo)':>8} | "
        f"{'H(hypo)':>8} {'C(hypo)':>8} {'V(hypo)':>8} | "
        f"{'ARI(pat)':>9} {'ARI(hemo)':>10}"
    )
    print(header)
    print("-" * len(header))

    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        sil = silhouette_score(X, labels, sample_size=min(1000, len(X)))

        h_pat, c_pat, v_pat = homogeneity_completeness_v_measure(
            subject_int, labels
        )
        h_hemo, c_hemo, v_hemo = homogeneity_completeness_v_measure(
            hemo_clusters, labels
        )
        h_hypo, c_hypo, v_hypo = homogeneity_completeness_v_measure(
            hypo_labels, labels
        )

        ari_pat = adjusted_rand_score(subject_int, labels)
        ari_hemo = adjusted_rand_score(hemo_clusters, labels)

        print(
            f"{k:>3} | {sil:>6.4f} | "
            f"{h_pat:>7.4f} {c_pat:>7.4f} {v_pat:>7.4f} | "
            f"{h_hemo:>8.4f} {c_hemo:>8.4f} {v_hemo:>8.4f} | "
            f"{h_hypo:>8.4f} {c_hypo:>8.4f} {v_hypo:>8.4f} | "
            f"{ari_pat:>9.4f} {ari_hemo:>10.4f}"
        )


if __name__ == "__main__":
    main()
