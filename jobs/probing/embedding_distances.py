"""
Embedding distance analysis: mean pairwise cosine distances between groups.

Tests whether same-group windows are embedded closer together than
different-group windows, for patient identity, hemodynamic phenotype,
and hypotension label.

Requires cached embeddings from cluster_embeddings.py.

Usage:
    python embedding_distances.py [--n-subsample 10000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

# ── Paths ─────────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
CLUSTERING_DIR = DERIVED_ROOT / "probing/clustering"
ICU_OUTPUT = Path("/gpfs/home/dk5565/icuDataExtraction/output_v2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subsample", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings-path", type=str,
                        default=str(CLUSTERING_DIR / "embeddings_nfull_seed42.npz"))
    return parser.parse_args()


def load_hemodynamic_clusters() -> dict:
    """Load per-patient dominant hemodynamic cluster from icuDataExtraction."""
    cluster_labels = np.load(ICU_OUTPUT / "cluster_labels.npy")
    patient_ids = np.load(ICU_OUTPUT / "patient_ids.npy", allow_pickle=True)

    patient_cluster = {}
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        counts = np.bincount(cluster_labels[mask], minlength=7)
        patient_cluster[str(pid)] = int(np.argmax(counts))

    return patient_cluster


def main():
    args = parse_args()

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings_path}")
    data = np.load(args.embeddings_path, allow_pickle=True)
    jepa_emb = data["jepa_embeddings"]
    ptst_emb = data["ptst_embeddings"]
    labels = data["labels"]
    patient_ids = data["patient_ids"]

    # Load hemodynamic clusters
    patient_cluster = load_hemodynamic_clusters()
    hemo_labels = np.array([patient_cluster.get(str(pid), -1) for pid in patient_ids])

    # Subsample
    rng = np.random.default_rng(args.seed)
    n_sub = min(args.n_subsample, len(jepa_emb))
    sub_idx = rng.choice(len(jepa_emb), size=n_sub, replace=False)

    results = []

    for model_name, emb_full in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        print(f"\n--- {model_name} ---")
        emb = emb_full[sub_idx]
        pids = patient_ids[sub_idx]
        hemo = hemo_labels[sub_idx]
        labs = labels[sub_idx]

        # Pairwise cosine distances
        dist = cosine_distances(emb)
        np.fill_diagonal(dist, np.nan)

        # Same patient vs different patient
        same_patient_mask = (pids[:, None] == pids[None, :])
        np.fill_diagonal(same_patient_mask, False)
        diff_patient_mask = ~same_patient_mask
        np.fill_diagonal(diff_patient_mask, False)

        mean_same_patient = dist[same_patient_mask].mean()
        mean_diff_patient = dist[diff_patient_mask].mean()

        # Same hemo vs different hemo
        valid = hemo >= 0
        hemo_v = hemo[valid]
        dist_v = cosine_distances(emb[valid])
        np.fill_diagonal(dist_v, np.nan)

        same_hemo_mask = (hemo_v[:, None] == hemo_v[None, :])
        np.fill_diagonal(same_hemo_mask, False)
        diff_hemo_mask = ~same_hemo_mask
        np.fill_diagonal(diff_hemo_mask, False)

        mean_same_hemo = dist_v[same_hemo_mask].mean()
        mean_diff_hemo = dist_v[diff_hemo_mask].mean()

        # Same hemo, DIFFERENT patient (pure phenotype signal)
        pids_v = pids[valid]
        same_hemo_diff_patient = same_hemo_mask & (pids_v[:, None] != pids_v[None, :])
        diff_hemo_diff_patient = diff_hemo_mask & (pids_v[:, None] != pids_v[None, :])

        mean_same_hemo_diff_patient = dist_v[same_hemo_diff_patient].mean()
        mean_diff_hemo_diff_patient = dist_v[diff_hemo_diff_patient].mean()

        # Same hypo label vs different
        same_label_mask = (labs[:, None] == labs[None, :])
        np.fill_diagonal(same_label_mask, False)
        diff_label_mask = ~same_label_mask
        np.fill_diagonal(diff_label_mask, False)

        mean_same_label = dist[same_label_mask].mean()
        mean_diff_label = dist[diff_label_mask].mean()

        print(f"  Same patient dist:       {mean_same_patient:.4f}")
        print(f"  Diff patient dist:       {mean_diff_patient:.4f}")
        print(f"  Ratio:                   {mean_same_patient / mean_diff_patient:.3f}")
        print()
        print(f"  Same hemo dist:          {mean_same_hemo:.4f}")
        print(f"  Diff hemo dist:          {mean_diff_hemo:.4f}")
        print(f"  Ratio:                   {mean_same_hemo / mean_diff_hemo:.3f}")
        print()
        print(f"  Same hemo, diff patient: {mean_same_hemo_diff_patient:.4f}")
        print(f"  Diff hemo, diff patient: {mean_diff_hemo_diff_patient:.4f}")
        print(f"  Ratio (pure phenotype):  {mean_same_hemo_diff_patient / mean_diff_hemo_diff_patient:.3f}")
        print()
        print(f"  Same hypo label dist:    {mean_same_label:.4f}")
        print(f"  Diff hypo label dist:    {mean_diff_label:.4f}")
        print(f"  Ratio:                   {mean_same_label / mean_diff_label:.3f}")

        results.append({
            "model": model_name,
            "same_patient_dist": mean_same_patient,
            "diff_patient_dist": mean_diff_patient,
            "same_hemo_dist": mean_same_hemo,
            "diff_hemo_dist": mean_diff_hemo,
            "same_hemo_diff_patient_dist": mean_same_hemo_diff_patient,
            "diff_hemo_diff_patient_dist": mean_diff_hemo_diff_patient,
            "same_label_dist": mean_same_label,
            "diff_label_dist": mean_diff_label,
        })

    df = pd.DataFrame(results)
    out_path = CLUSTERING_DIR / "embedding_distances.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
