"""
Within-patient distance analysis: do pre-hypotensive windows cluster
separately from normal windows within the same patient's trajectory?

For patients with both positive and negative windows, computes pairwise
cosine distances grouped by label (pos-pos, neg-neg, pos-neg).

Requires cached embeddings from cluster_embeddings.py.

Usage:
    python within_patient_distances.py
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-path", type=str,
                        default=str(CLUSTERING_DIR / "embeddings_nfull_seed42.npz"))
    parser.add_argument("--max-neg-per-patient", type=int, default=50,
                        help="Cap neg windows per patient to avoid O(n^2) explosion")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings_path}")
    data = np.load(args.embeddings_path, allow_pickle=True)
    jepa_emb = data["jepa_embeddings"]
    ptst_emb = data["ptst_embeddings"]
    labels = data["labels"]
    patient_ids = data["patient_ids"]

    print(f"Total: {len(labels)} windows, {len(np.unique(patient_ids))} patients")

    # Find patients with BOTH positive and negative windows
    unique_pids = np.unique(patient_ids)
    patients_with_both = []
    for pid in unique_pids:
        mask = patient_ids == pid
        pid_labels = labels[mask]
        if pid_labels.sum() > 0 and (1 - pid_labels).sum() > 0:
            patients_with_both.append(pid)

    print(f"Patients with both pos+neg windows: {len(patients_with_both)}")

    results = []

    for model_name, emb_full in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        print(f"\n--- {model_name} ---")

        pos_pos_dists = []
        neg_neg_dists = []
        pos_neg_dists = []

        for pid in patients_with_both:
            mask = patient_ids == pid
            pid_emb = emb_full[mask]
            pid_labels = labels[mask]

            pos_idx = np.where(pid_labels == 1)[0]
            neg_idx = np.where(pid_labels == 0)[0]

            # Compute pairwise distances within this patient
            dist = cosine_distances(pid_emb)

            # Pos-pos distances
            if len(pos_idx) >= 2:
                for i in range(len(pos_idx)):
                    for j in range(i + 1, len(pos_idx)):
                        pos_pos_dists.append(dist[pos_idx[i], pos_idx[j]])

            # Neg-neg distances (subsample if too many)
            neg_sample = neg_idx[: args.max_neg_per_patient]
            if len(neg_sample) >= 2:
                for i in range(len(neg_sample)):
                    for j in range(i + 1, len(neg_sample)):
                        neg_neg_dists.append(dist[neg_sample[i], neg_sample[j]])

            # Pos-neg distances
            if len(pos_idx) > 0 and len(neg_sample) > 0:
                for i in pos_idx:
                    for j in neg_sample:
                        pos_neg_dists.append(dist[i, j])

        mean_pp = np.mean(pos_pos_dists)
        mean_nn = np.mean(neg_neg_dists)
        mean_pn = np.mean(pos_neg_dists)

        print(f"  Within-patient distances:")
        print(f"    Pos-Pos (both pre-hypotension): {mean_pp:.4f} (n={len(pos_pos_dists)})")
        print(f"    Neg-Neg (both normal):          {mean_nn:.4f} (n={len(neg_neg_dists)})")
        print(f"    Pos-Neg (mixed):                {mean_pn:.4f} (n={len(pos_neg_dists)})")
        print(f"    Ratio Pos-Pos / Pos-Neg:        {mean_pp / mean_pn:.3f}")
        print(f"    Ratio Neg-Neg / Pos-Neg:        {mean_nn / mean_pn:.3f}")

        if mean_pp < mean_pn:
            print(f"    → Positive windows ARE closer to each other within patient")
        else:
            print(f"    → Positive windows are NOT closer to each other within patient")

        results.append({
            "model": model_name,
            "mean_pos_pos_dist": mean_pp,
            "mean_neg_neg_dist": mean_nn,
            "mean_pos_neg_dist": mean_pn,
            "ratio_pp_pn": mean_pp / mean_pn,
            "ratio_nn_pn": mean_nn / mean_pn,
            "n_patients": len(patients_with_both),
            "n_pos_pos_pairs": len(pos_pos_dists),
            "n_neg_neg_pairs": len(neg_neg_dists),
            "n_pos_neg_pairs": len(pos_neg_dists),
        })

    df = pd.DataFrame(results)
    out_path = CLUSTERING_DIR / "within_patient_distances.csv"
    df.to_csv(out_path, index=False)

    print(f"\n\n{'=' * 70}")
    print("WITHIN-PATIENT DISTANCE SUMMARY (HYPOTENSION)")
    print(f"{'=' * 70}")
    print(f"{'':>25} {'JEPA':>12} {'PatchTST':>12}")
    print("-" * 70)
    j, p = results[0], results[1]
    print(f"{'Pos-Pos distance':<25} {j['mean_pos_pos_dist']:>12.4f} {p['mean_pos_pos_dist']:>12.6f}")
    print(f"{'Neg-Neg distance':<25} {j['mean_neg_neg_dist']:>12.4f} {p['mean_neg_neg_dist']:>12.6f}")
    print(f"{'Pos-Neg distance':<25} {j['mean_pos_neg_dist']:>12.4f} {p['mean_pos_neg_dist']:>12.6f}")
    print(f"{'Pos-Pos / Pos-Neg ratio':<25} {j['ratio_pp_pn']:>12.3f} {p['ratio_pp_pn']:>12.3f}")
    print(f"{'Neg-Neg / Pos-Neg ratio':<25} {j['ratio_nn_pn']:>12.3f} {p['ratio_nn_pn']:>12.3f}")
    print(f"{'=' * 70}")
    print(f"\nRatio < 1 = same-label windows are closer within the patient.")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
