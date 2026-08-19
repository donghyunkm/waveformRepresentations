"""
k-NN neighborhood analysis of JEPA/PatchTST embeddings.

Tests local structure: do nearby embeddings share clinical labels,
hemodynamic phenotypes, or patient identity?

Requires cached embeddings from cluster_embeddings.py (--skip-extraction).

Usage:
    python knn_analysis.py [--n-subsample 20000] [--ks 5,10,20,50,100]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

# ── Paths ─────────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
CLUSTERING_DIR = DERIVED_ROOT / "probing/clustering"
ICU_OUTPUT = Path("/gpfs/home/dk5565/icuDataExtraction/output_v2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subsample", type=int, default=20000,
                        help="Number of samples for k-NN (0 = use all, slow)")
    parser.add_argument("--ks", type=str, default="5,10,20,50,100",
                        help="Comma-separated k values")
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


def cross_patient_knn(emb_sub, labels_sub, hemo_sub, pids_sub, valid_hemo_sub, ks, model_name):
    """Run k-NN analysis excluding same-patient neighbors."""
    n_sub = len(emb_sub)
    n_fetch = 300  # fetch extra to have enough after filtering

    nn = NearestNeighbors(n_neighbors=n_fetch + 1, metric="cosine", n_jobs=-1)
    nn.fit(emb_sub)
    distances, indices = nn.kneighbors(emb_sub)
    all_neighbors = indices[:, 1:]

    results = []
    for k in ks:
        knn_preds = np.zeros(n_sub)
        same_hemo_counts = []
        n_skipped = 0

        for i in range(n_sub):
            my_pid = pids_sub[i]
            neighbor_mask = pids_sub[all_neighbors[i]] != my_pid
            cross_patient_nn = all_neighbors[i][neighbor_mask]

            if len(cross_patient_nn) < k:
                knn_preds[i] = labels_sub.mean()
                n_skipped += 1
                continue

            cp_nn = cross_patient_nn[:k]
            knn_preds[i] = labels_sub[cp_nn].mean()

            if valid_hemo_sub[i]:
                same_hemo_counts.append((hemo_sub[cp_nn] == hemo_sub[i]).mean())

        knn_auroc = roc_auc_score(labels_sub, knn_preds)
        mean_same_hemo = np.mean(same_hemo_counts) if same_hemo_counts else np.nan

        hemo_valid = hemo_sub[valid_hemo_sub]
        cluster_props = np.bincount(hemo_valid, minlength=7) / len(hemo_valid)
        expected_same = (cluster_props ** 2).sum()

        results.append({
            "model": model_name, "k": k,
            "knn_auroc_hypo": knn_auroc,
            "same_hemo_cluster": mean_same_hemo,
            "expected_same_hemo": expected_same,
            "n_skipped": n_skipped,
        })

        print(
            f"  k={k:>3}: kNN_AUROC={knn_auroc:.3f}, "
            f"same_hemo={mean_same_hemo:.3f} (chance={expected_same:.3f}), "
            f"skipped={n_skipped}"
        )

    return results


def main():
    args = parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings_path}")
    data = np.load(args.embeddings_path, allow_pickle=True)
    jepa_emb = data["jepa_embeddings"]
    ptst_emb = data["ptst_embeddings"]
    labels = data["labels"]
    patient_ids = data["patient_ids"]

    print(f"  {jepa_emb.shape[0]} samples, {len(np.unique(patient_ids))} patients")

    # Load hemodynamic clusters
    patient_cluster = load_hemodynamic_clusters()
    hemo_labels = np.array([patient_cluster.get(str(pid), -1) for pid in patient_ids])
    valid_hemo = hemo_labels >= 0
    print(f"  Hemodynamic mapping: {valid_hemo.sum()}/{len(patient_ids)} "
          f"({100 * valid_hemo.mean():.1f}%)")

    # Subsample
    rng = np.random.default_rng(args.seed)
    n_sub = len(jepa_emb) if args.n_subsample == 0 else min(args.n_subsample, len(jepa_emb))
    if n_sub < len(jepa_emb):
        sub_idx = rng.choice(len(jepa_emb), size=n_sub, replace=False)
    else:
        sub_idx = np.arange(len(jepa_emb))

    jepa_sub = jepa_emb[sub_idx]
    ptst_sub = ptst_emb[sub_idx]
    labels_sub = labels[sub_idx]
    hemo_sub = hemo_labels[sub_idx]
    pids_sub = patient_ids[sub_idx]
    valid_hemo_sub = hemo_sub >= 0

    print(f"\nSubsample: {n_sub} windows")
    print(f"  Hypotension pos rate: {labels_sub.mean():.3f}")
    print(f"  Hemodynamic mapped: {valid_hemo_sub.sum()}/{n_sub}")

    results = []

    for model_name, emb_sub in [("JEPA", jepa_sub), ("PatchTST", ptst_sub)]:
        print(f"\n--- {model_name} ---")

        # Fit k-NN (use largest k, then slice)
        nn = NearestNeighbors(n_neighbors=max(ks) + 1, metric="cosine", n_jobs=-1)
        nn.fit(emb_sub)
        distances, indices = nn.kneighbors(emb_sub)

        # Remove self (index 0)
        neighbor_indices = indices[:, 1:]

        for k in ks:
            nn_idx = neighbor_indices[:, :k]

            # 1. Hypotension: k-NN AUROC
            nn_labels = labels_sub[nn_idx]
            pos_mask = labels_sub == 1
            neg_mask = labels_sub == 0

            if pos_mask.sum() > 0:
                pos_neighbor_rate = nn_labels[pos_mask].mean()
                neg_neighbor_rate = nn_labels[neg_mask].mean()
                knn_pred = nn_labels.mean(axis=1)
                knn_auroc = roc_auc_score(labels_sub, knn_pred)
            else:
                pos_neighbor_rate = neg_neighbor_rate = knn_auroc = np.nan

            # 2. Hemodynamic cluster: fraction of neighbors with same cluster
            if valid_hemo_sub.sum() > 100:
                nn_hemo = hemo_sub[nn_idx]
                same_cluster = (nn_hemo == hemo_sub[:, None])
                mean_same_cluster = same_cluster[valid_hemo_sub].mean()
                # Expected by chance
                hemo_valid = hemo_sub[valid_hemo_sub]
                cluster_props = np.bincount(hemo_valid, minlength=7) / len(hemo_valid)
                expected_same = (cluster_props ** 2).sum()
            else:
                mean_same_cluster = expected_same = np.nan

            # 3. Same patient: fraction of neighbors from same patient
            nn_pids = pids_sub[nn_idx]
            same_patient = (nn_pids == pids_sub[:, None]).mean()

            results.append({
                "model": model_name,
                "k": k,
                "knn_auroc_hypo": knn_auroc,
                "pos_neighbor_pos_rate": pos_neighbor_rate,
                "neg_neighbor_pos_rate": neg_neighbor_rate,
                "same_hemo_cluster": mean_same_cluster,
                "expected_same_hemo": expected_same,
                "same_patient_rate": same_patient,
            })

            print(
                f"  k={k:>3}: kNN_AUROC={knn_auroc:.3f}, "
                f"same_hemo={mean_same_cluster:.3f} (chance={expected_same:.3f}), "
                f"same_patient={same_patient:.3f}"
            )

    df = pd.DataFrame(results)
    out_path = CLUSTERING_DIR / "knn_analysis.csv"
    df.to_csv(out_path, index=False)

    # Summary table
    print(f"\n\n{'=' * 75}")
    print("k-NN NEIGHBORHOOD ANALYSIS")
    print(f"{'=' * 75}")
    print(
        f"{'Model':<10} {'k':>4} {'kNN AUROC':>10} {'Same Hemo':>10} "
        f"{'(chance)':>9} {'Same Patient':>13}"
    )
    print("-" * 75)
    for _, r in df.iterrows():
        print(
            f"{r['model']:<10} {r['k']:>4} {r['knn_auroc_hypo']:>10.3f} "
            f"{r['same_hemo_cluster']:>10.3f} {r['expected_same_hemo']:>9.3f} "
            f"{r['same_patient_rate']:>13.3f}"
        )
    print(f"{'=' * 75}")
    print(f"\nSaved to {out_path}")

    # ── Cross-patient analysis ────────────────────────────────────────────────
    print(f"\n\n{'=' * 75}")
    print("CROSS-PATIENT k-NN (same-patient neighbors excluded)")
    print(f"{'=' * 75}")

    cp_results = []
    for model_name, emb_sub_cp in [("JEPA", jepa_sub), ("PatchTST", ptst_sub)]:
        print(f"\n--- {model_name} (cross-patient) ---")
        cp_results.extend(
            cross_patient_knn(emb_sub_cp, labels_sub, hemo_sub, pids_sub,
                              valid_hemo_sub, ks, model_name)
        )

    cp_df = pd.DataFrame(cp_results)
    cp_out_path = CLUSTERING_DIR / "knn_cross_patient.csv"
    cp_df.to_csv(cp_out_path, index=False)

    print(f"\n\n{'=' * 75}")
    print("CROSS-PATIENT RESULTS")
    print(f"{'=' * 75}")
    print(
        f"{'Model':<10} {'k':>4} {'kNN AUROC':>10} {'Same Hemo':>10} "
        f"{'(chance)':>9}"
    )
    print("-" * 75)
    for _, r in cp_df.iterrows():
        print(
            f"{r['model']:<10} {r['k']:>4} {r['knn_auroc_hypo']:>10.3f} "
            f"{r['same_hemo_cluster']:>10.3f} {r['expected_same_hemo']:>9.3f}"
        )
    print(f"{'=' * 75}")
    print(f"\nSaved to {cp_out_path}")


if __name__ == "__main__":
    main()
