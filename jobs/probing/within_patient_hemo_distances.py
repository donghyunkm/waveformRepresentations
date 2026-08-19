"""
Within-patient hemodynamic cluster distance analysis.

Aligns PhysioJEPA windows to icuDataExtraction windows via per-patient
time offset correction, assigns window-level hemodynamic clusters, then
tests whether same-cluster windows are closer within a patient's trajectory.

Requires:
- Cached embeddings from cluster_embeddings.py
- icuDataExtraction output_v2 (window_times, patient_ids, cluster_labels)
- PhysioJEPA test sample cache

Usage:
    python within_patient_hemo_distances.py
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

# ── Paths ─────────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
CLUSTERING_DIR = DERIVED_ROOT / "probing/clustering"
ICU_OUTPUT = Path("/gpfs/home/dk5565/icuDataExtraction/output_v2")
SAMPLE_CACHE_DIR = DERIVED_ROOT / "models/fcn_hypotension_paper"
DATASET_FILENAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"


def get_physio_center(row):
    """Compute window center in icuDataExtraction reference frame.
    
    icuDataExtraction uses seconds since 2000-01-01.
    PhysioJEPA filenames give POSIX timestamps (since 1970-01-01).
    Subtract EPOCH_OFFSET to convert.
    """
    EPOCH_OFFSET = 946684800.0  # seconds between 1970-01-01 and 2000-01-01
    match = re.search(r"p\d+-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.zarr", row["file_path"])
    y, mo, d, h, mi = [int(x) for x in match.groups()]
    seg_start = datetime(y, mo, d, h, mi, 0).timestamp()
    center_posix = seg_start + (row["start_idx"] + row["end_idx"]) / 2 / 125
    return center_posix - EPOCH_OFFSET


def align_windows(samples: pd.DataFrame, max_tolerance_sec: float = 150) -> np.ndarray:
    """
    Align PhysioJEPA windows to icuDataExtraction windows using epoch offset
    correction. Returns window-level hemodynamic cluster labels (-1 for unmatched).

    icuDataExtraction uses seconds since 2000-01-01 as its time reference.
    PhysioJEPA container filenames encode dates that produce POSIX timestamps
    (since 1970-01-01). The get_physio_center() function already subtracts
    the epoch offset to produce times in icuDataExtraction's reference frame.

    Tolerance default is 150s (2.5 min = icuDataExtraction's anchor stride).
    """
    icu_window_times = np.load(ICU_OUTPUT / "window_times.npy")
    icu_patient_ids = np.load(ICU_OUTPUT / "patient_ids.npy", allow_pickle=True)
    icu_clusters = np.load(ICU_OUTPUT / "cluster_labels.npy")

    # Build per-patient sorted lookup
    icu_by_patient = defaultdict(list)
    for i in range(len(icu_patient_ids)):
        icu_by_patient[str(icu_patient_ids[i])].append(i)

    icu_times_by_patient = {}
    icu_clusters_by_patient = {}
    for pid, indices in icu_by_patient.items():
        indices = np.array(indices)
        times = icu_window_times[indices]
        sort_order = np.argsort(times)
        icu_times_by_patient[pid] = times[sort_order]
        icu_clusters_by_patient[pid] = icu_clusters[indices[sort_order]]

    # Find overlapping patients
    physio_pids = set(samples["subject_id"].unique())
    icu_pids_set = set(icu_by_patient.keys())
    overlap = list(physio_pids & icu_pids_set)

    # Compute window centers in icu reference frame
    samples_centers = samples.apply(get_physio_center, axis=1).values

    matched_clusters = np.full(len(samples), -1, dtype=int)
    n_matched = 0

    for pid in overlap:
        pid_mask = samples["subject_id"] == pid
        pid_indices = samples.index[pid_mask].values
        physio_centers = samples_centers[pid_mask]

        if pid not in icu_times_by_patient:
            continue
        icu_times = icu_times_by_patient[pid]
        icu_cls = icu_clusters_by_patient[pid]

        for idx, center in zip(pid_indices, physio_centers):
            search_idx = np.searchsorted(icu_times, center)
            best_dist = np.inf
            best_cluster = -1
            for candidate in [search_idx - 1, search_idx]:
                if 0 <= candidate < len(icu_times):
                    dist = abs(icu_times[candidate] - center)
                    if dist < best_dist:
                        best_dist = dist
                        best_cluster = icu_cls[candidate]

            if best_dist <= max_tolerance_sec:
                matched_clusters[idx] = best_cluster
                n_matched += 1

    print(f"Matched: {n_matched}/{len(samples)} ({100 * n_matched / len(samples):.1f}%)")
    return matched_clusters


def main():
    # Load test samples
    samples = pd.read_csv(
        SAMPLE_CACHE_DIR / f"{DATASET_FILENAME}-test_samples.csv.gz"
    )

    # Load or compute window-level hemo clusters
    cache_path = CLUSTERING_DIR / "window_hemo_clusters.npz"
    if cache_path.is_file():
        print(f"Loading cached window clusters from {cache_path}")
        hemo_data = np.load(cache_path, allow_pickle=True)
        hemo_clusters = hemo_data["hemo_clusters"]
    else:
        print("Aligning windows...")
        hemo_clusters = align_windows(samples)
        np.savez_compressed(cache_path,
                            hemo_clusters=hemo_clusters,
                            patient_ids=samples["subject_id"].values)

    # Load embeddings
    data = np.load(CLUSTERING_DIR / "embeddings_nfull_seed42.npz", allow_pickle=True)
    jepa_emb = data["jepa_embeddings"]
    ptst_emb = data["ptst_embeddings"]
    patient_ids = data["patient_ids"]

    print(f"Windows with hemo labels: {(hemo_clusters >= 0).sum()}/{len(hemo_clusters)}")

    # Find patients with multiple hemodynamic clusters
    unique_pids = np.unique(patient_ids)
    patients_multi_hemo = []
    for pid in unique_pids:
        mask = patient_ids == pid
        pid_hemo = hemo_clusters[mask]
        valid = pid_hemo[pid_hemo >= 0]
        if len(valid) > 10 and len(np.unique(valid)) >= 2:
            patients_multi_hemo.append(pid)

    print(f"Patients with 2+ hemo clusters and >10 labeled windows: {len(patients_multi_hemo)}")

    results = []

    for model_name, emb_full in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        print(f"\n--- {model_name} ---")

        same_cluster_dists = []
        diff_cluster_dists = []

        for pid in patients_multi_hemo:
            mask = patient_ids == pid
            pid_emb = emb_full[mask]
            pid_hemo = hemo_clusters[mask]

            valid = pid_hemo >= 0
            if valid.sum() < 5:
                continue

            emb_v = pid_emb[valid]
            hemo_v = pid_hemo[valid]

            # Cap to avoid huge distance matrices
            if len(emb_v) > 100:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(emb_v), size=100, replace=False)
                emb_v = emb_v[idx]
                hemo_v = hemo_v[idx]

            dist = cosine_distances(emb_v)
            np.fill_diagonal(dist, np.nan)

            same_mask = (hemo_v[:, None] == hemo_v[None, :])
            np.fill_diagonal(same_mask, False)
            diff_mask = ~same_mask
            np.fill_diagonal(diff_mask, False)

            if same_mask.any():
                same_cluster_dists.extend(dist[same_mask].tolist())
            if diff_mask.any():
                diff_cluster_dists.extend(dist[diff_mask].tolist())

        mean_same = np.mean(same_cluster_dists)
        mean_diff = np.mean(diff_cluster_dists)
        ratio = mean_same / mean_diff

        print(f"  Same hemo cluster (within patient): {mean_same:.4f} (n={len(same_cluster_dists)})")
        print(f"  Diff hemo cluster (within patient): {mean_diff:.4f} (n={len(diff_cluster_dists)})")
        print(f"  Ratio: {ratio:.3f}")

        if ratio < 1:
            print(f"  → Windows with same hemo cluster ARE closer within patient")
        else:
            print(f"  → Windows with same hemo cluster are NOT closer within patient")

        results.append({
            "model": model_name,
            "same_hemo_dist": mean_same,
            "diff_hemo_dist": mean_diff,
            "ratio": ratio,
            "n_patients": len(patients_multi_hemo),
            "n_same_pairs": len(same_cluster_dists),
            "n_diff_pairs": len(diff_cluster_dists),
        })

    df = pd.DataFrame(results)
    out_path = CLUSTERING_DIR / "within_patient_hemo_distances.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 70}")
    print("WITHIN-PATIENT HEMODYNAMIC CLUSTER DISTANCES")
    print(f"{'=' * 70}")
    print(f"{'':>35} {'JEPA':>10} {'PatchTST':>10}")
    print("-" * 70)
    j, p = results[0], results[1]
    print(f"{'Same hemo dist (within patient)':<35} {j['same_hemo_dist']:>10.4f} {p['same_hemo_dist']:>10.6f}")
    print(f"{'Diff hemo dist (within patient)':<35} {j['diff_hemo_dist']:>10.4f} {p['diff_hemo_dist']:>10.6f}")
    print(f"{'Ratio (same/diff)':<35} {j['ratio']:>10.3f} {p['ratio']:>10.3f}")
    print(f"{'=' * 70}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
