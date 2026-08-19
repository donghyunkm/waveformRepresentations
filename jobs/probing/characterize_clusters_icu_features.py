"""
Characterize pooled embedding clusters using icuDataExtraction's 19 features.

Instead of computing features from raw waveforms, matches each PhysioJEPA
embedding window to its nearest icuDataExtraction window (via epoch-offset
time alignment) and uses the precomputed X_stats features.

Reports both median and mean±std per cluster for all 19 features.

Usage:
    python characterize_clusters_icu_features.py --patient-ids p052529,p011342,p057886,p093560,p097441,p098276
    python characterize_clusters_icu_features.py --top-n 6 --model jepa
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

sys.path.insert(0, "/gpfs/home/dk5565/PhysioJEPA")
sys.path.insert(0, "/gpfs/home/dk5565/icuDataExtraction")
from config import FEATURE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
CLUSTERING_DIR = DERIVED_ROOT / "probing/clustering"
SAMPLE_CACHE_DIR = DERIVED_ROOT / "models/fcn_hypotension_paper"
DATASET_FILENAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"
EMBEDDINGS_PATH = CLUSTERING_DIR / "embeddings_nfull_seed42.npz"
HEMO_CLUSTERS_PATH = CLUSTERING_DIR / "window_hemo_clusters.npz"

ICU_OUTPUT = Path("/gpfs/home/dk5565/icuDataExtraction/output_v2")

OUTPUT_DIR = CLUSTERING_DIR / "pooled"

EPOCH_OFFSET = 946684800.0  # seconds between 1970-01-01 and 2000-01-01
FS = 125


def parse_args():
    parser = argparse.ArgumentParser(
        description="Characterize embedding clusters using icuDataExtraction features."
    )
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--patient-ids", type=str, default=None)
    parser.add_argument("--model", type=str, default="jepa", choices=["jepa", "patchtst"])
    parser.add_argument("--n-clusters", type=int, default=0,
                        help="Fixed k (0 = auto via silhouette sweep)")
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--tolerance-sec", type=float, default=150.0,
                        help="Max time difference for window matching (default 150s = 2.5 min)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Time alignment ────────────────────────────────────────────────────────────

def get_physio_center(file_path: str, start_idx: int, end_idx: int) -> float:
    """Compute window center in icuDataExtraction reference frame (seconds since 2000)."""
    match = re.search(r"p\d+-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.zarr", file_path)
    y, mo, d, h, mi = [int(x) for x in match.groups()]
    seg_start = datetime(y, mo, d, h, mi, 0).timestamp()
    center_posix = seg_start + (start_idx + end_idx) / 2 / FS
    return center_posix - EPOCH_OFFSET


def match_windows_to_icu(
    sample_df: pd.DataFrame,
    tolerance_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match PhysioJEPA windows to icuDataExtraction windows.
    
    Returns:
        matched_indices: array of icuDataExtraction indices (-1 for unmatched)
        match_offsets: time offset in seconds for matched windows
    """
    # Load ICU data
    icu_window_times = np.load(ICU_OUTPUT / "window_times.npy")
    icu_patient_ids = np.load(ICU_OUTPUT / "patient_ids.npy", allow_pickle=True)

    # Build per-patient sorted lookup
    icu_by_patient = defaultdict(list)
    for i in range(len(icu_patient_ids)):
        icu_by_patient[str(icu_patient_ids[i])].append(i)

    icu_times_by_patient = {}
    icu_sorted_indices = {}
    for pid, indices in icu_by_patient.items():
        indices = np.array(indices)
        times = icu_window_times[indices]
        sort_order = np.argsort(times)
        icu_times_by_patient[pid] = times[sort_order]
        icu_sorted_indices[pid] = indices[sort_order]

    # Compute PhysioJEPA window centers
    centers = np.array([
        get_physio_center(row["file_path"], row["start_idx"], row["end_idx"])
        for _, row in sample_df.iterrows()
    ])

    matched_indices = np.full(len(sample_df), -1, dtype=int)
    match_offsets = np.full(len(sample_df), np.nan)

    for pid in sample_df["subject_id"].unique():
        if pid not in icu_times_by_patient:
            continue
        pid_mask = sample_df["subject_id"] == pid
        pid_df_indices = np.where(pid_mask)[0]
        physio_centers = centers[pid_mask]

        icu_times = icu_times_by_patient[pid]
        icu_idx = icu_sorted_indices[pid]

        # Binary search for nearest match
        for i, center in enumerate(physio_centers):
            insert_pos = np.searchsorted(icu_times, center)
            best_dist = np.inf
            best_idx = -1

            for candidate in [insert_pos - 1, insert_pos]:
                if 0 <= candidate < len(icu_times):
                    dist = abs(icu_times[candidate] - center)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = candidate

            if best_dist <= tolerance_sec:
                df_idx = pid_df_indices[i]
                matched_indices[df_idx] = icu_idx[best_idx]
                match_offsets[df_idx] = best_dist

    return matched_indices, match_offsets


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print("Loading embeddings...")
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    all_embeddings = data["jepa_embeddings"] if args.model == "jepa" else data["ptst_embeddings"]
    all_labels = data["labels"]
    all_patient_ids = data["patient_ids"]

    # Load hemo clusters
    if HEMO_CLUSTERS_PATH.is_file():
        hemo_data = np.load(HEMO_CLUSTERS_PATH, allow_pickle=True)
        all_hemo_labels = hemo_data["hemo_clusters"]
    else:
        all_hemo_labels = np.full(len(all_labels), -1, dtype=int)

    # Load sample cache
    cache_path = SAMPLE_CACHE_DIR / f"{DATASET_FILENAME}-test_samples.csv.gz"
    sample_df = pd.read_csv(cache_path)
    print(f"  {len(all_labels)} windows, {len(np.unique(all_patient_ids))} patients")

    # Load icuDataExtraction features
    print("Loading icuDataExtraction X_stats...")
    X_stats = np.load(ICU_OUTPUT / "X_stats.npy", mmap_mode='r')
    print(f"  X_stats: {X_stats.shape} ({FEATURE_NAMES})")

    # Select patients
    if args.patient_ids:
        target_pids = [p.strip() for p in args.patient_ids.split(",")]
    else:
        pid_counts = pd.Series(all_patient_ids).value_counts()
        candidates = []
        for pid, count in pid_counts.items():
            if count < 200:
                continue
            mask = all_patient_ids == pid
            n_pos = all_labels[mask].sum()
            candidates.append((pid, count, n_pos))
        candidates.sort(key=lambda x: (x[2] > 0, x[1]), reverse=True)
        target_pids = [c[0] for c in candidates[:args.top_n]]

    # Pool selected patients
    pool_mask = np.isin(all_patient_ids, target_pids)
    embeddings = all_embeddings[pool_mask]
    labels = all_labels[pool_mask]
    patient_ids = all_patient_ids[pool_mask]
    hemo_labels = all_hemo_labels[pool_mask]
    pool_sample_df = sample_df.iloc[np.where(pool_mask)[0]].reset_index(drop=True)

    n_total = len(embeddings)
    print(f"\n  Pooled: {n_total} windows from {len(target_pids)} patients")

    # ── Match to icuDataExtraction ────────────────────────────────────────
    print(f"\n  Matching to icuDataExtraction (tolerance={args.tolerance_sec}s)...")
    matched_indices, match_offsets = match_windows_to_icu(pool_sample_df, args.tolerance_sec)
    n_matched = (matched_indices >= 0).sum()
    print(f"  Matched: {n_matched}/{n_total} ({100*n_matched/n_total:.1f}%)")
    if n_matched > 0:
        valid_offsets = match_offsets[matched_indices >= 0]
        print(f"  Mean offset: {np.nanmean(valid_offsets):.1f}s, "
              f"median: {np.nanmedian(valid_offsets):.1f}s")

    # ── Cluster ───────────────────────────────────────────────────────────
    if args.n_clusters > 0:
        k = args.n_clusters
    else:
        print(f"\n  Silhouette sweep k=2..{args.max_k}...")
        rng = np.random.default_rng(args.seed)
        n_sil = min(5000, n_total)
        sil_idx = rng.choice(n_total, size=n_sil, replace=False)
        best_k, best_sil = 2, -1.0
        for kk in range(2, args.max_k + 1):
            km = KMeans(n_clusters=kk, random_state=args.seed, n_init=10)
            all_labs = km.fit_predict(embeddings)
            s = silhouette_score(embeddings[sil_idx], all_labs[sil_idx])
            if s > best_sil:
                best_sil = s
                best_k = kk
        k = best_k
        print(f"  Selected k={k} (silhouette={best_sil:.3f})")

    km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
    cluster_labels = km.fit_predict(embeddings)

    # ── Extract features for matched windows ──────────────────────────────
    # For each matched window, get the mean feature across 109 sub-windows
    matched_mask = matched_indices >= 0
    matched_icu_idx = matched_indices[matched_mask]

    # Extract features: mean over 109 sub-windows per feature
    print(f"\n  Extracting mean features for {n_matched} matched windows...")
    # X_stats shape: (N_icu, 19, 109)
    # We want: for each matched window, mean across 109 sub-windows → (n_matched, 19)
    features_matched = np.nanmean(X_stats[matched_icu_idx], axis=2)  # (n_matched, 19)
    cluster_labels_matched = cluster_labels[matched_mask]
    labels_matched = labels[matched_mask]
    patient_ids_matched = patient_ids[matched_mask]

    print(f"  Feature matrix: {features_matched.shape}")
    print(f"  NaN fraction: {np.isnan(features_matched).mean():.4f}")

    # ── Per-cluster feature summary ───────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  CLUSTER CHARACTERIZATION — {args.model.upper()}, k={k}, "
          f"{n_matched} matched windows, 19 icuDataExtraction features")
    print(f"{'='*100}")

    # Feature names (skip RESP-dependent ones that are mostly NaN)
    all_feature_names = FEATURE_NAMES
    # Check which features have enough data
    valid_features = []
    for fi, fname in enumerate(all_feature_names):
        nan_rate = np.isnan(features_matched[:, fi]).mean()
        if nan_rate < 0.5:
            valid_features.append((fi, fname))
    print(f"\n  Valid features ({len(valid_features)}/19, <50% NaN):")
    for fi, fname in valid_features:
        nan_rate = np.isnan(features_matched[:, fi]).mean()
        print(f"    {fi:>2}. {fname:<12} NaN={nan_rate:.1%}")

    # ── Median table ──────────────────────────────────────────────────────
    print(f"\n  Per-cluster feature MEDIANS:")
    header = f"  {'C':>3} {'n':>5} {'Pos%':>5}"
    for fi, fname in valid_features:
        header += f" {fname:>10}"
    print(header)
    print(f"  {'-'*len(header)}")

    cluster_rows = []
    for c in range(k):
        c_mask = cluster_labels_matched == c
        n_c = c_mask.sum()
        if n_c == 0:
            continue
        pos_rate = labels_matched[c_mask].mean()
        row = {"cluster": c, "n_matched": int(n_c), "pos_rate": float(pos_rate)}
        line = f"  C{c:>2} {n_c:>5} {pos_rate*100:>4.1f}%"
        for fi, fname in valid_features:
            vals = features_matched[c_mask, fi]
            med = np.nanmedian(vals)
            row[f"{fname}_median"] = med
            if fname in ("PLETH_ACDC", "ShockIdx", "ABP_tau", "PLETH_amp"):
                line += f" {med:>10.3f}"
            elif fname in ("HRV_RMSSD", "dPdt_max"):
                line += f" {med:>10.1f}"
            else:
                line += f" {med:>10.1f}"
        print(line)
        cluster_rows.append(row)

    # ── Mean ± std table ──────────────────────────────────────────────────
    print(f"\n  Per-cluster feature MEAN ± STD:")
    header2 = f"  {'C':>3} {'n':>5}"
    for fi, fname in valid_features:
        header2 += f" {fname:>16}"
    print(header2)
    print(f"  {'-'*len(header2)}")

    # Build a lookup from cluster id to row index
    cluster_to_row = {row["cluster"]: i for i, row in enumerate(cluster_rows)}

    for c in range(k):
        c_mask = cluster_labels_matched == c
        n_c = c_mask.sum()
        if n_c == 0:
            continue
        line = f"  C{c:>2} {n_c:>5}"
        row_idx = cluster_to_row[c]
        for fi, fname in valid_features:
            vals = features_matched[c_mask, fi]
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            cluster_rows[row_idx][f"{fname}_mean"] = mean
            cluster_rows[row_idx][f"{fname}_std"] = std
            if fname in ("PLETH_ACDC", "ShockIdx", "ABP_tau", "PLETH_amp"):
                line += f" {mean:>6.3f}±{std:<6.3f}"
            elif fname in ("HRV_RMSSD", "dPdt_max"):
                line += f" {mean:>6.1f}±{std:<6.1f}"
            else:
                line += f" {mean:>6.1f}±{std:<6.1f}"
        print(line)

    # ── Feature variation across clusters ─────────────────────────────────
    print(f"\n  Feature variation across clusters (range of medians / grand mean):")
    for fi, fname in valid_features:
        medians = []
        for c in range(k):
            c_mask = cluster_labels_matched == c
            if c_mask.sum() > 0:
                medians.append(np.nanmedian(features_matched[c_mask, fi]))
        if not medians:
            continue
        spread = max(medians) - min(medians)
        grand_mean = np.nanmean(features_matched[:, fi])
        if abs(grand_mean) > 1e-6:
            cv = spread / abs(grand_mean) * 100
            print(f"    {fname:<12}: range={spread:>9.2f}  ({cv:>5.1f}% of mean={grand_mean:.2f})")

    # ── ARI analysis ──────────────────────────────────────────────────────
    pid_to_int = {pid: i for i, pid in enumerate(target_pids)}
    patient_int = np.array([pid_to_int[pid] for pid in patient_ids])
    file_paths = pool_sample_df["file_path"].values
    unique_files = list(pd.unique(file_paths))
    file_idx = np.array([unique_files.index(f) for f in file_paths])

    ari_patient = adjusted_rand_score(patient_int, cluster_labels)
    ari_file = adjusted_rand_score(file_idx, cluster_labels)
    ari_hypo = adjusted_rand_score(labels, cluster_labels)
    print(f"\n  ARI vs patient: {ari_patient:.4f}")
    print(f"  ARI vs recording: {ari_file:.4f}")
    print(f"  ARI vs hypotension: {ari_hypo:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(cluster_rows)
    out_path = OUTPUT_DIR / f"cluster_characterization_icu_features_{args.model}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
