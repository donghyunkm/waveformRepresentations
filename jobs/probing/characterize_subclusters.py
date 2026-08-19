"""
Characterize per-patient embedding sub-clusters.

For selected patients, computes physiological features per window and analyzes
what distinguishes each sub-cluster found by cluster_single_patient.py:
  - Mean HR, MAP, SBP, DBP, PP, PLETH amplitude
  - Temporal position (hours from first window)
  - Recording segment (file boundary alignment)
  - Signal quality (NaN fraction, amplitude range)
  - Waveform morphology stats

This helps explain what the encoder is using to organize the embedding space
within a single patient.

Requires cached embeddings and the Zarr containers.

Usage:
    python characterize_subclusters.py --patient-ids p097441,p011342,p057886
    python characterize_subclusters.py --top-n 5 --model jepa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.signal import find_peaks

sys.path.insert(0, "/gpfs/home/dk5565/PhysioJEPA")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Paths ─────────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
CLUSTERING_DIR = DERIVED_ROOT / "probing/clustering"
SAMPLE_CACHE_DIR = DERIVED_ROOT / "models/fcn_hypotension_paper"
DATASET_FILENAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"
EMBEDDINGS_PATH = CLUSTERING_DIR / "embeddings_nfull_seed42.npz"
HEMO_CLUSTERS_PATH = CLUSTERING_DIR / "window_hemo_clusters.npz"

OUTPUT_DIR = CLUSTERING_DIR / "per_patient"

FS = 125  # Hz
WINDOW_SAMPLES = 1800 * FS  # 225000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Characterize what per-patient embedding sub-clusters represent."
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--patient-ids", type=str, default=None,
                        help="Comma-separated patient IDs")
    parser.add_argument("--model", type=str, default="jepa", choices=["jepa", "patchtst"])
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--n-clusters", type=int, default=0,
                        help="Fixed k (0 = auto-select)")
    parser.add_argument("--max-windows-for-features", type=int, default=200,
                        help="Max windows per patient to read from Zarr (feature computation)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Feature extraction from waveforms ─────────────────────────────────────────

def compute_window_features(abp: np.ndarray, ecg: np.ndarray, pleth: np.ndarray) -> dict:
    """
    Compute lightweight physiological features from a 30-min window.
    Returns dict of scalar features or NaN if extraction fails.
    """
    features = {}

    # ── Signal quality ────────────────────────────────────────────────────
    features["abp_nan_frac"] = np.isnan(abp).mean()
    features["ecg_nan_frac"] = np.isnan(ecg).mean()
    features["pleth_nan_frac"] = np.isnan(pleth).mean()

    # Replace NaN for computation
    abp_clean = np.nan_to_num(abp, nan=np.nanmean(abp) if not np.all(np.isnan(abp)) else 0)
    ecg_clean = np.nan_to_num(ecg, nan=np.nanmean(ecg) if not np.all(np.isnan(ecg)) else 0)
    pleth_clean = np.nan_to_num(pleth, nan=np.nanmean(pleth) if not np.all(np.isnan(pleth)) else 0)

    # ── ABP features ─────────────────────────────────────────────────────
    # Use 10-second chunks to get beat-level stats
    chunk_size = 10 * FS  # 10 seconds
    sbps, dbps = [], []
    for i in range(0, len(abp_clean) - chunk_size, chunk_size):
        chunk = abp_clean[i:i + chunk_size]
        if np.std(chunk) < 1.0:  # flat/dead signal
            continue
        peaks, _ = find_peaks(chunk, distance=int(0.4 * FS), height=30)
        troughs, _ = find_peaks(-chunk, distance=int(0.4 * FS))
        if len(peaks) > 2:
            sbps.extend(chunk[peaks].tolist())
        if len(troughs) > 2:
            dbps.extend(chunk[troughs].tolist())

    if sbps:
        features["SBP"] = np.median(sbps)
        features["SBP_std"] = np.std(sbps)
    else:
        features["SBP"] = np.nan
        features["SBP_std"] = np.nan

    if dbps:
        features["DBP"] = np.median(dbps)
    else:
        features["DBP"] = np.nan

    if sbps and dbps:
        features["MAP"] = features["DBP"] + (features["SBP"] - features["DBP"]) / 3
        features["PP"] = features["SBP"] - features["DBP"]
    else:
        features["MAP"] = np.nan
        features["PP"] = np.nan

    # ── HR from ECG ──────────────────────────────────────────────────────
    # Detect R-peaks in 30-second segments, take median
    hrs = []
    seg_size = 30 * FS
    for i in range(0, len(ecg_clean) - seg_size, seg_size):
        seg = ecg_clean[i:i + seg_size]
        if np.std(seg) < 0.01:
            continue
        peaks, _ = find_peaks(seg, distance=int(0.4 * FS),
                              height=np.percentile(seg, 70))
        if len(peaks) > 3:
            rr_intervals = np.diff(peaks) / FS
            rr_valid = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
            if len(rr_valid) > 2:
                hrs.append(60.0 / np.median(rr_valid))

    if hrs:
        features["HR"] = np.median(hrs)
        features["HR_std"] = np.std(hrs)
        # HRV approximation (std of HR across segments)
        features["HR_range"] = np.max(hrs) - np.min(hrs)
    else:
        features["HR"] = np.nan
        features["HR_std"] = np.nan
        features["HR_range"] = np.nan

    # Shock index
    if features["HR"] and features["SBP"] and not np.isnan(features["HR"]) and not np.isnan(features["SBP"]) and features["SBP"] > 0:
        features["ShockIdx"] = features["HR"] / features["SBP"]
    else:
        features["ShockIdx"] = np.nan

    # ── PLETH features ───────────────────────────────────────────────────
    # Amplitude (peak-trough range in normalized signal)
    pleth_amps = []
    for i in range(0, len(pleth_clean) - chunk_size, chunk_size):
        chunk = pleth_clean[i:i + chunk_size]
        if np.std(chunk) < 0.001:
            continue
        pleth_amps.append(np.percentile(chunk, 95) - np.percentile(chunk, 5))

    if pleth_amps:
        features["PLETH_amp"] = np.median(pleth_amps)
        features["PLETH_amp_std"] = np.std(pleth_amps)
    else:
        features["PLETH_amp"] = np.nan
        features["PLETH_amp_std"] = np.nan

    # ── Waveform morphology (overall signal stats) ────────────────────────
    features["abp_mean"] = np.nanmean(abp)
    features["abp_std"] = np.nanstd(abp)
    features["ecg_std"] = np.nanstd(ecg)
    features["pleth_mean"] = np.nanmean(pleth)
    features["pleth_std"] = np.nanstd(pleth)

    return features


def read_window(file_path: str, start_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read ABP, II, PLETH from Zarr container for one window."""
    try:
        store = zarr.ZipStore(file_path, mode='r')
        root = zarr.open(store, mode='r')
        end_idx = start_idx + WINDOW_SAMPLES
        abp = np.array(root["ABP"][start_idx:end_idx], dtype=np.float32)
        ecg = np.array(root["II"][start_idx:end_idx], dtype=np.float32)
        pleth = np.array(root["PLETH"][start_idx:end_idx], dtype=np.float32)
        store.close()
        return abp, ecg, pleth
    except Exception as e:
        return None


# ── Per-patient characterization ──────────────────────────────────────────────

def characterize_patient(
    patient_id: str,
    embeddings: np.ndarray,
    hypo_labels: np.ndarray,
    hemo_labels: np.ndarray,
    sample_rows: pd.DataFrame,
    n_clusters: int,
    max_k: int,
    max_windows_for_features: int,
    seed: int,
    model_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Cluster a patient's embeddings and characterize each cluster."""

    n_windows = len(embeddings)
    print(f"\n{'═'*70}")
    print(f"  Patient {patient_id} — {n_windows} windows, model={model_name}")
    print(f"{'═'*70}")

    # Sort by temporal order
    time_order = np.argsort(sample_rows["start_idx"].values)
    embeddings = embeddings[time_order]
    hypo_labels = hypo_labels[time_order]
    hemo_labels = hemo_labels[time_order]
    sample_rows = sample_rows.iloc[time_order].reset_index(drop=True)

    start_indices = sample_rows["start_idx"].values
    file_paths = sample_rows["file_path"].values
    time_hours = (start_indices - start_indices[0]) / (FS * 3600)

    # ── Cluster ───────────────────────────────────────────────────────────
    if n_clusters > 0:
        k = n_clusters
    else:
        best_k, best_sil = 2, -1.0
        for kk in range(2, min(max_k + 1, n_windows // 10)):
            km = KMeans(n_clusters=kk, random_state=seed, n_init=10)
            labs = km.fit_predict(embeddings)
            s = silhouette_score(embeddings, labs)
            if s > best_sil:
                best_sil = s
                best_k = kk
        k = best_k

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(embeddings)
    print(f"  Clusters: k={k}")

    # ── Recording segment assignment ──────────────────────────────────────
    unique_files = list(pd.unique(file_paths))
    file_idx = np.array([unique_files.index(f) for f in file_paths])
    file_names = [Path(f).stem for f in unique_files]

    # ── Sample windows for feature extraction ─────────────────────────────
    rng = np.random.default_rng(seed)
    # Stratified by cluster: sample proportionally
    sample_indices = []
    per_cluster_budget = max_windows_for_features // k
    for c in range(k):
        c_idx = np.where(cluster_labels == c)[0]
        n_sample = min(per_cluster_budget, len(c_idx))
        chosen = rng.choice(c_idx, size=n_sample, replace=False)
        sample_indices.extend(chosen.tolist())
    sample_indices = sorted(sample_indices)

    print(f"  Reading {len(sample_indices)} windows from Zarr for feature extraction...")

    # ── Extract features ──────────────────────────────────────────────────
    feature_rows = []
    for idx in sample_indices:
        fp = file_paths[idx]
        si = start_indices[idx]
        signals = read_window(fp, si)
        if signals is None:
            continue
        abp, ecg, pleth = signals
        feats = compute_window_features(abp, ecg, pleth)
        feats["window_idx"] = idx
        feats["cluster"] = cluster_labels[idx]
        feats["time_hours"] = time_hours[idx]
        feats["file_idx"] = file_idx[idx]
        feats["file_name"] = file_names[file_idx[idx]]
        feats["hypo_label"] = hypo_labels[idx]
        feats["hemo_cluster"] = hemo_labels[idx]
        feature_rows.append(feats)

    if not feature_rows:
        print("  ERROR: No features extracted!")
        return pd.DataFrame(), {}

    feat_df = pd.DataFrame(feature_rows)
    print(f"  Extracted features for {len(feat_df)} windows")

    # ── Characterize each cluster ─────────────────────────────────────────
    print(f"\n  {'─'*66}")
    print(f"  CLUSTER CHARACTERIZATION")
    print(f"  {'─'*66}")

    # Key features to summarize
    key_features = ["HR", "SBP", "DBP", "MAP", "PP", "ShockIdx",
                    "PLETH_amp", "HR_std", "abp_std", "pleth_std"]

    cluster_summaries = []
    for c in range(k):
        c_mask_all = cluster_labels == c
        c_feat = feat_df[feat_df["cluster"] == c]

        # Recording segment distribution
        c_files = file_idx[c_mask_all]
        file_counts = np.bincount(c_files, minlength=len(unique_files))
        dominant_file = np.argmax(file_counts)
        file_purity = file_counts[dominant_file] / c_mask_all.sum()

        # Temporal span
        c_times = time_hours[c_mask_all]

        summary = {
            "cluster": c,
            "size": int(c_mask_all.sum()),
            "pos_rate": float(hypo_labels[c_mask_all].mean()),
            "time_mean_h": float(c_times.mean()),
            "time_std_h": float(c_times.std()),
            "time_min_h": float(c_times.min()),
            "time_max_h": float(c_times.max()),
            "dominant_file": file_names[dominant_file],
            "file_purity": file_purity,
            "n_files": int((file_counts > 0).sum()),
        }

        # Add feature medians and mean±std
        for feat in key_features:
            if feat in c_feat.columns:
                summary[f"{feat}_median"] = c_feat[feat].median()
                summary[f"{feat}_mean"] = c_feat[feat].mean()
                summary[f"{feat}_std"] = c_feat[feat].std()
                summary[f"{feat}_iqr"] = c_feat[feat].quantile(0.75) - c_feat[feat].quantile(0.25)

        cluster_summaries.append(summary)

    summary_df = pd.DataFrame(cluster_summaries)

    # Print median table
    print(f"\n  {'Clust':>5} {'Size':>5} {'Pos%':>5} {'Time(h)':>9} "
          f"{'FilePur':>7} {'#Files':>6} {'HR':>6} {'MAP':>6} {'SBP':>6} "
          f"{'PP':>5} {'PLETH':>6}")
    print(f"  {'-'*80}")
    for _, row in summary_df.iterrows():
        hr_str = f"{row.get('HR_median', np.nan):>6.0f}" if not pd.isna(row.get("HR_median")) else "   N/A"
        map_str = f"{row.get('MAP_median', np.nan):>6.0f}" if not pd.isna(row.get("MAP_median")) else "   N/A"
        sbp_str = f"{row.get('SBP_median', np.nan):>6.0f}" if not pd.isna(row.get("SBP_median")) else "   N/A"
        pp_str = f"{row.get('PP_median', np.nan):>5.0f}" if not pd.isna(row.get("PP_median")) else "  N/A"
        pleth_str = f"{row.get('PLETH_amp_median', np.nan):>6.2f}" if not pd.isna(row.get("PLETH_amp_median")) else "   N/A"
        print(f"  C{int(row['cluster']):>3} {int(row['size']):>5} "
              f"{row['pos_rate']*100:>4.1f}% "
              f"{row['time_mean_h']:>5.1f}±{row['time_std_h']:<3.1f} "
              f"{row['file_purity']:>6.1%} {int(row['n_files']):>5} "
              f"{hr_str} {map_str} {sbp_str} {pp_str} {pleth_str}")

    # Print mean ± std table
    mean_features = ["HR", "SBP", "DBP", "MAP", "PP", "ShockIdx", "PLETH_amp"]
    print(f"\n  Per-cluster features (mean ± std):")
    print(f"  {'Clust':>5}", end="")
    for feat in mean_features:
        print(f" {feat:>14}", end="")
    print()
    print(f"  {'-'*(5 + 15*len(mean_features))}")
    for _, row in summary_df.iterrows():
        c = int(row["cluster"])
        print(f"  C{c:>3}", end="")
        for feat in mean_features:
            mean_val = row.get(f"{feat}_mean")
            std_val = row.get(f"{feat}_std")
            if pd.isna(mean_val):
                print(f" {'N/A':>14}", end="")
            elif feat in ("ShockIdx", "PLETH_amp"):
                print(f" {mean_val:>5.2f}±{std_val:<5.2f}", end="")
            else:
                print(f" {mean_val:>5.1f}±{std_val:<5.1f}", end="")
        print()

    # ── File-cluster alignment analysis ───────────────────────────────────
    print(f"\n  Recording segment × Cluster contingency:")
    print(f"  {'File':<45} ", end="")
    for c in range(k):
        print(f" C{c:>2}", end="")
    print(f"  Total")

    for fi, fname in enumerate(file_names):
        fi_mask = file_idx == fi
        total_in_file = fi_mask.sum()
        if total_in_file == 0:
            continue
        print(f"  {fname:<45} ", end="")
        for c in range(k):
            n = ((cluster_labels == c) & fi_mask).sum()
            print(f" {n:>3}", end="")
        print(f"  {total_in_file:>5}")

    # ── Most discriminating features between clusters ─────────────────────
    print(f"\n  Feature differences between clusters (median):")
    for feat in key_features:
        if feat not in feat_df.columns:
            continue
        vals = []
        for c in range(k):
            c_feat = feat_df[feat_df["cluster"] == c][feat]
            vals.append(c_feat.median())
        if any(pd.isna(v) for v in vals):
            continue
        spread = max(vals) - min(vals)
        mean_val = np.nanmean(vals)
        if mean_val != 0:
            cv = spread / abs(mean_val) * 100
            print(f"    {feat:<12}: range={spread:>7.1f} ({cv:>5.1f}% of mean)  "
                  f"[{' | '.join(f'C{c}:{v:.1f}' for c, v in enumerate(vals))}]")

    # ── Summary interpretation ────────────────────────────────────────────
    # Check if clusters are primarily driven by file boundaries
    from sklearn.metrics import adjusted_rand_score
    ari_file = adjusted_rand_score(file_idx, cluster_labels)
    print(f"\n  ARI(clusters, recording_segments) = {ari_file:.3f}")
    if ari_file > 0.3:
        print(f"  → Sub-clusters are STRONGLY driven by recording segment boundaries")
    elif ari_file > 0.1:
        print(f"  → Sub-clusters are MODERATELY driven by recording segments")
    else:
        print(f"  → Sub-clusters are NOT primarily driven by recording segments")

    meta = {
        "patient_id": patient_id,
        "k": k,
        "ari_file_segment": ari_file,
        "n_files": len(unique_files),
        "n_windows": n_windows,
    }

    return summary_df, meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    print(f"Loading embeddings...")
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    if args.model == "jepa":
        all_embeddings = data["jepa_embeddings"]
    else:
        all_embeddings = data["ptst_embeddings"]
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

    # Select patients
    if args.patient_ids:
        target_pids = [p.strip() for p in args.patient_ids.split(",")]
    else:
        pid_counts = pd.Series(all_patient_ids).value_counts()
        # Patients with >=200 windows and both labels
        candidates = []
        for pid, count in pid_counts.items():
            if count < 200:
                continue
            mask = all_patient_ids == pid
            n_pos = all_labels[mask].sum()
            candidates.append((pid, count, n_pos))
        candidates.sort(key=lambda x: (x[2] > 0, x[1]), reverse=True)
        target_pids = [c[0] for c in candidates[:args.top_n]]

    print(f"  Patients to analyze: {target_pids}")

    # Run characterization
    all_summaries = []
    all_metas = []

    for pid in target_pids:
        mask = all_patient_ids == pid
        if mask.sum() < 30:
            print(f"  Skipping {pid}: {mask.sum()} windows")
            continue

        idx = np.where(mask)[0]
        emb = all_embeddings[idx]
        labs = all_labels[idx]
        hemo = all_hemo_labels[idx]
        rows = sample_df.iloc[idx].copy().reset_index(drop=True)

        summary_df, meta = characterize_patient(
            patient_id=pid,
            embeddings=emb,
            hypo_labels=labs,
            hemo_labels=hemo,
            sample_rows=rows,
            n_clusters=args.n_clusters,
            max_k=args.max_k,
            max_windows_for_features=args.max_windows_for_features,
            seed=args.seed,
            model_name=args.model,
        )

        if not summary_df.empty:
            summary_df["patient_id"] = pid
            all_summaries.append(summary_df)
            all_metas.append(meta)

    # Save combined results
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        out_path = OUTPUT_DIR / f"subcluster_characterization_{args.model}.csv"
        combined.to_csv(out_path, index=False)
        print(f"\n\nSaved characterization to {out_path}")

        # Summary of what drives clusters
        print(f"\n{'═'*70}")
        print(f"  OVERALL: What drives per-patient sub-clusters?")
        print(f"{'═'*70}")
        for m in all_metas:
            driver = ("RECORDING SEGMENT" if m["ari_file_segment"] > 0.3
                      else "MIXED" if m["ari_file_segment"] > 0.1
                      else "PHYSIOLOGICAL STATE")
            print(f"  {m['patient_id']}: ARI_file={m['ari_file_segment']:.3f} "
                  f"(k={m['k']}, {m['n_files']} files, {m['n_windows']} windows) "
                  f"→ {driver}")


if __name__ == "__main__":
    main()
