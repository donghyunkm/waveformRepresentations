"""
Pooled multi-patient clustering of embeddings.

Pools all windows from selected patients, clusters the combined embedding
space jointly, then characterizes what drives cluster membership:
  - Patient identity (are clusters = patients?)
  - Hemodynamic regime (HR, MAP, SBP, PP)
  - Recording segment (file boundaries)
  - Temporal position within each patient's stay
  - Hypotension label

This answers: when windows from different patients are mixed, does the encoder
group them by patient or by shared physiological state?

Requires cached embeddings and Zarr containers.

Usage:
    python cluster_pooled_patients.py --top-n 6 --model jepa
    python cluster_pooled_patients.py --patient-ids p052529,p011342,p057886,p093560,p097441,p098276
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
import umap
import zarr
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

sys.path.insert(0, "/gpfs/home/dk5565/PhysioJEPA")

# ── Paths ─────────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
CLUSTERING_DIR = DERIVED_ROOT / "probing/clustering"
SAMPLE_CACHE_DIR = DERIVED_ROOT / "models/fcn_hypotension_paper"
DATASET_FILENAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"
EMBEDDINGS_PATH = CLUSTERING_DIR / "embeddings_nfull_seed42.npz"
HEMO_CLUSTERS_PATH = CLUSTERING_DIR / "window_hemo_clusters.npz"

OUTPUT_DIR = CLUSTERING_DIR / "pooled"

FS = 125
WINDOW_SAMPLES = 1800 * FS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pooled multi-patient clustering of embeddings."
    )
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--patient-ids", type=str, default=None)
    parser.add_argument("--model", type=str, default="jepa", choices=["jepa", "patchtst"])
    parser.add_argument("--n-clusters", type=int, default=0,
                        help="Fixed k (0 = sweep 2..max_k)")
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--max-windows-for-features", type=int, default=300,
                        help="Max windows to read from Zarr for feature extraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-umap", action="store_true")
    return parser.parse_args()


# ── Feature extraction (same as characterize_subclusters.py) ──────────────────

def compute_window_features(abp: np.ndarray, ecg: np.ndarray, pleth: np.ndarray) -> dict:
    """Compute lightweight physiological features from a 30-min window."""
    features = {}

    features["abp_nan_frac"] = np.isnan(abp).mean()
    features["ecg_nan_frac"] = np.isnan(ecg).mean()
    features["pleth_nan_frac"] = np.isnan(pleth).mean()

    abp_clean = np.nan_to_num(abp, nan=np.nanmean(abp) if not np.all(np.isnan(abp)) else 0)
    ecg_clean = np.nan_to_num(ecg, nan=np.nanmean(ecg) if not np.all(np.isnan(ecg)) else 0)
    pleth_clean = np.nan_to_num(pleth, nan=np.nanmean(pleth) if not np.all(np.isnan(pleth)) else 0)

    chunk_size = 10 * FS
    sbps, dbps = [], []
    for i in range(0, len(abp_clean) - chunk_size, chunk_size):
        chunk = abp_clean[i:i + chunk_size]
        if np.std(chunk) < 1.0:
            continue
        peaks, _ = find_peaks(chunk, distance=int(0.4 * FS), height=30)
        troughs, _ = find_peaks(-chunk, distance=int(0.4 * FS))
        if len(peaks) > 2:
            sbps.extend(chunk[peaks].tolist())
        if len(troughs) > 2:
            dbps.extend(chunk[troughs].tolist())

    features["SBP"] = np.median(sbps) if sbps else np.nan
    features["SBP_std"] = np.std(sbps) if sbps else np.nan
    features["DBP"] = np.median(dbps) if dbps else np.nan

    if sbps and dbps:
        features["MAP"] = features["DBP"] + (features["SBP"] - features["DBP"]) / 3
        features["PP"] = features["SBP"] - features["DBP"]
    else:
        features["MAP"] = np.nan
        features["PP"] = np.nan

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

    features["HR"] = np.median(hrs) if hrs else np.nan
    features["HR_std"] = np.std(hrs) if hrs else np.nan

    if features["HR"] and features["SBP"] and not np.isnan(features.get("HR", np.nan)) and not np.isnan(features.get("SBP", np.nan)) and features["SBP"] > 0:
        features["ShockIdx"] = features["HR"] / features["SBP"]
    else:
        features["ShockIdx"] = np.nan

    pleth_amps = []
    for i in range(0, len(pleth_clean) - chunk_size, chunk_size):
        chunk = pleth_clean[i:i + chunk_size]
        if np.std(chunk) < 0.001:
            continue
        pleth_amps.append(np.percentile(chunk, 95) - np.percentile(chunk, 5))

    features["PLETH_amp"] = np.median(pleth_amps) if pleth_amps else np.nan

    features["abp_mean"] = np.nanmean(abp)
    features["abp_std"] = np.nanstd(abp)
    features["pleth_std"] = np.nanstd(pleth)

    return features


def read_window(file_path: str, start_idx: int):
    """Read ABP, II, PLETH from Zarr container."""
    try:
        store = zarr.ZipStore(file_path, mode='r')
        root = zarr.open(store, mode='r')
        end_idx = start_idx + WINDOW_SAMPLES
        abp = np.array(root["ABP"][start_idx:end_idx], dtype=np.float32)
        ecg = np.array(root["II"][start_idx:end_idx], dtype=np.float32)
        pleth = np.array(root["PLETH"][start_idx:end_idx], dtype=np.float32)
        store.close()
        return abp, ecg, pleth
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading embeddings...")
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    all_embeddings = data["jepa_embeddings"] if args.model == "jepa" else data["ptst_embeddings"]
    all_labels = data["labels"]
    all_patient_ids = data["patient_ids"]

    if HEMO_CLUSTERS_PATH.is_file():
        hemo_data = np.load(HEMO_CLUSTERS_PATH, allow_pickle=True)
        all_hemo_labels = hemo_data["hemo_clusters"]
    else:
        all_hemo_labels = np.full(len(all_labels), -1, dtype=int)

    cache_path = SAMPLE_CACHE_DIR / f"{DATASET_FILENAME}-test_samples.csv.gz"
    sample_df = pd.read_csv(cache_path)
    print(f"  {len(all_labels)} windows, {len(np.unique(all_patient_ids))} patients")

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

    # Pool windows from selected patients
    pool_mask = np.isin(all_patient_ids, target_pids)
    embeddings = all_embeddings[pool_mask]
    labels = all_labels[pool_mask]
    patient_ids = all_patient_ids[pool_mask]
    hemo_labels = all_hemo_labels[pool_mask]
    pool_sample_df = sample_df.iloc[np.where(pool_mask)[0]].reset_index(drop=True)

    n_total = len(embeddings)
    n_patients = len(target_pids)
    print(f"\n  Pooled: {n_total} windows from {n_patients} patients")
    print(f"  Patients: {target_pids}")
    print(f"  Positive rate: {labels.mean():.3f}")
    for pid in target_pids:
        m = patient_ids == pid
        print(f"    {pid}: {m.sum()} windows, pos_rate={labels[m].mean():.3f}")

    # File/recording segment info
    file_paths = pool_sample_df["file_path"].values
    unique_files = list(pd.unique(file_paths))
    file_idx = np.array([unique_files.index(f) for f in file_paths])
    file_names = [Path(f).stem for f in unique_files]

    # Patient index (integer encoding)
    pid_to_int = {pid: i for i, pid in enumerate(target_pids)}
    patient_int = np.array([pid_to_int[pid] for pid in patient_ids])

    # ── Silhouette sweep ──────────────────────────────────────────────────
    if args.n_clusters > 0:
        k = args.n_clusters
        k_scores = []
    else:
        print(f"\n  Silhouette sweep k=2..{args.max_k}...")
        k_scores = []
        # Use subsample for speed
        rng = np.random.default_rng(args.seed)
        n_sil = min(5000, n_total)
        sil_idx = rng.choice(n_total, size=n_sil, replace=False)
        best_k, best_sil = 2, -1.0
        for kk in range(2, args.max_k + 1):
            km = KMeans(n_clusters=kk, random_state=args.seed, n_init=10)
            all_labs = km.fit_predict(embeddings)
            s = silhouette_score(embeddings[sil_idx], all_labs[sil_idx])
            k_scores.append((kk, s))
            if s > best_sil:
                best_sil = s
                best_k = kk
            print(f"    k={kk:>2}: sil={s:.3f}", flush=True)
        k = best_k
        print(f"  Selected k={k} (silhouette={best_sil:.3f})")

    # ── KMeans clustering ─────────────────────────────────────────────────
    print(f"\n  Running KMeans (k={k})...")
    km = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
    cluster_labels = km.fit_predict(embeddings)

    # ── ARI/NMI against various groupings ─────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  AGREEMENT ANALYSIS: What do pooled clusters correspond to?")
    print(f"{'='*70}")

    ari_patient = adjusted_rand_score(patient_int, cluster_labels)
    nmi_patient = normalized_mutual_info_score(patient_int, cluster_labels)
    print(f"\n  vs Patient identity:    ARI={ari_patient:.4f}  NMI={nmi_patient:.4f}")

    ari_file = adjusted_rand_score(file_idx, cluster_labels)
    nmi_file = normalized_mutual_info_score(file_idx, cluster_labels)
    print(f"  vs Recording segment:   ARI={ari_file:.4f}  NMI={nmi_file:.4f}")

    ari_hypo = adjusted_rand_score(labels, cluster_labels)
    nmi_hypo = normalized_mutual_info_score(labels, cluster_labels)
    print(f"  vs Hypotension label:   ARI={ari_hypo:.4f}  NMI={nmi_hypo:.4f}")

    valid_hemo = hemo_labels >= 0
    if valid_hemo.sum() > 50:
        ari_hemo = adjusted_rand_score(hemo_labels[valid_hemo], cluster_labels[valid_hemo])
        nmi_hemo = normalized_mutual_info_score(hemo_labels[valid_hemo], cluster_labels[valid_hemo])
        print(f"  vs Hemodynamic cluster: ARI={ari_hemo:.4f}  NMI={nmi_hemo:.4f}")
    else:
        ari_hemo, nmi_hemo = np.nan, np.nan
        print(f"  vs Hemodynamic cluster: N/A (too few mapped)")

    # ── Patient × Cluster contingency ─────────────────────────────────────
    print(f"\n  Patient × Cluster contingency (% of patient's windows in each cluster):")
    print(f"  {'Patient':<10}", end="")
    for c in range(k):
        print(f" C{c:>2}", end="")
    print(f"  Total")

    for pid in target_pids:
        p_mask = patient_ids == pid
        n_p = p_mask.sum()
        print(f"  {pid:<10}", end="")
        for c in range(k):
            n_in_c = ((cluster_labels == c) & p_mask).sum()
            pct = 100 * n_in_c / n_p
            print(f" {pct:>3.0f}", end="")
        print(f"  {n_p:>5}")

    # ── Cluster profiles ──────────────────────────────────────────────────
    print(f"\n  Cluster profiles:")
    print(f"  {'Clust':>5} {'Size':>6} {'Pos%':>5} {'#Pat':>4} {'DomPat':>10} "
          f"{'PatPur':>6} {'#Files':>6}")
    print(f"  {'-'*55}")
    for c in range(k):
        c_mask = cluster_labels == c
        size = c_mask.sum()
        pos_rate = labels[c_mask].mean()
        pids_in_c = patient_ids[c_mask]
        unique_pids_in_c = np.unique(pids_in_c)
        pid_counts_in_c = pd.Series(pids_in_c).value_counts()
        dom_pid = pid_counts_in_c.index[0]
        pat_purity = pid_counts_in_c.iloc[0] / size
        n_files_in_c = len(np.unique(file_idx[c_mask]))
        print(f"  C{c:>3} {size:>6} {pos_rate*100:>4.1f}% {len(unique_pids_in_c):>4} "
              f"{dom_pid:>10} {pat_purity:>5.1%} {n_files_in_c:>5}")

    # ── Feature extraction for cluster characterization ───────────────────
    print(f"\n  Extracting physiological features ({args.max_windows_for_features} windows)...")
    rng = np.random.default_rng(args.seed)
    per_cluster = args.max_windows_for_features // k
    sample_indices = []
    for c in range(k):
        c_idx = np.where(cluster_labels == c)[0]
        n_sample = min(per_cluster, len(c_idx))
        chosen = rng.choice(c_idx, size=n_sample, replace=False)
        sample_indices.extend(chosen.tolist())
    sample_indices = sorted(sample_indices)

    feature_rows = []
    for idx in sample_indices:
        fp = file_paths[idx]
        si = pool_sample_df.iloc[idx]["start_idx"]
        signals = read_window(fp, int(si))
        if signals is None:
            continue
        abp, ecg, pleth = signals
        feats = compute_window_features(abp, ecg, pleth)
        feats["cluster"] = cluster_labels[idx]
        feats["patient_id"] = patient_ids[idx]
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows)
    print(f"  Extracted features for {len(feat_df)} windows")

    # ── Feature summary per cluster ───────────────────────────────────────
    key_features = ["HR", "SBP", "DBP", "MAP", "PP", "ShockIdx", "PLETH_amp"]

    # Median table
    print(f"\n  Physiological features per cluster (median):")
    print(f"  {'Clust':>5}", end="")
    for feat in key_features:
        print(f" {feat:>8}", end="")
    print()
    print(f"  {'-'*(5 + 9*len(key_features))}")

    cluster_feat_summary = []
    for c in range(k):
        c_feat = feat_df[feat_df["cluster"] == c]
        row = {"cluster": c, "n_sampled": len(c_feat)}
        print(f"  C{c:>3}", end="")
        for feat in key_features:
            med = c_feat[feat].median()
            mean = c_feat[feat].mean()
            std = c_feat[feat].std()
            row[f"{feat}_median"] = med
            row[f"{feat}_mean"] = mean
            row[f"{feat}_std"] = std
            if pd.isna(med):
                print(f" {'N/A':>8}", end="")
            elif feat in ("ShockIdx", "PLETH_amp"):
                print(f" {med:>8.2f}", end="")
            else:
                print(f" {med:>8.1f}", end="")
        print()
        cluster_feat_summary.append(row)

    # Mean ± std table
    print(f"\n  Physiological features per cluster (mean ± std):")
    print(f"  {'Clust':>5} {'n':>3}", end="")
    for feat in key_features:
        print(f" {feat:>14}", end="")
    print()
    print(f"  {'-'*(8 + 15*len(key_features))}")

    for row in cluster_feat_summary:
        c = row["cluster"]
        print(f"  C{c:>3} {row['n_sampled']:>3}", end="")
        for feat in key_features:
            mean = row.get(f"{feat}_mean")
            std = row.get(f"{feat}_std")
            if pd.isna(mean):
                print(f" {'N/A':>14}", end="")
            elif feat in ("ShockIdx", "PLETH_amp"):
                print(f" {mean:>5.2f}±{std:<5.2f}", end="")
            else:
                print(f" {mean:>5.1f}±{std:<5.1f}", end="")
        print()

    # ── Feature discrimination: which features vary most across clusters ──
    print(f"\n  Feature variation across clusters:")
    for feat in key_features:
        medians = [feat_df[feat_df["cluster"] == c][feat].median() for c in range(k)]
        medians_clean = [m for m in medians if not pd.isna(m)]
        if not medians_clean:
            continue
        spread = max(medians_clean) - min(medians_clean)
        mean_val = np.mean(medians_clean)
        if abs(mean_val) > 0:
            cv = spread / abs(mean_val) * 100
            print(f"    {feat:<10}: range={spread:>7.1f} ({cv:>5.1f}% of mean)")

    # ── UMAP ──────────────────────────────────────────────────────────────
    if not args.no_umap:
        print(f"\n  Computing UMAP ({n_total} points)...")
        # Subsample for UMAP if too many
        n_umap = min(10000, n_total)
        if n_umap < n_total:
            umap_idx = rng.choice(n_total, size=n_umap, replace=False)
        else:
            umap_idx = np.arange(n_total)

        reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                            metric="cosine", random_state=args.seed)
        coords = reducer.fit_transform(embeddings[umap_idx])

        # 5-panel figure
        fig, axes = plt.subplots(1, 5, figsize=(25, 4.5))

        # Panel 1: KMeans clusters
        sc = axes[0].scatter(coords[:, 0], coords[:, 1],
                             c=cluster_labels[umap_idx], cmap="tab10",
                             s=3, alpha=0.5, rasterized=True)
        axes[0].set_title(f"KMeans Clusters (k={k})")
        plt.colorbar(sc, ax=axes[0])

        # Panel 2: Patient identity
        sc2 = axes[1].scatter(coords[:, 0], coords[:, 1],
                              c=patient_int[umap_idx], cmap="tab10",
                              s=3, alpha=0.5, rasterized=True)
        axes[1].set_title(f"Patient Identity ({n_patients} patients)")
        plt.colorbar(sc2, ax=axes[1])

        # Panel 3: Hypotension label
        colors_hypo = np.where(labels[umap_idx] == 1, "crimson", "steelblue")
        axes[2].scatter(coords[:, 0], coords[:, 1], c=colors_hypo,
                        s=3, alpha=0.5, rasterized=True)
        axes[2].set_title("Hypotension (red=pos)")

        # Panel 4: Recording file
        sc4 = axes[3].scatter(coords[:, 0], coords[:, 1],
                              c=file_idx[umap_idx], cmap="tab20",
                              s=3, alpha=0.5, rasterized=True)
        axes[3].set_title(f"Recording ({len(unique_files)} files)")
        plt.colorbar(sc4, ax=axes[3])

        # Panel 5: Hemodynamic cluster (where available)
        valid_sub = valid_hemo[umap_idx]
        if valid_sub.sum() > 50:
            axes[4].scatter(coords[~valid_sub, 0], coords[~valid_sub, 1],
                            c="lightgray", s=2, alpha=0.2, rasterized=True)
            sc5 = axes[4].scatter(coords[valid_sub, 0], coords[valid_sub, 1],
                                  c=hemo_labels[umap_idx][valid_sub], cmap="Set1",
                                  s=5, alpha=0.7, rasterized=True)
            axes[4].set_title(f"Hemo clusters ({valid_sub.sum()} mapped)")
            plt.colorbar(sc5, ax=axes[4])
        else:
            axes[4].set_title("Hemo clusters (N/A)")

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"{args.model.upper()} — Pooled {n_patients} patients, "
                     f"{n_total} windows", fontsize=12)
        fig.tight_layout()
        out_path = OUTPUT_DIR / f"pooled_{args.model}_{n_patients}patients.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — Pooled Clustering ({args.model.upper()}, "
          f"{n_patients} patients, {n_total} windows, k={k})")
    print(f"{'='*70}")
    print(f"  ARI vs patient identity:    {ari_patient:.4f}")
    print(f"  ARI vs recording segment:   {ari_file:.4f}")
    print(f"  ARI vs hypotension label:   {ari_hypo:.4f}")
    print(f"  ARI vs hemodynamic cluster: {ari_hemo:.4f}" if not np.isnan(ari_hemo)
          else f"  ARI vs hemodynamic cluster: N/A")
    print(f"  NMI vs patient identity:    {nmi_patient:.4f}")
    print(f"  NMI vs recording segment:   {nmi_file:.4f}")
    print(f"  NMI vs hypotension label:   {nmi_hypo:.4f}")
    print(f"  NMI vs hemodynamic cluster: {nmi_hemo:.4f}" if not np.isnan(nmi_hemo)
          else f"  NMI vs hemodynamic cluster: N/A")

    # Rank the drivers
    aris = [
        ("Patient identity", ari_patient),
        ("Recording segment", ari_file),
        ("Hypotension label", ari_hypo),
    ]
    if not np.isnan(ari_hemo):
        aris.append(("Hemodynamic cluster", ari_hemo))
    aris.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Cluster drivers (ranked by ARI):")
    for name, val in aris:
        bar = "█" * int(val * 50)
        print(f"    {name:<22}: {val:.4f} {bar}")

    # Save
    results = {
        "model": args.model,
        "n_patients": n_patients,
        "n_windows": n_total,
        "k": k,
        "ari_patient": ari_patient,
        "nmi_patient": nmi_patient,
        "ari_file": ari_file,
        "nmi_file": nmi_file,
        "ari_hypo": ari_hypo,
        "nmi_hypo": nmi_hypo,
        "ari_hemo": ari_hemo,
        "nmi_hemo": nmi_hemo,
        "patients": ",".join(target_pids),
    }
    results_df = pd.DataFrame([results])
    out_csv = OUTPUT_DIR / f"pooled_clustering_{args.model}_summary.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    # Save cluster details
    cluster_detail = pd.DataFrame({
        "patient_id": patient_ids,
        "cluster": cluster_labels,
        "hypo_label": labels,
        "hemo_label": hemo_labels,
        "file_idx": file_idx,
    })
    detail_path = OUTPUT_DIR / f"pooled_cluster_assignments_{args.model}.csv.gz"
    cluster_detail.to_csv(detail_path, index=False)
    print(f"  Saved: {detail_path}")


if __name__ == "__main__":
    main()
