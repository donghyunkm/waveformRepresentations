"""
Probe hemodynamic cluster prediction using PRE-EXTRACTED embeddings.

Uses the full test-split embeddings (127,811 windows) already saved at:
  probing/clustering/embeddings_nfull_seed42.npz

These are positionally aligned with the test-split sample CSV. The hemo
cluster labels come from the positionally-aligned .npy file:
  models/fcn_hypotension_paper/{dataset}-test_hemo_clusters.npy

For raw signal statistics baseline, we compute stats directly from Zarr
for the subset of windows with valid hemo labels (22,323 windows).

Usage:
    # Full run (computes raw stats from Zarr — needs ~2h on CPU):
    python probe_hemo_clusters_precomputed.py

    # Skip raw stats (embeddings-only, fast):
    python probe_hemo_clusters_precomputed.py --skip-raw-stats

    # Use cached raw stats from prior run:
    python probe_hemo_clusters_precomputed.py --load-raw-stats
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ── Constants ─────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
SAMPLE_CACHE_DIR = DERIVED_ROOT / "models/fcn_hypotension_paper"
DATASET_NAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"

# Pre-extracted embeddings (test split, 127811 windows)
EMBEDDINGS_PATH = DERIVED_ROOT / "probing/clustering/embeddings_nfull_seed42.npz"

# Hemo cluster labels aligned to test split
HEMO_LABELS_PATH = SAMPLE_CACHE_DIR / f"{DATASET_NAME}-test_hemo_clusters.npy"

# Test split sample CSV
SAMPLES_PATH = SAMPLE_CACHE_DIR / f"{DATASET_NAME}-test_samples.csv.gz"

OUTPUT_DIR = DERIVED_ROOT / "probing/hemo_clusters"

WINDOW_SAMPLES = 225000  # 1800s at 125 Hz
NUM_CLASSES = 7

CLUSTER_NAMES = [
    "Failing Vasoconstriction",
    "Hemodynamic Quiescence",
    "High-Output Compensated",
    "Normal Baroreflex",
    "Tachycardia + Vasoconstriction",
    "Catecholamine-Driven",
    "PPG Dissociation",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-raw-stats", action="store_true",
                        help="Skip raw stats baseline (no Zarr reads needed)")
    parser.add_argument("--load-raw-stats", action="store_true",
                        help="Load raw stats from prior cached file")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of valid-label samples (for faster testing)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Raw Signal Statistics ─────────────────────────────────────────────────────

def read_window_from_zarr(file_path: str, start_idx: int, end_idx: int) -> np.ndarray | None:
    """Read (3, n) waveform from Zarr ZipStore. Returns None on failure."""
    import zarr
    try:
        store = zarr.ZipStore(file_path, mode='r')
        root = zarr.open(store, mode='r')
        abp = np.array(root["ABP"][start_idx:end_idx], dtype=np.float32)
        ii = np.array(root["II"][start_idx:end_idx], dtype=np.float32)
        pleth = np.array(root["PLETH"][start_idx:end_idx], dtype=np.float32)
        store.close()
        return np.stack([abp, ii, pleth], axis=0)
    except Exception:
        return None


def compute_raw_stats_single(sig_3ch: np.ndarray) -> np.ndarray:
    """Compute 30 raw stats for a single (3, T) waveform."""
    n_stats = 10
    stats = np.zeros(3 * n_stats, dtype=np.float32)
    n_timepoints = sig_3ch.shape[1]

    for ch in range(3):
        sig = sig_3ch[ch]
        offset = ch * n_stats

        stats[offset + 0] = np.nanmean(sig)
        stats[offset + 1] = np.nanstd(sig)
        stats[offset + 2] = np.nanmin(sig)
        stats[offset + 3] = np.nanmax(sig)
        stats[offset + 4] = np.nanmedian(sig)

        q75, q25 = np.nanpercentile(sig, [75, 25])
        stats[offset + 5] = q75 - q25
        stats[offset + 6] = skew(sig, nan_policy='omit')
        stats[offset + 7] = kurtosis(sig, nan_policy='omit')

        sig_centered = sig - np.nanmean(sig)
        zero_crossings = np.sum(np.diff(np.sign(sig_centered)) != 0)
        stats[offset + 8] = zero_crossings / n_timepoints

        try:
            peaks, _ = find_peaks(sig, distance=50)
            stats[offset + 9] = len(peaks) / n_timepoints
        except Exception:
            stats[offset + 9] = 0.0

    return stats


def extract_raw_stats_from_zarr(
    samples: pd.DataFrame, valid_indices: np.ndarray
) -> np.ndarray:
    """Read waveforms from Zarr for valid-label windows and compute raw stats."""
    n = len(valid_indices)
    raw_stats = np.zeros((n, 30), dtype=np.float32)

    t0 = time.time()
    n_failed = 0

    for i, idx in enumerate(valid_indices):
        if i % 500 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (n - i) / rate
            print(f"    [{i}/{n}] {rate:.1f} windows/s, ETA {eta / 60:.1f} min")

        row = samples.iloc[idx]
        sig = read_window_from_zarr(row["file_path"], int(row["start_idx"]), int(row["end_idx"]))
        if sig is None:
            n_failed += 1
            raw_stats[i] = np.nan
            continue

        # Pad/truncate to WINDOW_SAMPLES
        actual_len = sig.shape[1]
        if actual_len < WINDOW_SAMPLES:
            padded = np.zeros((3, WINDOW_SAMPLES), dtype=np.float32)
            padded[:, :actual_len] = sig
            sig = padded
        else:
            sig = sig[:, :WINDOW_SAMPLES]

        raw_stats[i] = compute_raw_stats_single(sig)

    elapsed = time.time() - t0
    print(f"    Done: {n - n_failed}/{n} valid in {elapsed:.0f}s ({n_failed} failed reads)")
    return raw_stats


# ── Classification Probing ────────────────────────────────────────────────────

def run_classification_probe(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    model_name: str,
) -> dict:
    """
    Logistic Regression (L2, multiclass) with patient-level 80/20 split.
    Reports macro AUROC (OVR), balanced accuracy, per-class metrics.
    """
    # Filter rows with NaN/inf
    valid = np.isfinite(X).all(axis=1)
    if not valid.all():
        n_bad = (~valid).sum()
        print(f"  [{model_name}] Filtering {n_bad}/{len(X)} rows with NaN/inf")
        X = X[valid]
        y = y[valid]
        patient_ids = patient_ids[valid]

    # Patient-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_train_patients = len(np.unique(patient_ids[train_idx]))
    n_test_patients = len(np.unique(patient_ids[test_idx]))

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression
    clf = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    bal_acc = balanced_accuracy_score(y_test, y_pred)

    try:
        present_classes = np.unique(y_test)
        if len(present_classes) == NUM_CLASSES:
            auroc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        else:
            auroc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro",
                                  labels=present_classes)
    except ValueError:
        auroc = float("nan")

    report = classification_report(
        y_test, y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=[f"C{i}" for i in range(NUM_CLASSES)],
        output_dict=True,
        zero_division=0,
    )

    return {
        "model": model_name,
        "auroc_macro": auroc,
        "balanced_accuracy": bal_acc,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_train_patients": n_train_patients,
        "n_test_patients": n_test_patients,
        "per_class": report,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load pre-extracted embeddings ─────────────────────────────────────────
    print("=== Loading pre-extracted embeddings ===")
    print(f"  Source: {EMBEDDINGS_PATH}")
    emb_data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    jepa_emb = emb_data["jepa_embeddings"]   # (127811, 512)
    ptst_emb = emb_data["ptst_embeddings"]   # (127811, 512)
    emb_patient_ids = emb_data["patient_ids"]  # (127811,) — from embeddings file
    print(f"  JEPA: {jepa_emb.shape}, PatchTST: {ptst_emb.shape}")
    print(f"  Patients in embeddings: {len(np.unique(emb_patient_ids))}")

    # ── Load hemo cluster labels (positionally aligned to test split) ─────────
    print(f"\n=== Loading hemo cluster labels ===")
    print(f"  Source: {HEMO_LABELS_PATH}")
    hemo_labels = np.load(HEMO_LABELS_PATH)
    print(f"  Shape: {hemo_labels.shape}")

    # Verify alignment
    assert len(hemo_labels) == len(jepa_emb), (
        f"Size mismatch: labels={len(hemo_labels)}, embeddings={len(jepa_emb)}"
    )

    # Filter to valid labels
    valid_mask = hemo_labels >= 0
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    print(f"  Valid labels: {n_valid}/{len(hemo_labels)} ({100 * n_valid / len(hemo_labels):.1f}%)")

    # Optionally subsample
    if args.max_samples and n_valid > args.max_samples:
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(n_valid, size=args.max_samples, replace=False)
        valid_indices = valid_indices[chosen]
        n_valid = len(valid_indices)
        print(f"  Subsampled to {n_valid} windows")

    # Extract aligned data
    labels = hemo_labels[valid_indices]
    jepa_valid = jepa_emb[valid_indices]
    ptst_valid = ptst_emb[valid_indices]
    patient_ids = emb_patient_ids[valid_indices].astype(str)

    print(f"\n  Dataset: {n_valid} windows, {len(np.unique(patient_ids))} patients")
    print(f"  Class distribution:")
    for c in range(NUM_CLASSES):
        n = (labels == c).sum()
        print(f"    C{c} ({CLUSTER_NAMES[c][:25]:<25}): {n:>5} ({100 * n / len(labels):.1f}%)")

    # ── Raw stats baseline ────────────────────────────────────────────────────
    raw_stats = None
    raw_stats_path = OUTPUT_DIR / f"raw_stats_test_n{n_valid}_seed{args.seed}.npy"

    if args.skip_raw_stats:
        print("\n=== Skipping raw stats baseline (--skip-raw-stats) ===")
    elif args.load_raw_stats and raw_stats_path.is_file():
        print(f"\n=== Loading cached raw stats from {raw_stats_path} ===")
        raw_stats = np.load(raw_stats_path)
        print(f"  Shape: {raw_stats.shape}")
    else:
        print(f"\n=== Computing raw stats from Zarr ({n_valid} windows) ===")
        print(f"  Loading sample CSV: {SAMPLES_PATH}")
        samples = pd.read_csv(SAMPLES_PATH)
        assert len(samples) == len(hemo_labels), "Sample CSV / label size mismatch"

        raw_stats = extract_raw_stats_from_zarr(samples, valid_indices)
        np.save(raw_stats_path, raw_stats)
        print(f"  Saved to {raw_stats_path}")

    # ── Run classification probes ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== Logistic Regression: Representation → 7-class Hemo Cluster ===")
    print("=" * 70)

    results = []

    if raw_stats is not None:
        print("\n--- Raw Signal Statistics (30 features) ---")
        res_raw = run_classification_probe(raw_stats, labels, patient_ids, "Raw Stats")
        results.append(res_raw)
        print(f"  Macro AUROC: {res_raw['auroc_macro']:.4f}")
        print(f"  Balanced Acc: {res_raw['balanced_accuracy']:.4f}")
        print(f"  (Train: {res_raw['n_train']} / {res_raw['n_train_patients']} patients, "
              f"Test: {res_raw['n_test']} / {res_raw['n_test_patients']} patients)")

    print("\n--- JEPA Mean-Pooled Embeddings (512-d) ---")
    res_jepa = run_classification_probe(jepa_valid, labels, patient_ids, "JEPA")
    results.append(res_jepa)
    print(f"  Macro AUROC: {res_jepa['auroc_macro']:.4f}")
    print(f"  Balanced Acc: {res_jepa['balanced_accuracy']:.4f}")
    print(f"  (Train: {res_jepa['n_train']} / {res_jepa['n_train_patients']} patients, "
          f"Test: {res_jepa['n_test']} / {res_jepa['n_test_patients']} patients)")

    print("\n--- PatchTST Mean-Pooled Embeddings (512-d) ---")
    res_ptst = run_classification_probe(ptst_valid, labels, patient_ids, "PatchTST")
    results.append(res_ptst)
    print(f"  Macro AUROC: {res_ptst['auroc_macro']:.4f}")
    print(f"  Balanced Acc: {res_ptst['balanced_accuracy']:.4f}")
    print(f"  (Train: {res_ptst['n_train']} / {res_ptst['n_train_patients']} patients, "
          f"Test: {res_ptst['n_test']} / {res_ptst['n_test_patients']} patients)")

    # ── Save results ──────────────────────────────────────────────────────────
    summary_rows = []
    for res in results:
        row = {
            "model": res["model"],
            "auroc_macro": res["auroc_macro"],
            "balanced_accuracy": res["balanced_accuracy"],
            "n_train": res["n_train"],
            "n_test": res["n_test"],
            "n_train_patients": res["n_train_patients"],
            "n_test_patients": res["n_test_patients"],
        }
        for c in range(NUM_CLASSES):
            key = f"C{c}"
            if key in res["per_class"]:
                row[f"f1_C{c}"] = res["per_class"][key]["f1-score"]
                row[f"precision_C{c}"] = res["per_class"][key]["precision"]
                row[f"recall_C{c}"] = res["per_class"][key]["recall"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    results_path = OUTPUT_DIR / "probe_results_precomputed.csv"
    summary_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # ── Print final table ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Hemodynamic Cluster Prediction (7-class, test split)")
    print(f"  N={n_valid} windows, {len(np.unique(patient_ids))} patients")
    print("  Random baseline: AUROC=0.500, Balanced Acc=0.143")
    print("=" * 70)
    print(f"{'Model':<20} {'AUROC':>8} {'Bal Acc':>8}")
    print("-" * 40)
    for res in results:
        print(f"{res['model']:<20} {res['auroc_macro']:>8.4f} {res['balanced_accuracy']:>8.4f}")
    print("=" * 70)

    print("\nPer-class F1 scores:")
    header = f"{'Cluster':<30}"
    for res in results:
        header += f" {res['model']:>10}"
    print(header)
    print("-" * 70)
    for c in range(NUM_CLASSES):
        key = f"C{c}"
        name = f"C{c} ({CLUSTER_NAMES[c][:22]})"
        line = f"{name:<30}"
        for res in results:
            f1 = res["per_class"].get(key, {}).get("f1-score", float("nan"))
            line += f" {f1:>10.3f}"
        print(line)
    print("=" * 70)

    print("\nInterpretation:")
    print("  - AUROC ≈ 0.50: cluster identity not linearly decodable")
    print("  - Raw Stats >> encoders: clusters trivially determined by signal stats")
    print("  - Encoders >> Raw Stats: encoders capture temporal dynamics")
    print("  - All > 0.50 but modest: partial encoding dominated by patient identity")


if __name__ == "__main__":
    main()
