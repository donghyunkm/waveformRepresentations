"""
Probe whether JEPA/PatchTST mean-pooled embeddings encode hemodynamic cluster identity.

For a random subset of PhysioJEPA windows with valid hemo cluster labels:
1. Read raw waveforms (ABP, II, PLETH from Zarr ZipStores)
2. Extract frozen encoder embeddings (mean-pooled to d_model=512)
3. Compute raw signal statistics (30 features) as a baseline
4. Train Logistic Regression probes: representation → 7-class cluster
5. Report macro AUROC, balanced accuracy, per-class accuracy (patient-level split)

Label alignment: The per-split .npy files
  {split}_hemo_clusters.npy
are positionally aligned with the sample CSVs:
  {split}_samples.csv.gz
Row i in the CSV corresponds to labels[i] in the .npy file.
Labels are -1 for unmatched windows, 0–6 for valid cluster labels.

Usage:
    python probe_hemo_clusters.py [--n-samples 10000] [--batch-size 32] [--seed 42]
    python probe_hemo_clusters.py --skip-extraction  # reuse cached embeddings
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

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

# ── PhysioJEPA ────────────────────────────────────────────────────────────────
sys.path.insert(0, "/gpfs/home/dk5565/PhysioJEPA")

# ── Constants ─────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
SAMPLE_CACHE_DIR = DERIVED_ROOT / "models/fcn_hypotension_paper"
DATASET_NAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"

JEPA_CKPT = DERIVED_ROOT / "models/jepa_native_paper/2026-08-04-native-jepa-paper-1gpu-debug-v1/best-val-epoch=13-loss=0.21508.ckpt"
PTST_CKPT = DERIVED_ROOT / "models/patchtst_self_supervised_paper/2026-08-05-patchtst-paper-1gpu-v1/best-val-epoch=03-loss=0.00329.ckpt"

OUTPUT_DIR = DERIVED_ROOT / "probing/hemo_clusters"

WINDOW_SAMPLES = 225000  # 1800s at 125 Hz
NUM_CLASSES = 7
CHANNEL_NAMES = ["ABP", "II", "PLETH"]

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
    parser = argparse.ArgumentParser(description="Hemo cluster probing with embeddings and raw stats")
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Max number of valid-label windows to use (sampled from train+test)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for encoder forward pass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip waveform/embedding extraction, load from cache")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        help="Which splits to draw samples from (default: train test)")
    return parser.parse_args()


# ── Data Loading & Label Alignment ────────────────────────────────────────────

def load_samples_with_labels(splits: list[str], n_samples: int, seed: int) -> pd.DataFrame:
    """
    Load sample CSVs and hemo cluster labels for the given splits.
    Filter to valid labels (>= 0), subsample to n_samples.

    Returns DataFrame with columns: file_path, start_idx, end_idx, subject_id,
                                     hemo_cluster, split
    """
    all_valid = []

    for split in splits:
        # Load sample CSV
        samples_path = SAMPLE_CACHE_DIR / f"{DATASET_NAME}-{split}_samples.csv.gz"
        if not samples_path.is_file():
            print(f"  WARNING: Sample cache not found for {split}: {samples_path}")
            continue
        samples = pd.read_csv(samples_path)

        # Load positionally-aligned hemo cluster labels
        labels_path = SAMPLE_CACHE_DIR / f"{DATASET_NAME}-{split}_hemo_clusters.npy"
        if not labels_path.is_file():
            print(f"  WARNING: Hemo cluster labels not found for {split}: {labels_path}")
            continue
        labels = np.load(labels_path)

        # Sanity check: sizes must match
        if len(samples) != len(labels):
            raise ValueError(
                f"Size mismatch in {split}: samples={len(samples)}, labels={len(labels)}. "
                f"Labels are not aligned!"
            )

        # Add labels and filter to valid (>= 0)
        samples["hemo_cluster"] = labels
        samples["split"] = split
        valid = samples[samples["hemo_cluster"] >= 0].copy()
        print(f"  {split}: {len(valid)}/{len(samples)} windows have valid hemo labels")
        all_valid.append(valid)

    if not all_valid:
        raise RuntimeError("No valid samples found in any split")

    combined = pd.concat(all_valid, ignore_index=True)
    print(f"  Total valid: {len(combined)} windows across {splits}")

    # Subsample if needed
    if len(combined) > n_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(combined), size=n_samples, replace=False)
        combined = combined.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled to {n_samples} windows")

    # Report class distribution
    print(f"\n  Class distribution ({len(combined)} windows):")
    for c in range(NUM_CLASSES):
        n = (combined["hemo_cluster"] == c).sum()
        print(f"    C{c} ({CLUSTER_NAMES[c][:25]:<25}): {n:>5} ({100*n/len(combined):.1f}%)")

    return combined


def read_window_signals(file_path: str, start_idx: int, end_idx: int) -> np.ndarray | None:
    """
    Read ABP, II, PLETH from Zarr ZipStore container.
    Returns (3, n_samples) float32 array or None on failure.
    Channel order: ABP, II, PLETH.
    """
    try:
        store = zarr.ZipStore(file_path, mode='r')
        root = zarr.open(store, mode='r')

        abp = np.array(root["ABP"][start_idx:end_idx], dtype=np.float32)
        ii = np.array(root["II"][start_idx:end_idx], dtype=np.float32)
        pleth = np.array(root["PLETH"][start_idx:end_idx], dtype=np.float32)
        store.close()

        return np.stack([abp, ii, pleth], axis=0)  # (3, n_samples)
    except Exception:
        return None


# ── Raw Signal Statistics ─────────────────────────────────────────────────────

def compute_raw_stats(waveforms_3ch: np.ndarray) -> np.ndarray:
    """
    Compute summary statistics from raw 3-channel waveforms.

    Input: (N, 3, 225000) float32
    Output: (N, 30) float32 — 10 stats per channel

    Stats per channel:
      0: mean, 1: std, 2: min, 3: max, 4: median,
      5: IQR, 6: skewness, 7: kurtosis,
      8: zero-crossing rate, 9: peak count (normalized)
    """
    n_samples, n_channels, n_timepoints = waveforms_3ch.shape
    n_stats = 10
    stats = np.zeros((n_samples, n_channels * n_stats), dtype=np.float32)

    for i in range(n_samples):
        if i % 2000 == 0 and i > 0:
            print(f"    Computing raw stats: {i}/{n_samples}")

        for ch in range(n_channels):
            sig = waveforms_3ch[i, ch]
            offset = ch * n_stats

            stats[i, offset + 0] = np.nanmean(sig)
            stats[i, offset + 1] = np.nanstd(sig)
            stats[i, offset + 2] = np.nanmin(sig)
            stats[i, offset + 3] = np.nanmax(sig)
            stats[i, offset + 4] = np.nanmedian(sig)

            q75, q25 = np.nanpercentile(sig, [75, 25])
            stats[i, offset + 5] = q75 - q25

            stats[i, offset + 6] = skew(sig, nan_policy='omit')
            stats[i, offset + 7] = kurtosis(sig, nan_policy='omit')

            sig_centered = sig - np.nanmean(sig)
            zero_crossings = np.sum(np.diff(np.sign(sig_centered)) != 0)
            stats[i, offset + 8] = zero_crossings / n_timepoints

            try:
                peaks, _ = find_peaks(sig, distance=50)
                stats[i, offset + 9] = len(peaks) / n_timepoints
            except Exception:
                stats[i, offset + 9] = 0.0

    return stats


# ── Encoder Loading ───────────────────────────────────────────────────────────

def load_jepa_encoder(device: torch.device):
    """Load frozen JEPA encoder."""
    from physiojepa.jepa import JEPASimpleLightning
    model = JEPASimpleLightning.load_from_checkpoint(str(JEPA_CKPT), map_location="cpu")
    model.eval()
    model.freeze()
    if hasattr(model, "pretrain"):
        model.pretrain = False
        if hasattr(model, "model"):
            model.model.pretrain = False
    return model.to(device)


def load_patchtst_encoder(device: torch.device):
    """Load frozen PatchTST encoder."""
    from physiojepa.patchtst import PatchTFTSimpleLightning
    model = PatchTFTSimpleLightning.load_from_checkpoint(str(PTST_CKPT), map_location="cpu")
    model.eval()
    model.freeze()
    if hasattr(model, "pretrain"):
        model.pretrain = False
        if hasattr(model, "model"):
            model.model.pretrain = False
    return model.to(device)


@torch.no_grad()
def extract_embeddings(encoder, waveforms: np.ndarray, batch_size: int,
                       device: torch.device) -> np.ndarray:
    """
    Run encoder on waveforms and return mean-pooled embeddings.

    waveforms: (N, 3, 225000) float32 - channels are ABP, II, PLETH
    Returns: (N, d_model) float32
    """
    n = len(waveforms)
    embeddings = []

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(waveforms[i:i + batch_size]).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            emb = encoder(batch)

        # Handle tuple output (PatchTST returns tuple)
        if isinstance(emb, tuple):
            emb = emb[0]

        # Mean-pool to (bs, d_model)
        if emb.dim() == 4:
            # (bs, n_channels, d_model, n_patches) → mean over channels and patches
            emb = emb.mean(dim=(1, 3))
        elif emb.dim() == 3:
            # (bs, d_model, n_patches) → mean over patches
            emb = emb.mean(dim=-1)

        embeddings.append(emb.float().cpu().numpy())

        if (i // batch_size) % 20 == 0:
            print(f"    Batch {i // batch_size}/{(n + batch_size - 1) // batch_size}")

    return np.concatenate(embeddings, axis=0)


# ── Classification Probing ────────────────────────────────────────────────────

def run_classification_probe(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    model_name: str,
) -> dict:
    """
    Train a regularized Logistic Regression (multiclass, L2 penalty) to predict
    7-class hemo clusters from the given representation.

    Uses patient-level 80/20 split for train/test (GroupShuffleSplit).
    Reports macro AUROC (OVR), balanced accuracy, per-class metrics.
    """
    # Filter rows with NaN/inf in features
    valid = np.isfinite(X).all(axis=1)
    if not valid.all():
        n_bad = (~valid).sum()
        print(f"  [{model_name}] Filtering {n_bad}/{len(X)} rows with NaN/inf")
        X = X[valid]
        y = y[valid]
        patient_ids = patient_ids[valid]

    # Patient-level split (same seed as medical feature probe for consistency)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_train_patients = len(np.unique(patient_ids[train_idx]))
    n_test_patients = len(np.unique(patient_ids[test_idx]))

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Logistic Regression with L2 penalty (Ridge-style classification)
    clf = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        multi_class="multinomial",
        class_weight="balanced",  # handle class imbalance
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # Metrics
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    # Macro AUROC (one-vs-rest)
    try:
        # Ensure all classes are present in test set for AUROC computation
        present_classes = np.unique(y_test)
        if len(present_classes) == NUM_CLASSES:
            auroc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        else:
            # Compute only for classes present in test set
            auroc = roc_auc_score(
                y_test, y_prob, multi_class="ovr", average="macro",
                labels=present_classes,
            )
    except ValueError:
        auroc = float("nan")

    # Per-class report
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
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cache_path = OUTPUT_DIR / f"cache_n{args.n_samples}_seed{args.seed}.npz"

    if args.skip_extraction and cache_path.is_file():
        print(f"\nLoading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        waveforms_3ch = data["waveforms_3ch"]
        labels = data["labels"]
        patient_ids = data["patient_ids"]
        print(f"  Loaded: {len(labels)} windows, {len(np.unique(patient_ids))} patients")
    else:
        # ── Step 1: Load samples with aligned hemo labels ─────────────────────
        print("\n=== Loading samples with hemo cluster labels ===")
        samples_df = load_samples_with_labels(args.splits, args.n_samples, args.seed)

        # ── Step 2: Read waveforms from Zarr ──────────────────────────────────
        print(f"\n=== Reading waveforms ({len(samples_df)} windows) ===")
        waveforms_3ch = np.zeros((len(samples_df), 3, WINDOW_SAMPLES), dtype=np.float32)
        labels = samples_df["hemo_cluster"].values.astype(np.int64)
        patient_ids = samples_df["subject_id"].values.astype(str)
        valid_mask = np.zeros(len(samples_df), dtype=bool)

        t0 = time.time()
        for i, row in samples_df.iterrows():
            if i % 500 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(samples_df) - i) / rate
                print(f"  [{i}/{len(samples_df)}] {rate:.1f} windows/s, ETA {eta / 60:.1f} min")

            sig = read_window_signals(
                row["file_path"], int(row["start_idx"]), int(row["end_idx"])
            )
            if sig is None:
                continue

            # Verify expected length
            expected_len = int(row["end_idx"]) - int(row["start_idx"])
            if sig.shape[1] != expected_len:
                continue

            # Pad or truncate to exactly WINDOW_SAMPLES if needed
            actual_len = sig.shape[1]
            if actual_len >= WINDOW_SAMPLES:
                waveforms_3ch[i] = sig[:, :WINDOW_SAMPLES]
            else:
                waveforms_3ch[i, :, :actual_len] = sig
            valid_mask[i] = True

        # Keep only valid windows
        valid_idx = np.where(valid_mask)[0]
        waveforms_3ch = waveforms_3ch[valid_idx]
        labels = labels[valid_idx]
        patient_ids = patient_ids[valid_idx]

        elapsed = time.time() - t0
        print(f"  Done: {len(valid_idx)}/{len(samples_df)} valid windows in {elapsed:.0f}s")

        # Save cache
        np.savez_compressed(cache_path,
                            waveforms_3ch=waveforms_3ch,
                            labels=labels,
                            patient_ids=patient_ids)
        print(f"  Cached to {cache_path}")

    print(f"\nDataset: {len(labels)} windows, {len(np.unique(patient_ids))} patients, "
          f"{NUM_CLASSES} classes")
    print(f"Class distribution:")
    for c in range(NUM_CLASSES):
        n = (labels == c).sum()
        print(f"  C{c} ({CLUSTER_NAMES[c][:25]:<25}): {n:>5} ({100 * n / len(labels):.1f}%)")

    # ── Step 3: Compute raw signal statistics ─────────────────────────────────
    print("\n=== Computing raw signal statistics (30 features) ===")
    raw_stats_path = OUTPUT_DIR / f"raw_stats_n{len(labels)}_seed{args.seed}.npy"
    if args.skip_extraction and raw_stats_path.is_file():
        raw_stats = np.load(raw_stats_path)
        print(f"  Loaded cached raw stats: {raw_stats.shape}")
    else:
        raw_stats = compute_raw_stats(waveforms_3ch)
        np.save(raw_stats_path, raw_stats)
        print(f"  Computed and saved: {raw_stats.shape}")

    # ── Step 4: Extract encoder embeddings ────────────────────────────────────
    print("\n=== JEPA Encoder ===")
    jepa_emb_path = OUTPUT_DIR / f"jepa_embeddings_n{len(labels)}_seed{args.seed}.npy"
    if args.skip_extraction and jepa_emb_path.is_file():
        jepa_emb = np.load(jepa_emb_path)
        print(f"  Loaded cached: {jepa_emb.shape}")
    else:
        encoder = load_jepa_encoder(device)
        jepa_emb = extract_embeddings(encoder, waveforms_3ch, args.batch_size, device)
        np.save(jepa_emb_path, jepa_emb)
        del encoder
        torch.cuda.empty_cache()
        print(f"  Extracted: {jepa_emb.shape}")

    print("\n=== PatchTST Encoder ===")
    ptst_emb_path = OUTPUT_DIR / f"ptst_embeddings_n{len(labels)}_seed{args.seed}.npy"
    if args.skip_extraction and ptst_emb_path.is_file():
        ptst_emb = np.load(ptst_emb_path)
        print(f"  Loaded cached: {ptst_emb.shape}")
    else:
        encoder = load_patchtst_encoder(device)
        ptst_emb = extract_embeddings(encoder, waveforms_3ch, args.batch_size, device)
        np.save(ptst_emb_path, ptst_emb)
        del encoder
        torch.cuda.empty_cache()
        print(f"  Extracted: {ptst_emb.shape}")

    # ── Step 5: Run classification probes ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== Logistic Regression Probes: Representation → 7-class Hemo Cluster ===")
    print("=" * 70)

    results = []

    print("\n--- Raw Signal Statistics (30 features) ---")
    res_raw = run_classification_probe(raw_stats, labels, patient_ids, "Raw Stats")
    results.append(res_raw)
    print(f"  Macro AUROC: {res_raw['auroc_macro']:.4f}")
    print(f"  Balanced Acc: {res_raw['balanced_accuracy']:.4f}")
    print(f"  (Train: {res_raw['n_train']} samples / {res_raw['n_train_patients']} patients, "
          f"Test: {res_raw['n_test']} samples / {res_raw['n_test_patients']} patients)")

    print("\n--- JEPA Mean-Pooled Embeddings (512-d) ---")
    res_jepa = run_classification_probe(jepa_emb, labels, patient_ids, "JEPA")
    results.append(res_jepa)
    print(f"  Macro AUROC: {res_jepa['auroc_macro']:.4f}")
    print(f"  Balanced Acc: {res_jepa['balanced_accuracy']:.4f}")
    print(f"  (Train: {res_jepa['n_train']} samples / {res_jepa['n_train_patients']} patients, "
          f"Test: {res_jepa['n_test']} samples / {res_jepa['n_test_patients']} patients)")

    print("\n--- PatchTST Mean-Pooled Embeddings (512-d) ---")
    res_ptst = run_classification_probe(ptst_emb, labels, patient_ids, "PatchTST")
    results.append(res_ptst)
    print(f"  Macro AUROC: {res_ptst['auroc_macro']:.4f}")
    print(f"  Balanced Acc: {res_ptst['balanced_accuracy']:.4f}")
    print(f"  (Train: {res_ptst['n_train']} samples / {res_ptst['n_train_patients']} patients, "
          f"Test: {res_ptst['n_test']} samples / {res_ptst['n_test_patients']} patients)")

    # ── Step 6: Save results ──────────────────────────────────────────────────
    # Summary CSV
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
        # Add per-class F1
        for c in range(NUM_CLASSES):
            key = f"C{c}"
            if key in res["per_class"]:
                row[f"f1_C{c}"] = res["per_class"][key]["f1-score"]
                row[f"precision_C{c}"] = res["per_class"][key]["precision"]
                row[f"recall_C{c}"] = res["per_class"][key]["recall"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    results_path = OUTPUT_DIR / "probe_results.csv"
    summary_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # ── Print final comparison table ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: Hemodynamic Cluster Prediction (7-class)")
    print("  Random baseline: AUROC=0.500, Balanced Acc=0.143")
    print("=" * 70)
    print(f"{'Model':<20} {'AUROC':>8} {'Bal Acc':>8} {'Δ AUROC':>9} {'Δ Acc':>7}")
    print("-" * 70)
    raw_auroc = res_raw["auroc_macro"]
    raw_acc = res_raw["balanced_accuracy"]
    for res in results:
        auroc = res["auroc_macro"]
        acc = res["balanced_accuracy"]
        d_auroc = auroc - raw_auroc if res["model"] != "Raw Stats" else 0.0
        d_acc = acc - raw_acc if res["model"] != "Raw Stats" else 0.0
        d_auroc_str = f"+{d_auroc:.4f}" if d_auroc >= 0 else f"{d_auroc:.4f}"
        d_acc_str = f"+{d_acc:.4f}" if d_acc >= 0 else f"{d_acc:.4f}"
        if res["model"] == "Raw Stats":
            d_auroc_str = "baseline"
            d_acc_str = "baseline"
        print(f"{res['model']:<20} {auroc:>8.4f} {acc:>8.4f} {d_auroc_str:>9} {d_acc_str:>7}")
    print("=" * 70)

    print("\nPer-class F1 scores:")
    print(f"{'Cluster':<30} {'Raw Stats':>10} {'JEPA':>8} {'PatchTST':>10}")
    print("-" * 70)
    for c in range(NUM_CLASSES):
        key = f"C{c}"
        raw_f1 = res_raw["per_class"].get(key, {}).get("f1-score", float("nan"))
        jepa_f1 = res_jepa["per_class"].get(key, {}).get("f1-score", float("nan"))
        ptst_f1 = res_ptst["per_class"].get(key, {}).get("f1-score", float("nan"))
        name = f"C{c} ({CLUSTER_NAMES[c][:22]})"
        print(f"{name:<30} {raw_f1:>10.3f} {jepa_f1:>8.3f} {ptst_f1:>10.3f}")
    print("=" * 70)

    print("\nInterpretation:")
    print("  - If AUROC ≈ 0.50 for all models: cluster identity is not linearly")
    print("    decodable from any representation (confirms attentive probe failure)")
    print("  - If Raw Stats >> encoders: clusters are trivially determined by")
    print("    signal statistics, but encoders discard this information")
    print("  - If encoders >> Raw Stats: encoders capture temporal dynamics")
    print("    beyond what raw statistics provide")
    print("  - If encoders ≈ Raw Stats > 0.50: all representations partially")
    print("    encode cluster identity but likely dominated by patient identity")


if __name__ == "__main__":
    main()
