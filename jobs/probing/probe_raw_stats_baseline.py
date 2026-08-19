"""
Baseline for medical feature probing: predict physiological features from
raw signal summary statistics (no learned encoder).

Uses the same cached data as probe_medical_features.py (waveforms + features +
patient_ids) and the same Ridge regression + patient-level split methodology.

Raw statistics computed per channel (ABP, II, PLETH):
  mean, std, min, max, median, IQR, skewness, kurtosis,
  zero-crossing rate, peak count

Total features: 10 stats × 3 channels = 30 raw features.

This provides a floor: if the encoder probe R² ≈ raw stats R², the encoder
adds nothing beyond what trivial signal statistics already capture.

Usage:
    python -u probe_raw_stats_baseline.py [--n-samples 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# ── icuDataExtraction config (for FEATURE_NAMES) ─────────────────────────────
sys.path.insert(0, "/gpfs/home/dk5565/icuDataExtraction")
from config import N_FEATURES, FEATURE_NAMES

# ── Constants ─────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
OUTPUT_DIR = DERIVED_ROOT / "probing/medical_features"

RESP_DEPENDENT_FEATURES = {"RR", "PPV", "PVI", "RESP_amp"}

CHANNEL_NAMES = ["ABP", "II", "PLETH"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="n-samples used in the cache filename")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed used in the cache filename")
    return parser.parse_args()


# ── Raw Signal Statistics ─────────────────────────────────────────────────────

def compute_raw_stats(waveforms_3ch: np.ndarray) -> np.ndarray:
    """
    Compute summary statistics from raw 3-channel waveforms.

    Input: (N, 3, 225000) float32
    Output: (N, 30) float32 — 10 stats per channel

    Stats per channel:
      0: mean
      1: std
      2: min
      3: max
      4: median
      5: IQR (75th - 25th percentile)
      6: skewness
      7: kurtosis
      8: zero-crossing rate (fraction of samples where signal crosses its mean)
      9: peak count (normalized by window length)
    """
    n_samples, n_channels, n_timepoints = waveforms_3ch.shape
    n_stats = 10
    stats = np.zeros((n_samples, n_channels * n_stats), dtype=np.float32)

    for i in range(n_samples):
        if i % 1000 == 0 and i > 0:
            print(f"  Computing raw stats: {i}/{n_samples}")

        for ch in range(n_channels):
            sig = waveforms_3ch[i, ch]
            offset = ch * n_stats

            # Basic moments
            stats[i, offset + 0] = np.nanmean(sig)
            stats[i, offset + 1] = np.nanstd(sig)
            stats[i, offset + 2] = np.nanmin(sig)
            stats[i, offset + 3] = np.nanmax(sig)
            stats[i, offset + 4] = np.nanmedian(sig)

            # IQR
            q75, q25 = np.nanpercentile(sig, [75, 25])
            stats[i, offset + 5] = q75 - q25

            # Skewness and kurtosis
            stats[i, offset + 6] = skew(sig, nan_policy='omit')
            stats[i, offset + 7] = kurtosis(sig, nan_policy='omit')

            # Zero-crossing rate (crossings of the mean)
            sig_centered = sig - np.nanmean(sig)
            zero_crossings = np.sum(np.diff(np.sign(sig_centered)) != 0)
            stats[i, offset + 8] = zero_crossings / n_timepoints

            # Peak count (normalized)
            try:
                peaks, _ = find_peaks(sig, distance=50)  # ~0.4s min distance at 125Hz
                stats[i, offset + 9] = len(peaks) / n_timepoints
            except Exception:
                stats[i, offset + 9] = 0.0

    return stats


# ── Probing ───────────────────────────────────────────────────────────────────

def run_probes(raw_stats: np.ndarray, features: np.ndarray,
               patient_ids: np.ndarray) -> pd.DataFrame:
    """
    Train Ridge regression probes: raw_stats → each physiological feature.
    Same methodology as probe_medical_features.py (patient-level 80/20 split).
    """
    results = []

    # Filter out rows where raw stats contain NaN or inf
    valid = np.isfinite(raw_stats).all(axis=1)
    if not valid.all():
        n_bad = (~valid).sum()
        print(f"  Filtering {n_bad}/{len(raw_stats)} rows with NaN/inf in raw stats")
        raw_stats = raw_stats[valid]
        features = features[valid]
        patient_ids = patient_ids[valid]

    # Patient-level split (same seed as encoder probe for comparability)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(raw_stats, groups=patient_ids))

    X_train, X_test = raw_stats[train_idx], raw_stats[test_idx]

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        if feat_name in RESP_DEPENDENT_FEATURES:
            continue

        y_all = features[:, feat_idx]

        # Filter NaN targets
        train_valid = ~np.isnan(y_all[train_idx])
        test_valid = ~np.isnan(y_all[test_idx])

        n_train = train_valid.sum()
        n_test = test_valid.sum()

        if n_train < 50 or n_test < 20:
            results.append({
                "feature": feat_name,
                "r2": np.nan,
                "pearson_r": np.nan,
                "n_train": int(n_train),
                "n_test": int(n_test),
                "note": "insufficient_data",
            })
            continue

        Xtr = X_train[train_valid]
        ytr = y_all[train_idx][train_valid]
        Xte = X_test[test_valid]
        yte = y_all[test_idx][test_valid]

        # Standardize targets
        y_mean, y_std = ytr.mean(), ytr.std()
        if y_std < 1e-8:
            results.append({
                "feature": feat_name,
                "r2": np.nan,
                "pearson_r": np.nan,
                "n_train": int(n_train),
                "n_test": int(n_test),
                "note": "zero_variance",
            })
            continue

        ytr_z = (ytr - y_mean) / y_std
        yte_z = (yte - y_mean) / y_std

        # Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xtr, ytr_z)
        pred_z = ridge.predict(Xte)

        r2 = r2_score(yte_z, pred_z)
        r, _ = pearsonr(yte_z, pred_z)

        results.append({
            "feature": feat_name,
            "r2": r2,
            "pearson_r": r,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "note": "",
        })

    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cached data from the medical probe run
    cache_path = OUTPUT_DIR / f"cache_n{args.n_samples}_seed{args.seed}.npz"
    if not cache_path.is_file():
        print(f"ERROR: Cache not found at {cache_path}")
        print("Run probe_medical_features.py first to generate the cache.")
        sys.exit(1)

    print(f"Loading cached data from {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    waveforms_3ch = data["waveforms_3ch"]
    features = data["features"]
    patient_ids = data["patient_ids"]

    print(f"Dataset: {len(features)} windows, {len(np.unique(patient_ids))} patients")
    print(f"Waveforms shape: {waveforms_3ch.shape}")
    print(f"Features shape: {features.shape}")

    # ── Compute raw signal statistics ─────────────────────────────────────────
    print("\nComputing raw signal statistics (10 stats × 3 channels = 30 features)...")
    raw_stats = compute_raw_stats(waveforms_3ch)
    print(f"  Raw stats shape: {raw_stats.shape}")

    # Save raw stats for reuse
    raw_stats_path = OUTPUT_DIR / f"raw_stats_n{len(features)}_seed{args.seed}.npy"
    np.save(raw_stats_path, raw_stats)
    print(f"  Saved to {raw_stats_path}")

    # ── Run probes ────────────────────────────────────────────────────────────
    print("\n=== Ridge Regression: Raw Stats → Medical Features ===")
    results = run_probes(raw_stats, features, patient_ids)
    results["model"] = "raw_stats"

    # Save results
    results_path = OUTPUT_DIR / "probe_results_raw_stats.csv"
    results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # ── Load encoder results for comparison ───────────────────────────────────
    encoder_results_path = OUTPUT_DIR / "probe_results.csv"
    if encoder_results_path.is_file():
        encoder_results = pd.read_csv(encoder_results_path)
        jepa_results = encoder_results[encoder_results["model"] == "JEPA"]
        ptst_results = encoder_results[encoder_results["model"] == "PatchTST"]
        has_encoder = True
    else:
        has_encoder = False

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("COMPARISON: Raw Stats vs Encoder Embeddings (R²)")
    print("=" * 72)
    if has_encoder:
        print(f"{'Feature':<12} {'Raw Stats':>10} {'JEPA':>8} {'PatchTST':>10} {'Best Δ':>8}")
        print("-" * 72)
    else:
        print(f"{'Feature':<12} {'Raw Stats R²':>13} {'Raw Stats r':>12}")
        print("-" * 72)

    for feat_name in FEATURE_NAMES:
        if feat_name in RESP_DEPENDENT_FEATURES:
            continue
        row = results[results["feature"] == feat_name]
        if row.empty:
            continue
        row = row.iloc[0]
        raw_r2 = row["r2"]
        raw_str = f"{raw_r2:.3f}" if not np.isnan(raw_r2) else "N/A"

        if has_encoder:
            j = jepa_results[jepa_results["feature"] == feat_name]
            p = ptst_results[ptst_results["feature"] == feat_name]
            jr2 = j.iloc[0]["r2"] if not j.empty else np.nan
            pr2 = p.iloc[0]["r2"] if not p.empty else np.nan
            jr2_str = f"{jr2:.3f}" if not np.isnan(jr2) else "N/A"
            pr2_str = f"{pr2:.3f}" if not np.isnan(pr2) else "N/A"

            # Delta: best encoder R² minus raw stats R²
            best_enc = max(jr2 if not np.isnan(jr2) else -999,
                           pr2 if not np.isnan(pr2) else -999)
            if not np.isnan(raw_r2) and best_enc > -999:
                delta = best_enc - raw_r2
                delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            else:
                delta_str = "N/A"

            print(f"{feat_name:<12} {raw_str:>10} {jr2_str:>8} {pr2_str:>10} {delta_str:>8}")
        else:
            r_str = f"{row['pearson_r']:.3f}" if not np.isnan(row['pearson_r']) else "N/A"
            print(f"{feat_name:<12} {raw_str:>13} {r_str:>12}")

    print("=" * 72)

    if has_encoder:
        print("\nΔ = (best encoder R²) - (raw stats R²)")
        print("  Positive Δ: encoder adds information beyond raw statistics")
        print("  Near-zero Δ: feature is trivially predictable from signal stats")
        print("  Negative Δ: encoder loses information present in raw signal")


if __name__ == "__main__":
    main()
