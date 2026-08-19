"""
Probe whether JEPA-d64 embeddings encode physiological (medical) information.

Same methodology as probe_medical_features.py but for the d64 encoder:
1. Read raw waveforms (ABP, II, PLETH from Zarr)
2. Compute all 15 physiological features (skip RESP-dependent)
3. Extract frozen encoder embeddings (mean-pooled, d_model=64)
4. Train Ridge regression probes: embedding → each feature scalar
5. Report R² per feature (train/test split by patient)

Usage:
    python probe_medical_features_d64.py [--n-samples 10000] [--batch-size 64]
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

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# ── icuDataExtraction feature code ────────────────────────────────────────────
sys.path.insert(0, "/gpfs/home/dk5565/icuDataExtraction")
from config import (
    CTX_SAMPLES, HALF_CTX, N_FEATURES, N_SUBWIN, FEATURE_NAMES, FS,
    WIN_SEC, SUB_WIN_SAMP, SUB_WIN_STR, NAN_FRAC,
    HR_MIN_DIST, RR_MIN_DIST, ABP_MIN_DIST, FEATURE_BOUNDS,
)

# ── PhysioJEPA ────────────────────────────────────────────────────────────────
sys.path.insert(0, "/gpfs/home/dk5565/PhysioJEPA")

# ── Constants ─────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
SAMPLE_CACHE_DIR = DERIVED_ROOT / "models/fcn_hypotension_paper"
DATASET_NAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"
SUBJECT_SPLIT_PATH = DERIVED_ROOT / "manifests/hypotension_subject_split_fixed_v1.csv"

JEPA_D64_CKPT = DERIVED_ROOT / "models/jepa_native_d64/2026-08-12-native-jepa-d64-v1/best-val-epoch=40-loss=0.18978.ckpt"

OUTPUT_DIR = DERIVED_ROOT / "probing/medical_features_d64"

WINDOW_SAMPLES = 225000  # 1800s at 125Hz

# Features that require RESP signal — skip entirely from probing
RESP_DEPENDENT_FEATURES = {"RR", "PPV", "PVI", "RESP_amp"}

# 15 non-RESP features
PROBE_FEATURE_NAMES = [
    "HR", "SBP", "DBP", "PP", "MAP", "ABP_area", "PLETH_ACDC", "PLETH_amp",
    "ECG_Ramp", "HRV_RMSSD", "HR_range", "ShockIdx", "PTT", "dPdt_max", "ABP_tau",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Number of windows to process")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for encoder forward pass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip feature/embedding extraction, load from cache")
    return parser.parse_args()


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_sample_cache() -> pd.DataFrame:
    """Load all splits of PhysioJEPA sample cache and combine."""
    dfs = []
    for split in ["train", "val", "test"]:
        path = SAMPLE_CACHE_DIR / f"{DATASET_NAME}-{split}_samples.csv.gz"
        if path.is_file():
            df = pd.read_csv(path)
            df["split"] = split
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def read_window_signals(file_path: str, start_idx: int, end_idx: int) -> dict | None:
    """Read ABP, II, PLETH from Zarr container."""
    try:
        store = zarr.ZipStore(file_path, mode='r')
        root = zarr.open(store, mode='r')

        abp = np.array(root["ABP"][start_idx:end_idx], dtype=np.float32)
        ii = np.array(root["II"][start_idx:end_idx], dtype=np.float32)
        pleth = np.array(root["PLETH"][start_idx:end_idx], dtype=np.float32)
        store.close()

        return {"II": ii, "PLETH": pleth, "ABP": abp}
    except Exception:
        return None


def compute_stats_no_resp(seg_signals: dict, anchor_center: int) -> np.ndarray:
    """
    Compute 15 physiological features (skipping RESP-dependent ones) over 109
    sub-windows within a 20-min context centered on anchor_center.
    Returns shape (19, 109) in physical units.
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d

    ctx_start = anchor_center - HALF_CTX
    ctx_end = anchor_center + HALF_CTX

    ii_ctx = seg_signals["II"][ctx_start:ctx_end]
    pleth_ctx = seg_signals["PLETH"][ctx_start:ctx_end]
    abp_ctx = seg_signals["ABP"][ctx_start:ctx_end]

    out = np.full((N_FEATURES, N_SUBWIN), np.nan, dtype=np.float32)

    ii_filled = np.where(np.isnan(ii_ctx), np.nanmedian(ii_ctx), ii_ctx).copy()
    from steps.features import interp_nans, _dominant_peaks, clip_or_nan
    _, ecg_sign = _dominant_peaks(interp_nans(ii_filled), HR_MIN_DIST)

    for k in range(N_SUBWIN):
        s = k * SUB_WIN_STR
        e = s + SUB_WIN_SAMP

        ii_w = ii_ctx[s:e].copy()
        pleth_w = pleth_ctx[s:e].copy()
        abp_w = abp_ctx[s:e].copy()

        if np.isnan(ii_w).mean() > NAN_FRAC:
            continue
        if np.isnan(pleth_w).mean() > NAN_FRAC:
            continue
        if np.isnan(abp_w).mean() > NAN_FRAC:
            continue

        ii_w = interp_nans(ii_w)
        pleth_w = interp_nans(pleth_w)
        abp_w = interp_nans(abp_w)

        peaks_r, _ = find_peaks(ecg_sign * ii_w, distance=HR_MIN_DIST)
        peaks_s, _ = find_peaks(abp_w, distance=ABP_MIN_DIST)
        troughs_d, _ = find_peaks(-abp_w, distance=ABP_MIN_DIST)
        peaks_p, _ = find_peaks(pleth_w, distance=ABP_MIN_DIST)
        troughs_p, _ = find_peaks(-pleth_w, distance=ABP_MIN_DIST)

        out[0, k] = clip_or_nan(len(peaks_r) * (60.0 / WIN_SEC), *FEATURE_BOUNDS["HR"])

        if len(peaks_s) > 0 and len(troughs_d) > 0:
            sbp = clip_or_nan(float(np.median(abp_w[peaks_s])), *FEATURE_BOUNDS["SBP"])
            dbp = clip_or_nan(float(np.median(abp_w[troughs_d])), *FEATURE_BOUNDS["DBP"])
            if not (np.isnan(sbp) or np.isnan(dbp)):
                pp = clip_or_nan(sbp - dbp, *FEATURE_BOUNDS["PP"])
                out[2, k] = sbp
                out[3, k] = dbp
                out[4, k] = pp

        out[5, k] = clip_or_nan(float(np.mean(abp_w)), *FEATURE_BOUNDS["MAP"])

        if len(troughs_d) >= 2:
            areas = []
            for i in range(len(troughs_d) - 1):
                beat = abp_w[troughs_d[i]:troughs_d[i + 1]]
                area = float(np.sum(np.maximum(beat - abp_w[troughs_d[i]], 0.0)) / FS)
                if area > 0:
                    areas.append(area)
            if areas:
                out[6, k] = clip_or_nan(float(np.median(areas)), *FEATURE_BOUNDS["ABP_area"])

        if len(peaks_p) > 0 and len(troughs_p) > 0:
            ac = float(np.median(pleth_w[peaks_p]) - np.median(pleth_w[troughs_p]))
            dc = float(np.mean(pleth_w))
            if dc > 1e-6 and ac >= 0:
                out[7, k] = clip_or_nan(ac / dc, *FEATURE_BOUNDS["PLETH_ACDC"])

        if len(peaks_p) > 0 and len(troughs_p) > 0:
            amp = float(np.median(pleth_w[peaks_p]) - np.median(pleth_w[troughs_p]))
            out[8, k] = clip_or_nan(max(0.0, amp), *FEATURE_BOUNDS["PLETH_amp"])

        if len(peaks_r) > 0:
            baseline = float(np.mean(ii_w))
            if ecg_sign == 1:
                r_amp = float(np.median(ii_w[peaks_r])) - baseline
            else:
                r_amp = baseline - float(np.median(ii_w[peaks_r]))
            out[9, k] = clip_or_nan(max(0.0, r_amp), *FEATURE_BOUNDS["ECG_Ramp"])

        if len(peaks_r) >= 3:
            rr_ms = np.diff(peaks_r).astype(float) / FS * 1000
            rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
            out[10, k] = clip_or_nan(rmssd, *FEATURE_BOUNDS["HRV_RMSSD"])

        if len(peaks_r) >= 3:
            rr_s = np.diff(peaks_r).astype(float) / FS
            med_rr = np.median(rr_s)
            rr_clean = rr_s[(rr_s > 0.5 * med_rr) & (rr_s < 2.0 * med_rr)]
            if len(rr_clean) >= 2:
                hr_inst = 60.0 / rr_clean
                out[11, k] = clip_or_nan(
                    float(np.max(hr_inst) - np.min(hr_inst)), *FEATURE_BOUNDS["HR_range"])

        if not np.isnan(out[0, k]) and not np.isnan(out[2, k]) and out[2, k] > 0:
            out[12, k] = clip_or_nan(out[0, k] / out[2, k], *FEATURE_BOUNDS["ShockIdx"])

        if len(peaks_r) > 0 and len(troughs_d) > 0:
            ptt_list = []
            for r_idx in peaks_r:
                cands = troughs_d[(troughs_d > r_idx + 4) & (troughs_d < r_idx + 32)]
                if len(cands) > 0:
                    ptt_list.append(float(cands[0] - r_idx) / FS * 1000)
            if ptt_list:
                out[15, k] = clip_or_nan(float(np.median(ptt_list)), *FEATURE_BOUNDS["PTT"])

        if len(peaks_s) > 0 and len(troughs_d) > 0:
            abp_sm = uniform_filter1d(abp_w.astype(np.float64), size=3)
            grad_abp = np.diff(abp_sm) * FS
            dpdt_list = []
            for s_idx in peaks_s:
                prec = troughs_d[troughs_d < s_idx]
                if len(prec) == 0:
                    continue
                t_idx = int(prec[-1])
                if s_idx > t_idx + 1:
                    seg_grad = grad_abp[t_idx:s_idx]
                    if len(seg_grad) > 0:
                        dpdt_list.append(float(np.max(seg_grad)))
            if dpdt_list:
                out[16, k] = clip_or_nan(float(np.median(dpdt_list)), *FEATURE_BOUNDS["dPdt_max"])

        if len(peaks_s) > 0 and len(troughs_d) > 0:
            tau_list = []
            for s_idx in peaks_s:
                foll = troughs_d[troughs_d > s_idx]
                if len(foll) == 0:
                    continue
                t_end = int(foll[0])
                if t_end - s_idx < 8:
                    continue
                notch_lo = s_idx + 5
                notch_hi = min(s_idx + max(8, (t_end - s_idx) // 3), t_end - 3)
                fit_start = notch_lo
                if notch_hi > notch_lo + 2:
                    notch_cands, _ = find_peaks(-abp_w[notch_lo:notch_hi], distance=2)
                    if len(notch_cands) > 0:
                        fit_start = notch_lo + int(notch_cands[0])
                seg = abp_w[fit_start:t_end + 1].astype(np.float64)
                if len(seg) < 5:
                    continue
                y = np.maximum(seg, 1.0)
                t_a = np.arange(len(seg)) / FS
                try:
                    slope, _ = np.polyfit(t_a, np.log(y), 1)
                    if slope < 0:
                        tau_list.append(-1.0 / slope)
                except Exception:
                    pass
            if tau_list:
                out[17, k] = clip_or_nan(float(np.median(tau_list)), *FEATURE_BOUNDS["ABP_tau"])

    return out


def compute_features_for_window(signals: dict) -> np.ndarray:
    """Compute 19 features for a PhysioJEPA window (1800s), skipping RESP-dependent."""
    window_len = len(signals["II"])
    anchor_center = window_len // 2

    try:
        stats = compute_stats_no_resp(signals, anchor_center)  # (19, 109)
        return np.nanmedian(stats, axis=1)  # (19,)
    except Exception:
        return np.full(N_FEATURES, np.nan)


# ── Encoder Loading ───────────────────────────────────────────────────────────

def load_jepa_d64_encoder(device: torch.device):
    """Load frozen JEPA-d64 encoder."""
    from physiojepa.jepa import JEPASimpleLightning
    model = JEPASimpleLightning.load_from_checkpoint(str(JEPA_D64_CKPT), map_location="cpu")
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
    waveforms: (N, 3, 225000) float32
    Returns: (N, d_model) float32
    """
    n = len(waveforms)
    embeddings = []

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(waveforms[i:i+batch_size]).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            emb = encoder(batch)

        if isinstance(emb, tuple):
            emb = emb[0]

        # Pool to (bs, d_model) by mean over patches and channels
        if emb.dim() == 4:
            emb = emb.mean(dim=(1, 3))
        elif emb.dim() == 3:
            emb = emb.mean(dim=-1)

        embeddings.append(emb.float().cpu().numpy())

        if (i // batch_size) % 20 == 0:
            print(f"    Batch {i//batch_size}/{(n + batch_size - 1)//batch_size}")

    return np.concatenate(embeddings, axis=0)


# ── Probing ───────────────────────────────────────────────────────────────────

def run_probes(embeddings: np.ndarray, features: np.ndarray,
               patient_ids: np.ndarray) -> pd.DataFrame:
    """
    Train Ridge regression probes: embedding → each feature.
    Split by patient (80/20). Report R² and Pearson r per feature.
    """
    results = []

    # Filter out rows where embeddings contain NaN
    emb_valid = ~np.isnan(embeddings).any(axis=1)
    if not emb_valid.all():
        n_nan = (~emb_valid).sum()
        print(f"  Filtering {n_nan}/{len(embeddings)} rows with NaN embeddings")
        embeddings = embeddings[emb_valid]
        features = features[emb_valid]
        patient_ids = patient_ids[emb_valid]

    # Patient-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(embeddings, groups=patient_ids))

    X_train, X_test = embeddings[train_idx], embeddings[test_idx]

    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        if feat_name in RESP_DEPENDENT_FEATURES:
            continue

        y_all = features[:, feat_idx]

        train_valid = ~np.isnan(y_all[train_idx])
        test_valid = ~np.isnan(y_all[test_idx])

        n_train = train_valid.sum()
        n_test = test_valid.sum()

        if n_train < 50 or n_test < 20:
            results.append({
                "feature": feat_name,
                "r2": np.nan,
                "pearson_r": np.nan,
                "n_train": n_train,
                "n_test": n_test,
                "note": "insufficient_data",
            })
            continue

        Xtr = X_train[train_valid]
        ytr = y_all[train_idx][train_valid]
        Xte = X_test[test_valid]
        yte = y_all[test_idx][test_valid]

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Reuse the feature/waveform cache from the original probe if available
    original_cache = DERIVED_ROOT / "probing/medical_features" / f"cache_n{args.n_samples}_seed{args.seed}.npz"
    cache_path = OUTPUT_DIR / f"cache_n{args.n_samples}_seed{args.seed}.npz"

    if args.skip_extraction and cache_path.is_file():
        print(f"Loading cached data from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        waveforms_3ch = data["waveforms_3ch"]
        features = data["features"]
        patient_ids = data["patient_ids"]
    elif original_cache.is_file():
        # Reuse the original probe's waveform/feature cache (same data, different encoder)
        print(f"Reusing original probe cache from {original_cache}")
        data = np.load(original_cache, allow_pickle=True)
        waveforms_3ch = data["waveforms_3ch"]
        features = data["features"]
        patient_ids = data["patient_ids"]
        # Save a copy for this output dir
        np.savez_compressed(cache_path,
                            waveforms_3ch=waveforms_3ch,
                            features=features,
                            patient_ids=patient_ids)
    else:
        # ── Step 1: Sample windows ────────────────────────────────────────────
        print("Loading sample cache...")
        samples = load_sample_cache()
        print(f"  Total samples: {len(samples)}")

        rng = np.random.default_rng(args.seed)
        n = min(args.n_samples, len(samples))
        idx = rng.choice(len(samples), size=n, replace=False)
        samples = samples.iloc[idx].reset_index(drop=True)
        print(f"  Selected {len(samples)} samples")

        # ── Step 2: Extract waveforms + compute features ──────────────────────
        print("Extracting waveforms and computing features...")
        waveforms_3ch = np.zeros((len(samples), 3, WINDOW_SAMPLES), dtype=np.float32)
        features = np.full((len(samples), N_FEATURES), np.nan, dtype=np.float32)
        patient_ids = samples["subject_id"].values.copy()
        valid_mask = np.zeros(len(samples), dtype=bool)

        t0 = time.time()

        for i, row in samples.iterrows():
            if i % 500 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(samples) - i) / rate
                print(f"  [{i}/{len(samples)}] {rate:.1f} windows/s, ETA {eta/60:.1f} min")

            signals = read_window_signals(
                row["file_path"], int(row["start_idx"]), int(row["end_idx"])
            )
            if signals is None:
                continue

            expected = int(row["end_idx"]) - int(row["start_idx"])
            if any(len(signals[k]) != expected for k in signals):
                continue

            waveforms_3ch[i, 0] = signals["ABP"]
            waveforms_3ch[i, 1] = signals["II"]
            waveforms_3ch[i, 2] = signals["PLETH"]

            features[i] = compute_features_for_window(signals)
            valid_mask[i] = True

        valid_idx = np.where(valid_mask)[0]
        waveforms_3ch = waveforms_3ch[valid_idx]
        features = features[valid_idx]
        patient_ids = patient_ids[valid_idx]

        elapsed = time.time() - t0
        print(f"  Done: {len(valid_idx)}/{len(samples)} valid windows in {elapsed:.0f}s")

        np.savez_compressed(cache_path,
                            waveforms_3ch=waveforms_3ch,
                            features=features,
                            patient_ids=patient_ids)
        print(f"  Cached to {cache_path}")

    print(f"\nDataset: {len(features)} windows, {len(np.unique(patient_ids))} patients")

    # ── Step 3: Extract embeddings ────────────────────────────────────────────
    print("\n=== JEPA-d64 Encoder ===")
    jepa_emb_path = OUTPUT_DIR / f"jepa_d64_embeddings_n{len(features)}.npy"
    if jepa_emb_path.is_file() and args.skip_extraction:
        jepa_emb = np.load(jepa_emb_path)
    else:
        encoder = load_jepa_d64_encoder(device)
        jepa_emb = extract_embeddings(encoder, waveforms_3ch, args.batch_size, device)
        np.save(jepa_emb_path, jepa_emb)
        del encoder
        torch.cuda.empty_cache()
    print(f"  JEPA-d64 embeddings: {jepa_emb.shape}")

    # ── Step 4: Train probes ──────────────────────────────────────────────────
    print("\n=== Ridge Regression Probes ===")

    print("\n--- JEPA-d64 ---")
    jepa_results = run_probes(jepa_emb, features, patient_ids)
    print(jepa_results.to_string(index=False))

    # ── Step 5: Save results ──────────────────────────────────────────────────
    jepa_results["model"] = "JEPA-d64"
    results_path = OUTPUT_DIR / "probe_results_d64.csv"
    jepa_results.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SUMMARY: R² per feature (JEPA-d64)")
    print("=" * 50)
    print(f"{'Feature':<12} {'R²':>8} {'Pearson r':>10}")
    print("-" * 50)
    for _, row in jepa_results.iterrows():
        if row["note"] == "insufficient_data":
            continue
        r2 = f"{row['r2']:.3f}" if not np.isnan(row['r2']) else "N/A"
        r = f"{row['pearson_r']:.3f}" if not np.isnan(row['pearson_r']) else "N/A"
        print(f"{row['feature']:<12} {r2:>8} {r:>10}")
    print("=" * 50)


if __name__ == "__main__":
    main()
