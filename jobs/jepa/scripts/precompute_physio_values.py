"""
Pre-compute HR and MAP values for the physio-contrastive JEPA dataset.

Reads HR and MAP directly from MIMIC bedside vital signs (numerics records at 1 Hz)
rather than computing them from raw waveforms. This is:
  - Fast: reads lightweight 1Hz numerics instead of opening ZipStore containers
  - Accurate: uses the bedside monitor's validated QRS detector for HR
    (avoids the T-wave double-counting bug in our R-peak detection)
  - Complete: every PhysioJEPA container has a paired numerics record

Strategy:
  1. Group samples by stay_id (= one numerics record per stay)
  2. For each stay: read the numerics record once, extract HR and ABP Mean
     for all samples from that stay via simple array slicing
  3. Parallelize across stays with multiprocessing

Output: physio_values_{train,val}.npz files in the sample cache directory,
compatible with SelfSupervisedDatasetWithPhysioValues.

Usage:
    python precompute_physio_values.py [--workers 16]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "train_physio_contrastive_jepa.yaml"

MANIFEST_PATH = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/manifests/waveform_manifest.csv"
)
SAMPLE_CACHE_DIR = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/sample_indices/"
    "jepa_native_paper_leakage_safe_v1"
)
OUTPUT_DIR = SAMPLE_CACHE_DIR  # Same directory as training script expects

TRAIN_CACHE = "zipstore_ABP_II_PLETH_125Hz_1800sec_stride1800_jepa_leakage_safe_v1-train_samples.csv.gz"
VAL_CACHE = "zipstore_ABP_II_PLETH_125Hz_1800sec_stride1800_jepa_leakage_safe_v1-val_samples.csv.gz"

FS_WAVEFORM = 125  # Hz
SAMPLE_DURATION_SEC = 1800  # 30 minutes per window


# ── Core Logic ────────────────────────────────────────────────────────────────

def load_manifest() -> pd.DataFrame:
    """Load waveform manifest with numeric header paths."""
    manifest = pd.read_csv(MANIFEST_PATH)
    return manifest[["stay_id", "header_path", "numeric_header_path"]].copy()


def get_base_datetime(header_path: str) -> datetime | None:
    """Read base_datetime from a WFDB header."""
    try:
        rec = wfdb.rdheader(header_path.replace(".hea", ""))
        return rec.base_datetime
    except Exception:
        return None


def process_stay(args: tuple) -> dict:
    """
    Process all samples from a single stay.

    Args:
        args: (stay_id, stay_samples_df, waveform_header_path, numeric_header_path)

    Returns:
        dict with keys: stay_id, indices, hr_values, map_values
    """
    stay_id, samples, waveform_header_path, numeric_header_path = args

    indices = samples.index.values
    n = len(samples)
    hr_values = np.full(n, np.nan, dtype=np.float32)
    map_values = np.full(n, np.nan, dtype=np.float32)

    # Get waveform record start time
    wave_dt = get_base_datetime(waveform_header_path)
    if wave_dt is None:
        return {"stay_id": stay_id, "indices": indices,
                "hr_values": hr_values, "map_values": map_values}

    # Read numerics header to find HR and ABP Mean channels
    try:
        num_header_path = numeric_header_path.replace(".hea", "")
        num_rec = wfdb.rdheader(num_header_path)
    except Exception:
        return {"stay_id": stay_id, "indices": indices,
                "hr_values": hr_values, "map_values": map_values}

    num_dt = num_rec.base_datetime
    if num_dt is None:
        return {"stay_id": stay_id, "indices": indices,
                "hr_values": hr_values, "map_values": map_values}

    sig_names = [s.upper().strip() for s in num_rec.sig_name]
    hr_chan = None
    map_chan = None

    # Find HR channel (prefer "HR", fall back to "PULSE")
    for i, name in enumerate(sig_names):
        if name == "HR":
            hr_chan = i
            break
    if hr_chan is None:
        for i, name in enumerate(sig_names):
            if name == "PULSE":
                hr_chan = i
                break

    # Find MAP channel (prefer "ABP MEAN", fall back to "ABP")
    for i, name in enumerate(sig_names):
        if name == "ABP MEAN":
            map_chan = i
            break
    if map_chan is None:
        for i, name in enumerate(sig_names):
            if name == "NBP MEAN":
                map_chan = i
                break

    if hr_chan is None and map_chan is None:
        return {"stay_id": stay_id, "indices": indices,
                "hr_values": hr_values, "map_values": map_values}

    # Compute time offset between waveform and numerics records
    # Both have base_datetime; numerics is at 1 Hz
    time_offset_sec = (wave_dt - num_dt).total_seconds()
    num_len = num_rec.sig_len

    # Read the full numerics record once
    try:
        signals, _ = wfdb.rdsamp(num_header_path)
    except Exception:
        return {"stay_id": stay_id, "indices": indices,
                "hr_values": hr_values, "map_values": map_values}

    # Process each sample
    for local_idx, (_, row) in enumerate(samples.iterrows()):
        start_idx = int(row["start_idx"])

        # Convert waveform start_idx to absolute time offset into numerics
        sample_time_in_wave = start_idx / FS_WAVEFORM
        sample_time_in_num = sample_time_in_wave + time_offset_sec

        num_start = int(round(sample_time_in_num))
        num_end = num_start + SAMPLE_DURATION_SEC

        # Bounds check
        if num_start < 0 or num_end > num_len or num_start >= num_end:
            continue

        # Extract HR
        if hr_chan is not None:
            hr_segment = signals[num_start:num_end, hr_chan]
            # Filter out zeros and NaN (invalid readings)
            valid_hr = hr_segment[(hr_segment > 20) & (hr_segment < 250) & ~np.isnan(hr_segment)]
            if len(valid_hr) > SAMPLE_DURATION_SEC * 0.5:  # need >50% valid
                hr_values[local_idx] = np.median(valid_hr)

        # Extract MAP
        if map_chan is not None:
            map_segment = signals[num_start:num_end, map_chan]
            valid_map = map_segment[(map_segment > 20) & (map_segment < 200) & ~np.isnan(map_segment)]
            if len(valid_map) > SAMPLE_DURATION_SEC * 0.5:
                map_values[local_idx] = np.median(valid_map)

    return {"stay_id": stay_id, "indices": indices,
            "hr_values": hr_values, "map_values": map_values}


def compute_physio_values(
    sample_df: pd.DataFrame,
    manifest: pd.DataFrame,
    n_workers: int = 16,
    desc: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute HR and MAP for all samples using bedside numerics.

    Groups by stay_id and processes in parallel.
    """
    n = len(sample_df)
    hr_values = np.full(n, np.nan, dtype=np.float32)
    map_values = np.full(n, np.nan, dtype=np.float32)

    # Build stay_id -> manifest row mapping
    manifest_by_stay = manifest.set_index("stay_id")

    # Group samples by stay_id
    grouped = sample_df.groupby("stay_id")
    n_stays = len(grouped)

    # Prepare arguments for multiprocessing
    args_list = []
    for stay_id, group_df in grouped:
        if stay_id not in manifest_by_stay.index:
            continue
        row = manifest_by_stay.loc[stay_id]
        args_list.append((
            stay_id,
            group_df,
            row["header_path"],
            row["numeric_header_path"],
        ))

    print(f"  Processing {len(args_list)} stays ({n} samples) with {n_workers} workers...")
    t0 = time.time()

    if n_workers <= 1:
        results = [process_stay(a) for a in args_list]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(process_stay, args_list, chunksize=16)

    # Collect results
    for result in results:
        indices = result["indices"]
        # Map from group-local indices back to the original dataframe positions
        for local_idx, global_idx in enumerate(indices):
            hr_values[global_idx] = result["hr_values"][local_idx]
            map_values[global_idx] = result["map_values"][local_idx]

    elapsed = time.time() - t0
    hr_valid = (~np.isnan(hr_values)).sum()
    map_valid = (~np.isnan(map_values)).sum()
    both_valid = ((~np.isnan(hr_values)) & (~np.isnan(map_values))).sum()

    print(f"  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  HR valid:  {hr_valid}/{n} ({100*hr_valid/n:.1f}%)")
    print(f"  MAP valid: {map_valid}/{n} ({100*map_valid/n:.1f}%)")
    print(f"  Both valid: {both_valid}/{n} ({100*both_valid/n:.1f}%)")
    if hr_valid > 0:
        print(f"  HR range: {np.nanmin(hr_values):.0f}–{np.nanmax(hr_values):.0f} bpm "
              f"(median {np.nanmedian(hr_values):.0f})")
    if map_valid > 0:
        print(f"  MAP range: {np.nanmin(map_values):.0f}–{np.nanmax(map_values):.0f} mmHg "
              f"(median {np.nanmedian(map_values):.0f})")

    return hr_values, map_values


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute physio values from bedside numerics")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel workers")
    parser.add_argument("--split", type=str, default="both",
                        choices=["train", "val", "both"],
                        help="Which split to compute")
    args = parser.parse_args()

    print("=" * 70)
    print("Pre-computing physiological values from bedside numerics")
    print("=" * 70)
    print(f"  Workers: {args.workers}")
    print(f"  Split: {args.split}")
    print(f"  Source: MIMIC numerics records (1 Hz HR + ABP Mean)")
    print()

    # Load manifest
    print("Loading waveform manifest...")
    manifest = load_manifest()
    print(f"  {len(manifest)} records with numerics")

    # Process each split
    splits = ["train", "val"] if args.split == "both" else [args.split]

    for split in splits:
        print(f"\n{'─' * 70}")
        print(f"  Processing {split} split")
        print(f"{'─' * 70}")

        cache_file = TRAIN_CACHE if split == "train" else VAL_CACHE
        sample_df = pd.read_csv(SAMPLE_CACHE_DIR / cache_file)
        print(f"  Loaded {len(sample_df)} samples, {sample_df['stay_id'].nunique()} stays")

        hr_values, map_values = compute_physio_values(
            sample_df, manifest, n_workers=args.workers, desc=split
        )

        # Save
        output_path = OUTPUT_DIR / f"physio_values_{split}.npz"
        np.savez_compressed(output_path, hr_values=hr_values, map_values=map_values)
        print(f"  Saved: {output_path}")
        print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
