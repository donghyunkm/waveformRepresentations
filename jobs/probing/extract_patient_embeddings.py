"""Extract full token-sequence embeddings for a random subset of test patients.

Saves N random windows from M random test patients with full token-level
embeddings, plus metadata for alignment with icuDataExtraction features and
hemo clusters.

Output (single .npz file):
    embeddings: float16 [N*M, n_channels, d_model, n_patches]  — full token sequences
    subject_id: object [N*M]  — patient ID (join key for icuDataExtraction)
    unique_identifier: object [N*M]  — ICU stay ID (subject + admission time)
    start_idx: int64 [N*M]  — sample start index in zarr container
    end_idx: int64 [N*M]  — sample end index in zarr container
    file_path: object [N*M]  — zarr container path
    test_sample_idx: int64 [N*M]  — row index into test_samples.csv.gz
        (aligns 1:1 with window_hemo_clusters.npz arrays)
    hypotension_label: int64 [N*M]  — outcome_val_300sec from sample cache
    hemo_cluster: int64 [N*M]  — hemodynamic cluster label (-1 = unmatched)

Usage:
    python extract_patient_embeddings.py [--n_patients 20] [--n_windows 50] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CACHE_DIR = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/fcn_hypotension_paper"
)
DATASET_NAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"
CONTAINERS_DIR = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/containers"
)
ENCODER_CKPT = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/"
    "jepa_native_paper/2026-08-04-native-jepa-paper-1gpu-debug-v1/"
    "best-val-epoch=13-loss=0.21508.ckpt"
)
HEMO_CLUSTERS_PATH = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/probing/"
    "clustering/window_hemo_clusters.npz"
)
OUTPUT_DIR = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/"
    "patient_token_sequences"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_window(file_path: str, start_idx: int, end_idx: int,
                channels=("ABP", "II", "PLETH")) -> np.ndarray:
    """Load a single window from a ZipStore container."""
    store = zarr.ZipStore(file_path, mode="r")
    root = zarr.open(store, mode="r")
    arrays = []
    for ch in channels:
        arr = root[ch][start_idx:end_idx]
        arrays.append(arr)
    store.close()
    return np.stack(arrays, axis=0).astype(np.float32)  # [n_channels, seq_len]


def normalize_iqr(x: np.ndarray) -> np.ndarray:
    """IQR-normalize each channel independently."""
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        ch = x[i]
        q25, q75 = np.nanpercentile(ch, [25, 75])
        iqr = q75 - q25
        if iqr < 1e-8:
            iqr = 1.0
        median = np.nanmedian(ch)
        out[i] = (ch - median) / iqr
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_patients", type=int, default=20,
                        help="Number of patients to sample")
    parser.add_argument("--n_windows", type=int, default=50,
                        help="Number of windows per patient")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Inference batch size (reduce if OOM)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_total = args.n_patients * args.n_windows

    # --- Load test samples ---
    print("Loading test samples...")
    df = pd.read_csv(CACHE_DIR / f"{DATASET_NAME}-test_samples.csv.gz")
    print(f"  {len(df)} windows, {df['subject_id'].nunique()} patients")

    # --- Load hemo clusters (aligned 1:1 with test samples) ---
    print("Loading hemo clusters...")
    hemo_data = np.load(HEMO_CLUSTERS_PATH, allow_pickle=True)
    hemo_clusters = hemo_data["hemo_clusters"]
    assert len(hemo_clusters) == len(df), "Hemo clusters length mismatch"

    # --- Select patients with enough windows ---
    windows_per_patient = df.groupby("subject_id").size()
    eligible = windows_per_patient[windows_per_patient >= args.n_windows].index.tolist()
    print(f"  {len(eligible)} patients with >= {args.n_windows} windows")

    if len(eligible) < args.n_patients:
        print(f"  WARNING: Only {len(eligible)} eligible patients, using all")
        selected_patients = eligible
    else:
        selected_patients = rng.choice(eligible, size=args.n_patients, replace=False).tolist()

    print(f"  Selected {len(selected_patients)} patients: {selected_patients[:5]}...")

    # --- Sample windows per patient ---
    selected_indices = []
    for pid in selected_patients:
        patient_indices = np.where(df["subject_id"].values == pid)[0]
        chosen = rng.choice(patient_indices, size=args.n_windows, replace=False)
        selected_indices.extend(chosen.tolist())

    selected_indices = np.array(selected_indices, dtype=np.int64)
    print(f"  Total windows to extract: {len(selected_indices)}")

    # --- Load encoder ---
    print("Loading encoder...")
    from physiojepa.jepa import JEPASimpleLightning

    encoder = JEPASimpleLightning.load_from_checkpoint(
        str(ENCODER_CKPT), map_location="cpu"
    )
    encoder = encoder.to(args.device).eval()
    # Output of encoder.forward(x): [bs, n_channels, d_model, n_patches]
    print(f"  Encoder loaded on {args.device}")
    print(f"  d_model={encoder.d_model}, patch_size={encoder.patch_size}, "
          f"n_patches={encoder.num_patch}, n_channels={encoder.c_in}")

    # --- Extract embeddings ---
    print("\nExtracting embeddings...")
    all_embeddings = []
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, len(selected_indices), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(selected_indices))
            batch_idx_slice = selected_indices[batch_start:batch_end]

            # Load and preprocess waveforms
            waveforms = []
            for idx in batch_idx_slice:
                row = df.iloc[idx]
                wf = load_window(row["file_path"], int(row["start_idx"]), int(row["end_idx"]))
                wf = normalize_iqr(wf)
                wf = np.nan_to_num(wf, nan=0.0)
                waveforms.append(wf)

            batch_tensor = torch.from_numpy(np.stack(waveforms)).to(args.device)
            # batch_tensor: [bs, n_channels, seq_len]

            # Forward through encoder → [bs, n_channels, d_model, n_patches]
            emb = encoder(batch_tensor)
            all_embeddings.append(emb.cpu().float().numpy())

            done = batch_end
            if (done % 50 == 0) or done == len(selected_indices):
                elapsed = time.time() - t0
                rate = done / elapsed
                print(f"  {done}/{len(selected_indices)} windows "
                      f"({elapsed:.0f}s, {rate:.1f} win/s)")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float16)
    print(f"\n  Final embedding shape: {embeddings.shape}")
    print(f"  dtype: {embeddings.dtype}")
    print(f"  Size in memory: {embeddings.nbytes / 1e9:.2f} GB")

    # --- Assemble metadata ---
    metadata_arrays = {
        "subject_id": df.iloc[selected_indices]["subject_id"].values.astype(str),
        "unique_identifier": df.iloc[selected_indices]["unique_identifier"].values.astype(str),
        "start_idx": df.iloc[selected_indices]["start_idx"].values.astype(np.int64),
        "end_idx": df.iloc[selected_indices]["end_idx"].values.astype(np.int64),
        "file_path": df.iloc[selected_indices]["file_path"].values.astype(str),
        "test_sample_idx": selected_indices,
        "hypotension_label": df.iloc[selected_indices]["outcome_val_300sec"].values.astype(np.int64),
        "hemo_cluster": hemo_clusters[selected_indices],
    }

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        OUTPUT_DIR
        / f"token_embeddings_{args.n_patients}patients_{args.n_windows}windows_seed{args.seed}.npz"
    )

    np.savez_compressed(out_path, embeddings=embeddings, **metadata_arrays)

    file_size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved: {out_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"\n--- Alignment guide ---")
    print(f"  subject_id → join with icuDataExtraction patient_ids.npy")
    print(f"  unique_identifier → ICU stay (subject + admission)")
    print(f"  test_sample_idx → positional index into:")
    print(f"    • {DATASET_NAME}-test_samples.csv.gz")
    print(f"    • window_hemo_clusters.npz (hemo_clusters, match_offsets, patient_ids)")
    print(f"  hemo_cluster → hemodynamic cluster (0–6, or -1 if unmatched)")
    print(f"  hypotension_label → 5-min-ahead hypotension event (0/1)")


if __name__ == "__main__":
    main()
