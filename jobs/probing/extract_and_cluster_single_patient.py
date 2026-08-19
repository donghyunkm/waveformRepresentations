"""Extract 1000 embeddings for a single patient and run within-patient clustering.

Loads the frozen JEPA encoder, extracts full token-sequence embeddings for
N randomly sampled windows from a single patient, saves them with metadata,
then runs K-Means homogeneity sweep.

Usage:
    # Extract token embeddings, then cluster them
    python extract_and_cluster_single_patient.py 
        --patient p072908 
        --n-windows 1000 
        --output /path/to/output.npz

    # Reuse a saved artifact without loading a model or waveforms
    python extract_and_cluster_single_patient.py 
        --input /path/to/output.npz 
        --results-output /path/to/clustering_results.csv
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
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn.metrics import (
    homogeneity_completeness_v_measure,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, ForecastingDataset
from physiojepa.data_preprocessing import zarr_record_name
from physiojepa.jepa import JEPASimpleLightning


torch.set_float32_matmul_precision("high")

# --- Paths ---
ENCODER_CKPT = "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/jepa_native_paper/2026-08-04-native-jepa-paper-1gpu-debug-v1/best-val-epoch=13-loss=0.21508.ckpt"
OUTCOME_PATH = "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/labels/hypotension_labels_mimic_all_events_rolling5min.csv.gz"
SUBJECT_SPLIT_PATH = "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/manifests/hypotension_subject_split_fixed_v1.csv"
SAMPLE_CACHE_DIR = "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/fcn_hypotension_paper"
DATASET_FILENAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"
CONTAINERS_DIR = "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/containers"

# Dataset config (matches train_hypotension_fixed.yaml)
CHANNELS = ["ABP", "II", "PLETH"]
FREQUENCY = 125
SAMPLE_SEQ_LEN_SEC = 1800
FORECAST_WINDOW_SEC = [300]
CONSTANT_NAN_TOLERANCE = 0.2


def load_encoder(device: torch.device) -> nn.Module:
    """Load frozen JEPA encoder."""
    print(f"Loading encoder from {ENCODER_CKPT}")
    lightning_module = JEPASimpleLightning.load_from_checkpoint(
        ENCODER_CKPT, map_location=device
    )
    lightning_module.eval()
    lightning_module.to(device)
    return lightning_module


def build_patient_dataset(
    patient_id: str, n_windows: int, seed: int = 42
) -> tuple[ForecastingDataset, pd.DataFrame]:
    """Build dataset for a single patient, subsampled to n_windows."""
    # Load sample cache
    cache_path = Path(SAMPLE_CACHE_DIR) / f"{DATASET_FILENAME}-test_samples.csv.gz"
    samples = pd.read_csv(cache_path)
    samples["subject_id"] = samples["file_path"].map(
        lambda p: zarr_record_name(p).split("-", 1)[0]
    )

    # Filter to patient
    patient_samples = samples[samples["subject_id"] == patient_id].copy()
    print(f"Patient {patient_id}: {len(patient_samples)} total windows")

    # Subsample
    rng = np.random.default_rng(seed)
    if len(patient_samples) > n_windows:
        idx = rng.choice(len(patient_samples), n_windows, replace=False)
        idx.sort()
        patient_samples = patient_samples.iloc[idx].reset_index(drop=True)
    print(f"Subsampled to {len(patient_samples)} windows")

    # Load outcomes
    outcomes = pd.read_csv(OUTCOME_PATH)
    if "subject_id" not in outcomes:
        outcomes["subject_id"] = outcomes["file_path"].map(
            lambda p: zarr_record_name(p).split("-", 1)[0]
        )
    outcomes = outcomes[outcomes["subject_id"] == patient_id].copy()
    outcomes["Time Stamp (seconds)"] = outcomes["Time Stamp (seconds)"].round()

    # Override container paths if env var set
    containers_override = os.environ.get("PHYSIOJEPA_CONTAINERS_OVERRIDE")
    if containers_override:
        patient_samples["file_path"] = patient_samples["file_path"].map(
            lambda p: os.path.join(containers_override, os.path.basename(p))
        )
        outcomes["file_path"] = outcomes["file_path"].map(
            lambda p: os.path.join(containers_override, os.path.basename(p))
        )

    dataset = ForecastingDataset(
        channels=CHANNELS,
        forecast_window_sec=FORECAST_WINDOW_SEC,
        outcome_df=outcomes,
        outcome_df_outcome_col="hypotension_label",
        file_col="file_path",
        y_date_column="date",
        outcome_df_seconds_since_column="Time Stamp (seconds)",
        outcome_df_duration_column="event_length",
        sample_df=patient_samples,
        sample_seq_len_sec=SAMPLE_SEQ_LEN_SEC,
        frequency=FREQUENCY,
        butterworth_filters=None,
        median_filter_kernel_size=None,
        clip_interpolations=CLIP_INTERPOLATE_RANGES,
        constant_nan_tolerance=CONSTANT_NAN_TOLERANCE,
        require_all_channels=True,
        infer_forecast_windows=False,
        normalize_signals=True,
    )

    return dataset, patient_samples


@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    dataset: ForecastingDataset,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and labels from the dataset.

    Returns:
        embeddings: [N, 3, 512, 1800] float16
        labels: [N] int (hypotension labels)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=False,
    )

    all_embeddings = []
    all_labels = []

    for batch_idx, batch in enumerate(loader):
        x = batch[0].to(device)  # [B, C, seq_len]
        y = batch[1]  # labels

        # Reshape for encoder: [B, C, seq_len] -> encoder expects patched input
        # The JEPA encoder handles patching internally
        emb = encoder(x)  # [B, C, d_model, n_patches]

        all_embeddings.append(emb.cpu().half().numpy())
        all_labels.append(y.numpy() if hasattr(y, "numpy") else np.array(y))

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)}")

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return embeddings, labels


def run_clustering_analysis(
    embeddings: np.ndarray,
    hypo_labels: np.ndarray,
    sample_df: pd.DataFrame,
):
    """Run K-Means homogeneity sweep on within-patient embeddings."""
    n_windows = embeddings.shape[0]
    hypo_labels = np.asarray(hypo_labels).reshape(-1)
    if len(hypo_labels) != n_windows:
        raise ValueError(f"Expected {n_windows} labels, got {np.asarray(hypo_labels).shape}")
    if len(sample_df) != n_windows:
        raise ValueError(f"Expected {n_windows} sample rows, got {len(sample_df)}")
    print(f"\n{'='*60}")
    print(f"WITHIN-PATIENT CLUSTERING ANALYSIS")
    print(f"{'='*60}")
    print(f"Windows: {n_windows}, Embedding shape: {embeddings.shape}")

    # Temporal subsample: 1800 -> 20 evenly spaced patches
    n_patches_sub = 20
    patch_indices = np.linspace(0, embeddings.shape[3] - 1, n_patches_sub, dtype=int)
    emb_sub = embeddings[:, :, :, patch_indices]  # [N, 3, 512, 20]

    # PCA on feature dim (512 -> 121)
    tokens_flat = emb_sub.astype(np.float32).transpose(0, 1, 3, 2).reshape(-1, 512)
    print(f"Fitting PCA on {tokens_flat.shape[0]} tokens...")
    pca = PCA(n_components=121, random_state=42)
    tokens_pca = pca.fit_transform(tokens_flat)
    pca_var = pca.explained_variance_ratio_.sum()
    print(f"PCA: 121 components explain {pca_var:.1%} variance")

    tokens_pca = tokens_pca.reshape(n_windows, 3, n_patches_sub, 121)
    X = tokens_pca.reshape(n_windows, -1)  # [N, 7260]
    X = StandardScaler().fit_transform(X)

    # Labels for comparison
    # ICU stay as a grouping variable
    stay_ids = sample_df["file_path"].map(lambda p: Path(p).stem).values
    unique_stays = np.unique(stay_ids)
    stay_int = np.array([np.where(unique_stays == s)[0][0] for s in stay_ids])

    # Hypotension (binary)
    hypo_int = (hypo_labels > 0).astype(int)

    # Temporal position (early/mid/late tercile within each stay)
    temporal_tercile = np.zeros(n_windows, dtype=int)
    for stay in unique_stays:
        mask = stay_ids == stay
        n = mask.sum()
        indices = np.where(mask)[0]
        tercile_size = n // 3
        if tercile_size > 0:
            temporal_tercile[indices[:tercile_size]] = 0  # early
            temporal_tercile[indices[tercile_size:2*tercile_size]] = 1  # mid
            temporal_tercile[indices[2*tercile_size:]] = 2  # late

    print(f"\nReference labels:")
    print(f"  ICU stays: {len(unique_stays)}")
    print(f"  Hypotension prevalence: {hypo_int.mean():.4f}")
    print(f"  Temporal terciles: {np.bincount(temporal_tercile)}")

    # K-Means sweep
    k_values = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50]

    print(f"\n{'k':>3} | {'Sil':>6} | {'H(stay)':>8} {'C(stay)':>8} {'V(stay)':>8} | {'H(hypo)':>8} {'C(hypo)':>8} | {'H(time)':>8} {'C(time)':>8} | {'ARI(stay)':>10} {'ARI(time)':>10}")
    print("-" * 120)

    results = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))

        h_stay, c_stay, v_stay = homogeneity_completeness_v_measure(stay_int, labels)
        if np.unique(hypo_int).size > 1:
            h_hypo, c_hypo, v_hypo = homogeneity_completeness_v_measure(hypo_int, labels)
        else:
            h_hypo = c_hypo = v_hypo = float("nan")
        h_time, c_time, v_time = homogeneity_completeness_v_measure(temporal_tercile, labels)

        ari_stay = adjusted_rand_score(stay_int, labels)
        ari_time = adjusted_rand_score(temporal_tercile, labels)

        print(
            f"{k:>3} | {sil:>6.4f} | "
            f"{h_stay:>8.4f} {c_stay:>8.4f} {v_stay:>8.4f} | "
            f"{h_hypo:>8.4f} {c_hypo:>8.4f} | "
            f"{h_time:>8.4f} {c_time:>8.4f} | "
            f"{ari_stay:>10.4f} {ari_time:>10.4f}"
        )

        results.append({
            "k": k, "silhouette": sil,
            "h_stay": h_stay, "c_stay": c_stay, "v_stay": v_stay,
            "h_hypo": h_hypo, "c_hypo": c_hypo, "v_hypo": v_hypo,
            "h_time": h_time, "c_time": c_time, "v_time": v_time,
            "ari_stay": ari_stay, "ari_time": ari_time,
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", default="p072908", help="Patient ID")
    parser.add_argument("--n-windows", type=int, default=1000)
    parser.add_argument("--output", help="Output .npz path")
    parser.add_argument("--input", help="Reuse an existing embedding .npz artifact")
    parser.add_argument("--results-output", help="Optional clustering-results CSV")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.input:
        artifact = np.load(args.input, allow_pickle=True)
        embeddings = artifact["embeddings"]
        labels = np.asarray(artifact["hypotension_label"]).reshape(-1)
        sample_df = pd.DataFrame({"file_path": artifact["file_path"].astype(str), "start_idx": artifact["start_idx"], "end_idx": artifact["end_idx"]})
        print(f"Loaded cached embeddings from {args.input}")
        results = run_clustering_analysis(embeddings, labels, sample_df)
        if args.results_output:
            Path(args.results_output).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(args.results_output, index=False)
            print(f"Saved clustering results to {args.results_output}")
        return
    if not args.output:
        parser.error("--output is required unless --input is provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build dataset
    dataset, sample_df = build_patient_dataset(
        args.patient, args.n_windows, args.seed
    )
    print(f"Dataset length: {len(dataset)}")

    # Load encoder
    encoder = load_encoder(device)

    # Extract embeddings
    print(f"\nExtracting embeddings...")
    t0 = time.time()
    embeddings, labels = extract_embeddings(
        encoder, dataset, device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    elapsed = time.time() - t0
    print(f"Extracted {embeddings.shape[0]} embeddings in {elapsed:.1f}s")
    print(f"Embedding shape: {embeddings.shape}")

    # Save embeddings with metadata
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        hypotension_label=labels,
        subject_id=np.array([args.patient] * len(embeddings)),
        file_path=sample_df["file_path"].values,
        start_idx=sample_df["start_idx"].values,
        end_idx=sample_df["end_idx"].values,
        unique_identifier=sample_df["unique_identifier"].values if "unique_identifier" in sample_df else sample_df["file_path"].values,
    )
    print(f"\nSaved embeddings to {output_path}")

    # Run clustering analysis
    results = run_clustering_analysis(embeddings, labels, sample_df)

    print("\nDone.")


if __name__ == "__main__":
    main()
