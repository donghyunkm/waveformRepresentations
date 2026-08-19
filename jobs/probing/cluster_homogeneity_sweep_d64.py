"""K-Means homogeneity analysis sweep for JEPA-d64 embeddings.

Extracts token embeddings from the d64 encoder for the same 20 patients / 50 windows
used in the original sweep, then runs K-Means k=2–50 measuring cluster alignment with
patient identity, hemodynamic state, and hypotension labels.

Supports resuming: if the embedding .npz already exists, skips extraction.

Usage:
    python cluster_homogeneity_sweep_d64.py [--batch_size 8] [--seed 42]
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

from sklearn.cluster import KMeans
from sklearn.metrics import (
    homogeneity_completeness_v_measure,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
CACHE_DIR = DERIVED_ROOT / "models/fcn_hypotension_paper"
DATASET_NAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"
CONTAINERS_DIR = DERIVED_ROOT / "containers"

ENCODER_CKPT = (
    DERIVED_ROOT / "models/jepa_native_d64/2026-08-12-native-jepa-d64-v1/"
    "best-val-epoch=40-loss=0.18978.ckpt"
)
HEMO_CLUSTERS_PATH = (
    DERIVED_ROOT / "probing/clustering/window_hemo_clusters.npz"
)
OUTPUT_DIR = DERIVED_ROOT / "embeddings/patient_token_sequences_d64"


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
    return np.stack(arrays, axis=0).astype(np.float32)


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
# Extraction
# ---------------------------------------------------------------------------

def extract_embeddings(args, df, selected_indices) -> np.ndarray:
    """Extract token-level embeddings from d64 encoder."""
    from physiojepa.jepa import JEPASimpleLightning

    print("Loading JEPA-d64 encoder...")
    encoder = JEPASimpleLightning.load_from_checkpoint(
        str(ENCODER_CKPT), map_location="cpu"
    )
    encoder = encoder.to(args.device).eval()
    print(f"  d_model={encoder.d_model}, patch_size={encoder.patch_size}, "
          f"n_patches={encoder.num_patch}, n_channels={encoder.c_in}")

    all_embeddings = []
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_start in range(0, len(selected_indices), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(selected_indices))
            batch_idx_slice = selected_indices[batch_start:batch_end]

            waveforms = []
            for idx in batch_idx_slice:
                row = df.iloc[idx]
                wf = load_window(row["file_path"], int(row["start_idx"]), int(row["end_idx"]))
                wf = normalize_iqr(wf)
                wf = np.nan_to_num(wf, nan=0.0)
                waveforms.append(wf)

            batch_tensor = torch.from_numpy(np.stack(waveforms)).to(args.device)
            emb = encoder(batch_tensor)  # [bs, n_channels, d_model, n_patches]
            all_embeddings.append(emb.cpu().float().numpy())

            done = batch_end
            if (done % 50 == 0) or done == len(selected_indices):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  {done}/{len(selected_indices)} windows "
                      f"({elapsed:.0f}s, {rate:.1f} win/s)")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float16)
    print(f"  Final embedding shape: {embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# Homogeneity Analysis
# ---------------------------------------------------------------------------

def run_homogeneity_sweep(embeddings, subject_ids, hemo_clusters, hypo_labels):
    """Run K-Means sweep k=2–50 and report alignment metrics."""
    n_windows = embeddings.shape[0]
    d_model = embeddings.shape[2]
    n_patches_total = embeddings.shape[3]

    print(f"\nEmbeddings: {embeddings.shape}")
    print(f"Unique patients: {len(np.unique(subject_ids))}")
    print(f"Hemo cluster distribution: {np.unique(hemo_clusters, return_counts=True)}")
    print(f"Hypotension prevalence: {hypo_labels.mean():.3f}")
    print()

    # Temporal subsample: n_patches -> 20 evenly spaced patches
    n_patches_sub = 20
    patch_indices = np.linspace(0, n_patches_total - 1, n_patches_sub, dtype=int)
    emb_sub = embeddings[:, :, :, patch_indices]  # [N, 3, d_model, 20]

    # PCA on feature dim (d_model -> min(121, d_model))
    n_pca = min(121, d_model)
    tokens_flat = emb_sub.transpose(0, 1, 3, 2).reshape(-1, d_model)  # [N*3*20, d_model]
    print(f"Running PCA: {d_model} -> {n_pca} components on {tokens_flat.shape[0]} tokens...")
    pca = PCA(n_components=n_pca, random_state=42)
    tokens_pca = pca.fit_transform(tokens_flat)
    tokens_pca = tokens_pca.reshape(n_windows, 3, n_patches_sub, n_pca)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Flatten per window
    X = tokens_pca.reshape(n_windows, -1)
    X = StandardScaler().fit_transform(X)
    print(f"  Feature matrix: {X.shape}")

    # Map subject_ids to integer labels
    unique_subjects = np.unique(subject_ids)
    subject_int = np.array(
        [np.where(unique_subjects == s)[0][0] for s in subject_ids]
    )

    # K-Means sweep
    k_values = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50]

    header = (
        f"{'k':>3} | {'Sil':>6} | "
        f"{'H(pat)':>7} {'C(pat)':>7} {'V(pat)':>7} | "
        f"{'H(hemo)':>8} {'C(hemo)':>8} {'V(hemo)':>8} | "
        f"{'H(hypo)':>8} {'C(hypo)':>8} {'V(hypo)':>8} | "
        f"{'ARI(pat)':>9} {'ARI(hemo)':>10}"
    )
    print("\n" + header)
    print("-" * len(header))

    results = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)

        sil = silhouette_score(X, labels, sample_size=min(1000, len(X)))

        h_pat, c_pat, v_pat = homogeneity_completeness_v_measure(
            subject_int, labels
        )
        h_hemo, c_hemo, v_hemo = homogeneity_completeness_v_measure(
            hemo_clusters, labels
        )
        h_hypo, c_hypo, v_hypo = homogeneity_completeness_v_measure(
            hypo_labels, labels
        )

        ari_pat = adjusted_rand_score(subject_int, labels)
        ari_hemo = adjusted_rand_score(hemo_clusters, labels)

        print(
            f"{k:>3} | {sil:>6.4f} | "
            f"{h_pat:>7.4f} {c_pat:>7.4f} {v_pat:>7.4f} | "
            f"{h_hemo:>8.4f} {c_hemo:>8.4f} {v_hemo:>8.4f} | "
            f"{h_hypo:>8.4f} {c_hypo:>8.4f} {v_hypo:>8.4f} | "
            f"{ari_pat:>9.4f} {ari_hemo:>10.4f}"
        )

        results.append({
            "k": k, "silhouette": sil,
            "h_patient": h_pat, "c_patient": c_pat, "v_patient": v_pat,
            "h_hemo": h_hemo, "c_hemo": c_hemo, "v_hemo": v_hemo,
            "h_hypo": h_hypo, "c_hypo": c_hypo, "v_hypo": v_hypo,
            "ari_patient": ari_pat, "ari_hemo": ari_hemo,
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "homogeneity_sweep_d64.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_patients", type=int, default=20)
    parser.add_argument("--n_windows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    out_path = (
        OUTPUT_DIR
        / f"token_embeddings_{args.n_patients}patients_{args.n_windows}windows_seed{args.seed}.npz"
    )

    if out_path.is_file():
        # Resume: embeddings already extracted, skip to analysis
        print(f"Loading cached embeddings from {out_path}")
        data = np.load(out_path, allow_pickle=True)
        embeddings = data["embeddings"].astype(np.float32)
        subject_ids = data["subject_id"]
        hemo_clusters = data["hemo_cluster"]
        hypo_labels = data["hypotension_label"]
    else:
        # --- Load test samples ---
        print("Loading test samples...")
        df = pd.read_csv(CACHE_DIR / f"{DATASET_NAME}-test_samples.csv.gz")
        print(f"  {len(df)} windows, {df['subject_id'].nunique()} patients")

        # --- Load hemo clusters ---
        print("Loading hemo clusters...")
        hemo_data = np.load(HEMO_CLUSTERS_PATH, allow_pickle=True)
        hemo_clusters_all = hemo_data["hemo_clusters"]
        assert len(hemo_clusters_all) == len(df), "Hemo clusters length mismatch"

        # --- Select patients with enough windows ---
        windows_per_patient = df.groupby("subject_id").size()
        eligible = windows_per_patient[windows_per_patient >= args.n_windows].index.tolist()
        print(f"  {len(eligible)} patients with >= {args.n_windows} windows")

        if len(eligible) < args.n_patients:
            print(f"  WARNING: Only {len(eligible)} eligible patients, using all")
            selected_patients = eligible
        else:
            selected_patients = rng.choice(eligible, size=args.n_patients, replace=False).tolist()

        print(f"  Selected {len(selected_patients)} patients")

        # --- Sample windows per patient ---
        selected_indices = []
        for pid in selected_patients:
            patient_indices = np.where(df["subject_id"].values == pid)[0]
            chosen = rng.choice(patient_indices, size=args.n_windows, replace=False)
            selected_indices.extend(chosen.tolist())

        selected_indices = np.array(selected_indices, dtype=np.int64)
        print(f"  Total windows to extract: {len(selected_indices)}")

        # --- Extract embeddings ---
        embeddings = extract_embeddings(args, df, selected_indices)

        # --- Assemble metadata ---
        subject_ids = df.iloc[selected_indices]["subject_id"].values.astype(str)
        hemo_clusters = hemo_clusters_all[selected_indices]
        hypo_labels = df.iloc[selected_indices]["outcome_val_300sec"].values.astype(np.int64)

        # --- Save ---
        np.savez_compressed(
            out_path,
            embeddings=embeddings,
            subject_id=subject_ids,
            hemo_cluster=hemo_clusters,
            hypotension_label=hypo_labels,
            unique_identifier=df.iloc[selected_indices]["unique_identifier"].values.astype(str),
            start_idx=df.iloc[selected_indices]["start_idx"].values.astype(np.int64),
            end_idx=df.iloc[selected_indices]["end_idx"].values.astype(np.int64),
            file_path=df.iloc[selected_indices]["file_path"].values.astype(str),
            test_sample_idx=selected_indices,
        )
        file_size_mb = out_path.stat().st_size / 1e6
        print(f"\n  Saved embeddings to {out_path} ({file_size_mb:.1f} MB)")

        embeddings = embeddings.astype(np.float32)

    # --- Run homogeneity sweep ---
    run_homogeneity_sweep(embeddings, subject_ids, hemo_clusters, hypo_labels)


if __name__ == "__main__":
    main()
