"""
Unsupervised clustering of JEPA/PatchTST embeddings and analysis of how
clusters relate to:
  1. Hypotension labels (binary clinical outcome)
  2. The 7 hemodynamic clusters from icuDataExtraction

Extracts embeddings with full metadata (patient_id, file_path, start_idx)
to enable cross-referencing with icuDataExtraction's cluster assignments.

Usage:
    python cluster_embeddings.py [--n-samples 10000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "/gpfs/home/dk5565/PhysioJEPA")
sys.path.insert(0, str(Path("/gpfs/home/dk5565/PhysioJEPA/jobs/jepa/scripts")))

from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, ForecastingDataset
from physiojepa.data_preprocessing import zarr_record_name
from physiojepa.jepa import JEPASimpleLightning
from physiojepa.patchtst import PatchTFTSimpleLightning

# ── Paths ─────────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
SUBJECT_SPLIT_PATH = DERIVED_ROOT / "manifests/hypotension_subject_split_fixed_v1.csv"
OUTCOME_DF_PATH = DERIVED_ROOT / "labels/hypotension_labels_mimic_all_events_rolling5min.csv.gz"
SAMPLE_CACHE_DIR = DERIVED_ROOT / "models/fcn_hypotension_paper"
DATASET_FILENAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"

JEPA_CKPT = DERIVED_ROOT / "models/jepa_native_paper/2026-08-04-native-jepa-paper-1gpu-debug-v1/best-val-epoch=13-loss=0.21508.ckpt"
PTST_CKPT = DERIVED_ROOT / "models/patchtst_self_supervised_paper/2026-08-05-patchtst-paper-1gpu-v1/best-val-epoch=03-loss=0.00329.ckpt"

OUTPUT_DIR = DERIVED_ROOT / "probing/clustering"

# icuDataExtraction paths
ICU_OUTPUT = Path("/gpfs/home/dk5565/icuDataExtraction/output_v2")

CHANNELS = ["ABP", "II", "PLETH"]
FREQUENCY = 125
SAMPLE_SEQ_LEN_SECONDS = 1800
FORECAST_WINDOW_SEC = [300]
BATCH_SIZE = 64
NUM_WORKERS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=0,
                        help="Number of samples (0 = use entire split)")
    parser.add_argument("--n-clusters", type=int, default=7,
                        help="Number of KMeans clusters (match icuDataExtraction)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-extraction", action="store_true")
    return parser.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset(split: str) -> tuple[ForecastingDataset, pd.DataFrame]:
    """Load a dataset split and return (dataset, sample_df)."""
    subject_split = pd.read_csv(SUBJECT_SPLIT_PATH)
    subjects = set(
        subject_split.loc[subject_split["split"] == split, "subject_id"].astype(str)
    )
    outcomes = pd.read_csv(OUTCOME_DF_PATH)
    if "subject_id" not in outcomes.columns:
        outcomes["subject_id"] = outcomes["file_path"].map(
            lambda p: zarr_record_name(p).split("-", 1)[0]
        )
    outcomes = outcomes.loc[outcomes["subject_id"].astype(str).isin(subjects)].copy()
    outcomes["Time Stamp (seconds)"] = outcomes["Time Stamp (seconds)"].round()

    cache_path = SAMPLE_CACHE_DIR / f"{DATASET_FILENAME}-{split}_samples.csv.gz"
    samples = pd.read_csv(cache_path)
    if "subject_id" not in samples.columns:
        samples["subject_id"] = samples["file_path"].map(
            lambda p: zarr_record_name(p).split("-", 1)[0]
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
        sample_df=samples,
        sample_seq_len_sec=SAMPLE_SEQ_LEN_SECONDS,
        frequency=FREQUENCY,
        butterworth_filters=None,
        median_filter_kernel_size=None,
        clip_interpolations=CLIP_INTERPOLATE_RANGES,
        constant_nan_tolerance=0.2,
        require_all_channels=True,
        infer_forecast_windows=False,
        normalize_signals=True,
    )
    return dataset, samples


@torch.no_grad()
def extract_embeddings(model, dataloader, device) -> np.ndarray:
    """Extract mean-pooled embeddings."""
    model.eval()
    all_emb = []
    for i, (batch_x, _) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out = model(batch_x)
        if isinstance(out, tuple):
            out = out[0]
        # (bs, n_ch, d_model, n_patches) → (bs, d_model)
        pooled = out.mean(dim=(1, 3))
        all_emb.append(pooled.float().cpu().numpy())
        if (i + 1) % 25 == 0:
            print(f"    batch {i+1}/{len(dataloader)}", flush=True)
    return np.concatenate(all_emb, axis=0)


# ── Cross-reference with icuDataExtraction ────────────────────────────────────

def load_icu_clusters() -> dict:
    """Load per-patient dominant hemodynamic cluster from icuDataExtraction."""
    cluster_labels = np.load(ICU_OUTPUT / "cluster_labels.npy")
    patient_ids = np.load(ICU_OUTPUT / "patient_ids.npy", allow_pickle=True)

    # Compute dominant cluster per patient (mode)
    patient_cluster = {}
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        clusters = cluster_labels[mask]
        # Mode
        counts = np.bincount(clusters, minlength=7)
        patient_cluster[str(pid)] = int(np.argmax(counts))

    return patient_cluster


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    icu_patient_clusters: dict,
    n_clusters: int,
    model_name: str,
) -> dict:
    """Run KMeans clustering and analyze relationships."""
    print(f"\n{'='*60}")
    print(f"  {model_name} — Clustering Analysis")
    print(f"{'='*60}")

    # KMeans
    print(f"  Running KMeans (k={n_clusters})...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_assignments = km.fit_predict(embeddings)

    # Silhouette score (on subsample for speed)
    n_sil = min(5000, len(embeddings))
    rng = np.random.default_rng(42)
    sil_idx = rng.choice(len(embeddings), size=n_sil, replace=False)
    sil = silhouette_score(embeddings[sil_idx], cluster_assignments[sil_idx])
    print(f"  Silhouette score: {sil:.3f}")

    # ── Relationship to hypotension labels ────────────────────────────────
    print(f"\n  Cluster vs Hypotension Label:")
    print(f"  {'Cluster':<8} {'Size':>6} {'Pos%':>6} {'Neg%':>6}")
    print(f"  {'-'*30}")
    for c in range(n_clusters):
        mask = cluster_assignments == c
        size = mask.sum()
        pos_rate = labels[mask].mean() * 100
        print(f"  {c:<8} {size:>6} {pos_rate:>5.1f}% {100-pos_rate:>5.1f}%")

    # ARI and NMI with hypotension labels
    ari_hypo = adjusted_rand_score(labels, cluster_assignments)
    nmi_hypo = normalized_mutual_info_score(labels, cluster_assignments)
    print(f"\n  ARI with hypotension labels: {ari_hypo:.4f}")
    print(f"  NMI with hypotension labels: {nmi_hypo:.4f}")

    # ── Relationship to icuDataExtraction hemodynamic clusters ─────────────
    # Map each embedding window to its patient's dominant hemodynamic cluster
    hemo_labels = np.full(len(patient_ids), -1, dtype=int)
    n_mapped = 0
    for i, pid in enumerate(patient_ids):
        pid_str = str(pid)
        if pid_str in icu_patient_clusters:
            hemo_labels[i] = icu_patient_clusters[pid_str]
            n_mapped += 1

    valid_hemo = hemo_labels >= 0
    print(f"\n  Mapped to hemodynamic clusters: {n_mapped}/{len(patient_ids)} "
          f"({100*n_mapped/len(patient_ids):.1f}%)")

    if valid_hemo.sum() > 100:
        ari_hemo = adjusted_rand_score(
            hemo_labels[valid_hemo], cluster_assignments[valid_hemo]
        )
        nmi_hemo = normalized_mutual_info_score(
            hemo_labels[valid_hemo], cluster_assignments[valid_hemo]
        )
        print(f"  ARI with hemodynamic clusters: {ari_hemo:.4f}")
        print(f"  NMI with hemodynamic clusters: {nmi_hemo:.4f}")

        # Contingency table
        print(f"\n  Contingency: Embedding Cluster (rows) × Hemodynamic Cluster (cols)")
        ct = np.zeros((n_clusters, 7), dtype=int)
        for i in range(len(cluster_assignments)):
            if valid_hemo[i]:
                ct[cluster_assignments[i], hemo_labels[i]] += 1
        # Normalize rows to percentages
        print(f"  {'':>4}", end="")
        for h in range(7):
            print(f" H{h:>3}", end="")
        print(f" {'Total':>6}")
        for c in range(n_clusters):
            row_total = ct[c].sum()
            print(f"  C{c}: ", end="")
            for h in range(7):
                pct = 100 * ct[c, h] / max(row_total, 1)
                print(f" {pct:>4.0f}", end="")
            print(f" {row_total:>6}")
    else:
        ari_hemo = np.nan
        nmi_hemo = np.nan

    return {
        "model": model_name,
        "silhouette": sil,
        "ari_hypotension": ari_hypo,
        "nmi_hypotension": nmi_hypo,
        "ari_hemodynamic": ari_hemo,
        "nmi_hemodynamic": nmi_hemo,
        "cluster_assignments": cluster_assignments,
        "hemo_labels": hemo_labels,
    }


def plot_umap(embeddings, cluster_assignments, labels, hemo_labels,
              model_name, output_dir):
    """UMAP visualization colored by different label sources."""
    print(f"  Computing UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3,
                        metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Colored by KMeans cluster
    scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=cluster_assignments,
                              cmap="tab10", s=3, alpha=0.5, rasterized=True)
    axes[0].set_title(f"KMeans Clusters (k={cluster_assignments.max()+1})")
    plt.colorbar(scatter, ax=axes[0])

    # 2. Colored by hypotension label
    colors = ["steelblue" if l == 0 else "crimson" for l in labels]
    axes[1].scatter(coords[:, 0], coords[:, 1], c=colors, s=3, alpha=0.5,
                    rasterized=True)
    axes[1].set_title("Hypotension Label (red=positive)")

    # 3. Colored by hemodynamic cluster (where available)
    valid = hemo_labels >= 0
    if valid.sum() > 0:
        scatter3 = axes[2].scatter(coords[valid, 0], coords[valid, 1],
                                   c=hemo_labels[valid], cmap="Set1", s=3,
                                   alpha=0.5, rasterized=True)
        axes[2].set_title(f"Hemodynamic Clusters ({valid.sum()} mapped)")
        plt.colorbar(scatter3, ax=axes[2])
    else:
        axes[2].set_title("Hemodynamic Clusters (no mapping)")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{model_name} — Embedding Space", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / f"clustering_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return coords


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    n_label = args.n_samples if args.n_samples > 0 else "full"
    cache_path = OUTPUT_DIR / f"embeddings_n{n_label}_seed{args.seed}.npz"

    if args.skip_extraction and cache_path.is_file():
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        jepa_emb = data["jepa_embeddings"]
        ptst_emb = data["ptst_embeddings"]
        labels = data["labels"]
        patient_ids = data["patient_ids"]
    else:
        # Load test dataset
        print("Loading test dataset...")
        dataset, sample_df = load_dataset("test")
        label_col = f"outcome_val_{FORECAST_WINDOW_SEC[0]}sec"

        # Random subset (not balanced — want natural distribution for clustering)
        rng = np.random.default_rng(args.seed)
        n = len(dataset) if args.n_samples == 0 else min(args.n_samples, len(dataset))
        if n < len(dataset):
            indices = rng.choice(len(dataset), size=n, replace=False).tolist()
        else:
            indices = list(range(len(dataset)))

        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

        # Get metadata for selected samples
        selected_df = sample_df.iloc[indices]
        labels = selected_df[label_col].values.astype(int)
        patient_ids = selected_df["subject_id"].values

        print(f"  N={n}, pos_rate={labels.mean():.3f}, "
              f"patients={len(np.unique(patient_ids))}")

        # Extract JEPA embeddings
        print("\nLoading JEPA encoder...")
        jepa_model = JEPASimpleLightning.load_from_checkpoint(
            str(JEPA_CKPT), map_location="cpu")
        jepa_model.eval()
        jepa_model.pretrain = False
        if hasattr(jepa_model, "model"):
            jepa_model.model.pretrain = False
        jepa_model = jepa_model.to(device)

        print("Extracting JEPA embeddings...")
        jepa_emb = extract_embeddings(jepa_model, loader, device)
        print(f"  Shape: {jepa_emb.shape}")
        del jepa_model
        torch.cuda.empty_cache()

        # Extract PatchTST embeddings
        print("\nLoading PatchTST encoder...")
        ptst_model = PatchTFTSimpleLightning.load_from_checkpoint(
            str(PTST_CKPT), map_location="cpu")
        ptst_model.eval()
        ptst_model.pretrain = False
        if hasattr(ptst_model, "model"):
            ptst_model.model.pretrain = False
        ptst_model = ptst_model.to(device)

        print("Extracting PatchTST embeddings...")
        ptst_emb = extract_embeddings(ptst_model, loader, device)
        print(f"  Shape: {ptst_emb.shape}")
        del ptst_model
        torch.cuda.empty_cache()

        # Cache
        np.savez_compressed(cache_path,
                            jepa_embeddings=jepa_emb,
                            ptst_embeddings=ptst_emb,
                            labels=labels,
                            patient_ids=patient_ids)
        print(f"\nCached to {cache_path}")

    print(f"\nDataset: {len(labels)} windows, {len(np.unique(patient_ids))} patients, "
          f"pos_rate={labels.mean():.3f}")

    # Load icuDataExtraction hemodynamic clusters
    print("\nLoading hemodynamic clusters from icuDataExtraction...")
    icu_patient_clusters = load_icu_clusters()
    print(f"  {len(icu_patient_clusters)} patients with cluster assignments")

    # ── Analysis ──────────────────────────────────────────────────────────────
    jepa_results = analyze_clusters(
        jepa_emb, labels, patient_ids, icu_patient_clusters,
        args.n_clusters, "JEPA"
    )
    ptst_results = analyze_clusters(
        ptst_emb, labels, patient_ids, icu_patient_clusters,
        args.n_clusters, "PatchTST"
    )

    # ── UMAP Plots ────────────────────────────────────────────────────────────
    print("\n\n--- UMAP Visualizations ---")
    plot_umap(jepa_emb, jepa_results["cluster_assignments"], labels,
              jepa_results["hemo_labels"], "JEPA", OUTPUT_DIR)
    plot_umap(ptst_emb, ptst_results["cluster_assignments"], labels,
              ptst_results["hemo_labels"], "PatchTST", OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'JEPA':>10} {'PatchTST':>10}")
    print("-" * 60)
    print(f"{'Silhouette score':<30} {jepa_results['silhouette']:>10.3f} "
          f"{ptst_results['silhouette']:>10.3f}")
    print(f"{'ARI vs hypotension':<30} {jepa_results['ari_hypotension']:>10.4f} "
          f"{ptst_results['ari_hypotension']:>10.4f}")
    print(f"{'NMI vs hypotension':<30} {jepa_results['nmi_hypotension']:>10.4f} "
          f"{ptst_results['nmi_hypotension']:>10.4f}")
    print(f"{'ARI vs hemodynamic clusters':<30} {jepa_results['ari_hemodynamic']:>10.4f} "
          f"{ptst_results['ari_hemodynamic']:>10.4f}")
    print(f"{'NMI vs hemodynamic clusters':<30} {jepa_results['nmi_hemodynamic']:>10.4f} "
          f"{ptst_results['nmi_hemodynamic']:>10.4f}")
    print("=" * 60)

    # Save summary
    summary = pd.DataFrame([
        {"model": "JEPA", **{k: v for k, v in jepa_results.items()
                             if k not in ["cluster_assignments", "hemo_labels"]}},
        {"model": "PatchTST", **{k: v for k, v in ptst_results.items()
                                  if k not in ["cluster_assignments", "hemo_labels"]}},
    ])
    summary.to_csv(OUTPUT_DIR / "clustering_summary.csv", index=False)
    print(f"\nSaved summary to {OUTPUT_DIR / 'clustering_summary.csv'}")


if __name__ == "__main__":
    main()
