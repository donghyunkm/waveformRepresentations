"""
Per-patient temporal clustering of JEPA embeddings.

For selected patients, clusters their individual windows to discover temporal
sub-states, then analyzes how those sub-clusters relate to:
  1. Hypotension labels (do sub-clusters separate pre-hypo from normal?)
  2. Hemodynamic phenotypes (do sub-clusters align with hemo cluster transitions?)
  3. Temporal structure (are sub-clusters temporally contiguous?)

Produces per-patient UMAP visualizations colored by sub-cluster, time, label,
and hemodynamic cluster.

Requires cached embeddings from cluster_embeddings.py.

Usage:
    python cluster_single_patient.py [--top-n 10] [--model jepa]
    python cluster_single_patient.py --patient-ids p052529,p011342
    python cluster_single_patient.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

# ── Paths ─────────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
CLUSTERING_DIR = DERIVED_ROOT / "probing/clustering"
SAMPLE_CACHE_DIR = DERIVED_ROOT / "models/fcn_hypotension_paper"
DATASET_FILENAME = "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"
HEMO_CLUSTERS_PATH = CLUSTERING_DIR / "window_hemo_clusters.npz"
EMBEDDINGS_PATH = CLUSTERING_DIR / "embeddings_nfull_seed42.npz"

OUTPUT_DIR = CLUSTERING_DIR / "per_patient"

FREQUENCY = 125  # Hz


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-patient temporal clustering of embeddings."
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Analyze the top-N patients with most windows and both labels "
             "(ignored if --patient-ids is set)"
    )
    parser.add_argument(
        "--patient-ids", type=str, default=None,
        help="Comma-separated patient IDs to analyze (e.g. p052529,p011342)"
    )
    parser.add_argument(
        "--model", type=str, default="jepa", choices=["jepa", "patchtst"],
        help="Which encoder's embeddings to cluster"
    )
    parser.add_argument(
        "--n-clusters", type=int, default=0,
        help="Fixed number of KMeans clusters (0 = auto-select via silhouette sweep)"
    )
    parser.add_argument(
        "--max-k", type=int, default=10,
        help="Maximum k for silhouette sweep when auto-selecting"
    )
    parser.add_argument(
        "--min-windows", type=int, default=100,
        help="Minimum windows required to include a patient"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--no-umap", action="store_true",
        help="Skip UMAP visualization (faster)"
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def select_k_silhouette(embeddings: np.ndarray, max_k: int, seed: int) -> int:
    """Select k by maximizing silhouette score over k=2..max_k."""
    best_k, best_sil = 2, -1.0
    scores = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labs = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labs)
        scores.append((k, sil))
        if sil > best_sil:
            best_sil = sil
            best_k = k
    return best_k, scores


def temporal_contiguity_score(cluster_labels: np.ndarray) -> float:
    """Fraction of adjacent window pairs that share the same cluster label.
    1.0 = perfectly contiguous blocks; lower = more fragmented."""
    if len(cluster_labels) < 2:
        return 1.0
    same = (cluster_labels[:-1] == cluster_labels[1:]).sum()
    return same / (len(cluster_labels) - 1)


def cluster_label_entropy(cluster_labels: np.ndarray, hypo_labels: np.ndarray,
                          n_clusters: int) -> pd.DataFrame:
    """For each cluster, compute positive rate and size."""
    rows = []
    for c in range(n_clusters):
        mask = cluster_labels == c
        size = mask.sum()
        if size == 0:
            continue
        pos_rate = hypo_labels[mask].mean()
        rows.append({
            "cluster": c,
            "size": int(size),
            "pos_rate": pos_rate,
            "neg_rate": 1 - pos_rate,
        })
    return pd.DataFrame(rows)


def analyze_single_patient(
    patient_id: str,
    embeddings: np.ndarray,
    hypo_labels: np.ndarray,
    hemo_labels: np.ndarray,
    start_indices: np.ndarray,
    n_clusters: int,
    max_k: int,
    seed: int,
    do_umap: bool,
    output_dir: Path,
    model_name: str,
) -> dict:
    """Run clustering + analysis for a single patient."""
    n_windows = len(embeddings)
    n_pos = int(hypo_labels.sum())
    n_neg = n_windows - n_pos
    n_hemo_valid = int((hemo_labels >= 0).sum())

    print(f"\n{'─'*60}")
    print(f"  Patient: {patient_id} | {n_windows} windows | "
          f"pos={n_pos} neg={n_neg} | hemo_mapped={n_hemo_valid}")
    print(f"{'─'*60}")

    # Sort by temporal order (start_idx within file)
    time_order = np.argsort(start_indices)
    embeddings = embeddings[time_order]
    hypo_labels = hypo_labels[time_order]
    hemo_labels = hemo_labels[time_order]
    start_indices = start_indices[time_order]

    # Compute temporal position (hours from first window)
    time_hours = (start_indices - start_indices[0]) / (FREQUENCY * 3600)

    # ── Determine k ───────────────────────────────────────────────────────
    if n_clusters > 0:
        k = n_clusters
        k_scores = None
    else:
        k, k_scores = select_k_silhouette(embeddings, min(max_k, n_windows // 10), seed)
        print(f"  Auto-selected k={k} (silhouette sweep: "
              f"{', '.join(f'k={s[0]}:{s[1]:.3f}' for s in k_scores)})")

    # ── KMeans ────────────────────────────────────────────────────────────
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(embeddings)
    sil = silhouette_score(embeddings, cluster_labels) if k >= 2 else 0.0
    print(f"  KMeans k={k}, silhouette={sil:.3f}")

    # ── Temporal contiguity ───────────────────────────────────────────────
    contiguity = temporal_contiguity_score(cluster_labels)
    # Random baseline: expected contiguity for k clusters of these sizes
    cluster_sizes = np.bincount(cluster_labels, minlength=k)
    expected_contiguity = (cluster_sizes ** 2).sum() / (n_windows ** 2)
    print(f"  Temporal contiguity: {contiguity:.3f} (random baseline: {expected_contiguity:.3f})")

    # ── Relationship to hypotension labels ────────────────────────────────
    if n_pos > 0 and n_neg > 0:
        ari_hypo = adjusted_rand_score(hypo_labels, cluster_labels)
        nmi_hypo = normalized_mutual_info_score(hypo_labels, cluster_labels)
        label_df = cluster_label_entropy(cluster_labels, hypo_labels, k)
        print(f"  ARI vs hypotension: {ari_hypo:.4f}")
        print(f"  NMI vs hypotension: {nmi_hypo:.4f}")
        print(f"  Per-cluster positive rates:")
        for _, row in label_df.iterrows():
            bar = "█" * int(row["pos_rate"] * 40)
            print(f"    C{int(row['cluster'])}: {row['size']:>5} windows, "
                  f"pos_rate={row['pos_rate']:.3f} {bar}")
    else:
        ari_hypo = np.nan
        nmi_hypo = np.nan
        label_df = pd.DataFrame()

    # ── Relationship to hemodynamic clusters ──────────────────────────────
    valid_hemo = hemo_labels >= 0
    if valid_hemo.sum() > 10:
        ari_hemo = adjusted_rand_score(hemo_labels[valid_hemo], cluster_labels[valid_hemo])
        nmi_hemo = normalized_mutual_info_score(hemo_labels[valid_hemo], cluster_labels[valid_hemo])
        n_hemo_types = len(np.unique(hemo_labels[valid_hemo]))
        print(f"  ARI vs hemodynamic ({n_hemo_types} phenotypes): {ari_hemo:.4f}")
        print(f"  NMI vs hemodynamic: {nmi_hemo:.4f}")

        # Cross-tabulation
        hemo_unique = np.unique(hemo_labels[valid_hemo])
        print(f"  Cluster × Hemodynamic contingency:")
        header = "      " + "".join(f"  H{h}" for h in hemo_unique) + "  Total"
        print(header)
        for c in range(k):
            c_mask = (cluster_labels == c) & valid_hemo
            row_vals = []
            for h in hemo_unique:
                row_vals.append(((hemo_labels == h) & c_mask).sum())
            total = sum(row_vals)
            row_str = f"   C{c}: " + "".join(f"{v:>4}" for v in row_vals) + f"  {total:>5}"
            print(row_str)
    else:
        ari_hemo = np.nan
        nmi_hemo = np.nan

    # ── HDBSCAN (if available) for comparison ─────────────────────────────
    hdbscan_n_clusters = np.nan
    if HAS_HDBSCAN and n_windows >= 50:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(10, n_windows // 20),
                                     metric="euclidean")
        hdb_labels = clusterer.fit_predict(embeddings)
        hdbscan_n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        hdb_noise = (hdb_labels == -1).sum()
        print(f"  HDBSCAN: {hdbscan_n_clusters} clusters, "
              f"{hdb_noise} noise points ({100*hdb_noise/n_windows:.1f}%)")

    # ── UMAP Visualization ────────────────────────────────────────────────
    umap_coords = None
    if do_umap and n_windows >= 30:
        n_neighbors = min(15, n_windows - 1)
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                            min_dist=0.1, metric="cosine", random_state=seed)
        umap_coords = reducer.fit_transform(embeddings)

        n_panels = 3 + (1 if valid_hemo.sum() > 10 else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))

        # Panel 1: colored by sub-cluster
        sc1 = axes[0].scatter(umap_coords[:, 0], umap_coords[:, 1],
                              c=cluster_labels, cmap="tab10", s=8, alpha=0.7,
                              rasterized=True)
        axes[0].set_title(f"Sub-clusters (k={k}, sil={sil:.2f})")
        plt.colorbar(sc1, ax=axes[0])

        # Panel 2: colored by time (hours from start)
        sc2 = axes[1].scatter(umap_coords[:, 0], umap_coords[:, 1],
                              c=time_hours, cmap="viridis", s=8, alpha=0.7,
                              rasterized=True)
        axes[1].set_title("Time (hours from start)")
        plt.colorbar(sc2, ax=axes[1], label="hours")

        # Panel 3: colored by hypotension label
        colors_hypo = np.where(hypo_labels == 1, "crimson", "steelblue")
        axes[2].scatter(umap_coords[:, 0], umap_coords[:, 1],
                        c=colors_hypo, s=8, alpha=0.7, rasterized=True)
        axes[2].set_title(f"Hypotension (red=pos, n={n_pos})")

        # Panel 4: hemodynamic clusters (if available)
        if valid_hemo.sum() > 10:
            # Plot unmapped as light gray, mapped colored by hemo cluster
            axes[3].scatter(umap_coords[~valid_hemo, 0], umap_coords[~valid_hemo, 1],
                            c="lightgray", s=4, alpha=0.3, rasterized=True,
                            label="unmapped")
            sc4 = axes[3].scatter(umap_coords[valid_hemo, 0], umap_coords[valid_hemo, 1],
                                  c=hemo_labels[valid_hemo], cmap="Set1", s=12,
                                  alpha=0.8, rasterized=True)
            axes[3].set_title(f"Hemo clusters ({valid_hemo.sum()} mapped)")
            plt.colorbar(sc4, ax=axes[3])

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"{model_name.upper()} — Patient {patient_id} "
                     f"({n_windows} windows, {time_hours.max():.1f}h span)",
                     fontsize=11)
        fig.tight_layout()
        out_path = output_dir / f"{patient_id}_{model_name}_subclusters.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # ── Temporal cluster transition diagram ───────────────────────────────
    # Show the sequence of clusters over time (compressed)
    transitions = []
    current = cluster_labels[0]
    count = 1
    for i in range(1, len(cluster_labels)):
        if cluster_labels[i] == current:
            count += 1
        else:
            transitions.append((current, count))
            current = cluster_labels[i]
            count = 1
    transitions.append((current, count))

    if len(transitions) <= 30:
        seq_str = " → ".join(f"C{c}({n})" for c, n in transitions)
    else:
        seq_str = (" → ".join(f"C{c}({n})" for c, n in transitions[:15])
                   + f" ... ({len(transitions)-30} more) ... "
                   + " → ".join(f"C{c}({n})" for c, n in transitions[-15:]))
    print(f"  Temporal sequence ({len(transitions)} segments):")
    print(f"    {seq_str}")

    return {
        "patient_id": patient_id,
        "n_windows": n_windows,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_hemo_mapped": int(valid_hemo.sum()),
        "time_span_hours": float(time_hours.max()),
        "k": k,
        "silhouette": sil,
        "temporal_contiguity": contiguity,
        "expected_contiguity": expected_contiguity,
        "contiguity_ratio": contiguity / max(expected_contiguity, 1e-8),
        "ari_hypotension": ari_hypo,
        "nmi_hypotension": nmi_hypo,
        "ari_hemodynamic": ari_hemo,
        "nmi_hemodynamic": nmi_hemo,
        "hdbscan_n_clusters": hdbscan_n_clusters,
        "n_temporal_segments": len(transitions),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load cached embeddings
    print(f"Loading embeddings from {EMBEDDINGS_PATH}")
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    if args.model == "jepa":
        all_embeddings = data["jepa_embeddings"]
    else:
        all_embeddings = data["ptst_embeddings"]
    all_labels = data["labels"]
    all_patient_ids = data["patient_ids"]
    print(f"  Total: {len(all_labels)} windows, {len(np.unique(all_patient_ids))} patients")

    # Load hemodynamic cluster mapping (window-level)
    if HEMO_CLUSTERS_PATH.is_file():
        hemo_data = np.load(HEMO_CLUSTERS_PATH, allow_pickle=True)
        all_hemo_labels = hemo_data["hemo_clusters"]
        print(f"  Hemo clusters loaded: {(all_hemo_labels >= 0).sum()} mapped windows")
    else:
        all_hemo_labels = np.full(len(all_labels), -1, dtype=int)
        print("  No hemo cluster mapping found")

    # Load sample cache for temporal ordering (start_idx)
    cache_path = SAMPLE_CACHE_DIR / f"{DATASET_FILENAME}-test_samples.csv.gz"
    sample_df = pd.read_csv(cache_path)
    all_start_idx = sample_df["start_idx"].values
    print(f"  Sample cache loaded: {len(sample_df)} rows")

    assert len(all_start_idx) == len(all_labels), (
        f"Sample cache ({len(all_start_idx)}) != embeddings ({len(all_labels)})"
    )

    # ── Select patients ───────────────────────────────────────────────────
    if args.patient_ids:
        target_pids = [p.strip() for p in args.patient_ids.split(",")]
    else:
        # Auto-select: patients with most windows + both labels
        pid_stats = []
        for pid in np.unique(all_patient_ids):
            mask = all_patient_ids == pid
            n = mask.sum()
            if n < args.min_windows:
                continue
            n_pos = all_labels[mask].sum()
            n_neg = n - n_pos
            pid_stats.append((pid, n, n_pos, n_neg))

        # Prioritize patients with both labels, then by window count
        pid_stats.sort(key=lambda x: (x[2] > 0 and x[3] > 0, x[1]), reverse=True)
        target_pids = [s[0] for s in pid_stats[:args.top_n]]

    print(f"\nAnalyzing {len(target_pids)} patients: {target_pids}")

    # ── Run per-patient analysis ──────────────────────────────────────────
    results = []
    for pid in target_pids:
        mask = all_patient_ids == pid
        if mask.sum() < 10:
            print(f"\n  Skipping {pid}: only {mask.sum()} windows")
            continue

        emb = all_embeddings[mask]
        labs = all_labels[mask]
        hemo = all_hemo_labels[mask]
        start_idx = all_start_idx[mask]

        result = analyze_single_patient(
            patient_id=pid,
            embeddings=emb,
            hypo_labels=labs,
            hemo_labels=hemo,
            start_indices=start_idx,
            n_clusters=args.n_clusters,
            max_k=args.max_k,
            seed=args.seed,
            do_umap=not args.no_umap,
            output_dir=OUTPUT_DIR,
            model_name=args.model,
        )
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────
    if not results:
        print("\nNo patients analyzed.")
        return

    df = pd.DataFrame(results)
    summary_path = OUTPUT_DIR / f"per_patient_clustering_{args.model}_summary.csv"
    df.to_csv(summary_path, index=False)

    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — {args.model.upper()} Per-Patient Clustering ({len(results)} patients)")
    print(f"{'='*70}")
    print(f"{'Patient':<10} {'N':>5} {'k':>3} {'Sil':>5} {'Contig':>6} "
          f"{'C/Rand':>6} {'ARI_hyp':>8} {'ARI_hem':>8} {'Span(h)':>7}")
    print("-" * 70)
    for r in results:
        ari_hem_str = "     N/A" if np.isnan(r["ari_hemodynamic"]) else f"{r['ari_hemodynamic']:>8.3f}"
        print(f"{r['patient_id']:<10} {r['n_windows']:>5} {r['k']:>3} "
              f"{r['silhouette']:>5.2f} {r['temporal_contiguity']:>6.3f} "
              f"{r['contiguity_ratio']:>6.1f} "
              f"{r['ari_hypotension']:>8.3f} "
              f"{ari_hem_str} "
              f"{r['time_span_hours']:>7.1f}")

    print(f"\n  Mean silhouette:          {df['silhouette'].mean():.3f}")
    print(f"  Mean contiguity ratio:    {df['contiguity_ratio'].mean():.1f}× random")
    valid_hypo = df["ari_hypotension"].dropna()
    if len(valid_hypo) > 0:
        print(f"  Mean ARI vs hypotension:  {valid_hypo.mean():.4f}")
    valid_hemo = df["ari_hemodynamic"].dropna()
    if len(valid_hemo) > 0:
        print(f"  Mean ARI vs hemodynamic:  {valid_hemo.mean():.4f}")
    print(f"\n  Saved: {summary_path}")
    print(f"  Plots: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
