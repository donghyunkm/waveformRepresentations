"""
Centered Kernel Alignment (CKA) analysis of JEPA/PatchTST embeddings.

CKA measures representational similarity — whether two sets of embeddings
organize their samples with the same internal geometry, regardless of
rotation or scaling. This is more nuanced than raw distance comparisons
because it captures *structural* similarity rather than absolute proximity.

Key questions:
1. Do same-phenotype patients share representational geometry?
   (CKA between embedding matrices of patients within the same hemo cluster)
2. Is the within-patient geometry shared across patients?
   (CKA between per-patient embedding matrices)
3. How similar are JEPA vs PatchTST representations per group?

Requires cached embeddings from cluster_embeddings.py (--skip-extraction).

Usage:
    python cka_analysis.py [--n-windows-per-patient 50] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from itertools import combinations

# ── Paths ─────────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
CLUSTERING_DIR = DERIVED_ROOT / "probing/clustering"
ICU_OUTPUT = Path("/gpfs/home/dk5565/icuDataExtraction/output_v2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-windows-per-patient", type=int, default=50,
                        help="Max windows per patient for CKA (subsample large patients)")
    parser.add_argument("--min-windows", type=int, default=20,
                        help="Minimum windows per patient to include in analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings-path", type=str,
                        default=str(CLUSTERING_DIR / "embeddings_nfull_seed42.npz"))
    return parser.parse_args()


# ── CKA Implementation ───────────────────────────────────────────────────────

def centering_matrix(n: int) -> np.ndarray:
    """Centering matrix H = I - 1/n * 11^T."""
    return np.eye(n) - np.ones((n, n)) / n


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """Linear kernel: K = X @ X.T"""
    return X @ X.T


def rbf_kernel(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """RBF kernel with median heuristic for sigma."""
    sq_dists = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    if sigma is None:
        # Median heuristic
        sigma = np.sqrt(np.median(sq_dists[np.triu_indices_from(sq_dists, k=1)]))
        if sigma == 0:
            sigma = 1.0
    return np.exp(-sq_dists / (2 * sigma ** 2))


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion (biased estimator).
    HSIC = 1/n^2 * tr(KHLH) where H is the centering matrix.
    """
    n = K.shape[0]
    H = centering_matrix(n)
    KH = K @ H
    LH = L @ H
    return np.trace(KH @ LH) / (n ** 2)


def cka(X: np.ndarray, Y: np.ndarray, kernel: str = "linear") -> float:
    """Compute CKA between two representation matrices.

    Args:
        X: (n_samples, d1) — first representation
        Y: (n_samples, d2) — second representation
        kernel: "linear" or "rbf"

    Returns:
        CKA score in [0, 1]. 1 = identical geometry, 0 = orthogonal.
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
    n = X.shape[0]
    if n < 3:
        return np.nan

    if kernel == "linear":
        K = linear_kernel(X)
        L = linear_kernel(Y)
    elif kernel == "rbf":
        K = rbf_kernel(X)
        L = rbf_kernel(Y)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    hsic_kl = hsic(K, L)
    hsic_kk = hsic(K, K)
    hsic_ll = hsic(L, L)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return np.nan
    return hsic_kl / denom


def cka_self(X: np.ndarray, Y: np.ndarray, kernel: str = "linear") -> float:
    """CKA between two sets of embeddings of the SAME samples.
    Measures whether two models organize the same inputs similarly."""
    return cka(X, Y, kernel=kernel)


def cka_cross(X: np.ndarray, Y: np.ndarray, n_common: int,
              kernel: str = "linear", rng: np.random.Generator | None = None) -> float:
    """CKA between two DIFFERENT sets of samples that need alignment.

    Subsamples n_common samples from each, computes CKA on aligned subsets.
    This measures whether two groups have similar internal geometry.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nx, ny = X.shape[0], Y.shape[0]
    if nx < n_common or ny < n_common:
        n_common = min(nx, ny)
    if n_common < 5:
        return np.nan

    idx_x = rng.choice(nx, size=n_common, replace=False)
    idx_y = rng.choice(ny, size=n_common, replace=False)

    X_sub = X[idx_x]
    Y_sub = Y[idx_y]

    # For cross-group CKA, we compare the internal geometry of each group
    # by computing their kernel matrices and checking structural similarity
    K_x = linear_kernel(X_sub) if kernel == "linear" else rbf_kernel(X_sub)
    K_y = linear_kernel(Y_sub) if kernel == "linear" else rbf_kernel(Y_sub)

    # Center both
    n = n_common
    H = centering_matrix(n)
    K_xc = H @ K_x @ H
    K_yc = H @ K_y @ H

    # Frobenius inner product of centered kernels
    num = np.sum(K_xc * K_yc)
    denom = np.sqrt(np.sum(K_xc * K_xc) * np.sum(K_yc * K_yc))
    if denom < 1e-10:
        return np.nan
    return num / denom


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_hemodynamic_clusters() -> dict:
    """Load per-patient dominant hemodynamic cluster from icuDataExtraction."""
    cluster_labels = np.load(ICU_OUTPUT / "cluster_labels.npy")
    patient_ids = np.load(ICU_OUTPUT / "patient_ids.npy", allow_pickle=True)

    patient_cluster = {}
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        counts = np.bincount(cluster_labels[mask], minlength=7)
        patient_cluster[str(pid)] = int(np.argmax(counts))

    return patient_cluster


def get_patient_embeddings(embeddings, patient_ids, pid, max_windows, rng):
    """Get embeddings for a single patient, subsampled if needed."""
    mask = patient_ids == pid
    emb = embeddings[mask]
    if len(emb) > max_windows:
        idx = rng.choice(len(emb), size=max_windows, replace=False)
        emb = emb[idx]
    return emb


# ── Analysis Functions ────────────────────────────────────────────────────────

def analysis_1_phenotype_geometry(embeddings, patient_ids, patient_cluster,
                                  max_windows, min_windows, rng, model_name):
    """Do same-phenotype patients share representational geometry?

    For each pair of patients, compute CKA between their embedding matrices.
    Compare CKA scores for same-phenotype pairs vs different-phenotype pairs.

    If same-phenotype pairs have higher CKA, the encoder organizes windows
    similarly for patients in the same physiological state — even though
    the raw centroids are far apart.
    """
    print(f"\n{'─' * 70}")
    print(f"Analysis 1: Cross-patient representational similarity ({model_name})")
    print(f"{'─' * 70}")

    # Get patients with enough windows AND hemodynamic mapping
    unique_pids = np.unique(patient_ids)
    eligible_patients = []
    for pid in unique_pids:
        n_windows = (patient_ids == pid).sum()
        if n_windows >= min_windows and str(pid) in patient_cluster:
            eligible_patients.append(pid)

    print(f"  Eligible patients (>={min_windows} windows + hemo mapping): "
          f"{len(eligible_patients)}")

    if len(eligible_patients) < 10:
        print("  Too few patients for meaningful analysis. Skipping.")
        return []

    # Compute CKA between all patient pairs (subsample for speed)
    max_pairs = 500
    all_pairs = list(combinations(range(len(eligible_patients)), 2))
    if len(all_pairs) > max_pairs:
        pair_idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        selected_pairs = [all_pairs[i] for i in pair_idx]
    else:
        selected_pairs = all_pairs

    print(f"  Computing CKA for {len(selected_pairs)} patient pairs...")

    results = []
    for i, (idx_a, idx_b) in enumerate(selected_pairs):
        pid_a = eligible_patients[idx_a]
        pid_b = eligible_patients[idx_b]

        emb_a = get_patient_embeddings(embeddings, patient_ids, pid_a,
                                       max_windows, rng)
        emb_b = get_patient_embeddings(embeddings, patient_ids, pid_b,
                                       max_windows, rng)

        # Align to same number of samples for CKA
        n_common = min(len(emb_a), len(emb_b))
        if n_common < 10:
            continue

        emb_a_sub = emb_a[:n_common]
        emb_b_sub = emb_b[:n_common]

        # CKA: compare internal geometry
        cka_score = cka_cross(emb_a, emb_b, n_common=n_common,
                              kernel="linear", rng=rng)

        cluster_a = patient_cluster[str(pid_a)]
        cluster_b = patient_cluster[str(pid_b)]
        same_pheno = cluster_a == cluster_b

        results.append({
            "model": model_name,
            "patient_a": str(pid_a),
            "patient_b": str(pid_b),
            "cluster_a": cluster_a,
            "cluster_b": cluster_b,
            "same_phenotype": same_pheno,
            "cka_linear": cka_score,
        })

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(selected_pairs)} pairs computed")

    df = pd.DataFrame(results)
    same_cka = df[df["same_phenotype"]]["cka_linear"].mean()
    diff_cka = df[~df["same_phenotype"]]["cka_linear"].mean()
    n_same = df["same_phenotype"].sum()
    n_diff = (~df["same_phenotype"]).sum()

    print(f"\n  Results ({model_name}):")
    print(f"    Same phenotype CKA:  {same_cka:.4f} (n={n_same} pairs)")
    print(f"    Diff phenotype CKA:  {diff_cka:.4f} (n={n_diff} pairs)")
    print(f"    Ratio (same/diff):   {same_cka / diff_cka:.4f}")
    print(f"    Delta:               {same_cka - diff_cka:+.4f}")

    return results


def analysis_2_model_comparison(jepa_emb, ptst_emb, patient_ids, patient_cluster,
                                max_windows, min_windows, rng):
    """How similarly do JEPA and PatchTST organize the same patients?

    For each patient, compute CKA(JEPA_windows, PatchTST_windows) on the
    same set of windows. High CKA = both models learn similar internal
    structure for that patient.

    Then check if this agreement varies by hemodynamic phenotype.
    """
    print(f"\n{'─' * 70}")
    print("Analysis 2: JEPA vs PatchTST representational agreement per patient")
    print(f"{'─' * 70}")

    unique_pids = np.unique(patient_ids)
    results = []

    for pid in unique_pids:
        mask = patient_ids == pid
        n_windows = mask.sum()
        if n_windows < min_windows:
            continue

        jepa_pat = jepa_emb[mask]
        ptst_pat = ptst_emb[mask]

        # Subsample if needed
        if n_windows > max_windows:
            idx = rng.choice(n_windows, size=max_windows, replace=False)
            jepa_pat = jepa_pat[idx]
            ptst_pat = ptst_pat[idx]

        # CKA between models on same samples
        score = cka(jepa_pat, ptst_pat, kernel="linear")

        cluster = patient_cluster.get(str(pid), -1)
        results.append({
            "patient_id": str(pid),
            "n_windows": n_windows,
            "hemo_cluster": cluster,
            "cka_jepa_vs_ptst": score,
        })

    df = pd.DataFrame(results)
    print(f"  Patients analyzed: {len(df)}")
    print(f"  Mean CKA (JEPA vs PatchTST): {df['cka_jepa_vs_ptst'].mean():.4f} "
          f"± {df['cka_jepa_vs_ptst'].std():.4f}")
    print(f"  Median: {df['cka_jepa_vs_ptst'].median():.4f}")
    print(f"  Range: [{df['cka_jepa_vs_ptst'].min():.4f}, "
          f"{df['cka_jepa_vs_ptst'].max():.4f}]")

    # Breakdown by hemodynamic cluster
    hemo_df = df[df["hemo_cluster"] >= 0]
    if len(hemo_df) > 0:
        print(f"\n  CKA by hemodynamic cluster:")
        for c in sorted(hemo_df["hemo_cluster"].unique()):
            cluster_df = hemo_df[hemo_df["hemo_cluster"] == c]
            print(f"    Cluster {c}: {cluster_df['cka_jepa_vs_ptst'].mean():.4f} "
                  f"± {cluster_df['cka_jepa_vs_ptst'].std():.4f} "
                  f"(n={len(cluster_df)})")

    return results


def analysis_3_within_patient_states(embeddings, labels, patient_ids,
                                     max_windows, rng, model_name):
    """Within a patient, do pre-hypotensive and normal windows have
    similar or different representational geometry?

    For patients with both positive and negative windows, compute:
    - CKA(pos_windows, neg_windows) within the same patient
    - Compare to CKA(neg_split1, neg_split2) as a baseline

    If pos-neg CKA is lower than neg-neg CKA, the encoder creates a
    structurally different geometry for pre-hypotensive states.
    """
    print(f"\n{'─' * 70}")
    print(f"Analysis 3: Within-patient state geometry ({model_name})")
    print(f"{'─' * 70}")

    unique_pids = np.unique(patient_ids)
    results = []

    for pid in unique_pids:
        mask = patient_ids == pid
        emb_pat = embeddings[mask]
        lab_pat = labels[mask]

        pos_mask = lab_pat == 1
        neg_mask = lab_pat == 0
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        # Need enough of both classes
        if n_pos < 10 or n_neg < 10:
            continue

        emb_pos = emb_pat[pos_mask]
        emb_neg = emb_pat[neg_mask]

        # Subsample negatives if too many
        if n_neg > max_windows:
            idx = rng.choice(n_neg, size=max_windows, replace=False)
            emb_neg = emb_neg[idx]
            n_neg = max_windows

        if n_pos > max_windows:
            idx = rng.choice(n_pos, size=max_windows, replace=False)
            emb_pos = emb_pos[idx]
            n_pos = max_windows

        # CKA between positive and negative windows
        n_common = min(n_pos, n_neg)
        cka_pos_neg = cka_cross(emb_pos, emb_neg, n_common=n_common,
                                kernel="linear", rng=rng)

        # Baseline: split negative windows in half and compute CKA
        if n_neg >= 20:
            half = n_neg // 2
            perm = rng.permutation(n_neg)
            neg_a = emb_neg[perm[:half]]
            neg_b = emb_neg[perm[half:2 * half]]
            cka_neg_neg = cka_cross(neg_a, neg_b, n_common=half,
                                    kernel="linear", rng=rng)
        else:
            cka_neg_neg = np.nan

        results.append({
            "model": model_name,
            "patient_id": str(pid),
            "n_pos": n_pos,
            "n_neg": min(n_neg, max_windows),
            "cka_pos_neg": cka_pos_neg,
            "cka_neg_neg_baseline": cka_neg_neg,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No patients with enough positive and negative windows.")
        return results

    print(f"  Patients analyzed: {len(df)}")
    print(f"\n  CKA(pos, neg):        {df['cka_pos_neg'].mean():.4f} "
          f"± {df['cka_pos_neg'].std():.4f}")
    print(f"  CKA(neg, neg) baseline: {df['cka_neg_neg_baseline'].mean():.4f} "
          f"± {df['cka_neg_neg_baseline'].std():.4f}")
    ratio = df['cka_pos_neg'].mean() / df['cka_neg_neg_baseline'].mean()
    print(f"  Ratio (pos-neg / neg-neg): {ratio:.4f}")
    print(f"\n  Interpretation: ", end="")
    if ratio < 0.9:
        print("Pre-hypotensive windows have DIFFERENT geometry than normal.")
    elif ratio > 1.1:
        print("Pre-hypotensive windows have MORE structured geometry than normal.")
    else:
        print("Pre-hypotensive and normal windows have SIMILAR geometry.")

    return results


def analysis_4_group_level_cka(embeddings, patient_ids, patient_cluster,
                               max_windows, rng, model_name):
    """Pool all windows from each phenotype group and compute CKA between groups.

    For each pair of hemodynamic clusters, pool all patient windows belonging
    to that cluster and compute CKA. High CKA between two clusters means they
    share representational structure despite being different physiological states.
    Low CKA means distinct geometry.

    Also compute within-cluster CKA (split in half) as baseline.
    """
    print(f"\n{'─' * 70}")
    print(f"Analysis 4: Group-level CKA between hemodynamic clusters ({model_name})")
    print(f"{'─' * 70}")

    # Assign per-window cluster labels
    hemo_labels = np.array([patient_cluster.get(str(pid), -1)
                           for pid in patient_ids])
    valid = hemo_labels >= 0

    emb_valid = embeddings[valid]
    hemo_valid = hemo_labels[valid]

    # Pool windows per cluster, subsample to manageable size
    cluster_embeddings = {}
    max_per_cluster = 2000
    clusters_present = sorted(np.unique(hemo_valid))

    for c in clusters_present:
        c_mask = hemo_valid == c
        c_emb = emb_valid[c_mask]
        if len(c_emb) > max_per_cluster:
            idx = rng.choice(len(c_emb), size=max_per_cluster, replace=False)
            c_emb = c_emb[idx]
        cluster_embeddings[c] = c_emb

    print(f"  Clusters: {clusters_present}")
    for c in clusters_present:
        print(f"    Cluster {c}: {len(cluster_embeddings[c])} windows")

    # Within-cluster CKA (split-half reliability)
    print(f"\n  Within-cluster CKA (split-half):")
    within_scores = {}
    for c in clusters_present:
        emb_c = cluster_embeddings[c]
        if len(emb_c) < 40:
            within_scores[c] = np.nan
            continue
        half = len(emb_c) // 2
        perm = rng.permutation(len(emb_c))
        a = emb_c[perm[:half]]
        b = emb_c[perm[half:2 * half]]
        n_common = half
        within_scores[c] = cka_cross(a, b, n_common=n_common,
                                     kernel="linear", rng=rng)
        print(f"    Cluster {c}: {within_scores[c]:.4f}")

    # Between-cluster CKA
    print(f"\n  Between-cluster CKA:")
    results = []
    for c1, c2 in combinations(clusters_present, 2):
        emb_1 = cluster_embeddings[c1]
        emb_2 = cluster_embeddings[c2]
        n_common = min(len(emb_1), len(emb_2), 500)
        score = cka_cross(emb_1, emb_2, n_common=n_common,
                          kernel="linear", rng=rng)
        results.append({
            "model": model_name,
            "cluster_a": c1,
            "cluster_b": c2,
            "type": "between",
            "cka_linear": score,
            "n_common": n_common,
        })
        print(f"    Cluster {c1} vs {c2}: {score:.4f}")

    # Add within scores
    for c in clusters_present:
        results.append({
            "model": model_name,
            "cluster_a": c,
            "cluster_b": c,
            "type": "within",
            "cka_linear": within_scores[c],
            "n_common": len(cluster_embeddings[c]) // 2,
        })

    # Summary
    between_scores = [r["cka_linear"] for r in results
                      if r["type"] == "between" and not np.isnan(r["cka_linear"])]
    within_vals = [v for v in within_scores.values() if not np.isnan(v)]

    print(f"\n  Summary ({model_name}):")
    print(f"    Mean within-cluster CKA:  {np.mean(within_vals):.4f}")
    print(f"    Mean between-cluster CKA: {np.mean(between_scores):.4f}")
    print(f"    Ratio (between/within):   "
          f"{np.mean(between_scores) / np.mean(within_vals):.4f}")

    return results


def analysis_6_temporal_segments(embeddings, labels, patient_ids, min_windows,
                                rng, model_name):
    """CKA between temporal segments (early/middle/late) within patients.

    Split each patient's windows into temporal thirds (assuming dataset order
    is temporal). Compute CKA between each pair of thirds.

    If CKA(early, late) is lower for deteriorating patients (those who
    eventually become hypotensive), the encoder captures temporal drift
    toward a distinct geometry before clinical events.
    """
    print(f"\n{'─' * 70}")
    print(f"Analysis 6: Temporal segment CKA ({model_name})")
    print(f"{'─' * 70}")

    unique_pids = np.unique(patient_ids)
    results = []

    for pid in unique_pids:
        mask = patient_ids == pid
        emb_pat = embeddings[mask]
        lab_pat = labels[mask]
        n = len(emb_pat)

        if n < min_windows * 3:  # need enough for 3 segments
            continue

        # Split into temporal thirds (assuming order = time)
        third = n // 3
        early = emb_pat[:third]
        middle = emb_pat[third:2 * third]
        late = emb_pat[2 * third:3 * third]

        # CKA between segment pairs
        n_common = min(third, 50)
        cka_em = cka_cross(early, middle, n_common=n_common,
                           kernel="linear", rng=rng)
        cka_ml = cka_cross(middle, late, n_common=n_common,
                           kernel="linear", rng=rng)
        cka_el = cka_cross(early, late, n_common=n_common,
                           kernel="linear", rng=rng)

        # Patient has hypotension events?
        has_hypo = lab_pat.sum() > 0
        hypo_rate = lab_pat.mean()

        results.append({
            "model": model_name,
            "patient_id": str(pid),
            "n_windows": n,
            "has_hypotension": has_hypo,
            "hypo_rate": hypo_rate,
            "cka_early_middle": cka_em,
            "cka_middle_late": cka_ml,
            "cka_early_late": cka_el,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No patients with enough windows.")
        return results

    print(f"  Patients analyzed: {len(df)}")

    # Overall
    print(f"\n  Overall temporal CKA:")
    print(f"    Early↔Middle:  {df['cka_early_middle'].mean():.4f} ± "
          f"{df['cka_early_middle'].std():.4f}")
    print(f"    Middle↔Late:   {df['cka_middle_late'].mean():.4f} ± "
          f"{df['cka_middle_late'].std():.4f}")
    print(f"    Early↔Late:    {df['cka_early_late'].mean():.4f} ± "
          f"{df['cka_early_late'].std():.4f}")

    # Split by hypotension status
    hypo_df = df[df["has_hypotension"]]
    stable_df = df[~df["has_hypotension"]]

    if len(hypo_df) >= 5 and len(stable_df) >= 5:
        print(f"\n  Patients with hypotension events (n={len(hypo_df)}):")
        print(f"    Early↔Late CKA: {hypo_df['cka_early_late'].mean():.4f} ± "
              f"{hypo_df['cka_early_late'].std():.4f}")
        print(f"  Stable patients (n={len(stable_df)}):")
        print(f"    Early↔Late CKA: {stable_df['cka_early_late'].mean():.4f} ± "
              f"{stable_df['cka_early_late'].std():.4f}")
        delta = (hypo_df['cka_early_late'].mean() -
                 stable_df['cka_early_late'].mean())
        print(f"    Delta (hypo - stable): {delta:+.4f}")
        if delta < -0.01:
            print(f"    → Deteriorating patients show MORE geometric drift over time")
        elif delta > 0.01:
            print(f"    → Deteriorating patients show LESS geometric drift (more stable)")
        else:
            print(f"    → No significant difference in temporal drift")

    return results


def analysis_7_clinical_similarity(embeddings, patient_ids, patient_cluster,
                                   min_windows, rng, model_name):
    """CKA stratified by clinical similarity.

    Compute per-patient mean physiological features from icuDataExtraction
    (19-dim feature vector). Group patient pairs into similarity tiers based
    on Euclidean distance in feature space. Check if clinically similar patients
    have higher CKA (shared representational geometry).
    """
    print(f"\n{'─' * 70}")
    print(f"Analysis 7: CKA by clinical similarity ({model_name})")
    print(f"{'─' * 70}")

    # Load per-patient mean features from icuDataExtraction
    X_stats = np.load(ICU_OUTPUT / "X_stats.npy", mmap_mode="r")
    icu_patient_ids = np.load(ICU_OUTPUT / "patient_ids.npy", allow_pickle=True)

    # Compute per-patient mean feature vector (mean over time and windows)
    icu_unique = np.unique(icu_patient_ids)
    patient_features = {}
    for pid in icu_unique:
        mask = icu_patient_ids == pid
        # X_stats: (n_windows, 19, 109) -> mean over windows and time
        pat_stats = X_stats[mask]
        pat_mean = np.nanmean(pat_stats, axis=(0, 2))  # (19,)
        if not np.any(np.isnan(pat_mean)):
            patient_features[str(pid)] = pat_mean

    # Find eligible patients (in embeddings + have features + enough windows)
    unique_pids = np.unique(patient_ids)
    eligible = []
    for pid in unique_pids:
        n_windows = (patient_ids == pid).sum()
        if n_windows >= min_windows and str(pid) in patient_features:
            eligible.append(str(pid))

    print(f"  Eligible patients (embeddings + features + >={min_windows} windows): "
          f"{len(eligible)}")

    if len(eligible) < 20:
        print("  Too few patients. Skipping.")
        return []

    # Compute all pairwise clinical distances
    feat_matrix = np.array([patient_features[p] for p in eligible])
    # Normalize features for fair distance
    feat_norm = (feat_matrix - feat_matrix.mean(axis=0)) / (feat_matrix.std(axis=0) + 1e-8)

    from sklearn.metrics.pairwise import euclidean_distances
    clin_dist = euclidean_distances(feat_norm)

    # Get upper triangle distances and corresponding CKA
    n_elig = len(eligible)
    triu_i, triu_j = np.triu_indices(n_elig, k=1)

    # Subsample pairs if too many
    max_pairs = 500
    n_pairs = len(triu_i)
    if n_pairs > max_pairs:
        sub_idx = rng.choice(n_pairs, size=max_pairs, replace=False)
        triu_i = triu_i[sub_idx]
        triu_j = triu_j[sub_idx]

    print(f"  Computing CKA for {len(triu_i)} patient pairs...")

    pair_data = []
    for idx, (i, j) in enumerate(zip(triu_i, triu_j)):
        pid_a = eligible[i]
        pid_b = eligible[j]

        emb_a = get_patient_embeddings(embeddings, patient_ids, pid_a,
                                       50, rng)
        emb_b = get_patient_embeddings(embeddings, patient_ids, pid_b,
                                       50, rng)

        n_common = min(len(emb_a), len(emb_b))
        if n_common < 10:
            continue

        score = cka_cross(emb_a, emb_b, n_common=n_common,
                          kernel="linear", rng=rng)

        pair_data.append({
            "model": model_name,
            "patient_a": pid_a,
            "patient_b": pid_b,
            "clinical_distance": clin_dist[i, j],
            "cka_linear": score,
        })

        if (idx + 1) % 100 == 0:
            print(f"    {idx + 1}/{len(triu_i)} pairs computed")

    df = pd.DataFrame(pair_data)

    # Stratify into quartiles of clinical distance
    df["distance_quartile"] = pd.qcut(df["clinical_distance"], q=4,
                                      labels=["Q1_most_similar", "Q2", "Q3",
                                              "Q4_least_similar"])

    print(f"\n  CKA by clinical similarity quartile:")
    for q in ["Q1_most_similar", "Q2", "Q3", "Q4_least_similar"]:
        qdf = df[df["distance_quartile"] == q]
        print(f"    {q}: CKA={qdf['cka_linear'].mean():.4f} ± "
              f"{qdf['cka_linear'].std():.4f} (n={len(qdf)}, "
              f"dist={qdf['clinical_distance'].mean():.2f})")

    # Correlation
    corr = df["clinical_distance"].corr(df["cka_linear"])
    print(f"\n  Pearson correlation (clinical_dist vs CKA): {corr:.4f}")
    if abs(corr) < 0.05:
        print(f"    → No relationship between clinical similarity and CKA")
    elif corr < -0.05:
        print(f"    → More clinically similar patients DO share more geometry")
    else:
        print(f"    → More clinically similar patients share LESS geometry (unexpected)")

    return pair_data


def analysis_8_cross_patient_hypo(embeddings, labels, patient_ids,
                                  min_windows, rng, model_name):
    """Cross-patient CKA: pre-hypotensive windows vs normal windows.

    Pool pre-hypotensive windows from different patients and compute their
    cross-patient CKA. Compare to CKA between normal windows from different
    patients. If pre-hypo cross-patient CKA > normal cross-patient CKA,
    there IS a shared 'approaching hypotension' geometry.

    This tests whether the null finding from distance analysis (no cross-patient
    hypotension signal) holds under the more nuanced CKA measure.
    """
    print(f"\n{'─' * 70}")
    print(f"Analysis 8: Cross-patient CKA for pre-hypo vs normal ({model_name})")
    print(f"{'─' * 70}")

    unique_pids = np.unique(patient_ids)

    # Collect per-patient positive and negative windows
    patients_with_hypo = []
    patients_normal_only = []

    for pid in unique_pids:
        mask = patient_ids == pid
        emb_pat = embeddings[mask]
        lab_pat = labels[mask]

        n_pos = (lab_pat == 1).sum()
        n_neg = (lab_pat == 0).sum()

        if n_pos >= 10:
            patients_with_hypo.append(pid)
        elif n_neg >= min_windows:
            patients_normal_only.append(pid)

    print(f"  Patients with >=10 pre-hypo windows: {len(patients_with_hypo)}")
    print(f"  Patients with only normal windows (>={min_windows}): "
          f"{len(patients_normal_only)}")

    if len(patients_with_hypo) < 5:
        print("  Too few hypotensive patients. Skipping.")
        return []

    # Cross-patient CKA between pre-hypo windows from different patients
    max_pairs = 200
    hypo_pairs = list(combinations(range(len(patients_with_hypo)), 2))
    if len(hypo_pairs) > max_pairs:
        pair_idx = rng.choice(len(hypo_pairs), size=max_pairs, replace=False)
        hypo_pairs = [hypo_pairs[i] for i in pair_idx]

    print(f"\n  Computing cross-patient CKA for pre-hypo windows "
          f"({len(hypo_pairs)} pairs)...")

    hypo_ckas = []
    for idx_a, idx_b in hypo_pairs:
        pid_a = patients_with_hypo[idx_a]
        pid_b = patients_with_hypo[idx_b]

        mask_a = (patient_ids == pid_a)
        mask_b = (patient_ids == pid_b)
        emb_a_pos = embeddings[mask_a][labels[mask_a] == 1]
        emb_b_pos = embeddings[mask_b][labels[mask_b] == 1]

        # Subsample
        if len(emb_a_pos) > 50:
            emb_a_pos = emb_a_pos[rng.choice(len(emb_a_pos), 50, replace=False)]
        if len(emb_b_pos) > 50:
            emb_b_pos = emb_b_pos[rng.choice(len(emb_b_pos), 50, replace=False)]

        n_common = min(len(emb_a_pos), len(emb_b_pos))
        if n_common < 5:
            continue

        score = cka_cross(emb_a_pos, emb_b_pos, n_common=n_common,
                          kernel="linear", rng=rng)
        hypo_ckas.append(score)

    # Cross-patient CKA between normal windows from different patients
    # Use both hypo patients' normal windows AND normal-only patients
    all_normal_patients = list(patients_with_hypo) + patients_normal_only
    normal_pairs = list(combinations(range(len(all_normal_patients)), 2))
    if len(normal_pairs) > max_pairs:
        pair_idx = rng.choice(len(normal_pairs), size=max_pairs, replace=False)
        normal_pairs = [normal_pairs[i] for i in pair_idx]

    print(f"  Computing cross-patient CKA for normal windows "
          f"({len(normal_pairs)} pairs)...")

    normal_ckas = []
    for idx_a, idx_b in normal_pairs:
        pid_a = all_normal_patients[idx_a]
        pid_b = all_normal_patients[idx_b]

        mask_a = (patient_ids == pid_a)
        mask_b = (patient_ids == pid_b)
        emb_a_neg = embeddings[mask_a][labels[mask_a] == 0]
        emb_b_neg = embeddings[mask_b][labels[mask_b] == 0]

        # Subsample
        if len(emb_a_neg) > 50:
            emb_a_neg = emb_a_neg[rng.choice(len(emb_a_neg), 50, replace=False)]
        if len(emb_b_neg) > 50:
            emb_b_neg = emb_b_neg[rng.choice(len(emb_b_neg), 50, replace=False)]

        n_common = min(len(emb_a_neg), len(emb_b_neg))
        if n_common < 5:
            continue

        score = cka_cross(emb_a_neg, emb_b_neg, n_common=n_common,
                          kernel="linear", rng=rng)
        normal_ckas.append(score)

    hypo_mean = np.mean(hypo_ckas) if hypo_ckas else np.nan
    hypo_std = np.std(hypo_ckas) if hypo_ckas else np.nan
    normal_mean = np.mean(normal_ckas) if normal_ckas else np.nan
    normal_std = np.std(normal_ckas) if normal_ckas else np.nan

    print(f"\n  Results ({model_name}):")
    print(f"    Cross-patient pre-hypo CKA:  {hypo_mean:.4f} ± {hypo_std:.4f} "
          f"(n={len(hypo_ckas)} pairs)")
    print(f"    Cross-patient normal CKA:    {normal_mean:.4f} ± {normal_std:.4f} "
          f"(n={len(normal_ckas)} pairs)")
    if not np.isnan(hypo_mean) and not np.isnan(normal_mean):
        ratio = hypo_mean / normal_mean
        print(f"    Ratio (hypo/normal):         {ratio:.4f}")
        delta = hypo_mean - normal_mean
        print(f"    Delta:                       {delta:+.4f}")
        if delta > 0.005:
            print(f"    → Pre-hypo windows share MORE geometry across patients!")
            print(f"      There IS a shared pre-hypotensive structure in CKA space.")
        elif delta < -0.005:
            print(f"    → Pre-hypo windows share LESS geometry across patients.")
        else:
            print(f"    → No difference — confirms distance-based null finding.")

    results = [{
        "model": model_name,
        "hypo_cka_mean": hypo_mean,
        "hypo_cka_std": hypo_std,
        "normal_cka_mean": normal_mean,
        "normal_cka_std": normal_std,
        "n_hypo_pairs": len(hypo_ckas),
        "n_normal_pairs": len(normal_ckas),
    }]
    return results


def analysis_9_within_patient_hypo_geometry(embeddings, labels, patient_ids,
                                            max_windows, rng, model_name):
    """Within-patient: do pre-hypo windows have more consistent internal
    geometry than normal windows?

    For each patient with both states:
    - CKA(pos_split1, pos_split2): internal consistency of pre-hypo geometry
    - CKA(neg_split1, neg_split2): internal consistency of normal geometry

    If pos-pos CKA > neg-neg CKA: pre-hypotensive windows form a more
    coherent geometric structure within the patient.

    Also computes CKA(pos, neg) to measure how different the two states'
    geometries are within each patient.
    """
    print(f"\n{'─' * 70}")
    print(f"Analysis 9: Within-patient hypo geometry consistency ({model_name})")
    print(f"{'─' * 70}")

    unique_pids = np.unique(patient_ids)
    results = []

    for pid in unique_pids:
        mask = patient_ids == pid
        emb_pat = embeddings[mask]
        lab_pat = labels[mask]

        pos_mask = lab_pat == 1
        neg_mask = lab_pat == 0
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()

        # Need enough of both to split
        if n_pos < 20 or n_neg < 20:
            continue

        emb_pos = emb_pat[pos_mask]
        emb_neg = emb_pat[neg_mask]

        # Subsample if too many
        if n_pos > max_windows:
            idx = rng.choice(n_pos, size=max_windows, replace=False)
            emb_pos = emb_pos[idx]
            n_pos = max_windows
        if n_neg > max_windows:
            idx = rng.choice(n_neg, size=max_windows, replace=False)
            emb_neg = emb_neg[idx]
            n_neg = max_windows

        # Split pos in half -> CKA(pos_a, pos_b)
        half_pos = n_pos // 2
        perm_pos = rng.permutation(n_pos)
        pos_a = emb_pos[perm_pos[:half_pos]]
        pos_b = emb_pos[perm_pos[half_pos:2 * half_pos]]
        cka_pos_pos = cka_cross(pos_a, pos_b, n_common=half_pos,
                                kernel="linear", rng=rng)

        # Split neg in half -> CKA(neg_a, neg_b)
        half_neg = n_neg // 2
        perm_neg = rng.permutation(n_neg)
        neg_a = emb_neg[perm_neg[:half_neg]]
        neg_b = emb_neg[perm_neg[half_neg:2 * half_neg]]
        cka_neg_neg = cka_cross(neg_a, neg_b, n_common=half_neg,
                                kernel="linear", rng=rng)

        # CKA(pos, neg) — how different are the two states?
        n_common = min(n_pos, n_neg, 50)
        cka_pos_neg = cka_cross(emb_pos, emb_neg, n_common=n_common,
                                kernel="linear", rng=rng)

        results.append({
            "model": model_name,
            "patient_id": str(pid),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "cka_pos_pos": cka_pos_pos,
            "cka_neg_neg": cka_neg_neg,
            "cka_pos_neg": cka_pos_neg,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No patients with enough windows of both classes.")
        return results

    print(f"  Patients analyzed: {len(df)}")
    print(f"\n  Internal consistency (split-half CKA):")
    print(f"    Pre-hypo (pos-pos):  {df['cka_pos_pos'].mean():.4f} ± "
          f"{df['cka_pos_pos'].std():.4f}")
    print(f"    Normal (neg-neg):    {df['cka_neg_neg'].mean():.4f} ± "
          f"{df['cka_neg_neg'].std():.4f}")
    pp_mean = df['cka_pos_pos'].mean()
    nn_mean = df['cka_neg_neg'].mean()
    print(f"    Ratio (pos/neg):     {pp_mean / nn_mean:.4f}")
    print(f"    Delta:               {pp_mean - nn_mean:+.4f}")

    print(f"\n  Cross-state CKA:")
    print(f"    CKA(pos, neg):       {df['cka_pos_neg'].mean():.4f} ± "
          f"{df['cka_pos_neg'].std():.4f}")

    # Is cross-state CKA lower than within-state?
    within_mean = (pp_mean + nn_mean) / 2
    cross_mean = df['cka_pos_neg'].mean()
    print(f"\n  Within-state avg CKA:  {within_mean:.4f}")
    print(f"  Cross-state CKA:       {cross_mean:.4f}")
    print(f"  Ratio (cross/within):  {cross_mean / within_mean:.4f}")

    if cross_mean / within_mean < 0.85:
        print(f"  → Strong geometric separation between states within patients")
    elif cross_mean / within_mean < 0.95:
        print(f"  → Moderate geometric separation between states")
    else:
        print(f"  → States share similar geometry within patients")

    return results


def analysis_12_within_patient_hemo_clusters(embeddings, patient_ids,
                                             max_windows, rng, model_name):
    """Within-patient CKA for hemodynamic cluster transitions.

    For patients whose windows span multiple hemodynamic clusters:
    - Split-half CKA within each cluster (self-consistency of that state)
    - CKA between different clusters within the same patient (state separation)

    If within-cluster CKA > between-cluster CKA, the encoder creates
    structurally distinct representations for different hemodynamic states
    within a patient's trajectory.
    """
    print(f"\n{'═' * 70}")
    print(f"Analysis 12: Within-patient hemodynamic cluster CKA ({model_name})")
    print(f"{'═' * 70}")
    print(f"  For each patient with windows in multiple clusters:")
    print(f"    Within-cluster: split cluster_i windows in half → CKA")
    print(f"    Between-cluster: CKA(cluster_i windows, cluster_j windows)")

    # Load window-level hemodynamic clusters
    hemo_data = np.load(CLUSTERING_DIR / "window_hemo_clusters.npz",
                        allow_pickle=True)
    hemo_clusters = hemo_data["hemo_clusters"]

    valid = hemo_clusters >= 0
    print(f"\n  Windows with cluster labels: {valid.sum()}/{len(hemo_clusters)}")

    # Build per-patient cluster → embedding mapping
    unique_pids = np.unique(patient_ids[valid])
    results = []

    patients_analyzed = 0
    for pid in unique_pids:
        mask = (patient_ids == pid) & valid
        indices = np.where(mask)[0]
        clusters_for_pid = hemo_clusters[indices]

        # Group indices by cluster
        cluster_map = {}
        for c in np.unique(clusters_for_pid):
            c_idx = indices[clusters_for_pid == c]
            if len(c_idx) >= 10:  # need enough for split-half
                cluster_map[int(c)] = c_idx

        if len(cluster_map) < 2:
            continue

        patients_analyzed += 1

        # Within-cluster CKA: split-half for each cluster
        within_ckas = []
        for c, c_indices in cluster_map.items():
            emb_c = embeddings[c_indices]
            if len(emb_c) > max_windows:
                sub = rng.choice(len(emb_c), max_windows, replace=False)
                emb_c = emb_c[sub]

            n = len(emb_c)
            if n < 10:
                continue
            half = n // 2
            perm = rng.permutation(n)
            a = emb_c[perm[:half]]
            b = emb_c[perm[half:2 * half]]
            score = cka_cross(a, b, n_common=half, kernel="linear", rng=rng)
            within_ckas.append(score)

        # Between-cluster CKA: for each pair of clusters
        between_ckas = []
        cluster_ids = list(cluster_map.keys())
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                ci = cluster_ids[i]
                cj = cluster_ids[j]

                emb_i = embeddings[cluster_map[ci]]
                emb_j = embeddings[cluster_map[cj]]

                if len(emb_i) > max_windows:
                    emb_i = emb_i[rng.choice(len(emb_i), max_windows, replace=False)]
                if len(emb_j) > max_windows:
                    emb_j = emb_j[rng.choice(len(emb_j), max_windows, replace=False)]

                n_common = min(len(emb_i), len(emb_j))
                if n_common < 5:
                    continue

                score = cka_cross(emb_i, emb_j, n_common=n_common,
                                  kernel="linear", rng=rng)
                between_ckas.append(score)

        if within_ckas and between_ckas:
            results.append({
                "model": model_name,
                "patient_id": str(pid),
                "n_clusters": len(cluster_map),
                "within_cluster_cka": np.mean(within_ckas),
                "between_cluster_cka": np.mean(between_ckas),
                "n_within": len(within_ckas),
                "n_between": len(between_ckas),
            })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No patients with sufficient multi-cluster windows.")
        return []

    within_mean = df["within_cluster_cka"].mean()
    between_mean = df["between_cluster_cka"].mean()

    print(f"\n  Patients analyzed: {patients_analyzed} attempted, {len(df)} valid")
    print(f"\n  Results ({model_name}):")
    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ WITHIN-CLUSTER CKA (split-half, same hemo state):       │")
    print(f"  │   Mean: {within_mean:.4f} ± {df['within_cluster_cka'].std():.4f}                            │")
    print(f"  │                                                         │")
    print(f"  │ BETWEEN-CLUSTER CKA (different hemo states):            │")
    print(f"  │   Mean: {between_mean:.4f} ± {df['between_cluster_cka'].std():.4f}                            │")
    print(f"  │                                                         │")
    print(f"  │ RATIO (between / within):  {between_mean / within_mean:.4f}                     │")
    print(f"  │ DELTA (within - between): {within_mean - between_mean:+.4f}                     │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # Paired test
    deltas = df["within_cluster_cka"] - df["between_cluster_cka"]
    n_positive = (deltas > 0).sum()
    print(f"\n  Paired comparison (per patient):")
    print(f"    Patients where within > between: {n_positive}/{len(df)} "
          f"({100 * n_positive / len(df):.1f}%)")
    print(f"    Mean delta: {deltas.mean():+.4f} ± {deltas.std():.4f}")

    if deltas.std() > 0:
        cohens_d = deltas.mean() / deltas.std()
        print(f"    Cohen's d (paired): {cohens_d:.3f}")

    ratio = between_mean / within_mean
    if ratio < 0.85:
        print(f"\n  ✓ Strong separation: different hemodynamic states have")
        print(f"    distinctly different relational structure within patients.")
    elif ratio < 0.95:
        print(f"\n  ~ Moderate separation: some structural difference between")
        print(f"    hemodynamic states within patients.")
    else:
        print(f"\n  ✗ No separation: hemodynamic states share similar structure")
        print(f"    within patients.")

    return results


def analysis_5_same_vs_diff_patient(embeddings, patient_ids, max_windows,
                                    min_windows, rng, model_name):
    """Compare same-patient CKA (split-half) vs different-patient CKA.

    Same-patient CKA: split a patient's windows into two halves and compute
    CKA between the halves. Measures how self-consistent the patient's
    representational geometry is.

    Different-patient CKA: compute CKA between windows from two different
    patients. Measures how much geometry is shared across patients.

    If same-patient >> diff-patient: the encoder creates patient-specific
    geometry (fingerprint). If they're similar: the encoder learns a
    universal structure shared across patients.
    """
    print(f"\n{'─' * 70}")
    print(f"Analysis 5: Same-patient vs different-patient CKA ({model_name})")
    print(f"{'─' * 70}")

    unique_pids = np.unique(patient_ids)
    eligible_patients = []
    for pid in unique_pids:
        n_windows = (patient_ids == pid).sum()
        if n_windows >= min_windows:
            eligible_patients.append(pid)

    print(f"  Eligible patients (>={min_windows} windows): {len(eligible_patients)}")

    # Same-patient CKA: split-half
    same_patient_ckas = []
    for pid in eligible_patients:
        emb_pat = get_patient_embeddings(embeddings, patient_ids, pid,
                                         max_windows * 2, rng)  # get more for split
        if len(emb_pat) < min_windows:
            continue

        half = len(emb_pat) // 2
        perm = rng.permutation(len(emb_pat))
        half_a = emb_pat[perm[:half]]
        half_b = emb_pat[perm[half:2 * half]]

        n_common = half
        score = cka_cross(half_a, half_b, n_common=n_common,
                          kernel="linear", rng=rng)
        same_patient_ckas.append(score)

    # Different-patient CKA: random pairs
    max_pairs = 500
    all_pairs = list(combinations(range(len(eligible_patients)), 2))
    if len(all_pairs) > max_pairs:
        pair_idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        selected_pairs = [all_pairs[i] for i in pair_idx]
    else:
        selected_pairs = all_pairs

    diff_patient_ckas = []
    for idx_a, idx_b in selected_pairs:
        pid_a = eligible_patients[idx_a]
        pid_b = eligible_patients[idx_b]

        emb_a = get_patient_embeddings(embeddings, patient_ids, pid_a,
                                       max_windows, rng)
        emb_b = get_patient_embeddings(embeddings, patient_ids, pid_b,
                                       max_windows, rng)

        n_common = min(len(emb_a), len(emb_b))
        if n_common < 10:
            continue

        score = cka_cross(emb_a, emb_b, n_common=n_common,
                          kernel="linear", rng=rng)
        diff_patient_ckas.append(score)

    same_mean = np.mean(same_patient_ckas)
    same_std = np.std(same_patient_ckas)
    diff_mean = np.mean(diff_patient_ckas)
    diff_std = np.std(diff_patient_ckas)

    print(f"\n  Results ({model_name}):")
    print(f"    Same-patient CKA (split-half):  {same_mean:.4f} ± {same_std:.4f} "
          f"(n={len(same_patient_ckas)} patients)")
    print(f"    Diff-patient CKA:               {diff_mean:.4f} ± {diff_std:.4f} "
          f"(n={len(diff_patient_ckas)} pairs)")
    print(f"    Ratio (same/diff):              {same_mean / diff_mean:.2f}x")
    print(f"    Delta:                          {same_mean - diff_mean:+.4f}")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((same_std**2 + diff_std**2) / 2)
    if pooled_std > 0:
        cohens_d = (same_mean - diff_mean) / pooled_std
        print(f"    Cohen's d:                      {cohens_d:.2f}")

    return {
        "model": model_name,
        "same_patient_cka_mean": same_mean,
        "same_patient_cka_std": same_std,
        "diff_patient_cka_mean": diff_mean,
        "diff_patient_cka_std": diff_std,
        "ratio": same_mean / diff_mean,
        "n_patients": len(same_patient_ckas),
        "n_pairs": len(diff_patient_ckas),
    }


def analysis_10_controlled_cross_patient_hypo(embeddings, labels, patient_ids,
                                              max_windows, rng, model_name):
    """Controlled cross-patient CKA: same-condition vs different-condition
    for hypotension labels.

    The direct test of patient-invariant condition encoding:
    For each pair of patients (A, B) who both have pre-hypo and normal windows:
      - CKA(A_pre-hypo, B_pre-hypo)  = same condition, different patient
      - CKA(A_normal, B_normal)      = same condition, different patient
      - CKA(A_pre-hypo, B_normal)    = different condition, different patient
      - CKA(A_normal, B_pre-hypo)    = different condition, different patient

    If same-condition CKA > different-condition CKA, the encoder creates
    condition-specific relational structure that generalizes across patients.
    """
    print(f"\n{'═' * 70}")
    print(f"Analysis 10: Controlled cross-patient CKA — Hypotension ({model_name})")
    print(f"{'═' * 70}")
    print(f"  For each patient pair (A, B) with both states:")
    print(f"    Same condition:  CKA(A_prehypo, B_prehypo) and CKA(A_normal, B_normal)")
    print(f"    Diff condition:  CKA(A_prehypo, B_normal)  and CKA(A_normal, B_prehypo)")

    unique_pids = np.unique(patient_ids)

    # Find patients with enough windows of both classes
    eligible = []
    for pid in unique_pids:
        mask = patient_ids == pid
        lab = labels[mask]
        n_pos = (lab == 1).sum()
        n_neg = (lab == 0).sum()
        if n_pos >= 10 and n_neg >= 10:
            eligible.append(pid)

    print(f"\n  Patients with >=10 windows of each class: {len(eligible)}")

    if len(eligible) < 5:
        print("  Too few patients. Skipping.")
        return []

    # Compute CKA for patient pairs
    from itertools import combinations
    all_pairs = list(combinations(range(len(eligible)), 2))
    max_pairs = 300
    if len(all_pairs) > max_pairs:
        pair_idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        selected_pairs = [all_pairs[i] for i in pair_idx]
    else:
        selected_pairs = all_pairs

    print(f"  Computing CKA for {len(selected_pairs)} patient pairs...")

    results = []
    for count, (idx_a, idx_b) in enumerate(selected_pairs):
        pid_a = eligible[idx_a]
        pid_b = eligible[idx_b]

        mask_a = patient_ids == pid_a
        mask_b = patient_ids == pid_b

        # Get pre-hypo and normal windows for each patient
        emb_a_pos = embeddings[mask_a & (labels == 1)]
        emb_a_neg = embeddings[mask_a & (labels == 0)]
        emb_b_pos = embeddings[mask_b & (labels == 1)]
        emb_b_neg = embeddings[mask_b & (labels == 0)]

        # Subsample if needed
        cap = max_windows
        if len(emb_a_pos) > cap:
            emb_a_pos = emb_a_pos[rng.choice(len(emb_a_pos), cap, replace=False)]
        if len(emb_a_neg) > cap:
            emb_a_neg = emb_a_neg[rng.choice(len(emb_a_neg), cap, replace=False)]
        if len(emb_b_pos) > cap:
            emb_b_pos = emb_b_pos[rng.choice(len(emb_b_pos), cap, replace=False)]
        if len(emb_b_neg) > cap:
            emb_b_neg = emb_b_neg[rng.choice(len(emb_b_neg), cap, replace=False)]

        # Same condition comparisons
        n_pp = min(len(emb_a_pos), len(emb_b_pos))
        n_nn = min(len(emb_a_neg), len(emb_b_neg))
        # Different condition comparisons
        n_pn = min(len(emb_a_pos), len(emb_b_neg))
        n_np = min(len(emb_a_neg), len(emb_b_pos))

        if min(n_pp, n_nn, n_pn, n_np) < 5:
            continue

        cka_pp = cka_cross(emb_a_pos, emb_b_pos, n_common=n_pp,
                           kernel="linear", rng=rng)
        cka_nn = cka_cross(emb_a_neg, emb_b_neg, n_common=n_nn,
                           kernel="linear", rng=rng)
        cka_pn = cka_cross(emb_a_pos, emb_b_neg, n_common=n_pn,
                           kernel="linear", rng=rng)
        cka_np = cka_cross(emb_a_neg, emb_b_pos, n_common=n_np,
                           kernel="linear", rng=rng)

        same_cond = (cka_pp + cka_nn) / 2
        diff_cond = (cka_pn + cka_np) / 2

        results.append({
            "model": model_name,
            "patient_a": str(pid_a),
            "patient_b": str(pid_b),
            "cka_prehypo_prehypo": cka_pp,
            "cka_normal_normal": cka_nn,
            "cka_prehypo_normal": cka_pn,
            "cka_normal_prehypo": cka_np,
            "same_condition_mean": same_cond,
            "diff_condition_mean": diff_cond,
        })

        if (count + 1) % 50 == 0:
            print(f"    {count + 1}/{len(selected_pairs)} pairs computed")

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No valid pairs computed.")
        return []

    # Aggregate results
    cka_pp_mean = df["cka_prehypo_prehypo"].mean()
    cka_nn_mean = df["cka_normal_normal"].mean()
    cka_pn_mean = df["cka_prehypo_normal"].mean()
    cka_np_mean = df["cka_normal_prehypo"].mean()
    same_mean = df["same_condition_mean"].mean()
    diff_mean = df["diff_condition_mean"].mean()

    print(f"\n  Results ({model_name}, {len(df)} pairs):")
    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ SAME CONDITION (patient-invariant):                     │")
    print(f"  │   CKA(A_prehypo, B_prehypo):  {cka_pp_mean:.4f} ± {df['cka_prehypo_prehypo'].std():.4f}        │")
    print(f"  │   CKA(A_normal,  B_normal):    {cka_nn_mean:.4f} ± {df['cka_normal_normal'].std():.4f}        │")
    print(f"  │   Average same-condition:       {same_mean:.4f}                │")
    print(f"  │                                                         │")
    print(f"  │ DIFFERENT CONDITION (patient-invariant):                 │")
    print(f"  │   CKA(A_prehypo, B_normal):    {cka_pn_mean:.4f} ± {df['cka_prehypo_normal'].std():.4f}        │")
    print(f"  │   CKA(A_normal,  B_prehypo):   {cka_np_mean:.4f} ± {df['cka_normal_prehypo'].std():.4f}        │")
    print(f"  │   Average diff-condition:       {diff_mean:.4f}                │")
    print(f"  │                                                         │")
    print(f"  │ RATIO (same / diff):            {same_mean / diff_mean:.4f}                │")
    print(f"  │ DELTA (same - diff):           {same_mean - diff_mean:+.4f}                │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # Paired statistical test
    deltas = df["same_condition_mean"] - df["diff_condition_mean"]
    n_positive = (deltas > 0).sum()
    print(f"\n  Paired comparison (per patient pair):")
    print(f"    Pairs where same > diff: {n_positive}/{len(df)} "
          f"({100 * n_positive / len(df):.1f}%)")
    print(f"    Mean delta: {deltas.mean():+.4f} ± {deltas.std():.4f}")

    # Effect size
    if deltas.std() > 0:
        cohens_d = deltas.mean() / deltas.std()
        print(f"    Cohen's d (paired): {cohens_d:.3f}")

    if same_mean > diff_mean * 1.05:
        print(f"\n  ✓ CONFIRMED: Same-condition windows share more structure")
        print(f"    across patients than different-condition windows.")
        print(f"    The encoder creates condition-specific relational patterns")
        print(f"    that generalize across patients (patient-invariant).")
    else:
        print(f"\n  ✗ NOT CONFIRMED: Same-condition ≈ different-condition.")

    return results


def analysis_11_controlled_cross_patient_hemo(embeddings, patient_ids,
                                              max_windows, rng, model_name):
    """Controlled cross-patient CKA: same-condition vs different-condition
    for hemodynamic clusters (window-level alignment).

    For each pair of patients (A, B) who have windows in overlapping clusters:
      - CKA(A_cluster_i, B_cluster_i)  = same cluster, different patient
      - CKA(A_cluster_i, B_cluster_j)  = different cluster, different patient

    Uses window-level cluster assignments from time-aligned icuDataExtraction.
    """
    print(f"\n{'═' * 70}")
    print(f"Analysis 11: Controlled cross-patient CKA — Hemodynamic clusters ({model_name})")
    print(f"{'═' * 70}")
    print(f"  For each patient pair (A, B) with windows in shared clusters:")
    print(f"    Same cluster:  CKA(A_cluster_i, B_cluster_i)")
    print(f"    Diff cluster:  CKA(A_cluster_i, B_cluster_j)")

    # Load window-level hemodynamic clusters
    hemo_data = np.load(CLUSTERING_DIR / "window_hemo_clusters.npz",
                        allow_pickle=True)
    hemo_clusters = hemo_data["hemo_clusters"]
    hemo_pids = hemo_data["patient_ids"]

    # Verify alignment with embeddings
    assert len(hemo_clusters) == len(embeddings), \
        "Hemo clusters array doesn't match embeddings length"

    valid = hemo_clusters >= 0
    print(f"\n  Windows with cluster labels: {valid.sum()}/{len(hemo_clusters)} "
          f"({100 * valid.mean():.1f}%)")

    # Build per-patient cluster → window index mapping
    unique_pids = np.unique(patient_ids[valid])
    patient_cluster_windows = {}  # {pid: {cluster: [indices]}}

    for pid in unique_pids:
        mask = (patient_ids == pid) & valid
        indices = np.where(mask)[0]
        clusters_for_pid = hemo_clusters[indices]

        cluster_map = {}
        for c in np.unique(clusters_for_pid):
            c_indices = indices[clusters_for_pid == c]
            if len(c_indices) >= 5:  # minimum windows per cluster
                cluster_map[int(c)] = c_indices
        if len(cluster_map) >= 2:  # need at least 2 clusters
            patient_cluster_windows[str(pid)] = cluster_map

    print(f"  Patients with >=2 clusters (>=5 windows each): "
          f"{len(patient_cluster_windows)}")

    if len(patient_cluster_windows) < 5:
        print("  Too few patients. Skipping.")
        return []

    # Find patient pairs that share at least one cluster
    eligible_pids = list(patient_cluster_windows.keys())
    from itertools import combinations
    all_pairs = list(combinations(range(len(eligible_pids)), 2))

    max_pairs = 300
    if len(all_pairs) > max_pairs:
        pair_idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        selected_pairs = [all_pairs[i] for i in pair_idx]
    else:
        selected_pairs = all_pairs

    print(f"  Evaluating {len(selected_pairs)} patient pairs...")

    results = []
    skipped = 0

    for count, (idx_a, idx_b) in enumerate(selected_pairs):
        pid_a = eligible_pids[idx_a]
        pid_b = eligible_pids[idx_b]

        clusters_a = patient_cluster_windows[pid_a]
        clusters_b = patient_cluster_windows[pid_b]

        # Find shared clusters
        shared_clusters = set(clusters_a.keys()) & set(clusters_b.keys())
        all_clusters_a = set(clusters_a.keys())
        all_clusters_b = set(clusters_b.keys())

        if len(shared_clusters) < 1:
            skipped += 1
            continue

        # Same-cluster CKA: for each shared cluster, compare A's windows to B's
        same_cluster_ckas = []
        for c in shared_clusters:
            idx_ac = clusters_a[c]
            idx_bc = clusters_b[c]

            emb_ac = embeddings[idx_ac]
            emb_bc = embeddings[idx_bc]

            # Subsample
            cap = max_windows
            if len(emb_ac) > cap:
                emb_ac = emb_ac[rng.choice(len(emb_ac), cap, replace=False)]
            if len(emb_bc) > cap:
                emb_bc = emb_bc[rng.choice(len(emb_bc), cap, replace=False)]

            n_common = min(len(emb_ac), len(emb_bc))
            if n_common < 5:
                continue

            score = cka_cross(emb_ac, emb_bc, n_common=n_common,
                              kernel="linear", rng=rng)
            same_cluster_ckas.append(score)

        # Different-cluster CKA: compare A's cluster_i to B's cluster_j (i != j)
        diff_cluster_ckas = []
        # Get all cross-cluster pairs between A and B
        cross_pairs = []
        for ca in clusters_a.keys():
            for cb in clusters_b.keys():
                if ca != cb:
                    cross_pairs.append((ca, cb))

        # Subsample cross pairs if too many
        if len(cross_pairs) > 10:
            cp_idx = rng.choice(len(cross_pairs), size=10, replace=False)
            cross_pairs = [cross_pairs[i] for i in cp_idx]

        for ca, cb in cross_pairs:
            idx_ac = clusters_a[ca]
            idx_bc = clusters_b[cb]

            emb_ac = embeddings[idx_ac]
            emb_bc = embeddings[idx_bc]

            cap = max_windows
            if len(emb_ac) > cap:
                emb_ac = emb_ac[rng.choice(len(emb_ac), cap, replace=False)]
            if len(emb_bc) > cap:
                emb_bc = emb_bc[rng.choice(len(emb_bc), cap, replace=False)]

            n_common = min(len(emb_ac), len(emb_bc))
            if n_common < 5:
                continue

            score = cka_cross(emb_ac, emb_bc, n_common=n_common,
                              kernel="linear", rng=rng)
            diff_cluster_ckas.append(score)

        if same_cluster_ckas and diff_cluster_ckas:
            same_mean_pair = np.mean(same_cluster_ckas)
            diff_mean_pair = np.mean(diff_cluster_ckas)

            results.append({
                "model": model_name,
                "patient_a": pid_a,
                "patient_b": pid_b,
                "n_shared_clusters": len(shared_clusters),
                "same_cluster_cka": same_mean_pair,
                "diff_cluster_cka": diff_mean_pair,
                "n_same_comparisons": len(same_cluster_ckas),
                "n_diff_comparisons": len(diff_cluster_ckas),
            })

        if (count + 1) % 50 == 0:
            print(f"    {count + 1}/{len(selected_pairs)} pairs computed")

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No valid pairs with shared clusters.")
        return []

    print(f"\n  Valid pairs: {len(df)} (skipped {skipped} with no shared clusters)")

    same_mean = df["same_cluster_cka"].mean()
    diff_mean = df["diff_cluster_cka"].mean()

    print(f"\n  Results ({model_name}, {len(df)} pairs):")
    print(f"  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ SAME CLUSTER, DIFFERENT PATIENT:                        │")
    print(f"  │   Mean CKA:  {same_mean:.4f} ± {df['same_cluster_cka'].std():.4f}                       │")
    print(f"  │                                                         │")
    print(f"  │ DIFFERENT CLUSTER, DIFFERENT PATIENT:                    │")
    print(f"  │   Mean CKA:  {diff_mean:.4f} ± {df['diff_cluster_cka'].std():.4f}                       │")
    print(f"  │                                                         │")
    print(f"  │ RATIO (same / diff):  {same_mean / diff_mean:.4f}                          │")
    print(f"  │ DELTA (same - diff): {same_mean - diff_mean:+.4f}                          │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    # Paired test
    deltas = df["same_cluster_cka"] - df["diff_cluster_cka"]
    n_positive = (deltas > 0).sum()
    print(f"\n  Paired comparison (per patient pair):")
    print(f"    Pairs where same > diff: {n_positive}/{len(df)} "
          f"({100 * n_positive / len(df):.1f}%)")
    print(f"    Mean delta: {deltas.mean():+.4f} ± {deltas.std():.4f}")

    if deltas.std() > 0:
        cohens_d = deltas.mean() / deltas.std()
        print(f"    Cohen's d (paired): {cohens_d:.3f}")

    if same_mean > diff_mean * 1.05:
        print(f"\n  ✓ CONFIRMED: Same-cluster windows share more structure")
        print(f"    across patients than different-cluster windows.")
        print(f"    The encoder creates cluster-specific relational patterns")
        print(f"    that generalize across patients (patient-invariant).")
    else:
        print(f"\n  ✗ NOT CONFIRMED: Same-cluster ≈ different-cluster.")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings_path}")
    data = np.load(args.embeddings_path, allow_pickle=True)
    jepa_emb = data["jepa_embeddings"]
    ptst_emb = data["ptst_embeddings"]
    labels = data["labels"]
    patient_ids = data["patient_ids"]

    n_patients = len(np.unique(patient_ids))
    print(f"  {jepa_emb.shape[0]} windows, {n_patients} patients, "
          f"dim={jepa_emb.shape[1]}")
    print(f"  Hypotension prevalence: {labels.mean():.3f}")

    # Load hemodynamic clusters
    patient_cluster = load_hemodynamic_clusters()
    mapped = sum(1 for pid in np.unique(patient_ids)
                 if str(pid) in patient_cluster)
    print(f"  Hemodynamic mapping: {mapped}/{n_patients} patients")

    all_results = {}

    # ── Analysis 1: Cross-patient phenotype geometry ──────────────────────
    pheno_results = []
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = analysis_1_phenotype_geometry(
            emb, patient_ids, patient_cluster,
            args.n_windows_per_patient, args.min_windows, rng, model_name
        )
        pheno_results.extend(res)

    if pheno_results:
        df1 = pd.DataFrame(pheno_results)
        out1 = CLUSTERING_DIR / "cka_phenotype_pairs.csv"
        df1.to_csv(out1, index=False)
        all_results["phenotype_pairs"] = df1
        print(f"\n  Saved: {out1}")

    # ── Analysis 2: JEPA vs PatchTST agreement ────────────────────────────
    model_cmp_results = analysis_2_model_comparison(
        jepa_emb, ptst_emb, patient_ids, patient_cluster,
        args.n_windows_per_patient, args.min_windows, rng
    )
    if model_cmp_results:
        df2 = pd.DataFrame(model_cmp_results)
        out2 = CLUSTERING_DIR / "cka_model_comparison.csv"
        df2.to_csv(out2, index=False)
        all_results["model_comparison"] = df2
        print(f"\n  Saved: {out2}")

    # ── Analysis 3: Within-patient state geometry ─────────────────────────
    state_results = []
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = analysis_3_within_patient_states(
            emb, labels, patient_ids,
            args.n_windows_per_patient, rng, model_name
        )
        state_results.extend(res)

    if state_results:
        df3 = pd.DataFrame(state_results)
        out3 = CLUSTERING_DIR / "cka_within_patient_states.csv"
        df3.to_csv(out3, index=False)
        all_results["within_patient_states"] = df3
        print(f"\n  Saved: {out3}")

    # ── Analysis 4: Group-level CKA ───────────────────────────────────────
    group_results = []
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = analysis_4_group_level_cka(
            emb, patient_ids, patient_cluster,
            args.n_windows_per_patient, rng, model_name
        )
        group_results.extend(res)

    if group_results:
        df4 = pd.DataFrame(group_results)
        out4 = CLUSTERING_DIR / "cka_group_level.csv"
        df4.to_csv(out4, index=False)
        all_results["group_level"] = df4
        print(f"\n  Saved: {out4}")

    # ── Analysis 6: Temporal segment CKA ──────────────────────────────────
    temporal_results = []
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = analysis_6_temporal_segments(
            emb, labels, patient_ids, args.min_windows, rng, model_name
        )
        temporal_results.extend(res)

    if temporal_results:
        df6 = pd.DataFrame(temporal_results)
        out6 = CLUSTERING_DIR / "cka_temporal_segments.csv"
        df6.to_csv(out6, index=False)
        all_results["temporal_segments"] = df6
        print(f"\n  Saved: {out6}")

    # ── Analysis 7: Clinical similarity strata ────────────────────────────
    clinical_results = []
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = analysis_7_clinical_similarity(
            emb, patient_ids, patient_cluster,
            args.min_windows, rng, model_name
        )
        clinical_results.extend(res)

    if clinical_results:
        df7 = pd.DataFrame(clinical_results)
        out7 = CLUSTERING_DIR / "cka_clinical_similarity.csv"
        df7.to_csv(out7, index=False)
        all_results["clinical_similarity"] = df7
        print(f"\n  Saved: {out7}")

    # ── Analysis 8: Cross-patient pre-hypo vs normal CKA ──────────────────
    cross_hypo_results = []
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = analysis_8_cross_patient_hypo(
            emb, labels, patient_ids, args.min_windows, rng, model_name
        )
        cross_hypo_results.extend(res)

    if cross_hypo_results:
        df8 = pd.DataFrame(cross_hypo_results)
        out8 = CLUSTERING_DIR / "cka_cross_patient_hypo.csv"
        df8.to_csv(out8, index=False)
        all_results["cross_patient_hypo"] = df8
        print(f"\n  Saved: {out8}")

    # ── Analysis 9: Within-patient hypo geometry consistency ───────────────
    within_hypo_results = []
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = analysis_9_within_patient_hypo_geometry(
            emb, labels, patient_ids,
            args.n_windows_per_patient, rng, model_name
        )
        within_hypo_results.extend(res)

    if within_hypo_results:
        df9 = pd.DataFrame(within_hypo_results)
        out9 = CLUSTERING_DIR / "cka_within_patient_hypo_geometry.csv"
        df9.to_csv(out9, index=False)
        all_results["within_patient_hypo_geometry"] = df9
        print(f"\n  Saved: {out9}")

    # ── Analysis 5: Same-patient vs different-patient CKA ─────────────────
    patient_cmp_results = []
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = analysis_5_same_vs_diff_patient(
            emb, patient_ids,
            args.n_windows_per_patient, args.min_windows, rng, model_name
        )
        patient_cmp_results.append(res)

    if patient_cmp_results:
        df5 = pd.DataFrame(patient_cmp_results)
        out5 = CLUSTERING_DIR / "cka_same_vs_diff_patient.csv"
        df5.to_csv(out5, index=False)
        all_results["same_vs_diff_patient"] = df5
        print(f"\n  Saved: {out5}")

    # ── Analysis 10: Controlled cross-patient CKA for hypotension ─────────
    controlled_hypo_results = []
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = analysis_10_controlled_cross_patient_hypo(
            emb, labels, patient_ids,
            args.n_windows_per_patient, rng, model_name
        )
        controlled_hypo_results.extend(res)

    if controlled_hypo_results:
        df10 = pd.DataFrame(controlled_hypo_results)
        out10 = CLUSTERING_DIR / "cka_controlled_hypo.csv"
        df10.to_csv(out10, index=False)
        all_results["controlled_hypo"] = df10
        print(f"\n  Saved: {out10}")

    # ── Analysis 11: Controlled cross-patient CKA for hemo clusters ───────
    controlled_hemo_results = []
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = analysis_11_controlled_cross_patient_hemo(
            emb, patient_ids,
            args.n_windows_per_patient, rng, model_name
        )
        controlled_hemo_results.extend(res)

    if controlled_hemo_results:
        df11 = pd.DataFrame(controlled_hemo_results)
        out11 = CLUSTERING_DIR / "cka_controlled_hemo.csv"
        df11.to_csv(out11, index=False)
        all_results["controlled_hemo"] = df11
        print(f"\n  Saved: {out11}")

    # ── Analysis 12: Within-patient hemodynamic cluster CKA ───────────────
    within_hemo_results = []
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = analysis_12_within_patient_hemo_clusters(
            emb, patient_ids,
            args.n_windows_per_patient, rng, model_name
        )
        within_hemo_results.extend(res)

    if within_hemo_results:
        df12 = pd.DataFrame(within_hemo_results)
        out12 = CLUSTERING_DIR / "cka_within_patient_hemo.csv"
        df12.to_csv(out12, index=False)
        all_results["within_patient_hemo"] = df12
        print(f"\n  Saved: {out12}")

    # ── Final Summary ─────────────────────────────────────────────────────
    print(f"\n\n{'=' * 75}")
    print("CKA ANALYSIS SUMMARY")
    print(f"{'=' * 75}")

    if "phenotype_pairs" in all_results:
        df1 = all_results["phenotype_pairs"]
        print("\n1. Cross-patient representational similarity (CKA between patient pairs):")
        print(f"   Question: Do same-phenotype patients share internal geometry?")
        for model in ["JEPA", "PatchTST"]:
            mdf = df1[df1["model"] == model]
            same = mdf[mdf["same_phenotype"]]["cka_linear"].mean()
            diff = mdf[~mdf["same_phenotype"]]["cka_linear"].mean()
            print(f"   {model}: same_pheno={same:.4f}, diff_pheno={diff:.4f}, "
                  f"delta={same - diff:+.4f}")

    if "model_comparison" in all_results:
        df2 = all_results["model_comparison"]
        print(f"\n2. JEPA vs PatchTST agreement:")
        print(f"   Question: Do both models organize patients the same way?")
        print(f"   Mean CKA: {df2['cka_jepa_vs_ptst'].mean():.4f} "
              f"± {df2['cka_jepa_vs_ptst'].std():.4f}")

    if "within_patient_states" in all_results:
        df3 = all_results["within_patient_states"]
        print(f"\n3. Within-patient state geometry:")
        print(f"   Question: Do pre-hypo windows have different geometry than normal?")
        for model in ["JEPA", "PatchTST"]:
            mdf = df3[df3["model"] == model]
            pn = mdf["cka_pos_neg"].mean()
            nn = mdf["cka_neg_neg_baseline"].mean()
            print(f"   {model}: CKA(pos,neg)={pn:.4f}, "
                  f"CKA(neg,neg)={nn:.4f}, ratio={pn / nn:.4f}")

    if "group_level" in all_results:
        df4 = all_results["group_level"]
        print(f"\n4. Group-level CKA (pooled phenotype clusters):")
        print(f"   Question: Do hemodynamic clusters have distinct geometry?")
        for model in ["JEPA", "PatchTST"]:
            mdf = df4[df4["model"] == model]
            within = mdf[mdf["type"] == "within"]["cka_linear"].mean()
            between = mdf[mdf["type"] == "between"]["cka_linear"].mean()
            print(f"   {model}: within={within:.4f}, between={between:.4f}, "
                  f"ratio={between / within:.4f}")

    if "same_vs_diff_patient" in all_results:
        df5 = all_results["same_vs_diff_patient"]
        print(f"\n5. Same-patient vs different-patient CKA:")
        print(f"   Question: Is representational geometry patient-specific?")
        for _, r in df5.iterrows():
            print(f"   {r['model']}: same_patient={r['same_patient_cka_mean']:.4f}, "
                  f"diff_patient={r['diff_patient_cka_mean']:.4f}, "
                  f"ratio={r['ratio']:.2f}x")

    if "temporal_segments" in all_results:
        df6 = all_results["temporal_segments"]
        print(f"\n6. Temporal segment CKA (geometry drift over ICU stay):")
        print(f"   Question: Does representational geometry drift over time?")
        for model in ["JEPA", "PatchTST"]:
            mdf = df6[df6["model"] == model]
            el = mdf["cka_early_late"].mean()
            em = mdf["cka_early_middle"].mean()
            print(f"   {model}: early↔late={el:.4f}, early↔middle={em:.4f}")
            hypo = mdf[mdf["has_hypotension"]]
            stable = mdf[~mdf["has_hypotension"]]
            if len(hypo) > 0 and len(stable) > 0:
                print(f"     Hypo patients early↔late: {hypo['cka_early_late'].mean():.4f}, "
                      f"Stable: {stable['cka_early_late'].mean():.4f}")

    if "clinical_similarity" in all_results:
        df7 = all_results["clinical_similarity"]
        print(f"\n7. CKA by clinical similarity:")
        print(f"   Question: Do clinically similar patients share more geometry?")
        for model in ["JEPA", "PatchTST"]:
            mdf = df7[df7["model"] == model]
            corr = mdf["clinical_distance"].corr(mdf["cka_linear"])
            print(f"   {model}: correlation(distance, CKA) = {corr:.4f}")

    if "cross_patient_hypo" in all_results:
        df8 = all_results["cross_patient_hypo"]
        print(f"\n8. Cross-patient CKA: pre-hypo vs normal windows:")
        print(f"   Question: Is there shared pre-hypotensive geometry across patients?")
        for _, r in df8.iterrows():
            print(f"   {r['model']}: pre-hypo={r['hypo_cka_mean']:.4f}, "
                  f"normal={r['normal_cka_mean']:.4f}, "
                  f"delta={r['hypo_cka_mean'] - r['normal_cka_mean']:+.4f}")

    if "within_patient_hypo_geometry" in all_results:
        df9 = all_results["within_patient_hypo_geometry"]
        print(f"\n9. Within-patient hypo geometry consistency:")
        print(f"   Question: Do pre-hypo windows form more coherent geometry than normal?")
        for model in ["JEPA", "PatchTST"]:
            mdf = df9[df9["model"] == model]
            if len(mdf) > 0:
                pp = mdf["cka_pos_pos"].mean()
                nn = mdf["cka_neg_neg"].mean()
                pn = mdf["cka_pos_neg"].mean()
                print(f"   {model}: pos-pos={pp:.4f}, neg-neg={nn:.4f}, "
                      f"pos-neg={pn:.4f}, ratio(pp/nn)={pp/nn:.2f}")

    if "controlled_hypo" in all_results:
        df10 = all_results["controlled_hypo"]
        print(f"\n10. Controlled cross-patient CKA — Hypotension:")
        print(f"    (same condition across patients vs different condition)")
        for model in ["JEPA", "PatchTST"]:
            mdf = df10[df10["model"] == model]
            if len(mdf) > 0:
                same = mdf["same_condition_mean"].mean()
                diff = mdf["diff_condition_mean"].mean()
                ratio = same / diff if diff > 0 else float("nan")
                print(f"    {model}: same_cond={same:.4f}, diff_cond={diff:.4f}, "
                      f"ratio={ratio:.3f}, delta={same - diff:+.4f}")

    if "controlled_hemo" in all_results:
        df11 = all_results["controlled_hemo"]
        print(f"\n11. Controlled cross-patient CKA — Hemodynamic clusters:")
        print(f"    (same cluster across patients vs different cluster)")
        for model in ["JEPA", "PatchTST"]:
            mdf = df11[df11["model"] == model]
            if len(mdf) > 0:
                same = mdf["same_cluster_cka"].mean()
                diff = mdf["diff_cluster_cka"].mean()
                ratio = same / diff if diff > 0 else float("nan")
                print(f"    {model}: same_cluster={same:.4f}, diff_cluster={diff:.4f}, "
                      f"ratio={ratio:.3f}, delta={same - diff:+.4f}")

    if "within_patient_hemo" in all_results:
        df12 = all_results["within_patient_hemo"]
        print(f"\n12. Within-patient hemodynamic cluster CKA:")
        print(f"    (do different hemo states have different structure within a patient?)")
        for model in ["JEPA", "PatchTST"]:
            mdf = df12[df12["model"] == model]
            if len(mdf) > 0:
                within = mdf["within_cluster_cka"].mean()
                between = mdf["between_cluster_cka"].mean()
                ratio = between / within if within > 0 else float("nan")
                print(f"    {model}: within_cluster={within:.4f}, "
                      f"between_cluster={between:.4f}, "
                      f"ratio(between/within)={ratio:.3f}")

    print(f"\n{'=' * 75}")
    print(f"All results saved to: {CLUSTERING_DIR}/cka_*.csv")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
