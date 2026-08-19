"""
Linear probing analysis of cross-patient generalizability.

Comprehensive test of whether condition/cluster information in JEPA/PatchTST
embeddings is patient-invariant. Parallel to the CKA analysis but using
classification-based evidence.

Goal: Show whether windows with the same condition have similar representations
and windows with different conditions have different representations — and
whether this holds across patients (patient-invariant).

Analyses for EACH condition (hypotension + hemodynamic clusters):
  1. Cross-patient probe (raw): patient-disjoint train/test, logistic regression
  2. Cross-patient probe (de-meaned): subtract patient centroid, then probe
  3. Leave-one-patient-out (LOPO): train on N-1 patients, test on held-out
  4. Within-patient baseline: random split ignoring patient (upper bound)
  5. Permutation baseline: shuffle labels within each patient (lower bound)
  6. Cross-patient with varying train sizes: generalization curve
  7. Per-patient AUROC distribution: which patients generalize?

Window-level hemodynamic clusters (time-aligned) used where available.

Requires cached embeddings from cluster_embeddings.py.

Usage:
    python linear_probe_generalizability.py [--seed 42] [--n-lopo-patients 50]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
DERIVED_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
CLUSTERING_DIR = DERIVED_ROOT / "probing/clustering"
ICU_OUTPUT = Path("/gpfs/home/dk5565/icuDataExtraction/output_v2")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embeddings-path", type=str,
                        default=str(CLUSTERING_DIR / "embeddings_nfull_seed42.npz"))
    parser.add_argument("--n-lopo-patients", type=int, default=50,
                        help="Max patients for LOPO (0=all)")
    parser.add_argument("--n-repeats", type=int, default=5,
                        help="Repeats for cross-patient probe (different splits)")
    parser.add_argument("--skip-demean", action="store_true",
                        help="Skip all de-meaned experiments to save time")
    return parser.parse_args()


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_hemodynamic_clusters_patient_level() -> dict:
    """Load per-patient dominant hemodynamic cluster from icuDataExtraction."""
    cluster_labels = np.load(ICU_OUTPUT / "cluster_labels.npy")
    patient_ids = np.load(ICU_OUTPUT / "patient_ids.npy", allow_pickle=True)

    patient_cluster = {}
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        counts = np.bincount(cluster_labels[mask], minlength=7)
        patient_cluster[str(pid)] = int(np.argmax(counts))

    return patient_cluster


def load_hemodynamic_clusters_window_level():
    """Load window-level hemodynamic cluster labels (time-aligned)."""
    path = CLUSTERING_DIR / "window_hemo_clusters.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return data["hemo_clusters"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def demean_by_patient(embeddings, patient_ids):
    """Subtract each patient's mean embedding (centroid removal)."""
    demeaned = embeddings.copy()
    for pid in np.unique(patient_ids):
        mask = patient_ids == pid
        patient_mean = embeddings[mask].mean(axis=0)
        demeaned[mask] -= patient_mean
    return demeaned


def fit_and_evaluate(X_train, y_train, X_test, y_test, binary=True,
                     random_state=42):
    """Fit logistic regression and return metrics."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    kwargs = dict(max_iter=1000, C=1.0, solver="lbfgs", random_state=random_state)
    if not binary:
        kwargs["multi_class"] = "multinomial"

    clf = LogisticRegression(**kwargs)
    clf.fit(X_tr, y_train)

    if binary:
        y_prob = clf.predict_proba(X_te)[:, 1]
        try:
            auroc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auroc = np.nan
    else:
        y_prob = clf.predict_proba(X_te)
        try:
            auroc = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                  average="macro")
        except ValueError:
            auroc = np.nan

    bacc = balanced_accuracy_score(y_test, clf.predict(X_te))
    return auroc, bacc


# ── Hypotension Analysis ──────────────────────────────────────────────────────

def probe_hypotension(embeddings, labels, patient_ids, rng, model_name,
                      n_lopo_patients=50, n_repeats=5, skip_demean=False):
    """Comprehensive cross-patient probing for hypotension.

    Tests:
    1. Cross-patient probe (raw): patient-disjoint 80/20, repeated
    2. Cross-patient probe (de-meaned): same but with centroid removed
    3. LOPO: train on all-but-one, test on held-out patient
    4. Within-patient baseline: random split (upper bound, data leaks patient)
    5. Permutation baseline: shuffle labels within each patient
    6. Generalization curve: vary number of training patients
    7. Per-patient AUROC distribution from LOPO
    """
    print(f"\n{'═' * 75}")
    print(f"  HYPOTENSION — Linear Probe Generalizability ({model_name})")
    print(f"{'═' * 75}")

    unique_pids = np.unique(patient_ids)
    n_patients = len(unique_pids)
    if not skip_demean:
        emb_dm = demean_by_patient(embeddings, patient_ids)

    print(f"  {n_patients} patients, {len(labels)} windows, "
          f"prevalence={labels.mean():.3f}")

    results = {"model": model_name, "task": "hypotension"}

    # ── 1 & 2. Cross-patient probe (raw + de-meaned), repeated ────────────
    print(f"\n  1. Cross-patient probe ({n_repeats} random patient splits):")

    aurocs_raw, aurocs_dm = [], []
    for rep in range(n_repeats):
        perm = rng.permutation(n_patients)
        n_train = int(0.8 * n_patients)
        train_pids = set(unique_pids[perm[:n_train]])

        train_mask = np.array([pid in train_pids for pid in patient_ids])
        test_mask = ~train_mask

        # Raw
        auroc, _ = fit_and_evaluate(
            embeddings[train_mask], labels[train_mask],
            embeddings[test_mask], labels[test_mask],
            binary=True, random_state=rep)
        aurocs_raw.append(auroc)

        # De-meaned
        if not skip_demean:
            auroc_dm, _ = fit_and_evaluate(
                emb_dm[train_mask], labels[train_mask],
                emb_dm[test_mask], labels[test_mask],
                binary=True, random_state=rep)
            aurocs_dm.append(auroc_dm)

    results["cross_patient_raw_auroc"] = np.mean(aurocs_raw)
    results["cross_patient_raw_auroc_std"] = np.std(aurocs_raw)
    if not skip_demean:
        results["cross_patient_demeaned_auroc"] = np.mean(aurocs_dm)
        results["cross_patient_demeaned_auroc_std"] = np.std(aurocs_dm)

    print(f"     Raw:      AUROC {np.mean(aurocs_raw):.4f} ± {np.std(aurocs_raw):.4f}")
    if not skip_demean:
        print(f"     De-meaned: AUROC {np.mean(aurocs_dm):.4f} ± {np.std(aurocs_dm):.4f}")

    # ── 3. Leave-one-patient-out ──────────────────────────────────────────
    print(f"\n  3. Leave-one-patient-out (LOPO):")

    # Select patients with enough of both classes
    lopo_eligible = []
    for pid in unique_pids:
        mask = patient_ids == pid
        if labels[mask].sum() >= 3 and (labels[mask] == 0).sum() >= 3:
            lopo_eligible.append(pid)

    if len(lopo_eligible) > n_lopo_patients and n_lopo_patients > 0:
        lopo_pids = rng.choice(lopo_eligible, size=n_lopo_patients, replace=False)
    else:
        lopo_pids = np.array(lopo_eligible)

    print(f"     Eligible: {len(lopo_eligible)}, evaluating: {len(lopo_pids)}")

    lopo_aurocs_raw, lopo_aurocs_dm = [], []
    for pid in lopo_pids:
        test_m = patient_ids == pid
        train_m = ~test_m
        y_te = labels[test_m]

        if len(np.unique(y_te)) < 2:
            continue

        # Raw
        auroc, _ = fit_and_evaluate(
            embeddings[train_m], labels[train_m],
            embeddings[test_m], y_te, binary=True)
        lopo_aurocs_raw.append(auroc)

        # De-meaned (train de-meaned by their patients, test de-meaned by own mean)
        if not skip_demean:
            emb_dm_train = demean_by_patient(embeddings[train_m], patient_ids[train_m])
            test_emb_dm = embeddings[test_m] - embeddings[test_m].mean(axis=0)
            auroc_dm, _ = fit_and_evaluate(
                emb_dm_train, labels[train_m],
                test_emb_dm, y_te, binary=True)
            lopo_aurocs_dm.append(auroc_dm)

    results["lopo_raw_auroc_mean"] = np.mean(lopo_aurocs_raw)
    results["lopo_raw_auroc_std"] = np.std(lopo_aurocs_raw)
    results["lopo_raw_auroc_median"] = np.median(lopo_aurocs_raw)
    if not skip_demean:
        results["lopo_demeaned_auroc_mean"] = np.mean(lopo_aurocs_dm)
        results["lopo_demeaned_auroc_std"] = np.std(lopo_aurocs_dm)
    results["lopo_n_patients"] = len(lopo_aurocs_raw)

    print(f"     Raw:       AUROC {np.mean(lopo_aurocs_raw):.4f} ± "
          f"{np.std(lopo_aurocs_raw):.4f} (median={np.median(lopo_aurocs_raw):.4f})")
    if not skip_demean:
        print(f"     De-meaned: AUROC {np.mean(lopo_aurocs_dm):.4f} ± "
              f"{np.std(lopo_aurocs_dm):.4f}")

    # ── 4. Within-patient baseline ────────────────────────────────────────
    print(f"\n  4. Within-patient baseline (random split, patient leakage):")
    n_total = len(labels)
    perm_all = rng.permutation(n_total)
    n_tr = int(0.8 * n_total)

    auroc_wp, bacc_wp = fit_and_evaluate(
        embeddings[perm_all[:n_tr]], labels[perm_all[:n_tr]],
        embeddings[perm_all[n_tr:]], labels[perm_all[n_tr:]],
        binary=True)
    results["within_patient_auroc"] = auroc_wp
    print(f"     AUROC: {auroc_wp:.4f}")

    # ── 5. Permutation baseline ───────────────────────────────────────────
    print(f"\n  5. Permutation baseline (labels shuffled within each patient):")
    labels_perm = labels.copy()
    for pid in unique_pids:
        mask = patient_ids == pid
        labels_perm[mask] = rng.permutation(labels_perm[mask])

    # Use same patient split as first repeat
    perm0 = rng.permutation(n_patients)
    n_train = int(0.8 * n_patients)
    train_pids_perm = set(unique_pids[perm0[:n_train]])
    train_mask_perm = np.array([pid in train_pids_perm for pid in patient_ids])
    test_mask_perm = ~train_mask_perm

    auroc_perm, _ = fit_and_evaluate(
        embeddings[train_mask_perm], labels_perm[train_mask_perm],
        embeddings[test_mask_perm], labels_perm[test_mask_perm],
        binary=True)
    results["permutation_auroc"] = auroc_perm
    print(f"     AUROC: {auroc_perm:.4f} (should be ~0.50)")

    # ── 6. Generalization curve (vary training patients) ──────────────────
    print(f"\n  6. Generalization curve (AUROC vs N training patients):")
    patient_fracs = [0.1, 0.2, 0.4, 0.6, 0.8]
    gen_curve = []
    for frac in patient_fracs:
        n_tr_patients = max(5, int(frac * n_patients))
        perm_gc = rng.permutation(n_patients)
        train_pids_gc = set(unique_pids[perm_gc[:n_tr_patients]])
        train_m_gc = np.array([pid in train_pids_gc for pid in patient_ids])
        test_m_gc = ~train_m_gc

        if labels[test_m_gc].sum() < 3 or (labels[test_m_gc] == 0).sum() < 3:
            continue

        auroc_gc, _ = fit_and_evaluate(
            embeddings[train_m_gc], labels[train_m_gc],
            embeddings[test_m_gc], labels[test_m_gc],
            binary=True)
        gen_curve.append({"frac": frac, "n_patients": n_tr_patients,
                          "auroc": auroc_gc})
        print(f"     {frac:.0%} ({n_tr_patients} patients): AUROC={auroc_gc:.4f}")

    results["gen_curve"] = gen_curve

    # ── 7. Per-patient AUROC distribution ─────────────────────────────────
    print(f"\n  7. Per-patient AUROC distribution (from LOPO):")
    if lopo_aurocs_raw:
        arr = np.array(lopo_aurocs_raw)
        above_06 = (arr > 0.6).sum()
        above_07 = (arr > 0.7).sum()
        above_08 = (arr > 0.8).sum()
        below_055 = (arr < 0.55).sum()
        print(f"     AUROC > 0.8: {above_08}/{len(arr)} patients ({100*above_08/len(arr):.0f}%)")
        print(f"     AUROC > 0.7: {above_07}/{len(arr)} ({100*above_07/len(arr):.0f}%)")
        print(f"     AUROC > 0.6: {above_06}/{len(arr)} ({100*above_06/len(arr):.0f}%)")
        print(f"     AUROC < 0.55 (near-chance): {below_055}/{len(arr)} ({100*below_055/len(arr):.0f}%)")
        results["lopo_pct_above_07"] = above_07 / len(arr)
        results["lopo_pct_below_055"] = below_055 / len(arr)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │ HYPOTENSION SUMMARY ({model_name})                              │")
    print(f"  │                                                             │")
    print(f"  │ Within-patient (data leakage):     AUROC {auroc_wp:.4f}          │")
    print(f"  │ Cross-patient (raw, mean±std):     AUROC {np.mean(aurocs_raw):.4f}±{np.std(aurocs_raw):.3f}  │")
    print(f"  │ Cross-patient (de-meaned):         AUROC {np.mean(aurocs_dm):.4f}±{np.std(aurocs_dm):.3f}  │")
    print(f"  │ LOPO (raw, mean±std):              AUROC {np.mean(lopo_aurocs_raw):.4f}±{np.std(lopo_aurocs_raw):.3f}  │")
    print(f"  │ LOPO (de-meaned):                  AUROC {np.mean(lopo_aurocs_dm):.4f}±{np.std(lopo_aurocs_dm):.3f}  │")
    print(f"  │ Permutation (null):                AUROC {auroc_perm:.4f}          │")
    print(f"  │ Chance:                            AUROC 0.5000          │")
    print(f"  └─────────────────────────────────────────────────────────────┘")

    return results


# ── Hemodynamic Cluster Analysis ──────────────────────────────────────────────

def probe_hemodynamic_clusters(embeddings, patient_ids, patient_cluster,
                               window_hemo_clusters, rng, model_name,
                               n_lopo_patients=30, n_repeats=5,
                               skip_demean=False):
    """Comprehensive cross-patient probing for hemodynamic clusters.

    Uses BOTH patient-level labels (dominant cluster) and window-level labels
    (time-aligned) to give the most complete picture.

    Same test battery as hypotension but multiclass (7 clusters).
    """
    print(f"\n{'═' * 75}")
    print(f"  HEMODYNAMIC CLUSTERS — Linear Probe Generalizability ({model_name})")
    print(f"{'═' * 75}")

    # ── Patient-level labels ──────────────────────────────────────────────
    hemo_labels = np.array([patient_cluster.get(str(pid), -1)
                           for pid in patient_ids])
    valid_patient = hemo_labels >= 0

    # ── Window-level labels (more precise, time-aligned) ──────────────────
    if window_hemo_clusters is not None:
        valid_window = window_hemo_clusters >= 0
    else:
        valid_window = np.zeros(len(patient_ids), dtype=bool)

    print(f"  Patient-level labels: {valid_patient.sum()} windows "
          f"({len(np.unique(patient_ids[valid_patient]))} patients)")
    print(f"  Window-level labels:  {valid_window.sum()} windows "
          f"(time-aligned, more precise)")

    all_results = []

    # Run on patient-level labels
    res_patient = _probe_hemo_subset(
        embeddings[valid_patient], hemo_labels[valid_patient],
        patient_ids[valid_patient], rng, model_name,
        label_type="patient_level", n_lopo_patients=n_lopo_patients,
        n_repeats=n_repeats, skip_demean=skip_demean)
    all_results.append(res_patient)

    # Run on window-level labels if available
    if valid_window.sum() > 1000:
        res_window = _probe_hemo_subset(
            embeddings[valid_window], window_hemo_clusters[valid_window],
            patient_ids[valid_window], rng, model_name,
            label_type="window_level", n_lopo_patients=n_lopo_patients,
            n_repeats=n_repeats, skip_demean=skip_demean)
        all_results.append(res_window)

    return all_results


def _probe_hemo_subset(embeddings, labels, patient_ids, rng, model_name,
                       label_type, n_lopo_patients=30, n_repeats=5,
                       skip_demean=False):
    """Inner function: probe hemodynamic clusters on a labeled subset."""
    print(f"\n  ── {label_type} labels ──")

    unique_pids = np.unique(patient_ids)
    n_patients = len(unique_pids)
    n_classes = len(np.unique(labels))
    chance = 1.0 / n_classes
    if not skip_demean:
        emb_dm = demean_by_patient(embeddings, patient_ids)

    print(f"     {n_patients} patients, {len(labels)} windows, "
          f"{n_classes} classes")
    print(f"     Distribution: {np.bincount(labels.astype(int), minlength=7)[:n_classes]}")

    results = {"model": model_name, "task": f"hemo_{label_type}",
               "n_classes": n_classes, "chance": chance}

    # ── 1. Cross-patient probe (raw + de-meaned) ────────────────────────
    print(f"\n     1. Cross-patient probe ({n_repeats} splits):")

    baccs_raw, baccs_dm, aurocs_raw, aurocs_dm = [], [], [], []
    for rep in range(n_repeats):
        perm = rng.permutation(n_patients)
        n_train = int(0.8 * n_patients)
        train_pids = set(unique_pids[perm[:n_train]])
        train_mask = np.array([pid in train_pids for pid in patient_ids])
        test_mask = ~train_mask

        if len(np.unique(labels[test_mask])) < 2:
            continue

        auroc, bacc = fit_and_evaluate(
            embeddings[train_mask], labels[train_mask],
            embeddings[test_mask], labels[test_mask],
            binary=False, random_state=rep)
        baccs_raw.append(bacc)
        aurocs_raw.append(auroc)

        if not skip_demean:
            auroc_dm, bacc_dm = fit_and_evaluate(
                emb_dm[train_mask], labels[train_mask],
                emb_dm[test_mask], labels[test_mask],
                binary=False, random_state=rep)
            baccs_dm.append(bacc_dm)
            aurocs_dm.append(auroc_dm)

    results["cross_patient_raw_bacc"] = np.mean(baccs_raw)
    results["cross_patient_raw_bacc_std"] = np.std(baccs_raw)
    results["cross_patient_raw_auroc"] = np.mean(aurocs_raw)
    if not skip_demean:
        results["cross_patient_demeaned_bacc"] = np.mean(baccs_dm)
        results["cross_patient_demeaned_bacc_std"] = np.std(baccs_dm)
        results["cross_patient_demeaned_auroc"] = np.mean(aurocs_dm)

    print(f"        Raw:       Bal.Acc {np.mean(baccs_raw):.4f}±{np.std(baccs_raw):.4f}, "
          f"AUROC {np.mean(aurocs_raw):.4f}")
    if not skip_demean:
        print(f"        De-meaned: Bal.Acc {np.mean(baccs_dm):.4f}±{np.std(baccs_dm):.4f}, "
              f"AUROC {np.mean(aurocs_dm):.4f}")

    # ── 3. LOPO ───────────────────────────────────────────────────────────
    print(f"\n     3. Leave-one-patient-out:")

    lopo_pids = unique_pids
    if len(lopo_pids) > n_lopo_patients and n_lopo_patients > 0:
        lopo_pids = rng.choice(lopo_pids, size=n_lopo_patients, replace=False)

    lopo_baccs_raw, lopo_baccs_dm = [], []
    for pid in lopo_pids:
        test_m = patient_ids == pid
        train_m = ~test_m

        if len(np.unique(labels[test_m])) < 1 or len(labels[test_m]) < 3:
            continue

        _, bacc = fit_and_evaluate(
            embeddings[train_m], labels[train_m],
            embeddings[test_m], labels[test_m],
            binary=False)
        lopo_baccs_raw.append(bacc)

        # De-meaned
        if not skip_demean:
            emb_dm_tr = demean_by_patient(embeddings[train_m], patient_ids[train_m])
            test_emb_dm = embeddings[test_m] - embeddings[test_m].mean(axis=0)
            _, bacc_dm = fit_and_evaluate(
                emb_dm_tr, labels[train_m],
                test_emb_dm, labels[test_m],
                binary=False)
            lopo_baccs_dm.append(bacc_dm)

    results["lopo_raw_bacc_mean"] = np.mean(lopo_baccs_raw)
    results["lopo_raw_bacc_std"] = np.std(lopo_baccs_raw)
    if not skip_demean:
        results["lopo_demeaned_bacc_mean"] = np.mean(lopo_baccs_dm)
        results["lopo_demeaned_bacc_std"] = np.std(lopo_baccs_dm)
    results["lopo_n_patients"] = len(lopo_baccs_raw)

    print(f"        Raw:       Bal.Acc {np.mean(lopo_baccs_raw):.4f}±{np.std(lopo_baccs_raw):.4f}")
    if not skip_demean:
        print(f"        De-meaned: Bal.Acc {np.mean(lopo_baccs_dm):.4f}±{np.std(lopo_baccs_dm):.4f}")
    print(f"        Chance: {chance:.4f}")

    # ── 4. Within-patient baseline ────────────────────────────────────────
    print(f"\n     4. Within-patient baseline:")
    n_total = len(labels)
    perm_all = rng.permutation(n_total)
    n_tr = int(0.8 * n_total)

    _, bacc_wp = fit_and_evaluate(
        embeddings[perm_all[:n_tr]], labels[perm_all[:n_tr]],
        embeddings[perm_all[n_tr:]], labels[perm_all[n_tr:]],
        binary=False)
    results["within_patient_bacc"] = bacc_wp
    print(f"        Bal.Acc: {bacc_wp:.4f}")

    # ── 5. Permutation baseline ───────────────────────────────────────────
    print(f"\n     5. Permutation baseline:")
    labels_perm = labels.copy()
    for pid in unique_pids:
        mask = patient_ids == pid
        labels_perm[mask] = rng.permutation(labels_perm[mask])

    perm0 = rng.permutation(n_patients)
    n_train = int(0.8 * n_patients)
    train_pids_p = set(unique_pids[perm0[:n_train]])
    train_m_p = np.array([pid in train_pids_p for pid in patient_ids])
    test_m_p = ~train_m_p

    _, bacc_perm = fit_and_evaluate(
        embeddings[train_m_p], labels_perm[train_m_p],
        embeddings[test_m_p], labels_perm[test_m_p],
        binary=False)
    results["permutation_bacc"] = bacc_perm
    print(f"        Bal.Acc: {bacc_perm:.4f} (should be ~{chance:.3f})")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n     ┌─────────────────────────────────────────────────────────┐")
    print(f"     │ HEMO {label_type:<14s} ({model_name})                      │")
    print(f"     │ Within-patient:       Bal.Acc {bacc_wp:.4f}               │")
    print(f"     │ Cross-patient (raw):  Bal.Acc {np.mean(baccs_raw):.4f}               │")
    print(f"     │ Cross-patient (de-m): Bal.Acc {np.mean(baccs_dm):.4f}               │")
    print(f"     │ LOPO (raw):           Bal.Acc {np.mean(lopo_baccs_raw):.4f}               │")
    print(f"     │ LOPO (de-meaned):     Bal.Acc {np.mean(lopo_baccs_dm):.4f}               │")
    print(f"     │ Permutation:          Bal.Acc {bacc_perm:.4f}               │")
    print(f"     │ Chance (1/{n_classes}):          Bal.Acc {chance:.4f}               │")
    print(f"     └─────────────────────────────────────────────────────────┘")

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

    print(f"  {jepa_emb.shape[0]} windows, {len(np.unique(patient_ids))} patients, "
          f"dim={jepa_emb.shape[1]}")
    print(f"  Hypotension prevalence: {labels.mean():.3f}")

    # Load hemodynamic clusters
    patient_cluster = load_hemodynamic_clusters_patient_level()
    window_hemo = load_hemodynamic_clusters_window_level()
    mapped = sum(1 for pid in np.unique(patient_ids)
                 if str(pid) in patient_cluster)
    print(f"  Hemodynamic mapping: {mapped}/{len(np.unique(patient_ids))} patients")
    if window_hemo is not None:
        print(f"  Window-level hemo labels: {(window_hemo >= 0).sum()} windows")

    all_results = []

    # ── Hypotension ───────────────────────────────────────────────────────
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res = probe_hypotension(emb, labels, patient_ids, rng, model_name,
                                n_lopo_patients=args.n_lopo_patients,
                                n_repeats=args.n_repeats,
                                skip_demean=args.skip_demean)
        all_results.append(res)

    # ── Hemodynamic clusters ──────────────────────────────────────────────
    for model_name, emb in [("JEPA", jepa_emb), ("PatchTST", ptst_emb)]:
        res_list = probe_hemodynamic_clusters(
            emb, patient_ids, patient_cluster, window_hemo,
            rng, model_name,
            n_lopo_patients=args.n_lopo_patients,
            n_repeats=args.n_repeats,
            skip_demean=args.skip_demean)
        all_results.extend(res_list)

    # ── Save ──────────────────────────────────────────────────────────────
    # Remove non-serializable fields for CSV
    save_results = []
    for r in all_results:
        r_save = {k: v for k, v in r.items() if k != "gen_curve"}
        save_results.append(r_save)

    df = pd.DataFrame(save_results)
    out_path = CLUSTERING_DIR / "linear_probe_generalizability.csv"
    df.to_csv(out_path, index=False)

    # Save generalization curves separately
    gen_curves = []
    for r in all_results:
        if "gen_curve" in r:
            for gc in r["gen_curve"]:
                gen_curves.append({"model": r["model"], "task": r["task"], **gc})
    if gen_curves:
        gc_df = pd.DataFrame(gen_curves)
        gc_path = CLUSTERING_DIR / "linear_probe_gen_curve.csv"
        gc_df.to_csv(gc_path, index=False)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n\n{'=' * 75}")
    print("LINEAR PROBE GENERALIZABILITY — FINAL SUMMARY")
    print(f"{'=' * 75}")

    print(f"\n  HYPOTENSION (AUROC):")
    print(f"  {'Method':<40} {'JEPA':>8} {'PatchTST':>10}")
    print(f"  {'─' * 60}")
    hypo = [r for r in all_results if r["task"] == "hypotension"]
    if len(hypo) == 2:
        j, p = hypo[0], hypo[1]
        print(f"  {'Within-patient (upper bound)':<40} {j['within_patient_auroc']:>8.4f} {p['within_patient_auroc']:>10.4f}")
        print(f"  {'Cross-patient (raw)':<40} {j['cross_patient_raw_auroc']:>8.4f} {p['cross_patient_raw_auroc']:>10.4f}")
        print(f"  {'Cross-patient (de-meaned)':<40} {j['cross_patient_demeaned_auroc']:>8.4f} {p['cross_patient_demeaned_auroc']:>10.4f}")
        print(f"  {'LOPO (raw)':<40} {j['lopo_raw_auroc_mean']:>8.4f} {p['lopo_raw_auroc_mean']:>10.4f}")
        print(f"  {'LOPO (de-meaned)':<40} {j['lopo_demeaned_auroc_mean']:>8.4f} {p['lopo_demeaned_auroc_mean']:>10.4f}")
        print(f"  {'Permutation (null)':<40} {j['permutation_auroc']:>8.4f} {p['permutation_auroc']:>10.4f}")
        print(f"  {'Chance':<40} {'0.5000':>8} {'0.5000':>10}")

    print(f"\n  HEMODYNAMIC CLUSTERS (Balanced Accuracy):")
    print(f"  {'Method':<40} {'JEPA':>8} {'PatchTST':>10}")
    print(f"  {'─' * 60}")
    hemo_pl = [r for r in all_results if r["task"] == "hemo_patient_level"]
    if len(hemo_pl) == 2:
        j, p = hemo_pl[0], hemo_pl[1]
        print(f"  Patient-level labels:")
        print(f"  {'  Within-patient (upper bound)':<40} {j['within_patient_bacc']:>8.4f} {p['within_patient_bacc']:>10.4f}")
        print(f"  {'  Cross-patient (raw)':<40} {j['cross_patient_raw_bacc']:>8.4f} {p['cross_patient_raw_bacc']:>10.4f}")
        print(f"  {'  Cross-patient (de-meaned)':<40} {j['cross_patient_demeaned_bacc']:>8.4f} {p['cross_patient_demeaned_bacc']:>10.4f}")
        print(f"  {'  LOPO (raw)':<40} {j['lopo_raw_bacc_mean']:>8.4f} {p['lopo_raw_bacc_mean']:>10.4f}")
        print(f"  {'  LOPO (de-meaned)':<40} {j['lopo_demeaned_bacc_mean']:>8.4f} {p['lopo_demeaned_bacc_mean']:>10.4f}")
        print(f"  {'  Permutation (null)':<40} {j['permutation_bacc']:>8.4f} {p['permutation_bacc']:>10.4f}")
        print(f"  {'  Chance (1/7)':<40} {j['chance']:>8.4f} {p['chance']:>10.4f}")

    hemo_wl = [r for r in all_results if r["task"] == "hemo_window_level"]
    if len(hemo_wl) == 2:
        j, p = hemo_wl[0], hemo_wl[1]
        print(f"  Window-level labels (time-aligned):")
        print(f"  {'  Within-patient (upper bound)':<40} {j['within_patient_bacc']:>8.4f} {p['within_patient_bacc']:>10.4f}")
        print(f"  {'  Cross-patient (raw)':<40} {j['cross_patient_raw_bacc']:>8.4f} {p['cross_patient_raw_bacc']:>10.4f}")
        print(f"  {'  Cross-patient (de-meaned)':<40} {j['cross_patient_demeaned_bacc']:>8.4f} {p['cross_patient_demeaned_bacc']:>10.4f}")
        print(f"  {'  LOPO (raw)':<40} {j['lopo_raw_bacc_mean']:>8.4f} {p['lopo_raw_bacc_mean']:>10.4f}")
        print(f"  {'  LOPO (de-meaned)':<40} {j['lopo_demeaned_bacc_mean']:>8.4f} {p['lopo_demeaned_bacc_mean']:>10.4f}")
        print(f"  {'  Permutation (null)':<40} {j['permutation_bacc']:>8.4f} {p['permutation_bacc']:>10.4f}")
        n_cls = j['n_classes']
        print(f"  {'  Chance (1/' + str(n_cls) + ')':<40} {j['chance']:>8.4f} {p['chance']:>10.4f}")

    print(f"\n  INTERPRETATION GUIDE:")
    print(f"  • Within-patient >> Cross-patient: patient fingerprint dominates")
    print(f"  • Cross-patient >> Chance: some condition info generalizes")
    print(f"  • De-meaned > Raw: condition info masked by patient centroid")
    print(f"  • De-meaned < Raw: patient centroid carries condition risk")
    print(f"  • LOPO ≈ Cross-patient: stable across evaluation methods")
    print(f"  • Permutation ≈ Chance: confirms real signal, not artifact")

    print(f"\n{'=' * 75}")
    print(f"Results saved to:")
    print(f"  {out_path}")
    if gen_curves:
        print(f"  {gc_path}")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
