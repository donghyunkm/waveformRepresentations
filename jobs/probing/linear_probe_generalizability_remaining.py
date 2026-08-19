"""
Linear probe generalizability — REMAINING analyses only.

Runs only:
  1. Hemodynamic clusters (JEPA) — window_level
  2. Hemodynamic clusters (PatchTST) — patient_level + window_level

Skips hypotension (both models) and hemo patient_level (JEPA) which already
completed in job 26294743.

Usage:
    python -u linear_probe_generalizability_remaining.py [--seed 42] [--n-lopo-patients 50]
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


# ── Hemodynamic Probing ───────────────────────────────────────────────────────

def _probe_hemo_subset(embeddings, labels, patient_ids, rng, model_name,
                       label_type, n_lopo_patients=30, n_repeats=5,
                       skip_demean=False):
    """Probe hemodynamic clusters on a labeled subset."""
    unique_pids = np.unique(patient_ids)
    n_patients = len(unique_pids)
    n_classes = len(np.unique(labels))

    print(f"\n  ── {label_type} labels ──")
    print(f"     {n_patients} patients, {len(labels)} windows, {n_classes} classes")
    print(f"     Distribution: {np.bincount(labels.astype(int), minlength=n_classes).tolist()}")

    results = {"model": model_name, "task": f"hemo_{label_type}"}

    if not skip_demean:
        emb_dm = demean_by_patient(embeddings, patient_ids)

    # ── 1. Cross-patient probe ────────────────────────────────────────────
    print(f"\n     1. Cross-patient probe ({n_repeats} splits):")
    baccs_raw = []
    baccs_dm = []
    for i in range(n_repeats):
        pids_shuffled = rng.permutation(unique_pids)
        split = int(0.8 * len(pids_shuffled))
        train_pids = set(pids_shuffled[:split])
        test_pids = set(pids_shuffled[split:])

        train_mask = np.array([pid in train_pids for pid in patient_ids])
        test_mask = np.array([pid in test_pids for pid in patient_ids])

        _, bacc = fit_and_evaluate(
            embeddings[train_mask], labels[train_mask],
            embeddings[test_mask], labels[test_mask],
            binary=False, random_state=42 + i)
        baccs_raw.append(bacc)

        if not skip_demean:
            _, bacc_dm = fit_and_evaluate(
                emb_dm[train_mask], labels[train_mask],
                emb_dm[test_mask], labels[test_mask],
                binary=False, random_state=42 + i)
            baccs_dm.append(bacc_dm)

    results["cross_patient_raw_bacc"] = np.mean(baccs_raw)
    results["cross_patient_raw_bacc_std"] = np.std(baccs_raw)
    print(f"        Raw:       Bal.Acc {np.mean(baccs_raw):.4f}±{np.std(baccs_raw):.4f}")

    if not skip_demean:
        results["cross_patient_demeaned_bacc"] = np.mean(baccs_dm)
        results["cross_patient_demeaned_bacc_std"] = np.std(baccs_dm)
        print(f"        De-meaned: Bal.Acc {np.mean(baccs_dm):.4f}±{np.std(baccs_dm):.4f}")
    else:
        results["cross_patient_demeaned_bacc"] = np.nan
        results["cross_patient_demeaned_bacc_std"] = np.nan

    # ── 3. Leave-one-patient-out ──────────────────────────────────────────
    print(f"\n     3. Leave-one-patient-out:")
    lopo_pids = rng.choice(unique_pids, size=min(n_lopo_patients, n_patients),
                           replace=False)
    lopo_baccs_raw = []
    lopo_baccs_dm = []

    for pid in lopo_pids:
        test_mask = patient_ids == pid
        train_mask = ~test_mask

        # Need at least 2 classes in train
        if len(np.unique(labels[train_mask])) < 2:
            continue
        # Need at least 1 sample in test
        if test_mask.sum() == 0:
            continue

        _, bacc = fit_and_evaluate(
            embeddings[train_mask], labels[train_mask],
            embeddings[test_mask], labels[test_mask],
            binary=False, random_state=42)
        lopo_baccs_raw.append(bacc)

        if not skip_demean:
            _, bacc_dm = fit_and_evaluate(
                emb_dm[train_mask], labels[train_mask],
                emb_dm[test_mask], labels[test_mask],
                binary=False, random_state=42)
            lopo_baccs_dm.append(bacc_dm)

    results["lopo_raw_bacc_mean"] = np.mean(lopo_baccs_raw) if lopo_baccs_raw else np.nan
    results["lopo_raw_bacc_std"] = np.std(lopo_baccs_raw) if lopo_baccs_raw else np.nan
    print(f"        Raw:       Bal.Acc {results['lopo_raw_bacc_mean']:.4f}±{results['lopo_raw_bacc_std']:.4f}")

    if not skip_demean:
        results["lopo_demeaned_bacc_mean"] = np.mean(lopo_baccs_dm) if lopo_baccs_dm else np.nan
        results["lopo_demeaned_bacc_std"] = np.std(lopo_baccs_dm) if lopo_baccs_dm else np.nan
        print(f"        De-meaned: Bal.Acc {results['lopo_demeaned_bacc_mean']:.4f}±{results['lopo_demeaned_bacc_std']:.4f}")
    else:
        results["lopo_demeaned_bacc_mean"] = np.nan
        results["lopo_demeaned_bacc_std"] = np.nan

    # ── 4. Within-patient baseline ────────────────────────────────────────
    print(f"\n     4. Within-patient baseline:")
    idx = rng.permutation(len(labels))
    split = int(0.8 * len(idx))
    _, bacc_within = fit_and_evaluate(
        embeddings[idx[:split]], labels[idx[:split]],
        embeddings[idx[split:]], labels[idx[split:]],
        binary=False, random_state=42)
    results["within_patient_bacc"] = bacc_within
    print(f"        Bal.Acc: {bacc_within:.4f}")

    # ── 5. Permutation baseline ───────────────────────────────────────────
    print(f"\n     5. Permutation baseline:")
    perm_labels = labels.copy()
    for pid in unique_pids:
        mask = patient_ids == pid
        perm_labels[mask] = rng.permutation(perm_labels[mask])

    pids_shuffled = rng.permutation(unique_pids)
    split = int(0.8 * len(pids_shuffled))
    train_pids = set(pids_shuffled[:split])
    train_mask = np.array([pid in train_pids for pid in patient_ids])
    test_mask = ~train_mask

    _, bacc_perm = fit_and_evaluate(
        embeddings[train_mask], perm_labels[train_mask],
        embeddings[test_mask], perm_labels[test_mask],
        binary=False, random_state=42)
    results["permutation_bacc"] = bacc_perm
    print(f"        Bal.Acc: {bacc_perm:.4f} (should be ~{1/n_classes:.3f})")

    # ── Summary box ───────────────────────────────────────────────────────
    chance = 1.0 / n_classes
    print(f"\n     ┌─────────────────────────────────────────────────────────┐")
    print(f"     │ HEMO {label_type}  ({model_name})"
          f"{'':>{40 - len(label_type) - len(model_name)}}│")
    print(f"     │ Within-patient:       Bal.Acc {bacc_within:.4f}"
          f"{'':>{15}}│")
    print(f"     │ Cross-patient (raw):  Bal.Acc {results['cross_patient_raw_bacc']:.4f}"
          f"{'':>{15}}│")
    print(f"     │ Cross-patient (de-m): Bal.Acc {results['cross_patient_demeaned_bacc']:.4f}"
          f"{'':>{15}}│")
    print(f"     │ LOPO (raw):           Bal.Acc {results['lopo_raw_bacc_mean']:.4f}"
          f"{'':>{15}}│")
    print(f"     │ LOPO (de-meaned):     Bal.Acc {results['lopo_demeaned_bacc_mean']:.4f}"
          f"{'':>{15}}│")
    print(f"     │ Permutation:          Bal.Acc {bacc_perm:.4f}"
          f"{'':>{15}}│")
    print(f"     │ Chance (1/{n_classes}):          Bal.Acc {chance:.4f}"
          f"{'':>{15}}│")
    print(f"     └─────────────────────────────────────────────────────────┘")

    return results


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

    # Load hemodynamic clusters
    patient_cluster = load_hemodynamic_clusters_patient_level()
    window_hemo = load_hemodynamic_clusters_window_level()

    all_results = []

    # ── Hemodynamic clusters — JEPA window_level only ─────────────────────
    print("\n" + "=" * 75)
    print("  REMAINING ANALYSIS: Hemo window_level (JEPA) + full Hemo (PatchTST)")
    print("=" * 75)

    # JEPA: window_level only (patient_level already done in job 26294743)
    if window_hemo is not None:
        hemo_wl = window_hemo
        valid_window = hemo_wl >= 0
        if valid_window.sum() > 1000:
            print("\n── JEPA: hemodynamic window_level ──")
            res = _probe_hemo_subset(
                jepa_emb[valid_window], hemo_wl[valid_window],
                patient_ids[valid_window], rng, "JEPA",
                label_type="window_level",
                n_lopo_patients=args.n_lopo_patients,
                n_repeats=args.n_repeats,
                skip_demean=args.skip_demean)
            all_results.append(res)

    # PatchTST: both patient_level and window_level
    hemo_labels_pl = np.array([patient_cluster.get(str(pid), -1)
                               for pid in patient_ids])
    valid_patient = hemo_labels_pl >= 0

    print("\n── PatchTST: hemodynamic patient_level ──")
    res_pl = _probe_hemo_subset(
        ptst_emb[valid_patient], hemo_labels_pl[valid_patient],
        patient_ids[valid_patient], rng, "PatchTST",
        label_type="patient_level",
        n_lopo_patients=args.n_lopo_patients,
        n_repeats=args.n_repeats,
        skip_demean=args.skip_demean)
    all_results.append(res_pl)

    if window_hemo is not None:
        valid_window = window_hemo >= 0
        if valid_window.sum() > 1000:
            print("\n── PatchTST: hemodynamic window_level ──")
            res_wl = _probe_hemo_subset(
                ptst_emb[valid_window], window_hemo[valid_window],
                patient_ids[valid_window], rng, "PatchTST",
                label_type="window_level",
                n_lopo_patients=args.n_lopo_patients,
                n_repeats=args.n_repeats,
                skip_demean=args.skip_demean)
            all_results.append(res_wl)

    # ── Save ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    out_path = CLUSTERING_DIR / "linear_probe_generalizability_remaining.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # Print summary
    print(f"\n{'=' * 75}")
    print("REMAINING ANALYSIS — SUMMARY")
    print(f"{'=' * 75}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
