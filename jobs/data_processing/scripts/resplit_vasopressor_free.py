#!/usr/bin/env python
"""
Re-split the vasopressor-free cohort using the same corrected stratified group
10-fold algorithm (seed=16) to produce balanced train/val/test splits.

Reads the vasopressor-free outcome labels, computes subject-level statistics,
runs the fold assignment, and writes:
  - Updated subject manifest with new fold/split assignments
  - Summary JSON with cohort statistics

Usage:
    python resplit_vasopressor_free.py [--seed 16] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
LABELS_PATH = OUTPUT_ROOT / "labels" / "hypotension_labels_vasopressor_free_stays_v1.csv.gz"
MANIFEST_OUT = OUTPUT_ROOT / "manifests" / "hypotension_subject_split_vasopressor_free_stays_v1.csv"
SUMMARY_OUT = OUTPUT_ROOT / "manifests" / "hypotension_subject_split_vasopressor_free_stays_v1.json"

SPLIT_NAMES = {0: "test", 1: "val"}


def corrected_stratified_group_folds(
    subject_stats: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    """
    Assign subjects to folds using a greedy algorithm that minimizes
    class-proportion variance across folds (same as the original split).
    """
    subjects = subject_stats.index.to_numpy()
    counts = subject_stats[["negative_events", "positive_events"]].to_numpy(
        dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(subjects))
    subjects = subjects[permutation]
    counts = counts[permutation]

    ordering = np.argsort(-np.std(counts, axis=1), kind="stable")
    totals = counts.sum(axis=0)
    fold_counts = np.zeros((n_splits, 2), dtype=np.float64)
    fold_subjects: list[list[str]] = [[] for _ in range(n_splits)]

    for index in ordering:
        group_counts = counts[index]
        best_fold = None
        best_score = np.inf
        best_size = np.inf
        for fold in range(n_splits):
            fold_counts[fold] += group_counts
            score = float(np.mean(np.std(fold_counts / totals, axis=0)))
            fold_counts[fold] -= group_counts
            size = float(fold_counts[fold].sum())
            if score < best_score or (
                np.isclose(score, best_score) and size < best_size
            ):
                best_fold = fold
                best_score = score
                best_size = size
        fold_counts[best_fold] += group_counts
        fold_subjects[best_fold].append(str(subjects[index]))

    return {
        subject: fold
        for fold, members in enumerate(fold_subjects)
        for subject in members
    }


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(content)
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--labels-path", type=Path, default=LABELS_PATH)
    parser.add_argument("--manifest-out", type=Path, default=MANIFEST_OUT)
    parser.add_argument("--summary-out", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--restrict-to-manifest", type=Path, default=None,
                        help="Only include subjects present in this reference manifest CSV")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading vasopressor-free outcome labels...")
    labels = pd.read_csv(args.labels_path)
    print(f"  {len(labels):,} rows, {labels['subject_id'].nunique()} subjects, "
          f"{labels['file_name'].nunique()} stays")

    # Optionally restrict to subjects from a reference manifest
    if args.restrict_to_manifest is not None:
        ref = pd.read_csv(args.restrict_to_manifest)
        valid_subjects = set(ref["subject_id"].unique())
        before = labels["subject_id"].nunique()
        labels = labels[labels["subject_id"].isin(valid_subjects)].copy()
        after = labels["subject_id"].nunique()
        print(f"  Restricted to reference manifest subjects: {before} → {after}")
        print(f"  Rows after restriction: {len(labels):,}")

    # Compute subject-level statistics
    print("Computing subject-level statistics...")
    subject_stats = labels.groupby("subject_id").agg(
        positive_events=("hypotension_label", "sum"),
        samples=("hypotension_label", "count"),
        icu_stays=("file_name", "nunique"),
    )
    subject_stats["positive_events"] = subject_stats["positive_events"].astype(int)
    subject_stats["negative_events"] = subject_stats["samples"] - subject_stats["positive_events"]

    print(f"  {len(subject_stats)} subjects")
    print(f"  {subject_stats['samples'].sum():,} total samples")
    print(f"  {subject_stats['positive_events'].sum():,} positive events "
          f"({100 * subject_stats['positive_events'].sum() / subject_stats['samples'].sum():.2f}%)")

    # Run the fold assignment
    print(f"\nRunning corrected stratified group {args.n_splits}-fold (seed={args.seed})...")
    assignments = corrected_stratified_group_folds(
        subject_stats, n_splits=args.n_splits, seed=args.seed
    )

    # Build manifest
    manifest = subject_stats.reset_index()
    manifest["fold"] = manifest["subject_id"].map(assignments).astype(int)
    manifest["split"] = manifest["fold"].map(SPLIT_NAMES).fillna("train")
    manifest["seed"] = args.seed
    manifest["algorithm"] = "corrected_stratified_group_10fold_v1"

    # Reorder columns
    manifest = manifest[["subject_id", "positive_events", "samples", "icu_stays",
                          "negative_events", "fold", "split", "seed", "algorithm"]]
    manifest.sort_values(["split", "subject_id"], inplace=True)

    # Verify no subject leakage
    for split_a in ["train", "val", "test"]:
        for split_b in ["train", "val", "test"]:
            if split_a >= split_b:
                continue
            subjects_a = set(manifest[manifest["split"] == split_a]["subject_id"])
            subjects_b = set(manifest[manifest["split"] == split_b]["subject_id"])
            overlap = subjects_a & subjects_b
            assert len(overlap) == 0, f"Leakage between {split_a} and {split_b}: {overlap}"

    # Summary
    summary = {
        "description": "Vasopressor-free cohort, re-split with corrected stratified group 10-fold",
        "algorithm": "corrected_stratified_group_10fold_v1",
        "seed": args.seed,
        "n_splits": args.n_splits,
        "fold_mapping": {"test": 0, "val": 1, "train": "2-9"},
        "total": {
            "subjects": len(manifest),
            "samples": int(manifest["samples"].sum()),
            "positive_events": int(manifest["positive_events"].sum()),
            "prevalence_pct": round(
                100 * manifest["positive_events"].sum() / manifest["samples"].sum(), 2
            ),
            "icu_stays": int(manifest["icu_stays"].sum()),
        },
        "splits": {},
    }

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"{'Split':<8} {'Subjects':<10} {'Stays':<8} {'Samples':<12} {'Pos Events':<12} {'Prevalence'}")
    print("-" * 60)
    for split_name in ["train", "val", "test"]:
        s = manifest[manifest["split"] == split_name]
        n_subj = len(s)
        n_stays = int(s["icu_stays"].sum())
        n_samples = int(s["samples"].sum())
        n_pos = int(s["positive_events"].sum())
        prev = round(100 * n_pos / n_samples, 2) if n_samples > 0 else 0
        print(f"{split_name:<8} {n_subj:<10} {n_stays:<8} {n_samples:<12,} {n_pos:<12,} {prev}%")
        summary["splits"][split_name] = {
            "subjects": n_subj,
            "icu_stays": n_stays,
            "samples": n_samples,
            "positive_events": n_pos,
            "prevalence_pct": prev,
        }
    print("-" * 60)
    total = summary["total"]
    print(f"{'TOTAL':<8} {total['subjects']:<10} {total['icu_stays']:<8} "
          f"{total['samples']:<12,} {total['positive_events']:<12,} {total['prevalence_pct']}%")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write outputs
    print("\nWriting outputs...")
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(args.manifest_out, index=False)
    print(f"  → {args.manifest_out}")

    with open(args.summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {args.summary_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
