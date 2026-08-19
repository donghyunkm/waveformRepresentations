#!/usr/bin/env python
"""
Identify ICU stays with vasopressor administration and exclude them from
the PhysioJEPA waveform cohort.

Strategy (stay-level exclusion):
1. Load INPUTEVENTS_MV and INPUTEVENTS_CV filtered to vasopressor item IDs.
2. Get unique ICUSTAY_IDs that received any vasopressor.
3. Join to ICUSTAYS to get SUBJECT_ID + INTIME + OUTTIME.
4. Match each vasopressor ICU stay to waveform stay_ids by subject + temporal
   overlap (waveform recording start falls within INTIME–OUTTIME window,
   with a tolerance for slight offsets).
5. Exclude matched waveform stays from the outcome labels.
6. Rebuild the subject-level manifest with updated counts.
7. Output summary statistics.

Outputs (all under OUTPUT_ROOT/manifests/):
- vasopressor_icustay_ids.csv: all ICUSTAY_IDs with vasopressor admin
- vasopressor_excluded_waveform_stays.csv: matched waveform stays to exclude
- hypotension_labels_vasopressor_free_stays_v1.csv.gz: filtered outcome labels
- hypotension_subject_split_vasopressor_free_stays_v1.csv: updated subject manifest
- vasopressor_exclusion_summary.json: cohort summary statistics

Usage:
    python exclude_vasopressor_stays.py [--output-root OUTPUT_ROOT]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np


# === Vasopressor item IDs ===

# MetaVision (INPUTEVENTS_MV)
MV_VASOPRESSOR_ITEMIDS = [
    221906,  # Norepinephrine
    221289,  # Epinephrine
    221749,  # Phenylephrine
    221662,  # Dopamine
    221653,  # Dobutamine
    221986,  # Milrinone
    222315,  # Vasopressin
]

# CareVue (INPUTEVENTS_CV)
CV_VASOPRESSOR_ITEMIDS = [
    30047,   # Levophed (Norepinephrine)
    30120,   # Levophed-k
    30044,   # Epinephrine
    30119,   # Epinephrine-k
    30127,   # Neosynephrine (Phenylephrine)
    30128,   # Neosynephrine-k
    30043,   # Dopamine
    30307,   # Dopamine Drip
    30042,   # Dobutamine
    30306,   # Dobutamine Drip
    30125,   # Milrinone
    30051,   # Vasopressin
]

# Default paths
MIMIC_CLINICAL_DIR = Path("/gpfs/data/eh3828lab/datasets/mimic_clinical")
OUTPUT_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
MANIFESTS_DIR = OUTPUT_ROOT / "manifests"
LABELS_PATH = OUTPUT_ROOT / "labels" / "hypotension_labels_mimic_all_events_rolling5min.csv.gz"
WAVEFORM_MANIFEST_PATH = MANIFESTS_DIR / "waveform_manifest.csv"
SUBJECT_SPLIT_PATH = MANIFESTS_DIR / "hypotension_subject_split_fixed_v1.csv"

# Temporal tolerance for matching waveform recording start to ICU stay window
TEMPORAL_TOLERANCE = timedelta(hours=1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                        help="Root output directory")
    parser.add_argument("--mimic-clinical-dir", type=Path, default=MIMIC_CLINICAL_DIR,
                        help="Path to MIMIC clinical CSV directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary without writing output files")
    return parser.parse_args()


def load_vasopressor_stays(mimic_dir: Path) -> pd.DataFrame:
    """Load and combine vasopressor administrations from MV and CV tables."""
    print("Loading INPUTEVENTS_MV (vasopressor items only)...")
    mv = pd.read_csv(
        mimic_dir / "INPUTEVENTS_MV.csv.gz",
        usecols=["SUBJECT_ID", "ICUSTAY_ID", "ITEMID"],
        dtype={"SUBJECT_ID": int, "ICUSTAY_ID": "Int64", "ITEMID": int},
    )
    mv_vaso = mv[mv["ITEMID"].isin(MV_VASOPRESSOR_ITEMIDS)].copy()
    print(f"  MV vasopressor rows: {len(mv_vaso):,}")
    del mv

    print("Loading INPUTEVENTS_CV (vasopressor items only)...")
    cv = pd.read_csv(
        mimic_dir / "INPUTEVENTS_CV.csv.gz",
        usecols=["SUBJECT_ID", "ICUSTAY_ID", "ITEMID"],
        dtype={"SUBJECT_ID": int, "ICUSTAY_ID": "Int64", "ITEMID": int},
    )
    cv_vaso = cv[cv["ITEMID"].isin(CV_VASOPRESSOR_ITEMIDS)].copy()
    print(f"  CV vasopressor rows: {len(cv_vaso):,}")
    del cv

    # Combine unique ICUSTAY_IDs
    combined = pd.concat([mv_vaso[["SUBJECT_ID", "ICUSTAY_ID"]],
                          cv_vaso[["SUBJECT_ID", "ICUSTAY_ID"]]])
    # Drop rows where ICUSTAY_ID is null
    combined = combined.dropna(subset=["ICUSTAY_ID"])
    combined["ICUSTAY_ID"] = combined["ICUSTAY_ID"].astype(int)

    unique_stays = combined.drop_duplicates(subset=["ICUSTAY_ID"])
    print(f"  Unique vasopressor ICU stays: {len(unique_stays):,}")
    print(f"  Unique vasopressor subjects: {unique_stays['SUBJECT_ID'].nunique():,}")
    return unique_stays


def load_icustays(mimic_dir: Path) -> pd.DataFrame:
    """Load ICUSTAYS table with admission/discharge times."""
    print("Loading ICUSTAYS...")
    icustays = pd.read_csv(
        mimic_dir / "ICUSTAYS.csv.gz",
        usecols=["SUBJECT_ID", "ICUSTAY_ID", "INTIME", "OUTTIME"],
        dtype={"SUBJECT_ID": int, "ICUSTAY_ID": int},
        parse_dates=["INTIME", "OUTTIME"],
    )
    print(f"  Total ICU stays: {len(icustays):,}")
    return icustays


def parse_waveform_stay_datetime(stay_id: str) -> pd.Timestamp:
    """
    Parse datetime from waveform stay_id format: pXXXXXX-YYYY-MM-DD-HH-MM
    Returns a Timestamp.
    """
    parts = stay_id.split("-")
    # Format: subject-YYYY-MM-DD-HH-MM
    year = int(parts[1])
    month = int(parts[2])
    day = int(parts[3])
    hour = int(parts[4])
    minute = int(parts[5])
    return pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)


def extract_subject_id_int(stay_id: str) -> int:
    """Extract integer subject_id from waveform stay_id (e.g., 'p000160-...' -> 160)."""
    return int(stay_id.split("-")[0].lstrip("p"))


def match_waveform_stays_to_vasopressor_icustays(
    waveform_manifest: pd.DataFrame,
    vaso_icustays: pd.DataFrame,
    tolerance: timedelta = TEMPORAL_TOLERANCE,
) -> set:
    """
    Match waveform stays to vasopressor ICU stays by subject + temporal overlap.

    A waveform stay is excluded if its recording start time falls within
    [INTIME - tolerance, OUTTIME + tolerance] of a vasopressor ICU stay
    for the same subject.

    Returns set of waveform stay_ids to exclude.
    """
    print("Matching waveform stays to vasopressor ICU stays...")

    # Parse waveform stay datetimes
    wf = waveform_manifest[["subject_id", "stay_id"]].copy()
    wf["subject_int"] = wf["subject_id"].apply(lambda x: int(x.lstrip("p")))
    wf["recording_start"] = wf["stay_id"].apply(parse_waveform_stay_datetime)

    # Filter vaso_icustays to subjects that appear in waveform data
    wf_subjects = set(wf["subject_int"].unique())
    vaso_relevant = vaso_icustays[vaso_icustays["SUBJECT_ID"].isin(wf_subjects)].copy()
    print(f"  Vasopressor ICU stays for waveform subjects: {len(vaso_relevant):,}")

    if len(vaso_relevant) == 0:
        return set()

    # For each waveform stay, check if it overlaps any vasopressor ICU stay
    excluded_stays = set()

    # Group vasopressor stays by subject for efficient lookup
    vaso_by_subject = vaso_relevant.groupby("SUBJECT_ID").apply(
        lambda g: list(zip(g["INTIME"], g["OUTTIME"])), include_groups=False
    ).to_dict()

    for _, row in wf.iterrows():
        subj = row["subject_int"]
        if subj not in vaso_by_subject:
            continue

        rec_start = row["recording_start"]
        for intime, outtime in vaso_by_subject[subj]:
            # Check if recording start falls within the ICU stay window (with tolerance)
            if (intime - tolerance) <= rec_start <= (outtime + tolerance):
                excluded_stays.add(row["stay_id"])
                break

    print(f"  Waveform stays matched to vasopressor ICU stays: {len(excluded_stays):,}")
    return excluded_stays


def filter_outcome_labels(labels_path: Path, excluded_stays: set) -> pd.DataFrame:
    """Filter outcome labels to remove excluded stays."""
    print(f"Loading outcome labels from {labels_path.name}...")
    labels = pd.read_csv(labels_path)
    n_before = len(labels)
    stays_before = labels["file_name"].nunique()

    # file_name column contains the stay_id
    labels_filtered = labels[~labels["file_name"].isin(excluded_stays)].copy()

    n_after = len(labels_filtered)
    stays_after = labels_filtered["file_name"].nunique()
    print(f"  Labels: {n_before:,} → {n_after:,} rows ({n_before - n_after:,} removed)")
    print(f"  Stays: {stays_before:,} → {stays_after:,} ({stays_before - stays_after:,} removed)")
    return labels_filtered


def rebuild_subject_manifest(
    labels_filtered: pd.DataFrame,
    original_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rebuild subject-level manifest from filtered labels, preserving fold/split
    assignments from the original manifest.
    """
    print("Rebuilding subject manifest...")

    # Compute per-subject stats from filtered labels
    subject_stats = labels_filtered.groupby("subject_id").agg(
        positive_events=("hypotension_label", "sum"),
        samples=("hypotension_label", "count"),
        icu_stays=("file_name", "nunique"),
    ).reset_index()
    subject_stats["negative_events"] = subject_stats["samples"] - subject_stats["positive_events"]
    subject_stats["positive_events"] = subject_stats["positive_events"].astype(int)

    # Merge with original manifest to get fold/split assignments
    orig_cols = original_manifest[["subject_id", "fold", "split", "seed", "algorithm"]].copy()
    manifest_new = subject_stats.merge(orig_cols, on="subject_id", how="inner")

    # Reorder columns to match original format
    manifest_new = manifest_new[["subject_id", "positive_events", "samples",
                                  "icu_stays", "negative_events", "fold", "split",
                                  "seed", "algorithm"]]

    # Subjects that lost all stays are dropped (inner join handles this)
    n_orig = len(original_manifest)
    n_new = len(manifest_new)
    print(f"  Subjects: {n_orig:,} → {n_new:,} ({n_orig - n_new:,} fully excluded)")
    return manifest_new


def compute_summary(
    manifest_new: pd.DataFrame,
    manifest_orig: pd.DataFrame,
    excluded_stays: set,
    vaso_unique_stays: pd.DataFrame,
    labels_filtered: pd.DataFrame,
) -> dict:
    """Compute summary statistics for the exclusion."""
    summary = {
        "description": "Stay-level vasopressor exclusion from PhysioJEPA cohort",
        "exclusion_level": "stay",
        "vasopressor_item_ids": {
            "metavision": MV_VASOPRESSOR_ITEMIDS,
            "carevue": CV_VASOPRESSOR_ITEMIDS,
        },
        "temporal_tolerance_hours": TEMPORAL_TOLERANCE.total_seconds() / 3600,
        "original_cohort": {
            "subjects": len(manifest_orig),
            "total_samples": int(manifest_orig["samples"].sum()),
            "positive_events": int(manifest_orig["positive_events"].sum()),
            "prevalence_pct": round(
                100 * manifest_orig["positive_events"].sum() / manifest_orig["samples"].sum(), 2
            ),
        },
        "vasopressor_stays_in_mimic": int(len(vaso_unique_stays)),
        "waveform_stays_excluded": len(excluded_stays),
        "filtered_cohort": {
            "subjects": len(manifest_new),
            "total_samples": int(manifest_new["samples"].sum()),
            "positive_events": int(manifest_new["positive_events"].sum()),
            "prevalence_pct": round(
                100 * manifest_new["positive_events"].sum() / manifest_new["samples"].sum(), 2
            ) if manifest_new["samples"].sum() > 0 else 0,
        },
        "per_split": {},
    }

    for split_name in ["train", "val", "test"]:
        orig_split = manifest_orig[manifest_orig["split"] == split_name]
        new_split = manifest_new[manifest_new["split"] == split_name]
        summary["per_split"][split_name] = {
            "subjects_orig": len(orig_split),
            "subjects_new": len(new_split),
            "subjects_removed": len(orig_split) - len(new_split),
            "samples_orig": int(orig_split["samples"].sum()),
            "samples_new": int(new_split["samples"].sum()),
            "positive_events_new": int(new_split["positive_events"].sum()),
            "prevalence_pct": round(
                100 * new_split["positive_events"].sum() / new_split["samples"].sum(), 2
            ) if new_split["samples"].sum() > 0 else 0,
        }

    # Subjects that were completely excluded (lost all stays)
    removed_subjects = set(manifest_orig["subject_id"]) - set(manifest_new["subject_id"])
    summary["subjects_fully_excluded"] = len(removed_subjects)

    # Subjects that lost some stays but retained others
    partial_subjects = set(manifest_new["subject_id"]) & set(
        manifest_orig[manifest_orig["icu_stays"] != manifest_new.set_index("subject_id").reindex(
            manifest_orig["subject_id"])["icu_stays"].values]["subject_id"]
    ) if len(manifest_new) > 0 else set()

    return summary


def main():
    args = parse_args()
    output_manifests_dir = args.output_root / "manifests"
    output_labels_dir = args.output_root / "labels"

    print("=" * 70)
    print("VASOPRESSOR STAY-LEVEL EXCLUSION")
    print("=" * 70)
    print()

    # Step 1: Identify vasopressor ICU stays
    vaso_stays = load_vasopressor_stays(args.mimic_clinical_dir)
    print()

    # Step 2: Load ICUSTAYS for temporal information
    icustays = load_icustays(args.mimic_clinical_dir)
    vaso_icustays = icustays[icustays["ICUSTAY_ID"].isin(vaso_stays["ICUSTAY_ID"])].copy()
    print(f"  Vasopressor ICU stays with temporal info: {len(vaso_icustays):,}")
    print()

    # Step 3: Load waveform manifest
    print("Loading waveform manifest...")
    wf_manifest = pd.read_csv(WAVEFORM_MANIFEST_PATH)
    print(f"  Waveform stays: {len(wf_manifest):,}")
    print()

    # Step 4: Match waveform stays to vasopressor ICU stays
    excluded_stays = match_waveform_stays_to_vasopressor_icustays(
        wf_manifest, vaso_icustays, tolerance=TEMPORAL_TOLERANCE
    )
    print()

    # Step 5: Filter outcome labels
    labels_filtered = filter_outcome_labels(LABELS_PATH, excluded_stays)
    print()

    # Step 6: Rebuild subject manifest
    manifest_orig = pd.read_csv(SUBJECT_SPLIT_PATH)
    manifest_new = rebuild_subject_manifest(labels_filtered, manifest_orig)
    print()

    # Step 7: Compute and display summary
    summary = compute_summary(manifest_new, manifest_orig, excluded_stays,
                              vaso_stays, labels_filtered)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Original cohort:  {summary['original_cohort']['subjects']} subjects, "
          f"{summary['original_cohort']['total_samples']:,} samples, "
          f"{summary['original_cohort']['prevalence_pct']}% positive")
    print(f"  Filtered cohort:  {summary['filtered_cohort']['subjects']} subjects, "
          f"{summary['filtered_cohort']['total_samples']:,} samples, "
          f"{summary['filtered_cohort']['prevalence_pct']}% positive")
    print(f"  Waveform stays excluded: {summary['waveform_stays_excluded']}")
    print(f"  Subjects fully excluded: {summary['subjects_fully_excluded']}")
    print()
    print("  Per-split breakdown:")
    for split_name, stats in summary["per_split"].items():
        print(f"    {split_name:5s}: {stats['subjects_orig']} → {stats['subjects_new']} subjects, "
              f"{stats['samples_orig']:,} → {stats['samples_new']:,} samples, "
              f"{stats['prevalence_pct']}% positive")
    print()

    if args.dry_run:
        print("[DRY RUN] No files written.")
        return

    # Step 8: Write outputs
    print("Writing outputs...")
    output_manifests_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Vasopressor ICU stay IDs (for reference)
    vaso_out = output_manifests_dir / "vasopressor_icustay_ids.csv"
    vaso_stays.to_csv(vaso_out, index=False)
    print(f"  → {vaso_out}")

    # Excluded waveform stays
    excluded_out = output_manifests_dir / "vasopressor_excluded_waveform_stays.csv"
    pd.DataFrame({"stay_id": sorted(excluded_stays)}).to_csv(excluded_out, index=False)
    print(f"  → {excluded_out}")

    # Filtered outcome labels
    labels_out = output_labels_dir / "hypotension_labels_vasopressor_free_stays_v1.csv.gz"
    labels_filtered.to_csv(labels_out, index=False, compression="gzip")
    print(f"  → {labels_out}")

    # New subject manifest
    manifest_out = output_manifests_dir / "hypotension_subject_split_vasopressor_free_stays_v1.csv"
    manifest_new.to_csv(manifest_out, index=False)
    print(f"  → {manifest_out}")

    # Summary JSON
    summary_out = output_manifests_dir / "vasopressor_exclusion_summary.json"
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → {summary_out}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
