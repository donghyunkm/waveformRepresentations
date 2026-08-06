"""Build a fixed corrected 80/10/10 subject split from valid FCN samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA"
)
DEFAULT_MODELS_DIR = DEFAULT_OUTPUT_ROOT / "models" / "fcn_hypotension_paper"
SOURCE_PREFIX = (
    "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_paper_full_v1"
)
TARGET_PREFIX = (
    "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"
)
SPLIT_NAMES = {0: "test", 1: "val"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, destination: Path, compression=None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    frame.to_csv(partial, index=False, compression=compression)
    os.replace(partial, destination)


def atomic_json(data: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(partial, destination)


def corrected_stratified_group_folds(
    subject_stats: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    """Assign groups while shuffling identities and label counts together."""
    subjects = subject_stats.index.to_numpy()
    counts = subject_stats[["negative_events", "positive_events"]].to_numpy(
        dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(subjects))
    subjects = subjects[permutation]
    counts = counts[permutation]

    # Match StratifiedGroupKFold's greedy objective, with a seeded stable
    # tie-break that preserves the mapping between each subject and its counts.
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


def summarize(frame: pd.DataFrame) -> dict:
    labels = frame["outcome_val_300sec"].value_counts().to_dict()
    return {
        "samples": int(len(frame)),
        "positive_events": int(labels.get(1, 0)),
        "negative_events": int(labels.get(0, 0)),
        "icu_stays": int(frame["file_path"].nunique()),
        "patients": int(frame["subject_id"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--source-prefix", default=SOURCE_PREFIX)
    parser.add_argument("--target-prefix", default=TARGET_PREFIX)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT
        / "manifests"
        / "hypotension_subject_split_fixed_v1.csv",
    )
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_paths = {
        split: args.models_dir / f"{args.source_prefix}-{split}_samples.csv.gz"
        for split in ("train", "val", "test")
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    frames = []
    for source_split, path in source_paths.items():
        frame = pd.read_csv(path)
        frame = frame.loc[
            :, ~frame.columns.str.match(r"^Unnamed(:.*)?$")
        ].copy()
        frame["source_split"] = source_split
        frames.append(frame)
    samples = pd.concat(frames, ignore_index=True)
    identity_columns = ["file_path", "start_idx", "end_idx"]
    duplicates = samples.duplicated(identity_columns, keep=False)
    if duplicates.any():
        raise ValueError(
            f"Found {int(duplicates.sum())} duplicate sample identities"
        )

    subject_stats = samples.groupby("subject_id").agg(
        positive_events=("outcome_val_300sec", "sum"),
        samples=("outcome_val_300sec", "size"),
        icu_stays=("file_path", "nunique"),
    )
    subject_stats["negative_events"] = (
        subject_stats["samples"] - subject_stats["positive_events"]
    )
    assignments = corrected_stratified_group_folds(
        subject_stats, n_splits=10, seed=args.seed
    )
    samples["fold"] = samples["subject_id"].map(assignments)
    if samples["fold"].isna().any():
        raise RuntimeError("At least one sample subject was not assigned")
    samples["fold"] = samples["fold"].astype(int)
    samples["split"] = samples["fold"].map(SPLIT_NAMES).fillna("train")

    manifest = subject_stats.reset_index()
    manifest["fold"] = manifest["subject_id"].map(assignments).astype(int)
    manifest["split"] = manifest["fold"].map(SPLIT_NAMES).fillna("train")
    manifest["seed"] = args.seed
    manifest["algorithm"] = "corrected_stratified_group_10fold_v1"
    manifest.sort_values(["split", "subject_id"], inplace=True)

    split_frames = {
        split: samples.loc[samples["split"] == split].drop(
            columns=["source_split", "fold", "split"]
        )
        for split in ("train", "val", "test")
    }
    split_subjects = {
        split: set(frame["subject_id"].unique())
        for split, frame in split_frames.items()
    }
    if (
        split_subjects["train"] & split_subjects["val"]
        or split_subjects["train"] & split_subjects["test"]
        or split_subjects["val"] & split_subjects["test"]
    ):
        raise RuntimeError("Subject leakage detected between splits")
    if set().union(*split_subjects.values()) != set(subject_stats.index):
        raise RuntimeError("Split subjects do not cover the full cohort")

    summary = {
        "algorithm": "corrected_stratified_group_10fold_v1",
        "seed": args.seed,
        "fold_mapping": {"test": 0, "val": 1, "train": "2-9"},
        "source_cache_sha256": {
            split: sha256(path) for split, path in source_paths.items()
        },
        "source_combined": summarize(samples),
        "splits": {
            split: summarize(frame) for split, frame in split_frames.items()
        },
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    targets = {
        split: args.models_dir / f"{args.target_prefix}-{split}_samples.csv.gz"
        for split in ("train", "val", "test")
    }
    summary_path = args.manifest.with_suffix(".json")
    destinations = [args.manifest, summary_path, *targets.values()]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing fixed-split artifacts:\n  "
            + "\n  ".join(str(path) for path in existing)
        )

    atomic_csv(manifest, args.manifest)
    for split, frame in split_frames.items():
        atomic_csv(frame, targets[split], compression="gzip")
    atomic_json(summary, summary_path)
    print(f"Wrote fixed subject manifest: {args.manifest}")
    for split, path in targets.items():
        print(f"Wrote {split} cache: {path}")


if __name__ == "__main__":
    main()
