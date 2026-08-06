"""Build a reproducible subject-level subset from fixed FCN sample caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_ROOT = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA"
)
SOURCE_MODELS_DIR = OUTPUT_ROOT / "models" / "fcn_hypotension_paper"
TARGET_MODELS_DIR = OUTPUT_ROOT / "models" / "fcn_hypotension_subset10"
SOURCE_PREFIX = (
    "zipstore_ABP_II_PLETH_125Hz_1800sec_"
    "hypotension_fixed_subject_split_v1"
)
TARGET_PREFIX = (
    "zipstore_ABP_II_PLETH_125Hz_1800sec_"
    "hypotension_fixed_subject_split_subset10_v1"
)
DEFAULT_MANIFEST = (
    OUTPUT_ROOT
    / "manifests"
    / "hypotension_subject_split_fixed_subset10_v1.csv"
)
SPLITS = ("train", "val", "test")
IDENTITY_COLUMNS = ["file_path", "start_idx", "end_idx"]
BALANCE_COLUMNS = [
    "positive_events",
    "negative_events",
    "samples",
    "icu_stays",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(
    frame: pd.DataFrame,
    destination: Path,
    compression: str | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    frame.to_csv(partial, index=False, compression=compression)
    os.replace(partial, destination)


def atomic_json(data: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(partial, destination)


def summarize(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["outcome_val_300sec"].value_counts().to_dict()
    return {
        "samples": int(len(frame)),
        "positive_events": int(counts.get(1, 0)),
        "negative_events": int(counts.get(0, 0)),
        "icu_stays": int(frame["file_path"].nunique()),
        "patients": int(frame["subject_id"].nunique()),
    }


def subject_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    stats = frame.groupby("subject_id").agg(
        positive_events=("outcome_val_300sec", "sum"),
        samples=("outcome_val_300sec", "size"),
        icu_stays=("file_path", "nunique"),
    )
    stats["negative_events"] = stats["samples"] - stats["positive_events"]
    return stats


def choose_subjects(
    stats: pd.DataFrame,
    fraction: float,
    seed: int,
    candidates: int,
) -> tuple[pd.Index, dict]:
    """Choose a fixed-size stratified subset closest to aggregate targets."""
    n_subjects = len(stats)
    n_select = max(2, int(round(n_subjects * fraction)))
    n_select = min(n_select, n_subjects)

    positive = np.flatnonzero(stats["positive_events"].to_numpy() > 0)
    negative = np.flatnonzero(stats["positive_events"].to_numpy() == 0)
    n_positive = int(round(n_select * len(positive) / n_subjects))
    if len(positive):
        n_positive = min(max(n_positive, 1), len(positive))
    n_negative = n_select - n_positive
    if n_negative > len(negative):
        deficit = n_negative - len(negative)
        n_negative = len(negative)
        n_positive += deficit
    if n_positive > len(positive):
        deficit = n_positive - len(positive)
        n_positive = len(positive)
        n_negative += deficit

    values = stats[BALANCE_COLUMNS].to_numpy(dtype=np.float64)
    target = values.sum(axis=0) * (n_select / n_subjects)
    scale = np.maximum(target, 1.0)
    rng = np.random.default_rng(seed)
    best_indices = None
    best_score = np.inf

    for _ in range(candidates):
        selected_positive = (
            rng.choice(positive, n_positive, replace=False)
            if n_positive
            else np.empty(0, dtype=int)
        )
        selected_negative = (
            rng.choice(negative, n_negative, replace=False)
            if n_negative
            else np.empty(0, dtype=int)
        )
        selected = np.concatenate([selected_positive, selected_negative])
        observed = values[selected].sum(axis=0)
        score = float(np.mean(np.square((observed - target) / scale)))
        if score < best_score:
            best_score = score
            best_indices = selected.copy()

    if best_indices is None:
        raise RuntimeError("Subset candidate search produced no selection")
    selected_stats = stats.iloc[best_indices]
    diagnostics = {
        "source_subjects": n_subjects,
        "selected_subjects": int(len(selected_stats)),
        "source_positive_subjects": int(len(positive)),
        "selected_positive_subjects": int(
            (selected_stats["positive_events"] > 0).sum()
        ),
        "target_fraction": fraction,
        "realized_aggregate_fractions": {
            column: float(selected_stats[column].sum() / stats[column].sum())
            for column in BALANCE_COLUMNS
        },
        "candidate_searches": candidates,
        "balance_score": best_score,
    }
    return selected_stats.index, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-models-dir", type=Path, default=SOURCE_MODELS_DIR)
    parser.add_argument("--target-models-dir", type=Path, default=TARGET_MODELS_DIR)
    parser.add_argument("--source-prefix", default=SOURCE_PREFIX)
    parser.add_argument("--target-prefix", default=TARGET_PREFIX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.fraction < 1:
        raise ValueError("--fraction must be strictly between 0 and 1")
    if args.candidates < 1:
        raise ValueError("--candidates must be positive")

    source_paths = {
        split: args.source_models_dir
        / f"{args.source_prefix}-{split}_samples.csv.gz"
        for split in SPLITS
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    source_frames = {}
    source_stats = {}
    for split, path in source_paths.items():
        frame = pd.read_csv(path)
        frame = frame.loc[:, ~frame.columns.str.match(r"^Unnamed(:.*)?$")].copy()
        source_frames[split] = frame
        source_stats[split] = subject_statistics(frame)

    source_subjects = {
        split: set(frame["subject_id"].unique())
        for split, frame in source_frames.items()
    }
    if (
        source_subjects["train"] & source_subjects["val"]
        or source_subjects["train"] & source_subjects["test"]
        or source_subjects["val"] & source_subjects["test"]
    ):
        raise RuntimeError("Source caches contain subject leakage")

    selected_subjects = {}
    selection_diagnostics = {}
    for offset, split in enumerate(SPLITS):
        selected_subjects[split], selection_diagnostics[split] = choose_subjects(
            source_stats[split],
            fraction=args.fraction,
            seed=args.seed + offset,
            candidates=args.candidates,
        )

    subset_frames = {
        split: source_frames[split]
        .loc[source_frames[split]["subject_id"].isin(selected_subjects[split])]
        .copy()
        for split in SPLITS
    }
    combined = pd.concat(
        [
            frame.assign(_split=split)
            for split, frame in subset_frames.items()
        ],
        ignore_index=True,
    )
    duplicates = combined.duplicated(IDENTITY_COLUMNS, keep=False)
    if duplicates.any():
        raise RuntimeError(
            f"Subset contains {int(duplicates.sum())} duplicate sample identities"
        )
    for split, frame in subset_frames.items():
        observed_subjects = set(frame["subject_id"].unique())
        expected_subjects = set(selected_subjects[split])
        if observed_subjects != expected_subjects:
            raise RuntimeError(f"{split} subset subject filtering was incomplete")
        if set(frame["outcome_val_300sec"].unique()) != {0, 1}:
            raise RuntimeError(f"{split} subset does not contain both classes")

    manifest_parts = []
    for split in SPLITS:
        part = source_stats[split].loc[selected_subjects[split]].reset_index()
        part["split"] = split
        part["subset_fraction"] = args.fraction
        part["seed"] = args.seed
        part["algorithm"] = "stratified_candidate_balance_subset_v1"
        manifest_parts.append(part)
    manifest = pd.concat(manifest_parts, ignore_index=True)
    manifest.sort_values(["split", "subject_id"], inplace=True)

    summary = {
        "algorithm": "stratified_candidate_balance_subset_v1",
        "fraction": args.fraction,
        "seed": args.seed,
        "source_cache_sha256": {
            split: sha256(path) for split, path in source_paths.items()
        },
        "source_splits": {
            split: summarize(frame) for split, frame in source_frames.items()
        },
        "selection": selection_diagnostics,
        "subset_splits": {
            split: summarize(frame) for split, frame in subset_frames.items()
        },
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    target_paths = {
        split: args.target_models_dir
        / f"{args.target_prefix}-{split}_samples.csv.gz"
        for split in SPLITS
    }
    summary_path = args.manifest.with_suffix(".json")
    destinations = [args.manifest, summary_path, *target_paths.values()]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing subset artifacts:\n  "
            + "\n  ".join(str(path) for path in existing)
        )

    atomic_csv(manifest, args.manifest)
    for split, frame in subset_frames.items():
        atomic_csv(frame, target_paths[split], compression="gzip")
    atomic_json(summary, summary_path)
    print(f"Wrote subset manifest: {args.manifest}")
    for split, path in target_paths.items():
        print(f"Wrote {split} subset cache: {path}")


if __name__ == "__main__":
    main()
