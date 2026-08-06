"""Shared helpers for cluster-ready native PhysioJEPA experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit


OUTPUT_ROOT = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA"
).resolve()


def load_config(default_name: str) -> tuple[dict[str, Any], Path]:
    """Load a YAML configuration, honoring ``PHYSIOJEPA_CONFIG``."""
    config_path = Path(os.environ.get("PHYSIOJEPA_CONFIG", default_name)).resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    return config, config_path


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def data_fingerprint(config: dict[str, Any]) -> str:
    """Fingerprint only settings that determine the pretraining split/caches."""
    path_keys = (
        "containers_dir",
        "waveform_manifest_path",
        "downstream_subject_split_path",
        "pretraining_manifest_path",
        "sample_cache_dir",
        "dataset_filename",
    )
    payload = {
        "paths": {key: config["paths"][key] for key in path_keys},
        "split": config["split"],
        "dataset": config["dataset"],
    }
    return config_fingerprint(payload)


def require_output_path(path: str | Path) -> Path:
    """Require derived artifacts to remain under the approved output root."""
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Derived artifact path must be under {OUTPUT_ROOT}: {resolved}"
        ) from exc
    return resolved


def atomic_write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    path = require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "".join(path.suffixes) or ".tmp"
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=suffix,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        frame.to_csv(
            temporary_path,
            index=False,
            compression="gzip" if path.name.endswith(".gz") else None,
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".json",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _has_required_channels(value: str, required_channels: list[str]) -> bool:
    available = set(str(value).split("|"))
    return set(required_channels).issubset(available)


def build_pretraining_split(config: dict[str, Any]) -> pd.DataFrame:
    """Build a subject-disjoint, downstream-leakage-safe JEPA record split."""
    paths = config["paths"]
    dataset = config["dataset"]
    split_config = config["split"]

    waveform_manifest = pd.read_csv(paths["waveform_manifest_path"])
    subject_split = pd.read_csv(paths["downstream_subject_split_path"])
    required_channels = list(dataset["channels"])
    minimum_duration = int(dataset["sample_seq_len_seconds"])

    required_manifest_columns = {
        "subject_id",
        "stay_id",
        "container_relpath",
        "duration_seconds",
        "available_channels",
    }
    missing = required_manifest_columns.difference(waveform_manifest.columns)
    if missing:
        raise ValueError(f"Waveform manifest is missing columns: {sorted(missing)}")
    if not {"subject_id", "split"}.issubset(subject_split.columns):
        raise ValueError("Downstream split must contain subject_id and split columns")

    excluded_splits = set(split_config.get("exclude_downstream_splits", ["val", "test"]))
    excluded_subjects = set(
        subject_split.loc[
            subject_split["split"].isin(excluded_splits), "subject_id"
        ].astype(str)
    )

    frame = waveform_manifest.copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame = frame.loc[
        frame["available_channels"].map(
            lambda value: _has_required_channels(value, required_channels)
        )
    ].copy()
    frame = frame.loc[frame["duration_seconds"] >= minimum_duration].copy()
    frame = frame.loc[~frame["subject_id"].isin(excluded_subjects)].copy()

    containers_dir = Path(paths["containers_dir"]).resolve()
    frame["file"] = frame["container_relpath"].map(
        lambda value: str((OUTPUT_ROOT / str(value)).resolve())
        if not Path(str(value)).is_absolute()
        else str(Path(str(value)).resolve())
    )
    if not all(Path(path).parent == containers_dir for path in frame["file"]):
        raise ValueError("Manifest contains a container outside the configured directory")

    missing_files = [path for path in frame["file"] if not Path(path).is_file()]
    if missing_files:
        preview = "\n  ".join(missing_files[:10])
        raise FileNotFoundError(f"Missing JEPA containers:\n  {preview}")

    subjects = sorted(frame["subject_id"].unique())
    if len(subjects) < 2:
        raise ValueError("At least two eligible subjects are required")
    validation_fraction = float(split_config.get("validation_fraction", 0.05))
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=int(config["run_config"]["random_state"]),
    )
    train_indices, validation_indices = next(
        splitter.split(subjects, groups=subjects)
    )
    train_subjects = {subjects[index] for index in train_indices}
    validation_subjects = {subjects[index] for index in validation_indices}
    if train_subjects.intersection(validation_subjects):
        raise AssertionError("JEPA train and validation subjects overlap")

    frame["pretrain_split"] = frame["subject_id"].map(
        lambda subject: "train" if subject in train_subjects else "val"
    )
    frame = frame[
        [
            "subject_id",
            "stay_id",
            "file",
            "pretrain_split",
            "sampling_frequency",
            "signal_length",
            "duration_seconds",
        ]
    ].sort_values(["pretrain_split", "subject_id", "stay_id"])
    return frame.reset_index(drop=True)


def load_pretraining_split(config: dict[str, Any]) -> pd.DataFrame:
    path = require_output_path(config["paths"]["pretraining_manifest_path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Pretraining manifest does not exist: {path}. Run sample preparation first."
        )
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Pretraining metadata does not exist: {metadata_path}. "
            "Run sample preparation first."
        )
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    expected_fingerprint = data_fingerprint(config)
    recorded_fingerprint = metadata.get("data_fingerprint")
    if recorded_fingerprint is None:
        raise ValueError(
            "Pretraining metadata predates data-fingerprint validation. "
            "Regenerate or explicitly migrate the sample metadata."
        )
    if recorded_fingerprint != expected_fingerprint:
        raise ValueError(
            "Pretraining manifest/cache fingerprint does not match the active "
            "dataset and split configuration"
        )
    recorded_manifest = Path(metadata.get("pretraining_manifest_path", "")).resolve()
    if recorded_manifest != path:
        raise ValueError(
            f"Pretraining metadata points to {recorded_manifest}, expected {path}"
        )
    for split in ("train", "val"):
        expected_cache = sample_cache_path(config, split)
        recorded_cache = Path(metadata.get("splits", {}).get(split, {}).get("path", "")).resolve()
        if recorded_cache != expected_cache or not expected_cache.is_file():
            raise ValueError(
                f"Pretraining {split} cache metadata is stale or missing: "
                f"recorded={recorded_cache}, expected={expected_cache}"
            )
    frame = pd.read_csv(path)
    required = {"subject_id", "stay_id", "file", "pretrain_split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Pretraining manifest is missing columns: {sorted(missing)}")
    return frame


def sample_cache_path(config: dict[str, Any], split: str) -> Path:
    cache_dir = require_output_path(config["paths"]["sample_cache_dir"])
    dataset_name = config["paths"]["dataset_filename"]
    return cache_dir / f"{dataset_name}-{split}_samples.csv.gz"
