"""Shared helpers for the MIMIC-III waveform processing jobs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

from zarr.storage import ZipStore


SOURCE_ROOT = Path("/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched")
OUTPUT_ROOT = Path(
    "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA"
)
DEFAULT_CHANNELS = ("ABP", "II", "PLETH")
MANIFEST_FIELDS = (
    "index",
    "subject_id",
    "stay_id",
    "record_name",
    "header_path",
    "numeric_header_path",
    "container_relpath",
    "sampling_frequency",
    "signal_length",
    "duration_seconds",
    "available_channels",
)


def parse_channels(value: str | Iterable[str]) -> tuple[str, ...]:
    """Return a non-empty tuple of channel names."""
    if isinstance(value, str):
        channels = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        channels = tuple(str(part).strip() for part in value if str(part).strip())
    if not channels:
        raise ValueError("At least one channel is required")
    if len(channels) != len(set(channels)):
        raise ValueError(f"Duplicate channels are not allowed: {channels}")
    return channels


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read and minimally validate a waveform manifest."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or ())
    missing = set(MANIFEST_FIELDS) - fieldnames
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {sorted(missing)}")
    for expected, row in enumerate(rows):
        if int(row["index"]) != expected:
            raise ValueError(
                f"Manifest index mismatch at row {expected}: {row['index']}"
            )
    return rows


def task_rows(
    rows: list[dict[str, str]], task_id: int, records_per_task: int
) -> list[dict[str, str]]:
    """Select the contiguous manifest slice assigned to an array task."""
    if task_id < 0:
        raise ValueError("task_id must be non-negative")
    if records_per_task < 1:
        raise ValueError("records_per_task must be positive")
    start = task_id * records_per_task
    return rows[start : start + records_per_task]


def atomic_csv_write(
    path: str | Path, rows: Iterable[dict], fieldnames: Iterable[str]
) -> None:
    """Write a CSV atomically in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def open_zip_group(path: str | Path, mode: str = "r"):
    """Return a ZipStore; callers should use it as a context manager."""
    return ZipStore(
        str(path),
        mode=mode,
        compression=0,
        allowZip64=True,
    )
