"""Extract minute-level ABP outcomes from one array slice of ZipStores."""

from __future__ import annotations

import argparse
import csv
import gzip
import os
from pathlib import Path

import numpy as np
import zarr

from physiojepa.signal import preprocess_abp_signal

from _common import OUTPUT_ROOT, open_zip_group, read_manifest, task_rows


FIELDS = (
    "Time Stamp (seconds)",
    "SBP (mmHg)",
    "DBP (mmHg)",
    "MBP (mmHg)",
    "hr_bpm",
    "hypotension",
    "shock_index",
    "shock_index_label",
    "file_name",
    "file_path",
    "subject_id",
    "date",
)


def date_from_stay_id(stay_id: str) -> str:
    parts = stay_id.split("-")
    if len(parts) != 6:
        raise ValueError(f"Unexpected stay identifier: {stay_id}")
    return f"{parts[2]}/{parts[3]}/{parts[1]} {parts[4]}:{parts[5]}"


def label_minute(abp: np.ndarray, sampling_frequency: float) -> dict:
    median_sys, median_dias, median_map, median_hr = preprocess_abp_signal(
        abp, sampling_frequency
    )
    if np.isfinite(median_sys) and np.isfinite(median_map):
        hypotension = int((median_map <= 65) or (median_sys <= 90))
    else:
        hypotension = np.nan
    if np.isfinite(median_sys) and np.isfinite(median_hr) and median_sys != 0:
        shock_index = median_hr / median_sys
        shock_label = int(shock_index >= 0.9)
    else:
        shock_index = np.nan
        shock_label = np.nan
    return {
        "SBP (mmHg)": median_sys,
        "DBP (mmHg)": median_dias,
        "MBP (mmHg)": median_map,
        "hr_bpm": median_hr,
        "hypotension": hypotension,
        "shock_index": shock_index,
        "shock_index_label": shock_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=OUTPUT_ROOT / "manifests" / "waveform_manifest.csv",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--records-per-task",
        type=int,
        default=int(os.environ.get("RECORDS_PER_TASK", "10")),
    )
    parser.add_argument("--step-seconds", type=int, default=60)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    selected = task_rows(rows, args.task_id, args.records_per_task)
    if not selected:
        print(f"Task {args.task_id} has no assigned records")
        return

    shard_path = (
        args.output_root.resolve()
        / "labels"
        / "minute_shards"
        / f"task_{args.task_id:05d}.csv.gz"
    )
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard_path.with_name(f".{shard_path.name}.partial.{os.getpid()}")
    written = 0
    try:
        with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in selected:
                container_path = (
                    args.output_root.resolve() / row["container_relpath"]
                )
                if not container_path.is_file():
                    raise FileNotFoundError(container_path)
                with open_zip_group(container_path, mode="r") as store:
                    root = zarr.open_consolidated(store, mode="r")
                    sampling_frequency = float(root.attrs["sampling_frequency"])
                    step_samples = int(args.step_seconds * sampling_frequency)
                    abp = root["ABP"]
                    for start in range(0, len(abp), step_samples):
                        end = min(start + step_samples, len(abp))
                        values = np.asarray(abp[start:end], dtype=np.float64)
                        labels = label_minute(values, sampling_frequency)
                        writer.writerow(
                            {
                                "Time Stamp (seconds)": start
                                / sampling_frequency,
                                **labels,
                                "file_name": row["stay_id"],
                                "file_path": str(container_path),
                                "subject_id": row["subject_id"],
                                "date": date_from_stay_id(row["stay_id"]),
                            }
                        )
                        written += 1
        os.replace(temporary, shard_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Wrote {written:,} minute rows to {shard_path}")


if __name__ == "__main__":
    main()
