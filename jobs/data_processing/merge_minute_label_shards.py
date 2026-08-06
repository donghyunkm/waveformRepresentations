"""Stream minute-level gzip CSV shards into one deterministic gzip CSV."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import os
from pathlib import Path

from _common import OUTPUT_ROOT, read_manifest
from extract_minute_labels_array import FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=OUTPUT_ROOT / "labels" / "minute_shards",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=OUTPUT_ROOT / "manifests" / "waveform_manifest.csv",
    )
    parser.add_argument(
        "--records-per-task",
        type=int,
        default=int(os.environ.get("RECORDS_PER_TASK", "10")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "labels" / "hypotension_si_labels_mimic.csv.gz",
    )
    args = parser.parse_args()

    if args.records_per_task < 1:
        raise ValueError("records-per-task must be positive")
    manifest_rows = read_manifest(args.manifest)
    expected_tasks = math.ceil(len(manifest_rows) / args.records_per_task)
    expected_names = {
        f"task_{task_id:05d}.csv.gz" for task_id in range(expected_tasks)
    }
    shards = sorted(args.shard_dir.glob("task_*.csv.gz"))
    if not shards:
        raise FileNotFoundError(f"No label shards found in {args.shard_dir}")
    observed_names = {path.name for path in shards}
    missing = sorted(expected_names - observed_names)
    unexpected = sorted(observed_names - expected_names)
    if missing or unexpected:
        raise RuntimeError(
            f"Label shard set does not match the manifest: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial.{os.getpid()}")
    rows_written = 0
    try:
        with gzip.open(temporary, "wt", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            for shard in shards:
                with gzip.open(shard, "rt", newline="", encoding="utf-8") as source:
                    reader = csv.DictReader(source)
                    if tuple(reader.fieldnames or ()) != FIELDS:
                        raise ValueError(f"Unexpected columns in {shard}")
                    for row in reader:
                        writer.writerow(row)
                        rows_written += 1
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"Merged {len(shards):,} shards and {rows_written:,} rows into "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
