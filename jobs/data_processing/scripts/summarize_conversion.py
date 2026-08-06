"""Validate conversion task reports and write an aggregate JSON summary."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

from _common import OUTPUT_ROOT, read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=OUTPUT_ROOT / "manifests" / "waveform_manifest.csv",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--records-per-task",
        type=int,
        default=int(os.environ.get("RECORDS_PER_TASK", "10")),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Default: OUTPUT_ROOT/manifests/conversion_summary.json",
    )
    args = parser.parse_args()

    if args.records_per_task < 1:
        raise ValueError("records-per-task must be positive")
    rows = read_manifest(args.manifest)
    expected_tasks = math.ceil(len(rows) / args.records_per_task)
    status_dir = args.output_root.resolve() / "manifests" / "conversion_status"
    reports = sorted(status_dir.glob("task_*.csv"))
    expected_names = {f"task_{task_id:05d}.csv" for task_id in range(expected_tasks)}
    observed_names = {path.name for path in reports}
    missing_reports = sorted(expected_names - observed_names)
    unexpected_reports = sorted(observed_names - expected_names)

    state_counts: dict[str, int] = {}
    reported_indices: set[int] = set()
    duplicate_indices: list[int] = []
    mismatched_records: list[dict[str, str | int]] = []
    valid_windows = 0
    total_windows = 0
    output_bytes = 0
    failed: list[dict[str, str]] = []
    for report in reports:
        with report.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                index = int(row["index"])
                if index in reported_indices:
                    duplicate_indices.append(index)
                reported_indices.add(index)
                if index < len(rows) and row["stay_id"] != rows[index]["stay_id"]:
                    mismatched_records.append(
                        {
                            "index": index,
                            "expected_stay_id": rows[index]["stay_id"],
                            "reported_stay_id": row["stay_id"],
                        }
                    )
                state = row["state"]
                state_counts[state] = state_counts.get(state, 0) + 1
                if row["valid_windows"]:
                    valid_windows += int(row["valid_windows"])
                if row["total_windows"]:
                    total_windows += int(row["total_windows"])
                if row["bytes"]:
                    output_bytes += int(row["bytes"])
                if state == "failed":
                    failed.append(
                        {
                            "index": row["index"],
                            "stay_id": row["stay_id"],
                            "message": row["message"],
                        }
                    )

    expected_indices = set(range(len(rows)))
    missing_indices = sorted(expected_indices - reported_indices)
    unexpected_indices = sorted(reported_indices - expected_indices)
    summary = {
        "schema": "physiojepa.conversion-summary.v1",
        "manifest": str(args.manifest.resolve()),
        "records_per_task": args.records_per_task,
        "expected_records": len(rows),
        "expected_tasks": expected_tasks,
        "reports_found": len(reports),
        "state_counts": state_counts,
        "valid_windows_reported": valid_windows,
        "total_windows_reported": total_windows,
        "output_bytes_reported": output_bytes,
        "missing_reports": missing_reports,
        "unexpected_reports": unexpected_reports,
        "missing_indices": missing_indices,
        "unexpected_indices": unexpected_indices,
        "duplicate_indices": sorted(set(duplicate_indices)),
        "mismatched_records": mismatched_records,
        "failures": failed,
    }
    summary_path = (
        args.summary
        if args.summary is not None
        else args.output_root.resolve()
        / "manifests"
        / "conversion_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(
        f".{summary_path.name}.partial.{os.getpid()}"
    )
    try:
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, summary_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(json.dumps(summary, indent=2, sort_keys=True))
    if (
        missing_reports
        or unexpected_reports
        or missing_indices
        or unexpected_indices
        or duplicate_indices
        or mismatched_records
        or failed
    ):
        raise SystemExit(
            f"Conversion is incomplete or inconsistent; see {summary_path}"
        )
    print(f"Conversion reports are complete; summary: {summary_path}")


if __name__ == "__main__":
    main()
