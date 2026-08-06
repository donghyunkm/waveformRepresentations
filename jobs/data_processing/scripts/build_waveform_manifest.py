"""Build a deterministic manifest of eligible MIMIC-III waveform records."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from _common import (
    DEFAULT_CHANNELS,
    MANIFEST_FIELDS,
    OUTPUT_ROOT,
    SOURCE_ROOT,
    atomic_csv_write,
    parse_channels,
)


MASTER_HEADER_RE = re.compile(
    r"^(p\d+-\d{4}-\d{2}-\d{2}-\d{2}-\d{2})\.hea$"
)


def find_master_headers(source_root: Path) -> list[Path]:
    """Use find(1) to avoid slow Python directory scans over segment files."""
    result = subprocess.run(
        [
            "find",
            str(source_root),
            "-mindepth",
            "3",
            "-maxdepth",
            "3",
            "-type",
            "f",
            "-name",
            "p*-*.hea",
            "!",
            "-name",
            "*n.hea",
            "-print0",
        ],
        check=True,
        capture_output=True,
    )
    return sorted(
        Path(raw.decode())
        for raw in result.stdout.split(b"\0")
        if raw and MASTER_HEADER_RE.match(Path(raw.decode()).name)
    )


def parse_master_header(path: Path) -> dict:
    """Read record dimensions and channel names without loading signal data."""
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Malformed master header: {path}")
    header = lines[0].split()
    if "/" not in header[0]:
        raise ValueError(f"Expected a multi-segment record: {path}")
    sampling_frequency = float(header[2])
    signal_length = int(header[3])
    layout_path = path.parent / f"{lines[1].split()[0]}.hea"
    layout_lines = layout_path.read_text(errors="replace").splitlines()
    available_channels = tuple(
        line.split()[-1] for line in layout_lines[1:] if line.split()
    )
    return {
        "sampling_frequency": sampling_frequency,
        "signal_length": signal_length,
        "available_channels": available_channels,
    }


def build_manifest(
    source_root: Path,
    channels: tuple[str, ...],
    require_numeric_pair: bool,
) -> list[dict]:
    """Return manifest rows for records advertising every requested channel."""
    rows = []
    for header_path in find_master_headers(source_root):
        numeric_path = header_path.with_name(f"{header_path.stem}n.hea")
        if require_numeric_pair and not numeric_path.exists():
            continue
        parsed = parse_master_header(header_path)
        available = parsed["available_channels"]
        if not set(channels).issubset(available):
            continue
        stay_id = header_path.stem
        rows.append(
            {
                "subject_id": header_path.parent.name,
                "stay_id": stay_id,
                "record_name": str(header_path.with_suffix("")),
                "header_path": str(header_path),
                "numeric_header_path": str(numeric_path)
                if numeric_path.exists()
                else "",
                "container_relpath": str(Path("containers") / f"{stay_id}.zarr.zip"),
                "sampling_frequency": parsed["sampling_frequency"],
                "signal_length": parsed["signal_length"],
                "duration_seconds": (
                    parsed["signal_length"] / parsed["sampling_frequency"]
                ),
                "available_channels": "|".join(available),
            }
        )
    rows.sort(key=lambda row: row["header_path"])
    for index, row in enumerate(rows):
        row["index"] = index
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=OUTPUT_ROOT / "manifests" / "waveform_manifest.csv",
    )
    parser.add_argument(
        "--channels",
        default=",".join(DEFAULT_CHANNELS),
        help="Comma-separated required channel names",
    )
    parser.add_argument(
        "--allow-missing-numeric-pair",
        action="store_true",
        help="Include waveform masters without a corresponding numeric header",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    manifest = args.manifest.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("Output root must not be inside the source waveform tree")

    channels = parse_channels(args.channels)
    rows = build_manifest(
        source_root=source_root,
        channels=channels,
        require_numeric_pair=not args.allow_missing_numeric_pair,
    )
    atomic_csv_write(manifest, rows, MANIFEST_FIELDS)
    print(f"Wrote {len(rows):,} eligible records to {manifest}")


if __name__ == "__main__":
    main()
