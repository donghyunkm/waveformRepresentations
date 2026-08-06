"""Convert one slice of a waveform manifest into per-stay Zarr ZipStores."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import wfdb
import zarr
from numcodecs import Blosc

from _common import (
    DEFAULT_CHANNELS,
    OUTPUT_ROOT,
    atomic_csv_write,
    open_zip_group,
    parse_channels,
    read_manifest,
    task_rows,
)


STATUS_FIELDS = (
    "index",
    "stay_id",
    "state",
    "container_path",
    "valid_windows",
    "total_windows",
    "bytes",
    "elapsed_seconds",
    "message",
)


def largest_constant_or_nan_fraction(values: np.ndarray) -> float:
    """Return the largest frequency of one value, treating NaN as a value."""
    if values.size == 0:
        return 1.0
    _, counts = np.unique(values, return_counts=True, equal_nan=True)
    return float(counts.max() / values.size)


def existing_container_metadata(
    path: Path, stay_id: str, channels: tuple[str, ...]
) -> dict | None:
    """Return metadata when an existing atomic output is complete."""
    if not path.is_file():
        return None
    try:
        with open_zip_group(path, mode="r") as store:
            root = zarr.open_consolidated(store, mode="r")
            recognized = (
                root.attrs.get("schema") == "physiojepa.waveform.v1"
                and root.attrs.get("stay_id") == stay_id
                and all(channel in root for channel in channels)
                and "quality/valid_window" in root
                and "valid_windows" in root.attrs
                and "total_windows" in root.attrs
            )
            return root.attrs.asdict() if recognized else None
    except Exception:
        return None


def read_window(
    record_name: str,
    start: int,
    end: int,
    channels: tuple[str, ...],
) -> np.ndarray:
    """Read a window and fill channels absent from a segment with NaNs."""
    output = np.full((len(channels), end - start), np.nan, dtype=np.float32)
    try:
        record = wfdb.rdrecord(
            record_name,
            sampfrom=start,
            sampto=end,
            channel_names=list(channels),
            m2s=True,
        )
    except (ValueError, IndexError):
        return output
    if record.p_signal is None:
        return output
    for output_index, channel in enumerate(channels):
        if channel in record.sig_name:
            source_index = record.sig_name.index(channel)
            output[output_index] = record.p_signal[:, source_index].astype(
                np.float32, copy=False
            )
    return output


def convert_record(
    row: dict[str, str],
    output_root: Path,
    channels: tuple[str, ...],
    window_seconds: int,
    threshold: float,
    overwrite: bool,
) -> dict:
    """Convert one record atomically and return a status row."""
    started = time.monotonic()
    final_path = output_root / row["container_relpath"]
    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = final_path.with_name(f".{final_path.name}.partial")

    existing = existing_container_metadata(final_path, row["stay_id"], channels)
    if existing is not None and not overwrite:
        return {
            "index": row["index"],
            "stay_id": row["stay_id"],
            "state": "skipped_complete",
            "container_path": str(final_path),
            "valid_windows": existing["valid_windows"],
            "total_windows": existing["total_windows"],
            "bytes": final_path.stat().st_size,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "message": "",
        }
    if final_path.exists() and not overwrite:
        raise RuntimeError(
            f"Refusing to replace an unrecognized container without --overwrite: "
            f"{final_path}"
        )
    if partial_path.exists():
        partial_path.unlink()
    if overwrite and final_path.exists():
        final_path.unlink()

    record_name = row["record_name"]
    sampling_frequency = float(row["sampling_frequency"])
    signal_length = int(row["signal_length"])
    if not sampling_frequency.is_integer():
        raise ValueError(
            f"Non-integer sampling frequency for {record_name}: "
            f"{sampling_frequency}"
        )
    window_samples = int(window_seconds * sampling_frequency)
    total_windows = signal_length // window_samples
    compressor = Blosc(
        cname="zstd",
        clevel=3,
        shuffle=Blosc.BITSHUFFLE,
    )

    try:
        with open_zip_group(partial_path, mode="w") as store:
            root = zarr.group(store=store, overwrite=True)
            arrays = {
                channel: root.create_dataset(
                    channel,
                    shape=(signal_length,),
                    chunks=(window_samples,),
                    dtype="f4",
                    compressor=compressor,
                    fill_value=np.nan,
                    overwrite=True,
                )
                for channel in channels
            }
            for channel, array in arrays.items():
                array.attrs.update(
                    {
                        "sampling_frequency": sampling_frequency,
                        "channel": channel,
                    }
                )

            quality = root.create_group("quality")
            valid_array = quality.create_dataset(
                "valid_window",
                shape=(total_windows,),
                chunks=(min(max(total_windows, 1), 4096),),
                dtype="bool",
                compressor=compressor,
                fill_value=False,
                overwrite=True,
            )
            constant_array = quality.create_dataset(
                "max_constant_nan_fraction",
                shape=(total_windows, len(channels)),
                chunks=(min(max(total_windows, 1), 1024), len(channels)),
                dtype="f4",
                compressor=compressor,
                fill_value=np.nan,
                overwrite=True,
            )
            nan_array = quality.create_dataset(
                "nan_fraction",
                shape=(total_windows, len(channels)),
                chunks=(min(max(total_windows, 1), 1024), len(channels)),
                dtype="f4",
                compressor=compressor,
                fill_value=np.nan,
                overwrite=True,
            )

            valid_windows = 0
            for start in range(0, signal_length, window_samples):
                end = min(start + window_samples, signal_length)
                values = read_window(record_name, start, end, channels)
                for channel_index, channel in enumerate(channels):
                    arrays[channel][start:end] = values[channel_index]
                if end - start != window_samples:
                    continue
                window_index = start // window_samples
                constant_fractions = np.asarray(
                    [
                        largest_constant_or_nan_fraction(channel_values)
                        for channel_values in values
                    ],
                    dtype=np.float32,
                )
                nan_fractions = np.isnan(values).mean(axis=1).astype(np.float32)
                is_valid = bool(np.all(constant_fractions <= threshold))
                constant_array[window_index] = constant_fractions
                nan_array[window_index] = nan_fractions
                valid_array[window_index] = is_valid
                valid_windows += int(is_valid)

            root.attrs.update(
                {
                    "schema": "physiojepa.waveform.v1",
                    "stay_id": row["stay_id"],
                    "subject_id": row["subject_id"],
                    "source_record": record_name,
                    "sampling_frequency": sampling_frequency,
                    "signal_length": signal_length,
                    "duration_seconds": signal_length / sampling_frequency,
                    "Duration": signal_length / sampling_frequency,
                    "channels": list(channels),
                    "quality_window_seconds": window_seconds,
                    "constant_nan_threshold": threshold,
                    "valid_windows": valid_windows,
                    "total_windows": total_windows,
                }
            )
            zarr.consolidate_metadata(store)

        os.replace(partial_path, final_path)
        return {
            "index": row["index"],
            "stay_id": row["stay_id"],
            "state": "converted",
            "container_path": str(final_path),
            "valid_windows": valid_windows,
            "total_windows": total_windows,
            "bytes": final_path.stat().st_size,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "message": "",
        }
    except Exception:
        if partial_path.exists():
            partial_path.unlink()
        raise


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
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    parser.add_argument("--window-seconds", type=int, default=1800)
    parser.add_argument("--constant-nan-threshold", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.constant_nan_threshold <= 1:
        raise ValueError("constant-nan threshold must be in (0, 1]")
    channels = parse_channels(args.channels)
    rows = read_manifest(args.manifest)
    selected = task_rows(rows, args.task_id, args.records_per_task)
    if not selected:
        print(f"Task {args.task_id} has no assigned records")
        return

    status_rows = []
    failures = 0
    for row in selected:
        try:
            status = convert_record(
                row=row,
                output_root=args.output_root.resolve(),
                channels=channels,
                window_seconds=args.window_seconds,
                threshold=args.constant_nan_threshold,
                overwrite=args.overwrite,
            )
            print(
                f"[{status['state']}] {row['stay_id']} "
                f"valid={status['valid_windows']}/{status['total_windows']} "
                f"elapsed={status['elapsed_seconds']}s"
            )
        except Exception as error:
            failures += 1
            status = {
                "index": row["index"],
                "stay_id": row["stay_id"],
                "state": "failed",
                "container_path": str(
                    args.output_root.resolve() / row["container_relpath"]
                ),
                "valid_windows": "",
                "total_windows": "",
                "bytes": "",
                "elapsed_seconds": "",
                "message": f"{type(error).__name__}: {error}",
            }
            print(f"[failed] {row['stay_id']}: {status['message']}")
        status_rows.append(status)

    status_path = (
        args.output_root.resolve()
        / "manifests"
        / "conversion_status"
        / f"task_{args.task_id:05d}.csv"
    )
    atomic_csv_write(status_path, status_rows, STATUS_FIELDS)
    if failures:
        raise SystemExit(f"{failures} record(s) failed; see {status_path}")


if __name__ == "__main__":
    main()
