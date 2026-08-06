#!/usr/bin/env python
"""Plot a supervised waveform input window and its following prediction horizon.

Example
-------
python jobs/plot_waveforms.py \\
    --split val --row 0 \\
    --output /tmp/physiojepa-waveform-val-0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from zarr.storage import ZipStore


OUTPUT_ROOT = Path("/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA")
DEFAULT_CONTAINERS_DIR = OUTPUT_ROOT / "containers"
DEFAULT_SAMPLE_CACHE_DIR = OUTPUT_ROOT / "models/fcn_hypotension_paper"
DEFAULT_DATASET_NAME = (
    "zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1"
)
CHANNELS = ("ABP", "II", "PLETH")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ABP, ECG lead II, and PLETH across input and forecast horizons."
    )
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--row", type=int, default=0, help="Row in the sample-index CSV.")
    parser.add_argument("--sample-index", type=Path, default=None)
    parser.add_argument("--sample-cache-dir", type=Path, default=DEFAULT_SAMPLE_CACHE_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--containers-dir", type=Path, default=DEFAULT_CONTAINERS_DIR)
    parser.add_argument("--frequency", type=int, default=125)
    parser.add_argument("--input-seconds", type=int, default=1800)
    parser.add_argument("--prediction-seconds", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path. If omitted, display the figure interactively.",
    )
    parser.add_argument(
        "--plot-stride",
        type=int,
        default=1,
        help="Plot every Nth sample to reduce rendering cost; data slicing is unchanged.",
    )
    return parser.parse_args()


def resolve_container(file_value: str | Path, containers_dir: Path) -> Path:
    """Resolve cached file paths against the cluster ZipStore directory."""
    candidate = Path(str(file_value))
    options = [candidate]
    if not candidate.is_absolute():
        options.extend((containers_dir / candidate, containers_dir / candidate.name))
    else:
        options.append(containers_dir / candidate.name)
    if candidate.suffix != ".zip":
        options.extend(
            (
                containers_dir / f"{candidate.name}.zip",
                containers_dir / f"{candidate.stem}.zarr.zip",
            )
        )
    for option in options:
        if option.exists():
            return option
    raise FileNotFoundError(f"Could not resolve waveform container: {file_value}")


def read_waveforms(
    container_path: Path,
    channels: tuple[str, ...],
    start_idx: int,
    end_idx: int,
    prediction_end_idx: int,
) -> dict[str, np.ndarray]:
    """Read each channel through the input and prediction horizon."""
    if container_path.name.endswith(".zip"):
        store = ZipStore(str(container_path), mode="r")
        root = zarr.open(store, mode="r")
    else:
        store = None
        root = zarr.open(str(container_path), mode="r")
    try:
        return {
            channel: np.asarray(root[channel][start_idx:prediction_end_idx])
            for channel in channels
        }
    finally:
        if store is not None:
            store.close()


def plot_waveforms(
    waveforms: dict[str, np.ndarray],
    frequency: int,
    input_seconds: float,
    prediction_seconds: float,
    title: str,
    plot_stride: int = 1,
):
    """Create the input/prediction horizon plot and return its figure."""
    if plot_stride < 1:
        raise ValueError("plot_stride must be at least 1")
    length = len(next(iter(waveforms.values())))
    time_seconds = np.arange(length) / frequency
    input_boundary = input_seconds
    fig, axes = plt.subplots(
        len(waveforms), 1, figsize=(18, 9), sharex=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, (channel, waveform) in zip(axes, waveforms.items()):
        axis.plot(
            time_seconds[::plot_stride],
            waveform[::plot_stride],
            linewidth=0.5,
            color="black",
        )
        axis.axvspan(0, input_boundary, color="tab:blue", alpha=0.08)
        axis.axvspan(
            input_boundary,
            input_boundary + prediction_seconds,
            color="tab:orange",
            alpha=0.10,
        )
        axis.axvline(input_boundary, color="tab:red", linestyle="--", linewidth=1.2)
        axis.set_ylabel(channel)
        axis.grid(alpha=0.25)
    axes[0].axvspan(0, input_boundary, color="tab:blue", alpha=0.08, label="input horizon")
    axes[0].axvspan(
        input_boundary,
        input_boundary + prediction_seconds,
        color="tab:orange",
        alpha=0.10,
        label="prediction horizon",
    )
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel("Seconds from input-window start")
    return fig


def main() -> None:
    args = parse_args()
    if args.frequency <= 0 or args.input_seconds <= 0 or args.prediction_seconds < 0:
        raise ValueError("frequency and input-seconds must be positive; prediction-seconds cannot be negative")

    sample_index = args.sample_index or (
        args.sample_cache_dir / f"{args.dataset_name}-{args.split}_samples.csv.gz"
    )
    if not sample_index.exists():
        raise FileNotFoundError(
            f"Sample index is missing: {sample_index}. "
            "Run the sample-index preparation stage first."
        )
    samples = pd.read_csv(sample_index)
    if not 0 <= args.row < len(samples):
        raise IndexError(f"row {args.row} is outside the sample index (0..{len(samples) - 1})")
    required = {"start_idx", "end_idx"}
    file_column = "file_path" if "file_path" in samples.columns else "file"
    required.add(file_column)
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Sample index is missing columns: {sorted(missing)}")

    sample = samples.iloc[args.row]
    start_idx = int(sample["start_idx"])
    end_idx = int(sample["end_idx"])
    expected_input_samples = args.input_seconds * args.frequency
    actual_input_samples = end_idx - start_idx
    prediction_end_idx = end_idx + args.prediction_seconds * args.frequency
    container_path = resolve_container(sample[file_column], args.containers_dir)
    waveforms = read_waveforms(
        container_path, CHANNELS, start_idx, end_idx, prediction_end_idx
    )
    actual_plot_seconds = len(next(iter(waveforms.values()))) / args.frequency
    print(f"container: {container_path}")
    print(f"sample row: {args.row}")
    print(f"input: {actual_input_samples / args.frequency:.1f} seconds")
    print(
        f"input samples: {actual_input_samples:,} "
        f"(expected {expected_input_samples:,})"
    )
    print(f"plot duration: {actual_plot_seconds:.1f} seconds")

    fig = plot_waveforms(
        waveforms,
        frequency=args.frequency,
        input_seconds=actual_input_samples / args.frequency,
        prediction_seconds=args.prediction_seconds,
        title=(
            f"{container_path.name}: input ({actual_input_samples / args.frequency:.0f} s) "
            f"+ prediction horizon ({args.prediction_seconds} s)"
        ),
        plot_stride=args.plot_stride,
    )
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved plot: {args.output}")


if __name__ == "__main__":
    main()
