"""Create leakage-safe native PhysioJEPA splits and sample-index caches."""

from __future__ import annotations

import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from physiojepa.data_preprocessing import close_zarr_group, open_zarr_group

from pipeline_common import (
    atomic_write_csv,
    atomic_write_json,
    build_pretraining_split,
    config_fingerprint,
    data_fingerprint,
    load_config,
    require_output_path,
    sample_cache_path,
)


_PREPARE_CONFIG: dict | None = None


def _initialize_worker(config: dict) -> None:
    global _PREPARE_CONFIG
    _PREPARE_CONFIG = config


def _samples_for_record(row: dict) -> tuple[list[dict], str]:
    if _PREPARE_CONFIG is None:
        raise RuntimeError("Sample-preparation worker was not initialized")
    dataset = _PREPARE_CONFIG["dataset"]
    window_seconds = int(dataset["sample_seq_len_seconds"])
    frequency = int(dataset["frequency"])
    window_samples = window_seconds * frequency

    root = open_zarr_group(row["file"], mode="r")
    try:
        if int(root.attrs["quality_window_seconds"]) != window_seconds:
            return [], "quality_window_mismatch"
        if float(root.attrs["constant_nan_threshold"]) != float(
            dataset["constant_nan_tolerance"]
        ):
            return [], "quality_threshold_mismatch"
        if int(root.attrs["sampling_frequency"]) != frequency:
            return [], "frequency_mismatch"
        if "quality/valid_window" not in root:
            return [], "missing_quality_mask"
        valid = np.asarray(root["quality/valid_window"][:], dtype=bool)
    finally:
        close_zarr_group(root)

    indices = np.flatnonzero(valid)
    samples = [
        {
            "file": row["file"],
            "start_idx": int(index * window_samples),
            "end_idx": int((index + 1) * window_samples),
            "subject_id": row["subject_id"],
            "stay_id": row["stay_id"],
        }
        for index in indices
    ]
    return samples, "ok"


def main() -> None:
    config, config_path = load_config("train_patch_jepa.yaml")
    paths = config["paths"]
    manifest_path = require_output_path(paths["pretraining_manifest_path"])
    metadata_path = manifest_path.with_suffix(".json")

    split_frame = build_pretraining_split(config)
    atomic_write_csv(split_frame, manifest_path)

    workers = int(config["sample_preparation"]["workers"])
    rows = split_frame.to_dict(orient="records")
    with mp.get_context("spawn").Pool(
        processes=workers,
        initializer=_initialize_worker,
        initargs=(config,),
    ) as pool:
        results = pool.map(_samples_for_record, rows, chunksize=4)

    samples_by_split = {"train": [], "val": []}
    reason_counts: Counter[str] = Counter()
    for row, (samples, reason) in zip(rows, results):
        reason_counts[reason] += 1
        samples_by_split[row["pretrain_split"]].extend(samples)

    summary: dict[str, object] = {
        "schema": "physiojepa.jepa-sample-index.v2",
        "config_path": str(config_path),
        "config_fingerprint": config_fingerprint(config),
        "data_fingerprint": data_fingerprint(config),
        "pretraining_manifest_path": str(manifest_path),
        "records": int(len(split_frame)),
        "subjects": int(split_frame["subject_id"].nunique()),
        "record_status_counts": dict(reason_counts),
        "splits": {},
    }
    for split, samples in samples_by_split.items():
        frame = pd.DataFrame(samples)
        if frame.empty:
            raise RuntimeError(f"No valid {split} JEPA samples were generated")
        frame = frame.sort_values(["subject_id", "stay_id", "start_idx"])
        output_path = sample_cache_path(config, split)
        atomic_write_csv(frame, output_path)
        summary["splits"][split] = {
            "path": str(output_path),
            "samples": int(len(frame)),
            "subjects": int(frame["subject_id"].nunique()),
            "stays": int(frame["stay_id"].nunique()),
        }

    atomic_write_json(summary, metadata_path)
    print(summary)


if __name__ == "__main__":
    main()
