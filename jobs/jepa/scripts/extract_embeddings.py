"""Extract and cache frozen encoder embeddings for downstream classifier training.

Supports both JEPA (JEPASimpleLightning) and PatchTST (PatchTFTSimpleLightning)
encoders. Embeddings are saved per-split as sharded .pt files to avoid RAM
exhaustion.

Output per sample: [n_channels, d_model, n_patches] in fp16.
For 1M train samples with [3, 512, 1800], total is ~5.5 TB across shards.
Each shard holds `shard_size` samples (default 512) and is written to disk
immediately so RAM stays bounded.

Usage:
    PHYSIOJEPA_CONFIG=configs/extract_embeddings_jepa.yaml python scripts/extract_embeddings.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, ForecastingDataset
from physiojepa.data_preprocessing import zarr_record_name

# Reuse pipeline utilities from the JEPA scripts directory.
_JEPA_SCRIPTS = Path(__file__).resolve().parent
if str(_JEPA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_JEPA_SCRIPTS))

from pipeline_common import load_config, require_output_path


torch.set_float32_matmul_precision("high")


def _load_split_inputs(config: dict, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load outcome and sample dataframes for a given split."""
    subject_split = pd.read_csv(config["paths"]["subject_split_path"])
    subjects = set(
        subject_split.loc[subject_split["split"] == split, "subject_id"].astype(str)
    )
    outcomes = pd.read_csv(config["paths"]["outcome_df_path"])
    if "subject_id" not in outcomes:
        outcomes["subject_id"] = outcomes["file_path"].map(
            lambda path: zarr_record_name(path).split("-", 1)[0]
        )
    outcomes = outcomes.loc[outcomes["subject_id"].astype(str).isin(subjects)].copy()
    outcomes["Time Stamp (seconds)"] = outcomes["Time Stamp (seconds)"].round()

    cache_dir = Path(config["paths"]["sample_cache_dir"])
    dataset_name = config["paths"]["dataset_filename"]
    cache_path = cache_dir / f"{dataset_name}-{split}_samples.csv.gz"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Required fixed-split forecasting cache is missing: {cache_path}"
        )
    samples = pd.read_csv(cache_path)
    if "subject_id" not in samples:
        samples["subject_id"] = samples["file_path"].map(
            lambda path: zarr_record_name(path).split("-", 1)[0]
        )
    unexpected = set(samples["subject_id"].astype(str)).difference(subjects)
    if unexpected:
        raise ValueError(f"{split} cache contains subjects outside the fixed split")
    return outcomes, samples.reset_index(drop=True)


def _dataset(config: dict, outcomes: pd.DataFrame, samples: pd.DataFrame):
    """Build a ForecastingDataset from config."""
    dataset = config["dataset"]
    return ForecastingDataset(
        channels=dataset["channels"],
        forecast_window_sec=dataset["forecast_window_sec"],
        outcome_df=outcomes,
        outcome_df_outcome_col=dataset["y_outcome"],
        file_col="file_path",
        y_date_column="date",
        outcome_df_seconds_since_column="Time Stamp (seconds)",
        outcome_df_duration_column="event_length",
        sample_df=samples,
        sample_seq_len_sec=dataset["sample_seq_len_seconds"],
        frequency=dataset["frequency"],
        butterworth_filters=None,
        median_filter_kernel_size=None,
        clip_interpolations=CLIP_INTERPOLATE_RANGES,
        constant_nan_tolerance=dataset["constant_nan_tolerance"],
        require_all_channels=dataset["require_all_channels"],
        infer_forecast_windows=dataset["infer_forecast_windows"],
        normalize_signals=dataset["normalize_signals"],
    )


def load_encoder(config: dict) -> nn.Module:
    """Load the pretrained encoder based on model_type in config."""
    model_type = config["run_config"]["model_type"]
    pretrained_path = Path(config["paths"]["pretrained_encoder_path"]).resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"Encoder checkpoint does not exist: {pretrained_path}")

    if model_type == "jepa":
        from physiojepa.jepa import JEPASimpleLightning
        encoder = JEPASimpleLightning.load_from_checkpoint(
            str(pretrained_path), map_location="cpu"
        )
    elif model_type == "patchtst":
        from physiojepa.patchtst import PatchTFTSimpleLightning
        encoder = PatchTFTSimpleLightning.load_from_checkpoint(
            str(pretrained_path), map_location="cpu"
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    encoder.eval()
    encoder.freeze()
    if hasattr(encoder, "pretrain"):
        encoder.pretrain = False
        if hasattr(encoder, "model"):
            encoder.model.pretrain = False
    return encoder


def _save_shard(split_dir: Path, shard_idx: int, embeddings_list, labels_list):
    """Concatenate and save a shard to disk."""
    emb = torch.cat(embeddings_list, dim=0)
    lbl = torch.cat(labels_list, dim=0)
    shard_path = split_dir / f"shard_{shard_idx:05d}.pt"
    torch.save({"embeddings": emb, "labels": lbl}, shard_path.with_suffix(".partial"))
    os.replace(shard_path.with_suffix(".partial"), shard_path)
    return emb.shape


@torch.no_grad()
def extract_split(
    encoder: nn.Module,
    dataset: ForecastingDataset,
    config: dict,
    split: str,
    output_dir: Path,
    device: torch.device,
) -> None:
    """Run encoder over an entire split and save embeddings + labels in shards."""
    extraction = config["extraction"]
    batch_size = int(extraction["batch_size"])
    num_workers = int(extraction["num_workers"])
    shard_size = int(extraction.get("shard_size", 512))
    precision = config["run_config"].get("precision", "bf16-mixed")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )

    n_samples = len(dataset)
    print(f"  [{split}] Extracting {n_samples} samples, {len(loader)} batches...")
    print(f"  [{split}] Shard size: {shard_size} samples")

    use_amp = "16" in precision or "bf16" in precision
    amp_dtype = torch.bfloat16 if "bf16" in precision else torch.float16

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    shard_embeddings = []
    shard_labels = []
    shard_idx = 0
    samples_in_shard = 0
    total_saved = 0
    emb_shape = None

    start_time = time.time()
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            emb = encoder(x)

        # PatchTST returns a tuple; take first element
        if isinstance(emb, tuple):
            emb = emb[0]

        # emb shape: [bs, n_channels, d_model, n_patches]
        shard_embeddings.append(emb.cpu().to(torch.float16))
        shard_labels.append(y.cpu())
        samples_in_shard += emb.size(0)

        # Flush shard to disk when full
        if samples_in_shard >= shard_size:
            emb_shape = _save_shard(split_dir, shard_idx, shard_embeddings, shard_labels)
            total_saved += samples_in_shard
            shard_embeddings = []
            shard_labels = []
            shard_idx += 1
            samples_in_shard = 0

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (batch_idx + 1) / elapsed
            eta = (len(loader) - batch_idx - 1) / rate
            print(
                f"    batch {batch_idx + 1}/{len(loader)} "
                f"({rate:.1f} batch/s, ETA {eta / 60:.1f} min), "
                f"shards: {shard_idx}, saved: {total_saved}"
            )

    # Save remaining samples
    if shard_embeddings:
        emb_shape = _save_shard(split_dir, shard_idx, shard_embeddings, shard_labels)
        total_saved += samples_in_shard
        shard_idx += 1

    elapsed = time.time() - start_time
    print(
        f"  [{split}] Done: {total_saved} samples in {shard_idx} shards, "
        f"{elapsed:.0f}s ({total_saved / elapsed:.0f} samples/s)"
    )
    if emb_shape is not None:
        print(f"  [{split}] Embedding shape per shard: {emb_shape}")

    # Write metadata
    metadata = {
        "n_samples": total_saved,
        "n_shards": shard_idx,
        "shard_size": shard_size,
        "embedding_shape_per_sample": list(emb_shape[1:]) if emb_shape else None,
    }
    meta_path = split_dir / "metadata.pt"
    torch.save(metadata, meta_path.with_suffix(".partial"))
    os.replace(meta_path.with_suffix(".partial"), meta_path)


def main() -> None:
    config, config_path = load_config("configs/extract_embeddings.yaml")
    print(f"Using configuration: {config_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load encoder
    encoder = load_encoder(config)
    encoder = encoder.to(device)
    print(f"Loaded encoder: {config['run_config']['model_type']}")

    # Prepare output directory
    output_dir = require_output_path(config["paths"]["embeddings_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config alongside embeddings for provenance
    meta_path = output_dir / "extraction_config.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Extract for each split
    splits = config["extraction"].get("splits", ["train", "val", "test"])
    for split in splits:
        print(f"\nProcessing split: {split}")
        outcomes, samples = _load_split_inputs(config, split)
        ds = _dataset(config, outcomes, samples)
        extract_split(encoder, ds, config, split, output_dir, device)

    print("\nAll splits extracted successfully.")


if __name__ == "__main__":
    main()
