"""Physiological-Distance Contrastive JEPA pretraining entry point.

Trains a JEPA encoder with an auxiliary contrastive loss based on continuous
physiological distance (HR + MAP). Windows from different patients with similar
physiology (d < epsilon) are pulled together; windows with different physiology
(d > delta) are pushed apart. Same-patient pairs are ignored entirely.

The dataset returns (X_normalized, patient_id, hr_value, map_value) tuples.
HR is computed from ECG II R-peak count; MAP from mean(ABP).
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset

from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, SelfSupervisedDataset
from physiojepa.jepa import PhysioContrastiveJEPALightning, loss_pred, mse_variance_loss

# Add scripts dir for shared pipeline utilities
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import (
    config_fingerprint,
    load_config,
    load_pretraining_split,
    require_output_path,
    sample_cache_path,
)
from resume_checkpoints import build_resume_callbacks, find_resume_checkpoint


torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)


# ── Physiological Feature Computation ─────────────────────────────────────────

def compute_hr_from_ecg(ecg_signal: np.ndarray, fs: int = 125) -> float:
    """
    Estimate heart rate (bpm) from ECG lead II by counting R-peaks.
    Returns NaN if fewer than 2 valid peaks detected.
    """
    if np.nanstd(ecg_signal) < 1e-6:
        return np.nan

    sig = ecg_signal - np.nanmean(ecg_signal)
    min_distance = int(0.4 * fs)  # 150 bpm max
    threshold = np.nanstd(sig) * 0.3

    peaks_pos, _ = find_peaks(sig, distance=min_distance, height=threshold)
    peaks_neg, _ = find_peaks(-sig, distance=min_distance, height=threshold)
    peaks = peaks_pos if len(peaks_pos) >= len(peaks_neg) else peaks_neg

    if len(peaks) < 2:
        return np.nan

    rr_intervals = np.diff(peaks) / fs
    rr_valid = rr_intervals[(rr_intervals > 0.3) & (rr_intervals < 2.0)]
    if len(rr_valid) < 1:
        return np.nan

    return 60.0 / np.median(rr_valid)


def compute_map_from_abp(abp_signal: np.ndarray) -> float:
    """Estimate MAP (mmHg) from ABP signal as the mean."""
    val = np.nanmean(abp_signal)
    if np.isnan(val) or val < 20 or val > 200:
        return np.nan
    return val


# ── Dataset Wrapper ───────────────────────────────────────────────────────────

class SelfSupervisedDatasetWithPhysioValues(Dataset):
    """Wraps SelfSupervisedDataset to return (X, patient_id, hr_value, map_value).

    Computes continuous HR and MAP from raw (un-normalized) waveforms at dataset
    construction time and caches them. Invalid values are stored as NaN and
    handled gracefully in the loss function.

    Channel order assumed: ABP=0, II=1, PLETH=2 (matching config).
    """

    def __init__(self, base_dataset: SelfSupervisedDataset,
                 cache_path: Path | None = None):
        """
        Args:
            base_dataset: The underlying SelfSupervisedDataset
            cache_path: Path to save/load cached HR/MAP values (.npz).
        """
        self.base_dataset = base_dataset

        # Build subject_id -> integer mapping
        subject_ids = base_dataset.sample_df["subject_id"].values
        unique_subjects = sorted(set(subject_ids))
        self.subject_to_int = {s: i for i, s in enumerate(unique_subjects)}
        self.patient_id_array = np.array(
            [self.subject_to_int[s] for s in subject_ids], dtype=np.int64
        )

        # Load or compute HR/MAP values
        if cache_path is not None and cache_path.is_file():
            print(f"  Loading cached physio values from {cache_path}")
            data = np.load(cache_path)
            self.hr_values = data["hr_values"]
            self.map_values = data["map_values"]
            assert len(self.hr_values) == len(base_dataset), \
                f"Cache size mismatch: {len(self.hr_values)} vs {len(base_dataset)}"
        else:
            print(f"  Computing HR/MAP for {len(base_dataset)} samples...")
            self.hr_values, self.map_values = self._compute_all_physio_values()
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path,
                                    hr_values=self.hr_values,
                                    map_values=self.map_values)
                print(f"  Saved physio values cache to {cache_path}")

        # Report statistics
        hr_valid = ~np.isnan(self.hr_values)
        map_valid = ~np.isnan(self.map_values)
        both_valid = hr_valid & map_valid
        print(f"  HR valid: {hr_valid.sum()}/{len(self.hr_values)} ({hr_valid.mean()*100:.1f}%)")
        print(f"  MAP valid: {map_valid.sum()}/{len(self.map_values)} ({map_valid.mean()*100:.1f}%)")
        print(f"  Both valid: {both_valid.sum()}/{len(self.hr_values)} ({both_valid.mean()*100:.1f}%)")
        if both_valid.sum() > 0:
            print(f"  HR range: {np.nanmin(self.hr_values):.0f} - {np.nanmax(self.hr_values):.0f} bpm "
                  f"(median {np.nanmedian(self.hr_values):.0f})")
            print(f"  MAP range: {np.nanmin(self.map_values):.0f} - {np.nanmax(self.map_values):.0f} mmHg "
                  f"(median {np.nanmedian(self.map_values):.0f})")

    def _compute_all_physio_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute HR and MAP for all samples from bedside numerics (1 Hz).

        Uses the validated bedside monitor values rather than waveform-derived
        features. Groups by stay to amortize I/O (one numerics record per stay).
        Falls back to waveform computation for stays without numerics.
        """
        from multiprocessing import Pool
        from precompute_physio_values import (
            load_manifest, process_stay, FS_WAVEFORM, SAMPLE_DURATION_SEC
        )

        n = len(self.base_dataset)
        sample_df = self.base_dataset.sample_df

        print(f"  Computing HR/MAP from bedside numerics for {n} samples...")
        manifest = load_manifest()
        manifest_by_stay = manifest.set_index("stay_id")

        # Group samples by stay_id
        grouped = sample_df.groupby("stay_id")

        args_list = []
        for stay_id, group_df in grouped:
            if stay_id not in manifest_by_stay.index:
                continue
            row = manifest_by_stay.loc[stay_id]
            args_list.append((
                stay_id, group_df,
                row["header_path"], row["numeric_header_path"],
            ))

        print(f"    {len(args_list)} stays to process...")

        n_workers = min(16, len(args_list))
        if n_workers > 1:
            with Pool(n_workers) as pool:
                results = pool.map(process_stay, args_list, chunksize=16)
        else:
            results = [process_stay(a) for a in args_list]

        # Collect into arrays
        hr_values = np.full(n, np.nan, dtype=np.float32)
        map_values = np.full(n, np.nan, dtype=np.float32)

        for result in results:
            for local_idx, global_idx in enumerate(result["indices"]):
                hr_values[global_idx] = result["hr_values"][local_idx]
                map_values[global_idx] = result["map_values"][local_idx]

        hr_ok = (~np.isnan(hr_values)).sum()
        map_ok = (~np.isnan(map_values)).sum()
        print(f"    HR valid: {hr_ok}/{n} ({100*hr_ok/n:.1f}%)")
        print(f"    MAP valid: {map_ok}/{n} ({100*map_ok/n:.1f}%)")

        return hr_values, map_values

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        X, _ = self.base_dataset[idx]
        patient_id = torch.tensor(self.patient_id_array[idx], dtype=torch.long)
        hr = torch.tensor(self.hr_values[idx], dtype=torch.float32)
        map_val = torch.tensor(self.map_values[idx], dtype=torch.float32)
        return X, patient_id, hr, map_val


# ── Training Setup ────────────────────────────────────────────────────────────

def _load_samples(config: dict, split: str, allowed_files: set[str]) -> pd.DataFrame:
    path = sample_cache_path(config, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JEPA sample cache is missing: {path}. "
            "Run prepare_pretraining_samples.py first."
        )
    samples = pd.read_csv(path)
    required = {"file", "start_idx", "end_idx", "subject_id"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"{path} is missing sample columns: {sorted(missing)}")
    unexpected = set(samples["file"]).difference(allowed_files)
    if unexpected:
        preview = "\n  ".join(sorted(unexpected)[:10])
        raise ValueError(f"{split} samples contain files outside the split:\n  {preview}")

    maximum = config["training"].get(f"max_{split}_samples")
    if maximum is not None:
        samples = samples.sample(
            n=min(int(maximum), len(samples)),
            random_state=int(config["run_config"]["random_state"]),
        ).sort_values(["file", "start_idx"])
    if samples.empty:
        raise ValueError(f"No {split} samples are available")
    return samples.reset_index(drop=True)


def _architecture(config: dict) -> tuple[dict, dict]:
    channels = config["dataset"]["channels"]
    frequency = int(config["dataset"]["frequency"])
    patch_size = int(config["dataset"]["patch_seconds"] * frequency)
    patch_stride = patch_size - int(config["dataset"]["overlap"] * patch_size)
    sequence_samples = int(config["dataset"]["sample_seq_len_seconds"] * frequency)
    num_patches = (max(sequence_samples, patch_size) - patch_size) // patch_stride + 1
    if (sequence_samples - patch_size) % patch_stride:
        num_patches += 1

    encoder_config = config["encoder"]
    encoder_arch = {
        "c_in": len(channels),
        "num_patches": int(num_patches),
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "d_model": encoder_config["d_model"],
        "nhead": encoder_config["nhead"],
        "use_tst_block": encoder_config["use_tst_block"],
        "shared_embedding": encoder_config["shared_embedding"],
        "num_layers": encoder_config["num_layers"],
        "pe_type": encoder_config["pe_type"],
        "mlp_ratio": encoder_config["mlp_ratio"],
        "qkv_bias": encoder_config["qkv_bias"],
        "qk_scale": encoder_config["qk_scale"],
        "drop_rate": encoder_config["drop_rate"],
        "attn_drop_rate": encoder_config["attn_drop_rate"],
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "jepa": encoder_config["jepa"],
        "tokenizer_type": encoder_config["tokenizer_type"],
        "tokenizer_kwargs": encoder_config["tokenizer_kwargs"],
        "embed_activation": nn.GELU(),
    }
    predictor_config = config["predictor"]
    predictor_arch = {
        "num_patches": int(num_patches),
        "encoder_embed_dim": encoder_arch["d_model"],
        "predictor_embed_dim": predictor_config["predictor_embed_dim"],
        "nhead": predictor_config["nhead"],
        "num_layers": predictor_config["num_layers"],
        "pe_type": predictor_config["pe_type"],
        "mlp_ratio": predictor_config["mlp_ratio"],
        "qkv_bias": predictor_config["qkv_bias"],
        "qk_scale": predictor_config["qk_scale"],
        "drop_rate": predictor_config["drop_rate"],
        "attn_drop_rate": predictor_config["attn_drop_rate"],
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "use_tst_block": predictor_config["use_tst_block"],
        "c_in_mask_tokens": predictor_config["c_in_mask_tokens"],
        "embed_activation": nn.GELU(),
    }
    return encoder_arch, predictor_arch


def main() -> None:
    config, config_path = load_config("configs/train_physio_contrastive_jepa.yaml")
    pl.seed_everything(int(config["run_config"]["random_state"]), workers=True)
    print(f"Using configuration: {config_path}")

    # Load pretraining split
    split_manifest = load_pretraining_split(config)
    train_files = set(
        split_manifest.loc[
            split_manifest["pretrain_split"] == "train", "file"
        ].astype(str)
    )
    val_files = set(
        split_manifest.loc[
            split_manifest["pretrain_split"] == "val", "file"
        ].astype(str)
    )
    if train_files.intersection(val_files):
        raise AssertionError("JEPA train and validation files overlap")

    # Load sample caches
    train_samples = _load_samples(config, "train", train_files)
    val_samples = _load_samples(config, "val", val_files)

    print(f"Train: {len(train_samples)} samples, {train_samples['subject_id'].nunique()} subjects")
    print(f"Val: {len(val_samples)} samples, {val_samples['subject_id'].nunique()} subjects")

    dataset_cfg = config["dataset"]

    # Build base datasets
    base_train_dataset = SelfSupervisedDataset(
        zarr_files=sorted(train_files),
        channels=dataset_cfg["channels"],
        sample_df=train_samples,
        max_seq_len_sec=None,
        sample_seq_len_sec=dataset_cfg["sample_seq_len_seconds"],
        sample_stride_sec=dataset_cfg["sample_stride_seconds"],
        frequency=dataset_cfg["frequency"],
        butterworth_filters=None,
        median_filter_kernel_size=None,
        normalize_signals=dataset_cfg["normalize_signals"],
        require_all_channels=dataset_cfg["require_all_channels"],
        clip_interpolations=CLIP_INTERPOLATE_RANGES,
        constant_nan_tolerance=dataset_cfg["constant_nan_tolerance"],
    )
    base_val_dataset = SelfSupervisedDataset(
        zarr_files=sorted(val_files),
        channels=dataset_cfg["channels"],
        sample_df=val_samples,
        max_seq_len_sec=None,
        sample_seq_len_sec=dataset_cfg["sample_seq_len_seconds"],
        sample_stride_sec=dataset_cfg["sample_stride_seconds"],
        frequency=dataset_cfg["frequency"],
        butterworth_filters=None,
        median_filter_kernel_size=None,
        normalize_signals=dataset_cfg["normalize_signals"],
        require_all_channels=dataset_cfg["require_all_channels"],
        clip_interpolations=CLIP_INTERPOLATE_RANGES,
        constant_nan_tolerance=dataset_cfg["constant_nan_tolerance"],
    )

    # Physio values cache paths
    cache_dir = Path(config["paths"]["sample_cache_dir"])
    train_cache = cache_dir / "physio_values_train.npz"
    val_cache = cache_dir / "physio_values_val.npz"

    # Wrap with continuous physio values
    print("\nBuilding training dataset with physiological values...")
    train_dataset = SelfSupervisedDatasetWithPhysioValues(
        base_train_dataset, cache_path=train_cache
    )
    print("\nBuilding validation dataset with physiological values...")
    val_dataset = SelfSupervisedDatasetWithPhysioValues(
        base_val_dataset, cache_path=val_cache
    )

    training = config["training"]
    batch_size = int(training["batch_size"])
    workers = int(training["num_workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
    )

    # Build model
    encoder_arch, predictor_arch = _architecture(config)
    scheduler = config["scheduler"]
    scheduler_kwargs = {
        "max_lr": scheduler["max_lr"],
        "div_factor": scheduler["div_factor"],
        "final_div_factor": scheduler["final_div_factor"],
        "pct_start": scheduler["pct_start"],
        "anneal_strategy": scheduler["anneal_strategy"],
    }
    loss_function = (
        mse_variance_loss
        if training["loss_fxn"].lower() == "variance"
        else loss_pred
    )

    contrastive_cfg = config["contrastive"]
    model = PhysioContrastiveJEPALightning(
        learning_rate=config["optimizer"]["learning_rate"],
        train_size=len(train_dataset),
        batch_size=batch_size,
        n_gpus=int(training["n_gpus"]),
        patchtsjepa_encoder_kwargs=encoder_arch,
        patchtsjepa_predictor_kwargs=predictor_arch,
        weight_decay=config["optimizer"]["weight_decay"],
        use_weight_decay_scheduler=config["optimizer"]["use_weight_decay_scheduler"],
        final_weight_decay=config["optimizer"]["final_weight_decay"],
        epochs=int(training["epochs"]),
        loss_fn=loss_function,
        optimizer_type=config["optimizer"]["optimizer_type"],
        scheduler_type=scheduler["scheduler_type"],
        target_mask_range=training["target_mask_range"],
        context_mask_range=training["context_mask_range"],
        mask_block_range=training["mask_block_range"],
        ema_decay=training["ema_decay"],
        scheduler_kwargs=scheduler_kwargs,
        transforms=None,
        contrastive_weight=float(contrastive_cfg["lambda_contrast"]),
        contrastive_temperature=float(contrastive_cfg["tau"]),
        contrastive_epsilon=float(contrastive_cfg["epsilon"]),
        contrastive_delta=float(contrastive_cfg["delta"]),
        sigma_hr=float(contrastive_cfg["sigma_hr"]),
        sigma_map=float(contrastive_cfg["sigma_map"]),
        projection_dim=int(contrastive_cfg["projection_dim"]),
        projection_hidden_dim=(
            int(contrastive_cfg["projection_hidden_dim"])
            if contrastive_cfg.get("projection_hidden_dim")
            else None
        ),
    )

    # Checkpoint setup
    resume_config = config["resume"]
    run_subdir = resume_config["run_subdir"]
    checkpoint_dir = require_output_path(config["paths"]["models_dir"]) / run_subdir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(config)
    resume_metadata, rolling_checkpoint = build_resume_callbacks(
        checkpoint_dir=checkpoint_dir,
        config_fingerprint=fingerprint,
        run_subdir=run_subdir,
        checkpoint_interval_minutes=float(
            resume_config["checkpoint_interval_minutes"]
        ),
    )
    best_validation = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-val-epoch={epoch:02d}-loss={val_loss_epoch:.5f}",
        monitor="val_loss_epoch",
        mode="min",
        save_top_k=2,
        auto_insert_metric_name=False,
    )
    epoch_last = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch-{epoch:02d}-train={train_loss_epoch:.5f}",
        monitor="train_loss_epoch",
        mode="min",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
    )
    checkpoint_path = (
        find_resume_checkpoint(checkpoint_dir, fingerprint)
        if resume_config.get("enabled", True)
        else None
    )
    print(f"Resume checkpoint: {checkpoint_path}")

    # Logger
    offline = bool(config["run_config"].get("wandb_offline", True))
    logger = WandbLogger(
        project=config["run_config"]["wandb_project"],
        name=config["run_config"]["name"],
        save_dir=str(checkpoint_dir),
        offline=offline,
    )
    logger.log_hyperparams(config)

    # Trainer
    devices = int(training["n_gpus"])
    trainer = pl.Trainer(
        precision=config["run_config"]["precision"],
        deterministic=bool(config["run_config"].get("deterministic", True)),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=logger,
        val_check_interval=training["val_check_interval"],
        log_every_n_steps=int(training.get("log_every_n_steps", 10)),
        num_sanity_val_steps=int(training.get("num_sanity_val_steps", 2)),
        strategy="ddp" if devices > 1 else "auto",
        gradient_clip_val=training["gradient_clip_val"],
        gradient_clip_algorithm=(
            training.get("gradient_clip_algorithm", "norm")
            if training["use_gradient_clipping"]
            else None
        ),
        accelerator="gpu",
        devices=devices,
        default_root_dir=str(checkpoint_dir),
        max_epochs=int(training["epochs"]),
        accumulate_grad_batches=int(training["accumulate_grad_batches"]),
        sync_batchnorm=devices > 1,
        callbacks=[
            resume_metadata,
            rolling_checkpoint,
            best_validation,
            epoch_last,
        ],
        limit_train_batches=training.get("limit_train_batches", 1.0),
        limit_val_batches=training.get("limit_val_batches", 1.0),
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )


if __name__ == "__main__":
    main()
