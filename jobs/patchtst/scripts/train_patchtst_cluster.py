"""Cluster-ready self-supervised PatchTST pretraining entry point.

Mirrors train_patchtst.py but adapted for the cluster:
- Reads ZipStore containers via the leakage-safe manifest
- Uses precomputed sample caches (same as native JEPA)
- Offline W&B logging
- Checkpoint resume with fingerprint validation
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, SelfSupervisedDataset
from physiojepa.patchtst import PatchTFTSimpleLightning

# Add the jepa directory to import the shared pipeline infrastructure.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "jepa" / "scripts"))
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


def _load_samples(config: dict, split: str, allowed_files: set[str]) -> pd.DataFrame:
    """Load and validate a precomputed sample cache."""
    path = sample_cache_path(config, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required sample cache is missing: {path}. "
            "Run jobs/jepa/prepare_pretraining_samples.py first."
        )
    samples = pd.read_csv(path)
    required = {"file", "start_idx", "end_idx"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"{path} is missing sample columns: {sorted(missing)}")
    unexpected = set(samples["file"]).difference(allowed_files)
    if unexpected:
        preview = "\n  ".join(sorted(unexpected)[:10])
        raise ValueError(f"{split} samples contain files outside the split:\n  {preview}")
    if samples.empty:
        raise ValueError(f"No {split} samples are available")
    return samples.reset_index(drop=True)


def main() -> None:
    config, config_path = load_config("configs/train_patchtst_cluster.yaml")
    pl.seed_everything(int(config["run_config"]["random_state"]), workers=True)
    print(f"Using configuration: {config_path}")

    # Load the leakage-safe pretraining manifest (shared with native JEPA).
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
        raise AssertionError("PatchTST train and validation files overlap")

    train_samples = _load_samples(config, "train", train_files)
    val_samples = _load_samples(config, "val", val_files)
    print(
        f"Loaded {len(train_samples)} train samples, "
        f"{len(val_samples)} val samples"
    )

    dataset = config["dataset"]
    train_dataset = SelfSupervisedDataset(
        zarr_files=sorted(train_files),
        channels=dataset["channels"],
        sample_df=train_samples,
        max_seq_len_sec=None,
        sample_seq_len_sec=dataset["sample_seq_len_seconds"],
        sample_stride_sec=dataset["sample_stride_seconds"],
        frequency=dataset["frequency"],
        butterworth_filters=None,
        median_filter_kernel_size=None,
        normalize_signals=dataset["normalize_signals"],
        require_all_channels=dataset["require_all_channels"],
        clip_interpolations=CLIP_INTERPOLATE_RANGES,
        constant_nan_tolerance=dataset["constant_nan_tolerance"],
    )
    val_dataset = SelfSupervisedDataset(
        zarr_files=sorted(val_files),
        channels=dataset["channels"],
        sample_df=val_samples,
        max_seq_len_sec=None,
        sample_seq_len_sec=dataset["sample_seq_len_seconds"],
        sample_stride_sec=dataset["sample_stride_seconds"],
        frequency=dataset["frequency"],
        butterworth_filters=None,
        median_filter_kernel_size=None,
        normalize_signals=dataset["normalize_signals"],
        require_all_channels=dataset["require_all_channels"],
        clip_interpolations=CLIP_INTERPOLATE_RANGES,
        constant_nan_tolerance=dataset["constant_nan_tolerance"],
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

    # Compute patch geometry.
    frequency = int(dataset["frequency"])
    patch_size = int(dataset["patch_seconds"] * frequency)
    overlap = dataset["overlap"]
    patch_stride = patch_size - int(overlap * patch_size)
    sequence_samples = int(dataset["sample_seq_len_seconds"] * frequency)
    num_patches = (max(sequence_samples, patch_size) - patch_size) // patch_stride + 1
    if (sequence_samples - patch_size) % patch_stride:
        num_patches += 1

    encoder_cfg = config["encoder"]
    encoder_arch = dict(
        c_in=len(dataset["channels"]),
        patch_size=patch_size,
        patch_stride=patch_stride,
        num_patches=int(num_patches),
        d_model=encoder_cfg["d_model"],
        n_heads=encoder_cfg["n_heads"],
        d_ff=encoder_cfg["d_ff"],
        num_layers=encoder_cfg["num_layers"],
        augmentations=encoder_cfg["augmentations"],
        mask_ratio=encoder_cfg["mask_ratio"],
        shared_embedding=encoder_cfg["shared_embedding"],
        pretrain_head=encoder_cfg["pretrain_head"],
        dropout=encoder_cfg["dropout"],
        attn_dropout=encoder_cfg["attn_dropout"],
        act=encoder_cfg["act"],
        pre_norm=encoder_cfg["pre_norm"],
        pe_type=encoder_cfg["pe_type"],
        qkv_bias=encoder_cfg["qkv_bias"],
        init_std=encoder_cfg["init_std"],
        tokenizer_type=encoder_cfg["tokenizer_type"],
        tokenizer_kwargs=encoder_cfg["tokenizer_kwargs"],
    )

    scheduler = config["scheduler"]
    scheduler_kwargs = {
        "max_lr": scheduler["max_lr"],
        "div_factor": scheduler["div_factor"],
        "final_div_factor": scheduler["final_div_factor"],
        "pct_start": scheduler["pct_start"],
        "anneal_strategy": scheduler["anneal_strategy"],
    }

    model = PatchTFTSimpleLightning(
        learning_rate=config["optimizer"]["learning_rate"],
        train_size=len(train_dataset),
        batch_size=batch_size,
        n_gpus=int(training["n_gpus"]),
        metrics={},
        loss_func=training["loss_fxn"],
        weight_decay=config["optimizer"]["weight_decay"],
        epochs=int(training["epochs"]),
        use_weight_decay_scheduler=config["optimizer"]["use_weight_decay_scheduler"],
        final_weight_decay=config["optimizer"]["final_weight_decay"],
        optimizer_type=config["optimizer"]["optimizer_type"],
        scheduler_type=scheduler["scheduler_type"],
        huber_delta=training["huber_delta"],
        scheduler_kwargs=scheduler_kwargs,
        transforms=None,
        **encoder_arch,
    )

    # Checkpoint and resume setup.
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
        filename="best-val-epoch={epoch:02d}-loss={val_loss:.5f}",
        monitor="val_loss",
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

    offline = bool(config["run_config"].get("wandb_offline", True))
    logger = WandbLogger(
        project=config["run_config"]["wandb_project"],
        name=config["run_config"]["name"],
        save_dir=str(checkpoint_dir),
        offline=offline,
    )
    logger.log_hyperparams(config)

    devices = int(training["n_gpus"])
    trainer = pl.Trainer(
        precision=config["run_config"]["precision"],
        deterministic=False,
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
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )


if __name__ == "__main__":
    main()
