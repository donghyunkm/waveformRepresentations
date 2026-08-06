"""Cluster-ready native PhysioJEPA pretraining entry point."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, SelfSupervisedDataset
from physiojepa.jepa import JEPASimpleLightning, loss_pred, mse_variance_loss

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
    path = sample_cache_path(config, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required JEPA sample cache is missing: {path}. "
            "Run prepare_pretraining_samples.py first."
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
    config, config_path = load_config("train_patch_jepa.yaml")
    pl.seed_everything(int(config["run_config"]["random_state"]), workers=True)
    print(f"Using configuration: {config_path}")

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

    train_samples = _load_samples(config, "train", train_files)
    val_samples = _load_samples(config, "val", val_files)
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
    model = JEPASimpleLightning(
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
    )

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
