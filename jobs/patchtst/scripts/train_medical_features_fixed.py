"""Attentive regression probe for medical feature prediction using a frozen PatchTST encoder.

Predicts 15 physiological features from the full token sequence using an attentive
pooling head + regression output. Same architecture as JEPA version but loads PatchTST.
"""

from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

# Reuse pipeline utilities from the JEPA scripts directory.
_JEPA_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "jepa" / "scripts"
if str(_JEPA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_JEPA_SCRIPTS))

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset

from physiojepa.augmentations import TransformsCallback, jitter_augmentation, channel_masking
from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, ForecastingDataset
from physiojepa.data_preprocessing import zarr_record_name
from physiojepa.heads import AttentiveClassifier
from physiojepa.patchtst import PatchTFTSimpleLightning
from physiojepa.train import PatchTFTSingleOutcomeLightning

from pipeline_common import config_fingerprint, load_config, require_output_path
from resume_checkpoints import build_resume_callbacks, find_resume_checkpoint

# Import shared components from JEPA version
from train_medical_features_fixed import (
    FEATURE_NAMES, NUM_FEATURES, ORIG_INDICES,
    RegressionProbe, MedicalFeatureDataset,
    compute_features_for_samples, _load_split_inputs, _base_dataset,
)


torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)


def main() -> None:
    config, config_path = load_config("configs/train_medical_features_fixed.yaml")
    pl.seed_everything(int(config["run_config"]["random_state"]), workers=True)
    print(f"Using configuration: {config_path}")

    # Load encoder
    pretrained_path = Path(config["paths"]["pretrained_encoder_path"]).resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"PatchTST checkpoint not found: {pretrained_path}")
    encoder = PatchTFTSimpleLightning.load_from_checkpoint(
        str(pretrained_path), map_location="cpu"
    )
    d_model = encoder.model.d_model
    print(f"Loaded PatchTST encoder: d_model={d_model}")

    # Load splits
    split_inputs = {
        split: _load_split_inputs(config, split)
        for split in ("train", "val", "test")
    }

    # Compute/load features for each split
    feature_cache_dir = Path(config["paths"]["feature_cache_dir"])
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    all_features = {}
    for split in ("train", "val", "test"):
        outcomes, samples = split_inputs[split]
        cache_path = feature_cache_dir / f"{config['paths']['dataset_filename']}-{split}_medical_features.npy"
        all_features[split] = compute_features_for_samples(samples, config, cache_path)

    # Build datasets
    datasets = {}
    for split in ("train", "val", "test"):
        outcomes, samples = split_inputs[split]
        base_ds = _base_dataset(config, outcomes, samples)
        feats = all_features[split]
        valid_mask = np.sum(~np.isnan(feats), axis=1) >= NUM_FEATURES // 2
        valid_indices = np.where(valid_mask)[0]
        datasets[split] = MedicalFeatureDataset(base_ds, feats, valid_indices)
        print(f"  {split}: {len(valid_indices)}/{len(samples)} valid windows")

    # Normalization stats from training set
    train_feats = all_features["train"][np.sum(~np.isnan(all_features["train"]), axis=1) >= NUM_FEATURES // 2]
    feature_means = torch.tensor(np.nanmean(train_feats, axis=0), dtype=torch.float32)
    feature_stds = torch.tensor(np.nanstd(train_feats, axis=0), dtype=torch.float32)
    feature_stds = torch.clamp(feature_stds, min=1e-6)
    print(f"\nFeature normalization stats:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name}: mean={feature_means[i]:.3f}, std={feature_stds[i]:.3f}")

    training = config["training"]
    batch_size = int(training["batch_size"])
    workers = int(training["num_workers"])

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=workers,
            persistent_workers=workers > 0,
            pin_memory=True,
        ),
    }

    # Build model
    head_config = config["lp_head"]
    head = AttentiveClassifier(
        embed_dim=d_model,
        num_heads=int(head_config["num_heads"]),
        mlp_ratio=head_config["mlp_ratio"],
        depth=head_config["depth"],
        c_in=len(config["dataset"]["channels"]),
        norm_layer=nn.LayerNorm,
        init_std=head_config["init_std"],
        qkv_bias=head_config["qkv_bias"],
        num_classes=NUM_FEATURES,
        complete_block=head_config["complete_block"],
        affine=head_config["affine"],
    )

    transforms = (
        TransformsCallback(
            transforms=[
                partial(jitter_augmentation, mask_ratio=0.05, jitter_ratio=0.05, p=0.5),
                partial(channel_masking, dim=1, p=0.1, specific_channels=None),
            ]
        )
        if config["dataset"]["use_transforms"]
        else None
    )

    scheduler = config["scheduler"]
    model = RegressionProbe(
        learning_rate=config["optimizer"]["learning_rate"],
        train_size=len(datasets["train"]),
        n_gpus=int(training["n_gpus"]),
        batch_size=batch_size,
        linear_probing_head=head,
        preloaded_model=encoder,
        num_features=NUM_FEATURES,
        feature_means=feature_means,
        feature_stds=feature_stds,
        metrics={},
        class_weights=None,
        fine_tune=training["fine_tune"],
        epochs=training["epochs"],
        scheduler_type=scheduler["scheduler_type"],
        optimizer_type=config["optimizer"]["optimizer_type"],
        weight_decay=config["optimizer"]["weight_decay"],
        use_weight_decay_scheduler=config["optimizer"]["use_weight_decay_scheduler"],
        final_weight_decay=config["optimizer"]["final_weight_decay"],
        scheduler_kwargs={
            "max_lr": scheduler["max_lr"],
            "div_factor": scheduler["div_factor"],
            "final_div_factor": scheduler["final_div_factor"],
            "pct_start": scheduler["pct_start"],
            "anneal_strategy": scheduler["anneal_strategy"],
        },
        mixup_callback=None,
        transforms=transforms,
    )

    # Resume / checkpointing
    resume_config = config["resume"]
    run_subdir = resume_config["run_subdir"]
    checkpoint_dir = require_output_path(config["paths"]["models_dir"]) / run_subdir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(config)
    resume_metadata, rolling_checkpoint = build_resume_callbacks(
        checkpoint_dir,
        fingerprint,
        run_subdir,
        float(resume_config["checkpoint_interval_minutes"]),
    )
    best_loss = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-loss-epoch={epoch:02d}-loss={val_loss:.5f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    best_r2 = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-r2-epoch={epoch:02d}-r2={val_r2:.5f}",
        monitor="val_r2",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    checkpoint_path = (
        find_resume_checkpoint(checkpoint_dir, fingerprint)
        if resume_config.get("enabled", True)
        else None
    )

    logger = WandbLogger(
        project=config["run_config"]["wandb_project"],
        name=config["run_config"]["name"],
        save_dir=str(checkpoint_dir),
        offline=bool(config["run_config"].get("wandb_offline", True)),
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
        log_every_n_steps=int(training.get("log_every_n_steps", 50)),
        num_sanity_val_steps=int(training.get("num_sanity_val_steps", 2)),
        strategy="ddp" if devices > 1 else "auto",
        gradient_clip_val=training["gradient_clip_val"],
        gradient_clip_algorithm="norm" if training["use_gradient_clipping"] else None,
        accelerator="gpu",
        devices=devices,
        default_root_dir=str(checkpoint_dir),
        max_epochs=int(training["epochs"]),
        accumulate_grad_batches=int(training["accumulate_grad_batches"]),
        sync_batchnorm=devices > 1,
        callbacks=[resume_metadata, rolling_checkpoint, best_loss, best_r2],
        limit_train_batches=training.get("limit_train_batches", 1.0),
        limit_val_batches=training.get("limit_val_batches", 1.0),
        limit_predict_batches=training.get("limit_predict_batches", 1.0),
    )
    trainer.fit(
        model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
        ckpt_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )

    # Save predictions
    if config["evaluation"].get("run_predictions", True):
        prediction_checkpoint = best_r2.best_model_path or best_loss.best_model_path
        if not prediction_checkpoint:
            raise RuntimeError("No checkpoint available for prediction")
        predictions: dict[str, torch.Tensor] = {}
        for split in ("val", "test"):
            outputs = trainer.predict(
                model=model,
                dataloaders=loaders[split],
                ckpt_path=prediction_checkpoint,
                return_predictions=True,
            )
            split_preds, split_targets = zip(*outputs)
            predictions[f"{split}_preds"] = torch.cat(split_preds).cpu()
            predictions[f"{split}_targets"] = torch.cat(split_targets).cpu()
        predictions["feature_names"] = FEATURE_NAMES
        predictions_dir = require_output_path(config["paths"]["predictions_dir"])
        predictions_dir.mkdir(parents=True, exist_ok=True)
        output_path = predictions_dir / f"{config['run_config']['name']}.pt"
        temporary_path = output_path.with_suffix(".partial")
        torch.save(predictions, temporary_path)
        os.replace(temporary_path, output_path)
        print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
