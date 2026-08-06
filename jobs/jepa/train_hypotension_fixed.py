"""Leakage-safe downstream hypotension probe for a native JEPA checkpoint."""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchmetrics.classification import AUROC, AveragePrecision

from physiojepa.augmentations import (
    MixupCallbackClassification,
    TransformsCallback,
    channel_masking,
    jitter_augmentation,
)
from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, ForecastingDataset
from physiojepa.data_preprocessing import zarr_record_name
from physiojepa.heads import AttentiveClassifier
from physiojepa.jepa import JEPASimpleLightning
from physiojepa.train import PatchTFTSingleOutcomeLightning

from pipeline_common import config_fingerprint, load_config, require_output_path
from resume_checkpoints import build_resume_callbacks, find_resume_checkpoint


class FrozenNativeJEPAProbe(PatchTFTSingleOutcomeLightning):
    """Keep the native JEPA encoder in evaluation mode during linear probing."""

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.fine_tune:
            self.encoder.eval()
        return self


torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)


def _load_split_inputs(config: dict, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    maximum = config["training"].get(f"max_{split}_samples")
    if maximum is not None:
        samples = samples.sample(
            n=min(int(maximum), len(samples)),
            random_state=int(config["run_config"]["random_state"]),
        ).sort_values(["file_path", "start_idx"])
    return outcomes, samples.reset_index(drop=True)


def _dataset(config: dict, outcomes: pd.DataFrame, samples: pd.DataFrame):
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


def main() -> None:
    config, config_path = load_config("train_hypotension_fixed.yaml")
    pl.seed_everything(int(config["run_config"]["random_state"]), workers=True)
    print(f"Using configuration: {config_path}")

    pretrained_path = Path(config["paths"]["pretrained_encoder_path"]).resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"Native JEPA checkpoint does not exist: {pretrained_path}")
    encoder = JEPASimpleLightning.load_from_checkpoint(
        str(pretrained_path), map_location="cpu"
    )

    split_inputs = {
        split: _load_split_inputs(config, split)
        for split in ("train", "val", "test")
    }
    datasets = {
        split: _dataset(config, outcomes, samples)
        for split, (outcomes, samples) in split_inputs.items()
    }
    training = config["training"]
    batch_size = int(training["batch_size"])
    workers = int(training["num_workers"])
    forecast_seconds = int(config["dataset"]["forecast_window_sec"][0])
    label_column = f"outcome_val_{forecast_seconds}sec"

    label_counts = datasets["train"].sample_df[label_column].value_counts().sort_index()
    if set(label_counts.index) != {0, 1}:
        raise ValueError(f"Training labels must contain binary classes: {label_counts}")
    label_weights = 1.0 / label_counts
    sample_weights = datasets["train"].sample_df[label_column].map(label_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights.to_numpy(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
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

    head_config = config["lp_head"]
    head = AttentiveClassifier(
        embed_dim=encoder.encoder.d_model,
        num_heads=int(head_config["num_heads"]),
        mlp_ratio=head_config["mlp_ratio"],
        depth=head_config["depth"],
        c_in=len(config["dataset"]["channels"]),
        norm_layer=torch.nn.LayerNorm,
        init_std=head_config["init_std"],
        qkv_bias=head_config["qkv_bias"],
        num_classes=1,
        complete_block=head_config["complete_block"],
        affine=head_config["affine"],
    )
    transforms = (
        TransformsCallback(
            transforms=[
                partial(
                    jitter_augmentation,
                    mask_ratio=0.05,
                    jitter_ratio=0.05,
                    p=0.5,
                ),
                partial(
                    channel_masking,
                    dim=1,
                    p=0.1,
                    specific_channels=None,
                ),
            ]
        )
        if config["dataset"]["use_transforms"]
        else None
    )
    mixup = (
        MixupCallbackClassification(
            num_classes=1,
            mixup_alpha=training["mixup_alpha"],
            ignore_index=2,
        )
        if training["mixup"]
        else None
    )
    class_weights = (
        torch.tensor([label_counts.loc[0] / label_counts.loc[1]])
        if training["use_class_weights"]
        else None
    )
    scheduler = config["scheduler"]
    model = FrozenNativeJEPAProbe(
        learning_rate=config["optimizer"]["learning_rate"],
        train_size=len(datasets["train"]),
        n_gpus=int(training["n_gpus"]),
        batch_size=batch_size,
        linear_probing_head=head,
        preloaded_model=encoder,
        metrics={
            "auroc": AUROC(task="binary"),
            "auprc": AveragePrecision(task="binary"),
        },
        class_weights=class_weights,
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
        mixup_callback=mixup,
        transforms=transforms,
    )

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
    best_auprc = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-auprc-epoch={epoch:02d}-auprc={val_auprc:.5f}",
        monitor="val_auprc",
        mode="max",
        save_top_k=1,
        auto_insert_metric_name=False,
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
        gradient_clip_algorithm=(
            "norm" if training["use_gradient_clipping"] else None
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
            best_auprc,
            best_loss,
        ],
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

    if config["evaluation"].get("run_predictions", True):
        prediction_checkpoint = best_auprc.best_model_path
        if not prediction_checkpoint:
            raise RuntimeError("No best-AUPRC checkpoint is available for prediction")
        predictions: dict[str, torch.Tensor] = {}
        for split in ("val", "test"):
            outputs = trainer.predict(
                model=model,
                dataloaders=loaders[split],
                ckpt_path=prediction_checkpoint,
                return_predictions=True,
            )
            split_predictions, split_targets = zip(*outputs)
            predictions[f"{split}_preds"] = torch.cat(split_predictions).cpu()
            predictions[f"{split}_targets"] = torch.cat(split_targets).cpu()
        predictions_dir = require_output_path(config["paths"]["predictions_dir"])
        predictions_dir.mkdir(parents=True, exist_ok=True)
        output_path = predictions_dir / f"{config['run_config']['name']}.pt"
        temporary_path = output_path.with_suffix(".partial")
        torch.save(predictions, temporary_path)
        os.replace(temporary_path, output_path)
        print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
