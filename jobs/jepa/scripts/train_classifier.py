"""Train an AttentiveClassifier head from pre-computed encoder embeddings.

This script loads cached embeddings (produced by extract_embeddings.py) and
trains only the classification head. This is 50-100x faster than training
through the full encoder because:
  - No ZipStore I/O during training
  - No encoder forward pass (21.6M params for JEPA, similar for PatchTST)
  - Embeddings fit in GPU memory or are loaded from fast local tensors

Supports augmentations on embeddings (jitter, channel masking, mixup) and
balanced sampling.

Usage:
    PHYSIOJEPA_CONFIG=configs/train_classifier_jepa.yaml python scripts/train_classifier.py
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchmetrics.classification import AUROC, AveragePrecision

from physiojepa.augmentations import (
    MixupCallbackClassification,
    TransformsCallback,
    channel_masking,
    jitter_augmentation,
)
from physiojepa.heads import AttentiveClassifier

# Reuse pipeline utilities from the JEPA scripts directory.
_JEPA_SCRIPTS = Path(__file__).resolve().parent
if str(_JEPA_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_JEPA_SCRIPTS))

from pipeline_common import config_fingerprint, load_config, require_output_path
from resume_checkpoints import build_resume_callbacks, find_resume_checkpoint


torch.set_float32_matmul_precision("high")


class EmbeddingClassifierLightning(pl.LightningModule):
    """Lightning module that trains a classifier head on pre-computed embeddings."""

    def __init__(
        self,
        classifier: AttentiveClassifier,
        learning_rate: float,
        train_size: int,
        batch_size: int,
        n_gpus: int,
        epochs: int,
        optimizer_type: str = "AdamW",
        weight_decay: float = 1e-4,
        use_weight_decay_scheduler: bool = False,
        final_weight_decay: float = 0.4,
        scheduler_type: str = "OneCycle",
        scheduler_kwargs: dict | None = None,
        class_weights: torch.Tensor | None = None,
        transforms: TransformsCallback | None = None,
        mixup_callback: MixupCallbackClassification | None = None,
    ):
        super().__init__()
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.train_size = train_size
        self.batch_size = batch_size
        self.n_gpus = max(1, n_gpus)
        self.epochs = epochs
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.use_weight_decay_scheduler = use_weight_decay_scheduler
        self.final_weight_decay = final_weight_decay
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.class_weights = class_weights
        self.transforms = transforms
        self.mixup_callback = mixup_callback

        self.ipe = math.ceil(train_size / (batch_size * self.n_gpus))

        self.metrics = nn.ModuleDict({
            "auroc": AUROC(task="binary"),
            "auprc": AveragePrecision(task="binary"),
        })

        self.save_hyperparameters(ignore=["classifier", "transforms", "mixup_callback"])

    def forward(self, x):
        # x: [bs, n_channels, d_model, n_patches] (fp16 embeddings cast to working precision)
        if torch.isnan(x).any():
            warnings.warn("NaN values in embedding input to classifier")
        return self.classifier(x)

    def on_train_batch_start(self, batch, batch_idx):
        if self.use_weight_decay_scheduler:
            step = self.global_step
            T_max = int(self.ipe * self.epochs)
            progress = step / T_max
            new_wd = (
                self.final_weight_decay
                + (self.weight_decay - self.final_weight_decay)
                * 0.5
                * (1.0 + math.cos(math.pi * progress))
            )
            if self.final_weight_decay <= self.weight_decay:
                new_wd = max(self.final_weight_decay, new_wd)
            else:
                new_wd = min(self.final_weight_decay, new_wd)
            for group in self.optimizer.param_groups:
                if ("WD_exclude" not in group) or not group["WD_exclude"]:
                    group["weight_decay"] = new_wd

    def training_step(self, batch, batch_idx):
        if self.transforms is not None:
            batch = self.transforms(batch)
        if self.mixup_callback is not None:
            batch = self.mixup_callback(batch)
        x, y = batch
        x = self(x)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self.class_weights.to(x.device) if self.class_weights is not None else None
        )
        loss_val = loss_fn(x, y.float())
        self.log("train_loss", loss_val, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self(x)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=self.class_weights.to(x.device) if self.class_weights is not None else None
        )
        loss_val = loss_fn(x, y.float())
        self.log("val_loss", loss_val, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        x_probs = torch.sigmoid(x)
        for metric in self.metrics:
            self.metrics[metric].update(x_probs, y.long())

    def on_validation_epoch_end(self):
        for name, metric in self.metrics.items():
            self.log(f"val_{name}", metric.compute(), prog_bar=True, sync_dist=True)
            metric.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self(x)
        return preds, y

    def configure_optimizers(self):
        if self.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.classifier.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.Adam(
                self.classifier.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        if self.scheduler_type == "OneCycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.scheduler_kwargs.get("max_lr", self.learning_rate * 25),
                total_steps=self.ipe * self.epochs,
                div_factor=self.scheduler_kwargs.get("div_factor", 25),
                final_div_factor=self.scheduler_kwargs.get("final_div_factor", 10000),
                pct_start=self.scheduler_kwargs.get("pct_start", 0.3),
                anneal_strategy=self.scheduler_kwargs.get("anneal_strategy", "cos"),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return optimizer


def main() -> None:
    config, config_path = load_config("configs/train_classifier.yaml")
    pl.seed_everything(int(config["run_config"]["random_state"]), workers=True)
    print(f"Using configuration: {config_path}")

    # Load pre-computed embeddings (supports both single-file and sharded formats)
    embeddings_dir = Path(config["paths"]["embeddings_dir"]).resolve()
    print(f"Loading embeddings from: {embeddings_dir}")

    splits_data = {}
    for split in ("train", "val", "test"):
        split_dir = embeddings_dir / split
        # Check for sharded format first
        shard_files = sorted(split_dir.glob("shard_*.pt"))
        if shard_files:
            print(f"  {split}: loading {len(shard_files)} shards...")
            all_emb = []
            all_lbl = []
            for shard_path in shard_files:
                shard = torch.load(shard_path, map_location="cpu", weights_only=True)
                all_emb.append(shard["embeddings"])
                all_lbl.append(shard["labels"])
            embeddings = torch.cat(all_emb, dim=0)
            labels = torch.cat(all_lbl, dim=0)
            del all_emb, all_lbl
        else:
            # Fall back to single-file format
            emb_path = split_dir / "embeddings.pt"
            lbl_path = split_dir / "labels.pt"
            if not emb_path.is_file() or not lbl_path.is_file():
                raise FileNotFoundError(
                    f"Missing cached embeddings for split '{split}' at {split_dir}. "
                    "Run extract_embeddings.py first."
                )
            embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
            labels = torch.load(lbl_path, map_location="cpu", weights_only=True)
        splits_data[split] = (embeddings, labels)
        print(f"  {split}: embeddings={embeddings.shape}, labels={labels.shape}")

    # Infer embedding dimensions from data
    train_emb, train_lbl = splits_data["train"]
    # train_emb: [N, n_channels, d_model, n_patches]
    _, n_channels, d_model, n_patches = train_emb.shape

    training = config["training"]
    batch_size = int(training["batch_size"])
    workers = int(training.get("num_workers", 4))

    # Build datasets
    datasets = {
        split: TensorDataset(emb, lbl)
        for split, (emb, lbl) in splits_data.items()
    }

    # Balanced sampling for training
    train_labels_flat = train_lbl.squeeze()
    label_counts = torch.bincount(train_labels_flat.long())
    if len(label_counts) != 2:
        raise ValueError(f"Expected binary labels, got {len(label_counts)} classes")
    label_weights = 1.0 / label_counts.float()
    sample_weights = label_weights[train_labels_flat.long()]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
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

    # Build classifier head
    head_config = config["lp_head"]
    head = AttentiveClassifier(
        embed_dim=d_model,
        num_heads=int(head_config["num_heads"]),
        mlp_ratio=head_config["mlp_ratio"],
        depth=head_config["depth"],
        c_in=n_channels,
        norm_layer=nn.LayerNorm,
        init_std=head_config["init_std"],
        qkv_bias=head_config["qkv_bias"],
        num_classes=1,
        complete_block=head_config["complete_block"],
        affine=head_config["affine"],
    )

    # Augmentations on embeddings
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
        if config.get("augmentations", {}).get("use_transforms", True)
        else None
    )
    mixup = (
        MixupCallbackClassification(
            num_classes=1,
            mixup_alpha=training.get("mixup_alpha", 0.2),
            ignore_index=2,
        )
        if training.get("mixup", True)
        else None
    )
    class_weights = (
        torch.tensor([label_counts[0].float() / label_counts[1].float()])
        if training.get("use_class_weights", False)
        else None
    )

    scheduler = config["scheduler"]
    model = EmbeddingClassifierLightning(
        classifier=head,
        learning_rate=config["optimizer"]["learning_rate"],
        train_size=len(datasets["train"]),
        batch_size=batch_size,
        n_gpus=int(training["n_gpus"]),
        epochs=int(training["epochs"]),
        optimizer_type=config["optimizer"]["optimizer_type"],
        weight_decay=config["optimizer"]["weight_decay"],
        use_weight_decay_scheduler=config["optimizer"]["use_weight_decay_scheduler"],
        final_weight_decay=config["optimizer"]["final_weight_decay"],
        scheduler_type=scheduler["scheduler_type"],
        scheduler_kwargs={
            "max_lr": scheduler["max_lr"],
            "div_factor": scheduler["div_factor"],
            "final_div_factor": scheduler["final_div_factor"],
            "pct_start": scheduler["pct_start"],
            "anneal_strategy": scheduler["anneal_strategy"],
        },
        class_weights=class_weights,
        transforms=transforms,
        mixup_callback=mixup,
    )

    # Checkpointing and resume
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
        val_check_interval=training.get("val_check_interval", 1.0),
        log_every_n_steps=int(training.get("log_every_n_steps", 50)),
        num_sanity_val_steps=int(training.get("num_sanity_val_steps", 2)),
        strategy="ddp" if devices > 1 else "auto",
        gradient_clip_val=training.get("gradient_clip_val", 1.0),
        gradient_clip_algorithm=(
            "norm" if training.get("use_gradient_clipping", True) else None
        ),
        accelerator="gpu",
        devices=devices,
        default_root_dir=str(checkpoint_dir),
        max_epochs=int(training["epochs"]),
        accumulate_grad_batches=int(training.get("accumulate_grad_batches", 1)),
        sync_batchnorm=devices > 1,
        callbacks=[
            resume_metadata,
            rolling_checkpoint,
            best_auprc,
            best_loss,
        ],
    )
    trainer.fit(
        model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["val"],
        ckpt_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )

    # Run predictions on val and test
    if config.get("evaluation", {}).get("run_predictions", True):
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
