"""Attentive probe for hemodynamic cluster classification using a frozen PatchTST encoder.

Predicts the 7-class hemodynamic cluster (KMeans on temporal correlation features
from icuDataExtraction) using the same attentive pooling head architecture as the
hypotension probe, but with CrossEntropyLoss and multiclass metrics.

Only windows with valid hemo cluster labels (aligned from icuDataExtraction via
window_hemo_clusters.npz) are used. The same patient-level train/val/test split
is applied.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy

from physiojepa.augmentations import (
    MixupCallbackClassification,
    TransformsCallback,
    channel_masking,
    jitter_augmentation,
)
from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, ForecastingDataset
from physiojepa.data_preprocessing import zarr_record_name
from physiojepa.heads import AttentiveClassifier
from physiojepa.patchtst import PatchTFTSimpleLightning
from physiojepa.train import PatchTFTSingleOutcomeLightning

from pipeline_common import config_fingerprint, load_config, require_output_path
from resume_checkpoints import build_resume_callbacks, find_resume_checkpoint

NUM_CLASSES = 7


# ── Multiclass Lightning module ───────────────────────────────────────────────

class MulticlassProbe(PatchTFTSingleOutcomeLightning):
    """Subclass that uses CrossEntropyLoss for multiclass classification.

    Overrides training_step, validation_step, and predict_step from the binary
    BCEWithLogitsLoss parent.
    """

    def __init__(self, *args, num_classes: int = NUM_CLASSES, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.fine_tune:
            self.encoder.eval()
        return self

    def training_step(self, batch, batch_idx):
        if self.transforms is not None:
            batch = self.transforms(batch)
        if self.mixup_callback is not None:
            batch = self.mixup_callback(batch)
        x, y = batch
        logits = self(x)  # (bs, num_classes)
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        # y may be float after mixup — use CE with soft targets
        if y.dtype in (torch.float32, torch.float16, torch.bfloat16) and y.dim() == 2:
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss_val = -(y * log_probs).sum(dim=-1).mean()
        else:
            loss_val = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
            )(logits, y.long())
        self.log("train_loss", loss_val, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        loss_val = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )(logits, y.long())
        self.log("val_loss", loss_val, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        probs = torch.softmax(logits, dim=-1)
        for metric in self.metrics:
            self.metrics[metric].update(probs, y.long())

    def on_validation_epoch_end(self):
        for name, metric in self.metrics.items():
            metric_val = metric.compute()
            self.log(f"val_{name}", metric_val, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            metric.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self(x)
        if logits.dim() == 3:
            logits = logits.squeeze(-1)
        probs = torch.softmax(logits, dim=-1)
        return probs, y


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class HemoClusterDataset(Dataset):
    """Wraps a ForecastingDataset but overrides labels with hemo cluster labels."""

    def __init__(self, base_dataset: ForecastingDataset, hemo_labels: np.ndarray,
                 valid_indices: np.ndarray):
        self.base_dataset = base_dataset
        self.hemo_labels = hemo_labels
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x, _ = self.base_dataset[real_idx]
        y = self.hemo_labels[real_idx]
        return x, torch.tensor(y, dtype=torch.long)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_split_inputs(config: dict, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load outcome df and sample cache for a given split."""
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

    containers_override = os.environ.get("PHYSIOJEPA_CONTAINERS_OVERRIDE")
    if containers_override:
        samples["file_path"] = samples["file_path"].map(
            lambda p: os.path.join(containers_override, os.path.basename(p))
        )

    return outcomes, samples.reset_index(drop=True)


def _base_dataset(config: dict, outcomes: pd.DataFrame, samples: pd.DataFrame):
    """Build the ForecastingDataset (labels will be overridden by wrapper)."""
    dataset_cfg = config["dataset"]
    return ForecastingDataset(
        channels=dataset_cfg["channels"],
        forecast_window_sec=dataset_cfg["forecast_window_sec"],
        outcome_df=outcomes,
        outcome_df_outcome_col=dataset_cfg["y_outcome"],
        file_col="file_path",
        y_date_column="date",
        outcome_df_seconds_since_column="Time Stamp (seconds)",
        outcome_df_duration_column="event_length",
        sample_df=samples,
        sample_seq_len_sec=dataset_cfg["sample_seq_len_seconds"],
        frequency=dataset_cfg["frequency"],
        butterworth_filters=None,
        median_filter_kernel_size=None,
        clip_interpolations=CLIP_INTERPOLATE_RANGES,
        constant_nan_tolerance=dataset_cfg["constant_nan_tolerance"],
        require_all_channels=dataset_cfg["require_all_channels"],
        infer_forecast_windows=dataset_cfg["infer_forecast_windows"],
        normalize_signals=dataset_cfg["normalize_signals"],
    )


def _parse_seg_start_posix(file_path: str) -> float | None:
    """Extract segment start time as POSIX timestamp from Zarr file path.

    Zarr containers are named like: p000188-2149-04-17-22-52.zarr.zip
    """
    import re
    from datetime import datetime as _dt
    match = re.search(r"p\d+-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", file_path)
    if match is None:
        return None
    y, mo, d, h, mi = (int(x) for x in match.groups())
    return _dt(y, mo, d, h, mi, 0).timestamp()


def align_hemo_to_split(
    samples: pd.DataFrame,
    icu_cluster_labels: np.ndarray,
    icu_patient_ids: np.ndarray,
    icu_window_times: np.ndarray,
    max_tolerance_sec: float = 150.0,
) -> np.ndarray:
    """Align icuDataExtraction window-level cluster labels to PhysioJEPA samples.

    Uses timestamp matching: for each PhysioJEPA sample, find the closest
    icuDataExtraction window for the same patient within tolerance.

    The window center is computed from the segment start time encoded in the
    Zarr file path (e.g. p000188-2149-04-17-22-52.zarr.zip) plus the sample
    offset within that container.  icuDataExtraction times are in seconds since
    2000-01-01, so we subtract the POSIX epoch offset (946684800).

    Returns array of shape (len(samples),) with cluster labels (0-6) or -1 if unmatched.
    """
    # Epoch offset: icuDataExtraction uses seconds since 2000-01-01
    EPOCH_OFFSET = 946684800.0
    FS = 125

    # Build per-patient index for icu data (sorted by time for binary search)
    icu_by_patient: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid in np.unique(icu_patient_ids):
        mask = icu_patient_ids == pid
        times = icu_window_times[mask]
        labels = icu_cluster_labels[mask]
        order = np.argsort(times)
        icu_by_patient[str(pid)] = (times[order], labels[order])

    # Compute all window centers, caching seg_start per unique file_path
    file_paths = samples["file_path"].values
    start_idxs = samples["start_idx"].values
    end_idxs = samples["end_idx"].values
    subject_ids = samples["subject_id"].values

    # Vectorized: parse unique paths once, map to all rows, then numpy arithmetic
    unique_fps = samples["file_path"].unique()
    fp_to_start = {fp: _parse_seg_start_posix(fp) for fp in unique_fps}
    n_no_path = sum(1 for v in fp_to_start.values() if v is None)
    seg_starts = samples["file_path"].map(fp_to_start).values.astype(np.float64)
    centers = seg_starts + (start_idxs + end_idxs) / 2 / FS - EPOCH_OFFSET

    # Per-patient vectorized nearest-neighbor matching
    matched = np.full(len(samples), -1, dtype=np.int64)
    n_matched = 0

    for pid, (icu_times, icu_cls) in icu_by_patient.items():
        pid_mask = subject_ids == pid
        if not pid_mask.any():
            continue
        pid_indices = np.where(pid_mask)[0]
        pid_centers = centers[pid_indices]

        valid = ~np.isnan(pid_centers)
        if not valid.any():
            continue
        valid_local = np.where(valid)[0]
        valid_centers = pid_centers[valid]

        insert_pos = np.searchsorted(icu_times, valid_centers)

        # Find nearest ICU time among candidates at insert_pos-1 and insert_pos
        best_dist = np.full(len(valid_centers), np.inf)
        best_cluster = np.full(len(valid_centers), -1, dtype=np.int64)

        for offset in (0, -1):
            cands = insert_pos + offset
            in_bounds = (cands >= 0) & (cands < len(icu_times))
            cands_clipped = np.clip(cands, 0, len(icu_times) - 1)
            dists = np.where(
                in_bounds,
                np.abs(icu_times[cands_clipped] - valid_centers),
                np.inf,
            )
            better = dists < best_dist
            best_dist[better] = dists[better]
            best_cluster[better] = icu_cls[cands_clipped[better]]

        within_tol = best_dist <= max_tolerance_sec
        matched_indices = pid_indices[valid_local[within_tol]]
        matched[matched_indices] = best_cluster[within_tol]
        n_matched += int(within_tol.sum())

    if n_no_path > 0:
        print(f"  Warning: {n_no_path} windows had unparseable file_path")
    print(f"  Aligned {n_matched}/{len(samples)} windows ({100*n_matched/len(samples):.1f}%)")
    return matched


torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config, config_path = load_config("configs/train_hemo_cluster_fixed.yaml")
    pl.seed_everything(int(config["run_config"]["random_state"]), workers=True)
    print(f"Using configuration: {config_path}")

    # Load encoder
    pretrained_path = Path(config["paths"]["pretrained_encoder_path"]).resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"PatchTST checkpoint does not exist: {pretrained_path}")
    encoder = PatchTFTSimpleLightning.load_from_checkpoint(
        str(pretrained_path), map_location="cpu"
    )
    d_model = encoder.model.d_model
    print(f"Loaded PatchTST encoder: d_model={d_model}")

    # Load icuDataExtraction cluster data
    icu_output = Path(config["paths"]["icu_output_dir"])
    icu_cluster_labels = np.load(icu_output / "cluster_labels.npy")
    icu_patient_ids = np.load(icu_output / "patient_ids.npy", allow_pickle=True)
    icu_window_times = np.load(icu_output / "window_times.npy")
    print(f"Loaded icuDataExtraction: {len(icu_cluster_labels)} windows, "
          f"{len(np.unique(icu_patient_ids))} patients, {NUM_CLASSES} clusters")

    # Load splits and build datasets
    split_inputs = {
        split: _load_split_inputs(config, split)
        for split in ("train", "val", "test")
    }

    hemo_cache_path = Path(config["paths"].get(
        "hemo_clusters_cache",
        "/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/probing/clustering/window_hemo_clusters.npz"
    ))

    hemo_labels = {}
    for split in ("train", "val", "test"):
        outcomes, samples = split_inputs[split]
        cache_path = Path(config["paths"]["sample_cache_dir"]) / \
            f"{config['paths']['dataset_filename']}-{split}_hemo_clusters.npy"

        if cache_path.is_file():
            print(f"Loading cached hemo labels for {split}: {cache_path}")
            hemo_labels[split] = np.load(cache_path)
        elif split == "test" and hemo_cache_path.is_file():
            print(f"Using pre-computed test hemo clusters from {hemo_cache_path}")
            data = np.load(hemo_cache_path, allow_pickle=True)
            hemo_labels[split] = data["hemo_clusters"]
            np.save(cache_path, hemo_labels[split])
        else:
            print(f"Aligning hemo clusters for {split} ({len(samples)} windows)...")
            hemo_labels[split] = align_hemo_to_split(
                samples, icu_cluster_labels, icu_patient_ids, icu_window_times
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, hemo_labels[split])
            print(f"  Cached to {cache_path}")

    # Build wrapped datasets
    datasets = {}
    for split in ("train", "val", "test"):
        outcomes, samples = split_inputs[split]
        base_ds = _base_dataset(config, outcomes, samples)
        valid_mask = hemo_labels[split] >= 0
        valid_indices = np.where(valid_mask)[0]
        datasets[split] = HemoClusterDataset(base_ds, hemo_labels[split], valid_indices)
        n_valid = len(valid_indices)
        class_dist = np.bincount(hemo_labels[split][valid_mask].astype(int), minlength=NUM_CLASSES)
        print(f"  {split}: {n_valid}/{len(samples)} valid windows, "
              f"class dist: {class_dist.tolist()}")

    training = config["training"]
    batch_size = int(training["batch_size"])
    workers = int(training["num_workers"])

    # Weighted sampler
    train_labels = hemo_labels["train"][hemo_labels["train"] >= 0]
    class_counts = np.bincount(train_labels.astype(int), minlength=NUM_CLASSES).astype(float)
    class_weights_sample = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights_sample[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
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
        num_classes=NUM_CLASSES,
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
    mixup = (
        MixupCallbackClassification(
            num_classes=NUM_CLASSES,
            mixup_alpha=training["mixup_alpha"],
            ignore_index=-1,
        )
        if training["mixup"]
        else None
    )

    loss_class_weights = (
        torch.tensor(class_weights_sample / class_weights_sample.sum() * NUM_CLASSES,
                     dtype=torch.float32)
        if training["use_class_weights"]
        else None
    )

    scheduler = config["scheduler"]
    model = MulticlassProbe(
        learning_rate=config["optimizer"]["learning_rate"],
        train_size=len(datasets["train"]),
        n_gpus=int(training["n_gpus"]),
        batch_size=batch_size,
        linear_probing_head=head,
        preloaded_model=encoder,
        num_classes=NUM_CLASSES,
        metrics={
            "auroc": MulticlassAUROC(num_classes=NUM_CLASSES, average="macro"),
            "bal_acc": MulticlassAccuracy(num_classes=NUM_CLASSES, average="macro"),
        },
        class_weights=loss_class_weights,
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
    best_bal_acc = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-balacc-epoch={epoch:02d}-balacc={val_bal_acc:.5f}",
        monitor="val_bal_acc",
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
        gradient_clip_algorithm="norm" if training["use_gradient_clipping"] else None,
        accelerator="gpu",
        devices=devices,
        default_root_dir=str(checkpoint_dir),
        max_epochs=int(training["epochs"]),
        accumulate_grad_batches=int(training["accumulate_grad_batches"]),
        sync_batchnorm=devices > 1,
        callbacks=[resume_metadata, rolling_checkpoint, best_bal_acc, best_loss],
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
        prediction_checkpoint = best_bal_acc.best_model_path
        if not prediction_checkpoint:
            raise RuntimeError("No best-bal_acc checkpoint available for prediction")
        predictions: dict[str, torch.Tensor] = {}
        for split in ("val", "test"):
            outputs = trainer.predict(
                model=model,
                dataloaders=loaders[split],
                ckpt_path=prediction_checkpoint,
                return_predictions=True,
            )
            split_probs, split_targets = zip(*outputs)
            predictions[f"{split}_preds"] = torch.cat(split_probs).cpu()
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
