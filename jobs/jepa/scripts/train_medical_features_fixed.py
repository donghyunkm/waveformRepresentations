"""Attentive regression probe for medical feature prediction using a frozen native JEPA encoder.

Predicts 15 physiological features (HR, SBP, DBP, PP, MAP, ABP_area, PLETH_ACDC,
PLETH_amp, ECG_Ramp, HRV_RMSSD, HR_range, ShockIdx, PTT, dPdt_max, ABP_tau) from
the full token sequence using an attentive pooling head + regression output.

Uses the same patient-level train/val/test split as the hypotension probe.
Features are pre-computed and aligned to sample indices.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

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
from physiojepa.jepa import JEPASimpleLightning
from physiojepa.train import PatchTFTSingleOutcomeLightning

from pipeline_common import config_fingerprint, load_config, require_output_path
from resume_checkpoints import build_resume_callbacks, find_resume_checkpoint


# 15 features (excluding RESP-dependent: RR, PPV, PVI, RESP_amp)
FEATURE_NAMES = [
    "HR", "SBP", "DBP", "PP", "MAP", "ABP_area", "PLETH_ACDC", "PLETH_amp",
    "ECG_Ramp", "HRV_RMSSD", "HR_range", "ShockIdx", "PTT", "dPdt_max", "ABP_tau",
]
NUM_FEATURES = len(FEATURE_NAMES)

# Mapping from 19-feature index to our 15-feature index
# Original indices: 0=HR, 1=RR(skip), 2=SBP, 3=DBP, 4=PP, 5=MAP, 6=ABP_area,
# 7=PLETH_ACDC, 8=PLETH_amp, 9=ECG_Ramp, 10=HRV_RMSSD, 11=HR_range, 12=ShockIdx,
# 13=PPV(skip), 14=PVI(skip), 15=PTT, 16=dPdt_max, 17=ABP_tau, 18=RESP_amp(skip)
ORIG_INDICES = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17]


# ── Regression Lightning module ──────────────────────────────────────────────

class RegressionProbe(PatchTFTSingleOutcomeLightning):
    """Subclass that uses MSE loss for multi-target regression.

    Overrides training_step, validation_step, and predict_step.
    """

    def __init__(self, *args, num_features: int = NUM_FEATURES,
                 feature_means: torch.Tensor = None,
                 feature_stds: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_features = num_features
        # Register normalization stats as buffers (move with model to device)
        if feature_means is not None:
            self.register_buffer("feature_means", feature_means)
            self.register_buffer("feature_stds", feature_stds)
        else:
            self.register_buffer("feature_means", torch.zeros(num_features))
            self.register_buffer("feature_stds", torch.ones(num_features))

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.fine_tune:
            self.encoder.eval()
        return self

    def _masked_mse(self, pred, target):
        """MSE loss ignoring NaN targets."""
        valid = ~torch.isnan(target)
        if not valid.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return nn.functional.mse_loss(pred[valid], target[valid])

    def training_step(self, batch, batch_idx):
        if self.transforms is not None:
            batch = self.transforms(batch)
        x, y = batch
        pred = self(x)  # (bs, num_features)
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        # Normalize targets
        y_norm = (y - self.feature_means) / self.feature_stds
        loss_val = self._masked_mse(pred, y_norm)
        self.log("train_loss", loss_val, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        y_norm = (y - self.feature_means) / self.feature_stds
        loss_val = self._masked_mse(pred, y_norm)
        self.log("val_loss", loss_val, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        # Compute R² per feature (un-normalized)
        pred_unnorm = pred * self.feature_stds + self.feature_means
        valid = ~torch.isnan(y)
        if valid.any():
            # Overall R² across all features
            ss_res = ((pred_unnorm[valid] - y[valid]) ** 2).sum()
            ss_tot = ((y[valid] - y[valid].mean()) ** 2).sum()
            r2 = 1.0 - ss_res / (ss_tot + 1e-8)
            self.log("val_r2", r2, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        pred = self(x)
        if pred.dim() == 3:
            pred = pred.squeeze(-1)
        # Return un-normalized predictions
        pred_unnorm = pred * self.feature_stds + self.feature_means
        return pred_unnorm, y


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class MedicalFeatureDataset(Dataset):
    """Wraps a ForecastingDataset but overrides labels with medical feature targets.

    Only includes windows that have valid (non-all-NaN) feature values.
    """

    def __init__(self, base_dataset: ForecastingDataset, features: np.ndarray,
                 valid_indices: np.ndarray):
        self.base_dataset = base_dataset
        self.features = features.astype(np.float32)
        self.valid_indices = valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x, _ = self.base_dataset[real_idx]
        y = torch.tensor(self.features[real_idx], dtype=torch.float32)
        return x, y


# ── Feature computation ───────────────────────────────────────────────────────

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


def compute_features_for_samples(
    samples: pd.DataFrame,
    config: dict,
    cache_path: Path,
) -> np.ndarray:
    """Compute or load cached physiological features for all samples.

    Uses icuDataExtraction alignment: matches each sample's center time to the
    nearest icuDataExtraction window and uses its pre-computed features.

    The window center is computed from the segment start time encoded in the
    Zarr file path (e.g. p000188-2149-04-17-22-52.zarr.zip) plus the sample
    offset within that container.
    """
    if cache_path.is_file():
        print(f"  Loading cached features from {cache_path}")
        return np.load(cache_path)

    print(f"  Computing features via icuDataExtraction alignment ({len(samples)} samples)...")

    icu_output = Path(config["paths"]["icu_output_dir"])
    icu_patient_ids = np.load(icu_output / "patient_ids.npy", allow_pickle=True)
    icu_window_times = np.load(icu_output / "window_times.npy")

    # Load X_stats (N_icu, 19, 109) and compute nanmedian per feature
    x_stats_path = icu_output / "X_stats.npy"
    if x_stats_path.exists():
        print(f"  Loading X_stats from {x_stats_path}...")
        x_stats = np.load(str(x_stats_path), mmap_mode="r")
        # Compute nanmedian across sub-windows for each feature
        # This gives (N_icu, 19) - same as what probe_medical_features does
        icu_features = np.nanmedian(x_stats[:, :, :], axis=2)
    else:
        raise FileNotFoundError(f"X_stats.npy not found at {x_stats_path}")

    # Alignment constants
    EPOCH_OFFSET = 946684800.0
    FS = config["dataset"]["frequency"]
    max_tolerance_sec = 150.0

    # Build per-patient ICU index (sorted by time for binary search)
    icu_by_patient: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid in np.unique(icu_patient_ids):
        mask = icu_patient_ids == pid
        times = icu_window_times[mask]
        indices = np.where(mask)[0]
        order = np.argsort(times)
        icu_by_patient[str(pid)] = (times[order], indices[order])

    # Compute window centers from file path timestamps (vectorized)
    unique_fps = samples["file_path"].unique()
    fp_to_start = {fp: _parse_seg_start_posix(fp) for fp in unique_fps}
    n_no_path = sum(1 for v in fp_to_start.values() if v is None)
    if n_no_path > 0:
        print(f"  Warning: {n_no_path} unique file_paths could not be parsed")

    seg_starts = samples["file_path"].map(fp_to_start).values.astype(np.float64)
    start_idxs = samples["start_idx"].values
    end_idxs = samples["end_idx"].values
    subject_ids = samples["subject_id"].values.astype(str)
    centers = seg_starts + (start_idxs + end_idxs) / 2 / FS - EPOCH_OFFSET

    # Per-patient vectorized nearest-neighbor matching
    features = np.full((len(samples), NUM_FEATURES), np.nan, dtype=np.float32)
    n_matched = 0

    for pid, (icu_times, icu_indices) in icu_by_patient.items():
        pid_mask = subject_ids == pid
        if not pid_mask.any():
            continue
        pid_sample_indices = np.where(pid_mask)[0]
        pid_centers = centers[pid_sample_indices]

        valid = ~np.isnan(pid_centers)
        if not valid.any():
            continue
        valid_local = np.where(valid)[0]
        valid_centers = pid_centers[valid]

        insert_pos = np.searchsorted(icu_times, valid_centers)

        # Find nearest ICU window among candidates at insert_pos-1 and insert_pos
        best_dist = np.full(len(valid_centers), np.inf)
        best_icu_idx = np.full(len(valid_centers), -1, dtype=np.int64)

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
            best_icu_idx[better] = icu_indices[cands_clipped[better]]

        within_tol = (best_dist <= max_tolerance_sec) & (best_icu_idx >= 0)
        matched_sample_indices = pid_sample_indices[valid_local[within_tol]]
        matched_icu_indices = best_icu_idx[within_tol]

        # Extract the 15 non-RESP features for matched windows
        for sample_idx, icu_idx in zip(matched_sample_indices, matched_icu_indices):
            features[sample_idx] = icu_features[icu_idx][ORIG_INDICES]

        n_matched += int(within_tol.sum())

    print(f"  Aligned {n_matched}/{len(samples)} windows ({100*n_matched/len(samples):.1f}%)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    print(f"  Cached to {cache_path}")
    return features


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        raise FileNotFoundError(f"Required cache missing: {cache_path}")
    samples = pd.read_csv(cache_path)
    if "subject_id" not in samples:
        samples["subject_id"] = samples["file_path"].map(
            lambda path: zarr_record_name(path).split("-", 1)[0]
        )

    containers_override = os.environ.get("PHYSIOJEPA_CONTAINERS_OVERRIDE")
    if containers_override:
        samples["file_path"] = samples["file_path"].map(
            lambda p: os.path.join(containers_override, os.path.basename(p))
        )

    return outcomes, samples.reset_index(drop=True)


def _base_dataset(config: dict, outcomes: pd.DataFrame, samples: pd.DataFrame):
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


torch.set_float32_matmul_precision("high")
torch.backends.cuda.enable_flash_sdp(True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config, config_path = load_config("configs/train_medical_features_fixed.yaml")
    pl.seed_everything(int(config["run_config"]["random_state"]), workers=True)
    print(f"Using configuration: {config_path}")

    # Load encoder
    pretrained_path = Path(config["paths"]["pretrained_encoder_path"]).resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"JEPA checkpoint not found: {pretrained_path}")
    encoder = JEPASimpleLightning.load_from_checkpoint(
        str(pretrained_path), map_location="cpu"
    )

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

    # Build datasets (only windows with at least 1 valid feature)
    datasets = {}
    for split in ("train", "val", "test"):
        outcomes, samples = split_inputs[split]
        base_ds = _base_dataset(config, outcomes, samples)
        feats = all_features[split]
        # Valid = at least half of features are non-NaN
        valid_mask = np.sum(~np.isnan(feats), axis=1) >= NUM_FEATURES // 2
        valid_indices = np.where(valid_mask)[0]
        datasets[split] = MedicalFeatureDataset(base_ds, feats, valid_indices)
        print(f"  {split}: {len(valid_indices)}/{len(samples)} valid windows")

    # Compute normalization stats from training set
    train_feats = all_features["train"][np.sum(~np.isnan(all_features["train"]), axis=1) >= NUM_FEATURES // 2]
    feature_means = torch.tensor(np.nanmean(train_feats, axis=0), dtype=torch.float32)
    feature_stds = torch.tensor(np.nanstd(train_feats, axis=0), dtype=torch.float32)
    feature_stds = torch.clamp(feature_stds, min=1e-6)  # Avoid division by zero
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
        embed_dim=encoder.encoder.d_model,
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
        metrics={},  # R² computed manually in validation_step
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
