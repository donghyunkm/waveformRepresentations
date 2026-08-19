"""Run prediction only from a saved best-auprc checkpoint (no training).

Constructs the model identically to train_hypotension_fixed.py, then runs
trainer.predict() with the best checkpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import AUROC, AveragePrecision

from physiojepa.bedside import CLIP_INTERPOLATE_RANGES, ForecastingDataset
from physiojepa.data_preprocessing import zarr_record_name
from physiojepa.heads import AttentiveClassifier
from physiojepa.jepa import JEPASimpleLightning
from physiojepa.train import PatchTFTSingleOutcomeLightning

sys.path.insert(0, str(Path(__file__).parent))
from pipeline_common import load_config, require_output_path


class FrozenNativeJEPAProbe(PatchTFTSingleOutcomeLightning):
    def train(self, mode: bool = True):
        super().train(mode)
        if not self.fine_tune:
            self.encoder.eval()
        return self


torch.set_float32_matmul_precision("high")


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
        raise FileNotFoundError(f"Missing cache: {cache_path}")
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
        outcomes["file_path"] = outcomes["file_path"].map(
            lambda p: os.path.join(containers_override, os.path.basename(p))
        )
    return samples, outcomes


def _build_dataset(config: dict, split: str) -> ForecastingDataset:
    samples, outcomes = _load_split_inputs(config, split)
    ds_config = config["dataset"]
    return ForecastingDataset(
        channels=ds_config["channels"],
        forecast_window_sec=ds_config["forecast_window_sec"],
        outcome_df=outcomes,
        outcome_df_outcome_col=ds_config["y_outcome"],
        file_col="file_path",
        y_date_column="date",
        outcome_df_seconds_since_column="Time Stamp (seconds)",
        outcome_df_duration_column="event_length",
        sample_df=samples,
        sample_seq_len_sec=ds_config["sample_seq_len_seconds"],
        frequency=ds_config["frequency"],
        butterworth_filters=None,
        median_filter_kernel_size=None,
        clip_interpolations=CLIP_INTERPOLATE_RANGES,
        constant_nan_tolerance=ds_config["constant_nan_tolerance"],
        require_all_channels=ds_config.get("require_all_channels", True),
        infer_forecast_windows=ds_config.get("infer_forecast_windows", False),
        normalize_signals=ds_config.get("normalize_signals", True),
    )


def _build_model(config: dict, model_type: str, train_size: int):
    """Construct model with dummy optimizer params (not used for prediction)."""
    from physiojepa.patchtst import PatchTFTSimpleLightning

    training = config["training"]
    lp_config = config["lp_head"]
    encoder_ckpt = config["paths"]["pretrained_encoder_path"]

    # Load full Lightning module as encoder (PatchTFTSingleOutcomeLightning
    # expects a module with .freeze() — i.e. a LightningModule)
    if model_type == "jepa":
        encoder = JEPASimpleLightning.load_from_checkpoint(
            encoder_ckpt, map_location="cpu"
        )
        d_model = encoder.d_model
    else:
        encoder = PatchTFTSimpleLightning.load_from_checkpoint(
            encoder_ckpt, map_location="cpu"
        )
        d_model = encoder.d_model if hasattr(encoder, "d_model") else encoder.model.d_model

    c_in = len(config["dataset"]["channels"])
    head = AttentiveClassifier(
        embed_dim=d_model,
        num_heads=lp_config["num_heads"],
        mlp_ratio=lp_config["mlp_ratio"],
        depth=lp_config["depth"],
        num_classes=1,
        init_std=lp_config["init_std"],
        qkv_bias=lp_config["qkv_bias"],
        complete_block=lp_config["complete_block"],
        affine=lp_config.get("affine", False),
        c_in=c_in,
    )

    optimizer = config["optimizer"]
    scheduler = config["scheduler"]
    cls = FrozenNativeJEPAProbe if model_type == "jepa" else PatchTFTSingleOutcomeLightning

    model = cls(
        learning_rate=optimizer["learning_rate"],
        train_size=train_size,
        n_gpus=1,
        batch_size=training["batch_size"],
        linear_probing_head=head,
        preloaded_model=encoder,
        metrics={
            "auroc": AUROC(task="binary"),
            "auprc": AveragePrecision(task="binary"),
        },
        class_weights=None,
        fine_tune=training.get("fine_tune", False),
        epochs=training["epochs"],
        scheduler_type=scheduler["scheduler_type"],
        optimizer_type=optimizer["optimizer_type"],
        weight_decay=optimizer["weight_decay"],
        use_weight_decay_scheduler=optimizer.get("use_weight_decay_scheduler", False),
        final_weight_decay=optimizer.get("final_weight_decay", 0.4),
        scheduler_kwargs={
            "max_lr": scheduler["max_lr"],
            "div_factor": scheduler["div_factor"],
            "final_div_factor": scheduler["final_div_factor"],
            "pct_start": scheduler["pct_start"],
            "anneal_strategy": scheduler["anneal_strategy"],
        },
    )
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", required=True, help="Path to best checkpoint")
    parser.add_argument("--output", required=True, help="Output .pt path for predictions")
    parser.add_argument("--model-type", default="jepa", choices=["jepa", "patchtst"])
    args = parser.parse_args()

    os.environ["PHYSIOJEPA_CONFIG"] = args.config
    config, _ = load_config(args.config)

    # Build dataloaders
    loaders = {}
    for split in ("val", "test"):
        dataset = _build_dataset(config, split)
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            num_workers=int(config["training"].get("num_workers", 8)),
            persistent_workers=False,
            pin_memory=True,
            shuffle=False,
        )

    # Build model architecture, then load trained weights via ckpt_path
    model = _build_model(config, args.model_type, train_size=1)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=config["run_config"]["precision"],
        enable_progress_bar=True,
        logger=False,
    )

    predictions: dict[str, torch.Tensor] = {}
    for split in ("val", "test"):
        outputs = trainer.predict(
            model=model,
            dataloaders=loaders[split],
            ckpt_path=args.checkpoint,
            return_predictions=True,
        )
        split_preds, split_targets = zip(*outputs)
        predictions[f"{split}_preds"] = torch.cat(split_preds).cpu()
        predictions[f"{split}_targets"] = torch.cat(split_targets).cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".partial")
    torch.save(predictions, tmp)
    os.replace(tmp, output_path)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
