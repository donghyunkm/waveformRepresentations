"""Fault-tolerant checkpoint helpers for native PhysioJEPA jobs."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


RESUME_METADATA_KEY = "physiojepa_jepa_resume"


class ResumeMetadataCallback(Callback):
    def __init__(self, config_fingerprint: str, run_subdir: str):
        super().__init__()
        self.config_fingerprint = config_fingerprint
        self.run_subdir = run_subdir

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        checkpoint[RESUME_METADATA_KEY] = {
            "config_fingerprint": self.config_fingerprint,
            "run_subdir": self.run_subdir,
        }


def build_resume_callbacks(
    checkpoint_dir: Path,
    config_fingerprint: str,
    run_subdir: str,
    checkpoint_interval_minutes: float,
) -> tuple[ResumeMetadataCallback, ModelCheckpoint]:
    if checkpoint_interval_minutes <= 0:
        raise ValueError("checkpoint_interval_minutes must be positive")
    metadata_callback = ResumeMetadataCallback(config_fingerprint, run_subdir)
    rolling_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="resume-epoch={epoch:02d}-step={step}",
        monitor="step",
        mode="max",
        save_top_k=2,
        save_last="link",
        save_weights_only=False,
        train_time_interval=timedelta(minutes=checkpoint_interval_minutes),
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )
    rolling_callback.CHECKPOINT_NAME_LAST = "resume-last"
    return metadata_callback, rolling_callback


def find_resume_checkpoint(
    checkpoint_dir: Path,
    expected_config_fingerprint: str,
) -> Path | None:
    candidates = set(checkpoint_dir.glob("resume-*.ckpt"))
    candidates.update(checkpoint_dir.glob("epoch-last*.ckpt"))
    # Lightning's Slurm signal handler writes the emergency checkpoint under
    # this name immediately before requeueing near the wall-time limit.
    candidates.update(checkpoint_dir.glob("hpc_ckpt_*.ckpt"))
    valid: list[tuple[int, float, Path]] = []
    rejected: list[str] = []
    for path in candidates:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            rejected.append(f"{path}: unreadable ({exc})")
            continue
        required = {
            "state_dict",
            "optimizer_states",
            "lr_schedulers",
            "epoch",
            "global_step",
        }
        missing = required.difference(checkpoint)
        metadata = checkpoint.get(RESUME_METADATA_KEY)
        if missing:
            rejected.append(f"{path}: missing {sorted(missing)}")
        elif not checkpoint["optimizer_states"] or not checkpoint["lr_schedulers"]:
            rejected.append(f"{path}: optimizer or scheduler state is empty")
        elif metadata is None:
            rejected.append(f"{path}: missing resume metadata")
        elif metadata.get("config_fingerprint") != expected_config_fingerprint:
            rejected.append(f"{path}: configuration fingerprint mismatch")
        else:
            valid.append(
                (int(checkpoint["global_step"]), path.stat().st_mtime, path)
            )
    if valid:
        return max(valid)[2]
    if rejected:
        raise RuntimeError(
            "Resume checkpoints exist, but none are safe to load:\n  "
            + "\n  ".join(rejected)
        )
    return None
