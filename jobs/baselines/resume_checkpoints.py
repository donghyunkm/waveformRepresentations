"""Utilities for fault-tolerant supervised baseline training."""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint


RESUME_METADATA_KEY = "physiojepa_resume"


def training_config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable fingerprint for the complete experiment configuration."""
    serialized = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class ResumeMetadataCallback(Callback):
    """Embed experiment identity in every Lightning checkpoint."""

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
    """Build metadata and rolling two-generation resume callbacks."""
    if checkpoint_interval_minutes <= 0:
        raise ValueError("checkpoint_interval_minutes must be positive")

    metadata_callback = ResumeMetadataCallback(
        config_fingerprint=config_fingerprint,
        run_subdir=run_subdir,
    )
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


def _candidate_paths(
    checkpoint_dir: Path,
    initial_checkpoint_path: Path | None,
) -> list[Path]:
    candidates = set(checkpoint_dir.glob("resume-*.ckpt"))
    epoch_last = checkpoint_dir / "epoch-last.ckpt"
    if epoch_last.exists():
        candidates.add(epoch_last)
    if initial_checkpoint_path is not None and initial_checkpoint_path.exists():
        candidates.add(initial_checkpoint_path)
    return list(candidates)


def find_resume_checkpoint(
    checkpoint_dir: Path,
    expected_config_fingerprint: str,
    initial_checkpoint_path: Path | None = None,
    allow_unverified_initial_checkpoint: bool = False,
) -> Path | None:
    """Return the highest-step compatible, readable full-state checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    initial_checkpoint_path = (
        Path(initial_checkpoint_path) if initial_checkpoint_path is not None else None
    )
    initial_resolved = (
        initial_checkpoint_path.resolve()
        if initial_checkpoint_path is not None and initial_checkpoint_path.exists()
        else None
    )

    valid_candidates: list[tuple[int, float, Path]] = []
    rejected_candidates: list[str] = []
    for path in _candidate_paths(checkpoint_dir, initial_checkpoint_path):
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            modified_time = path.stat().st_mtime
        except Exception as exc:
            rejected_candidates.append(f"{path}: unreadable ({exc})")
            continue

        required_keys = {
            "state_dict",
            "optimizer_states",
            "lr_schedulers",
            "epoch",
            "global_step",
        }
        missing_keys = sorted(required_keys.difference(checkpoint))
        if missing_keys:
            rejected_candidates.append(
                f"{path}: missing training state {missing_keys}"
            )
            continue
        if not checkpoint["optimizer_states"] or not checkpoint["lr_schedulers"]:
            rejected_candidates.append(
                f"{path}: optimizer or scheduler state is empty"
            )
            continue

        metadata = checkpoint.get(RESUME_METADATA_KEY)
        if metadata is None:
            is_allowed_initial = (
                allow_unverified_initial_checkpoint
                and initial_resolved is not None
                and path.resolve() == initial_resolved
            )
            if not is_allowed_initial:
                rejected_candidates.append(
                    f"{path}: missing {RESUME_METADATA_KEY!r} metadata"
                )
                continue
            print(f"WARNING: accepting explicitly configured legacy checkpoint {path}")
        elif metadata.get("config_fingerprint") != expected_config_fingerprint:
            rejected_candidates.append(f"{path}: experiment fingerprint mismatch")
            continue

        valid_candidates.append(
            (int(checkpoint["global_step"]), modified_time, path)
        )

    if valid_candidates:
        _, _, selected = max(valid_candidates)
        return selected
    if rejected_candidates:
        details = "\n  ".join(rejected_candidates)
        raise RuntimeError(
            "Resume checkpoints exist, but none are safe to load:\n  " + details
        )
    return None
