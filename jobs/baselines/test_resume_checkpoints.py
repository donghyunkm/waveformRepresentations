"""Focused regression tests for supervised baseline checkpoint resumption."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from resume_checkpoints import (
    RESUME_METADATA_KEY,
    build_resume_callbacks,
    find_resume_checkpoint,
    training_config_fingerprint,
)


def write_checkpoint(
    path: Path,
    *,
    fingerprint: str | None,
    global_step: int,
) -> None:
    checkpoint = {
        "state_dict": {"weight": torch.tensor([1.0])},
        "optimizer_states": [{"state": {}, "param_groups": []}],
        "lr_schedulers": [{"last_epoch": global_step}],
        "epoch": global_step // 10,
        "global_step": global_step,
    }
    if fingerprint is not None:
        checkpoint[RESUME_METADATA_KEY] = {
            "config_fingerprint": fingerprint,
            "run_subdir": "stable-run",
        }
    torch.save(checkpoint, path)


class ResumeCheckpointTests(unittest.TestCase):
    def test_selects_highest_step_compatible_checkpoint(self):
        fingerprint = training_config_fingerprint({"model": "fcn", "seed": 16})
        with TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            write_checkpoint(
                checkpoint_dir / "epoch-last.ckpt",
                fingerprint=fingerprint,
                global_step=20,
            )
            write_checkpoint(
                checkpoint_dir / "resume-epoch=02-step=35.ckpt",
                fingerprint=fingerprint,
                global_step=35,
            )
            write_checkpoint(
                checkpoint_dir / "resume-epoch=09-step=90.ckpt",
                fingerprint="different-run",
                global_step=90,
            )

            selected = find_resume_checkpoint(
                checkpoint_dir,
                expected_config_fingerprint=fingerprint,
            )

            self.assertEqual(
                selected.name,
                "resume-epoch=02-step=35.ckpt",
            )

    def test_rejects_unverified_checkpoint_unless_explicitly_allowed(self):
        fingerprint = training_config_fingerprint({"model": "fcn"})
        with TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            legacy = checkpoint_dir / "last.ckpt"
            write_checkpoint(legacy, fingerprint=None, global_step=10)

            selected = find_resume_checkpoint(
                checkpoint_dir,
                expected_config_fingerprint=fingerprint,
                initial_checkpoint_path=legacy,
                allow_unverified_initial_checkpoint=True,
            )

            self.assertEqual(selected, legacy)

    def test_rolling_callback_has_distinct_two_generation_last_name(self):
        fingerprint = training_config_fingerprint({"model": "patchtst"})
        with TemporaryDirectory() as temp_dir:
            _, callback = build_resume_callbacks(
                checkpoint_dir=Path(temp_dir),
                config_fingerprint=fingerprint,
                run_subdir="stable-run",
                checkpoint_interval_minutes=30,
            )

            self.assertEqual(callback.CHECKPOINT_NAME_LAST, "resume-last")
            self.assertEqual(callback.save_top_k, 2)
            self.assertEqual(callback.monitor, "step")


if __name__ == "__main__":
    unittest.main()
