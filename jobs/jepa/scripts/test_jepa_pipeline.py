"""Focused regression tests for the cluster-ready native JEPA path."""

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from physiojepa.jepa import (
    JEPASimpleLightning,
    PositionAwareMultiHeadAttention,
    apply_masks,
    create_masks,
    loss_pred,
    representation_channel_stats,
)

from pipeline_common import (
    build_pretraining_split,
    data_fingerprint,
    load_pretraining_split,
)
from train_hypotension_fixed import FrozenNativeJEPAProbe


HERE = Path(__file__).resolve().parent


def test_apply_masks_preserves_every_effective_batch() -> None:
    values = torch.arange(6 * 5 * 2).reshape(6, 5, 2)
    masks = torch.tensor(
        [
            [0, 2, 4],
            [1, 2, 3],
            [0, 1, 4],
            [2, 3, 4],
            [0, 3, 4],
            [1, 3, 4],
        ]
    )
    masked = apply_masks(values, masks)
    assert masked.shape == (6, 3, 2)
    for batch_index in range(values.shape[0]):
        assert torch.equal(masked[batch_index], values[batch_index, masks[batch_index]])


def test_representation_diagnostics_are_bounded_and_finite() -> None:
    values = torch.randn(8, 3, 200, 16)
    stats = representation_channel_stats(values, max_vectors=32)
    assert len(stats) == 3
    for mean, std, cosine in stats:
        assert all(torch.isfinite(torch.tensor([mean, std, cosine])))
        assert -1.0 <= cosine <= 1.0


def test_mask_ratios_do_not_collapse_at_production_batch_size() -> None:
    torch.manual_seed(12)
    values = torch.empty(128, 3, 1800)
    observed = []
    for _ in range(12):
        targets, contexts = create_masks(
            values, 10, 10, (0.1, 0.4), (0.1, 0.3), True
        )
        target_ratio = targets.shape[1] / 180
        context_ratio = contexts.shape[1] / (180 - targets.shape[1])
        assert 0.1 - 1 / 180 <= target_ratio <= 0.3
        assert 0.1 - 1 / 180 <= context_ratio <= 0.4
        observed.append((targets.shape[1], contexts.shape[1]))
    assert len(set(observed)) > 1


def test_rotary_attention_preserves_positions_when_tokens_are_permuted() -> None:
    torch.manual_seed(12)
    attention = PositionAwareMultiHeadAttention(16, 4, rotary_pes=True).eval()
    values = torch.randn(2, 31, 16)
    positions = torch.arange(31).expand(2, -1)
    permutation = torch.randperm(31)
    inverse = torch.argsort(permutation)
    expected = attention(values, positions=positions)
    actual = attention(
        values[:, permutation], positions=positions[:, permutation]
    )[:, inverse]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_unshared_three_channel_training_step_keeps_batch_shape() -> None:
    encoder = {
        "c_in": 3,
        "num_patches": 8,
        "patch_size": 4,
        "patch_stride": 4,
        "d_model": 8,
        "nhead": 2,
        "use_tst_block": True,
        "shared_embedding": False,
        "num_layers": 1,
        "pe_type": "rotary",
        "mlp_ratio": 2.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "norm_layer": nn.LayerNorm,
        "jepa": True,
        "tokenizer_type": "linear",
        "tokenizer_kwargs": {},
        "embed_activation": nn.GELU(),
    }
    predictor = {
        "num_patches": 8,
        "encoder_embed_dim": 8,
        "predictor_embed_dim": 4,
        "nhead": 1,
        "num_layers": 1,
        "pe_type": "rotary",
        "mlp_ratio": 2.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "norm_layer": nn.LayerNorm,
        "use_tst_block": True,
        "c_in_mask_tokens": 3,
        "embed_activation": nn.GELU(),
    }
    model = JEPASimpleLightning(
        learning_rate=1e-4,
        train_size=4,
        batch_size=2,
        n_gpus=1,
        patchtsjepa_encoder_kwargs=encoder,
        patchtsjepa_predictor_kwargs=predictor,
        epochs=1,
        target_mask_range=(0.25, 0.5),
        context_mask_range=(0.25, 0.5),
        loss_fn=loss_pred,
        scheduler_kwargs={
            "max_lr": 1e-4,
            "div_factor": 10,
            "final_div_factor": 10,
            "pct_start": 0.3,
            "anneal_strategy": "cos",
        },
    )
    loss = model.training_step((torch.randn(2, 3, 32), torch.zeros(2)), 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    model.train()
    assert model.encoder.training
    assert not model.target_encoder.training
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())


class _DummyFrozenEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def freeze(self):
        self.requires_grad_(False)
        self.eval()

    def forward(self, values):
        return self.dropout(values)


def test_probe_keeps_frozen_encoder_in_evaluation_mode() -> None:
    model = FrozenNativeJEPAProbe(
        learning_rate=1e-4,
        train_size=4,
        n_gpus=1,
        batch_size=2,
        linear_probing_head=nn.Identity(),
        preloaded_model=_DummyFrozenEncoder(),
        metrics={},
        fine_tune=False,
        epochs=1,
        scheduler_type="OneCycle",
        optimizer_type="AdamW",
        scheduler_kwargs={
            "max_lr": 1e-4,
            "div_factor": 10,
            "final_div_factor": 10,
            "pct_start": 0.3,
            "anneal_strategy": "cos",
        },
    )
    model.train()
    assert model.training
    assert not model.encoder.training
    assert not model.encoder.dropout.training


def test_data_fingerprint_ignores_training_but_tracks_dataset() -> None:
    with (HERE / "train_patch_jepa.yaml").open() as handle:
        config = yaml.safe_load(handle)
    changed_training = __import__("copy").deepcopy(config)
    changed_training["training"]["epochs"] += 1
    assert data_fingerprint(changed_training) == data_fingerprint(config)
    changed_dataset = __import__("copy").deepcopy(config)
    changed_dataset["dataset"]["sample_stride_seconds"] += 1
    assert data_fingerprint(changed_dataset) != data_fingerprint(config)


def test_pretraining_metadata_rejects_a_stale_data_configuration() -> None:
    with (HERE / "train_patch_jepa.yaml").open() as handle:
        config = yaml.safe_load(handle)
    assert not load_pretraining_split(config).empty
    stale = __import__("copy").deepcopy(config)
    stale["dataset"]["sample_stride_seconds"] += 1
    try:
        load_pretraining_split(stale)
    except ValueError as error:
        assert "fingerprint" in str(error)
    else:
        raise AssertionError("Stale JEPA sample metadata was accepted")


def test_paper_architecture_and_smoke_paths_are_aligned() -> None:
    with (HERE / "train_patch_jepa.yaml").open() as handle:
        pretrain = yaml.safe_load(handle)
    with (HERE / "train_hypotension_fixed.yaml").open() as handle:
        probe = yaml.safe_load(handle)
    with (HERE / "train_hypotension_fixed_smoke.yaml").open() as handle:
        smoke = yaml.safe_load(handle)
    assert pretrain["predictor"]["predictor_embed_dim"] == 256
    assert probe["lp_head"]["num_heads"] == smoke["lp_head"]["num_heads"] == 4
    assert probe["dataset"]["use_transforms"] == smoke["dataset"]["use_transforms"]
    assert probe["training"]["mixup"] == smoke["training"]["mixup"]
    assert probe["training"]["use_class_weights"] is False
    assert smoke["evaluation"]["run_predictions"] is True


def test_full_config_builds_a_leakage_safe_subject_split() -> None:
    with (HERE / "train_patch_jepa.yaml").open() as handle:
        config = yaml.safe_load(handle)
    frame = build_pretraining_split(config)
    downstream = __import__("pandas").read_csv(
        config["paths"]["downstream_subject_split_path"]
    )
    excluded = set(
        downstream.loc[downstream["split"].isin(["val", "test"]), "subject_id"].astype(str)
    )
    assert not set(frame["subject_id"]).intersection(excluded)
    train_subjects = set(
        frame.loc[frame["pretrain_split"] == "train", "subject_id"].astype(str)
    )
    validation_subjects = set(
        frame.loc[frame["pretrain_split"] == "val", "subject_id"].astype(str)
    )
    assert train_subjects
    assert validation_subjects
    assert not train_subjects.intersection(validation_subjects)
