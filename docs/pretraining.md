# Self-Supervised Pretraining

## Overview

Two self-supervised models were pretrained on MIMIC-III ICU waveforms:

- **Native PhysioJEPA**: JEPA-style masked latent prediction with EMA target encoder
- **Self-supervised PatchTST**: Masked patch reconstruction baseline

Both share the same data pipeline, tokenization, and encoder architecture.

---

## Data

- **Source**: 30-minute, non-overlapping windows at 125 Hz
- **Channels**: ABP, ECG lead II, PLETH (3 channels)
- **Preprocessing**: Exclude signals with ≥20% constant/null values, interpolate
  nulls, IQR-normalize per window, divide into patches
- **Patch size**: 125 samples = 1 second (non-overlapping)
- **Sequence length**: 1,800 patches per channel (30 min × 60 s × 125 Hz / 125)
- **Training samples**: ~272,068 (from ~3,035 subjects, 4,689 stays)
- **Validation samples**: ~14,540

### Pretraining Split

Self-supervised pretraining excludes the downstream val and test subjects
(511 subjects total) before building its own 95/5 train/val split. This means
the encoder never sees the subjects used for downstream evaluation during
pretraining.

---

## Architecture

Both models share a common encoder specification:

| Component | Specification |
|-----------|--------------|
| Encoder layers | 3 |
| Attention heads | 8 |
| Model width (d_model) | 512 |
| Feed-forward width | 2048 |
| Positional encoding | RoPE (Rotary Position Embeddings) |
| Activation | GELU |
| Embedding output | 512-dim per patch token |

### JEPA-Specific Components

| Component | Specification |
|-----------|--------------|
| Predictor layers | 2 |
| Predictor width | 256 |
| Predictor heads | 4 |
| Target encoder | EMA copy of context encoder |
| EMA decay | Fixed (not warmed up) |
| Target masking | 10–30% of patches |
| Context masking | 10–40% of patches |

### PatchTST-Specific Components

| Component | Specification |
|-----------|--------------|
| Reconstruction head | Per-channel linear projection |
| Mask ratio | 10–30% (YAML); original code used 10–40% |
| Reconstruction target | Full input waveform (not just masked patches) |

---

## JEPA Pretraining

### Training Configuration

- Optimizer: AdamW
- Scheduler: OneCycleLR
- Max epochs: 100 (planned)
- Batch size: configurable per hardware

### Results

- **Completed**: 35 epochs (hit 3-day wall time on A100)
- **Best checkpoint**: epoch 13, val_loss = 0.215
- **Training loss at best**: 0.246
- **Divergence**: Both train and val loss rose monotonically after epoch 13–14

### Divergence Analysis

The loss divergence is **not overfitting** (the train-val gap stays small, ~0.03).
It is caused by the **OneCycle LR scheduler conflicting with the EMA target
encoder dynamics**:

- OneCycle raises the learning rate aggressively in the first half, then decays
- The EMA target encoder uses a fixed decay rate
- As the online encoder's learning rate spikes, the target can't track fast
  enough, causing the prediction targets to become inconsistent
- Result: loss rises for both train and val simultaneously

**Potential fix** (not implemented): Replace OneCycle with cosine annealing,
and add EMA decay warmup (start with faster EMA, slow down as training
stabilizes).

### Checkpoints

```
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/jepa_native_paper/
  2026-08-04-native-jepa-paper-1gpu-debug-v1/
    best-val-epoch=13-loss=0.21508.ckpt
```

### Config and Scripts

```
jobs/jepa/configs/train_patch_jepa.yaml
jobs/jepa/scripts/train_patch_jepa.py
```

---

## JEPA d64 Pretraining (Compression Experiment)

A reduced-capacity version of the native JEPA, testing whether a narrower
encoder retains sufficient representational power.

### Architecture Differences from Full JEPA

| Component | Full JEPA | JEPA d64 |
|-----------|-----------|----------|
| d_model | 512 | **64** |
| Attention heads | 8 | **4** |
| Encoder layers | 3 | 3 |
| Predictor width | 256 | **32** |
| Predictor heads | 4 | **2** |
| Tokenizer bottleneck | 112 | 112 |

### Training

- **Job**: 26381948, `a100_short` (a100-4027)
- **Started**: 2026-08-13 10:21
- **Completed**: 2026-08-15 22:27 (2d 12h 05m)
- **Epochs**: 100/100 (full run)
- **Batch size**: 256
- **Optimizer**: AdamW, OneCycleLR (max_lr=0.0006)
- **Best val checkpoint**: epoch 40, val_loss = **0.18978**
- **Final checkpoint**: epoch 99, step 105299

Unlike the full JEPA (which diverged after epoch 13), the d64 model trained
stably through all 100 epochs — the lower learning rate (0.0006 vs 0.001) and
smaller model may have avoided the OneCycle/EMA conflict.

### Checkpoints

```
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/jepa_native_d64/
  2026-08-12-native-jepa-d64-v1/
    best-val-epoch=40-loss=0.18978.ckpt    ← best validation
    last.ckpt                               ← epoch 99
    resume-last.ckpt → resume-epoch=99-step=105299.ckpt
```

### Config

```
jobs/jepa/configs/train_patch_jepa_d64.yaml
jobs/jepa/slurm/train_patch_jepa_d64_a100_short.sbatch
```

---

## PatchTST Pretraining

### Training Configuration

- Optimizer: AdamW
- Scheduler: OneCycleLR
- Max epochs: 100 (planned)
- Mask ratio: [0.1, 0.3]

### Results

- **Completed**: 53 epochs (hit 3-day wall time)
- **Best checkpoint**: epoch 3, val_loss = 0.003
- **Overfitting**: Val MSE bottomed at 0.0033 (epoch 4–7), rose to 0.023 by
  epoch 33 (a 7× increase). Train loss plateaued at ~0.067.

### Paper vs Repository Discrepancy

The paper (Appendix A.1) states that PatchTST:
- Masks patches before tokenization (10–30% target masking)
- Reconstructs the masked patches with a per-channel linear head
- Computes MSE **on the masked patches only**

The released repository implementation does something different:

1. The input is copied to an unmasked `Y_true` target
2. Patches are masked only when `self.training` is True
3. The model reconstructs the **complete** target tensor
4. Training loss = MSE over **all patches** (not just masked positions)
5. Validation runs in eval mode (no masking), measuring full-signal reconstruction

This is effectively a denoising-autoencoder objective: visible and masked patches
both contribute to the gradient. At a 10–30% mask ratio, roughly 70–90% of the
reconstruction terms are visible patches. This means:

- The model can minimize training loss by simply copying visible patches through
- The validation loss (no masking) becomes trivially low early
- True generalization of masked prediction is never properly measured
- Fast overfitting is expected (and observed)

**The current cluster training path uses the released repository's loss behavior**
(faithful to GitHub code, not to the paper's stated masked-only loss).

Matching the repository and matching the paper are therefore different targets.

### Checkpoints

```
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/patchtst_pretraining/
└── best-val-epoch=03-loss=0.00329.ckpt
```

### Config and Scripts

```
jobs/patchtst/configs/train_patchtst.yaml
jobs/patchtst/train_patchtst.py
```

---

## Comparison

| Aspect | JEPA | PatchTST |
|--------|------|----------|
| Objective | Predict masked latent targets | Reconstruct all patches |
| Target encoder | EMA copy (momentum) | N/A |
| Best epoch | 13 (of 35 completed) | 3 (of 53 completed) |
| Best val loss | 0.215 | 0.003 |
| Failure mode | Loss divergence (OneCycle+EMA conflict) | Overfitting (full-patch loss) |
| Total params | ~9.6M encoder + ~1.8M predictor | ~9.6M encoder + linear head |
| Downstream probe AUROC | 0.844 (attentive, epoch 7) | 0.850 (attentive, epoch 3) |

Both models learn nearly identical relational structure per patient (CKA = 0.81),
suggesting the dominant structure comes from the waveform data itself rather than
the specific training objective.

---

## Paper alignment

This section documents known discrepancies between the current cluster
implementation and the published PhysioJEPA paper (Fox et al., ML4H 2024).

### Data inventory

The current sample-index cache contains approximately **286,608 samples** from
**4,689 stays** and **3,035 subjects**. The paper reports approximately
**356,903 segments** from **4,282 stays** and **2,631 patients**. The data
inventories are not identical — the current pipeline discovers more stays and
subjects but produces fewer qualifying 30-minute segments, likely due to
differences in signal-quality thresholds, minimum-duration requirements, or
manifest construction.

### Signal quality boundary

The paper states that windows with 20% or more constant/null values should be
**excluded**. The current code accepts windows with **exactly** 20% constant
samples (using a `<=` boundary rather than `<`). This is a minor off-by-one
difference that could admit borderline-quality windows the paper intended to
reject.

### Mask ratio

The current cluster YAML uses `mask_ratio: [0.1, 0.3]` for both JEPA target
masking and PatchTST masking. The original upstream YAML used `[0.1, 0.4]`.
The paper text specifies 10–30% target masking, so the current value is
consistent with the paper text but narrower than the original code.

### Pretraining split

The paper describes a 95/5 train/val split for self-supervised pretraining.
The current pipeline is **stricter**: downstream val and test subjects (511
subjects total) are excluded from the pretraining pool before the 95/5 split
is applied. This guarantees that downstream evaluation subjects are never seen
during pretraining — a stronger leakage prevention than the paper describes,
and a protocol difference rather than an accidental overlap.

---

## Status

| Model | Status | Notes |
|-------|--------|-------|
| JEPA pretraining | ✅ Completed | 35 epochs, best at epoch 13 |
| PatchTST pretraining | ✅ Completed | 53 epochs, best at epoch 3 |
| Physio-Contrastive JEPA | 🔄 In progress | See [contrastive_learning.md](contrastive_learning.md) |

### Potential Improvements (not implemented)

- **JEPA**: Cosine LR annealing + EMA warmup to prevent divergence
- **PatchTST**: Compute loss on masked patches only (as paper describes) to
  prevent overfitting and properly evaluate masked prediction capability
- Both: longer training on the improved objectives once scheduler/loss issues
  are resolved
