# Contrastive Learning: Physio-Contrastive JEPA

## Overview

The Physio-Contrastive JEPA adds a cross-patient contrastive loss to the vanilla
JEPA masked-prediction objective. This directly addresses the core finding from
representation analysis: embeddings are patient fingerprints that do not capture
shared physiological features generalizing across patients.

The contrastive loss pulls together windows from *different patients* that share
similar physiological state (HR and MAP within epsilon), while pushing apart
windows with dissimilar physiological state (HR and MAP beyond delta).

---

## Motivation

The representation analysis (see [representation_analysis.md](representation_analysis.md))
showed:
- Cross-patient k-NN AUROC drops from 0.91 → 0.57 when same-patient neighbors excluded
- Patient centroid hemodynamic ratio = 1.04–1.07 (anti-correlated)
- The masked-prediction pretraining objective naturally encourages learning
  patient-specific waveform morphology rather than shared physiological states

The Physio-Contrastive JEPA aims to break this patient fingerprinting by
explicitly optimizing for cross-patient physiological similarity.

---

## Design

### Architecture

Identical to `JEPASimpleLightning` (same Encoder, Predictor, EMA target encoder,
masking) plus:

- **Projection head**: 2-layer MLP (512→512→128) with BatchNorm + ReLU, applied
  to mean-pooled encoder tokens. Adds ~329K parameters (+1.5%).
- **Physiological distance metric**:
  `d_physio(i,j) = sqrt((ΔHR/σ_HR)² + (ΔMAP/σ_MAP)²)`
  where σ_HR=15 bpm, σ_MAP=12 mmHg (normalizing constants)

### Pair Selection

- **Positive pairs**: windows from *different patients* within epsilon
  physiological distance (similar HR + MAP)
- **Negative pairs**: windows from *different patients* beyond delta
  physiological distance (dissimilar HR + MAP)
- **Excluded**: same-patient pairs (excluded from denominator entirely), and
  margin pairs (epsilon ≤ d ≤ delta)

No discrete bins — continuous distance with configurable thresholds.

### Loss Function

```
L_total = L_JEPA + lambda_contrast * L_contrastive
```

The contrastive component is InfoNCE over the physiological-distance-selected
pairs.

### Gradient Flow and Encoder Roles

#### Data Flow Diagram

```
                         Input X (B, C, T)
                              │
                    ┌─────────┴─────────┐
                    │   Random Masking   │
                    └─────────┬─────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
     Context patches                   Full input X
     (non_masks, ~60-90%)                     │
              │                               │
     ┌────────┴────────┐            ┌────────┴────────┐
     │  Online Encoder │            │ EMA Target Enc. │
     │   (gradients)   │            │   (no grad)     │
     └────────┬────────┘            └────────┬────────┘
              │                               │
       context tokens                    Layer Norm
              │                               │
     ┌────────┴────────────┐          Select target
     │                     │           positions
     │                     │              │
     │              ┌──────┴──────┐   target tokens
     │              │  Mean-pool  │       │
     │              │  (patches)  │       │
     │              └──────┬──────┘       │
     │                     │              │
     │              ┌──────┴──────┐       │
     │              │ Mean over   │       │
     │              │  channels   │       │
     │              └──────┬──────┘       │
     │                     │              │
     │              ┌──────┴──────┐       │
     │              │ Projection  │       │
     │              │  MLP        │       │
     │              └──────┬──────┘       │
     │                     │              │
     │              ┌──────┴──────┐       │
     │              │ L2-normalize│       │
     │              └──────┬──────┘       │
     │                     │              │
┌────┴─────┐               │              │
│Predictor │          L_contrastive       │
│(+ mask   │          (InfoNCE)           │
│positions)│               │              │
└────┬─────┘               │              │
     │                     │              │
predicted targets          │              │
     │                     │              │
     └──────────┐          │              │
                ▼          │              │
         MSE(pred, ────────┼──────────────┘
          targets)         │
            │              │
        L_JEPA             │
            │              │
            └──────┬───────┘
                   │
      L_total = L_JEPA + λ * L_contrastive
```

#### Key Design Choice: Contrastive Loss on Context Encoder, Not Target Encoder

- The contrastive loss uses the **online encoder's** output on CONTEXT (visible)
  patches only — not the full sequence, not the target encoder output.
- This means the contrastive representation is built from a random ~60–90% subset
  of patches (the complement of `target_mask_range`).
- **Rationale**: We want gradients to flow through the online encoder to shape its
  representations. The EMA target encoder is frozen (no gradients) and provides
  stable regression targets only.
- **Alternative considered**: Computing contrastive loss on EMA target encoder
  output (full sequence) — rejected because no gradients would flow, making the
  contrastive objective invisible to parameter updates.

#### Why Context Patches Rather Than Full-Sequence Encoding

- Computing contrastive loss requires backpropagation through the encoder.
- The encoder is already called once with the context mask for JEPA — reusing
  those tokens avoids a second expensive forward pass through the full sequence.
- The masking introduces stochasticity: each sample's contrastive representation
  is based on a different random subset of its patches.
- This acts as **implicit data augmentation** — the same window will produce
  slightly different contrastive embeddings across epochs due to different masks,
  preventing the projection head from memorizing fixed representations.

#### Gradient Flow Summary

- **Online Encoder** receives gradients from BOTH `L_JEPA` (via predictor) and
  `L_contrastive` (via projection head)
- **Predictor** receives gradients from `L_JEPA` only
- **Projection Head** receives gradients from `L_contrastive` only
- **EMA Target Encoder** receives NO gradients — updated via momentum (EMA)
- The projection head is discarded at evaluation time; only the encoder is used
  downstream

---

## HR and MAP Computation: 3,400× Speedup

### Problem

Computing HR from waveforms via R-peak detection was prohibitively slow
(22+ hours for the full dataset). The original training attempt spent 2h50m
reaching only 11% of physio-value computation.

### Solution

HR and MAP are read directly from MIMIC **bedside numerics records** (1 Hz)
instead of computing from raw waveforms via R-peak detection.

- Script: `jobs/jepa/scripts/precompute_physio_values.py`
- Groups by stay, parallelizes with multiprocessing
- Runtime: **23 seconds** vs the original 22+ hours (3,400× speedup)
- Validated: r=0.976 vs waveform-derived MAP, HR from bedside monitor avoids
  T-wave double-counting bugs

### Coverage

- Training: 191,723/272,068 windows have valid HR+MAP (70.5%)
- Validation: 8,412/14,540 windows have valid HR+MAP (57.9%)
- ~30% of windows have NaN HR/MAP (no bedside numerics for that time period)

---

## Implementation

### Model Class

`PhysioContrastiveJEPALightning` in `physiojepa/jepa.py`

Key functions:
- `physio_distance_contrastive()` — computes InfoNCE loss over physiological
  distance pairs
- `SelfSupervisedDatasetWithPhysioValues` — dataset wrapper that returns
  `(X, physio_values)` tuples from precomputed cache

### Training Script

`jobs/jepa/scripts/train_physio_contrastive_jepa.py`

- Loads precomputed physio values cache at startup (instant)
- Constructs `SelfSupervisedDatasetWithPhysioValues` from existing sample indices
- Standard Lightning training with checkpoint resume

### Precomputation Script

`jobs/jepa/scripts/precompute_physio_values.py`

- Reads HR from bedside monitor numerics records (MIMIC wfdb format)
- Reads MAP from bedside ABP Mean records
- Matches to PhysioJEPA sample indices by timestamp
- Saves as `.npz` files for instant loading during training

---

## Configuration

`jobs/jepa/configs/train_physio_contrastive_jepa.yaml`

```yaml
contrastive:
  epsilon: 0.5         # positive pair threshold (normalized distance)
  delta: 1.5           # negative pair threshold
  lambda_contrast: 0.1 # loss weight
  tau: 0.1             # InfoNCE temperature
  sigma_hr: 15.0       # HR normalization constant (bpm)
  sigma_map: 12.0      # MAP normalization constant (mmHg)
  projection_dim: 128
  projection_hidden_dim: 512

training:
  batch_size: 64
  accumulate_grad_batches: 2  # effective batch = 128
  max_epochs: 100
  precision: 16-mixed
```

### Logged Metrics (every step)

- `train_jepa_loss`, `train_contrastive_loss`, `train_loss`
- `train_mean_pos_per_anchor` — average number of positive pairs per anchor
- `train_frac_anchors_with_pos` — fraction of anchors that have ≥1 positive
- `train_mean_neg_per_anchor`
- `train_frac_same_patient`, `train_frac_margin`

---

## Known Caveats

1. **Contrastive loss computed per micro-batch (64), not full effective batch (128)**:
   With `accumulate_grad_batches=2`, each forward pass only sees 64 windows for
   constructing positive/negative pairs. This means fewer cross-patient positives
   per step than a true bs=128. Monitor `train_frac_anchors_with_pos`.

2. **30% of windows have NaN HR/MAP** (no bedside numerics for that time period):
   The contrastive loss skips these, so effective contrastive batch may be <64.
   The JEPA prediction loss still trains on all 64.

3. **OneCycle scheduler + EMA**: The vanilla JEPA diverged after epoch 14 from
   this combination. The contrastive version uses the same scheduler — monitor
   for similar divergence.

4. **L40S memory constraint**: Reduced from batch_size=128 to batch_size=64 with
   accumulate_grad_batches=2 to fit on 48GB L40S (the pairwise contrastive
   distance matrix + 3-channel JEPA activations exceeded memory at bs=128).

---

## Training Status

🔄 **In progress.** Resumed from epoch 12 (step 26828) on 2026-08-17, job
26497397 on `gl40s_short` (3-day wall-time). Training 100 epochs total.

- Last checkpoint: `resume-epoch=12-step=26828.ckpt` (2026-08-12)
- Running on L40S GPUs via `gl40s_short` partition
- Monitor: `train_frac_anchors_with_pos` should be >0.3 for meaningful
  contrastive signal

### Evaluation Plan (after pretraining completes)

1. Extract embeddings from the best physio-contrastive checkpoint
2. Re-run kNN/distance/probing analysis
3. Target metrics:
   - Cross-patient kNN AUROC > 0.65 (currently 0.57 for vanilla JEPA)
   - Same-patient ratio < 0.60 (currently 0.79)
   - Within-patient hypotension ratio ≤ 0.80 (preserve)
4. Train downstream attentive probe on frozen contrastive encoder
5. Compare to vanilla JEPA probe (AUROC 0.844)

---

## Design Evolution

### Predecessor: Cross-Patient InfoNCE (deprecated)

The first contrastive implementation (`ContrastiveJEPALightning`) used a simpler
objective: all samples from different patients = positives, same-patient =
negatives. This was replaced because:

- It penalizes patient identity encoding indiscriminately
- Two cross-patient windows could be hemodynamically very different (one
  tachycardic, one bradycardic) — treating them as positives is incorrect
- No physiological grounding for similarity

The physio-distance variant addresses all three issues by grounding similarity
in measured HR and MAP.

### Old files (deleted)

- `train_contrastive_jepa.py`, `train_physio_state_contrastive_jepa.py`
- Old YAML configs and sbatch files
- Old checkpoint directory (`models/contrastive_jepa/`)

---

## Relevant Files

### Scripts

```
jobs/jepa/scripts/train_physio_contrastive_jepa.py    # Training entry point
jobs/jepa/scripts/precompute_physio_values.py         # HR/MAP precomputation (23s)
```

### Config and Submission

```
jobs/jepa/configs/train_physio_contrastive_jepa.yaml  # Main config
jobs/jepa/slurm/train_physio_contrastive_jepa.sbatch  # Slurm script (gl40s_dev)
jobs/jepa/slurm/train_physio_contrastive_jepa_chain.sbatch  # Chain variant
```

### Outputs

```
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/
├── models/physio_contrastive_jepa/2026-08-11-physio-contrastive-jepa-v1/
│   └── *.ckpt                           # Training checkpoints
└── sample_indices/.../
    ├── physio_values_train.npz          # Precomputed HR/MAP (train, 70.5% valid)
    └── physio_values_val.npz            # Precomputed HR/MAP (val, 57.9% valid)
```
