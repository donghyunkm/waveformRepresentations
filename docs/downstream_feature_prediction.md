# Downstream Medical Feature Prediction

## Overview

Probes measuring decodability of 15 physiological features from frozen
self-supervised encoder embeddings. Tests what medical information the encoder
has learned to represent.

---

## Results: Mean-Pooled Ridge Regression (R²)

Ridge regression on mean-pooled embeddings (512-d, averaged over all patch
tokens and channels). 10,000 windows, patient-level 80/20 split, α tuned per
feature.

| Feature | JEPA R² | PatchTST R² | Raw Stats R² | Best |
|---------|---------|-------------|:------------:|------|
| PLETH_amp | 0.957 | 0.965 | 0.963 | PatchTST |
| PLETH_ACDC | 0.924 | 0.853 | 0.714 | JEPA |
| PP | 0.745 | 0.788 | 0.729 | PatchTST |
| HR | 0.734 | 0.697 | 0.092 | JEPA |
| ABP_area | 0.709 | 0.751 | 0.644 | PatchTST |
| dPdt_max | 0.602 | 0.734 | 0.432 | PatchTST |
| HRV_RMSSD | 0.532 | 0.469 | 0.185 | JEPA |
| ABP_tau | 0.495 | 0.529 | 0.414 | PatchTST |
| HR_range | 0.493 | 0.438 | 0.207 | JEPA |
| ShockIdx | 0.419 | 0.459 | 0.490 | Raw Stats |
| DBP | 0.390 | 0.496 | 0.927 | Raw Stats |
| SBP | 0.270 | 0.380 | 0.825 | Raw Stats |
| ECG_Ramp | 0.232 | 0.318 | 0.224 | PatchTST |
| MAP | 0.136 | 0.282 | 0.963 | Raw Stats |
| PTT | -0.058 | -0.015 | 0.071 | Raw Stats |

**Mean R² (encoder)**: JEPA=0.51, PatchTST=0.54

### Raw Signal Statistics Baseline

To assess what the encoders add beyond trivial signal-level information, a
baseline using 30 summary statistics computed directly from raw waveforms was
evaluated with the same Ridge regression pipeline and patient split.

**Statistics computed (10 per channel × 3 channels = 30 features):**
- Mean, standard deviation, min, max, median
- IQR (interquartile range)
- Skewness, kurtosis
- Zero-crossing rate
- Peak count

These are simple, non-learned features that capture signal amplitude, spread,
and basic shape — but no temporal structure (beat detection, inter-beat timing,
waveform morphology). They serve as a floor: any feature well-predicted by raw
stats doesn't require a learned encoder.

### Interpretation

- Encoders excel at **temporal/morphological features** requiring beat detection
  (HR +0.64, HRV +0.35, dPdt_max +0.30 over raw stats)
- Raw stats dominate for **absolute signal levels** (MAP, DBP, SBP) because IQR
  normalization during pretraining strips absolute values — raw channel mean ≈
  MAP by definition, so a trivial statistic achieves R²=0.96
- **JEPA** better captures PLETH waveform shape and autonomic dynamics
- **PatchTST** better encodes ABP-derived hemodynamics
- Neither captures pulse transit time (requires sub-sample cross-channel alignment)

The encoder's value lies in learning waveform dynamics (rhythm, morphology,
beat-to-beat timing) rather than signal levels. This is consistent with IQR
normalization removing absolute scale during pretraining.

---

## Failed Experiment: Full-Embedding Attentive Probe

### Motivation

The mean-pooled Ridge probe discards temporal structure (1800 patch tokens →
single 512-d vector). An attentive probe on the full token sequence should
capture richer temporal/morphological information and improve on the mean-pooled
baseline.

### Setup

- **Architecture**: Frozen JEPA encoder → full token sequence (3 × n_patches × 512)
  → AttentiveClassifier (4 heads, depth=1, mlp_ratio=4, `num_classes=15`) → MSE
- **Split**: Same patient-level train/val/test as hypotension probes
- **Training**: 11 epochs reached (of 20), OneCycle LR (max_lr=0.005), bs=128
- **Normalization**: Z-score per feature using training set statistics
- **Job chain**: 26343381–26343385 on gl40s_dev

### Training Metrics (misleading)

The Lightning progress bar reported:
- Epoch 10: `val_R²=0.700`, `val_loss=1.200`
- Epoch 11: `val_R²=0.718`, `val_loss=1.250`

These appeared to show the model improving steadily and surpassing the Ridge
baseline (mean R²≈0.51).

### Actual Test Results

When predictions were saved and evaluated per-feature on held-out patients:

| Split | Mean R² | Mean Pearson r | Verdict |
|-------|---------|----------------|---------|
| Val | -0.37 | ~0.0 | Worse than mean predictor |
| Test | -0.21 | ~0.0 | Worse than mean predictor |

**All 15 features have near-zero correlation** between predictions and targets
on held-out patients. The model predictions have ~50% of the targets' standard
deviation (e.g., HR: pred_std=7.2 vs target_std=16.1), meaning it predicts
near the population mean with slight variation.

Within-patient evaluation (subtracting each patient's mean) also shows zero
predictive signal (within-patient R² ≈ -4 to -5), confirming the model has
learned nothing useful for feature prediction at any level.

### Root Cause

The training-time `val_R²=0.718` was an **aggregate metric computed incorrectly**:
the validation step pools predictions across all 15 features before computing R².
When features have very different scales (HR~100 vs PLETH_ACDC~0.9), a model
that predicts near each feature's mean achieves high pooled R² because the
between-feature variance dominates the residual.

The model essentially learned to output approximately correct feature means
(the normalization stats) but captured no per-window or per-patient variance.
This is consistent with the broader finding that JEPA embeddings are dominated
by patient identity, and the attentive probe could not learn cross-patient
feature prediction from the representation.

### Conclusions

1. **Mean-pooled Ridge regression remains the valid result** for medical feature
   probing. It properly evaluates per-feature R² with patient-level splits.
2. The full-embedding attentive probe adds no value — the additional temporal
   tokens don't help when the task requires generalizing feature values across
   patients.
3. The misleading aggregate R² metric is a training-code bug that needs fixing
   (should log per-feature R²).

---

## Features Predicted

| # | Feature | Description |
|---|---------|-------------|
| 1 | HR | Heart rate (bpm) |
| 2 | SBP | Systolic blood pressure |
| 3 | DBP | Diastolic blood pressure |
| 4 | PP | Pulse pressure (SBP-DBP) |
| 5 | MAP | Mean arterial pressure |
| 6 | ABP_area | ABP waveform area |
| 7 | PLETH_ACDC | Pleth AC/DC ratio |
| 8 | PLETH_amp | Pleth amplitude |
| 9 | ECG_Ramp | ECG R-wave amplitude |
| 10 | HRV_RMSSD | Heart rate variability |
| 11 | HR_range | HR range over context |
| 12 | ShockIdx | Shock index (HR/SBP) |
| 13 | PTT | Pulse transit time |
| 14 | dPdt_max | Max ABP upstroke slope |
| 15 | ABP_tau | ABP decay time constant |

Skipped (RESP-dependent, always NaN): RR, PPV, PVI, RESP_amp.

---

## Methodology Notes

### Mean-Pooled Probe (working)

- **Encoders tested**: JEPA (epoch 13) and PatchTST (epoch 3)
- **Probe type**: Ridge regression (α tuned per feature)
- **Data**: 10,000 randomly sampled 30-min windows
- **Split**: Patient-level 80/20 train/test
- **Embedding dimension**: 512 (mean-pooled over all tokens)
- **28% of JEPA embeddings are NaN** (filtered before Ridge fitting)

### Attentive Probe (failed)

- **Architecture**: Frozen encoder → AttentiveClassifier (4 heads, depth=1)
- **Input**: Full token sequence (3 channels × n_patches × 512 dims)
- **Split**: Same patient-level train/val/test as hypotension
- **Target normalization**: Z-score per feature (training set mean/std)
- **NaN handling**: MSE computed only on valid targets per sample
- **Training**: 11 of 20 epochs completed before chain exhausted

---

## Relevant Files

### Mean-Pooled Probe

```
jobs/probing/probe_medical_features.py
jobs/probing/probe_raw_stats_baseline.py

/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/probing/medical_features/
├── probe_results.csv
├── probe_results_raw_stats.csv
├── jepa_embeddings_n10000.npy
└── ptst_embeddings_n10000.npy
```

### Attentive Probe (failed)

```
jobs/jepa/scripts/train_medical_features_fixed.py
jobs/jepa/configs/train_medical_features_fixed.yaml

/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/predictions/
└── jepa_medical_feature_probe/native_jepa_medical_feature_attentive_probe_v1.pt
```

---

## Status

| Experiment | Status | Result |
|------------|--------|--------|
| Mean-pooled Ridge (JEPA) | ✅ Complete | Mean R²=0.51 |
| Mean-pooled Ridge (PatchTST) | ✅ Complete | Mean R²=0.54 |
| Raw stats baseline | ✅ Complete | Varies by feature |
| Attentive probe (JEPA, full tokens) | ❌ Failed | R²<0, no signal |
| Attentive probe (PatchTST) | 🚫 Not run | Cancelled; not worth pursuing |
| **Mean-pooled Ridge (JEPA d64)** | ✅ Complete | Mean R²=0.26 |

---

## JEPA d64 Mean-Pooled Ridge Probe — Complete

Same methodology as the full JEPA Ridge probe (mean-pooled embeddings → Ridge
regression → per-feature R²), applied to the d64 encoder (embed_dim=64 vs 512).

- **Script**: `jobs/probing/probe_medical_features_d64.py`
- **Sbatch**: `jobs/probing/slurm/probe_medical_features_d64_gl40s_short.sbatch`
- **Job**: 26487639, completed 2026-08-16 (17 min on L40S)
- **Output**: `probing/medical_features_d64/probe_results_d64.csv`
- **Data**: Same 10,000 windows, seed=42, patient-level 80/20 split
- **NaN filtering**: 2,814/10,000 rows with NaN embeddings removed (28%, same
  rate as full JEPA), leaving 5,870 train / 1,316 test.

### Results: JEPA d64 vs Full JEPA (d512)

| Feature | d64 R² | d512 R² | Δ R² | d64 Pearson r |
|---------|--------|---------|------|---------------|
| PLETH_amp | 0.944 | 0.957 | −0.013 | 0.971 |
| PLETH_ACDC | 0.872 | 0.924 | −0.052 | 0.934 |
| HR | 0.539 | 0.734 | −0.195 | 0.735 |
| ABP_area | 0.363 | 0.709 | −0.346 | 0.604 |
| HRV_RMSSD | 0.360 | 0.532 | −0.172 | 0.607 |
| HR_range | 0.354 | 0.493 | −0.139 | 0.603 |
| ShockIdx | 0.201 | 0.419 | −0.218 | 0.469 |
| ECG_Ramp | 0.173 | 0.232 | −0.059 | 0.418 |
| ABP_tau | 0.135 | 0.495 | −0.360 | 0.384 |
| PP | 0.087 | 0.745 | −0.658 | 0.309 |
| dPdt_max | 0.052 | 0.602 | −0.550 | 0.255 |
| SBP | 0.024 | 0.270 | −0.246 | 0.211 |
| MAP | −0.033 | 0.136 | −0.169 | 0.275 |
| DBP | −0.038 | 0.390 | −0.428 | 0.269 |
| PTT | −0.064 | −0.058 | −0.006 | 0.104 |

**Mean R² (d64)**: 0.26 (vs d512: 0.51)

### Interpretation

The 8× capacity reduction (512→64 dims) causes substantial degradation in
medical feature decodability, with a clear pattern:

- **Preserved** (Δ < 0.06): PLETH morphology features (PLETH_amp, PLETH_ACDC)
  remain excellent, suggesting waveform shape requires few dimensions to encode.
- **Moderate loss** (0.1–0.2): Temporal features (HR, HRV_RMSSD, HR_range) lose
  some fidelity but retain positive R², consistent with heartbeat timing being
  a lower-dimensional signal.
- **Severe loss** (0.3–0.7): ABP-derived hemodynamics (PP, dPdt_max, ABP_area,
  ABP_tau, DBP) drop dramatically. These features characterize detailed pressure
  waveform morphology which apparently requires higher-dimensional embedding
  space to encode linearly.

The d64 encoder prioritizes encoding what can be compressed into 64 dimensions:
waveform amplitude and basic rhythm. The full encoder's additional 448 dimensions
are meaningfully utilized for multi-channel hemodynamic detail. This argues
against extreme compression for tasks requiring detailed ABP analysis.
