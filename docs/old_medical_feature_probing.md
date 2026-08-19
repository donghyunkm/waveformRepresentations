# Medical Feature Probing

## Overview

Note, this uses mean-pooled embeddings so analysis needs to be redone

Ridge regression probes measuring linear decodability of physiological features
from frozen self-supervised encoder embeddings. Tests what medical information
the encoder has learned to represent, independent of any downstream task.

---

## Methodology

- **Encoders tested**: JEPA (epoch 13, val=0.215) and PatchTST (epoch 3, val=0.003)
- **Probe type**: Ridge regression (α tuned per feature)
- **Data**: 10,000 randomly sampled 30-min windows from the test set
- **Split**: Patient-level 80/20 train/test (no patient in both)
- **Embedding dimension**: 512 (mean-pooled transformer output — averaged over
  all patch tokens and channels, discarding temporal/positional structure)
- **Features**: 15 physiological features computed from ABP, ECG II, and PLETH
  channels using icuDataExtraction's `compute_stats_subwindow()` logic

### Feature Computation

Features are computed on the central 20 minutes of each 30-min window
(icuDataExtraction's context size). The 5-min margins are seen by the encoder
but not covered by feature targets.

Custom `compute_stats_no_resp()` inlined in the probe script skips RESP-dependent
features entirely (no RESP channel available in PhysioJEPA containers):
- Skipped: RR, PPV, PVI, RESP_amp (100% NaN)
- Computed: HR, SBP, DBP, PP, MAP, ABP_area, PLETH_ACDC, PLETH_amp, ECG_Ramp,
  HRV_RMSSD, HR_range, ShockIdx, PTT, dPdt_max, ABP_tau

### Caveats

- 28% of JEPA embeddings are NaN (filtered before Ridge fitting). Cause not
  fully diagnosed — likely specific corrupted containers or encoder instability
  with certain waveform patterns.
- icuDataExtraction's HR and RR features have known algorithmic bias (+19 bpm
  vs bedside truth). Probes test whether the encoder predicts the algorithm's
  output, not clinical ground truth. Both encoder and features see the same
  underlying signal.

---

## Results: Encoder Probes (R²)

| Feature | JEPA R² | PatchTST R² | JEPA r | PatchTST r |
|---------|---------|-------------|--------|------------|
| PLETH_amp | 0.957 | 0.965 | 0.978 | 0.983 |
| PLETH_ACDC | 0.924 | 0.853 | 0.961 | 0.927 |
| PP (pulse pressure) | 0.745 | 0.788 | 0.866 | 0.893 |
| HR | 0.734 | 0.697 | 0.858 | 0.835 |
| ABP_area | 0.709 | 0.751 | 0.845 | 0.872 |
| dPdt_max | 0.602 | 0.734 | 0.777 | 0.860 |
| HRV_RMSSD | 0.532 | 0.469 | 0.737 | 0.688 |
| ABP_tau | 0.495 | 0.529 | 0.715 | 0.731 |
| HR_range | 0.493 | 0.438 | 0.707 | 0.662 |
| ShockIdx | 0.419 | 0.459 | 0.666 | 0.688 |
| DBP | 0.390 | 0.496 | 0.660 | 0.723 |
| SBP | 0.270 | 0.380 | 0.553 | 0.638 |
| ECG_Ramp | 0.232 | 0.318 | 0.490 | 0.579 |
| MAP | 0.136 | 0.282 | 0.477 | 0.570 |
| PTT | -0.058 | -0.015 | 0.195 | 0.183 |

---

## Results: Raw Signal Statistics Baseline

To assess what the encoder adds beyond trivial signal-level information, a
baseline using 30 summary statistics from raw waveforms (10 stats × 3 channels:
mean, std, min, max, median, IQR, skewness, kurtosis, zero-crossing rate,
peak count) was evaluated with the same Ridge regression pipeline.

| Feature | Raw Stats R² | JEPA R² | PatchTST R² | Δ (best encoder − raw) |
|---------|:-----------:|:-------:|:-----------:|:----------------------:|
| MAP | **0.963** | 0.136 | 0.282 | -0.681 |
| PLETH_amp | **0.963** | 0.957 | 0.965 | +0.002 |
| DBP | **0.927** | 0.390 | 0.496 | -0.431 |
| SBP | **0.825** | 0.270 | 0.380 | -0.446 |
| PP | 0.729 | 0.745 | **0.788** | +0.059 |
| PLETH_ACDC | 0.714 | **0.924** | 0.853 | +0.210 |
| ABP_area | 0.644 | 0.709 | **0.751** | +0.107 |
| ShockIdx | **0.490** | 0.419 | 0.459 | -0.032 |
| dPdt_max | 0.432 | 0.602 | **0.734** | +0.302 |
| ABP_tau | 0.414 | 0.495 | **0.529** | +0.115 |
| ECG_Ramp | 0.224 | 0.232 | **0.318** | +0.094 |
| HR_range | 0.207 | **0.493** | 0.438 | +0.287 |
| HRV_RMSSD | 0.185 | **0.532** | 0.469 | +0.347 |
| HR | 0.092 | **0.734** | 0.697 | +0.642 |
| PTT | 0.071 | -0.058 | -0.015 | -0.086 |

---

## Interpretation

### Encoders excel at temporal/morphological features

Features requiring beat detection and timing analysis show the largest
encoder advantage:
- **HR (+0.64)**: requires detecting R-peaks and computing intervals
- **HRV_RMSSD (+0.35)**: requires RR-interval variability (second-order timing)
- **dPdt_max (+0.30)**: requires identifying systolic upstrokes and computing
  gradients
- **HR_range (+0.29)**: requires instantaneous HR from beat detection
- **PLETH_ACDC (+0.21)**: requires peak/trough detection in PPG

### Raw stats dominate for absolute signal levels

- MAP (0.96), DBP (0.93), SBP (0.83) are near-perfectly captured by channel
  mean/min/max (MAP ≈ mean ABP by definition). The encoders *lose* this
  information because IQR normalization during pretraining strips absolute
  values.

### Neither captures pulse transit time

PTT (R² ≈ 0 everywhere) requires precise cross-channel temporal alignment
(ECG R-peak to ABP foot) at sub-sample resolution — too fine-grained for
either approach.

### JEPA vs PatchTST Specialization

- **JEPA excels at**: PLETH waveform shape (ACDC R²=0.92) and autonomic
  dynamics (HRV R²=0.53, HR_range R²=0.49)
- **PatchTST excels at**: ABP-derived hemodynamics (SBP, DBP, MAP, dPdt_max,
  ABP_tau, ABP_area)

### Conclusion

The encoder's value lies in learning waveform dynamics (rhythm, morphology,
beat-to-beat timing) rather than signal levels. This is consistent with IQR
normalization removing absolute scale during pretraining. The self-supervised
objective teaches the encoder to understand temporal structure within and
between heartbeats — information that cannot be recovered from simple summary
statistics.

---

## 19-Feature Characterization for Cluster Analysis

The same icuDataExtraction framework provides 19 physiological features used
for cluster characterization (see [Representation Analysis](representation_analysis.md)):

ECG_Ramp, PPV, HRV_RMSSD, ABP_tau, RESP_amp, ABP_area, dPdt_max, PP,
PLETH_ACDC, HR_range, PLETH_amp, DBP, ShockIdx, PTT, HR, RR, SBP, MAP, PVI

These features — extracted from time-aligned windows using epoch offset
correction — were used to characterize pooled clusters' physiological profiles
and identify what distinguishes pre-hypotensive clusters from safe clusters
(primarily reduced autonomic variability: low HRV_RMSSD, narrow HR_range).

---

## Relevant Files

### Scripts

- `jobs/probing/probe_medical_features.py` — Main probe script (encoder + feature extraction + Ridge)
- `jobs/probing/probe_raw_stats_baseline.py` — Raw statistics baseline
- `jobs/probing/submit_probe.sbatch` — Slurm submission

### Results

```
/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/probing/medical_features/
├── probe_results.csv                # Encoder R² results (15 features × 2 models)
├── probe_results_raw_stats.csv      # Raw stats baseline results
├── cache_n10000_seed42.npz          # Cached waveforms + features
├── raw_stats_n10000_seed42.npy      # Cached raw statistics
├── jepa_embeddings_n10000.npy       # JEPA embeddings for probed windows
└── ptst_embeddings_n10000.npy       # PatchTST embeddings for probed windows
```

---

## Status

✅ **Completed.** All 15 features probed for both encoders and raw baseline.
Results documented in README.md.
