# waveformRepresentations

Research repository for physiological waveform representation learning, built on top of [PhysioJEPA](https://github.com/benmfox/PhysioJEPA) (Fox et al., ML4H 2024, [paper](https://openreview.net/forum?id=bdXsfrNaGY&noteId=bdXsfrNaGY)).

The goal is to train and evaluate self-supervised waveform encoders on MIMIC-III ICU data and use them for downstream hypotension prediction.

## Models

- **PhysioJEPA** — JEPA-style masked latent prediction with EMA target encoder (3-channel, 30-min ABP/ECG/PLETH windows at 125 Hz)
- **PatchTST** — masked patch reconstruction baseline
- **Multi-scale PatchTST** — hierarchical dual-path tokenizer (1s coarse + 200ms fine) fused with a learnable gate
- **FCN** — supervised fully convolutional network baseline
- **ECG-JEPA** — channel/time patch tokenizer with joint positional embeddings

## Results (hypotension prediction)

| Model | AUROC | AP | Notes |
|---|---|---|---|
| Supervised PatchTST (full) | 0.8688 | 0.2766 | |
| Multi-scale PatchTST | 0.8414 | 0.2340 | Best at epoch 4; declined after |
| JEPA attentive probe | 0.844 | 0.278 | Epoch 2 (training in progress) |
| JEPA linear probe (balanced) | 0.8260 | 0.8274 | 50/50 test set; AP inflated |
| PatchTST linear probe (balanced) | 0.8118 | 0.8238 | 50/50 test set; AP inflated |
| FCN baseline | 0.7903 | 0.1911 | |

## Results (medical feature probing)

Ridge regression R² measuring linear decodability of physiological features from
frozen encoder embeddings (10,000 windows, patient-level 80/20 split):

| Feature | JEPA R² | PatchTST R² |
|---------|---------|-------------|
| PLETH_amp | 0.957 | 0.965 |
| PLETH_ACDC | 0.924 | 0.853 |
| PP (pulse pressure) | 0.745 | 0.788 |
| HR | 0.734 | 0.697 |
| ABP_area | 0.709 | 0.751 |
| dPdt_max | 0.602 | 0.734 |
| HRV_RMSSD | 0.532 | 0.469 |
| ABP_tau | 0.495 | 0.529 |
| HR_range | 0.493 | 0.438 |
| ShockIdx | 0.419 | 0.459 |
| DBP | 0.390 | 0.496 |
| SBP | 0.270 | 0.380 |
| ECG_Ramp | 0.232 | 0.318 |
| MAP | 0.136 | 0.282 |
| PTT | -0.058 | -0.015 |

Both encoders richly encode hemodynamic and morphological information. JEPA
better captures PLETH waveform shape and HRV; PatchTST better encodes
ABP-derived hemodynamics (SBP, DBP, MAP, dPdt_max).

## Repository structure

This is an [nbdev](https://nbdev.fast.ai/) repository. The dependency chain is:

```
nbs/ → (nbdev_prepare) → physiojepa/ → jobs/ imports from physiojepa
```

- `nbs/` — source notebooks (source of truth for all library code)
- `physiojepa/` — generated Python modules (do not edit directly; overwritten by `nbdev_prepare`)
- `jobs/` — training and data-processing entry points that import from `physiojepa`

To make changes: edit the relevant notebook in `nbs/`, then run `nbdev_prepare` to regenerate the `physiojepa/` package. The `jobs/` scripts will pick up the changes on next run.

```
jobs/
  data_processing/      cluster-safe Slurm pipeline: manifest → Zarr conversion → labels
  label_processing/     minute-level and event-level label creation
  jepa/                 JEPA pretraining and downstream jobs
  patchtst/             PatchTST pretraining and downstream jobs
  ecgjepa/              ECG-JEPA pretraining and downstream jobs
  baselines/            supervised PatchTST, multi-scale PatchTST, FCN, InceptionTime
```

## Environment

```bash
source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa
cd /gpfs/home/dk5565/PhysioJEPA
```

Python 3.10, PyTorch 2.6.0, CUDA 12.4.

## Data and outputs

Source waveforms: `/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched` (read-only)

All derived data and run artifacts: `/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA`

## Data splits

All supervised and downstream probe experiments use the same fixed subject-level
split (`manifests/hypotension_subject_split_fixed_v1.csv`), ensuring results are
directly comparable across models.

| Split | Subjects | ICU Stays | Samples | Positive Events | Prevalence |
|-------|----------|-----------|---------|-----------------|------------|
| Train (folds 2–9) | 2,013 | 3,195 | 1,022,563 | 43,950 | 4.3% |
| Val (fold 1) | 255 | 399 | 127,831 | 5,493 | 4.3% |
| Test (fold 0) | 256 | 407 | 127,811 | 5,494 | 4.3% |

Algorithm: corrected stratified group 10-fold (seed=16), enforcing subject-level
disjointness — no patient appears in more than one split.

Self-supervised pretraining (JEPA, PatchTST) excludes the downstream val and
test subjects before building its own 95/5 train/val split. Downstream probes
train the classification head on the same 2,013 train subjects the encoder saw
during pretraining, but evaluate on the 255 val + 256 test subjects that were
never seen during pretraining.

See `PROGRESS.md` for current training status and session notes.
