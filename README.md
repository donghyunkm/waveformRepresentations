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

| Model | AUROC | AP |
|---|---|---|
| Supervised PatchTST (full) | 0.8688 | 0.2766 |
| FCN baseline | 0.7903 | 0.1911 |
| Multi-scale PatchTST | in progress | — |

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

See `PROGRESS.md` for current training status and session notes.
