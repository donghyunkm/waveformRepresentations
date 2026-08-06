#!/bin/bash
#SBATCH --job-name=physiojepa-patchtst-1gpu
#SBATCH --partition=a100_short
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@180
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

# Single-GPU PatchTST pretraining for a100_short (3-day segments).
# Use this if the 2-GPU path hits the same rank-1 CUDA issues as native JEPA.
# Overrides n_gpus=1. Resume-safe across multiple short-partition allocations.

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

base_config=/gpfs/home/dk5565/PhysioJEPA/jobs/patchtst/configs/train_patchtst_cluster.yaml
runtime_config="/tmp/physiojepa-patchtst-1gpu-${SLURM_JOB_ID}.yaml"
/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python -c "
import yaml
c = yaml.safe_load(open('${base_config}'))
c['training']['n_gpus'] = 1
c['run_config']['name'] = 'PatchTST_paper_1gpu'
c['resume']['run_subdir'] = '2026-08-05-patchtst-paper-1gpu-v1'
yaml.safe_dump(c, open('${runtime_config}', 'w'), sort_keys=False)
"

export PHYSIOJEPA_CONFIG="${runtime_config}"
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/patchtst

/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python -c "
import torch
assert torch.cuda.device_count() == 1, f'Expected 1 GPU, found {torch.cuda.device_count()}'
print(torch.cuda.get_device_name(0))
"

/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python scripts/train_patchtst_cluster.py
