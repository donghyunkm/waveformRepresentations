#!/bin/bash
#SBATCH --job-name=physiojepa-patchtst-full
#SBATCH --partition=a100_long
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@180
#SBATCH --exclude=a100-4029,a100-4003
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

# Full 100-epoch self-supervised PatchTST pretraining (paper configuration).
# Uses 2x A100 GPUs with DDP, batch_size 32 per GPU (64 effective).
# Requeue-safe: Lightning auto-saves on SIGUSR1 and resumes from checkpoint.

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

export PHYSIOJEPA_CONFIG=/gpfs/home/dk5565/PhysioJEPA/jobs/patchtst/configs/train_patchtst_cluster.yaml
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/patchtst

# Verify 2 GPUs are visible.
/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python -c "
import torch
assert torch.cuda.device_count() == 2, f'Expected 2 GPUs, found {torch.cuda.device_count()}'
print([torch.cuda.get_device_name(i) for i in range(2)])
"

/gpfs/home/dk5565/.conda/envs/physiojepa/bin/torchrun \
  --standalone --nproc_per_node=2 scripts/train_patchtst_cluster.py
