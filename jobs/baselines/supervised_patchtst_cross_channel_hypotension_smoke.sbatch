#!/bin/bash
#SBATCH --job-name=physiojepa-sptst-xch-smoke
#SBATCH --partition=a100_long
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=a100-4041
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

export PHYSIOJEPA_CONFIG=/gpfs/home/dk5565/PhysioJEPA/jobs/baselines/supervised_patchtst_cross_channel_hypotension_smoke.yaml
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/baselines

hostname
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable in the batch allocation'; torch.zeros(1, device='cuda'); print('CUDA preflight passed:', torch.cuda.get_device_name(0))"
python train_supervised_patchtst_cross_channel_hypotension.py
