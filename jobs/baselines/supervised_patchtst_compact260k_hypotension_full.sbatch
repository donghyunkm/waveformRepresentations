#!/bin/bash
#SBATCH --job-name=physiojepa-sptst-c260-full
#SBATCH --partition=a100_long
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=a100-4030,a100-4031,a100-4032,a100-4033,a100-4034,a100-4035,a100-4037,a100-4039,a100-4040,a100-4041,a100-4042,a100-4044,a100-4045
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

export PHYSIOJEPA_CONFIG=/gpfs/home/dk5565/PhysioJEPA/jobs/baselines/supervised_patchtst_compact260k_hypotension_full.yaml
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/baselines

python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable in the batch allocation'; torch.zeros(1, device='cuda'); print('CUDA preflight passed:', torch.cuda.get_device_name(0))"
python train_supervised_patchtst_hypotension.py
