#!/bin/bash
#SBATCH --job-name=physiojepa-homog-d64-sweep
#SBATCH --partition=gl40s_short
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@180
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/probing

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

# Resumable: if embeddings .npz exists, skips extraction and goes straight to sweep
python cluster_homogeneity_sweep_d64.py \
    --n_patients 20 \
    --n_windows 50 \
    --seed 42 \
    --batch_size 8
