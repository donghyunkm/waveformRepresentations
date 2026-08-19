#!/bin/bash
#SBATCH --job-name=physiojepa-physio-contrastive
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
cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa

echo "Physio-Contrastive JEPA training (resume from epoch 12)"
echo "Host: $(hostname), Job: ${SLURM_JOB_ID}"
echo "Time: $(date)"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python -u scripts/train_physio_contrastive_jepa.py

echo "Done: $(date)"
