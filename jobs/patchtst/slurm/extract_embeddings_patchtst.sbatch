#!/bin/bash
#SBATCH --job-name=physiojepa-ptst-extract-emb
#SBATCH --partition=gl40s_dev
#SBATCH --time=4:00:00
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

export PHYSIOJEPA_CONFIG=/gpfs/home/dk5565/PhysioJEPA/jobs/patchtst/configs/extract_embeddings_patchtst.yaml
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python scripts/extract_embeddings.py
