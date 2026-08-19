#!/bin/bash
#SBATCH --job-name=physiojepa-ptst-predict
#SBATCH --partition=gl40s_dev
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

python scripts/predict_hypotension_from_checkpoint.py \
    --config /gpfs/home/dk5565/PhysioJEPA/jobs/patchtst/configs/train_hypotension_fixed.yaml \
    --checkpoint /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/patchtst_hypotension_probe/2026-08-07-patchtst-hypotension-probe-fixed-v1/best-auprc-epoch=06-auprc=0.26830.ckpt \
    --output /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/predictions/patchtst_hypotension_probe/patchtst_attentive_probe_fixed_v1.pt \
    --model-type patchtst
