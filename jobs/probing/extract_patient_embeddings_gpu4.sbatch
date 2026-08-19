#!/bin/bash
#SBATCH --job-name=physiojepa-extract-emb
#SBATCH --partition=gpu4_dev
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
cd /gpfs/home/dk5565/PhysioJEPA

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python jobs/probing/extract_patient_embeddings.py --n_patients 20 --n_windows 50 --seed 42 --batch_size 4
