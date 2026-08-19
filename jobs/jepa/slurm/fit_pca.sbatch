#!/bin/bash
#SBATCH --job-name=physiojepa-fit-pca
#SBATCH --partition=gl40s_short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

export PHYSIOJEPA_CONFIG=/gpfs/home/dk5565/PhysioJEPA/jobs/jepa/configs/train_hypotension_fixed.yaml
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa

echo "Fit PCA on JEPA embeddings"
echo "Host: $(hostname), Job: ${SLURM_JOB_ID}"
echo "Time: $(date)"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python -u scripts/fit_pca.py --n_windows 5000 --batch_size 64 --seed 42

echo "Done: $(date)"
