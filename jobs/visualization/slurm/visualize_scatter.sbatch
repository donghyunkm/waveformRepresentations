#!/bin/bash
#SBATCH --job-name=physiojepa-viz-scatter
#SBATCH --partition=gpu4_short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"

cd /gpfs/home/dk5565/PhysioJEPA/jobs/visualization

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

python scripts/visualize_scatter.py
