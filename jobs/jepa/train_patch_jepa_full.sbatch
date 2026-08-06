#!/bin/bash
#SBATCH --job-name=physiojepa-jepa-full
#SBATCH --partition=a100_long
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@180
#SBATCH --exclude=a100-4030,a100-4031,a100-4032,a100-4033,a100-4034,a100-4035,a100-4037,a100-4039,a100-4040,a100-4041,a100-4042,a100-4044,a100-4045
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

export PHYSIOJEPA_CONFIG=/gpfs/home/dk5565/PhysioJEPA/jobs/jepa/train_patch_jepa.yaml
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa

python -c "import torch; assert torch.cuda.device_count() == 2, f'Expected 2 GPUs, found {torch.cuda.device_count()}'; print([torch.cuda.get_device_name(i) for i in range(2)])"
/gpfs/home/dk5565/.conda/envs/physiojepa/bin/torchrun \
  --standalone --nproc_per_node=2 train_patch_jepa.py
