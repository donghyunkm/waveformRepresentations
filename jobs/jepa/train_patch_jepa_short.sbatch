#!/bin/bash
# Resumable short-partition companion to train_patch_jepa_full.sbatch.
# It uses a distinct checkpoint subdirectory so the long job can remain queued.
#SBATCH --job-name=physiojepa-jepa-short
#SBATCH --partition=a100_short
#SBATCH --time=3-00:00:00
# Lightning launches the DDP worker processes from this normal sbatch task.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100:2
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@180
# Exclude only the node used by the first attempt; broaden eligibility across the A100 pool.
# The first attempt failed during CUDA/DDP setup on a100-4015.
# Exclude nodes that produced CUDA failures during two-process startup.
#SBATCH --exclude=a100-4015,a100-4029
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

base_config=/gpfs/home/dk5565/PhysioJEPA/jobs/jepa/train_patch_jepa.yaml
short_config=/tmp/physiojepa-jepa-short-${SLURM_JOB_ID}.yaml
/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python -c "import yaml; p='${base_config}'; c=yaml.safe_load(open(p)); c['run_config']['name']='native_jepa_paper_short'; c['resume']['run_subdir']='2026-08-04-native-jepa-paper-short-v1'; yaml.safe_dump(c, open('${short_config}','w'), sort_keys=False)"
export PHYSIOJEPA_CONFIG="${short_config}"
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa

/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python -c "import torch; assert torch.cuda.device_count() == 2, f'Expected 2 GPUs, found {torch.cuda.device_count()}'; print([torch.cuda.get_device_name(i) for i in range(2)])"
/gpfs/home/dk5565/.conda/envs/physiojepa/bin/torchrun \
  --standalone --nproc_per_node=2 train_patch_jepa.py
