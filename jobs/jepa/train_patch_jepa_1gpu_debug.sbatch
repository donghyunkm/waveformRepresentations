#!/bin/bash
# One-GPU JEPA startup/debug run for a100_dev.
# This keeps its checkpoints separate from the two-GPU experiments.
#SBATCH --job-name=physiojepa-jepa-1gpu-debug
#SBATCH --partition=a100_dev
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

base_config=/gpfs/home/dk5565/PhysioJEPA/jobs/jepa/train_patch_jepa.yaml
debug_config=/tmp/physiojepa-jepa-1gpu-debug-${SLURM_JOB_ID}.yaml
/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python -c "import yaml; p='${base_config}'; c=yaml.safe_load(open(p)); c['training']['n_gpus']=1; c['run_config']['name']='native_jepa_paper_1gpu_debug'; c['resume']['run_subdir']='2026-08-04-native-jepa-paper-1gpu-debug-v1'; yaml.safe_dump(c, open('${debug_config}','w'), sort_keys=False)"

export PHYSIOJEPA_CONFIG="${debug_config}"
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa

/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python -c "import torch; assert torch.cuda.device_count() == 1, f'Expected 1 GPU, found {torch.cuda.device_count()}'; print(torch.cuda.get_device_name(0))"
/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python train_patch_jepa.py
