#!/bin/bash
#SBATCH --job-name=physiojepa-patchtst-smoke
#SBATCH --partition=a100_dev
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

# Smoke test: validates memory, data loading, and 2-GPU DDP with the full
# per-GPU batch size. A successful smoke confirms the full job will fit.

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

# Create a temporary smoke config that limits training to a few batches.
base_config=/gpfs/home/dk5565/PhysioJEPA/jobs/patchtst/train_patchtst_cluster.yaml
smoke_config="/tmp/physiojepa-patchtst-smoke-${SLURM_JOB_ID}.yaml"
/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python -c "
import yaml
c = yaml.safe_load(open('${base_config}'))
c['training']['epochs'] = 1
c['run_config']['name'] = 'PatchTST_paper_smoke'
c['resume']['run_subdir'] = '2026-08-05-patchtst-paper-smoke-v1'
c['resume']['enabled'] = False
yaml.safe_dump(c, open('${smoke_config}', 'w'), sort_keys=False)
"

export PHYSIOJEPA_CONFIG="${smoke_config}"
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/patchtst

# Verify 2 GPUs are visible.
/gpfs/home/dk5565/.conda/envs/physiojepa/bin/python -c "
import torch
assert torch.cuda.device_count() == 2, f'Expected 2 GPUs, found {torch.cuda.device_count()}'
print([torch.cuda.get_device_name(i) for i in range(2)])
"

/gpfs/home/dk5565/.conda/envs/physiojepa/bin/torchrun \
  --standalone --nproc_per_node=2 train_patchtst_cluster.py
