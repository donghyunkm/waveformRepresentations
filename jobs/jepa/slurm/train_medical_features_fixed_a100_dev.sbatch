#!/bin/bash
#SBATCH --job-name=physiojepa-jepa-medfeat-probe
#SBATCH --partition=a100_dev
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --requeue
#SBATCH --signal=B:SIGUSR1@180
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

export PHYSIOJEPA_CONFIG=/gpfs/home/dk5565/PhysioJEPA/jobs/jepa/configs/train_medical_features_fixed.yaml
export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
export TMPDIR="/tmp/physiojepa-${SLURM_JOB_ID}"

mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa

# === Stage 1: Copy containers to local SSD ===
LOCAL_DIR="/tmp/physiojepa-containers-${SLURM_JOB_ID}"
GPFS_DIR="/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/containers"
FILELIST="/tmp/physiojepa-filelist-${SLURM_JOB_ID}.txt"

cleanup() {
    echo "Cleaning up ${LOCAL_DIR}..."
    rm -rf "${LOCAL_DIR}" "${FILELIST}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Generating file list..."
python -c "
import pandas as pd
from pathlib import Path

cache_dir = Path('/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/fcn_hypotension_paper')
dataset_name = 'zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1'

all_paths = set()
for split in ('train', 'val', 'test'):
    df = pd.read_csv(cache_dir / f'{dataset_name}-{split}_samples.csv.gz')
    all_paths.update(df['file_path'].unique())

with open('${FILELIST}', 'w') as f:
    for path in sorted(all_paths):
        f.write(Path(path).name + '\n')
print(f'{len(all_paths)} containers to copy')
"

echo "Copying containers to local SSD..."
mkdir -p "${LOCAL_DIR}"
START_COPY=$(date +%s)
cat "${FILELIST}" | xargs -P 16 -I {} cp "${GPFS_DIR}/{}" "${LOCAL_DIR}/"
END_COPY=$(date +%s)
echo "Copy done: $((END_COPY - START_COPY)) seconds, $(du -sh ${LOCAL_DIR} | cut -f1)"

# === Stage 2: Run training with local path override ===
export PHYSIOJEPA_CONTAINERS_OVERRIDE="${LOCAL_DIR}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python scripts/train_medical_features_fixed.py
