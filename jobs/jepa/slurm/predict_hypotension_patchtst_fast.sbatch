#!/bin/bash
#SBATCH --job-name=physiojepa-ptst-predict-fast
#SBATCH --partition=gl40s_dev
#SBATCH --time=4:00:00
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

# --- Local SSD staging for fast I/O ---
LOCAL_CONTAINER_DIR="/tmp/physiojepa-containers-${SLURM_JOB_ID}"
GPFS_CONTAINER_DIR="/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/containers"
FILELIST="/tmp/physiojepa-filelist-${SLURM_JOB_ID}.txt"

cleanup() {
    echo "Cleaning up local containers..."
    rm -rf "${LOCAL_CONTAINER_DIR}" "${FILELIST}" "${MPLCONFIGDIR}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Stage 1: Generate file list (val+test containers only) ==="
python -c "
import pandas as pd
from pathlib import Path

cache_dir = Path('/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/fcn_hypotension_paper')
dataset_name = 'zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1'

all_paths = set()
for split in ('val', 'test'):
    df = pd.read_csv(cache_dir / f'{dataset_name}-{split}_samples.csv.gz')
    all_paths.update(df['file_path'].unique())

with open('${FILELIST}', 'w') as f:
    for path in sorted(all_paths):
        f.write(Path(path).name + '\n')

print(f'File list: {len(all_paths)} containers for val+test')
"

N_FILES=$(wc -l < "${FILELIST}")
echo "Need to copy ${N_FILES} ZipStore files"

echo "=== Stage 2: Copy containers to local SSD ==="
mkdir -p "${LOCAL_CONTAINER_DIR}"

START_COPY=$(date +%s)
cat "${FILELIST}" | xargs -P 16 -I {} cp "${GPFS_CONTAINER_DIR}/{}" "${LOCAL_CONTAINER_DIR}/"
END_COPY=$(date +%s)
COPY_TIME=$((END_COPY - START_COPY))

COPY_SIZE=$(du -sh "${LOCAL_CONTAINER_DIR}" | cut -f1)
echo "Copied ${COPY_SIZE} in ${COPY_TIME} seconds"

export PHYSIOJEPA_CONTAINERS_OVERRIDE="${LOCAL_CONTAINER_DIR}"

echo "=== Stage 3: Run prediction ==="
cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

python scripts/predict_hypotension_from_checkpoint.py \
    --config /gpfs/home/dk5565/PhysioJEPA/jobs/patchtst/configs/train_hypotension_fixed.yaml \
    --checkpoint /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/patchtst_hypotension_probe/2026-08-07-patchtst-hypotension-probe-fixed-v1/best-auprc-epoch=06-auprc=0.26830.ckpt \
    --output /gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/predictions/patchtst_hypotension_probe/patchtst_attentive_probe_fixed_v1.pt \
    --model-type patchtst

echo "=== Done ==="
