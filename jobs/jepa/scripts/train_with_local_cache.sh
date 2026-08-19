#!/bin/bash
# Stage ZipStore containers to local /tmp for fast I/O, then run training.
#
# This script:
# 1. Generates a list of required ZipStore paths from the sample cache
# 2. Copies them to /tmp/physiojepa-containers/ using parallel cp
# 3. Creates a symlink so the training script finds containers at the same path
# 4. Runs the training script
# 5. Cleans up /tmp on exit
#
# Usage: Called from sbatch. Expects PHYSIOJEPA_CONFIG and PHYSIOJEPA_TRAIN_SCRIPT
# environment variables to be set.

set -euo pipefail

LOCAL_CONTAINER_DIR="/tmp/physiojepa-containers-${SLURM_JOB_ID}"
GPFS_CONTAINER_DIR="/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/containers"
FILELIST="/tmp/physiojepa-filelist-${SLURM_JOB_ID}.txt"

cleanup() {
    echo "Cleaning up local containers..."
    rm -rf "${LOCAL_CONTAINER_DIR}" "${FILELIST}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Stage 1: Generate file list ==="
python -c "
import pandas as pd
from pathlib import Path

cache_dir = Path('/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/models/fcn_hypotension_paper')
dataset_name = 'zipstore_ABP_II_PLETH_125Hz_1800sec_hypotension_fixed_subject_split_v1'

all_paths = set()
for split in ('train', 'val', 'test'):
    df = pd.read_csv(cache_dir / f'{dataset_name}-{split}_samples.csv.gz')
    all_paths.update(df['file_path'].unique())

# Write just filenames (relative to containers dir)
with open('${FILELIST}', 'w') as f:
    for path in sorted(all_paths):
        f.write(Path(path).name + '\n')

print(f'File list: {len(all_paths)} containers')
"

N_FILES=$(wc -l < "${FILELIST}")
echo "Need to copy ${N_FILES} ZipStore files"

echo "=== Stage 2: Copy containers to local SSD ==="
mkdir -p "${LOCAL_CONTAINER_DIR}"

# Use parallel cp with xargs for speed (16 parallel copies)
START_COPY=$(date +%s)
cat "${FILELIST}" | xargs -P 16 -I {} cp "${GPFS_CONTAINER_DIR}/{}" "${LOCAL_CONTAINER_DIR}/"
END_COPY=$(date +%s)
COPY_TIME=$((END_COPY - START_COPY))

COPY_SIZE=$(du -sh "${LOCAL_CONTAINER_DIR}" | cut -f1)
echo "Copied ${COPY_SIZE} in ${COPY_TIME} seconds"

echo "=== Stage 3: Create override symlink ==="
# The sample cache has absolute paths like:
#   /gpfs/data/.../containers/p012345-2100-01-01-00-00.zarr.zip
# We override by bind-mounting or symlinking. Simplest: set an env var that
# the dataset code checks, OR patch at the Python level.
#
# We'll use PHYSIOJEPA_CONTAINERS_OVERRIDE to tell the training script to
# remap container paths to local /tmp.
export PHYSIOJEPA_CONTAINERS_OVERRIDE="${LOCAL_CONTAINER_DIR}"

echo "=== Stage 4: Run training ==="
echo "Config: ${PHYSIOJEPA_CONFIG}"
echo "Script: ${PHYSIOJEPA_TRAIN_SCRIPT}"
echo "Containers override: ${PHYSIOJEPA_CONTAINERS_OVERRIDE}"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python "${PHYSIOJEPA_TRAIN_SCRIPT}"
