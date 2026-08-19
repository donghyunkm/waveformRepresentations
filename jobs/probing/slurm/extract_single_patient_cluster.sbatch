#!/bin/bash
#SBATCH --job-name=physiojepa-single-patient-cluster
#SBATCH --partition=gl40s_dev
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

export WANDB_MODE=offline
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

PATIENT="p072908"
N_WINDOWS=1000
OUTPUT_DIR="/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/single_patient"
OUTPUT_FILE="${OUTPUT_DIR}/token_embeddings_${PATIENT}_${N_WINDOWS}windows_seed42.npz"

# --- Local SSD staging (only this patient's containers) ---
LOCAL_CONTAINER_DIR="/tmp/physiojepa-containers-${SLURM_JOB_ID}"
GPFS_CONTAINER_DIR="/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/containers"

cleanup() {
    echo "Cleaning up..."
    rm -rf "${LOCAL_CONTAINER_DIR}" "${MPLCONFIGDIR}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Stage 1: Copy patient containers to local SSD ==="
mkdir -p "${LOCAL_CONTAINER_DIR}"

# Copy only p072908 containers
START_COPY=$(date +%s)
ls "${GPFS_CONTAINER_DIR}"/${PATIENT}-*.zarr.zip | xargs -P 8 -I {} cp {} "${LOCAL_CONTAINER_DIR}/"
END_COPY=$(date +%s)
N_FILES=$(ls "${LOCAL_CONTAINER_DIR}" | wc -l)
COPY_SIZE=$(du -sh "${LOCAL_CONTAINER_DIR}" | cut -f1)
echo "Copied ${N_FILES} files (${COPY_SIZE}) in $((END_COPY - START_COPY)) seconds"

export PHYSIOJEPA_CONTAINERS_OVERRIDE="${LOCAL_CONTAINER_DIR}"

echo "=== Stage 2: Extract embeddings and run clustering ==="
cd /gpfs/home/dk5565/PhysioJEPA

python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

mkdir -p "${OUTPUT_DIR}"

python jobs/probing/extract_and_cluster_single_patient.py \
    --patient "${PATIENT}" \
    --n-windows "${N_WINDOWS}" \
    --output "${OUTPUT_FILE}" \
    --batch-size 16 \
    --num-workers 8 \
    --seed 42

echo "=== Done ==="
