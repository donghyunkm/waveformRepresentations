#!/bin/bash
#SBATCH --job-name=physiojepa-event-labels
#SBATCH --partition=cpu_short
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=192G
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

REPOSITORY=/gpfs/home/dk5565/PhysioJEPA
OUTPUT_ROOT="${OUTPUT_ROOT:-/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA}"
LABEL_DIR="${OUTPUT_ROOT}/labels"
SOURCE_NOTEBOOK="${REPOSITORY}/jobs/label_processing/create_hypotension_shock_labels.ipynb"
RUN_NOTEBOOK="${LABEL_DIR}/create_hypotension_shock_labels-${SLURM_JOB_ID}.ipynb"
EXECUTED_NOTEBOOK="${LABEL_DIR}/create_hypotension_shock_labels-${SLURM_JOB_ID}-executed.ipynb"

mkdir -p "${LABEL_DIR}" "/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
cp "${SOURCE_NOTEBOOK}" "${RUN_NOTEBOOK}"

export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
cd "${LABEL_DIR}"

exec_nb "${RUN_NOTEBOOK}" --dest "${EXECUTED_NOTEBOOK}" --exc_stop
