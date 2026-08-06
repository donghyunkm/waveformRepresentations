#!/bin/bash
#SBATCH --job-name=physiojepa-convert
#SBATCH --partition=cpu_short
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --requeue
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%A_%a.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%A_%a.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa
cd /gpfs/home/dk5565/PhysioJEPA

python jobs/data_processing/scripts/convert_waveform_array.py \
    --manifest "${MANIFEST:-/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/manifests/waveform_manifest.csv}" \
    --output-root "${OUTPUT_ROOT:-/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA}" \
    --records-per-task "${RECORDS_PER_TASK:-10}"
