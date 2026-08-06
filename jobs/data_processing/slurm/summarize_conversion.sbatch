#!/bin/bash
#SBATCH --job-name=physiojepa-summary
#SBATCH --partition=cpu_short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa
cd /gpfs/home/dk5565/PhysioJEPA

python jobs/data_processing/summarize_conversion.py \
    --manifest "${MANIFEST:-/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/manifests/waveform_manifest.csv}" \
    --output-root "${OUTPUT_ROOT:-/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA}" \
    --records-per-task "${RECORDS_PER_TASK:-10}"
