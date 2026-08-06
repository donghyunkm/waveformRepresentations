#!/bin/bash
#SBATCH --job-name=physiojepa-manifest
#SBATCH --partition=cpu_short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa
cd /gpfs/home/dk5565/PhysioJEPA

python jobs/data_processing/build_waveform_manifest.py \
    --source-root "${SOURCE_ROOT:-/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched}" \
    --output-root "${OUTPUT_ROOT:-/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA}" \
    --manifest "${MANIFEST:-/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/manifests/waveform_manifest.csv}"
