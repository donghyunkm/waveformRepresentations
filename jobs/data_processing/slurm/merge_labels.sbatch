#!/bin/bash
#SBATCH --job-name=physiojepa-label-merge
#SBATCH --partition=cpu_short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa
cd /gpfs/home/dk5565/PhysioJEPA

python jobs/data_processing/scripts/merge_minute_label_shards.py \
    --shard-dir "${OUTPUT_ROOT:-/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA}/labels/minute_shards" \
    --manifest "${MANIFEST:-/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/manifests/waveform_manifest.csv}" \
    --records-per-task "${RECORDS_PER_TASK:-10}" \
    --output "${OUTPUT_ROOT:-/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA}/labels/hypotension_si_labels_mimic.csv.gz"
