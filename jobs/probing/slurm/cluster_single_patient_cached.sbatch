#!/bin/bash
#SBATCH --job-name=physiojepa-single-cluster-cached
#SBATCH --partition=cpu_dev
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err
set -euo pipefail
source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa
export MPLCONFIGDIR="/tmp/physiojepa-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
cd /gpfs/home/dk5565/PhysioJEPA
INPUT=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/single_patient/token_embeddings_p072908_1000windows_seed42.npz
OUTPUT=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/single_patient/clustering_results_p072908_1000windows_seed42.csv
python jobs/probing/extract_and_cluster_single_patient.py --input "$INPUT" --results-output "$OUTPUT"
