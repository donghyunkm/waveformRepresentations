#!/bin/bash
#SBATCH --job-name=physiojepa-cached-single-patient-cluster
#SBATCH --partition=cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

set -euo pipefail

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

cd /gpfs/home/dk5565/PhysioJEPA

INPUT=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/single_patient/token_embeddings_p072908_1000windows_seed42.npz
RESULTS=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/embeddings/single_patient/clustering_p072908_1000windows_seed42.csv

echo \"=== Cached single-patient clustering ===\"
echo \"Input: \\"
echo \"Results: \\"
python -u jobs/probing/extract_and_cluster_single_patient.py \\n    --input \"\\" \\n    --results-output \"\\"
