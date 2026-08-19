#!/bin/bash
#SBATCH --job-name=physiojepa-per-patient-cluster
#SBATCH --partition=cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

cd /gpfs/home/dk5565/PhysioJEPA/jobs/probing

echo "Starting per-patient clustering"
echo "Host: $(hostname)"
echo "Time: $(date)"

# JEPA — top 15 patients with UMAP
python -u cluster_single_patient.py \
    --top-n 15 \
    --model jepa \
    --seed 42

# PatchTST — same patients for comparison
python -u cluster_single_patient.py \
    --top-n 15 \
    --model patchtst \
    --seed 42

echo "Done: $(date)"
