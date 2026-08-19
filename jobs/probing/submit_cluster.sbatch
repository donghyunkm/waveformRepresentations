#!/bin/bash
#SBATCH --job-name=physiojepa-cluster-emb
#SBATCH --partition=gl40s_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

cd /gpfs/home/dk5565/PhysioJEPA/jobs/probing

echo "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Use full test set (127,811 samples)
python -u cluster_embeddings.py --n-samples 0 --seed 42
