#!/bin/bash
#SBATCH --job-name=physiojepa-physio-contrastive
#SBATCH --partition=gl40s_dev
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa
python -u scripts/train_physio_contrastive_jepa.py
