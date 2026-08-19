#!/bin/bash
#SBATCH --job-name=physiojepa-linear-probe-remaining
#SBATCH --partition=cpu_short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/physiojepa-linear-probe-remaining-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/physiojepa-linear-probe-remaining-%j.err

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

cd /gpfs/home/dk5565/PhysioJEPA/jobs/probing

python -u linear_probe_generalizability_remaining.py \
    --n-lopo-patients 50 \
    --n-repeats 5 \
    --skip-demean
