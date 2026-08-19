#!/bin/bash
#SBATCH --job-name=physiojepa-linear-probe-gen
#SBATCH --partition=cpu_dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

cd /gpfs/home/dk5565/PhysioJEPA/jobs/probing

echo "Starting linear probe generalizability analysis"
echo "Host: $(hostname)"
echo "Time: $(date)"

python -u linear_probe_generalizability.py \
    --seed 42 \
    --n-lopo-patients 50 \
    --n-repeats 5 \
    --skip-demean

echo "Done: $(date)"
