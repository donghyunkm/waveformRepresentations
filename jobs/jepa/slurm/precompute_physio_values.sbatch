#!/bin/bash
#SBATCH --job-name=physiojepa-precompute-physio
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/data/eh3828lab/derived_datasets/baselines/PhysioJEPA/logs/slurm/%x-%j.err

source /gpfs/share/apps/anaconda3/gpu/2025.06/etc/profile.d/conda.sh
conda activate physiojepa

cd /gpfs/home/dk5565/PhysioJEPA/jobs/jepa/scripts

echo "Pre-computing physiological values from bedside numerics"
echo "Host: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Time: $(date)"

python -u precompute_physio_values.py --workers 16 --split both

echo "Done: $(date)"
