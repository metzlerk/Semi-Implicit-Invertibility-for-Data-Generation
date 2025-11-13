#!/bin/bash
#SBATCH --job-name=implicit_vs_semiimplicit
#SBATCH --output=logs/implicit_semiimplicit_%j.out
#SBATCH --error=logs/implicit_semiimplicit_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load modules
module load python/3.9
module load cuda/11.7

# Activate virtual environment (if you have one)
# source ~/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run the hyperparameter tuning script
cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/scripts
python compare_implicit_semiimplicit.py

echo "Finished at: $(date)"
