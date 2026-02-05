#!/bin/bash
#SBATCH --job-name=continue_ae_training
#SBATCH --output=logs/continue_ae_training_%j.out
#SBATCH --error=logs/continue_ae_training_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some info
echo "Starting Continued Autoencoder Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Run the training
python scripts/continue_training_autoencoder.py

echo ""
echo "End time: $(date)"
echo "Job completed!"
