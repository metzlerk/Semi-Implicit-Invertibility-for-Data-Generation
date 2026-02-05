#!/bin/bash
#SBATCH --job-name=diffusion_improved_sep
#SBATCH --output=logs/diffusion_improved_separation_%j.out
#SBATCH --error=logs/diffusion_improved_separation_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=========================================="
echo "Improved Diffusion Training with Separation Loss"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Run the training
python scripts/diffusion_ims_improved_separation.py

echo ""
echo "End time: $(date)"
echo "Job completed!"
