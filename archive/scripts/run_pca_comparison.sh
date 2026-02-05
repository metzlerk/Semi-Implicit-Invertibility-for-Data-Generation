#!/bin/bash
#SBATCH --job-name=pca_comparison
#SBATCH --output=logs/pca_comparison_%j.out
#SBATCH --error=logs/pca_comparison_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=========================================="
echo "PCA Comparison: Real vs Gaussian vs Diffusion"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Run the comparison
python scripts/generate_pca_comparison.py

echo ""
echo "End time: $(date)"
echo "Job completed!"
