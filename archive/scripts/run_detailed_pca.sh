#!/bin/bash
#SBATCH --job-name=detailed_pca
#SBATCH --output=logs/detailed_pca_%j.out
#SBATCH --error=logs/detailed_pca_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

echo "=========================================="
echo "Detailed PCA by Chemical"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

# Run the detailed PCA script
python scripts/generate_detailed_pca.py

echo ""
echo "End time: $(date)"
echo "Job completed!"
