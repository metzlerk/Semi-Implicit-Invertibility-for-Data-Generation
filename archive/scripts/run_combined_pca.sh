#!/bin/bash
#SBATCH --job-name=combined_pca
#SBATCH --output=logs/combined_pca_%j.out
#SBATCH --error=logs/combined_pca_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "=========================================="
echo "Combined PCA Visualization"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

python scripts/generate_combined_pca.py

echo ""
echo "End time: $(date)"
echo "Done!"
