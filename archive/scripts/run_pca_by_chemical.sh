#!/bin/bash
#SBATCH --job-name=pca_by_chem
#SBATCH --output=logs/pca_by_chem_%j.out
#SBATCH --error=logs/pca_by_chem_%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "=========================================="
echo "PCA by Chemical: Real vs Diffusion vs Gaussian"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

python scripts/generate_pca_by_chemical.py

echo ""
echo "End time: $(date)"
echo "Done!"
