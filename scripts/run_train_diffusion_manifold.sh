#!/bin/bash
#SBATCH --job-name=train_diff_manifold
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/train_diff_manifold_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/train_diff_manifold_%j.err

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "Training diffusion with manifold-preserving settings..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

python scripts/train_diffusion_normalized_full.py

echo ""
echo "Training complete!"
echo "Time: $(date)"
