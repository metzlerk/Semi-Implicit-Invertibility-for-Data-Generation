#!/bin/bash
#SBATCH --job-name=single_deb
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/single_deb_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/single_deb_%j.err

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "Training single-chemical (DEB) diffusion with manifold-preserving settings..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

python scripts/train_single_deb_manifold.py

echo ""
echo "Training complete!"
echo "Time: $(date)"
