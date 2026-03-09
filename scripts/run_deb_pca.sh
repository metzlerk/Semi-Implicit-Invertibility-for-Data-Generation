#!/bin/bash
#SBATCH --job-name=deb_pca
#SBATCH --output=logs/deb_pca_%j.out
#SBATCH --error=logs/deb_pca_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

# Setup
source ~/.bashrc
cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

# Run comparison
echo "Starting DEB latent PCA comparison..."
python3 scripts/compare_deb_latent_pca.py

echo "Done!"
