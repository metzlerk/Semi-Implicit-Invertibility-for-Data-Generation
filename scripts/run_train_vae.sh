#!/bin/bash
#SBATCH --job-name=train_vae
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/train_vae_%j.out

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "Training VAE baseline model..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python3)"
echo ""

python3 scripts/train_vae_baseline.py

echo ""
echo "VAE training complete!"
echo "Time: $(date)"
