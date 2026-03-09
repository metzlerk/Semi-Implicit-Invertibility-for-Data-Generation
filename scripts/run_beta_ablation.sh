#!/bin/bash
#SBATCH --job-name=beta_abl
#SBATCH --output=logs/beta_ablation_%j.out
#SBATCH --error=logs/beta_ablation_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Setup
source ~/.bashrc
cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

# Run comparison
echo "Starting beta ablation comparison..."
python3 scripts/compare_beta_ablation.py

echo "Done!"
