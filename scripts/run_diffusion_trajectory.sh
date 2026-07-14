#!/bin/bash
#SBATCH --job-name=diff_traj
#SBATCH --output=logs/diffusion_trajectory_%j.out
#SBATCH --error=logs/diffusion_trajectory_%j.err
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

# Run visualization for beta=0.02 (weak diffusion)
echo "============================================"
echo "Generating trajectory for beta_end=0.02"
echo "============================================"
python3 scripts/visualize_diffusion_trajectory.py 0.02

echo ""
echo "============================================"
echo "Generating trajectory for beta_end=0.2"
echo "============================================"
python3 scripts/visualize_diffusion_trajectory.py 0.2

echo ""
echo "Done! Generated 4 images total."
