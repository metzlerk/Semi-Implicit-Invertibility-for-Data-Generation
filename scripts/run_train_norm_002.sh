#!/bin/bash
#SBATCH --job-name=train_norm_0.02
#SBATCH --output=logs/train_norm_002_%j.out
#SBATCH --error=logs/train_norm_002_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
python3 scripts/train_normalized_diffusion.py --beta_end 0.02
