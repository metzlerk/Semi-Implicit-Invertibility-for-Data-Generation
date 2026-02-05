#!/bin/bash
#SBATCH --job-name=train_diff_sep
#SBATCH --output=logs/train_diff_sep_%j.out
#SBATCH --error=logs/train_diff_sep_%j.err
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
python scripts/train_latent_diffusion.py
