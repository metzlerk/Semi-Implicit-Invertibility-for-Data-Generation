#!/bin/bash
#SBATCH --job-name=train_diff
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/train_diff_%j.out

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

python scripts/train_latent_diffusion.py
