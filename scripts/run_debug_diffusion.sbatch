#!/bin/bash
#SBATCH --job-name=debug_diff
#SBATCH --output=logs/debug_diff_%j.out
#SBATCH --error=logs/debug_diff_%j.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem=16G

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
python scripts/train_diffusion_mmd.py
