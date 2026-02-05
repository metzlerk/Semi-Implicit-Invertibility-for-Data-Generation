#!/bin/bash
#SBATCH --job-name=check_variance
#SBATCH --output=logs/variance_check_%j.out
#SBATCH --error=logs/variance_check_%j.err
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --gpus=1
#SBATCH --mem=16G

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
python scripts/check_diffusion_variance.py
