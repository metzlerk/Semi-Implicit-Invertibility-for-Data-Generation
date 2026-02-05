#!/bin/bash
#SBATCH --job-name=finetune_encoder
#SBATCH --output=logs/finetune_encoder_%j.out
#SBATCH --error=logs/finetune_encoder_%j.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem=32G

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
python scripts/finetune_encoder_separation.py
