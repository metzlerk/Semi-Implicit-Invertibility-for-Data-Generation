#!/bin/bash
#SBATCH --job-name=eval_quality
#SBATCH --output=logs/eval_quality_%j.out
#SBATCH --error=logs/eval_quality_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
python3 scripts/evaluate_generation_quality.py
