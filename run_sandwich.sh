#!/bin/bash
#SBATCH --job-name=sandwich_model
#SBATCH --output=sandwich_model_%j.out
#SBATCH --error=sandwich_model_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

export PYTHONUNBUFFERED=1

echo "Running sandwich model"
python -u sandwich_model.py \
    --data-feather Data/train_data_with_conditions.feather \
    --bottom-frac 0.5 \
    --middle-frac 0.2 \
    --top-frac 0.3 \
    --ham-chem DMMP

echo "All done."