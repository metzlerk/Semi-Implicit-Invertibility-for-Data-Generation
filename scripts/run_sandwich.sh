#!/bin/bash
#SBATCH --job-name=sandwich_model
#SBATCH --output=logs/sandwich_model_%j.out
#SBATCH --error=logs/sandwich_model_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
mkdir -p logs results

export PYTHONUNBUFFERED=1

echo "Running sandwich model"
python -u scripts/sandwich_model.py \
    --data-feather Data/train_data_with_conditions.feather \
    --embedding-file Data/name_smiles_embedding_file.csv \
    --bottom-frac 0.5 \
    --middle-frac 0.2 \
    --top-frac 0.3 \
    --ham-chem DMMP

echo "All done."
