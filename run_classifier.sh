#!/bin/bash
#SBATCH --job-name=gen_and_train
#SBATCH --output=gen_and_train_%j.out
#SBATCH --error=gen_and_train_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4

export PYTHONUNBUFFERED=1

echo "Step 1: Generating synthetic data"
python -u generate_synthetic.py

echo "Step 2: Training classifier"
python -u train_mlp_and_eval.py \
    --real-feather Data/train_data.feather \
    --test-feather Data/clean_split_5_of_5.feather \
    --synthetic-feather synthetic_high_temp.feather \
    --ratio 0.5 \
    --save-model models/mlp_with_synth.pkl \
    --out-csv results/mlp_with_synth.csv


echo "All done."
