#!/bin/bash
#SBATCH --job-name=confusion
#SBATCH --output=confusion.out
#SBATCH --error=confusion.err
#SBATCH --time=01:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module load python/3.10

# Activate your environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yourenv

python confusion_matrix_and_accuracy.py \
    --model best_classifier.pt \
    --test-feather Data/clean_split_5_of_5.feather \
    --out-prefix results/mlp_eval
