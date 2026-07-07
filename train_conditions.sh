#!/bin/bash
#SBATCH --job-name=conditional_mlp
#SBATCH --output=logs/conditional_mlp_%j.out
#SBATCH --error=logs/conditional_mlp_%j.err
#SBATCH --partition=short
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Move to project directory
cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

# Ensure directories exist
mkdir -p logs models results

# -----------------------------
# Activate conda base environment
# -----------------------------
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda activate base

# -----------------------------
# Run your training script
# -----------------------------
python -u train-conditions.py \
    --encoder-out models/conditional_encoder.pt \
    --decoder-out models/conditional_decoder.pt
