#!/bin/bash
#SBATCH --job-name=train_diff_no_geo
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/train_diff_no_geo_%j.out

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "Training diffusion WITHOUT geometry regularization (ablation)..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python3)"
echo ""

# Run training with SEPARATION_WEIGHT=0
python3 -c "
import os
import sys

# Read the training script
with open('scripts/train_latent_diffusion.py', 'r') as f:
    code = f.read()

# Replace SEPARATION_WEIGHT with 0
code = code.replace('SEPARATION_WEIGHT = 0.2', 'SEPARATION_WEIGHT = 0.0')

# Replace output model name
code = code.replace(\"'diffusion_latent_best.pt'\", \"'diffusion_no_geo_best.pt'\")
code = code.replace(\"'diffusion_latent_epoch_\", \"'diffusion_no_geo_epoch_\")

# Execute
exec(code)
"

echo ""
echo "Ablation training complete!"
echo "Time: $(date)"
