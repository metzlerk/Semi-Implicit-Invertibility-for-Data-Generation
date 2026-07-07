#!/bin/bash
#SBATCH --job-name=gen_synth
#SBATCH --output=/logs/gen_synth_%j.out
#SBATCH --error=/logs/gen_synth_%j.err
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Force Python to flush output immediately
export PYTHONUNBUFFERED=1

# Load modules if needed
# module load cuda/12.0

echo "Starting synthetic generation job on $(hostname)"
python -u generate_synthetic.py
echo "Job complete."
