#!/bin/bash
#SBATCH --job-name=eval_norm
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/eval_norm_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/eval_norm_%j.err

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "Starting normalized diffusion structure evaluation..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

python scripts/analyze_diffusion_structure.py --model normalized --all --n-samples 1000

echo ""
echo "Evaluation complete!"
echo "Time: $(date)"
