#!/bin/bash
#SBATCH --job-name=regen_test_latent
#SBATCH --output=logs/regen_test_latent_%j.out
#SBATCH --error=logs/regen_test_latent_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

source /home/kjmetzler/miniconda3/etc/profile.d/conda.sh
conda activate base
export LD_LIBRARY_PATH=/home/kjmetzler/miniconda3/lib:${LD_LIBRARY_PATH:-}

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
mkdir -p logs

python3 scripts/regenerate_test_latents.py
