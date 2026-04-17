#!/bin/bash
#SBATCH --job-name=final_pca
#SBATCH --output=logs/final_pca_%j.out
#SBATCH --error=logs/final_pca_%j.err
#SBATCH --time=00:20:00
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

python scripts/gen_pca_final.py
