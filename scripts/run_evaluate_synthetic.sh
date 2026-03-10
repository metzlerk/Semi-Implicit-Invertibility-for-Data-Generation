#!/bin/bash
#SBATCH --job-name=eval_synthetic
#SBATCH --output=logs/eval_synthetic_%j.out
#SBATCH --error=logs/eval_synthetic_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

python scripts/4f-kjm-evaluatesynthetic-data.py
