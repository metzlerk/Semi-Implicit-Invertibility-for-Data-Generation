#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/compare_%j.out

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

# Install missing dependencies for ChemicalDataGeneration
pip install gputil dask --quiet

python scripts/compare_diffusion_gaussian.py
