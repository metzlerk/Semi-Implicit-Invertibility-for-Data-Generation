#!/bin/bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -t 23:00:00
#SBATCH --mem 64G
#SBATCH --job-name="MNIST1D Diffusion Experiments"
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kjmetzler@wpi.edu

source /home/kjmetzler/anaconda3/bin/activate
conda activate venv

# Install wandb if not already installed
pip install wandb --quiet

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
python scripts/diffusion_mnist1d_experiments.py
