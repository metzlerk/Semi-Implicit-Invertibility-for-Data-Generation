#!/bin/bash
#SBATCH --job-name=test_enc_dec
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/test_enc_dec_%j.out

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

# Install missing dependency
pip install gputil --quiet

python scripts/test_encoder_decoder.py
