#!/bin/bash
#SBATCH --job-name=gen_decode_full
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/gen_decode_full_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/gen_decode_full_%j.err

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "Generating samples and decoding to spectra..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

python3 scripts/generate_and_decode_full.py

echo ""
echo "Complete!"
echo "Time: $(date)"
