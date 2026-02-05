#!/bin/bash
#SBATCH --job-name=decode_deb
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/decode_deb_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/decode_deb_%j.err

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

# Install missing dependencies
pip install gputil --quiet
pip install --upgrade pyarrow --quiet

echo "Decoding DEB latents to spectra..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

python scripts/decode_single_deb.py

echo ""
echo "Decoding complete!"
echo "Time: $(date)"
