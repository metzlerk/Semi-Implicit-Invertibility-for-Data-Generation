#!/bin/bash
#SBATCH --job-name=viz_deb
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/viz_deb_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/viz_deb_%j.err

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "Visualizing DEB spectra..."
echo "Time: $(date)"
echo ""

python scripts/visualize_single_deb_spectra.py

echo ""
echo "Visualization complete!"
echo "Time: $(date)"
