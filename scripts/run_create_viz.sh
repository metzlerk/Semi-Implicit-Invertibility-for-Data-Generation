#!/bin/bash
#SBATCH --job-name=create_viz
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/create_viz_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/create_viz_%j.err

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "Creating visualizations..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo""

echo "Creating eye tests for all 8 chemicals..."
python3 scripts/create_eye_test_all.py

echo ""
echo "Creating per-chemical PCA plots..."
python3 scripts/create_per_chemical_pca.py

echo ""
echo "Complete!"
echo "Time: $(date)"
