#!/bin/bash
#SBATCH --job-name=gen_vae
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/gen_vae_%j.out

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "Generating VAE samples and creating comparison plots..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python3)"
echo ""

python3 scripts/generate_vae_samples.py

echo ""
echo "VAE generation complete!"
echo "Time: $(date)"
