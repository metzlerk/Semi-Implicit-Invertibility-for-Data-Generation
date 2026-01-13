#!/bin/bash
#SBATCH --job-name=ims_diffusion_autoencoder_per_class
#SBATCH --output=logs/ims_diffusion_autoencoder_per_class_%j.out
#SBATCH --error=logs/ims_diffusion_autoencoder_per_class_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Load modules if needed
# module load python/3.9
# module load cuda/11.8

# Activate conda environment if you have one
# source ~/.bashrc
# conda activate your_env

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some info
echo "Starting IMS Diffusion with Autoencoder (Per-Class) experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Run the experiment
python scripts/diffusion_ims_autoencoder_per_class_experiments.py

echo ""
echo "End time: $(date)"
echo "Job completed!"
