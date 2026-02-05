#!/bin/bash
#SBATCH --job-name=ims_bijection
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/ims_bijection_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/ims_bijection_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Load modules
module load python/3.9
module load cuda/11.7

# Create logs directory if it doesn't exist
mkdir -p /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs

# Print some info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run the bijection-based IMS spectra experiment
cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
python scripts/bijection_ims_experiments.py

echo "Finished at: $(date)"
