#!/bin/bash
#SBATCH --job-name=inspect_encoder
#SBATCH --partition=short
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/inspect_encoder_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/inspect_encoder_%j.err

set -euo pipefail

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
mkdir -p /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs

CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "${CONDA_ENV}" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    conda activate "${CONDA_ENV}"
fi

CHECKPOINT="${CHECKPOINT:-/home/kjmetzler/ChemicalDataGeneration/models/trained_models/spectrum/hypertuned_encoder.pth}"

python scripts/inspect_encoder_checkpoint.py --checkpoint "${CHECKPOINT}"
