#!/bin/bash
#SBATCH --job-name=full_mlp_heatmap
#SBATCH --output=logs/full_mlp_heatmap_%j.out
#SBATCH --error=logs/full_mlp_heatmap_%j.err
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
mkdir -p logs results

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

OUT_CSV="results/full_dataset_heatmap.csv"

python scripts/build_full_dataset_heatmap_csv.py \
  --train-feather Data/train_data.feather \
  --test-feather Data/test_data.feather \
  --synthetic-spectra results/generated_spectra_std1.5.npy \
  --synthetic-labels results/generated_labels_std1.5.npy \
  --out-csv "${OUT_CSV}" \
  --step 1000

python scripts/plots2.py \
  --csv-path "${OUT_CSV}" \
  --out-dir results
