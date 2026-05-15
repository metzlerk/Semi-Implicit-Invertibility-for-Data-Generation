#!/bin/bash
#SBATCH --job-name=mlp_realonly
#SBATCH --output=logs/mlp_realonly_%j.out
#SBATCH --error=logs/mlp_realonly_%j.err
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
mkdir -p logs models results

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

MODEL_PATH="models/mlp_realonly.joblib"
OUT_CSV="results/eval_mlp_realonly.csv"

python scripts/train_mlp_and_eval.py \
  --real-feather Data/train_data.feather \
  --test-feather Data/test_data.feather \
  --ratio 1.0 \
  --save-model "${MODEL_PATH}" \
  --out-csv "${OUT_CSV}"
