#!/bin/bash
#SBATCH --job-name=mlp_train
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/train_mlp_%j.out

set -euo pipefail
ROOT_DIR=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-base}"
if [[ -n "$CONDA_ENV" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    conda activate "$CONDA_ENV"
fi

REAL_FEATHER="${REAL_FEATHER:-Data/train_data.feather}"
TEST_FEATHER="${TEST_FEATHER:-Data/test_data.feather}"
SYN_FEATHER="${SYN_FEATHER:-}" # if provided
SYN_SPECTRA="${SYN_SPECTRA:-}"
SYN_LABELS="${SYN_LABELS:-}"
RATIO="${RATIO:-0.4}"
SAVE_MODEL="${SAVE_MODEL:-models/mlp_split.joblib}"
OUT_CSV="${OUT_CSV:-results/mlp_split_results.csv}"

CMD=(python3 scripts/train_mlp_and_eval.py --real-feather "$REAL_FEATHER" --test-feather "$TEST_FEATHER" --ratio "$RATIO" --save-model "$SAVE_MODEL" --out-csv "$OUT_CSV" --random-state 42)
if [[ -n "$SYN_FEATHER" ]]; then
    CMD+=(--synthetic-feather "$SYN_FEATHER")
elif [[ -n "$SYN_SPECTRA" && -n "$SYN_LABELS" ]]; then
    CMD+=(--synthetic-spectra "$SYN_SPECTRA" --synthetic-labels "$SYN_LABELS")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Done training: $SAVE_MODEL"
