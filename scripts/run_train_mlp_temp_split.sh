#!/bin/bash
#SBATCH --job-name=mlp_temp_split
#SBATCH --output=logs/mlp_temp_split_%j.out
#SBATCH --error=logs/mlp_temp_split_%j.err
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
mkdir -p logs models results scratch_splits

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

TRAIN_COND="${TRAIN_COND:-$HOME/scratch/train_data_with_conditions.feather}"
TEST_COND="${TEST_COND:-$HOME/scratch/test_data_with_conditions.feather}"
SPLIT_TRAIN="scratch_splits/train_temp_split.feather"
SPLIT_TEST="scratch_splits/test_temp_split.feather"

python scripts/split_by_temperature.py --in-file "${TRAIN_COND}" --out-train "${SPLIT_TRAIN}" --out-test "${SPLIT_TEST}" --temp-col TemperatureKelvin --train-quantile 0.8
MODEL_PATH="models/mlp_realonly_temp.joblib"
OUT_CSV="results/eval_mlp_realonly_temp.csv"

python scripts/train_mlp_and_eval.py \
  --real-feather "${SPLIT_TRAIN}" \
  --test-feather "${SPLIT_TEST}" \
  --ratio 1.0 \
    --save-model "${MODEL_PATH}" \
  --out-csv "${OUT_CSV}"
