#!/bin/bash
#SBATCH --job-name=train_diff
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/train_diff_%j.out

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

mkdir -p logs

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

# Do not copy feather files into Data/ to avoid race conditions between jobs.
# The training script will read TRAIN_FEATHER/TEST_FEATHER directly when provided.
if [[ -n "${TRAIN_FEATHER:-}" ]]; then
    echo "TRAIN_FEATHER will be read by the training script: ${TRAIN_FEATHER}"
fi
if [[ -n "${TEST_FEATHER:-}" ]]; then
    echo "TEST_FEATHER will be read by the training script: ${TEST_FEATHER}"
fi

BETA_START="${BETA_START:-0.001}"
BETA_END="${BETA_END:-0.2}"
NOISE_WEIGHT="${NOISE_WEIGHT:-0.8}"
SEPARATION_WEIGHT="${SEPARATION_WEIGHT:-0.2}"
SEPARATION_MARGIN="${SEPARATION_MARGIN:-5.0}"
MARGIN_MODE="${MARGIN_MODE:-fixed}"
MARGIN_QUANTILE="${MARGIN_QUANTILE:-0.2}"
MARGIN_MIN="${MARGIN_MIN:-0.0}"
MARGIN_MAX="${MARGIN_MAX:-10.0}"
SWD_WEIGHT="${SWD_WEIGHT:-0.0}"
SWD_PROJECTIONS="${SWD_PROJECTIONS:-64}"
LOCAL_ALIGN_WEIGHT="${LOCAL_ALIGN_WEIGHT:-0.0}"
LOCAL_ALIGN_K="${LOCAL_ALIGN_K:-5}"
MODEL_TAG="${MODEL_TAG:-}"

python scripts/train_latent_diffusion.py \
    --beta-start "${BETA_START}" \
    --beta-end "${BETA_END}" \
    --noise-weight "${NOISE_WEIGHT}" \
    --separation-weight "${SEPARATION_WEIGHT}" \
    --separation-margin "${SEPARATION_MARGIN}" \
    --margin-mode "${MARGIN_MODE}" \
    --margin-quantile "${MARGIN_QUANTILE}" \
    --margin-min "${MARGIN_MIN}" \
    --margin-max "${MARGIN_MAX}" \
    --swd-weight "${SWD_WEIGHT}" \
    --swd-projections "${SWD_PROJECTIONS}" \
    --local-align-weight "${LOCAL_ALIGN_WEIGHT}" \
    --local-align-k "${LOCAL_ALIGN_K}" \
    ${MODEL_TAG:+--model-tag "${MODEL_TAG}"}
