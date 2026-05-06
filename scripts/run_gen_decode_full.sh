#!/bin/bash
#SBATCH --job-name=gen_decode_full
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/gen_decode_full_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/gen_decode_full_%j.err

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

echo "Generating samples and decoding to spectra..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo ""

MODEL_PATH="${MODEL_PATH:-models/diffusion_latent_normalized_best.pt}"
DECODER_PATH="${DECODER_PATH:-models/autoencoder_separated.pth}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-full_generated}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-500}"
DDIM_STEPS="${DDIM_STEPS:-100}"
SIGMA="${SIGMA:-1.0}"
SIGMA_BY_CLASS="${SIGMA_BY_CLASS:-}"
SIGMA_MODE="${SIGMA_MODE:-fixed}"
BUDGET="${BUDGET:-}"
BUDGET_MIN="${BUDGET_MIN:-1.0}"
BUDGET_MAX="${BUDGET_MAX:-50.0}"
SIGMA_MIN="${SIGMA_MIN:-1.0}"
SIGMA_MAX="${SIGMA_MAX:-2.0}"

GEN_ARGS=(
    --model-path "${MODEL_PATH}"
    --decoder-path "${DECODER_PATH}"
    --output-prefix "${OUTPUT_PREFIX}"
    --samples-per-class "${SAMPLES_PER_CLASS}"
    --ddim-steps "${DDIM_STEPS}"
    --sigma "${SIGMA}"
    --sigma-mode "${SIGMA_MODE}"
    --budget-min "${BUDGET_MIN}"
    --budget-max "${BUDGET_MAX}"
    --sigma-min "${SIGMA_MIN}"
    --sigma-max "${SIGMA_MAX}"
)

if [[ -n "${SIGMA_BY_CLASS}" ]]; then
    GEN_ARGS+=(--sigma-by-class "${SIGMA_BY_CLASS}")
fi
if [[ -n "${BUDGET}" ]]; then
    GEN_ARGS+=(--budget "${BUDGET}")
fi

python3 scripts/generate_and_decode_full.py "${GEN_ARGS[@]}"

echo ""
echo "Complete!"
echo "Time: $(date)"
