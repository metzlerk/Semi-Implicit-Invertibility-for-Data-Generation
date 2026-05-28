#!/bin/bash
#SBATCH --job-name=gen_split
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/gen_%j.out

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

DIFF_CKPT="${DIFF_CKPT:-models/diffusion_latent_separated_beta0.20_temp_split_5_best.pt}"
DECODER_CKPT="${DECODER_CKPT:-models/decoder_split_5.pth}"
PREFIX="${PREFIX:-gen_split}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-1000}"
SIGMA="${SIGMA:-1.5}"

echo "Using DIFF_CKPT=$DIFF_CKPT"
echo "Using DECODER_CKPT=$DECODER_CKPT"
echo "Output prefix: $PREFIX"

mkdir -p results

python3 scripts/generate_and_decode_full.py --model-path "$DIFF_CKPT" --decoder-path "$DECODER_CKPT" \
    --output-prefix "$PREFIX" --samples-per-class $SAMPLES_PER_CLASS --sigma $SIGMA

echo "Generation finished for $PREFIX"
