#!/bin/bash
#SBATCH --job-name=train_decoder
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_decoder_%j.out

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "${CONDA_ENV}" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV}"
fi

LATENTS_PATH="${LATENTS_PATH:-results/autoencoder_train_latent.npy}"
TRAIN_FEATHER="${TRAIN_FEATHER:-Data/clean_split_1_of_5.feather}"
OUT_PATH="${OUT_PATH:-models/decoder_split.pth}"

python3 scripts/train_decoder.py --latents "${LATENTS_PATH}" --train-feather "${TRAIN_FEATHER}" --out "${OUT_PATH}"
