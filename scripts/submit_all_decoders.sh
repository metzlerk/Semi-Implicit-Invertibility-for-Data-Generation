#!/bin/bash
set -euo pipefail
for i in 1 2 3 4 5; do
    echo "Submitting decoder job for split ${i}"
    TRAIN_LATENTS_PATH="$(pwd)/results/autoencoder_train_latent_${i}.npy"
    if [[ ! -f "${TRAIN_LATENTS_PATH}" ]]; then
        echo "Missing ${TRAIN_LATENTS_PATH} - skipping"; continue
    fi
    TRAIN_FEATHER="Data/clean_split_${i}_of_5.feather"
    OUT_MODEL="models/decoder_split_${i}.pth"
    EXPORT_VARS="LATENTS_PATH=${TRAIN_LATENTS_PATH},TRAIN_FEATHER=${TRAIN_FEATHER},OUT_PATH=${OUT_MODEL}"
    if [[ -n "${CONDA_ENV:-}" ]]; then
        EXPORT_VARS="CONDA_ENV=${CONDA_ENV},${EXPORT_VARS}"
    fi
    sbatch --export=${EXPORT_VARS} scripts/run_train_decoder.sh
    sleep 0.5
done
echo "Submitted decoder jobs for available splits (skipped missing latents)."
