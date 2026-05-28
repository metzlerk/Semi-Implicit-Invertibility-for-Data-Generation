#!/bin/bash
# Submit SLURM jobs to train 5 diffusion models, one per 20% temperature split.

set -euo pipefail
ROOT=$(pwd)
for i in 1 2 3 4 5; do
    echo "Preparing diffusion job for split ${i}"
    # point to split-specific latent files (avoid overwriting a shared filename)
    TRAIN_LATENTS_PATH="$(pwd)/results/autoencoder_train_latent_${i}.npy"
    TEST_LATENTS_PATH="$(pwd)/results/autoencoder_test_latent_${i}.npy"
    if [[ ! -f "${TRAIN_LATENTS_PATH}" ]]; then
        echo "Missing ${TRAIN_LATENTS_PATH}"; exit 1
    fi

    MODEL_TAG="temp_split_${i}"
    BETA_END="0.2"
    TRAIN_FEATHER="Data/clean_split_${i}_of_5.feather"
    TEST_FEATHER="Data/clean_test_data.feather"

    # Build export string
    if [[ -n "${CONDA_ENV:-}" ]]; then
        EXPORT_PREFIX="CONDA_ENV=${CONDA_ENV},"
    else
        EXPORT_PREFIX=""
    fi
    EXPORT_VARS="${EXPORT_PREFIX}MODEL_TAG=${MODEL_TAG},BETA_END=${BETA_END},TRAIN_FEATHER=${TRAIN_FEATHER},TEST_FEATHER=${TEST_FEATHER},TRAIN_LATENTS_PATH=${TRAIN_LATENTS_PATH},TEST_LATENTS_PATH=${TEST_LATENTS_PATH}"
    sbatch --export=${EXPORT_VARS} scripts/run_train_latent_diffusion.sh
    sleep 0.5
done

echo "Submitted diffusion jobs for splits 1-5. Check squeue." 
