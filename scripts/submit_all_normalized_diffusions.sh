#!/bin/bash
# Submit SLURM jobs to train 5 normalized diffusion models, one per 20% temperature split.

set -euo pipefail
ROOT_DIR=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
cd "$ROOT_DIR"

for i in 1 2 3 4 5; do
    echo "Preparing normalized diffusion job for split ${i}"

    TRAIN_LATENTS_PATH="$ROOT_DIR/results/autoencoder_train_latent_${i}.npy"
    TEST_LATENTS_PATH="$ROOT_DIR/results/autoencoder_test_latent_${i}.npy"
    TRAIN_FEATHER="$ROOT_DIR/Data/clean_split_${i}_of_5.feather"
    TEST_FEATHER="$ROOT_DIR/Data/clean_test_data.feather"

    if [[ ! -f "$TRAIN_LATENTS_PATH" ]]; then
        echo "Missing ${TRAIN_LATENTS_PATH}"
        exit 1
    fi
    if [[ ! -f "$TEST_LATENTS_PATH" ]]; then
        echo "Missing ${TEST_LATENTS_PATH}"
        exit 1
    fi
    if [[ ! -f "$TRAIN_FEATHER" ]]; then
        echo "Missing ${TRAIN_FEATHER}"
        exit 1
    fi
    if [[ ! -f "$TEST_FEATHER" ]]; then
        echo "Missing ${TEST_FEATHER}"
        exit 1
    fi

    MODEL_TAG="temp_split_${i}"
    BETA_END="0.2"

    if [[ -n "${CONDA_ENV:-}" ]]; then
        EXPORT_PREFIX="CONDA_ENV=${CONDA_ENV},"
    else
        EXPORT_PREFIX=""
    fi

    EXPORT_VARS="${EXPORT_PREFIX}MODEL_TAG=${MODEL_TAG},BETA_END=${BETA_END},TRAIN_FEATHER=${TRAIN_FEATHER},TEST_FEATHER=${TEST_FEATHER},TRAIN_LATENTS_PATH=${TRAIN_LATENTS_PATH},TEST_LATENTS_PATH=${TEST_LATENTS_PATH}"
    sbatch --job-name=train_norm_diff_${i} --export=${EXPORT_VARS} scripts/run_train_normalized_diffusion.sh
    sleep 0.5
done

echo "Submitted normalized diffusion jobs for splits 1-5. Check squeue and logs/train_diff_norm_*."
