#!/bin/bash
#SBATCH --job-name=build_latents
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/build_latents_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/build_latents_%j.err

set -euo pipefail

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

CHEMGEN_MODELS_PATH="${CHEMGEN_MODELS_PATH:-/home/kjmetzler/ChemicalDataGeneration/models}"
ENCODER_PATH="${ENCODER_PATH:-/home/kjmetzler/ChemicalDataGeneration/models/trained_models/spectrum/new_hypertuning_results_20251009_100119/baseline_exact_encoder.pth}"
TRAIN_DATA="${TRAIN_DATA:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/train_data.feather}"
TEST_DATA="${TEST_DATA:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/test_data.feather}"
EMBEDDING_FILE="${EMBEDDING_FILE:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/name_smiles_embedding_file.csv}"
OUT_TRAIN="${OUT_TRAIN:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/autoencoder_train_latent.npy}"
OUT_TEST="${OUT_TEST:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/autoencoder_test_latent.npy}"
BATCH_SIZE="${BATCH_SIZE:-256}"
START_IDX="${START_IDX:-2}"
STOP_IDX="${STOP_IDX:--9}"
SAVE_CSV="${SAVE_CSV:-false}"

ARGS=(
    --chemgen-models-path "${CHEMGEN_MODELS_PATH}"
    --encoder-path "${ENCODER_PATH}"
    --train-data "${TRAIN_DATA}"
    --test-data "${TEST_DATA}"
    --embedding-file "${EMBEDDING_FILE}"
    --output-train "${OUT_TRAIN}"
    --output-test "${OUT_TEST}"
    --batch-size "${BATCH_SIZE}"
    --start-idx "${START_IDX}"
    --stop-idx "${STOP_IDX}"
)

if [[ "${SAVE_CSV}" == "true" ]]; then
    ARGS+=(--save-csv)
fi

python scripts/build_autoencoder_latents.py "${ARGS[@]}"
