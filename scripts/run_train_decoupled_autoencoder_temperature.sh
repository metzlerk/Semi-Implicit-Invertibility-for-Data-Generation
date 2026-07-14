#!/bin/bash
#SBATCH --job-name=dec_auto_temp
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_decoupled_autoencoder_temperature_%j.out

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

CONDA_ENV="${CONDA_ENV:-}"
if [[ -n "${CONDA_ENV}" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
    conda activate "${CONDA_ENV}"
fi

TRAIN_DATA="${TRAIN_DATA:-Data/train_data_with_conditions.feather}"
EMBEDDING_FILE="${EMBEDDING_FILE:-Data/name_smiles_embedding_file.csv}"
OUT_PATH="${OUT_PATH:-models/decoupled_autoencoder_temperature.pth}"

python3 scripts/train_decoupled_autoencoder_temperature.py \
    --train-data "${TRAIN_DATA}" \
    --embedding-file "${EMBEDDING_FILE}" \
    --out "${OUT_PATH}" \
    --batch-size "${BATCH_SIZE:-256}" \
    --epochs "${EPOCHS:-200}" \
    --lr "${LR:-1e-4}" \
    --n-layers "${N_LAYERS:-9}" \
    --latent-dim "${LATENT_DIM:-513}" \
    --temperature-scale "${TEMPERATURE_SCALE:-300.0}" \
    --recon-weight "${RECON_WEIGHT:-1.0}" \
    --chem-weight "${CHEM_WEIGHT:-1.0}" \
    --temp-weight "${TEMP_WEIGHT:-1.0}" \
    --val-fraction "${VAL_FRACTION:-0.1}" \
    --seed "${SEED:-42}"