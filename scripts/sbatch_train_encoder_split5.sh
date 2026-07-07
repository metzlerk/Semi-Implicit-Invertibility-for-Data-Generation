#!/bin/bash
#SBATCH -p short
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --mem 250G
#SBATCH --job-name=encoder_split5
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/encoder_split5_%j.out
#SBATCH --error=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/encoder_split5_%j.err

set -euo pipefail

echo "Preparing encoder split5 data"
bash /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/encoder_split5/setup_files.sh

echo "Activating environment"
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell=bash)"
  micromamba activate data_gen_venv
fi

echo "Starting encoder training"
python3 /home/kjmetzler/ChemicalDataGeneration/models/run_encoder.py --target_embedding ChemNet --n_layers 9 --layers_string nine_layer_ --correct_label_prob 1.0

echo "Encoder training script finished"
