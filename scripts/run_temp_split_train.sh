#!/bin/bash
#SBATCH --job-name=mlp_temp_split
#SBATCH --output=logs/mlp_temp_split.%j.out
#SBATCH --error=logs/mlp_temp_split.%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

IN_FILE=/home/kjmetzler/scratch/train_data_with_conditions.feather
OUT_TRAIN=/home/kjmetzler/scratch/train_temp_bottom80.feather
OUT_TEST=/home/kjmetzler/scratch/test_temp_top20.feather

python3 scripts/split_by_temperature.py --in-file "$IN_FILE" --out-train "$OUT_TRAIN" --out-test "$OUT_TEST" --temp-col TemperatureKelvin --train-quantile 0.8

python3 scripts/train_mlp_and_eval.py \
  --real-feather "$OUT_TRAIN" \
  --test-feather "$OUT_TEST" \
  --std-label realonly_temp_split \
  --ratio 1.0 \
  --save-model models/mlp_realonly_temp_split.joblib \
  --out-csv results/eval_mlp_realonly_temp_split.csv

echo "done"
