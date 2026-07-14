#!/bin/bash
#SBATCH --job-name=mlp_temp_split
#SBATCH --output=logs/mlp_temp_split.%j.out
#SBATCH --error=logs/mlp_temp_split.%j.err
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
mkdir -p logs models results scratch_splits

# Real-only temperature-extrapolation baseline:
# train on the cooler 80% of temperatures, test on the hottest 20%.
IN_FILE="${IN_FILE:-Data/train_data_with_conditions.feather}"
OUT_TRAIN="scratch_splits/train_temp_bottom80.feather"
OUT_TEST="scratch_splits/test_temp_top20.feather"

python3 scripts/split_by_temperature.py \
  --in-file "$IN_FILE" \
  --out-train "$OUT_TRAIN" \
  --out-test "$OUT_TEST" \
  --temp-col TemperatureKelvin \
  --train-quantile 0.8

python3 scripts/train_mlp_and_eval.py \
  --real-feather "$OUT_TRAIN" \
  --test-feather "$OUT_TEST" \
  --std-label realonly_temp_split \
  --ratio 1.0 \
  --save-model models/mlp_realonly_temp_split.pt \
  --out-csv results/eval_mlp_realonly_temp_split.csv

echo "done"
