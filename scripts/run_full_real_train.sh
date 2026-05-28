#!/bin/bash
#SBATCH --job-name=mlp_full_real
#SBATCH --output=logs/mlp_full_real.%j.out
#SBATCH --error=logs/mlp_full_real.%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

python3 scripts/train_mlp_and_eval.py \
  --real-feather Data/train_data.feather \
  --test-feather Data/test_data.feather \
  --std-label selected \
  --ratio 1.0 \
  --save-model models/mlp_realonly_full.joblib \
  --out-csv results/eval_mlp_realonly_full.csv

echo "done"
