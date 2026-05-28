#!/bin/bash
# Generate synthetic data from a specified diffusion+decoder and run classifier experiment.
# Usage: scripts/run_experiment_generate_and_eval.sh <diffusion_checkpoint> <decoder_checkpoint> <out_prefix>

set -euo pipefail
DIFF_CKPT=${1}
DECODER_CKPT=${2}
PREFIX=${3:-hot_aug}

mkdir -p results
# prepare train/test: combine splits 1-4 as coldest 80% train, split 5 as test
python3 - <<'PY'
import pandas as pd
files = [f'Data/clean_split_{i}_of_5.feather' for i in (1,2,3,4)]
dfs = [pd.read_feather(f) for f in files]
train = pd.concat(dfs, ignore_index=True)
train.to_feather('Data/real_train_80.feather')
test = pd.read_feather('Data/clean_split_5_of_5.feather')
test.to_feather('Data/real_test_20.feather')
print('Prepared Data/real_train_80.feather and Data/real_test_20.feather')
PY

# generate synthetic data (use sigma=1.5 and samples per class large enough)
SAMPLES_PER_CLASS=1000
python3 scripts/generate_and_decode_full.py --model-path "${DIFF_CKPT}" --decoder-path "${DECODER_CKPT}" \
    --output-prefix "${PREFIX}" --samples-per-class ${SAMPLES_PER_CLASS} --sigma 1.5

# Train classifier replacing 60% with synthetic -> set ratio=0.4
python3 scripts/train_mlp_and_eval.py --real-feather Data/real_train_80.feather --test-feather Data/real_test_20.feather \
    --synthetic-spectra results/${PREFIX}_spectra.npy --synthetic-labels results/${PREFIX}_labels.npy --ratio 0.4 --out-csv results/${PREFIX}_classifier_results.csv

echo "Experiment complete. Results -> results/${PREFIX}_classifier_results.csv"
