#!/bin/bash
#SBATCH --job-name=mlp_full_60syn
#SBATCH --output=logs/mlp_full_60syn.%j.out
#SBATCH --error=logs/mlp_full_60syn.%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation


# User requested 60% synthetic -> desired real fraction in final set = 0.4
REAL_RATIO=0.4
SYN_FEATHER=results/synthetic_spectra_std1.5_DEB_DEM_DMMP_DPM_DtBP_JP8_MES_TEPO.feather

# Create a subsampled real training feather so that final combined set size
# equals the original full real-only training size (i.e., don't increase total)
SUB_REAL_FEATHER=results/train_40real_subsampled.feather
python3 - <<PY
import pandas as pd, math
R = float("${REAL_RATIO}")
df = pd.read_feather('Data/train_data.feather')
N = len(df)
n_real_sub = int(math.floor(R * N))
print(f'subsampling {n_real_sub} of {N} real rows (R={R})')
df.sample(n=n_real_sub, random_state=42).reset_index(drop=True).to_feather("${SUB_REAL_FEATHER}")
PY

python3 scripts/train_mlp_and_eval.py \
  --real-feather "$SUB_REAL_FEATHER" \
  --test-feather Data/test_data.feather \
  --synthetic-feather "$SYN_FEATHER" \
  --std-label std1.5 \
  --ratio "$REAL_RATIO" \
  --save-model models/mlp_std1.5_r0.4_full.joblib \
  --out-csv results/eval_mlp_std1.5_r0.4_full.csv \
  --mlp-hidden-sizes "500,250,100" \
  --mlp-max-iter 1000

echo "done"
