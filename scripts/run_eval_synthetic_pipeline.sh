#!/bin/bash
#SBATCH --job-name=eval_syn_pipeline
#SBATCH --output=logs/eval_syn_pipeline_%j.out
#SBATCH --error=logs/eval_syn_pipeline_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

CSV_PATH="/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/eval_synthetic_metrics.csv"
OUT_DIR="/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results"

if [[ -x ".venv_eval/bin/python" ]]; then
    PYTHON_BIN=".venv_eval/bin/python"
else
    PYTHON_BIN="python"
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Running synthetic evaluation..."
"${PYTHON_BIN}" scripts/4f-kjm-evaluatesynthetic-data.py --csv-path "${CSV_PATH}"

echo "Generating heatmaps..."
"${PYTHON_BIN}" scripts/plots2.py --csv-path "${CSV_PATH}" --out-dir "${OUT_DIR}"

echo "Done. Outputs:"
echo "  ${CSV_PATH}"
echo "  ${OUT_DIR}/heatmap_rf_accuracy.png"
echo "  ${OUT_DIR}/heatmap_mlp_accuracy.png"
echo "  ${OUT_DIR}/heatmap_rf_delta_accuracy.png"
echo "  ${OUT_DIR}/heatmap_mlp_delta_accuracy.png"
