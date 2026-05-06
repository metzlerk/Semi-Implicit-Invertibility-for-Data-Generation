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
mkdir -p logs

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

CSV_PATH="${CSV_PATH:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/eval_synthetic_metrics.csv}"
OUT_DIR="${OUT_DIR:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results}"

CLASSIFIERS="${CLASSIFIERS:-RandomForest,MLP}"
TUNE="${TUNE:-false}"
TUNE_NUM_POINTS="${TUNE_NUM_POINTS:-20}"
TUNE_RATIO="${TUNE_RATIO:-0.5}"
TUNE_ITERATIONS="${TUNE_ITERATIONS:-24}"
TUNE_CV="${TUNE_CV:-3}"
TUNE_JOBS="${TUNE_JOBS:--1}"
TUNED_SUFFIX="${TUNED_SUFFIX:-_tuned}"
RANDOM_STATE="${RANDOM_STATE:-42}"
SYN_STD1_SPECTRA="${SYN_STD1_SPECTRA:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std1.0.npy}"
SYN_STD1_LABELS="${SYN_STD1_LABELS:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std1.0.npy}"
SYN_STD1P5_SPECTRA="${SYN_STD1P5_SPECTRA:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std1.5.npy}"
SYN_STD1P5_LABELS="${SYN_STD1P5_LABELS:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std1.5.npy}"
SYN_STD2_SPECTRA="${SYN_STD2_SPECTRA:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std2.0.npy}"
SYN_STD2_LABELS="${SYN_STD2_LABELS:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std2.0.npy}"
CATE_PATH="${CATE_PATH:-/home/kjmetzler/scratch/CARL/universal_generator/_synthetic_test_spectra.feather}"
SKIP_CATE="${SKIP_CATE:-false}"

if [[ -x ".venv_eval/bin/python" ]]; then
    PYTHON_BIN=".venv_eval/bin/python"
else
    PYTHON_BIN="python"
fi

echo "Using Python: ${PYTHON_BIN}"
echo "Running synthetic evaluation..."
EVAL_ARGS=(
    --csv-path "${CSV_PATH}"
    --classifiers "${CLASSIFIERS}"
    --random-state "${RANDOM_STATE}"
    --synthetic-std1-spectra "${SYN_STD1_SPECTRA}"
    --synthetic-std1-labels "${SYN_STD1_LABELS}"
    --synthetic-std1p5-spectra "${SYN_STD1P5_SPECTRA}"
    --synthetic-std1p5-labels "${SYN_STD1P5_LABELS}"
    --synthetic-std2-spectra "${SYN_STD2_SPECTRA}"
    --synthetic-std2-labels "${SYN_STD2_LABELS}"
    --cate-path "${CATE_PATH}"
)
if [[ "${TUNE}" == "true" ]]; then
    EVAL_ARGS+=(
        --tune
        --tune-num-points "${TUNE_NUM_POINTS}"
        --tune-ratio "${TUNE_RATIO}"
        --tune-iterations "${TUNE_ITERATIONS}"
        --tune-cv "${TUNE_CV}"
        --tune-jobs "${TUNE_JOBS}"
        --tuned-suffix "${TUNED_SUFFIX}"
    )
fi
if [[ "${SKIP_CATE}" == "true" ]]; then
    EVAL_ARGS+=(--skip-cate)
fi
"${PYTHON_BIN}" scripts/4f-kjm-evaluatesynthetic-data.py "${EVAL_ARGS[@]}"

echo "Generating heatmaps..."
"${PYTHON_BIN}" scripts/plots2.py --csv-path "${CSV_PATH}" --out-dir "${OUT_DIR}" \
    --classifiers "${CLASSIFIERS}"

echo "Done. Outputs:"
echo "  ${CSV_PATH}"
echo "  ${OUT_DIR}/heatmap_*_accuracy.png"
echo "  ${OUT_DIR}/heatmap_*_delta_accuracy.png"
