#!/bin/bash
#SBATCH --job-name=eval_synthetic
#SBATCH --output=logs/eval_synthetic_%j.out
#SBATCH --error=logs/eval_synthetic_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

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

CLASSIFIERS="${CLASSIFIERS:-RandomForest,MLP}"
TUNE="${TUNE:-false}"
TUNE_NUM_POINTS="${TUNE_NUM_POINTS:-20}"
TUNE_RATIO="${TUNE_RATIO:-0.5}"
TUNE_ITERATIONS="${TUNE_ITERATIONS:-24}"
TUNE_CV="${TUNE_CV:-3}"
TUNE_JOBS="${TUNE_JOBS:--1}"
TUNED_SUFFIX="${TUNED_SUFFIX:-_tuned}"
RANDOM_STATE="${RANDOM_STATE:-42}"
CSV_PATH="${CSV_PATH:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/eval_synthetic_metrics.csv}"
SYN_STD1_SPECTRA="${SYN_STD1_SPECTRA:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std1.0.npy}"
SYN_STD1_LABELS="${SYN_STD1_LABELS:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std1.0.npy}"
SYN_STD1P5_SPECTRA="${SYN_STD1P5_SPECTRA:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std1.5.npy}"
SYN_STD1P5_LABELS="${SYN_STD1P5_LABELS:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std1.5.npy}"
SYN_STD2_SPECTRA="${SYN_STD2_SPECTRA:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_spectra_std2.0.npy}"
SYN_STD2_LABELS="${SYN_STD2_LABELS:-/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/results/generated_labels_std2.0.npy}"
CATE_PATH="${CATE_PATH:-/home/kjmetzler/scratch/CARL/universal_generator/_synthetic_test_spectra.feather}"
SKIP_CATE="${SKIP_CATE:-false}"

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

python scripts/4f-kjm-evaluatesynthetic-data.py "${EVAL_ARGS[@]}"
