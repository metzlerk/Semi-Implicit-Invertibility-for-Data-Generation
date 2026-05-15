#!/bin/bash
#SBATCH --job-name=mlp_selected
#SBATCH --output=logs/mlp_selected_%j.out
#SBATCH --error=logs/mlp_selected_%j.err
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
mkdir -p logs models results

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

# use selector to get best combo (use existing plot_data CSV if summary CSV missing)
eval "$(python scripts/select_best_mlp_combo.py --csv plot_data_full_evaluation_metrics.csv | sed -n 's/^/export /p')"

case "${selected_std_label}" in
  std1.0) SYN_SPECTRA="results/generated_spectra_std1.0.npy"; SYN_LABELS="results/generated_labels_std1.0.npy" ;;
  std1.5) SYN_SPECTRA="results/generated_spectra_std1.5.npy"; SYN_LABELS="results/generated_labels_std1.5.npy" ;;
  std2.0) SYN_SPECTRA="results/generated_spectra_std2.0.npy"; SYN_LABELS="results/generated_labels_std2.0.npy" ;;
  *) SYN_SPECTRA=""; SYN_LABELS="" ;;
esac

MODEL_PATH="models/mlp_${selected_std_label}_r${selected_real_ratio}.joblib"
OUT_CSV="results/eval_mlp_selected_${selected_std_label}_r${selected_real_ratio}.csv"

python scripts/train_mlp_and_eval.py \
  --real-feather Data/train_data.feather \
  --test-feather Data/test_data.feather \
  ${SYN_SPECTRA:+--synthetic-spectra "${SYN_SPECTRA}"} \
  ${SYN_LABELS:+--synthetic-labels "${SYN_LABELS}"} \
  --std-label "${selected_std_label}" \
  --ratio "${selected_real_ratio}" \
  --save-model "${MODEL_PATH}" \
  --out-csv "${OUT_CSV}"
