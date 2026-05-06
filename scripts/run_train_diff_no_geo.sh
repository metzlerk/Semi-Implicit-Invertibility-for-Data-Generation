#!/bin/bash
#SBATCH --job-name=train_diff_no_geo
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs/train_diff_no_geo_%j.out

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
mkdir -p /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/logs

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

echo "Training diffusion WITHOUT geometry regularization (ablation)..."
echo "Time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python3)"
echo ""

BETA_START="${BETA_START:-0.001}"
BETA_END="${BETA_END:-0.2}"
MODEL_TAG="${MODEL_TAG:-no_geo}"

python3 scripts/train_latent_diffusion.py \
    --beta-start "${BETA_START}" \
    --beta-end "${BETA_END}" \
    --separation-weight 0.0 \
    --swd-weight 0.0 \
    --local-align-weight 0.0 \
    --model-tag "${MODEL_TAG}"

echo ""
echo "Ablation training complete!"
echo "Time: $(date)"
