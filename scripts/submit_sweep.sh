#!/usr/bin/env bash
# Submit a simple grid sweep to Slurm by varying BETA_END, NOISE_WEIGHT, and SEPARATION_WEIGHT.
# Usage: ./scripts/submit_sweep.sh

set -euo pipefail

# Define grid (edit as needed)
BETA_ENDS=(0.02 0.2)
NOISE_WEIGHTS=(0.6 0.8)
SEPARATION_WEIGHTS=(0.1 0.2)

# Optional common settings
CONDA_ENV="${CONDA_ENV:-base}"
PARTITION="short"

echo "Submitting sweep: ${#BETA_ENDS[@]} x ${#NOISE_WEIGHTS[@]} x ${#SEPARATION_WEIGHTS[@]} jobs"

for be in "${BETA_ENDS[@]}"; do
  for nw in "${NOISE_WEIGHTS[@]}"; do
    for sw in "${SEPARATION_WEIGHTS[@]}"; do
      echo "Submitting: BETA_END=${be}, NOISE_WEIGHT=${nw}, SEPARATION_WEIGHT=${sw}"
      sbatch --export=ALL,BETA_END=${be},NOISE_WEIGHT=${nw},SEPARATION_WEIGHT=${sw},CONDA_ENV=${CONDA_ENV} scripts/run_train_latent_diffusion.sh
      sleep 0.05
    done
  done
done

echo "Sweep submission complete."
