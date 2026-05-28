#!/bin/bash
set -euo pipefail
# Re-submit diffusion jobs for any splits missing their best checkpoint until all complete.

ROOT=$(pwd)
CONDA_ENV=${CONDA_ENV:-base}
SLEEP_POLL=${SLEEP_POLL:-30}

splits=(1 2 3 4 5)
model_template="models/diffusion_latent_separated_beta0.20_temp_split_%d_best.pt"

submit_split() {
  i=$1
  TRAIN_LATENTS_PATH="$ROOT/results/autoencoder_train_latent_${i}.npy"
  TEST_LATENTS_PATH="$ROOT/results/autoencoder_test_latent_${i}.npy"
  TRAIN_FEATHER="$ROOT/Data/clean_split_${i}_of_5.feather"
  TEST_FEATHER="$ROOT/Data/clean_test_data.feather"
  MODEL_TAG="temp_split_${i}"
  BETA_END="0.2"

  if [[ ! -f "${TRAIN_LATENTS_PATH}" ]]; then
    echo "Missing latents for split ${i}: ${TRAIN_LATENTS_PATH}"; return 1
  fi

  # avoid duplicate submissions: check squeue for train_diff_split_<i>
  if squeue -u "$USER" -o "%j" | grep -q "train_diff_split_${i}"; then
    echo "Job for split ${i} already queued/running; skipping submit"
    return 0
  fi

  EXPORT_VARS="CONDA_ENV=${CONDA_ENV},MODEL_TAG=${MODEL_TAG},BETA_END=${BETA_END},TRAIN_FEATHER=${TRAIN_FEATHER},TEST_FEATHER=${TEST_FEATHER},TRAIN_LATENTS_PATH=${TRAIN_LATENTS_PATH},TEST_LATENTS_PATH=${TEST_LATENTS_PATH}"
  echo "Submitting diffusion job for split ${i}"
  sbatch --job-name=train_diff_split_${i} --export=${EXPORT_VARS} scripts/run_train_latent_diffusion.sh
}

all_done() {
  for i in "${splits[@]}"; do
    model=$(printf "$model_template" "$i")
    if [[ ! -f "$model" ]]; then
      return 1
    fi
  done
  return 0
}

echo "Starting ensure_diffusions_complete loop (CONDA_ENV=${CONDA_ENV})"
mkdir -p logs
while true; do
  missing=()
  for i in "${splits[@]}"; do
    model=$(printf "$model_template" "$i")
    if [[ ! -f "$model" ]]; then
      missing+=("$i")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "All diffusion best checkpoints present. Done."
    exit 0
  fi

  echo "Missing splits: ${missing[*]}"
  for i in "${missing[@]}"; do
    submit_split "$i" || true
  done

  # Wait and poll until no train_diff_split jobs in queue
  echo "Waiting for jobs to start/finish... polling every ${SLEEP_POLL}s"
  while squeue -u "$USER" -o "%j" | grep -q "train_diff_split_"; do
    sleep ${SLEEP_POLL}
  done

  # small pause before re-checking
  sleep 5
done
