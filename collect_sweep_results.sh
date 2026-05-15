#!/bin/bash
# Collect and summarize sweep results

set -euo pipefail

REPO_DIR="/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation"
SUMMARY_FILE="${REPO_DIR}/SWEEP_RESULTS.txt"

{
  echo "================================================================"
  echo "SWEEP RESULTS SUMMARY - $(date)"
  echo "================================================================"
  echo ""
  
  echo "--- Job Status (sacct) ---"
  sacct -j 2015646-2015653 -o JobID,State,ExitCode,Elapsed,NodeList -P | grep -v batch || true
  echo ""
  
  echo "--- Active Queue ---"
  squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R" || true
  echo ""
  
  echo "--- Saved Models ---"
  ls -lh "${REPO_DIR}/models/diffusion_normalized_beta"*.pt 2>/dev/null | awk '{print $9, "("$5")"}' || echo "No normalized beta models saved yet"
  echo ""
  
  echo "--- Job Log Summary ---"
  for jid in 2015646 2015647 2015648 2015649 2015650 2015651 2015652 2015653; do
    outfile="${REPO_DIR}/logs/train_diff_norm_${jid}.out"
    errfile="${REPO_DIR}/logs/train_diff_norm_${jid}.err"
    if [ -f "$errfile" ]; then
      err_lines=$(wc -l < "$errfile" 2>/dev/null || echo 0)
      echo "Job $jid: stderr=$err_lines lines"
    fi
  done
  echo ""
  
  echo "--- Recent Errors (first 300 chars per job) ---"
  for jid in 2015646 2015647 2015648 2015649 2015650 2015651 2015652 2015653; do
    errfile="${REPO_DIR}/logs/train_diff_norm_${jid}.err"
    if [ -f "$errfile" ] && [ -s "$errfile" ]; then
      echo "Job $jid:"
      head -c 300 "$errfile" || true
      echo ""
    fi
  done
  echo ""
  
  echo "--- W&B Offline Runs (if any) ---"
  find "${REPO_DIR}/wandb" -name "run-*" -type d 2>/dev/null | wc -l | xargs echo "Found runs:"
  echo ""
  
  echo "--- Paper Improvements Status ---"
  echo "✓ Multi-schedule support: linear, cosine, quadratic"
  echo "✓ Class-specific sigma scaling (variance-scaled mode)"
  echo "✓ SLURM grid submitter (scripts/submit_sweep.sh)"
  echo "✓ W&B sweep template (wandb_sweep.yaml)"
  echo "✓ 8-job test sweep submitted (2×2×2 grid)"
  echo ""
  
} | tee "$SUMMARY_FILE"

echo "Summary written to $SUMMARY_FILE"
