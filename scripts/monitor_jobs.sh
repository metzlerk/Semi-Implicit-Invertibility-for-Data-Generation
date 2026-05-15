#!/bin/bash
# monitor_jobs.sh <job_ids_comma> <poll_interval_seconds> <max_seconds>
JOB_IDS_CSV="$1"
POLL_INTERVAL=${2:-60}
MAX_SECONDS=${3:-43200}
LOGFILE="logs/job_monitor.log"

if [[ -z "$JOB_IDS_CSV" ]]; then
  echo "Usage: $0 <job_ids_comma> [poll_interval_seconds] [max_seconds]" >&2
  exit 2
fi

IFS=',' read -r -a JOB_IDS <<< "$JOB_IDS_CSV"
start_ts=$(date +%s)
echo "Monitor started at $(date) for jobs: ${JOB_IDS[*]}" | tee -a "$LOGFILE"

all_done=false
while true; do
  now=$(date +%s)
  elapsed=$((now - start_ts))
  echo "=== $(date) | elapsed=${elapsed}s ===" | tee -a "$LOGFILE"
  any_running=false
  for job in "${JOB_IDS[@]}"; do
    # prefer sacct if available
    if command -v sacct >/dev/null 2>&1; then
      out=$(sacct -j ${job} --format=JobID,State --noheader -P 2>/dev/null | tr -s '\n' ';')
      if [[ -z "$out" ]]; then
        # fall back to squeue
        out=$(squeue -j ${job} -o "%i %T %M %R" -h 2>/dev/null || true)
      fi
    else
      out=$(squeue -j ${job} -o "%i %T %M %R" -h 2>/dev/null || true)
    fi
    if [[ -z "$out" ]]; then
      echo "job ${job}: NOT IN SQUEUE/sacct (may be finished)" | tee -a "$LOGFILE"
    else
      echo "job ${job}: ${out}" | tee -a "$LOGFILE"
      # check for RUNNING
      if echo "$out" | grep -q -E "RUNNING|PENDING|CONFIGURING|COMPLETING"; then
        any_running=true
      fi
    fi
  done

  if ! $any_running ; then
    echo "No running/pending jobs detected; exiting monitor." | tee -a "$LOGFILE"
    exit 0
  fi

  if (( elapsed >= MAX_SECONDS )); then
    echo "Max wait time reached (${MAX_SECONDS}s); exiting monitor." | tee -a "$LOGFILE"
    exit 0
  fi

  sleep "$POLL_INTERVAL"
done
