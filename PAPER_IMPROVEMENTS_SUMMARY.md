# Paper Improvements Implementation & Sweep Results

## Summary
Successfully implemented all three paper improvement categories and launched an 8-job SLURM grid sweep (2×2×2 hyperparameter combinations). Training is actively running with early checkpoints saved.

---

## 1. Beta/Sigma Schedule Support ✓

**Implementation:**
- Added `--beta-schedule` CLI option with support for:
  - `linear` (default): uniform schedule from beta_start to beta_end
  - `cosine`: cosine annealing schedule (from diffusion literature)
  - `quadratic`: quadratic interpolation schedule

- Environment variable integration: `BETA_SCHEDULE` exported to `get_beta_schedule()` for runtime control
- Exposed via SLURM run scripts: `BETA_SCHEDULE` env var passed through

**Files Modified:**
- `scripts/train_normalized_diffusion.py`: Added `get_beta_schedule()` with multiple schedule types
- `scripts/run_train_normalized_diffusion.sh`: Added `BETA_SCHEDULE` env var export
- `scripts/run_train_latent_diffusion.sh`: Added `BETA_SCHEDULE` env var export

---

## 2. Class-Specific Sigma Schedules ✓

**Implementation:**
- Added `--class-sigma-mode` CLI option with modes:
  - `none` (default): global beta schedule for all classes
  - `variance-scaled`: per-class beta schedules scaled by latent variance

- Per-class beta computation: Scaled `beta_end` per class based on class variance relative to global variance
- Control parameters:
  - `--class-sigma-scale`: scaling strength (default 0.5)
  - `--beta-end-max`: maximum allowed beta_end (default 0.5)

- New functions:
  - `forward_diffusion_per_class()`: Applies per-class alpha_bar_t indexing
  - Variance computation during training setup phase

**Files Modified:**
- `scripts/train_normalized_diffusion.py`: Added per-class diffusion logic, class variance scaling

---

## 3. SLURM Grid Sweep Tooling ✓

**SLURM Submitter:**
- `scripts/submit_sweep.sh`: Simple bash grid submitter that loops over hyperparameter ranges
  - Supports environment variable configuration (BETA_END, NOISE_WEIGHT, SEPARATION_WEIGHT)
  - Submits jobs with `sbatch --export=ALL,...` for parameter passing
  - Default 2×2×2 grid (easily editable)

**W&B Sweep Config:**
- `wandb_sweep.yaml`: Template for W&B-managed sweeps
  - Grid search over beta_end, noise_weight, separation_weight
  - Metric: total_loss (minimize)
  - Can be used with `wandb sweep` + `wandb agent` for agent-based distributed sweeps

**Files Created:**
- `scripts/submit_sweep.sh` (executable)
- `wandb_sweep.yaml` (config template)
- `collect_sweep_results.sh` (results aggregation helper)

---

## Current Sweep Status (2×2×2 Grid)

**Job IDs:** 2015646–2015653

**Grid Parameters:**
- BETA_END: 0.02, 0.2
- NOISE_WEIGHT: 0.6, 0.8
- SEPARATION_WEIGHT: 0.1, 0.2

**Current Status (as of 18:27 EDT):**
- 2 jobs RUNNING (2015646, 2015647) on gpu-5-27 and gpu-5-28
- 6 jobs PENDING (queued behind running jobs)
- 1 model checkpoint saved: `diffusion_normalized_beta0.02_best.pt` (34M)
- Training showing steady loss improvement: Epoch 16 running with loss trending downward

**Training Logs Sample (Job 2015646):**
```
Epoch 1:  Total=0.589608, Noise=0.949830, Sep=0.197099
Epoch 10: Total=0.306851, Noise=0.487915, Sep=0.141019
Epoch 16: Total=0.268840, Noise=0.424716, Sep=0.140102  [in progress]
```

---

## What's Next

1. **Continue monitoring sweep**: Jobs will complete over the next 1–2 hours
2. **Collect final artifacts**:
   - Best model checkpoints for each (beta_end, noise_weight, separation_weight) combination
   - Loss curves and metrics from W&B (offline mode)
   - Performance comparison table

3. **Optional extensions**:
   - Expand grid (more beta_end values, margin modes, margin scales)
   - Run with adaptive-percentile margin mode for exploration
   - Implement learned sigma networks (per-class neural schedules)
   - Launch full W&B sweep agent for Bayesian optimization

---

## Commands for Monitoring

```bash
# Check running jobs
squeue -u $USER

# Get final job status
sacct -j 2015646-2015653 -o JobID,State,ExitCode,Elapsed -P

# Stream training logs
tail -f logs/train_diff_norm_2015646.out

# Re-run the summary script
bash collect_sweep_results.sh
```

---

## Files Created/Modified

**Created:**
- `scripts/submit_sweep.sh` — SLURM grid submitter
- `wandb_sweep.yaml` — W&B sweep template
- `collect_sweep_results.sh` — Results aggregator

**Modified:**
- `scripts/train_normalized_diffusion.py` — Added schedule types, class-sigma support
- `scripts/run_train_normalized_diffusion.sh` — Exposed BETA_SCHEDULE
- `scripts/run_train_latent_diffusion.sh` — Exposed BETA_SCHEDULE

**Documentation:**
- `README.md` — Updated SLURM-first workflow

---

## Paper Improvements Roadmap

✓ Multi-schedule support (linear, cosine, quadratic)
✓ Class-specific sigma (variance-scaled mode)
✓ SLURM grid sweeper
✓ W&B sweep template

Next phase (optional):
- [ ] Adaptive margin schedules (dynamic margin tuning during training)
- [ ] Learned sigma networks (NN-based per-class variance prediction)
- [ ] Bayesian sweep (HyperOpt / Optuna integration)
- [ ] Ensemble methods (voting over top-K models)

