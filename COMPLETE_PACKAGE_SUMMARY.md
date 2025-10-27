# MNIST1D Diffusion Experiments - Complete Package

## Files Created (Corrected Implementation)

### 1. Main Experiment Script
**`diffusion_mnist1d_experiments.py`**
- Runs 8 experiments with corrected semi-implicit training
- Uses MNIST1D dataset with 10 deterministic prime inverse patterns
- Integrates Weights & Biases for experiment tracking
- ~1000 lines, fully documented

### 2. SLURM Job Script
**`run_diffusion_mnist1d.sh`**
- Submits job to supercomputer
- 4 CPUs, 64GB RAM, short partition
- Email notifications

### 3. Documentation Files
**`CORRECTED_IMPLEMENTATION.md`**
- Explains what changed from initial misunderstanding
- Key concepts: prime inverse, directional training, four losses

**`SEMI_IMPLICIT_EXPLANATION.md`**
- Detailed explanation of corrected semi-implicit training
- Step-by-step process for all four losses
- Dataset structure and prime inverse patterns

**`SEMI_IMPLICIT_VISUAL.md`**
- Visual diagrams and flowcharts (kept from before, needs updating)

**`VERIFICATION_WALKTHROUGH.md`**  ← **MOST IMPORTANT FOR YOU**
- Step-by-step verification of implementation
- Shows exactly how your requirements are met
- Includes code snippets for each loss

**`README_EXPERIMENTS.md`**
- Quick start guide
- Troubleshooting
- Output analysis

### 4. Test Script
**`test_setup.py`**
- Pre-flight checks before running experiments
- Verifies imports, W&B login, dataset download

## Corrected Semi-Implicit Training

### Core Concept
**z-coordinate = direction indicator**
- z=0: Forward (noisy → clean)
- z=1: Inverse (clean → prime inverse)

### Prime Inverse
Deterministic noise patterns, one per digit (0-9):
- Digit 0: `-x`
- Digit 1: `roll(x, 10)`
- Digit 9: `cos(x*π/5) + 1` (like your cos(x)+1)

### Four Losses
1. **Forward**: MSE(model(noisy, t, z=0), clean)
2. **Inverse**: MSE(model(clean, t, z=1), prime_inverse)
3. **Roundtrip F→I**: clean → denoise(z=0) → to_prime(z=1) → loss
4. **Roundtrip I→F**: clean → to_prime(z=1) → denoise(z=0) → loss

Total: `forward + inverse + 0.5*roundtrip_fi + 0.5*roundtrip_if`

### Implicit vs Semi-Implicit
**Implicit**:
- Only z=0 samples (forward)
- Single loss: standard denoising
- No z-switch

**Semi-Implicit**:
- Both z=0 and z=1 samples
- Four losses
- Bidirectional learning

## Experiments

### 8 Configurations
1. MSE + Gaussian + Implicit
2. MSE + Gaussian + Semi-Implicit
3. MSE + Convex + Implicit
4. MSE + Convex + Semi-Implicit
5. KL Divergence + Gaussian + Implicit
6. KL Divergence + Gaussian + Semi-Implicit
7. KL Divergence + Convex + Implicit
8. KL Divergence + Convex + Semi-Implicit

### Metrics Tracked
- Training loss (overall)
- Individual losses (forward, inverse, roundtrip_fi, roundtrip_if) for semi-implicit
- Test MSE (overall, forward, inverse)
- Test MAE
- Test KL divergence
- Runtime
- Convergence rate

## Weights & Biases

**Configuration**:
- API Key: 57680a36aa570ba8df25adbdd143df3d0bf6b6e8
- Entity: metzlerk
- Project: mnist1d-diffusion-experiments

**Logging**:
- Per-epoch training losses
- Individual loss components
- Test metrics after training
- Hyperparameters and configuration

## Quick Start

### Pre-flight Check
```bash
cd ~/4f-files
python test_setup.py
```

### Run Experiments
```bash
sbatch run_diffusion_mnist1d.sh
```

### Monitor Progress
```bash
# Check job status
squeue -u kjmetzler

# View output (once job starts)
tail -f slurm-<jobid>.out

# Or check W&B dashboard
# https://wandb.ai/metzlerk/mnist1d-diffusion-experiments
```

### Expected Runtime
- Per experiment: ~5-10 minutes
- Total: ~40-80 minutes
- 8 experiments sequentially

## Output Files

### Generated Locally
1. `mnist1d_diffusion_experiments_YYYYMMDD_HHMMSS.json`
   - Complete experimental data
   - All metrics for all experiments

2. `mnist1d_experiment_report_YYYYMMDD_HHMMSS.md`
   - Summary table
   - Best configurations
   - Markdown formatted

3. `mnist1d_data.pkl`
   - Cached MNIST1D dataset
   - Downloaded on first run

### Weights & Biases
- Interactive dashboard
- Training curves
- Metric comparisons
- Configuration tracking

## Verification Steps

Before running, verify using `VERIFICATION_WALKTHROUGH.md`:

1. ✓ Prime inverse is deterministic per digit
2. ✓ z=0 means forward (noisy → clean)
3. ✓ z=1 means inverse (clean → prime inverse)
4. ✓ Four losses in semi-implicit
5. ✓ Implicit uses only z=0
6. ✓ Both sets trained simultaneously
7. ✓ Timesteps used in both directions
8. ✓ W&B logging integrated

## Key Differences from Original

| Aspect | Original Notebook | New Implementation |
|--------|------------------|-------------------|
| Dataset | Sine/cosine | MNIST1D |
| Architecture | Iterative + Variable | Iterative only |
| Experiments | 16 | 8 |
| Semi-implicit | Roundtrip only | Full bidirectional (4 losses) |
| Direction | Not explicit | z-coordinate indicator |
| Prime inverse | N/A | 10 deterministic patterns |
| Logging | Local only | W&B integrated |
| Format | Jupyter | Python script |

## What to Check After Running

### 1. Training Convergence
- Do losses decrease?
- Are semi-implicit losses stable?
- Do roundtrip losses converge?

### 2. Test Performance
- Which loss function works better?
- Does convex sampling help?
- Is semi-implicit better than implicit?

### 3. Forward vs Inverse
- Is forward MSE lower? (should be easier)
- Does inverse reach prime inverse correctly?
- Are they balanced?

### 4. Bidirectional Consistency
- Do roundtrip losses decrease?
- Can model do both directions?
- Is z-switch working?

## Troubleshooting

### If experiments fail
1. Check `slurm-<jobid>.out` for errors
2. Verify environment: `conda activate venv`
3. Check disk space: `df -h`
4. Test imports: `python test_setup.py`

### If W&B fails
- Experiments continue without W&B
- Check API key in code
- Verify internet connection
- Results still saved locally

### If out of memory
- Reduce `batch_size` in script
- Increase `--mem` in `.sh` file
- Use fewer samples

## Next Steps

After experiments complete:

1. **Analyze Results**
   ```python
   import json
   with open('mnist1d_diffusion_experiments_*.json') as f:
       results = json.load(f)
   ```

2. **Compare Methods**
   - Plot MSE by configuration
   - Compare implicit vs semi-implicit
   - Check forward vs inverse performance

3. **Hyperparameter Tuning**
   - Adjust loss weights
   - Try different prime inverse patterns
   - Experiment with timesteps

4. **Extend**
   - Test on other datasets
   - Try different architectures
   - Explore other transformations

## Contact

Questions or issues? Check the verification walkthrough first:
`VERIFICATION_WALKTHROUGH.md`

This shows exactly how the implementation meets your specifications.

## Summary

✓ Semi-implicit training corrected with bidirectional learning  
✓ Prime inverse implemented as deterministic patterns  
✓ Four losses: forward, inverse, two roundtrips  
✓ z-coordinate indicates direction (0=forward, 1=inverse)  
✓ Implicit training uses only forward direction  
✓ W&B integration for experiment tracking  
✓ Ready to run on supercomputer  

Everything is set up according to your specifications!
