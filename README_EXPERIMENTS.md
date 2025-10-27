# Files Created for MNIST1D Diffusion Experiments

## Overview
This package contains scripts and documentation for running diffusion model experiments on the MNIST1D dataset, adapted from the original sine/cosine experiments in `4g-kjm-Diffusion.ipynb`.

## Files Created

### 1. `diffusion_mnist1d_experiments.py` (Main Script)
**Purpose**: Runs 8 diffusion model experiments comparing different configurations

**Key Changes from Original:**
- ✓ Removed architecture comparison (only iterative, no variable)
- ✓ Uses MNIST1D dataset instead of synthetic sine/cosine data
- ✓ Adapted for 41-dimensional features (40 MNIST1D features + 1 label)
- ✓ Maintains semi-implicit training method
- ✓ Tests 2×2×2 = 8 configurations (Loss × Sampling × Training)

**Experiments Run:**
1. MSE + Gaussian + Implicit
2. MSE + Gaussian + Semi-Implicit
3. MSE + Convex + Implicit
4. MSE + Convex + Semi-Implicit
5. KL Divergence + Gaussian + Implicit
6. KL Divergence + Gaussian + Semi-Implicit
7. KL Divergence + Convex + Implicit
8. KL Divergence + Convex + Semi-Implicit

**Output Files:**
- `mnist1d_diffusion_experiments_YYYYMMDD_HHMMSS.json` - Raw experimental data
- `mnist1d_experiment_report_YYYYMMDD_HHMMSS.md` - Markdown summary report
- `mnist1d_experiment_results_partial.json` - Intermediate saves

### 2. `run_diffusion_mnist1d.sh` (SLURM Job Script)
**Purpose**: Submits the experiment to the supercomputer's job scheduler

**Configuration:**
- Partition: short
- Nodes: 1
- CPUs: 4
- Memory: 64GB
- No GPU (commented out, can be enabled)
- Email notifications on completion/failure

**Usage:**
```bash
sbatch run_diffusion_mnist1d.sh
```

### 3. `SEMI_IMPLICIT_EXPLANATION.md` (Detailed Documentation)
**Purpose**: Comprehensive explanation of semi-implicit training

**Contains:**
- Step-by-step breakdown of the algorithm
- Comparison with standard implicit training
- Mathematical intuition
- Data setup for MNIST1D
- Verification checklist
- Running instructions

### 4. `SEMI_IMPLICIT_VISUAL.md` (Visual Guide)
**Purpose**: Visual diagrams and flow charts for understanding semi-implicit training

**Contains:**
- Complete training flow diagram
- Data flow example with numbers
- Comparison diagrams (implicit vs semi-implicit)
- Gradient flow explanation
- Why .detach() is critical

### 5. `README_EXPERIMENTS.md` (This File)
**Purpose**: Overview and quick reference for all files

## Quick Start

### Option 1: Run Directly (Login Node - for testing only)
```bash
cd ~/4f-files
python diffusion_mnist1d_experiments.py
```
⚠️ Not recommended for full experiments - use Option 2 instead

### Option 2: Submit to Compute Node (Recommended)
```bash
cd ~/4f-files
sbatch run_diffusion_mnist1d.sh
```

Check job status:
```bash
squeue -u kjmetzler
```

### Option 3: Interactive Testing
```bash
srun -p short --mem=32G -c 4 --pty bash
source /home/kjmetzler/anaconda3/bin/activate
conda activate venv
cd ~/4f-files
python diffusion_mnist1d_experiments.py
```

## Dataset Details

### MNIST1D Dataset
- **Source**: https://github.com/greydanus/mnist1d
- **Format**: 1D sequences representing digits 0-9
- **Samples**: 2000 training (1000 per set)
- **Features**: 40 per sample (from original sequential data)
- **Additional**: +1 label dimension (z=0 or z=1)

### Data Preparation
**Set 1 (Digits 0-4, z=0):**
- Input: Original signal + z=0 label → (N, 41)
- Target: Phase-shifted signal + z=0 label → (N, 41)
- Transformation: Circular shift by 5 positions

**Set 2 (Digits 5-9, z=1):**
- Input: Original signal + z=1 label → (N, 41)
- Target: Inverted and scaled signal + z=1 label → (N, 41)
- Transformation: Multiply by -0.8

This mirrors the original experiment's structure:
- Original Set 1: (x,y,0) → (x,sin(x),0)
- Original Set 2: (x,sin(x),1) → (x,cos(x)+1,1)

## Understanding Semi-Implicit Training

### Core Concept
The model learns to be **self-inverse**: `denoise(noise(x)) ≈ x`

### Training Flow
1. **Forward Denoising**: Standard diffusion training
   - Add noise to target → model predicts noise → calculate loss
   
2. **Inverse Constraint**: Self-consistency check
   - Denoise to get predicted_clean (no gradients)
   - Add noise again to predicted_clean
   - Denoise again → should recover predicted_clean
   - Calculate consistency loss

3. **Combined Loss**: 
   ```
   total_loss = forward_loss + 0.5 * inverse_loss
   ```

### Why It Works
- Enforces consistency across noise levels
- Regularizes the model
- Improves generalization
- Makes training more stable

### Critical Implementation Detail
```python
predicted_clean = predicted_clean.detach()
```
This prevents gradient cycles and ensures stable training.

## Expected Runtime

- **Per Experiment**: ~5-10 minutes
- **Total (8 experiments)**: ~40-80 minutes
- **Depends on**: CPU speed, data loading time, whether MNIST1D is cached

## Monitoring Progress

The script prints detailed progress:
```
================================================================================
EXPERIMENT 1/8
Loss: mse, Sampling: gaussian, Training: implicit
================================================================================

Training model...
  Epoch 0/50, Loss: 0.234567
  Epoch 10/50, Loss: 0.123456
  ...

Testing model...

Results:
  Parameters: 123,456
  Test MSE: 0.012345
  Test MAE: 0.098765
  Test KL Divergence: 0.543210
  Runtime: 321.5s
```

## Output Analysis

### JSON File Structure
```json
{
  "metadata": {
    "start_time": "...",
    "dataset": "mnist1d",
    "n_samples": 2000,
    ...
  },
  "results": [
    {
      "experiment_id": 1,
      "loss_function": "mse",
      "sampling_method": "gaussian",
      "training_method": "implicit",
      "test_mse": 0.012345,
      "test_kl_divergence": 0.543210,
      ...
    },
    ...
  ],
  "summary": {
    "total_experiments": 8,
    "successful": 8,
    "failed": 0
  }
}
```

### Markdown Report
Contains:
- Configuration summary
- Results table
- Best configurations by metric
- Timestamp and metadata

## Troubleshooting

### Issue: MNIST1D download fails
**Solution**: Pre-download the dataset
```python
import requests
import pickle
url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
response = requests.get(url)
with open('mnist1d_data.pkl', 'wb') as f:
    f.write(response.content)
```

### Issue: Out of memory
**Solution**: Reduce batch size in the script
```python
batch_size = 64  # Instead of 128
```

### Issue: Job stuck in queue
**Solution**: Check queue status and partition availability
```bash
squeue
sinfo
```

### Issue: Conda environment not activated
**Solution**: Check the path in `.sh` file matches your conda installation
```bash
which conda
# Update the .sh file accordingly
```

## Verification Steps

Before running full experiments, verify:

1. ✓ Conda environment has required packages:
   ```bash
   conda activate venv
   python -c "import torch, numpy, pandas, matplotlib, scipy, requests"
   ```

2. ✓ MNIST1D dataset can be downloaded:
   ```bash
   python -c "import requests; print(requests.get('https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl').status_code)"
   ```

3. ✓ Script runs on small scale:
   - Edit script temporarily: `max_epochs = 2`
   - Run one experiment to verify

4. ✓ Output directory is writable:
   ```bash
   touch ~/4f-files/test.txt && rm ~/4f-files/test.txt
   ```

## Comparing to Original Experiments

| Aspect | Original (4g-kjm-Diffusion.ipynb) | New (diffusion_mnist1d_experiments.py) |
|--------|-----------------------------------|----------------------------------------|
| Dataset | Synthetic sine/cosine | MNIST1D (real data) |
| Features | 3 (x, y, z) | 41 (40 features + z) |
| Experiments | 16 | 8 |
| Architectures | 2 (iterative, variable) | 1 (iterative only) |
| Training Sets | 2 (sine transforms) | 2 (digit transforms) |
| Format | Jupyter Notebook | Python script |
| Execution | Interactive | Batch job |

## Next Steps After Completion

1. **Analyze Results**:
   ```bash
   # View the report
   cat mnist1d_experiment_report_*.md
   
   # Parse JSON for detailed analysis
   python -m json.tool mnist1d_diffusion_experiments_*.json
   ```

2. **Compare Metrics**:
   - Which loss function works better?
   - Does convex sampling help?
   - Is semi-implicit training beneficial?

3. **Visualize**:
   Create plots comparing:
   - Test MSE across configurations
   - KL divergence across configurations
   - Runtime efficiency

4. **Extend**:
   - Try different hyperparameters
   - Test on other datasets
   - Experiment with architecture variants

## Contact & Support

If you encounter issues or have questions:
- Check SEMI_IMPLICIT_EXPLANATION.md for algorithm details
- Check SEMI_IMPLICIT_VISUAL.md for visual explanations
- Review error messages in slurm output files
- Email: kjmetzler@wpi.edu

## References

- Original notebook: `4g-kjm-Diffusion.ipynb`
- MNIST1D: https://github.com/greydanus/mnist1d
- Experiment guide: `EXPERIMENT_GUIDE.md`
