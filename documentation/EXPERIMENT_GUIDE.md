# Diffusion Model Experimental Framework - User Guide

## Overview

This experimental framework compares 16 different configurations of diffusion models across 4 key dimensions:

### Experimental Dimensions

1. **Architecture** (2 variants)
   - **Iterative**: Single shared U-Net for all timesteps (standard approach)
   - **Variable**: 5 separate U-Nets for different noise levels (more parameters, specialized)

2. **Loss Function** (2 variants)
   - **MSE**: Standard noise prediction loss (recreates training distribution)
   - **KL Divergence**: Distribution matching loss (optimizes for distributional similarity)

3. **Sampling Method** (2 variants)
   - **Gaussian**: Standard Gaussian noise for diffusion (probabilistic)
   - **Convex**: Convex combinations of points (geometric/deterministic)

4. **Training Method** (2 variants)
   - **Implicit**: Standard training (forward diffusion only)
   - **Semi-Implicit**: Self-inverse training (learns denoise(noise(x)) ≈ x)

**Total Experiments**: 2 × 2 × 2 × 2 = **16 configurations**

---

## Dataset

The experiments use a 3D point transformation task:
- **Set 1**: (x, y, 0) → (x, sin(x), 0)
- **Set 2**: (x, sin(x), 1) → (x, cos(x)+1, 1)
- **Training samples**: 2000
- **Test samples**: 200
- **Epochs per experiment**: 50

---

## How to Run

### Prerequisites
1. Ensure you have executed all cells up to the "Experimental Comparison Framework" section
2. The baseline diffusion model should be trained (cells 1-17)
3. Required libraries: torch, numpy, matplotlib, pandas, scipy, seaborn

### Execution Steps

```python
# Step 1: Run the experimental framework setup
# Execute cells defining:
# - EXPERIMENT_CONFIG
# - VariableMapDiffusion
# - KL Divergence Loss
# - Convex Combination Sampling
# - Semi-Implicit Training
# - Main Experiment Runner

# Step 2: Run all experiments
# Execute the cell "Run all 16 experiments"
# This will take approximately 20-40 minutes

# Step 3: Generate outputs
# Execute cells for:
# - JSON export
# - Visualizations
# - Markdown report
# - Summary display
```

---

## Output Files

### 1. JSON Results (`diffusion_experiments_YYYYMMDD_HHMMSS.json`)
Complete experimental data including:
- Metadata (dataset, hyperparameters, timestamps)
- Results for all 16 experiments
- Detailed metrics per experiment

**Structure:**
```json
{
  "metadata": {
    "start_time": "...",
    "dataset": "sin_transformation",
    "n_samples": 2000,
    ...
  },
  "results": [
    {
      "experiment_id": 1,
      "architecture": "iterative",
      "loss_function": "mse",
      "sampling_method": "gaussian",
      "training_method": "implicit",
      "num_parameters": 123456,
      "test_mse": 0.001234,
      "test_kl_divergence": 0.0234,
      ...
    },
    ...
  ]
}
```

### 2. Markdown Report (`experiment_report_YYYYMMDD_HHMMSS.md`)
Formatted report containing:
- Executive summary with best configuration
- Detailed results table (all 16 experiments sorted by MSE)
- Analysis by each dimension
- Key findings and recommendations
- Parameter efficiency analysis
- Trade-off discussions

**Sections:**
1. Executive Summary
2. Experimental Setup
3. Detailed Results
4. Analysis by Dimension
5. Key Findings
6. Recommendations for Advisor Discussion

### 3. Visualizations (4 PNG files)

**a) `experiment_comparison_quality.png`** (2×2 grid)
- Test MSE by Architecture
- Test KL Divergence by Loss Function
- Test MSE by Sampling Method
- Test MSE by Training Method

**b) `experiment_comparison_efficiency.png`** (1×2 grid)
- Model Parameters by Architecture
- Runtime by Configuration

**c) `experiment_heatmap_mse.png`**
- Complete MSE heatmap
- Rows: Architecture-Loss combinations
- Columns: Sampling-Training combinations

**d) `experiment_training_curves.png`** (2×2 grid)
- Training loss curves for top 4 experiments
- Shows convergence behavior

---

## Metrics Explained

### Sample Quality Metrics

1. **Test MSE (Mean Squared Error)**
   - Measures how accurately generated samples match target distribution
   - Lower is better
   - Primary metric for sample quality

2. **Test KL Divergence**
   - Measures distributional similarity
   - Lower is better
   - Indicates how well the model captures the target distribution

3. **MAE per Dimension (X, Y, Z)**
   - Mean absolute error for each coordinate
   - Identifies which dimensions are harder to model

### Efficiency Metrics

4. **Number of Parameters**
   - Total trainable parameters in the model
   - Variable architecture has ~5× more parameters

5. **Runtime**
   - Time to train for 50 epochs
   - Includes data loading and evaluation

6. **Convergence Rate**
   - (Initial loss - Final loss) / epochs
   - Higher indicates faster learning

---

## Interpreting Results

### What to Look For

1. **Best Overall Performance**
   - Check which experiment has lowest Test MSE
   - This is your best configuration for sample quality

2. **Architecture Comparison**
   - Does Variable (more parameters) outperform Iterative?
   - By how much? Is the parameter increase justified?

3. **Loss Function Impact**
   - Does KL Divergence loss lead to better distributional matching?
   - Trade-off: MSE optimizes for noise prediction, KL for distribution

4. **Sampling Method**
   - Does geometric sampling (convex) preserve structure better?
   - Or does stochastic (Gaussian) provide better exploration?

5. **Training Method**
   - Does semi-implicit (self-inverse) improve roundtrip consistency?
   - Trade-off: More complex training vs potential better inverse properties

### Key Questions for Advisor

1. **Parameter Efficiency**: Is the performance gain from Variable architecture worth 5× parameters?

2. **Loss Function**: Should we optimize for noise prediction (MSE) or distribution matching (KL)?

3. **Geometric vs Stochastic**: Does the problem benefit from geometric structure (convex) or need randomness (Gaussian)?

4. **Self-Inverse Property**: Is the additional training complexity of semi-implicit worth it for consistency?

5. **Best Trade-off**: Which configuration offers the best balance of quality, parameters, and runtime?

---

## Common Issues and Solutions

### Issue 1: Out of Memory
**Solution**: Reduce batch_size or use fewer noise levels in Variable architecture

### Issue 2: Training Taking Too Long
**Solution**: Reduce max_epochs or run fewer experiments (comment out some combinations)

### Issue 3: Poor Convergence
**Solution**: Check learning_rate, may need tuning for specific configurations

### Issue 4: NaN Losses
**Solution**: Typically occurs in KL divergence; increase epsilon in histogram computation

---

## Customization

### To Add More Experiments

```python
EXPERIMENT_CONFIG = {
    'architecture': ['iterative', 'variable', 'YOUR_NEW_ARCH'],
    'loss': ['mse', 'kl_divergence', 'YOUR_NEW_LOSS'],
    'sampling': ['gaussian', 'convex', 'YOUR_NEW_SAMPLING'],
    'training': ['implicit', 'semi_implicit', 'YOUR_NEW_TRAINING']
}
```

Then implement the corresponding classes/functions.

### To Change Hyperparameters

Modify the hyperparameter cell before running experiments:
- `timesteps`: Number of diffusion steps
- `hidden_dim`: Model capacity
- `max_epochs`: Training duration
- `learning_rate`: Optimization speed

### To Use Different Dataset

Modify the `generate_3d_data()` function to generate your custom transformations.

---

## Expected Runtime

- **Per experiment**: ~2-3 minutes (GPU) or 5-10 minutes (CPU)
- **Total (16 experiments)**: ~30-50 minutes (GPU) or 1.5-3 hours (CPU)
- **Post-processing**: ~1-2 minutes

**Note**: Variable architecture experiments take longer due to more parameters.

---

## Presenting to Advisor

### Recommended Flow

1. **Start with Executive Summary** (from markdown report)
   - Best configuration found
   - Key metrics achieved

2. **Show Sample Quality Visualization**
   - `experiment_comparison_quality.png`
   - Discuss which dimension matters most

3. **Discuss Trade-offs**
   - Parameters vs Performance (from efficiency plot)
   - Architecture comparison (Iterative vs Variable)

4. **Deep Dive on Interesting Findings**
   - Show heatmap for complete picture
   - Training curves for convergence analysis

5. **Recommendations**
   - Best for quality
   - Best for efficiency
   - Best overall balance

### Key Talking Points

- "We systematically tested 16 configurations across 4 dimensions"
- "The best configuration achieved MSE of X.XXXX with Y parameters"
- "Variable architecture has 5× more parameters but only improves by X%"
- "KL divergence loss better matches the target distribution"
- "Convex sampling preserves geometric structure"
- "Semi-implicit training ensures self-inverse property"

---

## Next Steps After Experiments

1. **Analyze the markdown report** - Understand which dimensions matter most
2. **Review visualizations** - Identify patterns and outliers
3. **Check JSON for details** - Deep dive into specific experiments
4. **Prepare presentation** - Use findings for advisor discussion
5. **Iterate if needed** - Run focused experiments on promising configurations

---

## Support and Troubleshooting

If you encounter issues:
1. Check that all prerequisite cells are executed
2. Verify GPU/CPU memory is sufficient
3. Review error messages in failed experiments (stored in JSON)
4. Consider reducing experiment scope for testing

---

**Created**: October 8, 2025
**Framework Version**: 1.0
**Compatible with**: PyTorch 1.x+, Python 3.7+
