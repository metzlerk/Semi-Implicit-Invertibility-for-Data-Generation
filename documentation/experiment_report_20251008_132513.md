# Diffusion Model Experimental Comparison Report

**Generated:** 2025-10-08 13:25:13

---

## Executive Summary

This report presents the results of a comprehensive experimental comparison of diffusion models for 3D point transformation tasks. We tested 16 different configurations across 4 experimental dimensions:

- **Architecture**: Iterative (shared U-Net) vs Variable (timestep-specific U-Nets)
- **Loss Function**: MSE (noise prediction) vs KL Divergence (distribution matching)
- **Sampling Method**: Gaussian noise vs Convex combinations
- **Training Method**: Implicit (standard) vs Semi-implicit (self-inverse)

### Best Performing Configuration

- **Experiment ID**: 6
- **Architecture**: iterative
- **Loss Function**: kl_divergence
- **Sampling Method**: gaussian
- **Training Method**: semi_implicit
- **Test MSE**: 0.020527
- **Test KL Divergence**: 2.615584
- **Parameters**: 69,187
- **Runtime**: 5.5s

---

## Experimental Setup

### Dataset

- **Type**: 3D point transformation
- **Samples**: 2000
- **Transformation 1**: (x, y, 0) → (x, sin(x), 0)
- **Transformation 2**: (x, sin(x), 1) → (x, cos(x)+1, 1)

### Training Configuration

- **Epochs**: 50
- **Batch Size**: 80
- **Base Timesteps**: 20
- **Device**: cpu
- **Learning Rate**: 0.001

---

## Detailed Results

### All Experiments (Sorted by Test MSE)

| Rank | ID | Architecture | Loss | Sampling | Training | Test MSE | KL Div | Params | Runtime |
|------|-----|--------------|------|----------|----------|----------|--------|--------|----------|
| 1 | 6 | iterative | kl_divergence | gaussian | semi_implicit | 0.020527 | 2.6156 | 69,187 | 5.5s |
| 2 | 2 | iterative | mse | gaussian | semi_implicit | 0.031564 | 2.7618 | 69,187 | 5.6s |
| 3 | 5 | iterative | kl_divergence | gaussian | implicit | 0.031586 | 2.0714 | 69,187 | 22.5s |
| 4 | 1 | iterative | mse | gaussian | implicit | 0.037462 | 2.2842 | 69,187 | 3.7s |
| 5 | 8 | iterative | kl_divergence | convex | semi_implicit | 2.828346 | 21.3846 | 69,187 | 7.0s |
| 6 | 3 | iterative | mse | convex | implicit | 3.108665 | 12.5479 | 69,187 | 3.8s |
| 7 | 4 | iterative | mse | convex | semi_implicit | 3.303612 | 9.8605 | 69,187 | 5.6s |
| 8 | 7 | iterative | kl_divergence | convex | implicit | 3.312262 | 12.5773 | 69,187 | 21.9s |

---

## Analysis by Experimental Dimension

### 1. Architecture Comparison

**Iterative (Shared U-Net)**:
- Mean Test MSE: 1.584253 ± 1.667900
- Mean KL Divergence: 8.2629
- Parameters: 69,187
- Average Runtime: 9.5s

**Variable (Timestep-Specific U-Nets)**:
### 2. Loss Function Comparison

**MSE Loss (Noise Prediction)**:
- Mean Test MSE: 1.620326 ± 1.832869
- Mean KL Divergence: 6.8636
- Average Runtime: 4.7s

**KL Divergence Loss (Distribution Matching)**:
- Mean Test MSE: 1.548180 ± 1.768671
- Mean KL Divergence: 9.6622
- Average Runtime: 14.2s

### 3. Sampling Method Comparison

**Gaussian Noise Sampling**:
- Mean Test MSE: 0.030285 ± 0.007073
- Mean KL Divergence: 2.4333
- Average Runtime: 9.3s

**Convex Combination Sampling**:
- Mean Test MSE: 3.138221 ± 0.226966
- Mean KL Divergence: 14.0926
- Average Runtime: 9.6s

### 4. Training Method Comparison

**Implicit Training (Standard)**:
- Mean Test MSE: 1.622494 ± 1.835514
- Mean KL Divergence: 7.3702
- Average Runtime: 13.0s

**Semi-Implicit Training (Self-Inverse)**:
- Mean Test MSE: 1.546012 ± 1.765804
- Mean KL Divergence: 9.1556
- Average Runtime: 5.9s

---

## Key Findings

### Best Configurations by Dimension

1. **Architecture**: iterative (MSE: 0.020527)
2. **Loss Function**: kl_divergence (MSE: 0.020527)
3. **Sampling Method**: gaussian (MSE: 0.020527)
4. **Training Method**: semi_implicit (MSE: 0.020527)

### Parameter Efficiency

**Most Parameter-Efficient Configuration** (Exp 6):
- Configuration: iterative-kl_divergence-gaussian-semi_implicit
- Test MSE: 0.020527
- Parameters: 69,187
- Efficiency Score: 7.04e-04

### Runtime Analysis

**Fastest Configuration** (Exp 1):
- Configuration: iterative-mse-gaussian-implicit
- Runtime: 3.7s
- Test MSE: 0.037462

---

## Visualizations

The following figures summarize the experimental results:

1. **experiment_comparison_quality.png**: Sample quality metrics (MSE, KL divergence) by dimension
2. **experiment_comparison_efficiency.png**: Parameter counts and runtimes
3. **experiment_heatmap_mse.png**: Complete MSE heatmap across all configurations
4. **experiment_training_curves.png**: Training curves for top-performing experiments

---

## Recommendations for Advisor Discussion

### For Best Sample Quality

Use **Experiment 6** configuration:
- Architecture: iterative
- Loss: kl_divergence
- Sampling: gaussian
- Training: semi_implicit
- Achieves MSE: 0.020527

### For Parameter Efficiency

Use **Experiment 6** configuration:
- Architecture: iterative
- Loss: kl_divergence
- Sampling: gaussian
- Training: semi_implicit
- Parameters: 69,187
- MSE: 0.020527

### Trade-offs to Discuss

1. **Variable vs Iterative Architecture**: Variable has nanx more parameters but has nan% worse MSE
2. **MSE vs KL Divergence Loss**: Both optimize for different objectives; consider task requirements
3. **Gaussian vs Convex Sampling**: Geometric structure preservation vs traditional diffusion
4. **Implicit vs Semi-Implicit Training**: Self-inverse property may help with roundtrip consistency

---

## Data Files

- **Complete Results**: `diffusion_experiments_20251008_132509.json`
- **This Report**: `experiment_report_20251008_132513.md`
- **Generated**: 2025-10-08 13:25:13

---

*End of Report*
