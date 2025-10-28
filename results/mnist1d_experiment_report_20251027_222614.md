# MNIST1D Diffusion Model Experimental Results

**Date**: 2025-10-27 22:26:14

**Dataset**: MNIST1D (digits 0-9)
**Total Experiments**: 16
**Successful**: 8
**Failed**: 8

## Configuration

- Architecture: Iterative (single U-Net)
- Loss Functions: fid, kl_divergence
- Sampling Methods: gaussian, convex
- Training Methods: implicit, semi_implicit, semi_explicit, explicit
- Epochs: 1000
- Batch Size: 128
- Feature Dimension: 41

## Results Summary

| Exp | Loss | Sampling | Training | Test MSE | Test KL | Runtime (s) |
|-----|------|----------|----------|----------|---------|-------------|
| 1 | fid | gaussian | implicit | 6.010303 | 51.728123 | 89.3 |
| 2 | fid | gaussian | semi_implicit | 5.116842 | 74.817375 | 125.2 |
| 5 | fid | convex | implicit | 4.097003 | 490.157013 | 140.2 |
| 6 | fid | convex | semi_implicit | 4.165574 | 509.927155 | 130.6 |
| 9 | kl_divergence | gaussian | implicit | 5.925710 | 54.580471 | 88.9 |
| 10 | kl_divergence | gaussian | semi_implicit | 5.093981 | 74.838249 | 124.6 |
| 13 | kl_divergence | convex | implicit | 4.061733 | 509.516052 | 142.5 |
| 14 | kl_divergence | convex | semi_implicit | 4.152691 | 469.728943 | 130.8 |

## Best Configurations

**Best MSE**: Experiment 13
- Configuration: kl_divergence, convex, implicit
- Test MSE: 4.061733

**Best KL Divergence**: Experiment 1
- Configuration: fid, gaussian, implicit
- Test KL: 51.728123

