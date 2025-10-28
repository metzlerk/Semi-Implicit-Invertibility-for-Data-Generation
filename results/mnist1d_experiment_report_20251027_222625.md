# MNIST1D Diffusion Model Experimental Results

**Date**: 2025-10-27 22:26:25

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
| 1 | fid | gaussian | implicit | 6.063733 | 53.689453 | 90.0 |
| 2 | fid | gaussian | semi_implicit | 5.089886 | 74.658951 | 125.0 |
| 5 | fid | convex | implicit | 4.071080 | 496.363953 | 142.6 |
| 6 | fid | convex | semi_implicit | 4.137633 | 502.980530 | 130.7 |
| 9 | kl_divergence | gaussian | implicit | 5.810835 | 45.964066 | 89.3 |
| 10 | kl_divergence | gaussian | semi_implicit | 4.968314 | 79.361969 | 124.6 |
| 13 | kl_divergence | convex | implicit | 4.030762 | 511.492157 | 142.5 |
| 14 | kl_divergence | convex | semi_implicit | 4.127789 | 502.173035 | 130.8 |

## Best Configurations

**Best MSE**: Experiment 13
- Configuration: kl_divergence, convex, implicit
- Test MSE: 4.030762

**Best KL Divergence**: Experiment 9
- Configuration: kl_divergence, gaussian, implicit
- Test KL: 45.964066

