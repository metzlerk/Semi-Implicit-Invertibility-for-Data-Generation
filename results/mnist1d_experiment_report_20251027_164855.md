# MNIST1D Diffusion Model Experimental Results

**Date**: 2025-10-27 16:48:55

**Dataset**: MNIST1D (digits 0-9)
**Total Experiments**: 8
**Successful**: 8
**Failed**: 0

## Configuration

- Architecture: Iterative (single U-Net)
- Loss Functions: fid, kl_divergence
- Sampling Methods: gaussian, convex
- Training Methods: implicit, semi_implicit
- Epochs: 1000
- Batch Size: 128
- Feature Dimension: 41

## Results Summary

| Exp | Loss | Sampling | Training | Test MSE | Test KL | Runtime (s) |
|-----|------|----------|----------|----------|---------|-------------|
| 1 | fid | gaussian | implicit | 5.490264 | 57.268070 | 670.2 |
| 2 | fid | gaussian | semi_implicit | 5.008788 | 72.282295 | 122.5 |
| 3 | fid | convex | implicit | 3.753853 | nan | 698.2 |
| 4 | fid | convex | semi_implicit | 3.686280 | 314.408844 | 126.7 |
| 5 | kl_divergence | gaussian | implicit | 5.114575 | 58.699291 | 733.8 |
| 6 | kl_divergence | gaussian | semi_implicit | 5.104734 | 71.517189 | 122.3 |
| 7 | kl_divergence | convex | implicit | 3.684244 | 439.592987 | 758.2 |
| 8 | kl_divergence | convex | semi_implicit | 3.847854 | 335.214600 | 128.3 |

## Best Configurations

**Best MSE**: Experiment 7
- Configuration: kl_divergence, convex, implicit
- Test MSE: 3.684244

**Best KL Divergence**: Experiment 1
- Configuration: fid, gaussian, implicit
- Test KL: 57.268070

