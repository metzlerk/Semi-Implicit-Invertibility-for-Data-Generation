# MNIST1D Diffusion Model Experimental Results

**Date**: 2025-10-27 16:49:13

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
| 1 | fid | gaussian | implicit | 5.237579 | 51.410488 | 667.9 |
| 2 | fid | gaussian | semi_implicit | 5.160124 | 69.034836 | 122.6 |
| 3 | fid | convex | implicit | 3.775000 | 343.279907 | 702.2 |
| 4 | fid | convex | semi_implicit | 3.734430 | nan | 128.2 |
| 5 | kl_divergence | gaussian | implicit | 5.004122 | 56.756592 | 739.5 |
| 6 | kl_divergence | gaussian | semi_implicit | 5.195210 | 69.974823 | 122.7 |
| 7 | kl_divergence | convex | implicit | 3.770426 | nan | 767.0 |
| 8 | kl_divergence | convex | semi_implicit | 3.692004 | nan | 128.3 |

## Best Configurations

**Best MSE**: Experiment 8
- Configuration: kl_divergence, convex, semi_implicit
- Test MSE: 3.692004

**Best KL Divergence**: Experiment 1
- Configuration: fid, gaussian, implicit
- Test KL: 51.410488

