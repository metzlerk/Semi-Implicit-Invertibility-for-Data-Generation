# MNIST1D Diffusion Model Experimental Results

**Date**: 2025-10-27 22:05:58

**Dataset**: MNIST1D (digits 0-9)
**Total Experiments**: 16
**Successful**: 6
**Failed**: 10

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
| 2 | fid | gaussian | semi_implicit | 5.083399 | 66.010590 | 124.4 |
| 5 | fid | convex | implicit | 4.204008 | 500.346466 | 77.2 |
| 6 | fid | convex | semi_implicit | 3.880648 | 565.838257 | 130.0 |
| 10 | kl_divergence | gaussian | semi_implicit | 5.222494 | 64.107155 | 126.7 |
| 13 | kl_divergence | convex | implicit | 4.181244 | 513.768127 | 76.5 |
| 14 | kl_divergence | convex | semi_implicit | 3.800267 | 534.942383 | 130.8 |

## Best Configurations

**Best MSE**: Experiment 14
- Configuration: kl_divergence, convex, semi_implicit
- Test MSE: 3.800267

**Best KL Divergence**: Experiment 10
- Configuration: kl_divergence, gaussian, semi_implicit
- Test KL: 64.107155

