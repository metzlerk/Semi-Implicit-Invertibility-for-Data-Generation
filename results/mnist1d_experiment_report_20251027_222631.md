# MNIST1D Diffusion Model Experimental Results

**Date**: 2025-10-27 22:26:31

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
| 1 | fid | gaussian | implicit | 5.868996 | 50.582218 | 89.9 |
| 2 | fid | gaussian | semi_implicit | 4.955934 | 80.602089 | 125.6 |
| 5 | fid | convex | implicit | 4.074709 | 515.334473 | 143.0 |
| 6 | fid | convex | semi_implicit | 4.146239 | 516.411194 | 131.9 |
| 9 | kl_divergence | gaussian | implicit | 5.756726 | 51.039623 | 89.7 |
| 10 | kl_divergence | gaussian | semi_implicit | 5.213452 | 75.399361 | 126.2 |
| 13 | kl_divergence | convex | implicit | 4.055724 | 493.756134 | 142.8 |
| 14 | kl_divergence | convex | semi_implicit | 4.140845 | 518.037903 | 131.7 |

## Best Configurations

**Best MSE**: Experiment 13
- Configuration: kl_divergence, convex, implicit
- Test MSE: 4.055724

**Best KL Divergence**: Experiment 1
- Configuration: fid, gaussian, implicit
- Test KL: 50.582218

