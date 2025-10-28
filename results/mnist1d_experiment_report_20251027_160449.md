# MNIST1D Diffusion Model Experimental Results

**Date**: 2025-10-27 16:04:49

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
| 1 | fid | gaussian | implicit | 5.265600 | 50.365688 | 667.6 |
| 2 | fid | gaussian | semi_implicit | 5.015625 | 68.022270 | 121.5 |
| 3 | fid | convex | implicit | 3.757640 | nan | 697.1 |
| 4 | fid | convex | semi_implicit | 3.691088 | 389.608795 | 127.0 |
| 5 | kl_divergence | gaussian | implicit | 5.191652 | 57.600277 | 733.2 |
| 6 | kl_divergence | gaussian | semi_implicit | 5.276428 | 73.443291 | 122.3 |
| 7 | kl_divergence | convex | implicit | 3.761023 | nan | 760.9 |
| 8 | kl_divergence | convex | semi_implicit | 3.721842 | nan | 126.9 |

## Best Configurations

**Best MSE**: Experiment 4
- Configuration: fid, convex, semi_implicit
- Test MSE: 3.691088

**Best KL Divergence**: Experiment 1
- Configuration: fid, gaussian, implicit
- Test KL: 50.365688

