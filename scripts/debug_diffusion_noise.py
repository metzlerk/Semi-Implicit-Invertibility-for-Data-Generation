"""
Debug: Check if forward diffusion is adding enough noise
"""

import sys
import os
import torch
import numpy as np

ROOT_DIR = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation'
sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))
from train_latent_diffusion import get_beta_schedule

# Get beta schedule
TIMESTEPS = 50
betas = get_beta_schedule(TIMESTEPS)

# Compute alphas and cumulative product
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

print("="*80)
print("DIFFUSION SCHEDULE ANALYSIS")
print("="*80)
print(f"\nTimesteps: {TIMESTEPS}")
print(f"Beta range: [{betas.min():.6f}, {betas.max():.6f}]")
print(f"\nAlpha_bar (cumulative product of alphas):")
print(f"  At t=0:  {alphas_cumprod[0]:.6f} (should be close to 1.0)")
print(f"  At t=10: {alphas_cumprod[10]:.6f}")
print(f"  At t=20: {alphas_cumprod[20]:.6f}")
print(f"  At t=30: {alphas_cumprod[30]:.6f}")
print(f"  At t=40: {alphas_cumprod[40]:.6f}")
print(f"  At t=49: {alphas_cumprod[49]:.6f} (should be close to 0.0)")

print("\n" + "="*80)
print("NOISE vs SIGNAL RATIOS")
print("="*80)
print(f"{'t':<5} {'alpha_bar':<12} {'sqrt(a_bar)':<15} {'sqrt(1-a_bar)':<15} {'Signal %':<10} {'Noise %':<10}")
print("-"*80)

for t in [0, 10, 20, 30, 40, 49]:
    alpha_bar = alphas_cumprod[t].item()
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar)
    signal_pct = sqrt_alpha_bar / (sqrt_alpha_bar + sqrt_one_minus_alpha_bar) * 100
    noise_pct = sqrt_one_minus_alpha_bar / (sqrt_alpha_bar + sqrt_one_minus_alpha_bar) * 100
    
    print(f"{t:<5} {alpha_bar:<12.6f} {sqrt_alpha_bar:<15.6f} {sqrt_one_minus_alpha_bar:<15.6f} {signal_pct:<10.1f} {noise_pct:<10.1f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("""
For proper diffusion:
- At t=0:  alpha_bar ≈ 1.0 → x_t ≈ x_0 (pure signal, no noise)
- At t=T:  alpha_bar ≈ 0.0 → x_t ≈ epsilon (pure noise, no signal)

Current schedule shows:
""")

final_alpha = alphas_cumprod[49].item()
if final_alpha > 0.1:
    print(f"⚠ WARNING: alpha_bar at t=49 is {final_alpha:.3f}")
    print(f"  This means x_49 is still {np.sqrt(final_alpha)*100:.1f}% signal!")
    print(f"  The samples are NOT being fully diffused to noise.")
    print(f"\n  SOLUTION: Increase beta_max or use more timesteps")
else:
    print(f"✓ alpha_bar at t=49 is {final_alpha:.3f} - good diffusion")

print("\n" + "="*80)
print("EXPECTED VARIANCE CHANGE")
print("="*80)

# Simulate what variance should do
x0_std = 7.0  # Approximate std of DEB in PC1
print(f"\nAssuming clean latent has std = {x0_std:.1f}")
print(f"{'t':<5} {'Expected Std':<15} {'Observed Std':<15}")
print("-"*50)

observed_stds = [7.217, 7.126, 6.902, 6.563, 6.103, 5.592]

for i, t in enumerate([0, 10, 20, 30, 40, 49]):
    alpha_bar = alphas_cumprod[t].item()
    # Variance of x_t = alpha_bar * var(x_0) + (1 - alpha_bar) * var(epsilon)
    # If x_0 has variance σ_0^2 and epsilon ~ N(0,1), then:
    # var(x_t) = alpha_bar * σ_0^2 + (1 - alpha_bar) * 1
    expected_var = alpha_bar * (x0_std**2) + (1 - alpha_bar) * 1.0
    expected_std = np.sqrt(expected_var)
    
    print(f"{t:<5} {expected_std:<15.2f} {observed_stds[i]:<15.3f}")

print("\n⚠ If observed std is DECREASING, that's because PCA projects")
print("  high-dimensional noise differently than the original space.")
print("  Need to check variance in ORIGINAL 512-D space, not PCA!")
print("="*80)
