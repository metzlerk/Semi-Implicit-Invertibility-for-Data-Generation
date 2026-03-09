"""
Check variance in ORIGINAL 512-D space (not PCA projection)
"""

import sys
import os
import torch
import numpy as np
import pandas as pd

ROOT_DIR = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation'
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))
from train_latent_diffusion import get_beta_schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_CHEMICAL = 'DEB'
N_SAMPLES = 500
TIMESTEPS = 50
TIMESTEPS_TO_SHOW = [0, 10, 20, 30, 40, 49]

def apply_forward_diffusion(x0, t, betas):
    """Apply forward diffusion"""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alpha_bar_t = alphas_cumprod[t]
    epsilon = torch.randn_like(x0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * epsilon
    return x_t, epsilon

# Load data
test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'))
test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
test_labels = test_df['Label'].values
deb_mask = test_labels == TARGET_CHEMICAL
deb_latent = test_latent[deb_mask]
indices = np.random.choice(len(deb_latent), N_SAMPLES, replace=False)
deb_samples = deb_latent[indices]

# Apply diffusion
betas = get_beta_schedule(TIMESTEPS)
x0 = torch.FloatTensor(deb_samples).to(device)

print("="*80)
print("VARIANCE IN ORIGINAL 512-D LATENT SPACE")
print("="*80)
print(f"\n{'t':<5}  {'Mean (per dim)':<15}  {'Std (per dim)':<15}  {'Mean norm':<15}  {'Frobenius norm':<15}")
print("-"*80)

for t in TIMESTEPS_TO_SHOW:
    if t == 0:
        x_t = x0
    else:
        x_t, _ = apply_forward_diffusion(x0, t, betas)
    
    x_t_np = x_t.cpu().numpy()
    
    # Statistics across all dimensions
    mean_per_dim = np.abs(x_t_np).mean()  # Mean absolute value
    std_per_dim = x_t_np.std()  # Std across all values
    mean_norm = np.linalg.norm(x_t_np, axis=1).mean()  # Mean L2 norm of each sample
    frob_norm = np.linalg.norm(x_t_np, 'fro')  # Frobenius norm of entire batch
    
    print(f"{t:<5}  {mean_per_dim:<15.4f}  {std_per_dim:<15.4f}  {mean_norm:<15.2f}  {frob_norm:<15.2f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("""
For N(0, 1) noise in 512-D with 500 samples:
- Mean (per dim) should be ≈ 0.80 (mean absolute value of N(0,1))
- Std (per dim) should be ≈ 1.0
- Mean norm should be ≈ sqrt(512) ≈ 22.6

At clean latent (t=0):
- Will have original data statistics (non-Gaussian)

At fully diffused (t=49):
- Should match pure Gaussian noise statistics
""")

# Check if t=49 looks like pure noise
if TIMESTEPS_TO_SHOW[-1] == 49:
    x_49, _ = apply_forward_diffusion(x0, 49, betas)
    x_49_np = x_49.cpu().numpy()
    std_49 = x_49_np.std()
    expected_std = 1.0
    
    print(f"\n✓ At t=49:")
    print(f"  Observed std: {std_49:.3f}")
    print(f"  Expected std: {expected_std:.3f}")
    print(f"  Difference:   {abs(std_49 - expected_std):.3f}")
    
    if abs(std_49 - expected_std) < 0.1:
        print(f"\n  ✓ GOOD: Samples are properly diffused to noise!")
    else:
        print(f"\n  ⚠ WARNING: Std is off by {abs(std_49-expected_std):.2f}")

print("="*80)
