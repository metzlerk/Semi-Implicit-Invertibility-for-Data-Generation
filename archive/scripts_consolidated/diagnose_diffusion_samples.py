#!/usr/bin/env python3
"""
Quick diagnostic: Check if diffusion model generates varied samples
"""

import torch
import numpy as np
from scipy.spatial.distance import pdist
import sys

sys.path.insert(0, '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/scripts')
from train_latent_diffusion import ClassConditionedDiffusion, get_beta_schedule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load model
model = ClassConditionedDiffusion(512, 512, 8, 512, 6, 50).to(device)
checkpoint = torch.load('models/diffusion_latent_separated_best.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Generating 100 samples for class 0...")

# Generate samples
smile_emb = torch.randn(100, 512, device=device)  # dummy SMILE embedding
class_onehot = torch.zeros(100, 8, device=device)
class_onehot[:, 0] = 1.0

# Sample using DDPM
with torch.no_grad():
    z_t = torch.randn(100, 512, device=device)
    
    for t in reversed(range(50)):
        t_batch = torch.full((100,), t, dtype=torch.long, device=device)
        predicted_noise = model(z_t, t_batch, smile_emb, class_onehot)
        
        alpha_t = model.alphas[t]
        alpha_bar_t = model.alphas_cumprod[t]
        beta_t = model.betas[t]
        
        if t > 0:
            alpha_bar_prev = model.alphas_cumprod[t-1]
            x_0_pred = (z_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * predicted_noise
            z_t = torch.sqrt(alpha_bar_prev) * x_0_pred + dir_xt
            z_t = z_t + torch.sqrt(beta_t) * torch.randn_like(z_t)
        else:
            z_t = (z_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
    
    samples = z_t.cpu().numpy()

print(f"\nGenerated samples shape: {samples.shape}")
print(f"Mean: {samples.mean():.4f}")
print(f"Std: {samples.std():.4f}")
print(f"Range: [{samples.min():.4f}, {samples.max():.4f}]")

# Check pairwise distances
dists = pdist(samples)
print(f"\nPairwise distances:")
print(f"  Mean: {dists.mean():.4f}")
print(f"  Std: {dists.std():.4f}")
print(f"  Min: {dists.min():.4f}")
print(f"  Max: {dists.max():.4f}")

if dists.mean() < 1.0:
    print("\n⚠️  WARNING: Samples appear COLLAPSED (very low pairwise distances)")
else:
    print("\n✓ Samples appear to have reasonable variance")

# Check if all samples are identical
unique_samples = np.unique(samples, axis=0)
print(f"\nUnique samples: {len(unique_samples)} out of 100")
if len(unique_samples) < 10:
    print("⚠️  WARNING: Very few unique samples - likely collapsed!")
