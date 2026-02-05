#!/usr/bin/env python3
"""
Check if diffusion samples capture variance from real data distribution.
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
from collections import OrderedDict
import re

# Import from train script
import sys
sys.path.insert(0, 'scripts')
from train_latent_diffusion import ClassConditionedDiffusion, load_smile_embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load real data
test_latent = np.load('results/autoencoder_test_latent.npy')
test_df = pd.read_feather('Data/test_data.feather')
onehot_cols = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
test_labels = test_df[onehot_cols].values.argmax(axis=1)
chemicals = onehot_cols
samples_per_class = 150

# Sample real data
np.random.seed(42)
real_latents = []
real_labels_list = []
for class_idx in range(8):
    mask = test_labels == class_idx
    class_latents = test_latent[mask]
    n_samples = min(samples_per_class, len(class_latents))
    indices = np.random.choice(len(class_latents), n_samples, replace=False)
    real_latents.append(class_latents[indices])
    real_labels_list.extend([class_idx] * n_samples)

real_latents = np.vstack(real_latents)
real_labels = np.array(real_labels_list)

# Load model
print("Loading diffusion model...")
model = ClassConditionedDiffusion(latent_dim=512, smile_dim=512, num_classes=8, timesteps=50, hidden_dim=512, num_layers=6).to(device)
checkpoint = torch.load('models/diffusion_latent_separated_best.pt', map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get diffusion schedule (recreate if not in buffers)
timesteps = 50
beta_start, beta_end = 0.0001, 0.02
betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Load SMILE embeddings
smile_embeddings = load_smile_embeddings()

# Generate diffusion samples
print("Generating diffusion samples...")
diffusion_latents = []
diffusion_labels_list = []

with torch.no_grad():
    for class_idx, chem in enumerate(chemicals):
        if chem not in smile_embeddings:
            continue
        
        smile_emb_np = smile_embeddings[chem]
        smile_emb = torch.FloatTensor(smile_emb_np).unsqueeze(0).repeat(samples_per_class, 1).to(device)
        class_onehot = torch.zeros(samples_per_class, 8, device=device)
        class_onehot[:, class_idx] = 1.0
        
        # Start from noise
        z_t = torch.randn(samples_per_class, 512, device=device)
        
        # DDPM sampling
        for t in reversed(range(timesteps)):
            t_batch = torch.full((samples_per_class,), t, dtype=torch.long, device=device)
            
            # Predict noise
            predicted_noise = model(z_t, t_batch, smile_emb, class_onehot)
            
            alpha_bar_t = alphas_cumprod[t]
            beta_t = betas[t]
            
            if t > 0:
                alpha_bar_prev = alphas_cumprod[t-1]
                
                # Predict x_0
                x_0_pred = (z_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
                
                # Direction to x_t
                dir_xt = torch.sqrt(1 - alpha_bar_prev) * predicted_noise
                
                # Compute x_{t-1}
                z_t = torch.sqrt(alpha_bar_prev) * x_0_pred + dir_xt
                
                # Add stochastic noise
                z_t = z_t + torch.sqrt(beta_t) * torch.randn_like(z_t)
            else:
                # Final step
                z_t = (z_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        diffusion_latents.append(z_t.cpu().numpy())
        diffusion_labels_list.extend([class_idx] * samples_per_class)

diffusion_latents = np.vstack(diffusion_latents)
diffusion_labels = np.array(diffusion_labels_list)

# Analyze variance in raw latent space
print("\n" + "="*80)
print("VARIANCE ANALYSIS (Raw Latent Space, 512-dim)")
print("="*80)
print(f"{'Chemical':<8} | {'Real μ':<8} {'σ':<8} | {'Diff μ':<8} {'σ':<8} | Ratio")
print("-"*80)

for class_idx, chem in enumerate(chemicals):
    mask_r = real_labels == class_idx
    real_class = real_latents[mask_r]
    real_centroid = real_class.mean(axis=0)
    real_dists = np.sqrt(((real_class - real_centroid) ** 2).sum(axis=1))
    
    mask_d = diffusion_labels == class_idx
    diff_class = diffusion_latents[mask_d]
    diff_centroid = diff_class.mean(axis=0)
    diff_dists = np.sqrt(((diff_class - diff_centroid) ** 2).sum(axis=1))
    
    ratio = diff_dists.mean() / real_dists.mean() if real_dists.mean() > 0 else 0
    
    print(f"{chem:<8} | {real_dists.mean():6.2f}   {real_dists.std():6.2f} | {diff_dists.mean():6.2f}   {diff_dists.std():6.2f} | {ratio:.3f}x")

# PCA Analysis
print("\n" + "="*80)
print("PCA ANALYSIS (2D projection)")
print("="*80)

all_latents = np.vstack([real_latents, diffusion_latents])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_latents)

n_real = len(real_latents)
real_pca = all_pca[:n_real]
diffusion_pca = all_pca[n_real:]

# Silhouette scores
real_sil = silhouette_score(real_pca, real_labels)
diffusion_sil = silhouette_score(diffusion_pca, diffusion_labels)

print(f"\nSeparation (Silhouette):")
print(f"  Real:      {real_sil:7.4f}")
print(f"  Diffusion: {diffusion_sil:7.4f}")
print(f"  Diff:      {diffusion_sil - real_sil:+7.4f}")
print(f"\nVariance Explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

avg_ratio = np.mean([
    (np.sqrt(((diffusion_latents[diffusion_labels==i] - diffusion_latents[diffusion_labels==i].mean(axis=0))**2).sum(axis=1)).mean() /
     np.sqrt(((real_latents[real_labels==i] - real_latents[real_labels==i].mean(axis=0))**2).sum(axis=1)).mean())
    for i in range(8)
])

if avg_ratio < 0.1:
    print("❌ Diffusion samples are COLLAPSED (variance ratio < 0.1)")
    print("   Problem: Model is too deterministic, not capturing data distribution")
elif avg_ratio < 0.5:
    print("⚠️  Diffusion samples have LOW variance (ratio < 0.5)")
    print("   Recommendation: Increase noise schedule or reduce separation loss weight")
elif avg_ratio > 1.5:
    print("⚠️  Diffusion samples have HIGH variance (ratio > 1.5)")
    print("   May be adding too much noise or model not fully converged")
else:
    print(f"✅ Diffusion samples capture REASONABLE variance (ratio = {avg_ratio:.2f}x)")
    print("   Model is generating samples that match the real data distribution")

print("="*80 + "\n")
