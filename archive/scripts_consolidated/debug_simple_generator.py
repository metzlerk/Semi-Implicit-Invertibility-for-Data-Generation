#!/usr/bin/env python3
"""
Alternative approach: Direct generative model (simpler than diffusion)
Train a model that directly maps noise -> latent distribution
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# =============================================================================
# SIMPLE GENERATOR MODEL
# =============================================================================

class SimpleGenerator(nn.Module):
    """Direct generator: noise -> latent"""
    def __init__(self, noise_dim=512, latent_dim=512, hidden_dim=1024):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, noise):
        return self.net(noise)

# =============================================================================
# DATA
# =============================================================================

print("Loading DEB latent data...")
test_latent = np.load('results/autoencoder_test_latent_separated.npy')
test_df = pd.read_feather('Data/test_data.feather')
test_labels = test_df[['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']].values.argmax(axis=1)

deb_mask = test_labels == 0
deb_latents = test_latent[deb_mask]

print(f"DEB samples: {len(deb_latents)}")
print(f"  Mean: {deb_latents.mean():.4f}")
print(f"  Std: {deb_latents.std():.4f}")

sample_subset = deb_latents[np.random.choice(len(deb_latents), min(500, len(deb_latents)), replace=False)]
dists = pdist(sample_subset)
print(f"  Pairwise distances: mean={dists.mean():.2f}")

TARGET_MEAN_DIST = dists.mean()
TARGET_STD = deb_latents.std()

# Normalize data for stable training
data_mean = deb_latents.mean(axis=0)
data_std = deb_latents.std(axis=0) + 1e-6

deb_normalized = (deb_latents - data_mean) / data_std

train_tensor = torch.FloatTensor(deb_normalized)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_tensor),
    batch_size=256,
    shuffle=True
)

# =============================================================================
# TRAINING WITH MMD LOSS
# =============================================================================

def gaussian_kernel(x, y, sigma=1.0):
    """RBF kernel for MMD"""
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]
    
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    
    return torch.exp(-torch.mean((tiled_x - tiled_y)**2, dim=2) / (2 * sigma))

def mmd_loss(x, y, sigmas=[0.1, 1.0, 10.0]):
    """Maximum Mean Discrepancy loss"""
    loss = 0
    for sigma in sigmas:
        xx = gaussian_kernel(x, x, sigma).mean()
        yy = gaussian_kernel(y, y, sigma).mean()
        xy = gaussian_kernel(x, y, sigma).mean()
        loss += xx + yy - 2 * xy
    return loss / len(sigmas)

print("\n" + "="*80)
print("TRAINING SIMPLE GENERATOR WITH MMD LOSS")
print("="*80)

model = SimpleGenerator(512, 512, 1024).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500)

best_score = float('inf')

for epoch in range(500):
    model.train()
    epoch_loss = 0
    n_batches = 0
    
    for (batch_real,) in train_loader:
        batch_real = batch_real.to(device)
        batch_size = len(batch_real)
        
        # Generate samples
        noise = torch.randn(batch_size, 512, device=device)
        generated = model(noise)
        
        # MMD loss (match distributions)
        loss = mmd_loss(batch_real, generated)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    avg_loss = epoch_loss / n_batches
    
    # Evaluate
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            noise = torch.randn(500, 512, device=device)
            samples = model(noise).cpu().numpy()
            
            # Denormalize
            samples = samples * data_std + data_mean
            
            sample_std = samples.std()
            sample_mean_dist = pdist(samples).mean()
            
            std_ratio = sample_std / TARGET_STD
            dist_ratio = sample_mean_dist / TARGET_MEAN_DIST
            score = (std_ratio + dist_ratio) / 2
            
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, mean_dist={sample_mean_dist:.2f} (target={TARGET_MEAN_DIST:.2f}), "
                  f"std={sample_std:.2f} (target={TARGET_STD:.2f}), score={score:.3f}")
            
            if abs(score - 1.0) < abs(best_score - 1.0):
                best_score = score
                torch.save(model.state_dict(), 'models/simple_generator_best.pt')

# Final evaluation
model.eval()
with torch.no_grad():
    noise = torch.randn(500, 512, device=device)
    final_samples = model(noise).cpu().numpy()
    final_samples = final_samples * data_std + data_mean

final_std = final_samples.std()
final_mean_dist = pdist(final_samples).mean()
final_score = ((final_std / TARGET_STD) + (final_mean_dist / TARGET_MEAN_DIST)) / 2

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Mean distance: {final_mean_dist:.2f} (target: {TARGET_MEAN_DIST:.2f}, ratio: {final_mean_dist/TARGET_MEAN_DIST:.3f})")
print(f"Std: {final_std:.2f} (target: {TARGET_STD:.2f}, ratio: {final_std/TARGET_STD:.3f})")
print(f"Score: {final_score:.3f} (best: {best_score:.3f})")

# Visualize
from sklearn.decomposition import PCA

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

real_subset = deb_latents[np.random.choice(len(deb_latents), 500, replace=False)]
all_data = np.vstack([real_subset, final_samples])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_data)

axes[0].scatter(all_pca[:500, 0], all_pca[:500, 1], alpha=0.5, s=30, label='Real', color='blue')
axes[0].scatter(all_pca[500:, 0], all_pca[500:, 1], alpha=0.5, s=30, label='Generated', color='red')
axes[0].set_title(f'PCA: Real vs Simple Generator (DEB)\nScore: {final_score:.3f}', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(real_subset.flatten(), bins=50, alpha=0.5, label='Real', density=True, color='blue')
axes[1].hist(final_samples.flatten(), bins=50, alpha=0.5, label='Generated', density=True, color='red')
axes[1].set_title(f'Value Distribution\nReal std={TARGET_STD:.2f}, Generated std={final_std:.2f}', fontweight='bold')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/simple_generator_single_chemical.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: images/simple_generator_single_chemical.png")

if final_score > 0.8 and final_score < 1.2:
    print("\n✓ SUCCESS! Simple generator works!")
    print("  This approach can replace diffusion")
else:
    print(f"\n⚠ Score {final_score:.3f} - needs more tuning")
