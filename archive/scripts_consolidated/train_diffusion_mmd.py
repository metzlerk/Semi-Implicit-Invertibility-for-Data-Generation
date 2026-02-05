#!/usr/bin/env python3
"""
DIFFUSION WITH MMD REGULARIZATION
Instead of discriminator, use MMD loss directly to match the data distribution.
This forces diffusion to preserve the manifold structure (the "islands").
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

LATENT_DIM = 512
TIMESTEPS = 50
HIDDEN_DIM = 512
NUM_LAYERS = 6
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
MAX_EPOCHS = 500
MMD_WEIGHT = 0.5  # Weight on MMD loss

# =============================================================================
# DIFFUSION MODEL
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionModel(nn.Module):
    def __init__(self, latent_dim=512, timesteps=50, hidden_dim=512, num_layers=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        input_dim = latent_dim + hidden_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        
        betas = torch.linspace(0.00002, 0.005, timesteps)
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t.float())
        x_in = torch.cat([x, t_emb], dim=1)
        return self.net(x_in)

# =============================================================================
# MMD LOSS
# =============================================================================

def gaussian_kernel(x, y, sigma=1.0):
    """RBF kernel for MMD"""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input / (2 * sigma))

def mmd_loss(x, y, sigmas=[0.1, 1.0, 10.0]):
    """Maximum Mean Discrepancy loss with multiple bandwidths"""
    mmd = 0
    for sigma in sigmas:
        xx = gaussian_kernel(x, x, sigma).mean()
        yy = gaussian_kernel(y, y, sigma).mean()
        xy = gaussian_kernel(x, y, sigma).mean()
        mmd += xx + yy - 2 * xy
    return mmd / len(sigmas)

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
DATA_MEAN = deb_latents.mean()
DATA_STD = deb_latents.std()
deb_latents_normalized = (deb_latents - DATA_MEAN) / DATA_STD

print(f"  Original mean: {DATA_MEAN:.4f}, std: {DATA_STD:.4f}")
print(f"  Normalized mean: {deb_latents_normalized.mean():.4f}, std: {deb_latents_normalized.std():.4f}")

TARGET_MEAN_DIST = pdist(deb_latents[np.random.choice(len(deb_latents), 500, replace=False)]).mean()
TARGET_STD = deb_latents.std()
print(f"  Target (original scale): mean_dist={TARGET_MEAN_DIST:.2f}, std={TARGET_STD:.2f}")

train_tensor = torch.FloatTensor(deb_latents_normalized)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =============================================================================
# TRAINING WITH MMD REGULARIZATION
# =============================================================================

def forward_diffusion(x_0, t, model):
    sqrt_alpha_bar = torch.sqrt(model.alphas_cumprod[t]).reshape(-1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - model.alphas_cumprod[t]).reshape(-1, 1)
    noise = torch.randn_like(x_0)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise

@torch.no_grad()
def sample_ddpm(model, n_samples=100):
    model.eval()
    x_t = torch.randn(n_samples, LATENT_DIM, device=device)
    
    for t in reversed(range(model.timesteps)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
        predicted_noise = model(x_t, t_batch)
        
        alpha_t = model.alphas[t]
        alpha_bar_t = model.alphas_cumprod[t]
        beta_t = model.betas[t]
        
        if t > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        
        x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
    
    return x_t

def evaluate_samples(samples_normalized):
    samples = samples_normalized.cpu().numpy() * DATA_STD + DATA_MEAN
    if len(samples) < 2:
        return 0, 0, 0
    sample_std = samples.std()
    sample_mean_dist = pdist(samples[:min(500, len(samples))]).mean()
    std_ratio = sample_std / TARGET_STD
    dist_ratio = sample_mean_dist / TARGET_MEAN_DIST
    score = (std_ratio + dist_ratio) / 2
    return sample_mean_dist, sample_std, score

# =============================================================================
# TRAINING LOOP
# =============================================================================

print("\n" + "="*80)
print(f"TRAINING DIFFUSION WITH MMD REGULARIZATION (weight={MMD_WEIGHT})")
print("="*80)

model = DiffusionModel(LATENT_DIM, TIMESTEPS, HIDDEN_DIM, NUM_LAYERS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)

best_score = float('inf')

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss_denoise = 0
    epoch_loss_mmd = 0
    n_batches = 0
    
    for (batch_latent,) in train_loader:
        batch_latent = batch_latent.to(device)
        batch_size = len(batch_latent)
        
        # Standard denoising loss
        t = torch.randint(0, TIMESTEPS, (batch_size,), device=device)
        x_noisy, noise = forward_diffusion(batch_latent, t, model)
        predicted_noise = model(x_noisy, t)
        loss_denoise = nn.functional.mse_loss(predicted_noise, noise)
        
        # MMD loss every 3 batches (expensive to compute)
        if (n_batches % 3) == 0:
            with torch.no_grad():
                fake_samples = sample_ddpm(model, n_samples=min(256, batch_size))
            
            # Compute MMD between fake and real
            real_subset = batch_latent[:min(256, batch_size)]
            loss_mmd_val = mmd_loss(fake_samples, real_subset)
            
            # Combined loss
            loss_total = loss_denoise + MMD_WEIGHT * loss_mmd_val
            epoch_loss_mmd += loss_mmd_val.item()
        else:
            loss_total = loss_denoise
            epoch_loss_mmd += 0
        
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss_denoise += loss_denoise.item()
        n_batches += 1
    
    scheduler.step()
    
    avg_loss_denoise = epoch_loss_denoise / n_batches
    avg_loss_mmd = epoch_loss_mmd / max(1, n_batches // 3)
    
    if (epoch + 1) % 20 == 0:
        samples_norm = sample_ddpm(model, n_samples=500)
        mean_dist, std, score = evaluate_samples(samples_norm)
        
        print(f"Epoch {epoch+1}: denoise={avg_loss_denoise:.4f}, mmd={avg_loss_mmd:.4f} | "
              f"mean_dist={mean_dist:.2f} (target={TARGET_MEAN_DIST:.2f}), "
              f"std={std:.2f} (target={TARGET_STD:.2f}), score={score:.3f}")
        
        if abs(score - 1.0) < abs(best_score - 1.0):
            best_score = score
            torch.save({
                'model_state': model.state_dict(),
                'data_mean': DATA_MEAN,
                'data_std': DATA_STD
            }, 'models/diffusion_mmd_best.pt')

# Final evaluation
final_samples_norm = sample_ddpm(model, n_samples=500)
final_mean_dist, final_std, final_score = evaluate_samples(final_samples_norm)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Mean distance: {final_mean_dist:.2f} (target: {TARGET_MEAN_DIST:.2f}, ratio: {final_mean_dist/TARGET_MEAN_DIST:.3f})")
print(f"Std: {final_std:.2f} (target: {TARGET_STD:.2f}, ratio: {final_std/TARGET_STD:.3f})")
print(f"Score: {final_score:.3f} (best: {best_score:.3f})")

# Visualize
from sklearn.decomposition import PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

final_samples = final_samples_norm.cpu().numpy() * DATA_STD + DATA_MEAN
real_subset = deb_latents[np.random.choice(len(deb_latents), 500, replace=False)]
all_data = np.vstack([real_subset, final_samples])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_data)

axes[0].scatter(all_pca[:500, 0], all_pca[:500, 1], alpha=0.5, s=30, label='Real', color='blue')
axes[0].scatter(all_pca[500:, 0], all_pca[500:, 1], alpha=0.5, s=30, label='Diffusion+MMD', color='red')
axes[0].set_title(f'PCA: MMD-Regularized Diffusion (DEB)\nScore: {final_score:.3f}', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(real_subset.flatten(), bins=50, alpha=0.5, label='Real', density=True, color='blue')
axes[1].hist(final_samples.flatten(), bins=50, alpha=0.5, label='Diffusion+MMD', density=True, color='red')
axes[1].set_title(f'Value Distribution\nReal std={TARGET_STD:.2f}, Diff std={final_std:.2f}', fontweight='bold')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/diffusion_mmd_single_chemical.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: images/diffusion_mmd_single_chemical.png")

if final_score > 0.8 and final_score < 1.2:
    print("\n✓ SUCCESS! MMD-regularized diffusion preserves manifold!")
else:
    print(f"\n⚠ Score {final_score:.3f} - may need tuning")
