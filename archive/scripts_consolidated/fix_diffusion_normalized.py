#!/usr/bin/env python3
"""
NORMALIZE DATA BEFORE DIFFUSION - The real fix
High variance (std=10.67) breaks diffusion's assumptions
Normalize to std=1, train diffusion, then denormalize samples
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

# =============================================================================
# DIFFUSION MODEL (standard DDPM)
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
        
        # Best schedule - score 1.205
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
# DATA WITH NORMALIZATION
# =============================================================================

print("Loading DEB latent data...")
test_latent = np.load('results/autoencoder_test_latent_separated.npy')
test_df = pd.read_feather('Data/test_data.feather')
test_labels = test_df[['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']].values.argmax(axis=1)

deb_mask = test_labels == 0
deb_latents = test_latent[deb_mask]

print(f"DEB samples: {len(deb_latents)}")
print(f"  Original mean: {deb_latents.mean():.4f}, std: {deb_latents.std():.4f}")

# NORMALIZE
DATA_MEAN = deb_latents.mean()
DATA_STD = deb_latents.std()
deb_latents_normalized = (deb_latents - DATA_MEAN) / DATA_STD

print(f"  Normalized mean: {deb_latents_normalized.mean():.4f}, std: {deb_latents_normalized.std():.4f}")

# Target metrics on ORIGINAL scale
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
# TRAINING & SAMPLING
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
    
    return x_t.cpu().numpy()

def evaluate_samples(samples_normalized):
    """Denormalize and evaluate on original scale"""
    samples = samples_normalized * DATA_STD + DATA_MEAN
    
    if len(samples) < 2:
        return 0, 0, 0
    
    sample_std = samples.std()
    sample_mean_dist = pdist(samples[:min(500, len(samples))]).mean()
    std_ratio = sample_std / TARGET_STD
    dist_ratio = sample_mean_dist / TARGET_MEAN_DIST
    score = (std_ratio + dist_ratio) / 2
    return sample_mean_dist, sample_std, score

# =============================================================================
# TRAINING
# =============================================================================

print("\n" + "="*80)
print("TRAINING DIFFUSION ON NORMALIZED DATA (std=1)")
print("="*80)

model = DiffusionModel(LATENT_DIM, TIMESTEPS, HIDDEN_DIM, NUM_LAYERS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)

best_score = float('inf')

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0
    n_batches = 0
    
    for (batch_latent,) in train_loader:
        batch_latent = batch_latent.to(device)
        
        t = torch.randint(0, TIMESTEPS, (len(batch_latent),), device=device)
        x_noisy, noise = forward_diffusion(batch_latent, t, model)
        predicted_noise = model(x_noisy, t)
        
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    avg_loss = epoch_loss / n_batches
    
    if (epoch + 1) % 20 == 0:
        samples_norm = sample_ddpm(model, n_samples=500)
        mean_dist, std, score = evaluate_samples(samples_norm)
        
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, mean_dist={mean_dist:.2f} (target={TARGET_MEAN_DIST:.2f}), "
              f"std={std:.2f} (target={TARGET_STD:.2f}), score={score:.3f}")
        
        if abs(score - 1.0) < abs(best_score - 1.0):
            best_score = score
            torch.save({
                'model_state': model.state_dict(),
                'data_mean': DATA_MEAN,
                'data_std': DATA_STD
            }, 'models/diffusion_normalized_best.pt')

# Final evaluation
final_samples_norm = sample_ddpm(model, n_samples=500)
final_mean_dist, final_std, final_score = evaluate_samples(final_samples_norm)

print("\n" + "="*80)
print("FINAL RESULTS (DENORMALIZED)")
print("="*80)
print(f"Mean distance: {final_mean_dist:.2f} (target: {TARGET_MEAN_DIST:.2f}, ratio: {final_mean_dist/TARGET_MEAN_DIST:.3f})")
print(f"Std: {final_std:.2f} (target: {TARGET_STD:.2f}, ratio: {final_std/TARGET_STD:.3f})")
print(f"Score: {final_score:.3f} (best: {best_score:.3f})")

# Visualize
from sklearn.decomposition import PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Denormalize for visualization
final_samples = final_samples_norm * DATA_STD + DATA_MEAN
real_subset = deb_latents[np.random.choice(len(deb_latents), 500, replace=False)]
all_data = np.vstack([real_subset, final_samples])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_data)

axes[0].scatter(all_pca[:500, 0], all_pca[:500, 1], alpha=0.5, s=30, label='Real', color='blue')
axes[0].scatter(all_pca[500:, 0], all_pca[500:, 1], alpha=0.5, s=30, label='Diffusion', color='red')
axes[0].set_title(f'PCA: Real vs Normalized Diffusion (DEB)\nScore: {final_score:.3f}', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(real_subset.flatten(), bins=50, alpha=0.5, label='Real', density=True, color='blue')
axes[1].hist(final_samples.flatten(), bins=50, alpha=0.5, label='Diffusion', density=True, color='red')
axes[1].set_title(f'Value Distribution\nReal std={TARGET_STD:.2f}, Diffusion std={final_std:.2f}', fontweight='bold')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/diffusion_normalized_single_chemical.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: images/diffusion_normalized_single_chemical.png")

if final_score > 0.8 and final_score < 1.2:
    print("\n✓ SUCCESS! Normalized diffusion works!")
    print("  Key: Train on normalized data (std=1), denormalize samples")
else:
    print(f"\n⚠ Score {final_score:.3f} - may need more tuning")
