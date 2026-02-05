#!/usr/bin/env python3
"""
Single-Chemical Diffusion Training (DEB)
Fast iteration to test manifold preservation with improved settings
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Paths
DATA_DIR = 'Data'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
IMAGES_DIR = 'images'

# Hyperparameters - IMPROVED
LATENT_DIM = 512
TIMESTEPS = 1000  # Key fix: 50 -> 1000
HIDDEN_DIM = 512
NUM_LAYERS = 6
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
MAX_EPOCHS = 1000  # Increased to learn structure better

# =============================================================================
# MODEL
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

class SimpleDiffusion(nn.Module):
    def __init__(self, latent_dim=512, timesteps=1000, hidden_dim=512, num_layers=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        
        time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        layers = []
        layers.append(nn.Linear(latent_dim + time_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
        
        # COSINE SCHEDULE (better for manifolds)
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999).float()
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x_in = torch.cat([x, t_emb], dim=1)
        return self.net(x_in)

# =============================================================================
# DATA LOADING
# =============================================================================

print("Loading DEB latent data...")
train_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy'))
train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))

# Get DEB only
mask = train_df['DEB'].values == 1
deb_latents = train_latent[mask]

print(f"DEB samples: {len(deb_latents)}")
print(f"  Original mean: {deb_latents.mean():.4f}, std: {deb_latents.std():.4f}")

# Normalize
DATA_MEAN = deb_latents.mean()
DATA_STD = deb_latents.std()
deb_latents_norm = (deb_latents - DATA_MEAN) / DATA_STD

print(f"  Normalized mean: {deb_latents_norm.mean():.4f}, std: {deb_latents_norm.std():.4f}")

# Target metrics
target_dists = pdist(deb_latents[np.random.choice(len(deb_latents), 3000, replace=False)])
print(f"  Target: mean_dist={target_dists.mean():.2f}, std={DATA_STD:.2f}")

# Dataset
train_tensor = torch.FloatTensor(deb_latents_norm)
train_loader = torch.utils.data.DataLoader(
    train_tensor, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)

# =============================================================================
# TRAINING
# =============================================================================

def forward_diffusion(x_0, t, model):
    """Forward diffusion: q(x_t | x_0)"""
    sqrt_alpha_bar = torch.sqrt(model.alphas_cumprod[t]).reshape(-1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - model.alphas_cumprod[t]).reshape(-1, 1)
    noise = torch.randn_like(x_0)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t, x_0  # Return x_0 instead of noise for x_0-prediction training

@torch.no_grad()
def sample_ddpm(model, n_samples):
    """DDIM sampling with x_0 prediction - deterministic, better for structure"""
    model.eval()
    x_t = torch.randn(n_samples, LATENT_DIM, device=device)
    
    # Use DDIM with 100 steps through 1000 timesteps
    ddim_steps = 100
    step_size = model.timesteps // ddim_steps
    timesteps = list(range(0, model.timesteps, step_size))
    
    for i in reversed(range(len(timesteps))):
        t = timesteps[i]
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
        
        # Model directly predicts x_0
        x_0_pred = model(x_t, t_batch)
        x_0_pred = torch.clamp(x_0_pred, -5, 5)  # Clamp to normalized data range
        
        if i > 0:
            t_prev = timesteps[i-1]
            alpha_bar_t = model.alphas_cumprod[t]
            alpha_bar_prev = model.alphas_cumprod[t_prev]
            
            # DDIM update: interpolate between x_0 and direction to x_t
            predicted_noise = (x_t - torch.sqrt(alpha_bar_t) * x_0_pred) / torch.sqrt(1 - alpha_bar_t)
            x_t = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_prev) * predicted_noise
        else:
            x_t = x_0_pred
    
    return x_t

print("\n" + "="*80)
print(f"TRAINING SINGLE-CHEMICAL DIFFUSION (DEB)")
print(f"Timesteps: {TIMESTEPS} | Cosine schedule | X_0 PREDICTION")
print("="*80 + "\n")

model = SimpleDiffusion(
    latent_dim=LATENT_DIM,
    timesteps=TIMESTEPS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)

best_loss = float('inf')

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0
    n_batches = 0
    
    for latent_batch in train_loader:
        latent_batch = latent_batch.to(device)
        
        t = torch.randint(0, TIMESTEPS, (len(latent_batch),), device=device)
        # Forward diffusion returns (x_t, x_0) for x_0-prediction training
        x_noisy, target_x0 = forward_diffusion(latent_batch, t, model)
        
        # Model predicts x_0 directly
        predicted_x0 = model(x_noisy, t)
        
        # Loss: MSE between predicted x_0 and actual x_0
        loss = nn.functional.mse_loss(predicted_x0, target_x0)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    avg_loss = epoch_loss / n_batches
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'data_mean': DATA_MEAN,
            'data_std': DATA_STD,
            'epoch': epoch,
            'loss': avg_loss
        }, os.path.join(MODELS_DIR, 'diffusion_single_deb_best.pt'))
    
    # Evaluate every 50 epochs
    if (epoch + 1) % 50 == 0 or epoch == 0:
        samples_norm = sample_ddpm(model, 1000).cpu().numpy()
        samples = samples_norm * DATA_STD + DATA_MEAN
        sample_dists = pdist(samples)
        
        print(f"Epoch {epoch+1}/{MAX_EPOCHS}: loss={avg_loss:.4f} | "
              f"dist={sample_dists.mean():.2f} (target={target_dists.mean():.2f}) | "
              f"std={samples.std():.2f} (target={DATA_STD:.2f})")

print("\n" + "="*80)
print("GENERATING FINAL SAMPLES AND VISUALIZATION")
print("="*80)

# Generate final samples
samples_norm = sample_ddpm(model, 2000).cpu().numpy()
samples = samples_norm * DATA_STD + DATA_MEAN

print(f"\nGenerated: mean={samples.mean():.2f}, std={samples.std():.2f}")
print(f"Real:      mean={deb_latents.mean():.2f}, std={deb_latents.std():.2f}")

sample_dists = pdist(samples)
print(f"\nDist ratio: {sample_dists.mean() / target_dists.mean():.3f}")

# PCA visualization
pca = PCA(n_components=2)
real_subset = deb_latents[np.random.choice(len(deb_latents), 2000, replace=False)]
combined = np.vstack([real_subset, samples])
pca.fit(combined)

real_pca = pca.transform(real_subset)
gen_pca = pca.transform(samples)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Real
ax = axes[0]
ax.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', alpha=0.3, s=5)
ax.set_title('Real DEB Latents', fontsize=14, fontweight='bold')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)

# Generated
ax = axes[1]
ax.scatter(gen_pca[:, 0], gen_pca[:, 1], c='red', alpha=0.3, s=5)
ax.set_title('Generated DEB Latents', fontsize=14, fontweight='bold')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)

# Overlay
ax = axes[2]
ax.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', alpha=0.2, s=5, label='Real')
ax.scatter(gen_pca[:, 0], gen_pca[:, 1], c='red', alpha=0.2, s=5, marker='x', label='Generated')
ax.set_title('Overlay Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle(f'DEB: {TIMESTEPS} Timesteps, Cosine Schedule', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'single_deb_1000steps.png'), dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: images/single_deb_1000steps.png")

# Save samples
np.save(os.path.join(RESULTS_DIR, 'single_deb_generated.npy'), samples)
print(f"✓ Saved: results/single_deb_generated.npy")

print("\nDONE! Check if tendrils appear in the PCA plot.")
