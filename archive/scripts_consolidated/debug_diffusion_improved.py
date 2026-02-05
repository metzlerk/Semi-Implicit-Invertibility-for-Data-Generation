#!/usr/bin/env python3
"""
Debug diffusion collapse - Try variance-preserving formulation
The issue: Standard DDPM parameterization may lose variance
Solution: Use variance-preserving (VP) formulation with larger model
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

# Configuration
LATENT_DIM = 512
TIMESTEPS = 1000  # Increase timesteps for smoother schedule
HIDDEN_DIM = 1024  # Increase capacity
NUM_LAYERS = 12    # Deeper network
LEARNING_RATE = 5e-5
BATCH_SIZE = 128
MAX_EPOCHS = 300

# =============================================================================
# MODEL - Variance Preserving Formulation
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

class ImprovedDiffusion(nn.Module):
    """Variance-preserving diffusion with larger capacity"""
    def __init__(self, latent_dim=512, timesteps=1000, hidden_dim=1024, num_layers=12):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        
        # Time embedding
        time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Main network with residual connections
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for time concat
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Cosine schedule (better than linear for VP)
        betas = self.cosine_beta_schedule(timesteps)
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
    
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t.float())
        
        # Input projection
        h = self.input_proj(x)
        
        # Residual blocks with time conditioning
        for block in self.blocks:
            h_with_time = torch.cat([h, t_emb], dim=-1)
            h = h + block(h_with_time)
        
        # Output projection
        return self.output_proj(h)

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
print(f"  Range: [{deb_latents.min():.4f}, {deb_latents.max():.4f}]")

sample_subset = deb_latents[np.random.choice(len(deb_latents), min(500, len(deb_latents)), replace=False)]
dists = pdist(sample_subset)
print(f"  Pairwise distances: mean={dists.mean():.2f}, std={dists.std():.2f}")

TARGET_MEAN_DIST = dists.mean()
TARGET_STD = deb_latents.std()

train_tensor = torch.FloatTensor(deb_latents)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =============================================================================
# TRAINING & SAMPLING
# =============================================================================

def forward_diffusion(x_0, t, model):
    """Variance-preserving forward process"""
    sqrt_alpha_bar = model.sqrt_alphas_cumprod[t].reshape(-1, 1)
    sqrt_one_minus_alpha_bar = model.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
    
    noise = torch.randn_like(x_0)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    
    return x_t, noise

@torch.no_grad()
def sample_ddim(model, n_samples=100, ddim_steps=50):
    """DDIM sampling (deterministic, often better than DDPM)"""
    model.eval()
    
    # Start from noise
    x_t = torch.randn(n_samples, LATENT_DIM, device=device)
    
    # Create DDIM timestep schedule
    step_size = model.timesteps // ddim_steps
    timesteps = list(range(0, model.timesteps, step_size))[::-1]
    
    for i, t in enumerate(timesteps):
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
        
        # Predict noise
        predicted_noise = model(x_t, t_batch)
        
        # Predict x_0
        alpha_bar_t = model.alphas_cumprod[t]
        x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        if i < len(timesteps) - 1:
            # Get next alpha
            t_next = timesteps[i + 1]
            alpha_bar_next = model.alphas_cumprod[t_next]
            
            # DDIM update (deterministic)
            x_t = torch.sqrt(alpha_bar_next) * x_0_pred + torch.sqrt(1 - alpha_bar_next) * predicted_noise
        else:
            x_t = x_0_pred
    
    return x_t.cpu().numpy()

def evaluate_samples(samples, target_mean_dist, target_std):
    """Evaluate sample quality"""
    if len(samples) < 2:
        return 0, 0, 0
    
    sample_std = samples.std()
    sample_mean_dist = pdist(samples[:min(500, len(samples))]).mean()
    
    std_ratio = sample_std / target_std
    dist_ratio = sample_mean_dist / target_mean_dist
    score = (std_ratio + dist_ratio) / 2
    
    return sample_mean_dist, sample_std, score

# =============================================================================
# TRAINING
# =============================================================================

print("\n" + "="*80)
print("TRAINING IMPROVED VARIANCE-PRESERVING DIFFUSION")
print(f"Architecture: {NUM_LAYERS} layers, {HIDDEN_DIM} hidden dim")
print(f"Schedule: Cosine, {TIMESTEPS} timesteps")
print("="*80)

model = ImprovedDiffusion(LATENT_DIM, TIMESTEPS, HIDDEN_DIM, NUM_LAYERS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)

best_score = float('inf')
best_epoch = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0
    n_batches = 0
    
    for (batch_latent,) in train_loader:
        batch_latent = batch_latent.to(device)
        
        # Sample random timesteps
        t = torch.randint(0, TIMESTEPS, (len(batch_latent),), device=device)
        
        # Forward diffusion
        x_noisy, noise = forward_diffusion(batch_latent, t, model)
        
        # Predict noise
        predicted_noise = model(x_noisy, t)
        
        # Loss
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    avg_loss = epoch_loss / n_batches
    
    # Evaluate every 20 epochs
    if (epoch + 1) % 20 == 0:
        samples = sample_ddim(model, n_samples=500, ddim_steps=50)
        mean_dist, std, score = evaluate_samples(samples, TARGET_MEAN_DIST, TARGET_STD)
        
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, mean_dist={mean_dist:.2f} (target={TARGET_MEAN_DIST:.2f}), "
              f"std={std:.2f} (target={TARGET_STD:.2f}), score={score:.3f}")
        
        if abs(score - 1.0) < abs(best_score - 1.0):
            best_score = score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'models/diffusion_debug_best.pt')

print(f"\nBest score: {best_score:.3f} at epoch {best_epoch}")

# Final evaluation
final_samples = sample_ddim(model, n_samples=500, ddim_steps=50)
final_mean_dist, final_std, final_score = evaluate_samples(final_samples, TARGET_MEAN_DIST, TARGET_STD)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Mean distance: {final_mean_dist:.2f} (target: {TARGET_MEAN_DIST:.2f}, ratio: {final_mean_dist/TARGET_MEAN_DIST:.3f})")
print(f"Std: {final_std:.2f} (target: {TARGET_STD:.2f}, ratio: {final_std/TARGET_STD:.3f})")
print(f"Score: {final_score:.3f}")

# Visualize
from sklearn.decomposition import PCA

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

real_subset = deb_latents[np.random.choice(len(deb_latents), 500, replace=False)]
all_data = np.vstack([real_subset, final_samples])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_data)

axes[0].scatter(all_pca[:500, 0], all_pca[:500, 1], alpha=0.5, s=30, label='Real', color='blue')
axes[0].scatter(all_pca[500:, 0], all_pca[500:, 1], alpha=0.5, s=30, label='Diffusion', color='red')
axes[0].set_title(f'PCA: Real vs Improved Diffusion (DEB)\nCosine schedule, {TIMESTEPS} steps, DDIM sampling', fontweight='bold')
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
plt.savefig('images/diffusion_improved_single_chemical.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: images/diffusion_improved_single_chemical.png")

if final_score > 0.8 and final_score < 1.2:
    print("\n✓ SUCCESS! Variance-preserving formulation works!")
    print(f"  Use this architecture for full training")
else:
    print(f"\n⚠ Score {final_score:.3f} - still needs improvement")
