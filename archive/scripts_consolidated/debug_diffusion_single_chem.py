#!/usr/bin/env python3
"""
Debug diffusion collapse - Single chemical (DEB) version
Iterate until diffusion captures real distribution
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
# CONFIGURATION
# =============================================================================

LATENT_DIM = 512
TIMESTEPS = 50
HIDDEN_DIM = 512
NUM_LAYERS = 6
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
MAX_EPOCHS = 200
BETA_START = 0.01   # Will experiment with this
BETA_END = 1.0      # Will experiment with this

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
    """Simplified diffusion model for single chemical"""
    def __init__(self, latent_dim=512, timesteps=50, hidden_dim=512, num_layers=6):
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
        
        # Main network: [noisy_latent, time_emb] -> predicted_noise
        input_dim = latent_dim + hidden_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Beta schedule
        betas = torch.linspace(BETA_START, BETA_END, timesteps)
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t.float())
        x_in = torch.cat([x, t_emb], dim=1)
        return self.net(x_in)

# =============================================================================
# DATA
# =============================================================================

print("Loading DEB latent data...")
test_latent = np.load('results/autoencoder_test_latent_separated.npy')
test_df = pd.read_feather('Data/test_data.feather')
test_labels = test_df[['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']].values.argmax(axis=1)

# Get DEB samples (class 0)
deb_mask = test_labels == 0
deb_latents = test_latent[deb_mask]

print(f"DEB samples: {len(deb_latents)}")
print(f"  Mean: {deb_latents.mean():.4f}")
print(f"  Std: {deb_latents.std():.4f}")
print(f"  Range: [{deb_latents.min():.4f}, {deb_latents.max():.4f}]")

# Check pairwise distances
sample_subset = deb_latents[np.random.choice(len(deb_latents), min(500, len(deb_latents)), replace=False)]
dists = pdist(sample_subset)
print(f"  Pairwise distances: mean={dists.mean():.2f}, std={dists.std():.2f}")

TARGET_MEAN_DIST = dists.mean()
TARGET_STD = deb_latents.std()

# Create dataloader
train_tensor = torch.FloatTensor(deb_latents)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =============================================================================
# DIFFUSION PROCESS
# =============================================================================

def forward_diffusion(x_0, t, betas):
    """Add noise to x_0 according to timestep t"""
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    alpha_bar_t = alpha_bars[t].reshape(-1, 1)
    noise = torch.randn_like(x_0)
    
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    return x_t, noise

@torch.no_grad()
def sample_diffusion(model, n_samples=100):
    """Sample from diffusion model using DDPM"""
    model.eval()
    
    z_t = torch.randn(n_samples, LATENT_DIM, device=device)
    
    for t in reversed(range(model.timesteps)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
        predicted_noise = model(z_t, t_batch)
        
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
    
    return z_t.cpu().numpy()

def evaluate_samples(samples, target_mean_dist, target_std):
    """Evaluate how well samples match target distribution"""
    if len(samples) < 2:
        return 0, 0, 0
    
    sample_std = samples.std()
    sample_mean_dist = pdist(samples[:min(500, len(samples))]).mean()
    
    std_ratio = sample_std / target_std
    dist_ratio = sample_mean_dist / target_mean_dist
    
    # Combined score (closer to 1 is better)
    score = (std_ratio + dist_ratio) / 2
    
    return sample_mean_dist, sample_std, score

# =============================================================================
# TRAINING
# =============================================================================

def train_iteration(beta_start, beta_end, trial_name):
    """Train with specific beta schedule"""
    print(f"\n{'='*80}")
    print(f"TRIAL: {trial_name}")
    print(f"Beta schedule: [{beta_start:.4f}, {beta_end:.4f}]")
    print('='*80)
    
    # Update beta schedule
    global BETA_START, BETA_END
    BETA_START = beta_start
    BETA_END = beta_end
    
    # Initialize model
    model = SimpleDiffusion(LATENT_DIM, TIMESTEPS, HIDDEN_DIM, NUM_LAYERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    betas = torch.linspace(beta_start, beta_end, TIMESTEPS).to(device)
    
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
            x_noisy, noise = forward_diffusion(batch_latent, t, betas)
            
            # Predict noise
            predicted_noise = model(x_noisy, t)
            
            # Loss
            loss = nn.functional.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            samples = sample_diffusion(model, n_samples=500)
            mean_dist, std, score = evaluate_samples(samples, TARGET_MEAN_DIST, TARGET_STD)
            
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, mean_dist={mean_dist:.2f} (target={TARGET_MEAN_DIST:.2f}), "
                  f"std={std:.2f} (target={TARGET_STD:.2f}), score={score:.3f}")
            
            # Track best
            if abs(score - 1.0) < abs(best_score - 1.0):
                best_score = score
                best_epoch = epoch + 1
    
    # Final evaluation
    final_samples = sample_diffusion(model, n_samples=500)
    final_mean_dist, final_std, final_score = evaluate_samples(final_samples, TARGET_MEAN_DIST, TARGET_STD)
    
    print(f"\nFinal results:")
    print(f"  Mean distance: {final_mean_dist:.2f} (target: {TARGET_MEAN_DIST:.2f}, ratio: {final_mean_dist/TARGET_MEAN_DIST:.3f})")
    print(f"  Std: {final_std:.2f} (target: {TARGET_STD:.2f}, ratio: {final_std/TARGET_STD:.3f})")
    print(f"  Score: {final_score:.3f} (best: {best_score:.3f} at epoch {best_epoch})")
    
    return final_score, model, final_samples

# =============================================================================
# MAIN
# =============================================================================

print("\n" + "="*80)
print("GOAL: Match target distribution")
print(f"  Target mean pairwise distance: {TARGET_MEAN_DIST:.2f}")
print(f"  Target std: {TARGET_STD:.2f}")
print("="*80)

# Try different beta schedules - focus on conservative range
trials = [
    (0.0001, 0.02, "Original baseline"),
    (0.0005, 0.1, "5x increased"),
    (0.001, 0.2, "10x increased"),
    (0.0002, 0.05, "2-5x increased"),
    (0.0003, 0.08, "3-8x increased"),
]

results = []
for beta_start, beta_end, name in trials:
    score, model, samples = train_iteration(beta_start, beta_end, name)
    results.append((name, beta_start, beta_end, score, samples))
    
    # Early stopping if we get close
    if abs(score - 1.0) < 0.1:
        print(f"\n✓ Found good solution with {name}!")
        break

# Summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
for name, beta_start, beta_end, score, _ in results:
    print(f"{name:30s} | β:[{beta_start:.3f}, {beta_end:.3f}] | Score: {score:.3f}")

# Save best
best_idx = np.argmin([abs(score - 1.0) for _, _, _, score, _ in results])
best_name, best_beta_start, best_beta_end, best_score, best_samples = results[best_idx]

print(f"\nBest: {best_name} with score {best_score:.3f}")

# Visualize best
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PCA
from sklearn.decomposition import PCA
real_subset = deb_latents[np.random.choice(len(deb_latents), 500, replace=False)]
all_data = np.vstack([real_subset, best_samples])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_data)

axes[0].scatter(all_pca[:500, 0], all_pca[:500, 1], alpha=0.5, s=30, label='Real', color='blue')
axes[0].scatter(all_pca[500:, 0], all_pca[500:, 1], alpha=0.5, s=30, label='Diffusion', color='red')
axes[0].set_title(f'PCA: Real vs Diffusion (DEB)\n{best_name}', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribution comparison
axes[1].hist(real_subset.flatten(), bins=50, alpha=0.5, label='Real', density=True, color='blue')
axes[1].hist(best_samples.flatten(), bins=50, alpha=0.5, label='Diffusion', density=True, color='red')
axes[1].set_title(f'Value Distribution\nReal std={TARGET_STD:.2f}, Diffusion std={best_samples.std():.2f}', fontweight='bold')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('images/diffusion_debug_single_chemical.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: images/diffusion_debug_single_chemical.png")

print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)
if best_score > 0.8 and best_score < 1.2:
    print(f"✓ Beta schedule [{best_beta_start:.3f}, {best_beta_end:.3f}] works well!")
    print(f"  Use this for full multi-class training")
else:
    print(f"⚠ Best score {best_score:.3f} is still far from 1.0")
    print(f"  May need different approach:")
    print(f"  - Try different architecture (more layers, different hidden_dim)")
    print(f"  - Try different sampling strategy (DDIM instead of DDPM)")
    print(f"  - Check if training data has issues")
