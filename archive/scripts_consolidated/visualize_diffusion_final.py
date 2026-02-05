#!/usr/bin/env python3
"""
Comprehensive visualization of best diffusion model
- PCA comparison (real vs diffusion)
- Decoded spectra comparison
- Distribution statistics
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LATENT_DIM = 512
TIMESTEPS = 50
HIDDEN_DIM = 512
NUM_LAYERS = 6

# =============================================================================
# MODEL DEFINITIONS
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

class FlexibleNLayersGenerator(nn.Module):
    def __init__(self, latent_dim=512, output_dim=1676, num_layers=9):
        super().__init__()
        layer_sizes = np.linspace(latent_dim, output_dim, num_layers + 1, dtype=int)
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < num_layers - 1:
                layers.append(nn.LeakyReLU())
        
        self.generator = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.generator(z)

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data...")
test_latent = np.load('results/autoencoder_test_latent_separated.npy')
test_df = pd.read_feather('Data/test_data.feather')
test_labels = test_df[['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']].values.argmax(axis=1)

# Get DEB data
deb_mask = test_labels == 0
deb_latents = test_latent[deb_mask]
deb_spectra = test_df[deb_mask].iloc[:, :1676].values

print(f"DEB samples: {len(deb_latents)}")
print(f"  Mean: {deb_latents.mean():.4f}, Std: {deb_latents.std():.4f}")

# =============================================================================
# LOAD MODELS
# =============================================================================

print("\nLoading models...")
checkpoint = torch.load('models/diffusion_normalized_best.pt', map_location=device, weights_only=False)
DATA_MEAN = checkpoint['data_mean']
DATA_STD = checkpoint['data_std']

diffusion = DiffusionModel(LATENT_DIM, TIMESTEPS, HIDDEN_DIM, NUM_LAYERS).to(device)
diffusion.load_state_dict(checkpoint['model_state'])
diffusion.eval()

generator = FlexibleNLayersGenerator(latent_dim=512, output_dim=1676, num_layers=9).to(device)
ae_state = torch.load('models/autoencoder_separated.pth', map_location=device, weights_only=False)
# Load only the generator weights (ignore bias_layer)
generator_weights = {k.replace('generator.', ''): v for k, v in ae_state['generator_state_dict'].items() 
                     if k.startswith('generator.')}
generator.generator.load_state_dict(generator_weights)
generator.eval()

print(f"Normalization: mean={DATA_MEAN:.4f}, std={DATA_STD:.4f}")

# =============================================================================
# GENERATE SAMPLES
# =============================================================================

@torch.no_grad()
def sample_ddpm(model, n_samples=1000):
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

print("\nGenerating diffusion samples...")
diffusion_samples_norm = sample_ddpm(diffusion, n_samples=1000)
diffusion_samples = diffusion_samples_norm * DATA_STD + DATA_MEAN

print(f"Diffusion samples mean: {diffusion_samples.mean():.4f}, std: {diffusion_samples.std():.4f}")

# Compute metrics
real_subset = deb_latents[np.random.choice(len(deb_latents), 1000, replace=False)]
real_mean_dist = pdist(real_subset[:500]).mean()
real_std = real_subset.std()
diff_mean_dist = pdist(diffusion_samples[:500]).mean()
diff_std = diffusion_samples.std()

print(f"\nMetrics:")
print(f"  Real: mean_dist={real_mean_dist:.2f}, std={real_std:.2f}")
print(f"  Diffusion: mean_dist={diff_mean_dist:.2f}, std={diff_std:.2f}")
print(f"  Ratios: dist={diff_mean_dist/real_mean_dist:.3f}, std={diff_std/real_std:.3f}")
print(f"  Score: {((diff_std/real_std + diff_mean_dist/real_mean_dist)/2):.3f}")

# =============================================================================
# DECODE TO SPECTRA
# =============================================================================

print("\nDecoding samples to spectra...")
n_decode = 100
real_decode_latents = torch.FloatTensor(real_subset[:n_decode]).to(device)
diff_decode_latents = torch.FloatTensor(diffusion_samples[:n_decode]).to(device)

with torch.no_grad():
    real_decoded = generator(real_decode_latents).cpu().numpy()
    diff_decoded = generator(diff_decode_latents).cpu().numpy()

# =============================================================================
# VISUALIZATIONS
# =============================================================================

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. PCA
ax1 = fig.add_subplot(gs[0, 0])
all_data = np.vstack([real_subset[:500], diffusion_samples[:500]])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_data)

ax1.scatter(all_pca[:500, 0], all_pca[:500, 1], alpha=0.4, s=20, label='Real', color='blue')
ax1.scatter(all_pca[500:, 0], all_pca[500:, 1], alpha=0.4, s=20, label='Diffusion', color='red')
ax1.set_title(f'PCA: Real vs Diffusion (DEB)\nScore: {((diff_std/real_std + diff_mean_dist/real_mean_dist)/2):.3f}', 
              fontweight='bold', fontsize=11)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Value distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(real_subset.flatten(), bins=50, alpha=0.5, label='Real', density=True, color='blue')
ax2.hist(diffusion_samples.flatten(), bins=50, alpha=0.5, label='Diffusion', density=True, color='red')
ax2.set_title(f'Latent Value Distribution\nReal std={real_std:.2f}, Diff std={diff_std:.2f} (ratio={diff_std/real_std:.3f})', 
              fontweight='bold', fontsize=11)
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Pairwise distances
ax3 = fig.add_subplot(gs[0, 2])
real_dists = pdist(real_subset[:500])
diff_dists = pdist(diffusion_samples[:500])
ax3.hist(real_dists, bins=50, alpha=0.5, label='Real', density=True, color='blue')
ax3.hist(diff_dists, bins=50, alpha=0.5, label='Diffusion', density=True, color='red')
ax3.set_title(f'Pairwise Distances\nReal={real_mean_dist:.1f}, Diff={diff_mean_dist:.1f} (ratio={diff_mean_dist/real_mean_dist:.3f})', 
              fontweight='bold', fontsize=11)
ax3.set_xlabel('Distance')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4-6. Decoded spectra (3 random samples)
for i in range(3):
    ax = fig.add_subplot(gs[1, i])
    idx = np.random.randint(0, n_decode)
    
    x_axis = np.arange(1676)
    ax.plot(x_axis, deb_spectra[idx], label='Original Real', alpha=0.7, linewidth=1.5, color='green')
    ax.plot(x_axis, real_decoded[idx], label='Real→Decoded', alpha=0.7, linewidth=1.5, color='blue')
    ax.plot(x_axis, diff_decoded[idx], label='Diffusion→Decoded', alpha=0.7, linewidth=1.5, color='red')
    
    ax.set_title(f'Decoded Spectrum {i+1} (DEB)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Intensity')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Diffusion Model Evaluation - Normalized Training (β: 0.00002→0.005, 50 steps)', 
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig('images/diffusion_final_comprehensive.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: images/diffusion_final_comprehensive.png")

# =============================================================================
# DETAILED SPECTRA COMPARISON
# =============================================================================

fig2, axes = plt.subplots(4, 2, figsize=(16, 14))
axes = axes.flatten()

for i in range(8):
    idx = np.random.randint(0, n_decode)
    x_axis = np.arange(1676)
    
    axes[i].plot(x_axis, deb_spectra[idx], label='Original Real', alpha=0.8, linewidth=1.2, color='green')
    axes[i].plot(x_axis, real_decoded[idx], label='Real→Decoded', alpha=0.8, linewidth=1.2, color='blue')
    axes[i].plot(x_axis, diff_decoded[idx], label='Diffusion→Decoded', alpha=0.8, linewidth=1.2, color='red')
    
    axes[i].set_title(f'Sample {i+1}', fontweight='bold')
    axes[i].set_xlabel('Feature Index')
    axes[i].set_ylabel('Intensity')
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Detailed Decoded Spectra Comparison - DEB Chemical', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/diffusion_final_spectra_detailed.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/diffusion_final_spectra_detailed.png")

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Model: Diffusion with normalized training (β: 0.00002→0.005, 50 timesteps)")
print(f"Normalization: mean={DATA_MEAN:.4f}, std={DATA_STD:.4f}")
print(f"\nDistribution Matching:")
print(f"  Std ratio: {diff_std/real_std:.3f} (target: 1.000)")
print(f"  Mean_dist ratio: {diff_mean_dist/real_mean_dist:.3f} (target: 1.000)")
print(f"  Overall score: {((diff_std/real_std + diff_mean_dist/real_mean_dist)/2):.3f}")
print(f"\nVisualization saved to:")
print(f"  - images/diffusion_final_comprehensive.png")
print(f"  - images/diffusion_final_spectra_detailed.png")
