#!/usr/bin/env python3
"""
Generate PCA plot using pre-computed latent codes.
Shows: Real (circles) vs Diffusion (squares) vs Gaussian (triangles) by chemical.
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import json

print("Loading pre-computed latent codes...")
# Use separated latents if available (from fine-tuned encoder)
if os.path.exists('results/autoencoder_test_latent_separated.npy'):
    print("  → Using SEPARATED latents from fine-tuned encoder")
    test_latent = np.load('results/autoencoder_test_latent_separated.npy')
else:
    print("  → Using original latents")
    test_latent = np.load('results/autoencoder_test_latent.npy')
print(f"Test latent shape: {test_latent.shape}")

# Load labels
test_df = pd.read_feather('Data/test_data.feather')
onehot_cols = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
test_labels = test_df[onehot_cols].values.argmax(axis=1)

chemicals = onehot_cols
samples_per_class = 150

print("\nPreparing real latent data...")
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
real_labels_arr = np.array(real_labels_list)
print(f"Real latents: {real_latents.shape}")

print("\nLoading diffusion model and SMILE embeddings...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load SMILE embeddings
smile_df = pd.read_csv('Data/name_smiles_embedding_file.csv')
import ast

label_mapping = {
    'DEB': '1,2,3,4-Diepoxybutane',
    'DEM': 'Diethyl Malonate',
    'DMMP': 'Dimethyl methylphosphonate',
    'DPM': 'Oxybispropanol',
    'DtBP': 'Di-tert-butyl peroxide',
    'JP8': 'JP8',
    'MES': '2-(N-morpholino)ethanesulfonic acid',
    'TEPO': 'Triethyl phosphate'
}

embedding_dict = {}
for _, row in smile_df.iterrows():
    if pd.notna(row['embedding']):
        try:
            embedding = np.array(ast.literal_eval(row['embedding']), dtype=np.float32)
            embedding_dict[row['Name']] = embedding
        except:
            pass

smile_embeddings = {}
for label, full_name in label_mapping.items():
    if full_name in embedding_dict:
        smile_embeddings[label] = embedding_dict[full_name]

# Import diffusion model from training script (without running it)
import sys
sys.path.insert(0, 'scripts')

# Minimal imports to avoid running training code
import torch.nn as nn
from collections import OrderedDict
import re

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class ClassConditionedDiffusion(nn.Module):
    def __init__(self, latent_dim=512, smile_dim=512, num_classes=8, 
                 timesteps=50, hidden_dim=512, num_layers=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        
        # Diffusion schedule
        beta_start, beta_end = 0.0001, 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        
        # Time embedding
        time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Input projection: latent + SMILE + class_onehot
        input_dim = latent_dim + smile_dim + num_classes
        
        # Network layers
        layers = []
        layers.append(nn.Linear(input_dim + time_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t, smile_emb, class_onehot):
        """
        Args:
            x: (batch, latent_dim) - noisy latent
            t: (batch,) - timesteps
            smile_emb: (batch, smile_dim) - SMILE embeddings
            class_onehot: (batch, num_classes) - one-hot class labels
        Returns:
            predicted_noise: (batch, latent_dim)
        """
        # Get time embedding
        t_emb = self.time_mlp(t)
        
        # Concatenate all inputs
        x_in = torch.cat([x, smile_emb, class_onehot], dim=1)
        x_in = torch.cat([x_in, t_emb], dim=1)
        
        # Predict noise
        return self.net(x_in)

print("Initializing diffusion model...")
diffusion = ClassConditionedDiffusion(
    latent_dim=512,
    smile_dim=512,
    num_classes=8,
    timesteps=50,
    hidden_dim=512,
    num_layers=6
).to(device)

# Load trained model if it exists
diffusion_path = 'models/diffusion_latent_separated_best.pt'
if not os.path.exists(diffusion_path):
    diffusion_path = 'models/diffusion_separation_best.pt'  # fallback to old name

if os.path.exists(diffusion_path):
    print(f"Loading trained diffusion model from: {diffusion_path}")
    checkpoint = torch.load(diffusion_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    new_sd = OrderedDict()
    for k, v in checkpoint.items():
        new_sd[re.sub(r'^module\.', '', k)] = v
    diffusion.load_state_dict(new_sd, strict=False)

diffusion.eval()

# Generate diffusion samples
print("\nGenerating diffusion samples...")
diffusion_latents = []
diffusion_labels_list = []

with torch.no_grad():
    for class_idx, chem in enumerate(chemicals):
        if chem not in smile_embeddings:
            print(f"  Skipping {chem}: no SMILE embedding")
            continue
        
        smile_emb_np = smile_embeddings[chem]
        smile_emb = torch.FloatTensor(smile_emb_np).unsqueeze(0).repeat(samples_per_class, 1).to(device)
        class_onehot = torch.zeros(samples_per_class, 8, device=device)
        class_onehot[:, class_idx] = 1.0
        
        # Denoise from noise using DDPM sampling
        z_t = torch.randn(samples_per_class, 512, device=device)
        
        for t in reversed(range(diffusion.timesteps)):
            t_batch = torch.full((samples_per_class,), t, dtype=torch.long, device=device)
            
            # Predict noise
            predicted_noise = diffusion(z_t, t_batch, smile_emb, class_onehot)
            
            # Get schedule values
            alpha_t = diffusion.alphas[t]
            alpha_bar_t = diffusion.alphas_cumprod[t]
            beta_t = diffusion.betas[t]
            
            if t > 0:
                alpha_bar_prev = diffusion.alphas_cumprod[t-1]
                
                # DDPM sampling: predict x_0 from x_t and noise
                x_0_pred = (z_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
                
                # Direction pointing to x_t
                dir_xt = torch.sqrt(1 - alpha_bar_prev) * predicted_noise
                
                # Compute x_{t-1}
                z_t = torch.sqrt(alpha_bar_prev) * x_0_pred + dir_xt
                
                # Add noise for stochasticity (this maintains variance!)
                z_t = z_t + torch.sqrt(beta_t) * torch.randn_like(z_t)
            else:
                # Final step - no noise added
                z_t = (z_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        diffusion_latents.append(z_t.cpu().numpy())
        diffusion_labels_list.extend([class_idx] * samples_per_class)

if diffusion_latents:
    diffusion_latents = np.vstack(diffusion_latents)
    diffusion_labels_arr = np.array(diffusion_labels_list)
    print(f"Diffusion latents: {diffusion_latents.shape}")
    
    # DIAGNOSTIC: Check if samples have variance
    from scipy.spatial.distance import pdist
    for class_idx in range(8):
        mask = diffusion_labels_arr == class_idx
        class_samples = diffusion_latents[mask]
        if len(class_samples) > 1:
            dists = pdist(class_samples)
            print(f"  Class {class_idx} ({chemicals[class_idx]}): mean_dist={dists.mean():.4f}, std={class_samples.std():.4f}")

else:
    print("Warning: No diffusion samples generated")
    diffusion_latents = np.zeros((0, 512))
    diffusion_labels_arr = np.array([])

# Generate Gaussian samples (from per-chemical fitted distributions)
print("\nGenerating Gaussian samples (from fitted distributions)...")
gaussian_latents = []
gaussian_labels_arr_list = []

for class_idx in range(8):
    mask = real_labels_arr == class_idx
    if mask.sum() == 0:
        continue
    
    class_real = real_latents[mask]
    mu = class_real.mean(axis=0)
    sigma = class_real.std(axis=0)
    
    samples = np.random.normal(mu, sigma, size=(samples_per_class, 512))
    gaussian_latents.append(samples)
    gaussian_labels_arr_list.extend([class_idx] * samples_per_class)

gaussian_latents = np.vstack(gaussian_latents)
gaussian_labels_arr = np.array(gaussian_labels_arr_list)
print(f"Gaussian latents: {gaussian_latents.shape}")

# PCA
print("\nComputing PCA...")
all_latents = np.vstack([real_latents, diffusion_latents, gaussian_latents])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_latents)

n_real = len(real_latents)
n_diffusion = len(diffusion_latents)
real_pca = all_pca[:n_real]
diffusion_pca = all_pca[n_real:n_real+n_diffusion]
gaussian_pca = all_pca[n_real+n_diffusion:]

print(f"PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}")

# Plot 1: With Gaussian samples (all three sources)
print("\nCreating visualization WITH Gaussian samples...")
fig, ax = plt.subplots(1, 1, figsize=(18, 13))
colors = plt.cm.tab10(np.linspace(0, 1, 8))

for class_idx, chem in enumerate(chemicals):
    color = colors[class_idx]
    
    # Real (circles)
    mask = real_labels_arr == class_idx
    if mask.sum() > 0:
        ax.scatter(real_pca[mask, 0], real_pca[mask, 1], c=[color], marker='o', 
                  s=70, alpha=0.7, edgecolors='black', linewidth=0.7)
    
    # Diffusion (squares)
    if len(diffusion_latents) > 0:
        mask = diffusion_labels_arr == class_idx
        if mask.sum() > 0:
            ax.scatter(diffusion_pca[mask, 0], diffusion_pca[mask, 1], c=[color], marker='s',
                      s=70, alpha=0.7, edgecolors='black', linewidth=0.7)
    
    # Gaussian (triangles)
    mask = gaussian_labels_arr == class_idx
    if mask.sum() > 0:
        ax.scatter(gaussian_pca[mask, 0], gaussian_pca[mask, 1], c=[color], marker='^',
                  s=70, alpha=0.5, edgecolors='black', linewidth=0.7)

# Legends
from matplotlib.lines import Line2D

chemical_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                         markersize=12, label=chem, markeredgecolor='black', markeredgewidth=0.7)
                  for i, chem in enumerate(chemicals)]

source_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=12, 
           label='Real (Encoder)', markeredgecolor='black', markeredgewidth=0.7),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=12, 
           label='Diffusion (Conditioned)', markeredgecolor='black', markeredgewidth=0.7),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=12, 
           label='Gaussian (Fitted per Chemical)', markeredgecolor='black', markeredgewidth=0.7)
]

first_legend = ax.legend(handles=chemical_legend, title='Chemicals', 
                        loc='upper left', bbox_to_anchor=(1.02, 1), 
                        frameon=True, fontsize=13, title_fontsize=14)
ax.add_artist(first_legend)
ax.legend(handles=source_legend, title='Source', 
         loc='upper left', bbox_to_anchor=(1.02, 0.5), 
         frameon=True, fontsize=13, title_fontsize=14)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
             fontsize=16, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
             fontsize=16, fontweight='bold')
ax.set_title('Latent Space PCA: Chemical-Specific Real, Diffusion, and Gaussian Samples\n' + 
            'Circles = Real Data | Squares = Diffusion Samples | Triangles = Gaussian Samples', 
            fontsize=17, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f9f9f9')

plt.tight_layout()
plt.savefig('images/pca_real_vs_diffusion_vs_gaussian_by_chemical.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: images/pca_real_vs_diffusion_vs_gaussian_by_chemical.png")

# Plot 2: WITHOUT Gaussian samples (only real vs diffusion)
print("\nCreating visualization WITHOUT Gaussian samples...")
fig, ax = plt.subplots(1, 1, figsize=(18, 13))

for class_idx, chem in enumerate(chemicals):
    color = colors[class_idx]
    
    # Real (circles)
    mask = real_labels_arr == class_idx
    if mask.sum() > 0:
        ax.scatter(real_pca[mask, 0], real_pca[mask, 1], c=[color], marker='o', 
                  s=100, alpha=0.8, edgecolors='black', linewidth=1.0, label=f'{chem} (Real)')
    
    # Diffusion (squares)
    if len(diffusion_latents) > 0:
        mask = diffusion_labels_arr == class_idx
        if mask.sum() > 0:
            ax.scatter(diffusion_pca[mask, 0], diffusion_pca[mask, 1], c=[color], marker='s',
                      s=100, alpha=0.8, edgecolors='black', linewidth=1.0)

# Legends
chemical_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                         markersize=14, label=chem, markeredgecolor='black', markeredgewidth=1.0)
                  for i, chem in enumerate(chemicals)]

source_legend = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=14, 
           label='Real (Encoder)', markeredgecolor='black', markeredgewidth=1.0),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=14, 
           label='Diffusion (Conditioned)', markeredgecolor='black', markeredgewidth=1.0)
]

first_legend = ax.legend(handles=chemical_legend, title='Chemicals', 
                        loc='upper left', bbox_to_anchor=(1.02, 1), 
                        frameon=True, fontsize=14, title_fontsize=15)
ax.add_artist(first_legend)
ax.legend(handles=source_legend, title='Source', 
         loc='upper left', bbox_to_anchor=(1.02, 0.65), 
         frameon=True, fontsize=14, title_fontsize=15)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
             fontsize=17, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', 
             fontsize=17, fontweight='bold')
ax.set_title('Latent Space PCA: Real vs Diffusion Samples by Chemical\n' + 
            'Circles = Real Encoder Outputs | Squares = Diffusion Generated Samples', 
            fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f9f9f9')

plt.tight_layout()
plt.savefig('images/pca_real_vs_diffusion_by_chemical.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: images/pca_real_vs_diffusion_by_chemical.png")

print("\n" + "=" * 80)
print("✓ BOTH plots saved:")
print("  1. images/pca_real_vs_diffusion_vs_gaussian_by_chemical.png (with Gaussian)")
print("  2. images/pca_real_vs_diffusion_by_chemical.png (without Gaussian)")
print("=" * 80)

# Save metrics
metrics = {
    'explained_variance_pc1': float(pca.explained_variance_ratio_[0]),
    'explained_variance_pc2': float(pca.explained_variance_ratio_[1]),
    'total_variance': float(pca.explained_variance_ratio_[:2].sum()),
    'chemicals': chemicals,
    'samples_per_class_per_source': samples_per_class,
    'note': 'Gaussian samples drawn from N(μ, σ) fitted to each chemical\'s real latent distribution'
}
with open('results/pca_by_chemical_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Done!")
