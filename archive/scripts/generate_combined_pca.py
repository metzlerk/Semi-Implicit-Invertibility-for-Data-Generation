#!/usr/bin/env python3
"""
Generate a SINGLE PCA plot with all 8 chemicals, showing:
- Real data (circles)
- Diffusion samples (squares)  
- Gaussian samples (triangles)
Each chemical has its own color.
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import sys
sys.path.insert(0, 'scripts')

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json

from diffusion_ims_improved_separation import (
    FlexibleNLayersEncoder,
    ClassConditionedDiffusion,
    load_smile_embeddings,
    load_ims_data,
    _load_state_dict_flex
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load data
print("Loading data...")
train_ims, test_ims, train_labels, test_labels, train_smiles, test_smiles = load_ims_data()
smile_embeddings = load_smile_embeddings()
chemicals = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']

# Load encoder
print("Loading encoder...")
encoder = FlexibleNLayersEncoder(input_size=test_ims.shape[1], output_size=512, n_layers=9).to(device)
checkpoint = torch.load('models/flexible_9layers_encoder.pt', map_location=device)
encoder = _load_state_dict_flex(encoder, checkpoint)
encoder.eval()

# Encode real data
print("Encoding real data...")
samples_per_class = 150  # Reduced to minimize memory usage
real_latents = []
real_labels_list = []

with torch.no_grad():
    for class_idx in range(8):
        mask = test_labels == class_idx
        class_data = test_ims[mask]
        n_samples = min(samples_per_class, len(class_data))
        indices = np.random.choice(len(class_data), n_samples, replace=False)
        sampled = class_data[indices]
        sampled_tensor = torch.FloatTensor(sampled).to(device)
        latent = encoder(sampled_tensor)
        real_latents.append(latent.cpu().numpy())
        real_labels_list.extend([class_idx] * n_samples)

real_latents = np.vstack(real_latents)
real_labels_arr = np.array(real_labels_list)

# Load diffusion model
print("Loading diffusion model...")
diffusion = ClassConditionedDiffusion(
    latent_dim=512,
    smile_dim=512,
    num_classes=8,
    timesteps=50,
    hidden_dim=512,
    num_layers=6
).to(device)
checkpoint = torch.load('models/diffusion_separation_best.pt', map_location=device)
diffusion.load_state_dict(checkpoint['model_state_dict'])
diffusion.eval()

# Generate diffusion samples
print("Generating diffusion samples...")
diffusion_latents = []
diffusion_labels_list = []

with torch.no_grad():
    for class_idx, chem in enumerate(chemicals):
        smile_emb_np = smile_embeddings[chem]
        smile_emb = torch.from_numpy(smile_emb_np).unsqueeze(0).repeat(samples_per_class, 1).to(device)
        class_labels_tens = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
        z_t = torch.randn(samples_per_class, 512, device=device)
        
        for t in reversed(range(diffusion.timesteps)):
            t_batch = torch.full((samples_per_class,), t, dtype=torch.long, device=device)
            pred_z0, _ = diffusion(z_t, t_batch, smile_emb, class_labels_tens)
            
            if t > 0:
                alpha_t = diffusion.alphas[t]
                alpha_prev = diffusion.alphas[t-1]
                beta_t = diffusion.betas[t]
                z_t = (torch.sqrt(alpha_prev) * beta_t / (1 - alpha_t)) * pred_z0 + \
                      (torch.sqrt(1 - beta_t) * (1 - alpha_prev) / (1 - alpha_t)) * z_t
                if t > 1:
                    z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
            else:
                z_t = pred_z0
        
        diffusion_latents.append(z_t.cpu().numpy())
        diffusion_labels_list.extend([class_idx] * samples_per_class)

diffusion_latents = np.vstack(diffusion_latents)
diffusion_labels_arr = np.array(diffusion_labels_list)

# Generate Gaussian samples fitted to each chemical's real distribution
print("Fitting Gaussians to real data and generating samples...")
gaussian_latents = []
gaussian_labels_arr_list = []

for class_idx in range(8):
    # Get real latents for this chemical
    mask = real_labels_arr == class_idx
    class_real_latents = real_latents[mask]
    
    # Fit Gaussian: compute mean and covariance
    mu = class_real_latents.mean(axis=0)
    sigma = class_real_latents.std(axis=0)
    
    # Draw samples from N(mu, sigma)
    samples = np.random.normal(mu, sigma, size=(samples_per_class, 512))
    gaussian_latents.append(samples)
    gaussian_labels_arr_list.extend([class_idx] * samples_per_class)

gaussian_latents = np.vstack(gaussian_latents)
gaussian_labels_arr = np.array(gaussian_labels_arr_list)

# PCA
print("Computing PCA...")
all_latents = np.vstack([real_latents, diffusion_latents, gaussian_latents])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_latents)

n_real = len(real_latents)
n_diffusion = len(diffusion_latents)
real_pca = all_pca[:n_real]
diffusion_pca = all_pca[n_real:n_real+n_diffusion]
gaussian_pca = all_pca[n_real+n_diffusion:]

print(f"PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}\n")

# Plot
print("Creating combined plot...")
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
           label='Diffusion', markeredgecolor='black', markeredgewidth=0.7),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=12, 
           label='Gaussian', markeredgecolor='black', markeredgewidth=0.7)
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
ax.set_title('Latent Space PCA by Chemical and Source\n' + 
            'Circles = Real Data, Squares = Diffusion, Triangles = Gaussian', 
            fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f9f9f9')

plt.tight_layout()
plt.savefig('images/combined_pca_by_chemical_and_source.png', dpi=300, bbox_inches='tight')

print("=" * 80)
print("Plot saved: images/combined_pca_by_chemical_and_source.png")
print("=" * 80)

# Save metrics
metrics = {
    'explained_variance_pc1': float(pca.explained_variance_ratio_[0]),
    'explained_variance_pc2': float(pca.explained_variance_ratio_[1]),
    'total_variance': float(pca.explained_variance_ratio_[:2].sum()),
    'chemicals': chemicals,
    'samples_per_class_per_source': samples_per_class
}
with open('results/combined_pca_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Done!")
