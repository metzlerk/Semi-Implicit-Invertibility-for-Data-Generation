#!/usr/bin/env python3
"""
Generate combined PCA plot showing:
- Real data (circles) - actual encoder outputs
- Diffusion samples (squares) - from trained diffusion model (chemical-conditioned)
- Gaussian samples (triangles) - from fitted Gaussian per chemical

All in one plot with colors representing different chemicals.
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import ast
import json
import re
from collections import OrderedDict
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ==============================================================================
# MODEL DEFINITIONS (inline to avoid importing training script)
# ==============================================================================

class FlexibleNLayersEncoder(nn.Module):
    def __init__(self, input_size=1676, output_size=512, n_layers=9, init_style=None, bkg=None, trainable=False):
        super().__init__()
        # Optional bias layer
        if init_style == 'bkg' and bkg is not None:
            self.bias_layer = nn.Parameter(bkg.clone().detach() if torch.is_tensor(bkg) else torch.FloatTensor(bkg))
        else:
            self.bias_layer = None
        
        layers = OrderedDict()
        if n_layers > 1:
            size_reduction_per_layer = (input_size - output_size) / n_layers
            for i in range(n_layers - 1):
                layer_input_size = input_size - int(size_reduction_per_layer) * i
                layer_output_size = input_size - int(size_reduction_per_layer) * (i + 1)
                layers[f'fc{i}'] = nn.Linear(layer_input_size, layer_output_size)
                layers[f'relu{i}'] = nn.LeakyReLU(inplace=True)
            layers['final'] = nn.Linear(layer_output_size, output_size)
        else:
            layers['final'] = nn.Linear(input_size, output_size)
        self.encoder = nn.Sequential(layers)
    
    def forward(self, x, use_bias=False):
        if use_bias and self.bias_layer is not None:
            x = x + self.bias_layer
        return self.encoder(x)


class ClassConditionedDiffusion(nn.Module):
    def __init__(self, latent_dim=512, smile_dim=512, num_classes=8, timesteps=50, hidden_dim=512, num_layers=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
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
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )
        
        # SMILE embedding projection
        self.smile_proj = nn.Sequential(
            nn.Linear(smile_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Class embedding projection
        self.class_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim + 256 + hidden_dim + hidden_dim, hidden_dim)
        
        # Main network
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Output heads
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        self.class_head = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, noisy_latent, t, smile_embedding, class_onehot):
        t_normalized = t.float().view(-1, 1) / self.timesteps
        t_emb = self.time_mlp(t_normalized)
        smile_emb = self.smile_proj(smile_embedding)
        class_emb = self.class_proj(class_onehot)
        
        x = torch.cat([noisy_latent, t_emb, smile_emb, class_emb], dim=1)
        h = self.input_proj(x)
        
        for layer, norm in zip(self.layers, self.layer_norms):
            h = h + layer(norm(h))
        
        pred_latent = self.latent_head(h)
        class_logits = self.class_head(h)
        
        return pred_latent, class_logits


# ==============================================================================
# DATA LOADING
# ==============================================================================

print("Loading data...")
test_df = pd.read_feather('Data/test_data.feather')
smile_df = pd.read_csv('Data/name_smiles_embedding_file.csv')

# Extract spectra
p_cols = [c for c in test_df.columns if c.startswith('p_')]
n_cols = [c for c in test_df.columns if c.startswith('n_')]
test_ims = np.concatenate([test_df[p_cols].values, test_df[n_cols].values], axis=1)

# Extract labels
onehot_cols = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
test_labels = test_df[onehot_cols].values.argmax(axis=1)

# Load SMILE embeddings
chemicals = onehot_cols
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

print(f"Chemicals: {chemicals}\n")

# ==============================================================================
# LOAD MODELS
# ==============================================================================

print("Loading encoder...")
encoder = FlexibleNLayersEncoder(input_size=test_ims.shape[1], output_size=512, n_layers=9).to(device)
checkpoint = torch.load('/scratch/cmdunham/trained_models/spectrum/current_best_models/nine_layer_ims_to_chemnet_encoder.pth', map_location=device, weights_only=False)

# Extract encoder state dict
if isinstance(checkpoint, dict):
    if 'encoder_state_dict' in checkpoint:
        state_dict = checkpoint['encoder_state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Handle module prefix
new_sd = OrderedDict()
for k, v in state_dict.items():
    # Remove 'module.' prefix if present
    key = re.sub(r'^module\.', '', k)
    # Remove 'bias_layer.' since we don't have it
    if 'bias_layer' not in key:
        new_sd[key] = v

encoder.load_state_dict(new_sd, strict=False)
encoder.eval()

# ==============================================================================
# ENCODE REAL DATA
# ==============================================================================

print("Encoding real data...")
samples_per_class = 150
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
print(f"Real latents: {real_latents.shape}")

# ==============================================================================
# LOAD DIFFUSION MODEL
# ==============================================================================

print("Loading diffusion model...")
diffusion = ClassConditionedDiffusion(
    latent_dim=512,
    smile_dim=512,
    num_classes=8,
    timesteps=50,
    hidden_dim=512,
    num_layers=6
).to(device)

# Try to load from our trained model
diffusion_path = 'models/diffusion_separation_best.pt'
if os.path.exists(diffusion_path):
    checkpoint = torch.load(diffusion_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    new_sd = OrderedDict()
    for k, v in checkpoint.items():
        new_sd[re.sub(r'^module\.', '', k)] = v
    diffusion.load_state_dict(new_sd, strict=False)
else:
    print(f"Warning: diffusion model not found at {diffusion_path}, using untrained model")

diffusion.eval()

# ==============================================================================
# GENERATE DIFFUSION SAMPLES
# ==============================================================================

print("Generating diffusion samples...")
diffusion_latents = []
diffusion_labels_list = []

with torch.no_grad():
    for class_idx, chem in enumerate(chemicals):
        if chem not in smile_embeddings:
            continue
        
        smile_emb_np = smile_embeddings[chem]
        smile_emb = torch.FloatTensor(smile_emb_np).unsqueeze(0).repeat(samples_per_class, 1).to(device)
        class_labels_tens = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
        class_onehot = torch.zeros(samples_per_class, 8, device=device)
        class_onehot[:, class_idx] = 1.0
        
        # Start from noise and denoise
        z_t = torch.randn(samples_per_class, 512, device=device)
        
        for t in reversed(range(diffusion.timesteps)):
            t_batch = torch.full((samples_per_class,), t, dtype=torch.long, device=device)
            pred_z0, _ = diffusion(z_t, t_batch, smile_emb, class_onehot)
            
            if t > 0:
                alpha_t = diffusion.alphas[t]
                alpha_prev = diffusion.alphas[t-1] if t > 0 else torch.tensor(1.0, device=device)
                beta_t = diffusion.betas[t]
                
                # DDPM reverse step
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
print(f"Diffusion latents: {diffusion_latents.shape}")

# ==============================================================================
# GENERATE GAUSSIAN SAMPLES (per-chemical fitted Gaussians)
# ==============================================================================

print("Fitting Gaussians to real data and generating samples...")
gaussian_latents = []
gaussian_labels_arr_list = []

for class_idx in range(8):
    # Get real latents for this chemical
    mask = real_labels_arr == class_idx
    if mask.sum() == 0:
        continue
    
    class_real_latents = real_latents[mask]
    
    # Fit Gaussian: mean and standard deviation
    mu = class_real_latents.mean(axis=0)
    sigma = class_real_latents.std(axis=0)
    
    # Draw samples from N(mu, sigma)
    samples = np.random.normal(mu, sigma, size=(samples_per_class, 512))
    gaussian_latents.append(samples)
    gaussian_labels_arr_list.extend([class_idx] * samples_per_class)

gaussian_latents = np.vstack(gaussian_latents)
gaussian_labels_arr = np.array(gaussian_labels_arr_list)
print(f"Gaussian latents: {gaussian_latents.shape}")

# ==============================================================================
# PCA ANALYSIS
# ==============================================================================

print("Computing PCA on combined data...")
all_latents = np.vstack([real_latents, diffusion_latents, gaussian_latents])
pca = PCA(n_components=2)
all_pca = pca.fit_transform(all_latents)

n_real = len(real_latents)
n_diffusion = len(diffusion_latents)
real_pca = all_pca[:n_real]
diffusion_pca = all_pca[n_real:n_real+n_diffusion]
gaussian_pca = all_pca[n_real+n_diffusion:]

print(f"PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%}\n")

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print("Creating plot...")
fig, ax = plt.subplots(1, 1, figsize=(18, 13))
colors = plt.cm.tab10(np.linspace(0, 1, 8))

# Plot each chemical with all three sources
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

print("\n" + "=" * 80)
print("Plot saved: images/pca_real_vs_diffusion_vs_gaussian_by_chemical.png")
print("=" * 80)

# Save metrics
metrics = {
    'explained_variance_pc1': float(pca.explained_variance_ratio_[0]),
    'explained_variance_pc2': float(pca.explained_variance_ratio_[1]),
    'total_variance': float(pca.explained_variance_ratio_[:2].sum()),
    'chemicals': chemicals,
    'samples_per_class_per_source': samples_per_class,
    'note': 'Gaussian samples are drawn from N(mu, sigma) fitted to each chemical\'s real latent data'
}
with open('results/pca_by_chemical_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Done!")
