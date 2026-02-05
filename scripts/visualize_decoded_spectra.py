#!/usr/bin/env python3
"""
Decode latent samples back to IMS spectra and visualize
Shows: Real vs Diffusion vs Gaussian spectra
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load decoder
from train_latent_diffusion import ClassConditionedDiffusion, get_beta_schedule
import torch.nn as nn

class BiasOnlyLayer(nn.Module):
    def __init__(self, bias_init, trainable=True):
        super().__init__()
        self.bias = nn.Parameter(bias_init, requires_grad=trainable)
    
    def forward(self, x):
        return x + self.bias

class FlexibleNLayersGenerator(nn.Module):
    """Generator: Latent space (512-dim) -> IMS spectra (1676-dim)"""
    def __init__(self, input_size=512, output_size=1676, n_layers=9, init_style=None, bkg=None, trainable=False):
        super().__init__()
        if init_style == 'bkg':
            if bkg is None:
                raise ValueError("Must provide `bkg` tensor when init_style='bkg'")
            self.bias_layer = BiasOnlyLayer(bias_init=bkg, trainable=trainable)
        
        # Use np.linspace to match original training
        layers = []
        layer_sizes = np.linspace(input_size, output_size, n_layers + 1, dtype=int)
        for i in range(n_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < n_layers - 1:
                layers.append(nn.LeakyReLU(inplace=True))
        
        self.generator = nn.Sequential(*layers)
    
    def forward(self, x, use_bias=False):
        x = self.generator(x)
        if use_bias and hasattr(self, 'bias_layer'):
            x = self.bias_layer(x)
        return x

print("Loading decoder...")
generator = FlexibleNLayersGenerator(init_style='bkg', bkg=torch.zeros(1676), trainable=False).to(device)
checkpoint = torch.load('models/autoencoder_separated.pth', map_location=device, weights_only=False)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()

# Load SMILE embeddings
print("Loading SMILE embeddings...")
smile_df = pd.read_csv('Data/name_smiles_embedding_file.csv')
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

smile_embeddings = {}
for short_name, full_name in label_mapping.items():
    row = smile_df[smile_df['Name'] == full_name]
    if not row.empty:
        emb_str = row['embedding'].values[0]
        smile_embeddings[short_name] = np.array(ast.literal_eval(emb_str))

chemicals = list(label_mapping.keys())

# Load real latents
print("Loading real latents...")
test_latent = np.load('results/autoencoder_test_latent_separated.npy')
test_df = pd.read_feather('Data/test_data.feather')
test_labels = test_df[chemicals].values.argmax(axis=1)

samples_per_class = 5  # Show 5 examples per class

# Collect real samples
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

# Load diffusion model and generate samples
print("Loading diffusion model...")

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

class ClassConditionedDiffusion(nn.Module):
    def __init__(self, latent_dim, smile_dim, num_classes, timesteps=50, hidden_dim=512, num_layers=6):
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
        
        # Main network
        input_dim = latent_dim + smile_dim + num_classes + hidden_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Beta schedule
        betas = torch.linspace(0.001, 0.2, timesteps)
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
    
    def forward(self, x, t, smile_emb, class_onehot):
        t_emb = self.time_mlp(t.float())
        x_in = torch.cat([x, smile_emb, class_onehot, t_emb], dim=1)
        return self.net(x_in)

diffusion = ClassConditionedDiffusion(512, 512, 8, 50, 512, 6).to(device)
checkpoint = torch.load('models/diffusion_latent_separated_best.pt', map_location=device, weights_only=False)
diffusion.load_state_dict(checkpoint['model_state_dict'], strict=False)
diffusion.eval()

print("Generating diffusion samples...")
diffusion_latents = []
with torch.no_grad():
    for class_idx, chem in enumerate(chemicals):
        if chem not in smile_embeddings:
            continue
        
        smile_emb = torch.FloatTensor(smile_embeddings[chem]).unsqueeze(0).repeat(samples_per_class, 1).to(device)
        class_onehot = torch.zeros(samples_per_class, 8, device=device)
        class_onehot[:, class_idx] = 1.0
        
        z_t = torch.randn(samples_per_class, 512, device=device)
        
        for t in reversed(range(50)):
            t_batch = torch.full((samples_per_class,), t, dtype=torch.long, device=device)
            predicted_noise = diffusion(z_t, t_batch, smile_emb, class_onehot)
            
            alpha_bar_t = diffusion.alphas_cumprod[t]
            beta_t = diffusion.betas[t]
            
            if t > 0:
                alpha_bar_prev = diffusion.alphas_cumprod[t-1]
                x_0_pred = (z_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
                dir_xt = torch.sqrt(1 - alpha_bar_prev) * predicted_noise
                z_t = torch.sqrt(alpha_bar_prev) * x_0_pred + dir_xt
                z_t = z_t + torch.sqrt(beta_t) * torch.randn_like(z_t)
            else:
                z_t = (z_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        diffusion_latents.append(z_t.cpu().numpy())

diffusion_latents = np.vstack(diffusion_latents)

# Generate Gaussian samples
print("Generating Gaussian samples...")
gaussian_latents = []
for class_idx in range(8):
    mask = np.array(real_labels_list) == class_idx
    class_real = real_latents[mask]
    
    mu = class_real.mean(axis=0)
    sigma = class_real.std(axis=0)
    
    samples = np.random.normal(mu, sigma, size=(samples_per_class, 512))
    gaussian_latents.append(samples)

gaussian_latents = np.vstack(gaussian_latents)

# Decode all samples
print("\nDecoding latents to IMS spectra...")
with torch.no_grad():
    real_spectra = generator(torch.FloatTensor(real_latents).to(device)).cpu().numpy()
    diffusion_spectra = generator(torch.FloatTensor(diffusion_latents).to(device)).cpu().numpy()
    gaussian_spectra = generator(torch.FloatTensor(gaussian_latents).to(device)).cpu().numpy()

print(f"Real spectra: {real_spectra.shape}")
print(f"Diffusion spectra: {diffusion_spectra.shape}")
print(f"Gaussian spectra: {gaussian_spectra.shape}")

# Plot spectra for each chemical
print("\nCreating visualizations...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

colors = {'real': '#2E86AB', 'diffusion': '#A23B72', 'gaussian': '#F18F01'}

for class_idx, chem in enumerate(chemicals):
    ax = axes[class_idx]
    
    # Get samples for this class
    start_idx = class_idx * samples_per_class
    end_idx = start_idx + samples_per_class
    
    real_samples = real_spectra[start_idx:end_idx]
    diff_samples = diffusion_spectra[start_idx:end_idx]
    gauss_samples = gaussian_spectra[start_idx:end_idx]
    
    # Plot each sample
    x = np.arange(1676)
    for i in range(samples_per_class):
        ax.plot(x, real_samples[i], color=colors['real'], alpha=0.3, linewidth=1)
        ax.plot(x, diff_samples[i], color=colors['diffusion'], alpha=0.3, linewidth=1)
        ax.plot(x, gauss_samples[i], color=colors['gaussian'], alpha=0.3, linewidth=1)
    
    # Plot means
    ax.plot(x, real_samples.mean(axis=0), color=colors['real'], linewidth=2, label='Real')
    ax.plot(x, diff_samples.mean(axis=0), color=colors['diffusion'], linewidth=2, label='Diffusion')
    ax.plot(x, gauss_samples.mean(axis=0), color=colors['gaussian'], linewidth=2, label='Gaussian')
    
    ax.set_title(f'{chem}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Index', fontsize=10)
    ax.set_ylabel('Intensity', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Decoded IMS Spectra: Real vs Diffusion vs Gaussian\n(Lines show mean, shaded show individual samples)', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('images/decoded_spectra_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: images/decoded_spectra_comparison.png")

# Also create a single-chemical detailed view
print("\nCreating detailed single-chemical view...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

chem_idx = 0  # DEB
chem = chemicals[chem_idx]
start_idx = chem_idx * samples_per_class
end_idx = start_idx + samples_per_class

real_samples = real_spectra[start_idx:end_idx]
diff_samples = diffusion_spectra[start_idx:end_idx]
gauss_samples = gaussian_spectra[start_idx:end_idx]

x = np.arange(1676)

# Real
for i in range(samples_per_class):
    axes[0].plot(x, real_samples[i], color=colors['real'], alpha=0.5, linewidth=1.5)
axes[0].plot(x, real_samples.mean(axis=0), color='black', linewidth=3, label='Mean', linestyle='--')
axes[0].set_title(f'{chem} - Real Encoder Outputs', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Intensity', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Diffusion
for i in range(samples_per_class):
    axes[1].plot(x, diff_samples[i], color=colors['diffusion'], alpha=0.5, linewidth=1.5)
axes[1].plot(x, diff_samples.mean(axis=0), color='black', linewidth=3, label='Mean', linestyle='--')
axes[1].set_title(f'{chem} - Diffusion Generated', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Intensity', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Gaussian
for i in range(samples_per_class):
    axes[2].plot(x, gauss_samples[i], color=colors['gaussian'], alpha=0.5, linewidth=1.5)
axes[2].plot(x, gauss_samples.mean(axis=0), color='black', linewidth=3, label='Mean', linestyle='--')
axes[2].set_title(f'{chem} - Gaussian Samples', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Feature Index', fontsize=12)
axes[2].set_ylabel('Intensity', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle(f'Detailed Spectra Comparison for {chem}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'images/decoded_spectra_detailed_{chem}.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: images/decoded_spectra_detailed_{chem}.png")

print("\n" + "="*80)
print("✓ All visualizations complete!")
print("="*80)
