#!/usr/bin/env python3
"""
Generate samples from trained 8-chemical diffusion and decode to spectra
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import ast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

MODELS_DIR = 'models'
RESULTS_DIR = 'results'
DATA_DIR = 'Data'

# Model architecture (must match training)
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
    def __init__(self, latent_dim=512, smile_dim=512, num_classes=8, 
                 timesteps=1000, hidden_dim=512, num_layers=6):
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
        
        input_dim = latent_dim + smile_dim + num_classes
        
        layers = []
        layers.append(nn.Linear(input_dim + time_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Cosine schedule
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
    
    def forward(self, x, t, smile_emb, class_onehot):
        t_emb = self.time_mlp(t)
        x_in = torch.cat([x, smile_emb, class_onehot], dim=1)
        x_in = torch.cat([x_in, t_emb], dim=1)
        return self.net(x_in)

# Decoder architecture
class BiasOnlyLayer(nn.Module):
    def __init__(self, bias_init, trainable=True):
        super().__init__()
        self.bias = nn.Parameter(bias_init, requires_grad=trainable)
    
    def forward(self, x):
        return x + self.bias

class FlexibleNLayersGenerator(nn.Module):
    def __init__(self, input_size=512, output_size=1676, n_layers=9, init_style=None, bkg=None, trainable=False):
        super().__init__()
        if init_style == 'bkg':
            if bkg is None:
                raise ValueError("Must provide `bkg` tensor when init_style='bkg'")
            self.bias_layer = BiasOnlyLayer(bias_init=bkg, trainable=trainable)
        
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

# Load SMILE embeddings
print("Loading SMILE embeddings...")
smile_df = pd.read_csv(os.path.join(DATA_DIR, 'name_smiles_embedding_file.csv'))
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
        smile_embeddings[short_name] = torch.FloatTensor(np.array(ast.literal_eval(emb_str)))

chemicals = list(label_mapping.keys())

# Load diffusion model
print("Loading diffusion model...")
checkpoint = torch.load(os.path.join(MODELS_DIR, 'diffusion_latent_normalized_best.pt'), 
                        map_location=device, weights_only=False)
DATA_MEAN = checkpoint['data_mean']
DATA_STD = checkpoint['data_std']
print(f"Normalization: mean={DATA_MEAN:.4f}, std={DATA_STD:.4f}")

model = ClassConditionedDiffusion(512, 512, 8, 1000, 512, 6).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load decoder
print("Loading decoder...")
decoder = FlexibleNLayersGenerator(init_style='bkg', bkg=torch.zeros(1676), trainable=False).to(device)
decoder_ckpt = torch.load(os.path.join(MODELS_DIR, 'autoencoder_separated.pth'), 
                         map_location=device, weights_only=False)
decoder.load_state_dict(decoder_ckpt['generator_state_dict'])
decoder.eval()

# DDIM sampling
@torch.no_grad()
def sample_ddim(model, smile_emb, class_onehot, n_samples):
    model.eval()
    x_t = torch.randn(n_samples, 512, device=device)
    
    ddim_steps = 100
    step_size = model.timesteps // ddim_steps
    timesteps = list(range(0, model.timesteps, step_size))
    
    for i in reversed(range(len(timesteps))):
        t = timesteps[i]
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
        
        x_0_pred = model(x_t, t_batch, smile_emb, class_onehot)
        x_0_pred = torch.clamp(x_0_pred, -5, 5)
        
        if i > 0:
            t_prev = timesteps[i-1]
            alpha_bar_t = model.alphas_cumprod[t]
            alpha_bar_prev = model.alphas_cumprod[t_prev]
            
            predicted_noise = (x_t - torch.sqrt(alpha_bar_t) * x_0_pred) / torch.sqrt(1 - alpha_bar_t)
            x_t = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_prev) * predicted_noise
        else:
            x_t = x_0_pred
    
    return x_t

# Generate samples for each chemical
print("\nGenerating samples for each chemical...")
samples_per_class = 500
all_latents = []
all_spectra = []
all_labels = []

for class_idx, chem in enumerate(chemicals):
    print(f"  {chem}... ", end='', flush=True)
    
    smile_emb = smile_embeddings[chem].unsqueeze(0).repeat(samples_per_class, 1).to(device)
    class_onehot = torch.zeros(samples_per_class, 8, device=device)
    class_onehot[:, class_idx] = 1.0
    
    # Generate latents
    latents_norm = sample_ddim(model, smile_emb, class_onehot, samples_per_class)
    latents = latents_norm * DATA_STD + DATA_MEAN
    
    # Decode to spectra
    spectra = decoder(latents).detach().cpu().numpy()
    
    all_latents.append(latents.detach().cpu().numpy())
    all_spectra.append(spectra)
    all_labels.extend([class_idx] * samples_per_class)
    
    print(f"mean={latents.mean():.2f}, std={latents.std():.2f}")

all_latents = np.vstack(all_latents)
all_spectra = np.vstack(all_spectra)
all_labels = np.array(all_labels)

print(f"\nGenerated {len(all_latents)} total samples")
print(f"  Latents: {all_latents.shape}")
print(f"  Spectra: {all_spectra.shape}")

# Save
np.save(os.path.join(RESULTS_DIR, 'full_generated_latents.npy'), all_latents)
np.save(os.path.join(RESULTS_DIR, 'full_generated_spectra.npy'), all_spectra)
np.save(os.path.join(RESULTS_DIR, 'full_generated_labels.npy'), all_labels)

print(f"\n✓ Saved to results/full_generated_*.npy")
