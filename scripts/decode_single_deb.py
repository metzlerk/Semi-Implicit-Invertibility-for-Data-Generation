#!/usr/bin/env python3
"""
Decode generated DEB latent points back to spectra using the pre-trained decoder.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Paths
DECODER_PATH = 'models/autoencoder_separated.pth'
LATENTS_PATH = 'results/single_deb_generated.npy'
OUTPUT_PATH = 'results/single_deb_spectra.npy'

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

print("Loading generated DEB latents...")
latents = np.load(LATENTS_PATH)
print(f"Loaded {len(latents)} latent vectors with shape {latents.shape}")
print(f"Latent stats: mean={latents.mean():.4f}, std={latents.std():.4f}")

print(f"\nLoading decoder from {DECODER_PATH}...")
generator = FlexibleNLayersGenerator(init_style='bkg', bkg=torch.zeros(1676), trainable=False).to(device)
checkpoint = torch.load(DECODER_PATH, map_location=device, weights_only=False)
generator.load_state_dict(checkpoint['generator_state_dict'])
generator.eval()
print(f"✓ Decoder loaded: {type(generator).__name__}")

print(f"\nDecoding {len(latents)} latent vectors to spectra...")
with torch.no_grad():
    latents_tensor = torch.FloatTensor(latents).to(device)
    
    # Decode in batches to avoid memory issues
    batch_size = 256
    spectra_list = []
    
    for i in range(0, len(latents), batch_size):
        batch = latents_tensor[i:i+batch_size]
        batch_spectra = generator(batch).cpu().numpy()
        spectra_list.append(batch_spectra)
        
        if (i // batch_size + 1) % 4 == 0:
            print(f"  Decoded {i + len(batch)}/{len(latents)} samples...")
    
    spectra = np.vstack(spectra_list)

print(f"\nGenerated spectra shape: {spectra.shape}")
print(f"Spectra stats: mean={spectra.mean():.4f}, std={spectra.std():.4f}, min={spectra.min():.4f}, max={spectra.max():.4f}")

print(f"\nSaving spectra to {OUTPUT_PATH}...")
np.save(OUTPUT_PATH, spectra)
print("✓ Done!")
