"""
Check what data the different models were trained on
"""

import torch
import numpy as np
import os

ROOT_DIR = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

models = [
    'diffusion_latent_normalized_best.pt',
    'diffusion_latent_separated_best.pt',
    'diffusion_latent_separated_beta0.02_best.pt',
    'diffusion_latent_separated_beta0.20_best.pt'
]

print("="*80)
print("Checking Model Checkpoints")
print("="*80)

for model_name in models:
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"\n{model_name}:")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Loss: {checkpoint.get('loss', 'unknown'):.6f}" if isinstance(checkpoint.get('loss'), float) else f"  Loss: {checkpoint.get('loss', 'unknown')}")
        print(f"  Beta_end: {checkpoint.get('beta_end', 'not stored')}")
        print(f"  Keys: {checkpoint.keys()}")

# Check which latent data would be loaded during training
print("\n" + "="*80)
print("Training Data Selection Logic")
print("="*80)

separated_train = os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy')
if os.path.exists(separated_train):
    print("✓ SEPARATED latents exist → training would use SEPARATED")
    train_latent_sep = np.load(separated_train)
    print(f"  Shape: {train_latent_sep.shape}, mean: {train_latent_sep.mean():.3f}, std: {train_latent_sep.std():.3f}")
else:
    print("✗ SEPARATED latents don't exist → would use ORIGINAL")

train_latent_orig = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent.npy'))
print(f"\nORIGINAL latents:")
print(f"  Shape: {train_latent_orig.shape}, mean: {train_latent_orig.mean():.3f}, std: {train_latent_orig.std():.3f}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("The β=0.02 models (old and new) were trained on SEPARATED latents (std≈18)")
print("For fair comparison, generated samples should be compared to SEPARATED latents.")
print("If samples have std≈1, the model is generating in the wrong scale!")
print("="*80)
