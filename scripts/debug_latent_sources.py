"""
Debug: Check what latent data the diffusion model was trained on
"""

import numpy as np
import os

ROOT_DIR = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation'
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

print("="*80)
print("Checking Latent Data Sources")
print("="*80)

# Check what files exist
print("\nAvailable latent files:")
for filename in ['autoencoder_train_latent.npy', 'autoencoder_test_latent.npy',
                 'autoencoder_train_latent_separated.npy', 'autoencoder_test_latent_separated.npy']:
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        data = np.load(path)
        print(f"  ✓ {filename:<45} shape: {data.shape}, "
              f"mean: {data.mean():.6f}, std: {data.std():.6f}")
    else:
        print(f"  ✗ {filename:<45} NOT FOUND")

# Load what the training used
print("\n" + "="*80)
print("What the TRAINING code would load:")
print("="*80)

separated_train = os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy')
if os.path.exists(separated_train):
    print("→ Would use SEPARATED latents")
    train_latent = np.load(separated_train)
    test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent_separated.npy'))
else:
    print("→ Would use ORIGINAL latents")
    train_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent.npy'))
    test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'))

print(f"\nTRAIN latents: shape {train_latent.shape}")
print(f"  Mean: {train_latent.mean():.6f}")
print(f"  Std:  {train_latent.std():.6f}")
print(f"  Min:  {train_latent.min():.6f}")
print(f"  Max:  {train_latent.max():.6f}")

print(f"\nTEST latents: shape {test_latent.shape}")
print(f"  Mean: {test_latent.mean():.6f}")
print(f"  Std:  {test_latent.std():.6f}")
print(f"  Min:  {test_latent.min():.6f}")
print(f"  Max:  {test_latent.max():.6f}")

# Check what the comparison script loaded
print("\n" + "="*80)
print("What the COMPARISON script loaded:")
print("="*80)

# In compare_deb_latent_pca.py, it loads:
# test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'))
comparison_data = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'))
print(f"Hardcoded to: autoencoder_test_latent.npy")
print(f"  Mean: {comparison_data.mean():.6f}")
print(f"  Std:  {comparison_data.std():.6f}")

if os.path.exists(separated_train):
    print("\n⚠ WARNING: Training used SEPARATED latents, but comparison used ORIGINAL!")
    print("   This could explain the difference!")

print("\n" + "="*80)
