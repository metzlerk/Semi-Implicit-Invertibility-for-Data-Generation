#!/usr/bin/env python3
"""
Create eye test for sponsor - can they spot the synthetic spectrum?
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Loading data...")
# Load synthetic DEB spectra from single-DEB model
gen_spectra = np.load('results/single_deb_spectra.npy')

# Load real DEB spectra
test_df = pd.read_feather('Data/test_data.feather')
chemicals = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']

# Get spectrum columns in same order as decoder (unsorted)
p_cols = [col for col in test_df.columns if col.startswith('p_')]
n_cols = [col for col in test_df.columns if col.startswith('n_')]
spectrum_cols = p_cols + n_cols

labels = test_df[chemicals].values.argmax(axis=1)
deb_mask = labels == 0
real_deb_spectra = test_df.loc[deb_mask, spectrum_cols].values

print(f"Real DEB spectra: {real_deb_spectra.shape}")
print(f"Generated spectra: {gen_spectra.shape}")

# Randomly select 5 real and 1 synthetic
np.random.seed(42)  # For reproducibility
real_indices = np.random.choice(len(real_deb_spectra), 5, replace=False)
gen_index = np.random.randint(0, len(gen_spectra))

# The 6 spectra (5 real + 1 fake)
spectra_list = []
is_fake = []

# Randomly place the fake in one of the 6 positions
fake_position = 3  # Put it in position 4 (0-indexed)

for i in range(6):
    if i == fake_position:
        spectra_list.append(gen_spectra[gen_index])
        is_fake.append(True)
    else:
        real_idx = i if i < fake_position else i - 1
        spectra_list.append(real_deb_spectra[real_indices[real_idx]])
        is_fake.append(False)

# VERSION 1: WITH LABELS (answer key)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (ax, spectrum, fake) in enumerate(zip(axes, spectra_list, is_fake)):
    ax.plot(spectrum, 'r-' if fake else 'b-', linewidth=1.5, alpha=0.8)
    
    title = f"Spectrum {idx + 1}"
    if fake:
        title += " (SYNTHETIC)"
        ax.set_facecolor('#fff0f0')  # Light red background
    
    ax.set_title(title, fontsize=14, fontweight='bold', 
                 color='red' if fake else 'black')
    ax.set_xlabel('Drift Time', fontsize=11)
    ax.set_ylabel('Ion Intensity', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([-500, 5000])

plt.suptitle('DEB Spectra Comparison - WITH ANSWER\n(Red = Synthetic, Blue = Real)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/deb_eye_test_labeled.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: images/deb_eye_test_labeled.png")
print(f"  Answer: Spectrum {fake_position + 1} is synthetic")

# VERSION 2: WITHOUT LABELS (blind test)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (ax, spectrum) in enumerate(zip(axes, spectra_list)):
    ax.plot(spectrum, 'k-', linewidth=1.5, alpha=0.8)  # All black
    
    ax.set_title(f"Spectrum {idx + 1}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Drift Time', fontsize=11)
    ax.set_ylabel('Ion Intensity', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([-500, 5000])

plt.suptitle('DEB Spectra - Can You Spot the Synthetic One?', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('images/deb_eye_test_blind.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: images/deb_eye_test_blind.png")

print(f"\nEye test created!")
print(f"Blind test: images/deb_eye_test_blind.png")
print(f"Answer key: images/deb_eye_test_labeled.png (Spectrum {fake_position + 1} is synthetic)")
