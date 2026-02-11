#!/usr/bin/env python3
"""
Visualize generated spectra for all 8 chemicals
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Loading data...")
# Load generated
gen_spectra = np.load('results/full_generated_spectra.npy')
gen_labels = np.load('results/full_generated_labels.npy')

# Load real
test_df = pd.read_feather('Data/test_data.feather')
chemicals = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']

# Get spectrum columns in SAME ORDER as decoder training (unsorted)
p_cols = [col for col in test_df.columns if col.startswith('p_')]
n_cols = [col for col in test_df.columns if col.startswith('n_')]
spectrum_cols = p_cols + n_cols

real_labels = test_df[chemicals].values.argmax(axis=1)
real_spectra = test_df[spectrum_cols].values

print(f"Generated: {gen_spectra.shape}")
print(f"Real: {real_spectra.shape}")

# Create comprehensive visualization
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for class_idx, chem in enumerate(chemicals):
    # Get data for this chemical
    gen_mask = gen_labels == class_idx
    real_mask = real_labels == class_idx
    
    gen_chem = gen_spectra[gen_mask]
    real_chem = real_spectra[real_mask]
    
    # Plot 1: Sample spectra
    ax = axes[class_idx]
    for i in range(min(5, len(gen_chem))):
        label = 'Generated' if i == 0 else None
        ax.plot(gen_chem[i], alpha=0.4, color='red', linewidth=0.8, label=label)
    for i in range(min(5, len(real_chem))):
        label = 'Real' if i == 0 else None
        ax.plot(real_chem[i], alpha=0.4, color='blue', linewidth=0.8, label=label)
    ax.set_title(f'{chem} - Sample Spectra', fontsize=11, fontweight='bold')
    ax.set_xlabel('Drift Time', fontsize=9)
    ax.set_ylabel('Ion Intensity', fontsize=9)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    
    # Plot 2: Mean comparison
    ax = axes[class_idx + 8]
    ax.plot(gen_chem.mean(axis=0), 'r-', label='Generated Mean', linewidth=2, alpha=0.8)
    ax.plot(real_chem.mean(axis=0), 'b-', label='Real Mean', linewidth=2, alpha=0.8)
    ax.fill_between(range(len(gen_chem[0])),
                     gen_chem.mean(axis=0) - gen_chem.std(axis=0),
                     gen_chem.mean(axis=0) + gen_chem.std(axis=0),
                     alpha=0.2, color='red')
    ax.fill_between(range(len(real_chem[0])),
                     real_chem.mean(axis=0) - real_chem.std(axis=0),
                     real_chem.mean(axis=0) + real_chem.std(axis=0),
                     alpha=0.2, color='blue')
    ax.set_title(f'{chem} - Mean ± Std', fontsize=11, fontweight='bold')
    ax.set_xlabel('Drift Time', fontsize=9)
    ax.set_ylabel('Ion Intensity', fontsize=9)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    
    # Print stats
    print(f"\n{chem}:")
    print(f"  Generated: mean={gen_chem.mean():.1f}, std={gen_chem.std():.1f}")
    print(f"  Real:      mean={real_chem.mean():.1f}, std={real_chem.std():.1f}")

plt.tight_layout()
plt.savefig('images/full_8chem_spectra_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: images/full_8chem_spectra_comparison.png")

# Create distribution comparison
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for class_idx, chem in enumerate(chemicals):
    ax = axes[class_idx]
    
    gen_mask = gen_labels == class_idx
    real_mask = real_labels == class_idx
    
    gen_chem = gen_spectra[gen_mask]
    real_chem = real_spectra[real_mask]
    
    ax.hist(gen_chem.flatten(), bins=100, alpha=0.5, label='Generated', 
            density=True, color='red')
    ax.hist(real_chem.flatten(), bins=100, alpha=0.5, label='Real',
            density=True, color='blue')
    ax.set_title(f'{chem} - Intensity Distribution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Ion Intensity', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('images/full_8chem_distributions.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: images/full_8chem_distributions.png")
