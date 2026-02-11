#!/usr/bin/env python3
"""
Create eye test for all 8 chemicals - can sponsor spot the synthetic?
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Loading data...")
# Load synthetic spectra from full 8-chemical model
gen_spectra = np.load('results/full_generated_spectra.npy')
gen_labels = np.load('results/full_generated_labels.npy')

# Load real spectra
test_df = pd.read_feather('Data/test_data.feather')
chemicals = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']

# Get spectrum columns in same order as decoder (unsorted)
p_cols = [col for col in test_df.columns if col.startswith('p_')]
n_cols = [col for col in test_df.columns if col.startswith('n_')]
spectrum_cols = p_cols + n_cols

test_labels = test_df[chemicals].values.argmax(axis=1)
real_spectra = test_df[spectrum_cols].values

print(f"Real spectra: {real_spectra.shape}")
print(f"Generated spectra: {gen_spectra.shape}")

# Set random seed for reproducibility
np.random.seed(42)

# For each chemical, create a 2x3 grid with 5 real + 1 fake
for chem_idx, chem in enumerate(chemicals):
    print(f"\nCreating eye test for {chem}...")
    
    # Get data for this chemical
    real_mask = test_labels == chem_idx
    gen_mask = gen_labels == chem_idx
    
    real_chem = real_spectra[real_mask]
    gen_chem = gen_spectra[gen_mask]
    
    # Randomly select 5 real and 1 synthetic
    real_indices = np.random.choice(len(real_chem), 5, replace=False)
    gen_index = np.random.randint(0, len(gen_chem))
    
    # Randomly place the fake in one of the 6 positions
    fake_position = np.random.randint(0, 6)
    
    # Build the 6 spectra list
    spectra_list = []
    is_fake = []
    
    for i in range(6):
        if i == fake_position:
            spectra_list.append(gen_chem[gen_index])
            is_fake.append(True)
        else:
            real_idx = i if i < fake_position else i - 1
            spectra_list.append(real_chem[real_indices[real_idx]])
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
    
    plt.suptitle(f'{chem} Spectra - WITH ANSWER\\n(Red = Synthetic, Blue = Real)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'images/{chem.lower()}_eye_test_labeled.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # VERSION 2: WITHOUT LABELS (blind test)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (ax, spectrum) in enumerate(zip(axes, spectra_list)):
        ax.plot(spectrum, 'k-', linewidth=1.5, alpha=0.8)  # All black
        
        ax.set_title(f"Spectrum {idx + 1}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Drift Time', fontsize=11)
        ax.set_ylabel('Ion Intensity', fontsize=11)
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'{chem} Spectra - Can You Spot the Synthetic One?', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'images/{chem.lower()}_eye_test_blind.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: images/{chem.lower()}_eye_test_blind.png")
    print(f"  ✓ Saved: images/{chem.lower()}_eye_test_labeled.png")
    print(f"  Answer: Spectrum {fake_position + 1} is synthetic")

print(f"\n✓ Created eye tests for all 8 chemicals!")
