#!/usr/bin/env python3
"""
Create separate PCA plots for each chemical (cleaner than combined)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

print("Loading data...")
# Load generated latents
gen_latents = np.load('results/full_generated_latents.npy')
gen_labels = np.load('results/full_generated_labels.npy')

# Load real latents
test_latents = np.load('results/autoencoder_test_latent_separated.npy')
test_df = pd.read_feather('Data/test_data.feather')
chemicals = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
test_labels = test_df[chemicals].values.argmax(axis=1)

print(f"Generated latents: {gen_latents.shape}")
print(f"Real latents: {test_latents.shape}")

# Create 8 separate PCA plots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for chem_idx, (chem, ax) in enumerate(zip(chemicals, axes)):
    print(f"\nProcessing {chem}...")
    
    # Get data for this chemical
    real_mask = test_labels == chem_idx
    gen_mask = gen_labels == chem_idx
    
    real_chem = test_latents[real_mask]
    gen_chem = gen_latents[gen_mask]
    
    print(f"  Real: {real_chem.shape}, Generated: {gen_chem.shape}")
    
    # Fit PCA on combined data for this chemical only
    pca = PCA(n_components=2)
    combined = np.vstack([real_chem, gen_chem])
    pca.fit(combined)
    
    real_pca = pca.transform(real_chem)
    gen_pca = pca.transform(gen_chem)
    
    # Plot
    ax.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', alpha=0.3, s=5, label='Real')
    ax.scatter(gen_pca[:, 0], gen_pca[:, 1], c='red', alpha=0.3, s=5, marker='x', label='Generated')
    
    ax.set_title(f'{chem}', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)
    
    # Print statistics
    print(f"  Real: mean={real_chem.mean():.2f}, std={real_chem.std():.2f}")
    print(f"  Gen:  mean={gen_chem.mean():.2f}, std={gen_chem.std():.2f}")
    print(f"  PCA explained: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

plt.suptitle('Per-Chemical Latent Space PCA: Real vs Generated', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('images/per_chemical_latent_pca.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: images/per_chemical_latent_pca.png")
