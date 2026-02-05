#!/usr/bin/env python3
"""
PCA visualization of full 8-chemical latent space
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

# Fit PCA on combined data
print("\nFitting PCA...")
pca = PCA(n_components=2)
combined = np.vstack([test_latents, gen_latents])
pca.fit(combined)

real_pca = pca.transform(test_latents)
gen_pca = pca.transform(gen_latents)

print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

# Colors for each chemical
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
          '#ff7f00', '#ffff33', '#a65628', '#f781bf']

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Real latents by chemical
ax = axes[0, 0]
for class_idx, chem in enumerate(chemicals):
    mask = test_labels == class_idx
    ax.scatter(real_pca[mask, 0], real_pca[mask, 1], 
              c=colors[class_idx], alpha=0.3, s=5, label=chem)
ax.set_title('Real Latent Space (Test Set)', fontsize=14, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

# Plot 2: Generated latents by chemical
ax = axes[0, 1]
for class_idx, chem in enumerate(chemicals):
    mask = gen_labels == class_idx
    ax.scatter(gen_pca[mask, 0], gen_pca[mask, 1],
              c=colors[class_idx], alpha=0.3, s=5, label=chem)
ax.set_title('Generated Latent Space (Diffusion)', fontsize=14, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax.legend(loc='best', fontsize=9)
ax.grid(alpha=0.3)

# Plot 3: Overlay all chemicals
ax = axes[1, 0]
ax.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', alpha=0.15, s=3, label='Real')
ax.scatter(gen_pca[:, 0], gen_pca[:, 1], c='red', alpha=0.15, s=3, marker='x', label='Generated')
ax.set_title('Overlay: Real vs Generated', fontsize=14, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Plot 4: Per-chemical comparison (one example)
ax = axes[1, 1]
# Show DEB as example
deb_real_mask = test_labels == 0
deb_gen_mask = gen_labels == 0
ax.scatter(real_pca[deb_real_mask, 0], real_pca[deb_real_mask, 1], 
          c='blue', alpha=0.3, s=10, label='Real DEB')
ax.scatter(gen_pca[deb_gen_mask, 0], gen_pca[deb_gen_mask, 1],
          c='red', alpha=0.3, s=10, marker='x', label='Generated DEB')
ax.set_title('Example: DEB Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/full_8chem_latent_pca.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: images/full_8chem_latent_pca.png")

# Print statistics per chemical
print("\nPer-chemical statistics:")
for class_idx, chem in enumerate(chemicals):
    real_mask = test_labels == class_idx
    gen_mask = gen_labels == class_idx
    
    real_mean = test_latents[real_mask].mean()
    real_std = test_latents[real_mask].std()
    gen_mean = gen_latents[gen_mask].mean()
    gen_std = gen_latents[gen_mask].std()
    
    print(f"{chem:5s}: Real (mean={real_mean:6.2f}, std={real_std:5.2f}) | "
          f"Gen (mean={gen_mean:6.2f}, std={gen_std:5.2f})")
