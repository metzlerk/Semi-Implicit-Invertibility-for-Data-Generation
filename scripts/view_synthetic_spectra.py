#!/usr/bin/env python3
"""
Simple script to visualize synthetic IMS spectra
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Load synthetic spectra
print("Loading synthetic spectra...")
spectra = np.load('synthetic_ims_data/synthetic_ims_spectra.npy')
labels = np.load('synthetic_ims_data/synthetic_labels.npy')

# Chemical class names
class_names = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']

print(f"Loaded {len(spectra)} synthetic spectra")
print(f"Spectra shape: {spectra.shape}")
print(f"Number of features: {spectra.shape[1]}")

# Plot one spectrum from each chemical class
print("\nCreating plot...")
fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.flatten()

for class_idx, class_name in enumerate(class_names):
    # Get first sample of this class
    class_mask = labels == class_idx
    class_spectra = spectra[class_mask]
    
    if len(class_spectra) > 0:
        # Plot the first spectrum
        axes[class_idx].plot(class_spectra[0], linewidth=1.5, color=f'C{class_idx}')
        axes[class_idx].set_title(f'{class_name}', fontsize=14, fontweight='bold')
        axes[class_idx].set_xlabel('Feature Index', fontsize=11)
        axes[class_idx].set_ylabel('Intensity', fontsize=11)
        axes[class_idx].grid(True, alpha=0.3)
        axes[class_idx].set_xlim(0, len(class_spectra[0]))

plt.suptitle('Synthetic IMS Spectra - One Sample per Chemical Class', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()

# Save figure
output_file = 'synthetic_spectra_samples.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved plot to: {output_file}")
print("  One synthetic spectrum displayed for each of the 8 chemical classes")
