#!/usr/bin/env python3
"""
Quick visualization of decoded DEB spectra
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Loading data...")
# Load generated spectra
gen_spectra = np.load('results/single_deb_spectra.npy')
print(f"Generated spectra: {gen_spectra.shape}")
print(f"  Stats: mean={gen_spectra.mean():.2e}, std={gen_spectra.std():.2e}")

# Load real DEB spectra for comparison
test_df = pd.read_feather('Data/test_data.feather')
p_cols = sorted([col for col in test_df.columns if col.startswith('p_')])
n_cols = sorted([col for col in test_df.columns if col.startswith('n_')])
spectrum_cols = p_cols + n_cols  # p columns first, then n columns
deb_mask = test_df['DEB'] == 1
real_deb_spectra = test_df.loc[deb_mask, spectrum_cols].values
print(f"\nReal DEB spectra: {real_deb_spectra.shape}")
print(f"  Stats: mean={real_deb_spectra.mean():.2e}, std={real_deb_spectra.std():.2e}")

# Plot comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sample spectra comparison
ax = axes[0, 0]
for i in range(5):
    ax.plot(gen_spectra[i], alpha=0.6, label=f'Gen {i+1}')
ax.set_title('Generated DEB Spectra (5 samples)', fontsize=12, fontweight='bold')
ax.set_xlabel('Drift Time')
ax.set_ylabel('Ion Intensity')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
for i in range(5):
    ax.plot(real_deb_spectra[i], alpha=0.6, label=f'Real {i+1}')
ax.set_title('Real DEB Spectra (5 samples)', fontsize=12, fontweight='bold')
ax.set_xlabel('Drift Time')
ax.set_ylabel('Ion Intensity')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Mean spectra comparison
ax = axes[1, 0]
ax.plot(gen_spectra.mean(axis=0), 'b-', label='Generated Mean', linewidth=2)
ax.fill_between(range(len(gen_spectra[0])), 
                 gen_spectra.mean(axis=0) - gen_spectra.std(axis=0),
                 gen_spectra.mean(axis=0) + gen_spectra.std(axis=0),
                 alpha=0.3)
ax.set_title('Generated: Mean ± Std', fontsize=12, fontweight='bold')
ax.set_xlabel('Drift Time')
ax.set_ylabel('Ion Intensity')
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(real_deb_spectra.mean(axis=0), 'g-', label='Real Mean', linewidth=2)
ax.fill_between(range(len(real_deb_spectra[0])), 
                 real_deb_spectra.mean(axis=0) - real_deb_spectra.std(axis=0),
                 real_deb_spectra.mean(axis=0) + real_deb_spectra.std(axis=0),
                 alpha=0.3)
ax.set_title('Real: Mean ± Std', fontsize=12, fontweight='bold')
ax.set_xlabel('Drift Time')
ax.set_ylabel('Ion Intensity')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/single_deb_spectra_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: images/single_deb_spectra_comparison.png")

# Create distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(gen_spectra.flatten(), bins=100, alpha=0.6, label='Generated', density=True)
ax.hist(real_deb_spectra.flatten(), bins=100, alpha=0.6, label='Real', density=True)
ax.set_xlabel('Intensity Value')
ax.set_ylabel('Density')
ax.set_title('Intensity Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
# Only compare matching dimensions
min_dim = min(real_deb_spectra.shape[1], gen_spectra.shape[1])
ax.scatter(real_deb_spectra[:100, :min_dim].flatten(), gen_spectra[:100, :min_dim].flatten(), alpha=0.1, s=1)
ax.set_xlabel('Real Ion Intensity')
ax.set_ylabel('Generated Ion Intensity')
ax.set_title('Real vs Generated (scatter)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/single_deb_spectra_distributions.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: images/single_deb_spectra_distributions.png")
