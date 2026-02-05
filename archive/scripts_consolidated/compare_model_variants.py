#!/usr/bin/env python3
"""
Compare Separated vs Normalized Diffusion Models
=================================================

Direct comparison of the two model architectures:
- Separated: 8 independent models (one per chemical)
- Normalized: Single model on normalized latent space

Which better preserves structure?
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import seaborn as sns

RESULTS_DIR = 'results'
DATA_DIR = 'Data'
IMAGES_DIR = 'images'

print("="*80)
print("SEPARATED VS NORMALIZED DIFFUSION COMPARISON")
print("="*80)

# Load real data
print("\nLoading real latents...")
train_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy'))
train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))
label_columns = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
train_labels = train_df[label_columns].values

print(f"Real samples: {len(train_latent)}")

# Check what samples exist
has_normalized = os.path.exists(os.path.join(RESULTS_DIR, 'normalized_diffusion_samples.npy'))
has_separated = os.path.exists(os.path.join(RESULTS_DIR, 'separated_diffusion_samples.npy'))

if not has_normalized:
    print("\n⚠️  Normalized diffusion samples not found!")
    print("Run: sbatch scripts/run_eval_normalized.sh")

if not has_separated:
    print("\n⚠️  Separated diffusion samples not found!")
    print("Note: Need to generate these from the separated model")

if not (has_normalized or has_separated):
    print("\nCannot proceed without sample data. Exiting.")
    exit(1)

# Load available samples
samples_dict = {'Real': (train_latent, train_labels)}

if has_normalized:
    norm_samples = np.load(os.path.join(RESULTS_DIR, 'normalized_diffusion_samples.npy'))
    norm_labels = np.load(os.path.join(RESULTS_DIR, 'normalized_diffusion_labels.npy'))
    samples_dict['Normalized Diffusion'] = (norm_samples, norm_labels)
    print(f"\n✓ Loaded normalized samples: {len(norm_samples)}")

if has_separated:
    sep_samples = np.load(os.path.join(RESULTS_DIR, 'separated_diffusion_samples.npy'))
    sep_labels = np.load(os.path.join(RESULTS_DIR, 'separated_diffusion_labels.npy'))
    samples_dict['Separated Diffusion'] = (sep_samples, sep_labels)
    print(f"✓ Loaded separated samples: {len(sep_samples)}")

# Compute global statistics
print("\n" + "="*80)
print("GLOBAL STATISTICS COMPARISON")
print("="*80)
print(f"{'Dataset':<25} {'Mean':<12} {'Std':<12} {'Pairwise Dist':<15}")
print("-"*80)

for name, (latents, labels) in samples_dict.items():
    subset = latents[np.random.choice(len(latents), min(3000, len(latents)), replace=False)]
    dists = pdist(subset)
    print(f"{name:<25} {latents.mean():>11.4f} {latents.std():>11.4f} {dists.mean():>14.2f}")

# Per-class comparison
print("\n" + "="*80)
print("PER-CLASS STATISTICS")
print("="*80)

for class_idx, chemical in enumerate(label_columns):
    print(f"\n{chemical}:")
    print(f"{'  Dataset':<23} {'Mean':<12} {'Std':<12} {'N Samples':<12}")
    print("  " + "-"*60)
    
    for name, (latents, labels) in samples_dict.items():
        mask = labels[:, class_idx] == 1
        class_latents = latents[mask]
        print(f"  {name:<23} {class_latents.mean():>11.4f} {class_latents.std():>11.4f} {len(class_latents):>11d}")

# PCA visualization
if len(samples_dict) > 1:
    print("\n" + "="*80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*80)
    
    # Sample for visualization
    n_vis = 2000
    np.random.seed(42)
    
    vis_data = {}
    for name, (latents, labels) in samples_dict.items():
        idx = np.random.choice(len(latents), min(n_vis, len(latents)), replace=False)
        vis_data[name] = (latents[idx], labels[idx])
    
    # Fit PCA on all data
    print("\nFitting PCA...")
    all_vis_data = np.vstack([v[0] for v in vis_data.values()])
    pca = PCA(n_components=2)
    pca.fit(all_vis_data)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Transform all datasets
    pca_data = {name: pca.transform(latents) for name, (latents, labels) in vis_data.items()}
    
    # Create comparison figure
    n_datasets = len(vis_data)
    fig, axes = plt.subplots(1, n_datasets, figsize=(7*n_datasets, 6))
    if n_datasets == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.arange(len(label_columns)))
    
    for ax, (name, (latents, labels)) in zip(axes, vis_data.items()):
        pca_proj = pca_data[name]
        
        for class_idx, chemical in enumerate(label_columns):
            mask = labels[:, class_idx] == 1
            ax.scatter(pca_proj[mask, 0], pca_proj[mask, 1],
                      c=[colors[class_idx]], label=chemical, alpha=0.4, s=10)
        
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Latent Space Structure Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'diffusion_model_comparison.png'), 
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(IMAGES_DIR, 'diffusion_model_comparison.png')}")
    plt.close()
    
    # Per-chemical overlay comparison
    if 'Normalized Diffusion' in samples_dict and 'Separated Diffusion' in samples_dict:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        real_pca = pca_data['Real']
        real_labels = vis_data['Real'][1]
        norm_pca = pca_data['Normalized Diffusion']
        norm_labels = vis_data['Normalized Diffusion'][1]
        sep_pca = pca_data['Separated Diffusion']
        sep_labels = vis_data['Separated Diffusion'][1]
        
        for class_idx, chemical in enumerate(label_columns):
            ax = axes[class_idx]
            
            # Real (blue circles)
            mask_real = real_labels[:, class_idx] == 1
            ax.scatter(real_pca[mask_real, 0], real_pca[mask_real, 1],
                      c='blue', alpha=0.3, s=30, marker='o', label='Real')
            
            # Normalized (red x)
            mask_norm = norm_labels[:, class_idx] == 1
            ax.scatter(norm_pca[mask_norm, 0], norm_pca[mask_norm, 1],
                      c='red', alpha=0.5, s=40, marker='x', label='Normalized')
            
            # Separated (green +)
            mask_sep = sep_labels[:, class_idx] == 1
            ax.scatter(sep_pca[mask_sep, 0], sep_pca[mask_sep, 1],
                      c='green', alpha=0.5, s=40, marker='+', label='Separated')
            
            ax.set_title(chemical, fontsize=12, fontweight='bold')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Per-Chemical: Real vs Normalized vs Separated', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'diffusion_detailed_comparison.png'), 
                    dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {os.path.join(IMAGES_DIR, 'diffusion_detailed_comparison.png')}")
        plt.close()

print("\n" + "="*80)
print("COMPARISON COMPLETE!")
print("="*80)
