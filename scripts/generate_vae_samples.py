"""
Generate VAE Samples and Create Baseline Comparison Figures
============================================================

Generate samples from VAE baseline by sampling from N(0,I) prior,
decode to spectra, and create comparison plots vs diffusion method.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ast

# Paths
ROOT_DIR = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation'
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import VAE architecture from training script
import sys
sys.path.append(os.path.join(ROOT_DIR, 'scripts'))
from train_vae_baseline import VAE, VAEDecoder

# Chemical labels
CHEMICALS = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
NUM_SAMPLES_PER_CLASS = 500

def load_vae_model():
    """Load trained VAE model"""
    checkpoint = torch.load(os.path.join(MODELS_DIR, 'vae_baseline_best.pth'))
    model = VAE(input_dim=1676, latent_dim=512, n_layers=9).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_vae_samples(model, num_per_class=500):
    """
    Generate samples from VAE by sampling from N(0,I) prior
    
    Returns:
        latents: (num_classes * num_per_class, 512)
        spectra: (num_classes * num_per_class, 1676)
        labels: (num_classes * num_per_class,)
    """
    all_latents = []
    all_spectra = []
    all_labels = []
    
    print(f"Generating {num_per_class} samples per chemical...")
    
    with torch.no_grad():
        for i, chem in enumerate(CHEMICALS):
            # Sample from standard Gaussian prior
            z = torch.randn(num_per_class, 512).to(device)
            
            # Decode to spectra
            spectra = model.decode(z)
            
            # Store results
            all_latents.append(z.cpu().numpy())
            all_spectra.append(spectra.cpu().numpy())
            all_labels.extend([chem] * num_per_class)
            
            print(f"  {chem}: {num_per_class} samples")
    
    latents = np.concatenate(all_latents, axis=0)
    spectra = np.concatenate(all_spectra, axis=0)
    labels = np.array(all_labels)
    
    return latents, spectra, labels

def compute_statistics(latents, labels):
    """Compute per-chemical latent statistics"""
    stats = []
    
    for chem in CHEMICALS:
        chem_latents = latents[labels == chem]
        
        mean = np.mean(chem_latents, axis=0).mean()
        std = np.std(chem_latents, axis=0).mean()
        
        # PCA variance
        if len(chem_latents) > 1:
            pca = PCA(n_components=2)
            pca.fit(chem_latents)
            pc1_var = pca.explained_variance_ratio_[0] * 100
            pc2_var = pca.explained_variance_ratio_[1] * 100
        else:
            pc1_var = pc2_var = 0
        
        stats.append({
            'Chemical': chem,
            'Mean': mean,
            'Std': std,
            'PC1_Var': pc1_var,
            'PC2_Var': pc2_var
        })
    
    return pd.DataFrame(stats)

def create_pca_comparison(vae_latents, vae_labels, real_latents, real_labels):
    """Create per-chemical PCA comparison: Real vs VAE"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, chem in enumerate(CHEMICALS):
        ax = axes[i]
        
        # Get chemical-specific data
        vae_chem = vae_latents[vae_labels == chem]
        real_chem = real_latents[real_labels == chem]
        
        # Fit PCA on combined data
        combined = np.vstack([real_chem, vae_chem])
        pca = PCA(n_components=2)
        pca.fit(combined)
        
        # Transform
        real_pca = pca.transform(real_chem)
        vae_pca = pca.transform(vae_chem)
        
        # Plot
        ax.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', alpha=0.3, s=10, label='Real')
        ax.scatter(vae_pca[:, 0], vae_pca[:, 1], c='red', alpha=0.3, s=10, label='VAE')
        
        ax.set_title(f'{chem}\nPC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'vae_per_chemical_pca.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: vae_per_chemical_pca.png")

def create_baseline_comparison_figure():
    """Create 3-way comparison: Real | VAE | Diffusion"""
    
    # Load VAE results
    vae_latents = np.load(os.path.join(RESULTS_DIR, 'vae_generated_latents.npy'))
    vae_labels = np.load(os.path.join(RESULTS_DIR, 'vae_generated_labels.npy'))
    
    # Load Diffusion results
    diff_latents = np.load(os.path.join(RESULTS_DIR, 'full_generated_latents.npy'))
    diff_labels = np.load(os.path.join(RESULTS_DIR, 'full_generated_labels.npy'))
    
    # Load real data
    real_latents = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent_separated.npy'))
    test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
    real_labels = test_df['Label'].values
    
    # Create 3x8 grid: each column is a chemical, each row is a method
    fig, axes = plt.subplots(3, 8, figsize=(28, 12))
    
    methods = ['Real', 'VAE', 'Diffusion (Ours)']
    datasets = [
        (real_latents, real_labels),
        (vae_latents, vae_labels),
        (diff_latents, diff_labels)
    ]
    colors = ['blue', 'red', 'green']
    
    for row, (method, (latents, labels), color) in enumerate(zip(methods, datasets, colors)):
        for col, chem in enumerate(CHEMICALS):
            ax = axes[row, col]
            
            # Get chemical-specific data
            chem_latents = latents[labels == chem]
            
            # PCA
            if len(chem_latents) > 1:
                pca = PCA(n_components=2)
                pca_proj = pca.fit_transform(chem_latents)
                
                ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=color, alpha=0.4, s=5)
                
                # Title only on top row
                if row == 0:
                    ax.set_title(chem, fontsize=12, fontweight='bold')
                
                # Method label on left column
                if col == 0:
                    ax.set_ylabel(method, fontsize=11, fontweight='bold')
                
                # Variance on plot
                var_text = f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%"
                ax.text(0.05, 0.95, var_text, transform=ax.transAxes, 
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            ax.grid(True, alpha=0.2)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Baseline Comparison: Real Data vs VAE vs Geometry-Preserving Diffusion', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'baseline_comparison_3way.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: baseline_comparison_3way.png")

def main():
    """Main generation and comparison workflow"""
    
    # Load VAE model
    print("Loading VAE model...")
    model = load_vae_model()
    
    # Generate samples
    print("\nGenerating VAE samples...")
    latents, spectra, labels = generate_vae_samples(model, num_per_class=NUM_SAMPLES_PER_CLASS)
    
    # Save results
    np.save(os.path.join(RESULTS_DIR, 'vae_generated_latents.npy'), latents)
    np.save(os.path.join(RESULTS_DIR, 'vae_generated_spectra.npy'), spectra)
    np.save(os.path.join(RESULTS_DIR, 'vae_generated_labels.npy'), labels)
    
    print(f"\nGenerated {len(latents)} total samples")
    print(f"Latents shape: {latents.shape}")
    print(f"Spectra shape: {spectra.shape}")
    
    # Compute statistics
    print("\nComputing VAE statistics...")
    stats_df = compute_statistics(latents, labels)
    stats_df.to_csv(os.path.join(RESULTS_DIR, 'vae_statistics.csv'), index=False)
    print(stats_df)
    
    # Load real data for comparison
    print("\nLoading real data for comparison...")
    real_latents = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent_separated.npy'))
    test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
    real_labels = test_df['Label'].values
    
    # Create PCA comparison
    print("\nCreating PCA comparison plots...")
    create_pca_comparison(latents, labels, real_latents, real_labels)
    
    # Create 3-way comparison if diffusion results exist
    if os.path.exists(os.path.join(RESULTS_DIR, 'full_generated_latents.npy')):
        print("\nCreating 3-way baseline comparison...")
        create_baseline_comparison_figure()
    
    print("\n✓ VAE generation complete!")

if __name__ == '__main__':
    main()
