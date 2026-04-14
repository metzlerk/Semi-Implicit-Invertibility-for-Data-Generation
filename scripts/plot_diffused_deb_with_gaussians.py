"""
Plot Diffused DEB PCA with Normal Distribution Draws
=====================================================

Show diffused DEB samples (beta=0.02) alongside draws from N(0,1), N(0,1.5), N(0,2)
to illustrate sampling approach in diffused latent space.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Paths
ROOT_DIR = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation'
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')

# Parameters
TARGET_CHEMICAL = 'DEB'
N_SAMPLES_DEB = 500
N_SAMPLES_GAUSSIAN = 300
LATENT_DIM = 512
TIMESTEPS = 50
BETA_START = 0.001
BETA_END = 0.02

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_beta_schedule(timesteps, beta_start=0.001, beta_end=0.02):
    """Linear beta schedule"""
    return torch.linspace(beta_start, beta_end, timesteps)


def apply_forward_diffusion(x0, t, betas):
    """Apply forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon"""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alpha_bar_t = alphas_cumprod[t]
    
    epsilon = torch.randn_like(x0)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
    
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * epsilon
    return x_t


def main():
    print(f"Loading {TARGET_CHEMICAL} samples...")
    
    # Load DEB latent samples from test set
    test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'))
    test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
    test_labels = test_df['Label'].values
    
    deb_mask = test_labels == TARGET_CHEMICAL
    deb_latent = test_latent[deb_mask]
    
    # Sample N_SAMPLES_DEB randomly
    indices = np.random.choice(len(deb_latent), N_SAMPLES_DEB, replace=False)
    deb_samples = deb_latent[indices]
    
    print(f"  Loaded {len(deb_samples)} {TARGET_CHEMICAL} samples, shape: {deb_samples.shape}")
    
    # Apply forward diffusion at final timestep
    print(f"\nApplying forward diffusion with beta_end={BETA_END}...")
    betas = get_beta_schedule(TIMESTEPS, BETA_START, BETA_END)
    x0 = torch.FloatTensor(deb_samples).to(device)
    x_t = apply_forward_diffusion(x0, TIMESTEPS - 1, betas)  # t=49 (last timestep)
    diffused_deb = x_t.cpu().numpy()
    
    print(f"  Diffused DEB shape: {diffused_deb.shape}")
    print(f"  Diffused DEB mean: {diffused_deb.mean():.4f}, std: {diffused_deb.std():.4f}")
    
    # Draw samples from N(0, sigma^2) for different sigma values
    print(f"\nDrawing {N_SAMPLES_GAUSSIAN} samples from different Gaussians...")
    gaussian_1 = np.random.normal(0, 1.0, size=(N_SAMPLES_GAUSSIAN, LATENT_DIM))
    gaussian_1_5 = np.random.normal(0, 1.5, size=(N_SAMPLES_GAUSSIAN, LATENT_DIM))
    gaussian_2 = np.random.normal(0, 2.0, size=(N_SAMPLES_GAUSSIAN, LATENT_DIM))
    
    print(f"  N(0,1) - mean: {gaussian_1.mean():.4f}, std: {gaussian_1.std():.4f}")
    print(f"  N(0,1.5) - mean: {gaussian_1_5.mean():.4f}, std: {gaussian_1_5.std():.4f}")
    print(f"  N(0,2) - mean: {gaussian_2.mean():.4f}, std: {gaussian_2.std():.4f}")
    
    # Combine all samples for PCA
    print("\nComputing PCA...")
    all_samples = np.vstack([diffused_deb, gaussian_1, gaussian_1_5, gaussian_2])
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_samples)
    
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
    
    # Split PCA results
    idx = 0
    diffused_deb_pca = all_pca[idx:idx+N_SAMPLES_DEB]
    idx += N_SAMPLES_DEB
    gaussian_1_pca = all_pca[idx:idx+N_SAMPLES_GAUSSIAN]
    idx += N_SAMPLES_GAUSSIAN
    gaussian_1_5_pca = all_pca[idx:idx+N_SAMPLES_GAUSSIAN]
    idx += N_SAMPLES_GAUSSIAN
    gaussian_2_pca = all_pca[idx:idx+N_SAMPLES_GAUSSIAN]
    
    # Remove outliers from diffused DEB using IQR method on PC1
    print("\nRemoving outliers from diffused DEB...")
    pc1_values = diffused_deb_pca[:, 0]
    q1, q3 = np.percentile(pc1_values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    outlier_mask = (pc1_values >= lower_bound) & (pc1_values <= upper_bound)
    n_outliers = (~outlier_mask).sum()
    diffused_deb_pca = diffused_deb_pca[outlier_mask]
    
    print(f"  Removed {n_outliers} outliers from {N_SAMPLES_DEB} samples")
    print(f"  Kept {len(diffused_deb_pca)} samples")
    
    # Create visualization - 3x1 column layout
    print("\nCreating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    # Get global axis limits for consistency
    all_pca_coords = [gaussian_1_pca, gaussian_1_5_pca, gaussian_2_pca]
    all_combined = np.vstack(all_pca_coords)
    x_min, x_max = all_combined[:, 0].min(), all_combined[:, 0].max()
    y_min, y_max = all_combined[:, 1].min(), all_combined[:, 1].max()
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    # Plot data for each subplot
    plot_configs = [
        (gaussian_1_pca, '#ff7f0e', 's', 'N(0, 1)'),
        (gaussian_1_5_pca, '#2ca02c', '^', 'N(0, 1.5)'),
        (gaussian_2_pca, '#d62728', 'D', 'N(0, 2)')
    ]
    
    for idx, (pca_coords, color, marker, title) in enumerate(plot_configs):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                  c=color, alpha=0.6, s=50, 
                  edgecolors='white', linewidths=0.5, marker=marker)
        
        # Add centroid
        centroid = pca_coords.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], 
                  c=color, s=400, marker='*',
                  edgecolors='black', linewidths=2.5, zorder=10)
        
        # Add statistics
        std_pc1 = pca_coords[:, 0].std()
        std_pc2 = pca_coords[:, 1].std()
        stats_text = f'σ_PC1={std_pc1:.2f}\nσ_PC2={std_pc2:.2f}\nn={len(pca_coords)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Gaussian Draws in Latent Space', 
                fontsize=18, fontweight='bold', y=0.998)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(IMAGES_DIR, 'diffused_deb_gaussian_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    plt.close()
    
    print("Done!")


if __name__ == '__main__':
    main()
