"""
Visualize Diffusion Trajectory in Latent Space
================================================

Show how a single chemical (DEB) is transformed through the diffusion process.
For different timesteps t, show where latent samples move in PCA space.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Paths
ROOT_DIR = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation'
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')

# Import diffusion utilities
sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))
from train_latent_diffusion import get_beta_schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters (can be overridden by command line)
import sys
TARGET_CHEMICAL = 'DEB'
N_SAMPLES = 500
TIMESTEPS = 50
TIMESTEPS_TO_SHOW = [0, 10, 20, 30, 40, 49]  # Show progression (0-indexed, max is 49)
BETA_END = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2  # Default to 0.2, can pass 0.02 for weak

# =============================================================================
# FORWARD DIFFUSION
# =============================================================================

def apply_forward_diffusion(x0, t, betas):
    """
    Apply forward diffusion: q(x_t | x_0)
    
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Get alpha_bar for timestep t
    alpha_bar_t = alphas_cumprod[t]
    
    # Sample noise
    epsilon = torch.randn_like(x0)
    
    # Apply diffusion
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
    
    x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * epsilon
    
    return x_t, epsilon


# =============================================================================
# LOAD DATA
# =============================================================================

def load_deb_samples():
    """Load DEB latent samples from test set"""
    print(f"Loading {TARGET_CHEMICAL} samples...")
    
    # Load test latents
    test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'))
    test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
    test_labels = test_df['Label'].values
    
    # Get DEB samples
    deb_mask = test_labels == TARGET_CHEMICAL
    deb_latent = test_latent[deb_mask]
    
    # Sample N_SAMPLES randomly
    indices = np.random.choice(len(deb_latent), N_SAMPLES, replace=False)
    deb_samples = deb_latent[indices]
    
    print(f"  Loaded {len(deb_samples)} {TARGET_CHEMICAL} samples")
    print(f"  Shape: {deb_samples.shape}")
    
    return deb_samples


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_diffusion_trajectory(deb_samples, timesteps_to_show):
    """
    Create visualization showing how DEB moves through diffusion process
    """
    print(f"\nGenerating diffusion trajectory with beta_end={BETA_END}...")
    
    # Get beta schedule
    betas = get_beta_schedule(TIMESTEPS, beta_start=0.001, beta_end=BETA_END)
    
    # Convert to torch
    x0 = torch.FloatTensor(deb_samples).to(device)
    
    # Apply diffusion at different timesteps
    diffused_samples = {}
    diffused_samples[0] = x0.cpu().numpy()  # t=0 is clean
    
    for t in timesteps_to_show[1:]:  # Skip t=0
        print(f"  Applying diffusion at t={t}...")
        x_t, _ = apply_forward_diffusion(x0, t, betas)
        diffused_samples[t] = x_t.cpu().numpy()
    
    # Combine all samples for PCA
    print("\nComputing PCA on combined samples...")
    all_samples = np.vstack(list(diffused_samples.values()))
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_samples)
    
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
    
    # Split PCA results back
    diffused_pca = {}
    idx = 0
    for t in timesteps_to_show:
        n = len(diffused_samples[t])
        diffused_pca[t] = all_pca[idx:idx+n]
        idx += n
    
    # Create visualization
    print("\nCreating visualization...")
    
    # Two plots: one with all timesteps overlaid, one with separate subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Color map for timesteps
    colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps_to_show)))
    
    # Plot 1: All timesteps overlaid - show centroids and spread
    ax1 = plt.subplot(2, 4, (1, 5))
    
    # First, plot all points with low alpha
    for i, t in enumerate(timesteps_to_show):
        pca_coords = diffused_pca[t]
        ax1.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                   c=[colors[i]], alpha=0.3, s=15, 
                   edgecolors='none')
    
    # Then plot centroids with lines showing movement
    centroids = []
    for i, t in enumerate(timesteps_to_show):
        pca_coords = diffused_pca[t]
        centroid = pca_coords.mean(axis=0)
        centroids.append(centroid)
        ax1.scatter(centroid[0], centroid[1], 
                   c=[colors[i]], s=200, marker='*', 
                   edgecolors='black', linewidths=2,
                   label=f't={t}', zorder=10)
    
    # Draw lines connecting centroids
    centroids = np.array(centroids)
    ax1.plot(centroids[:, 0], centroids[:, 1], 'k--', alpha=0.5, linewidth=2, zorder=5)
    
    ax1.set_title(f'{TARGET_CHEMICAL} Through Diffusion Process (β={BETA_END})\n'
                 f'Overlay: Stars = centroids, Lines = trajectory', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax1.legend(fontsize=9, loc='best', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2-7: Individual timesteps with adaptive axis limits
    for i, t in enumerate(timesteps_to_show):
        if i < 3:
            ax = plt.subplot(2, 4, i + 2)
        else:
            ax = plt.subplot(2, 4, i + 3)
        
        pca_coords = diffused_pca[t]
        ax.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                  c=[colors[i]], alpha=0.6, s=30, 
                  edgecolors='white', linewidths=0.3)
        
        # Add centroid
        centroid = pca_coords.mean(axis=0)
        ax.scatter(centroid[0], centroid[1], c='red', s=150, marker='*',
                  edgecolors='black', linewidths=1.5, zorder=10)
        
        # Compute spread statistics
        std_pc1 = pca_coords[:, 0].std()
        std_pc2 = pca_coords[:, 1].std()
        
        if t == 0:
            title = f't={t} (Clean Latent)'
        else:
            title = f't={t}'
        
        ax.set_title(f'{title}\nσ_PC1={std_pc1:.2f}, σ_PC2={std_pc2:.2f}', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel(f'PC1', fontsize=10)
        ax.set_ylabel(f'PC2', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Use per-timestep axis limits for better detail
        # But keep a 3-std range to show the distribution clearly
        x_range = max(std_pc1 * 3, 5)
        y_range = max(std_pc2 * 3, 5)
        ax.set_xlim(centroid[0] - x_range, centroid[0] + x_range)
        ax.set_ylim(centroid[1] - y_range, centroid[1] + y_range)
    
    plt.suptitle(f'Diffusion Trajectory: {TARGET_CHEMICAL} in Latent Space (β_end={BETA_END})\n'
                f'How Clean Latent Structure Transforms Through Forward Diffusion', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(IMAGES_DIR, f'diffusion_trajectory_{TARGET_CHEMICAL.lower()}_beta{BETA_END:.2f}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    # Create second plot: Density evolution with adaptive ranges
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, t in enumerate(timesteps_to_show):
        ax = axes[i]
        pca_coords = diffused_pca[t]
        
        # Compute centroid for centering
        centroid = pca_coords.mean(axis=0)
        std_pc1 = pca_coords[:, 0].std()
        std_pc2 = pca_coords[:, 1].std()
        
        # Adaptive axis limits (±3 std)
        x_range = max(std_pc1 * 3, 5)
        y_range = max(std_pc2 * 3, 5)
        xlim = [centroid[0] - x_range, centroid[0] + x_range]
        ylim = [centroid[1] - y_range, centroid[1] + y_range]
        
        # 2D histogram
        h = ax.hist2d(pca_coords[:, 0], pca_coords[:, 1], 
                     bins=40, cmap='viridis', alpha=0.8,
                     range=[xlim, ylim])
        plt.colorbar(h[3], ax=ax, label='Count')
        
        # Mark centroid
        ax.scatter(centroid[0], centroid[1], c='red', s=200, marker='*',
                  edgecolors='white', linewidths=2, zorder=10)
        
        if t == 0:
            title = f't={t} (Clean Latent)'
        else:
            title = f't={t}'
        
        ax.set_title(f'{title}\nσ={std_pc1:.2f} (PC1), {std_pc2:.2f} (PC2)', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)
        ax.grid(True, alpha=0.3, color='white')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
    plt.suptitle(f'Density Evolution: {TARGET_CHEMICAL} Through Diffusion Process (β_end={BETA_END})', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(IMAGES_DIR, f'diffusion_density_{TARGET_CHEMICAL.lower()}_beta{BETA_END:.2f}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Diffusion Statistics:")
    print("="*60)
    print(f"{'Timestep':<10} {'Mean PC1':<12} {'Std PC1':<12} {'Mean PC2':<12} {'Std PC2':<12}")
    print("-"*60)
    for t in timesteps_to_show:
        pca_coords = diffused_pca[t]
        mean_pc1 = pca_coords[:, 0].mean()
        std_pc1 = pca_coords[:, 0].std()
        mean_pc2 = pca_coords[:, 1].mean()
        std_pc2 = pca_coords[:, 1].std()
        print(f"t={t:<8} {mean_pc1:<12.3f} {std_pc1:<12.3f} {mean_pc2:<12.3f} {std_pc2:<12.3f}")
    print("="*60)
    
    return diffused_samples, diffused_pca


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print(f"Visualizing Diffusion Trajectory for {TARGET_CHEMICAL}")
    print(f"Beta schedule: [0.001, {BETA_END}]")
    print("="*70)
    
    # Load data
    deb_samples = load_deb_samples()
    
    # Visualize
    diffused_samples, diffused_pca = visualize_diffusion_trajectory(
        deb_samples, TIMESTEPS_TO_SHOW
    )
    
    print("\n" + "="*70)
    print("✓ Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
