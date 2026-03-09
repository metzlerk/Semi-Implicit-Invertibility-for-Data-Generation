"""
Compare DEB Latent Space: Beta=0.02 vs Beta=0.2
================================================

Show how the two different beta schedules affect the geometry of 
generated DEB samples in latent space.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ast

ROOT_DIR = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation'
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')

sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))
from train_latent_diffusion import ClassConditionedDiffusion, get_beta_schedule, sample_diffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Parameters
LATENT_DIM = 512
SMILE_DIM = 512
NUM_CLASSES = 8
TIMESTEPS = 50
N_SAMPLES = 1000  # More samples for better PCA visualization

TARGET_CHEMICAL = 'DEB'
CHEMICALS = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']

# =============================================================================
# LOAD DATA
# =============================================================================

def load_smile_embeddings():
    """Load SMILE embeddings"""
    smile_path = os.path.join(DATA_DIR, 'name_smiles_embedding_file.csv')
    smile_df = pd.read_csv(smile_path)
    
    label_mapping = {
        'DEB': '1,2,3,4-Diepoxybutane',
        'DEM': 'Diethyl Malonate',
        'DMMP': 'Dimethyl methylphosphonate',
        'DPM': 'Oxybispropanol',
        'DtBP': 'Di-tert-butyl peroxide',
        'JP8': 'JP8',
        'MES': '2-(N-morpholino)ethanesulfonic acid',
        'TEPO': 'Triethyl phosphate'
    }
    
    embedding_dict = {}
    for _, row in smile_df.iterrows():
        if pd.notna(row['embedding']):
            embedding = np.array(ast.literal_eval(row['embedding']), dtype=np.float32)
            embedding_dict[row['Name']] = embedding
    
    label_embeddings = {}
    for label, full_name in label_mapping.items():
        if full_name in embedding_dict:
            label_embeddings[label] = embedding_dict[full_name]
    
    return label_embeddings


def load_real_deb_latents():
    """Load real DEB latent codes from test set - use SAME data model was trained on"""
    
    # Use separated latents if available (same as training uses)
    separated_test = os.path.join(RESULTS_DIR, 'autoencoder_test_latent_separated.npy')
    if os.path.exists(separated_test):
        print("  → Loading SEPARATED latents (same as training)")
        test_latent = np.load(separated_test)
    else:
        print("  → Loading ORIGINAL latents")
        test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'))
    
    test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
    test_labels = test_df['Label'].values
    
    deb_mask = test_labels == TARGET_CHEMICAL
    deb_latents = test_latent[deb_mask]
    
    print(f"  Real {TARGET_CHEMICAL} latents: mean={deb_latents.mean():.3f}, std={deb_latents.std():.3f}")
    
    # Sample randomly to match N_SAMPLES
    if len(deb_latents) > N_SAMPLES:
        indices = np.random.choice(len(deb_latents), N_SAMPLES, replace=False)
        deb_latents = deb_latents[indices]
    
    return deb_latents


def load_model(model_path, beta_end):
    """Load a trained diffusion model"""
    print(f"Loading model: {os.path.basename(model_path)}")
    print(f"  Beta: [0.001, {beta_end}]")
    
    model = ClassConditionedDiffusion(
        latent_dim=LATENT_DIM,
        smile_dim=SMILE_DIM,
        num_classes=NUM_CLASSES,
        timesteps=TIMESTEPS
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}, Loss: {checkpoint.get('loss', 0):.6f}")
    
    return model


# =============================================================================
# GENERATE SAMPLES
# =============================================================================

def generate_deb_samples(model, smile_embeddings, beta_end):
    """Generate DEB samples from a diffusion model"""
    print(f"\nGenerating {N_SAMPLES} DEB samples with beta={beta_end}...")
    
    betas = get_beta_schedule(TIMESTEPS, beta_start=0.001, beta_end=beta_end)
    betas = betas.to(device)
    
    smile_emb = smile_embeddings[TARGET_CHEMICAL]
    smile_tensor = torch.FloatTensor(smile_emb).unsqueeze(0).repeat(N_SAMPLES, 1).to(device)
    
    class_idx = CHEMICALS.index(TARGET_CHEMICAL)
    class_onehot = torch.zeros(N_SAMPLES, NUM_CLASSES).to(device)
    class_onehot[:, class_idx] = 1
    
    samples = sample_diffusion(model, smile_tensor, class_onehot, betas, device, N_SAMPLES)
    
    return samples.cpu().numpy()


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comparison(real_latents, weak_latents, strong_latents):
    """Create comprehensive PCA comparison"""
    
    fig = plt.figure(figsize=(22, 14))
    
    # Combine for unified PCA
    all_latents = np.vstack([real_latents, weak_latents, strong_latents])
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_latents)
    
    n_real = len(real_latents)
    n_weak = len(weak_latents)
    
    real_pca = all_pca[:n_real]
    weak_pca = all_pca[n_real:n_real+n_weak]
    strong_pca = all_pca[n_real+n_weak:]
    
    # Plot 1: All three overlaid
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(real_pca[:, 0], real_pca[:, 1], 
               c='blue', alpha=0.4, s=20, label='Real', edgecolors='none')
    ax1.scatter(weak_pca[:, 0], weak_pca[:, 1], 
               c='green', alpha=0.4, s=20, label='Weak (β=0.02)', edgecolors='none')
    ax1.scatter(strong_pca[:, 0], strong_pca[:, 1], 
               c='red', alpha=0.4, s=20, label='Strong (β=0.2)', edgecolors='none')
    
    # Mark centroids
    ax1.scatter(real_pca[:, 0].mean(), real_pca[:, 1].mean(), 
               c='blue', s=400, marker='*', edgecolors='black', linewidths=2, zorder=10)
    ax1.scatter(weak_pca[:, 0].mean(), weak_pca[:, 1].mean(), 
               c='green', s=400, marker='*', edgecolors='black', linewidths=2, zorder=10)
    ax1.scatter(strong_pca[:, 0].mean(), strong_pca[:, 1].mean(), 
               c='red', s=400, marker='*', edgecolors='black', linewidths=2, zorder=10)
    
    ax1.set_title(f'{TARGET_CHEMICAL}: All Methods Overlaid\n(Stars = centroids)', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Real only
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(real_pca[:, 0], real_pca[:, 1], 
               c='blue', alpha=0.5, s=30, edgecolors='white', linewidths=0.3)
    ax2.scatter(real_pca[:, 0].mean(), real_pca[:, 1].mean(), 
               c='darkblue', s=400, marker='*', edgecolors='black', linewidths=2, zorder=10)
    ax2.set_title(f'Real {TARGET_CHEMICAL} Latents\n'
                 f'μ={real_latents.mean():.3f}, σ={real_latents.std():.3f}', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Weak diffusion
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(weak_pca[:, 0], weak_pca[:, 1], 
               c='green', alpha=0.5, s=30, edgecolors='white', linewidths=0.3)
    ax3.scatter(weak_pca[:, 0].mean(), weak_pca[:, 1].mean(), 
               c='darkgreen', s=400, marker='*', edgecolors='black', linewidths=2, zorder=10)
    ax3.set_title(f'Weak Diffusion (β=0.02)\n'
                 f'μ={weak_latents.mean():.3f}, σ={weak_latents.std():.3f}', 
                 fontsize=14, fontweight='bold', color='darkgreen')
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Strong diffusion
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(strong_pca[:, 0], strong_pca[:, 1], 
               c='red', alpha=0.5, s=30, edgecolors='white', linewidths=0.3)
    ax4.scatter(strong_pca[:, 0].mean(), strong_pca[:, 1].mean(), 
               c='darkred', s=400, marker='*', edgecolors='black', linewidths=2, zorder=10)
    ax4.set_title(f'Strong Diffusion (β=0.2)\n'
                 f'μ={strong_latents.mean():.3f}, σ={strong_latents.std():.3f}', 
                 fontsize=14, fontweight='bold', color='darkred')
    ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Real vs Weak
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(real_pca[:, 0], real_pca[:, 1], 
               c='blue', alpha=0.4, s=25, label='Real', edgecolors='none')
    ax5.scatter(weak_pca[:, 0], weak_pca[:, 1], 
               c='green', alpha=0.4, s=25, label='Weak', edgecolors='none')
    ax5.set_title('Real vs Weak (β=0.02)\n✓ Good overlap', 
                 fontsize=14, fontweight='bold', color='green')
    ax5.set_xlabel(f'PC1', fontsize=12)
    ax5.set_ylabel(f'PC2', fontsize=12)
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Real vs Strong
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(real_pca[:, 0], real_pca[:, 1], 
               c='blue', alpha=0.4, s=25, label='Real', edgecolors='none')
    ax6.scatter(strong_pca[:, 0], strong_pca[:, 1], 
               c='red', alpha=0.4, s=25, label='Strong', edgecolors='none')
    ax6.set_title('Real vs Strong (β=0.2)\n⚠ Over-dispersed', 
                 fontsize=14, fontweight='bold', color='red')
    ax6.set_xlabel(f'PC1', fontsize=12)
    ax6.set_ylabel(f'PC2', fontsize=12)
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Beta Ablation Study: {TARGET_CHEMICAL} Latent Space Geometry\n'
                'Comparing Weak (β=0.02) vs Strong (β=0.2) Diffusion', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    output_path = os.path.join(IMAGES_DIR, f'deb_latent_pca_beta_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print(f"{TARGET_CHEMICAL} LATENT SPACE STATISTICS (512-D)")
    print("="*80)
    print(f"{'Metric':<30} {'Real':<15} {'Weak (0.02)':<15} {'Strong (0.2)':<15}")
    print("-"*80)
    print(f"{'Mean':<30} {real_latents.mean():<15.6f} {weak_latents.mean():<15.6f} {strong_latents.mean():<15.6f}")
    print(f"{'Std Dev':<30} {real_latents.std():<15.6f} {weak_latents.std():<15.6f} {strong_latents.std():<15.6f}")
    print(f"{'Min':<30} {real_latents.min():<15.6f} {weak_latents.min():<15.6f} {strong_latents.min():<15.6f}")
    print(f"{'Max':<30} {real_latents.max():<15.6f} {weak_latents.max():<15.6f} {strong_latents.max():<15.6f}")
    
    # L2 norms
    real_norms = np.linalg.norm(real_latents, axis=1)
    weak_norms = np.linalg.norm(weak_latents, axis=1)
    strong_norms = np.linalg.norm(strong_latents, axis=1)
    print(f"{'Mean L2 norm':<30} {real_norms.mean():<15.2f} {weak_norms.mean():<15.2f} {strong_norms.mean():<15.2f}")
    print(f"{'Std L2 norm':<30} {real_norms.std():<15.2f} {weak_norms.std():<15.2f} {strong_norms.std():<15.2f}")
    
    print("\n" + "="*80)
    print("PCA SPACE (2-D Projection)")
    print("="*80)
    print(f"{'PC1 Mean':<30} {real_pca[:, 0].mean():<15.3f} {weak_pca[:, 0].mean():<15.3f} {strong_pca[:, 0].mean():<15.3f}")
    print(f"{'PC1 Std':<30} {real_pca[:, 0].std():<15.3f} {weak_pca[:, 0].std():<15.3f} {strong_pca[:, 0].std():<15.3f}")
    print(f"{'PC2 Mean':<30} {real_pca[:, 1].mean():<15.3f} {weak_pca[:, 1].mean():<15.3f} {strong_pca[:, 1].mean():<15.3f}")
    print(f"{'PC2 Std':<30} {real_pca[:, 1].std():<15.3f} {weak_pca[:, 1].std():<15.3f} {strong_pca[:, 1].std():<15.3f}")
    print("="*80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print(f"Comparing {TARGET_CHEMICAL} Latent Space Geometry: Beta Ablation")
    print("="*80)
    
    # Load data
    smile_embeddings = load_smile_embeddings()
    real_latents = load_real_deb_latents()
    print(f"\nLoaded {len(real_latents)} real {TARGET_CHEMICAL} latent codes")
    
    # Load models and generate
    model_weak = load_model(
        os.path.join(MODELS_DIR, 'diffusion_latent_separated_beta0.02_best.pt'),
        beta_end=0.02
    )
    weak_latents = generate_deb_samples(model_weak, smile_embeddings, beta_end=0.02)
    
    model_strong = load_model(
        os.path.join(MODELS_DIR, 'diffusion_latent_separated_beta0.20_best.pt'),
        beta_end=0.2
    )
    strong_latents = generate_deb_samples(model_strong, smile_embeddings, beta_end=0.2)
    
    # Visualize
    plot_comparison(real_latents, weak_latents, strong_latents)
    
    print("\n" + "="*80)
    print("✓ Comparison complete!")
    print("="*80)


if __name__ == "__main__":
    main()
