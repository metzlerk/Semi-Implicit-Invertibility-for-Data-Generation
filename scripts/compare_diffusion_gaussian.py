"""
Compare Diffusion vs Gaussian Sampling for IMS Generation
==========================================================

Compare three approaches:
1. Real: IMS → Encoder → Latent (real distribution)
2. Gaussian: Draw from N(μ, σ) fitted to each chemical's latent distribution  
3. Diffusion: Sample from trained diffusion model conditioned on chemical

For each, we visualize:
- PCA in latent space
- PCA in IMS space (after decoding)
- Comparison to SMILE embedding structure
"""

import sys
sys.path.insert(0, '/home/kjmetzler/ChemicalDataGeneration/models')

import os
import torch
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

# Import diffusion model
sys.path.insert(0, os.path.join(ROOT_DIR, 'scripts'))
from train_latent_diffusion import ClassConditionedDiffusion, get_beta_schedule, sample_diffusion

# Model paths
ENCODER_PATH = '/scratch/cmdunham/trained_models/spectrum/current_best_models/nine_layer_ims_to_chemnet_encoder.pth'
DECODER_PATH = '/scratch/cmdunham/trained_models/spectrum/current_best_models/nine_layer_ChemNet_to_ims_generator_from_nine_layer__encoder.pth'
DIFFUSION_PATH = os.path.join(MODELS_DIR, 'diffusion_latent_separated_best.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 512
SMILE_DIM = 512
NUM_CLASSES = 8
TIMESTEPS = 50
N_SAMPLES_PER_CLASS = 150

# =============================================================================
# LOAD MODELS AND DATA
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


def load_models():
    """Load diffusion model only (skip decoder - focus on latent space comparison)"""
    print("Loading diffusion model...")
    
    # Load diffusion
    diffusion = ClassConditionedDiffusion(
        latent_dim=LATENT_DIM,
        smile_dim=SMILE_DIM,
        num_classes=NUM_CLASSES,
        timesteps=TIMESTEPS
    ).to(device)
    
    checkpoint = torch.load(DIFFUSION_PATH, map_location=device, weights_only=False)
    diffusion.load_state_dict(checkpoint['model_state_dict'])
    diffusion.eval()
    
    print("✓ Diffusion model loaded")
    return diffusion


def load_test_data():
    """Load test latents and labels"""
    test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'))
    test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
    test_labels = test_df['Label'].values
    
    # Also get IMS spectra
    spectrum_cols = [col for col in test_df.columns if col.startswith('p_')]
    test_ims = test_df[spectrum_cols].values
    
    return test_latent, test_labels, test_ims


# =============================================================================
# GENERATE SAMPLES
# =============================================================================

def generate_samples(diffusion, smile_embeddings):
    """Generate samples from all three methods"""
    
    # Load test data
    test_latent, test_labels, test_ims = load_test_data()
    
    unique_classes = sorted(np.unique(test_labels))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    
    # Storage
    all_real_latent = []
    all_gaussian_latent = []
    all_diffusion_latent = []
    all_labels = []
    
    betas = get_beta_schedule(TIMESTEPS).to(device)
    
    print("\nGenerating samples for each chemical...")
    
    for chemical in unique_classes:
        print(f"  {chemical}...")
        
        # Get real samples for this chemical
        class_mask = test_labels == chemical
        class_latent = test_latent[class_mask]
        
        # Sample N_SAMPLES_PER_CLASS real examples
        indices = np.random.choice(len(class_latent), N_SAMPLES_PER_CLASS, replace=False)
        real_latent_samples = class_latent[indices]
        
        # Gaussian: Fit N(μ, σ) to this chemical's latent distribution
        mu = class_latent.mean(axis=0)
        sigma = class_latent.std(axis=0)
        gaussian_latent_samples = np.random.normal(mu, sigma, size=(N_SAMPLES_PER_CLASS, LATENT_DIM))
        
        # Diffusion: Sample conditioned on this chemical
        smile_emb = smile_embeddings[chemical]
        smile_tensor = torch.FloatTensor(smile_emb).unsqueeze(0).repeat(N_SAMPLES_PER_CLASS, 1).to(device)
        
        class_idx = class_to_idx[chemical]
        class_onehot = torch.zeros(N_SAMPLES_PER_CLASS, NUM_CLASSES).to(device)
        class_onehot[:, class_idx] = 1
        
        diffusion_latent_samples = sample_diffusion(diffusion, smile_tensor, class_onehot, betas, device, N_SAMPLES_PER_CLASS)
        diffusion_latent_samples = diffusion_latent_samples.cpu().numpy()
        
        # Store
        all_real_latent.append(real_latent_samples)
        all_gaussian_latent.append(gaussian_latent_samples)
        all_diffusion_latent.append(diffusion_latent_samples)
        all_labels.extend([chemical] * N_SAMPLES_PER_CLASS)
    
    # Concatenate
    results = {
        'real_latent': np.vstack(all_real_latent),
        'gaussian_latent': np.vstack(all_gaussian_latent),
        'diffusion_latent': np.vstack(all_diffusion_latent),
        'labels': np.array(all_labels)
    }
    
    print(f"\n✓ Generated {len(all_labels)} samples total")
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_pca_comparison(results, smile_embeddings):
    """Create latent space PCA comparison plot"""
    
    unique_classes = sorted(np.unique(results['labels']))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_to_color = {c: colors[i] for i, c in enumerate(unique_classes)}
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Latent space PCA (Real vs Gaussian vs Diffusion)
    ax = axes[0]
    combined_latent = np.vstack([results['real_latent'], results['gaussian_latent'], results['diffusion_latent']])
    pca_latent = PCA(n_components=2)
    latent_pca = pca_latent.fit_transform(combined_latent)
    
    n_per_source = len(results['real_latent'])
    real_pca = latent_pca[:n_per_source]
    gaussian_pca = latent_pca[n_per_source:2*n_per_source]
    diffusion_pca = latent_pca[2*n_per_source:]
    
    for i, chemical in enumerate(unique_classes):
        mask = results['labels'] == chemical
        color = class_to_color[chemical]
        
        ax.scatter(real_pca[mask, 0], real_pca[mask, 1], c=[color], marker='o', s=50, alpha=0.6, label=f'{chemical} (Real)', edgecolors='white', linewidths=0.5)
        ax.scatter(gaussian_pca[mask, 0], gaussian_pca[mask, 1], c=[color], marker='^', s=50, alpha=0.6, edgecolors='white', linewidths=0.5)
        ax.scatter(diffusion_pca[mask, 0], diffusion_pca[mask, 1], c=[color], marker='s', s=50, alpha=0.6, edgecolors='white', linewidths=0.5)
    
    ax.set_title(f'Latent Space PCA\nPC1: {pca_latent.explained_variance_ratio_[0]:.1%}, PC2: {pca_latent.explained_variance_ratio_[1]:.1%}', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. SMILE embeddings PCA
    ax = axes[1]
    smile_matrix = np.array([smile_embeddings[c] for c in unique_classes])
    pca_smile = PCA(n_components=2)
    smile_pca = pca_smile.fit_transform(smile_matrix)
    
    for i, chemical in enumerate(unique_classes):
        color = class_to_color[chemical]
        ax.scatter(smile_pca[i, 0], smile_pca[i, 1], c=[color], marker='*', s=800, edgecolors='black', linewidths=3, label=chemical, zorder=10)
        ax.text(smile_pca[i, 0], smile_pca[i, 1], f'  {chemical}', fontsize=12, fontweight='bold', va='center')
    
    ax.set_title(f'SMILE Embeddings PCA (True Chemical Structure)\nPC1: {pca_smile.explained_variance_ratio_[0]:.1%}, PC2: {pca_smile.explained_variance_ratio_[1]:.1%}', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add legend box
    legend_text = """
    Markers:
    • Circle (o): Real latent codes (from encoder)
    • Triangle (^): Gaussian N(μ,σ) fitted per chemical
    • Square (s): Diffusion samples (conditioned)
    • Star (*): SMILE embedding (true structure)
    
    Goal: Diffusion should capture chemical structure better than Gaussian
    """
    fig.text(0.5, -0.05, legend_text, ha='center', fontsize=11, family='monospace', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'latent_space_pca_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: latent_space_pca_comparison.png")


def main():
    print("="*60)
    print("Comparing Diffusion vs Gaussian for IMS Generation")
    print("="*60)
    
    # Load everything
    smile_embeddings = load_smile_embeddings()
    diffusion = load_models()
    
    # Generate samples
    results = generate_samples(diffusion, smile_embeddings)
    
    # Visualize
    plot_pca_comparison(results, smile_embeddings)
    
    print("\n" + "="*60)
    print("✓ Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
