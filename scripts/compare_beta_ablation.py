"""
Ablation Study: Compare beta_end=0.02 vs beta_end=0.2
=======================================================

Compare two diffusion models trained with different beta schedules:
- Weak diffusion (beta_end=0.02): Gentler noise, 45% noise at t=49
- Strong diffusion (beta_end=0.2): More aggressive, 94% noise at t=49

For each model, generate samples and evaluate:
1. Latent space geometry (PCA, per-chemical structure)
2. Statistical matching (mean, std per chemical)
3. Separation between classes
4. Generate decoded spectra and compare quality
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
N_SAMPLES_PER_CLASS = 500

CHEMICALS = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']

# Model paths
MODEL_WEAK = os.path.join(MODELS_DIR, 'diffusion_latent_separated_beta0.02_best.pt')
MODEL_STRONG = os.path.join(MODELS_DIR, 'diffusion_latent_separated_beta0.20_best.pt')

# =============================================================================
# LOAD DATA AND MODELS
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


def load_test_data():
    """Load test latents and labels"""
    test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'))
    test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
    test_labels = test_df['Label'].values
    return test_latent, test_labels


def load_model(model_path, beta_end):
    """Load a trained diffusion model"""
    print(f"Loading model: {os.path.basename(model_path)}")
    print(f"  Beta schedule: [0.001, {beta_end}]")
    
    model = ClassConditionedDiffusion(
        latent_dim=LATENT_DIM,
        smile_dim=SMILE_DIM,
        num_classes=NUM_CLASSES,
        timesteps=TIMESTEPS
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loss: {checkpoint.get('loss', 'unknown'):.6f}")
    
    return model


# =============================================================================
# GENERATE SAMPLES
# =============================================================================

def generate_samples_from_model(model, smile_embeddings, beta_end):
    """Generate samples from a diffusion model"""
    print(f"\nGenerating samples with beta_end={beta_end}...")
    
    betas = get_beta_schedule(TIMESTEPS, beta_start=0.001, beta_end=beta_end)
    betas = betas.to(device)
    
    all_samples = []
    all_labels = []
    
    class_to_idx = {c: i for i, c in enumerate(CHEMICALS)}
    
    for chemical in CHEMICALS:
        print(f"  {chemical}...")
        
        smile_emb = smile_embeddings[chemical]
        smile_tensor = torch.FloatTensor(smile_emb).unsqueeze(0).repeat(N_SAMPLES_PER_CLASS, 1).to(device)
        
        class_idx = class_to_idx[chemical]
        class_onehot = torch.zeros(N_SAMPLES_PER_CLASS, NUM_CLASSES).to(device)
        class_onehot[:, class_idx] = 1
        
        samples = sample_diffusion(model, smile_tensor, class_onehot, betas, device, N_SAMPLES_PER_CLASS)
        
        all_samples.append(samples.cpu().numpy())
        all_labels.extend([chemical] * N_SAMPLES_PER_CLASS)
    
    return np.vstack(all_samples), np.array(all_labels)


# =============================================================================
# ANALYSIS AND COMPARISON
# =============================================================================

def compute_statistics(samples, labels):
    """Compute per-chemical statistics"""
    stats = {}
    for chemical in CHEMICALS:
        mask = labels == chemical
        chem_samples = samples[mask]
        stats[chemical] = {
            'mean': np.mean(chem_samples),
            'std': np.std(chem_samples),
            'samples': chem_samples
        }
    return stats


def compute_separation(samples, labels):
    """Compute inter-class separation (mean distance between centroids)"""
    centroids = {}
    for chemical in CHEMICALS:
        mask = labels == chemical
        centroids[chemical] = samples[mask].mean(axis=0)
    
    # Pairwise distances
    distances = []
    for i, chem1 in enumerate(CHEMICALS):
        for chem2 in CHEMICALS[i+1:]:
            dist = np.linalg.norm(centroids[chem1] - centroids[chem2])
            distances.append(dist)
    
    return np.mean(distances), np.std(distances)


def plot_comparison(real_samples, real_labels, weak_samples, weak_labels, 
                   strong_samples, strong_labels):
    """Create comparison visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Combine all samples for unified PCA
    all_combined = np.vstack([real_samples, weak_samples, strong_samples])
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_combined)
    
    n_real = len(real_samples)
    n_weak = len(weak_samples)
    
    real_pca = all_pca[:n_real]
    weak_pca = all_pca[n_real:n_real+n_weak]
    strong_pca = all_pca[n_real+n_weak:]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(CHEMICALS)))
    chem_to_color = {c: colors[i] for i, c in enumerate(CHEMICALS)}
    
    # Plot 1: Real data
    ax = axes[0, 0]
    for chemical in CHEMICALS:
        mask = real_labels == chemical
        ax.scatter(real_pca[mask, 0], real_pca[mask, 1], 
                  c=[chem_to_color[chemical]], alpha=0.5, s=20, 
                  label=chemical, edgecolors='white', linewidths=0.3)
    ax.set_title('Real Latent Codes\n(From Encoder)', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Weak diffusion (beta_end=0.02)
    ax = axes[0, 1]
    for chemical in CHEMICALS:
        mask = weak_labels == chemical
        ax.scatter(weak_pca[mask, 0], weak_pca[mask, 1], 
                  c=[chem_to_color[chemical]], alpha=0.5, s=20,
                  edgecolors='white', linewidths=0.3)
    ax.set_title('Weak Diffusion (β_end=0.02)\n45% noise at t=49', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Strong diffusion (beta_end=0.2)
    ax = axes[0, 2]
    for chemical in CHEMICALS:
        mask = strong_labels == chemical
        ax.scatter(strong_pca[mask, 0], strong_pca[mask, 1], 
                  c=[chem_to_color[chemical]], alpha=0.5, s=20,
                  edgecolors='white', linewidths=0.3)
    ax.set_title('Strong Diffusion (β_end=0.2)\n94% noise at t=49', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 4-6: Per-chemical overlay for one chemical (DEB)
    target_chem = 'DEB'
    for i, (samples, pca_coords, labels, title) in enumerate([
        (real_samples, real_pca, real_labels, 'Real'),
        (weak_samples, weak_pca, weak_labels, 'Weak (β=0.02)'),
        (strong_samples, strong_pca, strong_labels, 'Strong (β=0.2)')
    ]):
        ax = axes[1, i]
        
        # Plot all chemicals in gray
        for chemical in CHEMICALS:
            mask = labels == chemical
            ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                      c='lightgray', alpha=0.2, s=10)
        
        # Highlight target chemical
        mask = labels == target_chem
        ax.scatter(pca_coords[mask, 0], pca_coords[mask, 1], 
                  c=[chem_to_color[target_chem]], alpha=0.7, s=30,
                  label=target_chem, edgecolors='white', linewidths=0.5)
        
        ax.set_title(f'{title}: {target_chem} Highlighted', fontsize=12, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=10)
        ax.set_ylabel('PC2', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Beta Ablation Study: Weak vs Strong Diffusion\n'
                'Latent Space Geometry Comparison', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = os.path.join(IMAGES_DIR, 'beta_ablation_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    return output_path


def print_statistics_table(real_stats, weak_stats, strong_stats, 
                          weak_sep, strong_sep):
    """Print comparison table"""
    print("\n" + "="*100)
    print("COMPARISON TABLE: Weak (β_end=0.02) vs Strong (β_end=0.2)")
    print("="*100)
    print(f"{'Chemical':<10} {'Real Mean':<12} {'Weak Mean':<12} {'Strong Mean':<12} "
          f"{'Real Std':<12} {'Weak Std':<12} {'Strong Std':<12}")
    print("-"*100)
    
    for chemical in CHEMICALS:
        print(f"{chemical:<10} "
              f"{real_stats[chemical]['mean']:<12.4f} "
              f"{weak_stats[chemical]['mean']:<12.4f} "
              f"{strong_stats[chemical]['mean']:<12.4f} "
              f"{real_stats[chemical]['std']:<12.4f} "
              f"{weak_stats[chemical]['std']:<12.4f} "
              f"{strong_stats[chemical]['std']:<12.4f}")
    
    print("\n" + "="*100)
    print("INTER-CLASS SEPARATION (Mean centroid distance)")
    print("="*100)
    print(f"  Weak beta (0.02):   {weak_sep[0]:.3f} ± {weak_sep[1]:.3f}")
    print(f"  Strong beta (0.2):  {strong_sep[0]:.3f} ± {strong_sep[1]:.3f}")
    print("="*100)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*100)
    print("BETA ABLATION STUDY: Comparing β_end = 0.02 vs 0.2")
    print("="*100)
    
    # Check if both models exist
    if not os.path.exists(MODEL_WEAK):
        print(f"✗ Model not found: {MODEL_WEAK}")
        return
    if not os.path.exists(MODEL_STRONG):
        print(f"✗ Model not found: {MODEL_STRONG}")
        print(f"  → Model is still training. Please run this script after training completes.")
        return
    
    # Load data
    smile_embeddings = load_smile_embeddings()
    real_samples, real_labels = load_test_data()
    
    # Sample real data to match generated sample count
    indices = np.random.choice(len(real_samples), N_SAMPLES_PER_CLASS * NUM_CLASSES, replace=False)
    real_samples = real_samples[indices]
    real_labels = real_labels[indices]
    
    # Load models and generate
    model_weak = load_model(MODEL_WEAK, beta_end=0.02)
    weak_samples, weak_labels = generate_samples_from_model(model_weak, smile_embeddings, beta_end=0.02)
    
    model_strong = load_model(MODEL_STRONG, beta_end=0.2)
    strong_samples, strong_labels = generate_samples_from_model(model_strong, smile_embeddings, beta_end=0.2)
    
    # Compute statistics
    real_stats = compute_statistics(real_samples, real_labels)
    weak_stats = compute_statistics(weak_samples, weak_labels)
    strong_stats = compute_statistics(strong_samples, strong_labels)
    
    weak_sep = compute_separation(weak_samples, weak_labels)
    strong_sep = compute_separation(strong_samples, strong_labels)
    
    # Visualize
    plot_comparison(real_samples, real_labels, 
                   weak_samples, weak_labels,
                   strong_samples, strong_labels)
    
    # Print results
    print_statistics_table(real_stats, weak_stats, strong_stats, weak_sep, strong_sep)
    
    print("\n" + "="*100)
    print("✓ Ablation study complete!")
    print("="*100)


if __name__ == "__main__":
    main()
