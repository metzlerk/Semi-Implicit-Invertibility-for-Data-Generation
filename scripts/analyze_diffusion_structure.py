#!/usr/bin/env python3
"""
Comprehensive Diffusion Structure Analysis
==========================================

Single script for all structural evaluation:
- Generate samples from normalized/separated diffusion models
- Compare structure to real latent space (PCA, distributions)
- Compute advanced metrics (MMD, coverage, precision)
- Compare model variants
- Visualize everything

Usage:
    python analyze_diffusion_structure.py --model normalized --generate
    python analyze_diffusion_structure.py --model separated --generate
    python analyze_diffusion_structure.py --compare  # Compare both models
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, cdist
from scipy.stats import ks_2samp
import ast
import seaborn as sns
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DATA_DIR = 'Data'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
IMAGES_DIR = 'images'

# Model hyperparameters
LATENT_DIM = 512
SMILE_DIM = 512
NUM_CLASSES = 8
TIMESTEPS = 1000  # Updated from 50 to 1000
HIDDEN_DIM = 512
NUM_LAYERS = 6

# =============================================================================
# MODEL DEFINITION
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ClassConditionedDiffusion(nn.Module):
    def __init__(self, latent_dim=512, smile_dim=512, num_classes=8, 
                 timesteps=50, hidden_dim=512, num_layers=6,
                 beta_start=0.00002, beta_end=0.005):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        
        time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        input_dim = latent_dim + smile_dim + num_classes
        
        layers = []
        layers.append(nn.Linear(input_dim + time_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
        
        # COSINE SCHEDULE (matching train_diffusion_normalized_full.py)
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999).float()
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def forward(self, x, t, smile_emb, class_onehot):
        t_emb = self.time_mlp(t)
        x_in = torch.cat([x, smile_emb, class_onehot], dim=1)
        x_in = torch.cat([x_in, t_emb], dim=1)
        return self.net(x_in)

# =============================================================================
# SAMPLING
# =============================================================================

@torch.no_grad()
def sample_diffusion(model, n_samples, smile_emb, class_onehot, device):
    """Sample from diffusion model using DDIM (deterministic, better for 1000 timesteps)"""
    model.eval()
    z_t = torch.randn(n_samples, model.latent_dim, device=device)
    
    # Use DDIM with 100 steps through 1000 timesteps
    ddim_steps = 100
    step_size = model.timesteps // ddim_steps
    timesteps = list(range(0, model.timesteps, step_size))
    
    for i in reversed(range(len(timesteps))):
        t = timesteps[i]
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
        
        predicted_noise = model(z_t, t_batch, smile_emb, class_onehot)
        
        alpha_bar_t = model.alphas_cumprod[t]
        
        if i > 0:
            t_prev = timesteps[i-1]
            alpha_bar_prev = model.alphas_cumprod[t_prev]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)
        
        # DDIM update (deterministic)
        x_0_pred = (z_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        x_0_pred = torch.clamp(x_0_pred, -10, 10)
        
        dir_xt = torch.sqrt(1 - alpha_bar_prev) * predicted_noise
        z_t = torch.sqrt(alpha_bar_prev) * x_0_pred + dir_xt
    
    return z_t.cpu().numpy()

# =============================================================================
# METRICS
# =============================================================================

def compute_mmd(X, Y, gamma=None):
    """Compute Maximum Mean Discrepancy"""
    n, m = len(X), len(Y)
    
    if gamma is None:
        all_data = np.vstack([X[:1000], Y[:1000]])
        dists = cdist(all_data, all_data, 'euclidean')
        gamma = 1.0 / np.median(dists[dists > 0])
    
    XX = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
    YY = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
    XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
    
    mmd = XX.sum() / (n * n) + YY.sum() / (m * m) - 2 * XY.sum() / (n * m)
    return mmd

def compute_coverage_precision(real, fake, k=5):
    """Compute coverage and precision metrics"""
    nbrs_real = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(real)
    nbrs_fake = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(fake)
    
    dists_real_to_fake, _ = nbrs_fake.kneighbors(real)
    coverage = (dists_real_to_fake[:, 0] < np.percentile(dists_real_to_fake, 50)).mean()
    
    dists_fake_to_real, _ = nbrs_real.kneighbors(fake)
    precision = (dists_fake_to_real[:, 0] < np.percentile(dists_fake_to_real, 50)).mean()
    
    return coverage, precision

# =============================================================================
# DATA LOADING
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
            label_embeddings[label] = torch.FloatTensor(embedding_dict[full_name])
    
    return label_embeddings

# =============================================================================
# GENERATION
# =============================================================================

def generate_samples(model_type='normalized', n_samples_per_class=1000):
    """Generate samples from specified diffusion model"""
    
    print(f"\n{'='*80}")
    print(f"GENERATING SAMPLES: {model_type.upper()} DIFFUSION")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy'))
    test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent_separated.npy'))
    
    train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))
    label_columns = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
    
    smile_dict = load_smile_embeddings()
    
    # Get normalization params
    all_latents = np.vstack([train_latent, test_latent])
    DATA_MEAN = all_latents.mean()
    DATA_STD = all_latents.std()
    print(f"Normalization: mean={DATA_MEAN:.4f}, std={DATA_STD:.4f}")
    
    # Load model
    print(f"\nLoading {model_type} diffusion model...")
    model = ClassConditionedDiffusion(
        latent_dim=LATENT_DIM,
        smile_dim=SMILE_DIM,
        num_classes=NUM_CLASSES,
        timesteps=TIMESTEPS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    
    model_path = os.path.join(MODELS_DIR, f'diffusion_latent_{model_type}_best.pt')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded")
    
    # Generate samples
    print(f"\nGenerating {n_samples_per_class} samples per class...")
    all_generated = []
    all_labels = []
    
    for class_idx, chemical in enumerate(label_columns):
        print(f"  {chemical}...", end=' ', flush=True)
        
        smile_emb = smile_dict[chemical].unsqueeze(0).repeat(n_samples_per_class, 1).to(device)
        class_onehot = torch.zeros(n_samples_per_class, NUM_CLASSES, device=device)
        class_onehot[:, class_idx] = 1.0
        
        samples_normalized = sample_diffusion(model, n_samples_per_class, smile_emb, class_onehot, device)
        
        # Denormalize if normalized model
        if model_type == 'normalized':
            samples = samples_normalized * DATA_STD + DATA_MEAN
        else:
            samples = samples_normalized
        
        pairwise_dists = pdist(samples)
        print(f"mean={samples.mean():.2f}, std={samples.std():.2f}, dist={pairwise_dists.mean():.2f}")
        
        all_generated.append(samples)
        label_vec = np.zeros((n_samples_per_class, NUM_CLASSES))
        label_vec[:, class_idx] = 1
        all_labels.append(label_vec)
    
    generated_latents = np.vstack(all_generated)
    generated_labels = np.vstack(all_labels)
    
    # Save
    np.save(os.path.join(RESULTS_DIR, f'{model_type}_diffusion_samples.npy'), generated_latents)
    np.save(os.path.join(RESULTS_DIR, f'{model_type}_diffusion_labels.npy'), generated_labels)
    
    print(f"\n✓ Saved: {len(generated_latents)} samples")
    return generated_latents, generated_labels

# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_structure(model_type='normalized'):
    """Comprehensive structure analysis"""
    
    print(f"\n{'='*80}")
    print(f"STRUCTURE ANALYSIS: {model_type.upper()}")
    print("="*80)
    
    # Load data
    train_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy'))
    train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))
    label_columns = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
    train_labels = train_df[label_columns].values
    
    gen_file = os.path.join(RESULTS_DIR, f'{model_type}_diffusion_samples.npy')
    if not os.path.exists(gen_file):
        print(f"ERROR: {gen_file} not found. Run with --generate first.")
        return None
    
    gen_latents = np.load(gen_file)
    gen_labels = np.load(os.path.join(RESULTS_DIR, f'{model_type}_diffusion_labels.npy'))
    
    print(f"\nReal samples: {len(train_latent)}")
    print(f"Generated samples: {len(gen_latents)}")
    
    results = {}
    
    # Basic statistics
    print("\n1. Global Statistics")
    print("-"*80)
    real_dists = pdist(train_latent[np.random.choice(len(train_latent), 5000, replace=False)])
    gen_dists = pdist(gen_latents[np.random.choice(len(gen_latents), 5000, replace=False)])
    
    print(f"Real:      mean={train_latent.mean():.4f}, std={train_latent.std():.4f}, dist={real_dists.mean():.2f}")
    print(f"Generated: mean={gen_latents.mean():.4f}, std={gen_latents.std():.4f}, dist={gen_dists.mean():.2f}")
    print(f"Distance ratio: {gen_dists.mean()/real_dists.mean():.3f}")
    
    results['global_stats'] = {
        'real_mean': float(train_latent.mean()),
        'gen_mean': float(gen_latents.mean()),
        'real_std': float(train_latent.std()),
        'gen_std': float(gen_latents.std()),
        'dist_ratio': float(gen_dists.mean()/real_dists.mean())
    }
    
    # Advanced metrics
    print("\n2. Advanced Metrics (MMD, Coverage, Precision)")
    print("-"*80)
    n_sample = min(5000, len(train_latent), len(gen_latents))
    real_subset = train_latent[np.random.choice(len(train_latent), n_sample, replace=False)]
    gen_subset = gen_latents[np.random.choice(len(gen_latents), n_sample, replace=False)]
    
    mmd = compute_mmd(real_subset, gen_subset)
    coverage, precision = compute_coverage_precision(real_subset, gen_subset, k=5)
    
    print(f"MMD:       {mmd:.6f} {'(EXCELLENT)' if mmd < 0.01 else '(GOOD)' if mmd < 0.05 else '(NEEDS WORK)'}")
    print(f"Coverage:  {coverage:.4f} {'(EXCELLENT)' if coverage > 0.9 else '(GOOD)' if coverage > 0.7 else '(NEEDS WORK)'}")
    print(f"Precision: {precision:.4f} {'(EXCELLENT)' if precision > 0.9 else '(GOOD)' if precision > 0.7 else '(NEEDS WORK)'}")
    
    # Manifold distance
    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(real_subset)
    dists_gen_to_real, _ = nbrs.kneighbors(gen_subset)
    
    nbrs_baseline = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(real_subset)
    dists_real_to_real, _ = nbrs_baseline.kneighbors(real_subset)
    dists_real_to_real = dists_real_to_real[:, 1]
    
    manifold_ratio = dists_gen_to_real.mean() / dists_real_to_real.mean()
    print(f"Manifold ratio: {manifold_ratio:.3f} {'(ON MANIFOLD)' if 0.8 < manifold_ratio < 1.5 else '(OFF MANIFOLD)'}")
    
    results['advanced_metrics'] = {
        'mmd': float(mmd),
        'coverage': float(coverage),
        'precision': float(precision),
        'manifold_ratio': float(manifold_ratio)
    }
    
    # Per-class analysis
    print("\n3. Per-Class Structure")
    print("-"*80)
    print(f"{'Chemical':<10} {'Real Mean':<12} {'Gen Mean':<12} {'MMD':<12} {'KS p-val':<12}")
    print("-"*80)
    
    per_class = {}
    for class_idx, chemical in enumerate(label_columns):
        real_mask = train_labels[:, class_idx] == 1
        gen_mask = gen_labels[:, class_idx] == 1
        
        real_class = train_latent[real_mask]
        gen_class = gen_latents[gen_mask]
        
        n_class = min(1000, len(real_class), len(gen_class))
        real_class_sample = real_class[np.random.choice(len(real_class), n_class, replace=False)]
        gen_class_sample = gen_class[np.random.choice(len(gen_class), n_class, replace=False)]
        
        mmd_class = compute_mmd(real_class_sample, gen_class_sample)
        ks_stat, ks_pval = ks_2samp(real_class.flatten(), gen_class.flatten())
        
        print(f"{chemical:<10} {real_class.mean():>11.4f} {gen_class.mean():>11.4f} {mmd_class:>11.6f} {ks_pval:>11.4f}")
        
        per_class[chemical] = {
            'mmd': float(mmd_class),
            'ks_pval': float(ks_pval)
        }
    
    results['per_class'] = per_class
    
    # Visualizations
    print("\n4. Creating Visualizations")
    print("-"*80)
    create_visualizations(train_latent, train_labels, gen_latents, gen_labels, label_columns, model_type)
    
    # Save results
    with open(os.path.join(RESULTS_DIR, f'{model_type}_structure_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Metrics saved: {model_type}_structure_metrics.json")
    
    return results

def create_visualizations(real_latents, real_labels, gen_latents, gen_labels, label_columns, model_type):
    """Create comprehensive visualizations"""
    
    # Sample for speed
    n_vis = 2000
    np.random.seed(42)
    real_idx = np.random.choice(len(real_latents), min(n_vis, len(real_latents)), replace=False)
    gen_idx = np.random.choice(len(gen_latents), min(n_vis, len(gen_latents)), replace=False)
    
    real_vis = real_latents[real_idx]
    real_labels_vis = real_labels[real_idx]
    gen_vis = gen_latents[gen_idx]
    gen_labels_vis = gen_labels[gen_idx]
    
    # PCA
    pca = PCA(n_components=2)
    combined = np.vstack([real_vis, gen_vis])
    pca.fit(combined)
    real_pca = pca.transform(real_vis)
    gen_pca = pca.transform(gen_vis)
    
    colors = plt.cm.tab10(np.arange(len(label_columns)))
    
    # Figure 1: Overall comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for ax, (data, labels, title) in zip(axes, [
        (real_pca, real_labels_vis, 'Real'),
        (gen_pca, gen_labels_vis, f'{model_type.capitalize()} Diffusion'),
        (None, None, 'Overlay')
    ]):
        if title != 'Overlay':
            for class_idx, chemical in enumerate(label_columns):
                mask = labels[:, class_idx] == 1
                ax.scatter(data[mask, 0], data[mask, 1], c=[colors[class_idx]], 
                          label=chemical, alpha=0.4, s=10)
        else:
            for class_idx, chemical in enumerate(label_columns):
                mask_real = real_labels_vis[:, class_idx] == 1
                mask_gen = gen_labels_vis[:, class_idx] == 1
                ax.scatter(real_pca[mask_real, 0], real_pca[mask_real, 1],
                          c=[colors[class_idx]], alpha=0.3, s=10, marker='o')
                ax.scatter(gen_pca[mask_gen, 0], gen_pca[mask_gen, 1],
                          c=[colors[class_idx]], alpha=0.3, s=10, marker='x')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        if title == 'Real':
            ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f'{model_type}_structure_pca.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ {model_type}_structure_pca.png")
    plt.close()
    
    # Figure 2: Per-chemical
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for class_idx, chemical in enumerate(label_columns):
        ax = axes[class_idx]
        mask_real = real_labels_vis[:, class_idx] == 1
        mask_gen = gen_labels_vis[:, class_idx] == 1
        
        ax.scatter(real_pca[mask_real, 0], real_pca[mask_real, 1],
                  c='blue', alpha=0.4, s=20, marker='o', label='Real')
        ax.scatter(gen_pca[mask_gen, 0], gen_pca[mask_gen, 1],
                  c='red', alpha=0.4, s=20, marker='x', label='Generated')
        
        ax.set_title(chemical, fontsize=12, fontweight='bold')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_type.capitalize()} Diffusion: Per-Chemical Structure', 
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f'{model_type}_per_chemical.png'), dpi=150, bbox_inches='tight')
    print(f"  ✓ {model_type}_per_chemical.png")
    plt.close()

def compare_models():
    """Compare different diffusion model variants"""
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print("="*80)
    
    # Check what's available
    models = []
    for model_type in ['normalized', 'separated']:
        if os.path.exists(os.path.join(RESULTS_DIR, f'{model_type}_diffusion_samples.npy')):
            models.append(model_type)
    
    if len(models) < 2:
        print(f"\n⚠️  Need at least 2 models. Found: {models}")
        print("Generate samples with --model <type> --generate")
        return
    
    # Load real data
    train_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy'))
    train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))
    label_columns = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
    train_labels = train_df[label_columns].values
    
    # Load all model samples
    all_data = {'Real': (train_latent, train_labels)}
    for model_type in models:
        samples = np.load(os.path.join(RESULTS_DIR, f'{model_type}_diffusion_samples.npy'))
        labels = np.load(os.path.join(RESULTS_DIR, f'{model_type}_diffusion_labels.npy'))
        all_data[model_type.capitalize()] = (samples, labels)
    
    print(f"\nComparing: {list(all_data.keys())}")
    
    # Statistics table
    print("\n" + "-"*80)
    print(f"{'Model':<15} {'Mean':<12} {'Std':<12} {'Pairwise Dist':<15}")
    print("-"*80)
    for name, (latents, labels) in all_data.items():
        subset = latents[np.random.choice(len(latents), min(3000, len(latents)), replace=False)]
        dists = pdist(subset)
        print(f"{name:<15} {latents.mean():>11.4f} {latents.std():>11.4f} {dists.mean():>14.2f}")
    
    # Comparison visualization
    print("\nCreating comparison visualization...")
    n_vis = 2000
    vis_data = {}
    for name, (latents, labels) in all_data.items():
        idx = np.random.choice(len(latents), min(n_vis, len(latents)), replace=False)
        vis_data[name] = (latents[idx], labels[idx])
    
    # PCA on all
    all_vis = np.vstack([v[0] for v in vis_data.values()])
    pca = PCA(n_components=2)
    pca.fit(all_vis)
    pca_data = {name: pca.transform(latents) for name, (latents, labels) in vis_data.items()}
    
    # Plot
    n_models = len(vis_data)
    fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
    if n_models == 1:
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
        if name == 'Real':
            ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Diffusion Model Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: model_comparison.png")
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Diffusion Structure Analysis')
    parser.add_argument('--model', type=str, choices=['normalized', 'separated'], 
                       help='Model type to analyze')
    parser.add_argument('--generate', action='store_true', 
                       help='Generate samples from model')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze structure of generated samples')
    parser.add_argument('--compare', action='store_true',
                       help='Compare different model variants')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Samples per class to generate')
    parser.add_argument('--all', action='store_true',
                       help='Run everything: generate + analyze')
    
    args = parser.parse_args()
    
    print(f"Using device: {device}")
    
    if args.all and args.model:
        generate_samples(args.model, args.n_samples)
        analyze_structure(args.model)
    elif args.generate and args.model:
        generate_samples(args.model, args.n_samples)
    elif args.analyze and args.model:
        analyze_structure(args.model)
    elif args.compare:
        compare_models()
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python analyze_diffusion_structure.py --model normalized --all")
        print("  python analyze_diffusion_structure.py --compare")

if __name__ == "__main__":
    main()
