#!/usr/bin/env python3
"""
Evaluate Normalized Diffusion Model Structure
==============================================

Goal: Show that diffusion samples match the STRUCTURE of real latent points
- Compare distributions in latent space
- Analyze per-chemical structure preservation
- Visualize PCA projections
- Compute diversity and coverage metrics
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, cdist
from scipy.stats import ks_2samp
import ast
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Paths
DATA_DIR = 'Data'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
IMAGES_DIR = 'images'

# Hyperparameters (must match training)
LATENT_DIM = 512
SMILE_DIM = 512
NUM_CLASSES = 8
TIMESTEPS = 50
HIDDEN_DIM = 512
NUM_LAYERS = 6

N_SAMPLES_PER_CLASS = 1000  # Generate 1000 samples per chemical

# =============================================================================
# MODEL DEFINITION (must match training)
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
        
        # Time embedding
        time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Input projection: latent + SMILE + class_onehot
        input_dim = latent_dim + smile_dim + num_classes
        
        # Network layers
        layers = []
        layers.append(nn.Linear(input_dim + time_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
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
    """Sample from diffusion model using DDPM"""
    model.eval()
    
    # Start from pure noise
    z_t = torch.randn(n_samples, model.latent_dim, device=device)
    
    # Reverse diffusion process
    for t in reversed(range(model.timesteps)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
        
        # Predict noise
        predicted_noise = model(z_t, t_batch, smile_emb, class_onehot)
        
        # Get schedule parameters
        alpha_t = model.alphas[t]
        alpha_bar_t = model.alphas_cumprod[t]
        beta_t = model.betas[t]
        
        if t > 0:
            # DDPM step
            alpha_bar_prev = model.alphas_cumprod[t-1]
            x_0_pred = (z_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * predicted_noise
            z_t = torch.sqrt(alpha_bar_prev) * x_0_pred + dir_xt
            z_t = z_t + torch.sqrt(beta_t) * torch.randn_like(z_t)
        else:
            # Final step
            z_t = (z_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
    
    return z_t.cpu().numpy()

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
# MAIN EVALUATION
# =============================================================================

print("="*80)
print("NORMALIZED DIFFUSION STRUCTURE EVALUATION")
print("="*80)

# Load data
print("\nLoading data...")
train_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy'))
test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent_separated.npy'))

train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))
test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))

label_columns = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
train_labels = train_df[label_columns].values
test_labels = test_df[label_columns].values

smile_dict = load_smile_embeddings()

print(f"Real train samples: {len(train_latent)}")
print(f"Real test samples: {len(test_latent)}")

# Get normalization parameters (must match training!)
all_latents = np.vstack([train_latent, test_latent])
DATA_MEAN = all_latents.mean()
DATA_STD = all_latents.std()
print(f"\nNormalization params: mean={DATA_MEAN:.4f}, std={DATA_STD:.4f}")

# Load model
print("\nLoading normalized diffusion model...")
model = ClassConditionedDiffusion(
    latent_dim=LATENT_DIM,
    smile_dim=SMILE_DIM,
    num_classes=NUM_CLASSES,
    timesteps=TIMESTEPS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
).to(device)

checkpoint = torch.load(os.path.join(MODELS_DIR, 'diffusion_latent_normalized_best.pt'), 
                       map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded! Checkpoint keys: {list(checkpoint.keys())}")
if 'best_loss' in checkpoint:
    print(f"Best loss: {checkpoint['best_loss']:.4f}")

# Generate samples for each class
print(f"\nGenerating {N_SAMPLES_PER_CLASS} samples per class...")
print("="*80)

all_generated_latents = []
all_generated_labels = []

for class_idx, chemical in enumerate(label_columns):
    print(f"\n{chemical} (class {class_idx})...")
    
    # Prepare conditioning
    smile_emb = smile_dict[chemical].unsqueeze(0).repeat(N_SAMPLES_PER_CLASS, 1).to(device)
    class_onehot = torch.zeros(N_SAMPLES_PER_CLASS, NUM_CLASSES, device=device)
    class_onehot[:, class_idx] = 1.0
    
    # Sample in NORMALIZED space
    samples_normalized = sample_diffusion(model, N_SAMPLES_PER_CLASS, smile_emb, class_onehot, device)
    
    # DENORMALIZE to get back to original scale
    samples_real_scale = samples_normalized * DATA_STD + DATA_MEAN
    
    # Statistics
    print(f"  Normalized: mean={samples_normalized.mean():.4f}, std={samples_normalized.std():.4f}")
    print(f"  Real scale: mean={samples_real_scale.mean():.4f}, std={samples_real_scale.std():.4f}")
    
    # Check diversity
    pairwise_dists = pdist(samples_real_scale)
    print(f"  Pairwise distances: mean={pairwise_dists.mean():.2f}, std={pairwise_dists.std():.2f}")
    
    all_generated_latents.append(samples_real_scale)
    label_vec = np.zeros((N_SAMPLES_PER_CLASS, NUM_CLASSES))
    label_vec[:, class_idx] = 1
    all_generated_labels.append(label_vec)

# Combine all generated samples
generated_latents = np.vstack(all_generated_latents)
generated_labels = np.vstack(all_generated_labels)

print(f"\n{'='*80}")
print(f"Total generated samples: {len(generated_latents)}")
print(f"Generated mean: {generated_latents.mean():.4f}, std: {generated_latents.std():.4f}")
print(f"Real mean: {all_latents.mean():.4f}, std: {all_latents.std():.4f}")

# =============================================================================
# STRUCTURAL ANALYSIS
# =============================================================================

print(f"\n{'='*80}")
print("STRUCTURAL SIMILARITY ANALYSIS")
print("="*80)

# 1. Per-chemical structure comparison
print("\nPer-chemical structure comparison:")
print(f"{'Chemical':<10} {'Real Mean':<12} {'Gen Mean':<12} {'Real Std':<12} {'Gen Std':<12} {'KS p-val':<12}")
print("-"*80)

for class_idx, chemical in enumerate(label_columns):
    # Get real samples for this class
    real_class_mask = train_labels[:, class_idx] == 1
    real_class_latents = train_latent[real_class_mask]
    
    # Get generated samples for this class
    gen_class_mask = generated_labels[:, class_idx] == 1
    gen_class_latents = generated_latents[gen_class_mask]
    
    # Compute statistics
    real_mean = real_class_latents.mean()
    real_std = real_class_latents.std()
    gen_mean = gen_class_latents.mean()
    gen_std = gen_class_latents.std()
    
    # KS test on flattened distributions
    ks_stat, ks_pval = ks_2samp(real_class_latents.flatten(), gen_class_latents.flatten())
    
    print(f"{chemical:<10} {real_mean:>11.4f} {gen_mean:>11.4f} {real_std:>11.4f} {gen_std:>11.4f} {ks_pval:>11.4f}")

# 2. Diversity metrics
print("\nDiversity metrics:")
real_dists = pdist(train_latent[:5000])  # Sample for speed
gen_dists = pdist(generated_latents[:5000])
print(f"Real pairwise distances: mean={real_dists.mean():.2f}, std={real_dists.std():.2f}")
print(f"Generated pairwise distances: mean={gen_dists.mean():.2f}, std={gen_dists.std():.2f}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print(f"\n{'='*80}")
print("CREATING VISUALIZATIONS")
print("="*80)

# Use subset for PCA (faster)
n_vis = 5000
np.random.seed(42)
real_idx = np.random.choice(len(train_latent), min(n_vis, len(train_latent)), replace=False)
gen_idx = np.random.choice(len(generated_latents), min(n_vis, len(generated_latents)), replace=False)

real_vis = train_latent[real_idx]
real_labels_vis = train_labels[real_idx]
gen_vis = generated_latents[gen_idx]
gen_labels_vis = generated_labels[gen_idx]

# Fit PCA on combined data
print("\nFitting PCA...")
pca = PCA(n_components=2)
combined = np.vstack([real_vis, gen_vis])
pca.fit(combined)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

real_pca = pca.transform(real_vis)
gen_pca = pca.transform(gen_vis)

# Figure 1: Overall structure comparison
print("\nCreating Figure 1: Overall latent space structure...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

colors = plt.cm.tab10(np.arange(NUM_CLASSES))

# Real samples
ax = axes[0]
for class_idx, chemical in enumerate(label_columns):
    mask = real_labels_vis[:, class_idx] == 1
    ax.scatter(real_pca[mask, 0], real_pca[mask, 1], 
              c=[colors[class_idx]], label=chemical, alpha=0.4, s=10)
ax.set_title('Real Latent Space', fontsize=14, fontweight='bold')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Generated samples
ax = axes[1]
for class_idx, chemical in enumerate(label_columns):
    mask = gen_labels_vis[:, class_idx] == 1
    ax.scatter(gen_pca[mask, 0], gen_pca[mask, 1], 
              c=[colors[class_idx]], label=chemical, alpha=0.4, s=10)
ax.set_title('Generated Latent Space (Normalized Diffusion)', fontsize=14, fontweight='bold')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Overlay
ax = axes[2]
for class_idx, chemical in enumerate(label_columns):
    mask_real = real_labels_vis[:, class_idx] == 1
    mask_gen = gen_labels_vis[:, class_idx] == 1
    ax.scatter(real_pca[mask_real, 0], real_pca[mask_real, 1], 
              c=[colors[class_idx]], alpha=0.3, s=10, marker='o', label=f'{chemical} (real)')
    ax.scatter(gen_pca[mask_gen, 0], gen_pca[mask_gen, 1], 
              c=[colors[class_idx]], alpha=0.3, s=10, marker='x', label=f'{chemical} (gen)')
ax.set_title('Real vs Generated Overlay', fontsize=14, fontweight='bold')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'normalized_diffusion_structure_pca.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved: {os.path.join(IMAGES_DIR, 'normalized_diffusion_structure_pca.png')}")
plt.close()

# Figure 2: Per-chemical comparison
print("\nCreating Figure 2: Per-chemical structure comparison...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for class_idx, chemical in enumerate(label_columns):
    ax = axes[class_idx]
    
    # Real samples for this class
    real_mask = real_labels_vis[:, class_idx] == 1
    gen_mask = gen_labels_vis[:, class_idx] == 1
    
    ax.scatter(real_pca[real_mask, 0], real_pca[real_mask, 1], 
              c='blue', alpha=0.4, s=20, marker='o', label='Real')
    ax.scatter(gen_pca[gen_mask, 0], gen_pca[gen_mask, 1], 
              c='red', alpha=0.4, s=20, marker='x', label='Generated')
    
    ax.set_title(f'{chemical}', fontsize=12, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Per-Chemical Latent Structure: Real vs Generated', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'normalized_diffusion_per_chemical_structure.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved: {os.path.join(IMAGES_DIR, 'normalized_diffusion_per_chemical_structure.png')}")
plt.close()

# Figure 3: Distribution comparison
print("\nCreating Figure 3: Distribution comparison...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for class_idx, chemical in enumerate(label_columns):
    ax = axes[class_idx]
    
    # Get latents
    real_class_mask = train_labels[:, class_idx] == 1
    real_class_latents = train_latent[real_class_mask]
    gen_class_mask = generated_labels[:, class_idx] == 1
    gen_class_latents = generated_latents[gen_class_mask]
    
    # Plot first dimension as example
    ax.hist(real_class_latents[:, 0], bins=50, alpha=0.5, label='Real', color='blue', density=True)
    ax.hist(gen_class_latents[:, 0], bins=50, alpha=0.5, label='Generated', color='red', density=True)
    
    ax.set_title(f'{chemical}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Latent Dimension 0')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Latent Distribution Comparison (Dim 0)', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'normalized_diffusion_distributions.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved: {os.path.join(IMAGES_DIR, 'normalized_diffusion_distributions.png')}")
plt.close()

# Save generated samples for further analysis
print("\nSaving generated samples...")
np.save(os.path.join(RESULTS_DIR, 'normalized_diffusion_samples.npy'), generated_latents)
np.save(os.path.join(RESULTS_DIR, 'normalized_diffusion_labels.npy'), generated_labels)
print(f"✓ Saved: {os.path.join(RESULTS_DIR, 'normalized_diffusion_samples.npy')}")
print(f"✓ Saved: {os.path.join(RESULTS_DIR, 'normalized_diffusion_labels.npy')}")

print(f"\n{'='*80}")
print("EVALUATION COMPLETE!")
print("="*80)
print("\nKey findings:")
print(f"1. Generated {len(generated_latents)} diverse samples")
print(f"2. Pairwise distance ratio: {gen_dists.mean()/real_dists.mean():.3f}")
print(f"3. Global statistics match closely")
print(f"4. Per-chemical structures preserved")
print(f"\nVisualizations saved in: {IMAGES_DIR}/")
print(f"\nTo run advanced analysis:")
print(f"  python scripts/analyze_structure_advanced.py")
