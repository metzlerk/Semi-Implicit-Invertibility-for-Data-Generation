"""
Compare PCA visualizations of:
1. Real embedded spectra (training data) in latent space
2. Gaussian samples in latent space
3. Diffusion-generated samples in latent space

All colored by chemical class to assess separation quality.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns
import ast
from collections import OrderedDict
import re

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# ENCODER/GENERATOR MODELS
# =============================================================================

class BiasOnlyLayer(nn.Module):
    def __init__(self, bias_init=None, trainable=False):
        super().__init__()
        bias = bias_init.clone().detach()
        if trainable:
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer("bias", bias)
    
    def forward(self, x):
        return x + self.bias


class FlexibleNLayersEncoder(nn.Module):
    def __init__(self, input_size=1676, output_size=512, n_layers=9, init_style=None, bkg=None, trainable=False):
        super().__init__()
        if init_style == 'random':
            self.bias_layer = BiasOnlyLayer(bias_init=torch.randn(input_size), trainable=trainable)
        if init_style == 'bkg':
            if bkg is None:
                raise ValueError("Must provide `bkg` tensor when init_style='bkg'")
            self.bias_layer = BiasOnlyLayer(bias_init=-bkg, trainable=trainable)
        
        layers = OrderedDict()
        if n_layers > 1:
            size_reduction_per_layer = (input_size - output_size) / n_layers
            for i in range(n_layers - 1):
                layer_input_size = input_size - int(size_reduction_per_layer) * i
                layer_output_size = input_size - int(size_reduction_per_layer) * (i + 1)
                layers[f'fc{i}'] = nn.Linear(layer_input_size, layer_output_size)
                layers[f'relu{i}'] = nn.LeakyReLU(inplace=True)
            layers['final'] = nn.Linear(layer_output_size, output_size)
        else:
            layers['final'] = nn.Linear(input_size, output_size)
        
        self.encoder = nn.Sequential(layers)
    
    def forward(self, x, use_bias=False):
        if use_bias and hasattr(self, 'bias_layer'):
            x = self.bias_layer(x)
        x = self.encoder(x)
        return x


class FlexibleNLayersGenerator(nn.Module):
    def __init__(self, input_size=512, output_size=1676, n_layers=9, init_style=None, bkg=None, trainable=False):
        super().__init__()
        if init_style == 'random':
            self.bias_layer = BiasOnlyLayer(bias_init=torch.randn(output_size), trainable=trainable)
        if init_style == 'bkg':
            if bkg is None:
                raise ValueError("Must provide `bkg` tensor when init_style='bkg'")
            self.bias_layer = BiasOnlyLayer(bias_init=bkg, trainable=trainable)
        
        layers = []
        layer_sizes = np.linspace(input_size, output_size, n_layers + 1, dtype=int)
        for i in range(n_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < n_layers - 1:
                layers.append(nn.LeakyReLU(inplace=True))
        self.generator = nn.Sequential(*layers)
    
    def forward(self, x, use_bias=False):
        x = self.generator(x)
        if use_bias and hasattr(self, 'bias_layer'):
            x = self.bias_layer(x)
        return x


# =============================================================================
# DIFFUSION MODEL (SAME AS TRAINING)
# =============================================================================

class ClassConditionedDiffusion(nn.Module):
    def __init__(self, latent_dim=512, embedding_dim=512, num_classes=8, 
                 hidden_dim=512, num_layers=6, timesteps=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.timesteps = timesteps
        
        time_embed_dim = embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        self.smile_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.class_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        input_dim = latent_dim + time_embed_dim + hidden_dim + hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.class_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, noisy_latent, t, smile_embedding, class_onehot):
        batch_size = noisy_latent.shape[0]
        
        t_normalized = t.float().view(-1, 1) / self.timesteps
        t_emb = self.time_mlp(t_normalized)
        
        smile_emb = self.smile_proj(smile_embedding)
        class_emb = self.class_proj(class_onehot)
        
        x = torch.cat([noisy_latent, t_emb, smile_emb, class_emb], dim=1)
        h = self.input_proj(x)
        
        for layer, norm in zip(self.layers, self.layer_norms):
            h = h + layer(norm(h))
        
        pred_latent = self.latent_head(h)
        class_logits = self.class_head(h)
        
        return pred_latent, class_logits


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and process data"""
    print("Loading data...")
    
    train_path = os.path.join(DATA_DIR, 'train_data.feather')
    test_path = os.path.join(DATA_DIR, 'test_data.feather')
    
    train_df = pd.read_feather(train_path)
    test_df = pd.read_feather(test_path)
    
    p_cols = [c for c in train_df.columns if c.startswith('p_')]
    n_cols = [c for c in train_df.columns if c.startswith('n_')]
    onehot_cols = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
    
    train_ims = np.concatenate([train_df[p_cols].values, train_df[n_cols].values], axis=1)
    test_ims = np.concatenate([test_df[p_cols].values, test_df[n_cols].values], axis=1)
    
    # Load SMILE embeddings
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
    
    smile_embedding_dim = len(next(iter(label_embeddings.values())))
    
    train_onehot = train_df[onehot_cols].values
    test_onehot = test_df[onehot_cols].values
    
    train_labels = train_onehot.argmax(axis=1)
    test_labels = test_onehot.argmax(axis=1)
    
    train_embeddings = np.zeros((len(train_df), smile_embedding_dim), dtype=np.float32)
    test_embeddings = np.zeros((len(test_df), smile_embedding_dim), dtype=np.float32)
    
    for idx, label in enumerate(onehot_cols):
        if label in label_embeddings:
            train_mask = train_labels == idx
            test_mask = test_labels == idx
            train_embeddings[train_mask] = label_embeddings[label]
            test_embeddings[test_mask] = label_embeddings[label]
    
    # Normalize
    ims_mean = train_ims.mean(axis=0)
    ims_std = train_ims.std(axis=0) + 1e-8
    
    train_ims_norm = (train_ims - ims_mean) / ims_std
    test_ims_norm = (test_ims - ims_mean) / ims_std
    
    train_bkg = train_ims_norm.mean(axis=0)
    
    return {
        'train_ims': train_ims_norm,
        'train_embeddings': train_embeddings,
        'train_labels': train_labels,
        'train_onehot': train_onehot,
        'test_ims': test_ims_norm,
        'test_embeddings': test_embeddings,
        'test_labels': test_labels,
        'test_onehot': test_onehot,
        'ims_mean': ims_mean,
        'ims_std': ims_std,
        'train_bkg': train_bkg,
        'class_names': onehot_cols,
        'num_classes': len(onehot_cols),
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    data = load_data()
    batch_size = 256
    latent_dim = 512
    timesteps = 50
    hidden_dim = 512
    num_layers = 6
    
    # Load models
    print("\nLoading encoder and generator...")
    encoder = FlexibleNLayersEncoder(
        input_size=data['train_ims'].shape[1],
        output_size=latent_dim,
        n_layers=9,
        init_style='bkg',
        bkg=torch.FloatTensor(data['train_bkg']),
        trainable=True
    ).to(device)
    
    generator = FlexibleNLayersGenerator(
        input_size=latent_dim,
        output_size=data['train_ims'].shape[1],
        n_layers=9,
        init_style='bkg',
        bkg=torch.FloatTensor(data['train_bkg']),
        trainable=True
    ).to(device)
    
    # Encode data
    print("Encoding data to latent space...")
    encoder.eval()
    with torch.no_grad():
        train_ims_tensor = torch.FloatTensor(data['train_ims']).to(device)
        
        train_latent_list = []
        for i in range(0, len(train_ims_tensor), batch_size):
            batch = train_ims_tensor[i:i+batch_size]
            latent = encoder(batch, use_bias=True)
            train_latent_list.append(latent.cpu())
        train_latent = torch.cat(train_latent_list, dim=0).numpy()
    
    # Load diffusion model
    print("Loading trained diffusion model...")
    diffusion_model = ClassConditionedDiffusion(
        latent_dim=latent_dim,
        embedding_dim=data['train_embeddings'].shape[1],
        num_classes=data['num_classes'],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        timesteps=timesteps
    ).to(device)
    
    model_path = os.path.join(MODELS_DIR, 'diffusion_separation_best.pt')
    if os.path.exists(model_path):
        diffusion_model.load_state_dict(torch.load(model_path))
        print(f"Loaded diffusion model from {model_path}")
    else:
        print(f"Warning: Diffusion model not found at {model_path}")
    
    diffusion_model.eval()
    
    # Generate Gaussian and diffusion samples
    print("\nGenerating samples...")
    n_samples = 2000
    gaussian_latent = np.random.randn(n_samples, latent_dim).astype(np.float32) * train_latent.std(axis=0) + train_latent.mean(axis=0)
    
    # Generate diffusion samples (simple sampling without full DDIM)
    diffusion_latent = torch.randn(n_samples, latent_dim, device=device)
    
    # Sample labels uniformly
    sample_labels = np.random.choice(data['num_classes'], n_samples)
    sample_embeddings = np.zeros((n_samples, data['train_embeddings'].shape[1]), dtype=np.float32)
    for c in range(data['num_classes']):
        mask = sample_labels == c
        sample_embeddings[mask] = data['train_embeddings'][data['train_labels'] == c].mean(axis=0)
    
    sample_onehot = np.eye(data['num_classes'])[sample_labels]
    
    with torch.no_grad():
        embeddings_tensor = torch.FloatTensor(sample_embeddings).to(device)
        onehot_tensor = torch.FloatTensor(sample_onehot).to(device)
        t_tensor = torch.zeros(n_samples, dtype=torch.long, device=device)
        
        pred_latent, _ = diffusion_model(diffusion_latent, t_tensor, embeddings_tensor, onehot_tensor)
        diffusion_latent = pred_latent.cpu().numpy()
    
    # PCA
    print("Computing PCA...")
    pca = PCA(n_components=2)
    
    # Real data PCA
    real_pca = pca.fit_transform(train_latent)
    
    # Gaussian PCA (using same fitted PCA)
    gaussian_pca = pca.transform(gaussian_latent)
    
    # Diffusion PCA
    diffusion_pca = pca.transform(diffusion_latent)
    
    # Compute silhouette scores
    print("\nComputing separation metrics...")
    real_silhouette = silhouette_score(real_pca, data['train_labels'][:len(real_pca)])
    gaussian_silhouette = silhouette_score(gaussian_pca, sample_labels)
    diffusion_silhouette = silhouette_score(diffusion_pca, sample_labels)
    
    print(f"  Real data silhouette score: {real_silhouette:.4f}")
    print(f"  Gaussian samples silhouette score: {gaussian_silhouette:.4f}")
    print(f"  Diffusion samples silhouette score: {diffusion_silhouette:.4f}")
    
    # Plot
    print("Creating visualizations...")
    colors = plt.cm.tab10(np.linspace(0, 1, data['num_classes']))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Real data
    ax = axes[0]
    for c in range(data['num_classes']):
        mask = data['train_labels'] == c
        ax.scatter(real_pca[mask, 0], real_pca[mask, 1], c=[colors[c]], label=data['class_names'][c], alpha=0.6, s=30)
    ax.set_title(f'Real Data (Silhouette: {real_silhouette:.3f})', fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Gaussian samples
    ax = axes[1]
    for c in range(data['num_classes']):
        mask = sample_labels == c
        ax.scatter(gaussian_pca[mask, 0], gaussian_pca[mask, 1], c=[colors[c]], label=data['class_names'][c], alpha=0.6, s=30)
    ax.set_title(f'Gaussian Samples (Silhouette: {gaussian_silhouette:.3f})', fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Diffusion samples
    ax = axes[2]
    for c in range(data['num_classes']):
        mask = sample_labels == c
        ax.scatter(diffusion_pca[mask, 0], diffusion_pca[mask, 1], c=[colors[c]], label=data['class_names'][c], alpha=0.6, s=30)
    ax.set_title(f'Diffusion Samples (Silhouette: {diffusion_silhouette:.3f})', fontsize=14, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'pca_separation_comparison_improved.png'), dpi=150, bbox_inches='tight')
    print(f"Saved to {IMAGES_DIR}/pca_separation_comparison_improved.png")
    plt.close()
    
    # Save metrics
    metrics = {
        "real_silhouette": float(real_silhouette),
        "gaussian_silhouette": float(gaussian_silhouette),
        "diffusion_silhouette": float(diffusion_silhouette),
        "improvement_over_gaussian": float(diffusion_silhouette - gaussian_silhouette),
        "explained_variance_pc1": float(pca.explained_variance_ratio_[0]),
        "explained_variance_pc2": float(pca.explained_variance_ratio_[1]),
    }
    
    import json
    with open(os.path.join(RESULTS_DIR, 'separation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {RESULTS_DIR}/separation_metrics.json")
    print(f"Improvement over Gaussian: {metrics['improvement_over_gaussian']:.4f}")
