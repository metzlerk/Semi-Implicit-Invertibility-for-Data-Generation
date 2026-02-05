#!/usr/bin/env python3
"""
Generate detailed PCA plot showing all 8 chemicals with distinct colors and markers
for encoder (real), diffusion, and Gaussian sources.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
import json
import os
import pandas as pd

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.FlexibleNLayersEncoder import FlexibleNLayersEncoder
from models.FlexibleNLayersGenerator import FlexibleNLayersGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_smile_embeddings():
    """Load SMILE embeddings for all chemicals."""
    smile_path = "Data/name_smiles_embedding_file.csv"
    df = pd.read_csv(smile_path)
    
    chemicals = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
    embeddings = {}
    for chem in chemicals:
        row = df[df['name'] == chem]
        if len(row) > 0:
            emb = row.iloc[0, 2:].values.astype(np.float32)
            embeddings[chem] = torch.from_numpy(emb).to(device)
    
    return embeddings, chemicals

def load_ims_data():
    """Load IMS spectra data."""
    train_path = 'Data/train_data.feather'
    test_path = 'Data/test_data.feather'
    
    train_df = pd.read_feather(train_path)
    test_df = pd.read_feather(test_path)
    
    p_cols = [c for c in train_df.columns if c.startswith('p_')]
    n_cols = [c for c in train_df.columns if c.startswith('n_')]
    onehot_cols = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
    
    train_ims = np.concatenate([train_df[p_cols].values, train_df[n_cols].values], axis=1)
    test_ims = np.concatenate([test_df[p_cols].values, test_df[n_cols].values], axis=1)
    
    train_labels = train_df[onehot_cols].values.argmax(axis=1)
    test_labels = test_df[onehot_cols].values.argmax(axis=1)
    
    return train_ims, test_ims, train_labels, test_labels

def main():
    print("=" * 80)
    print("Detailed PCA Visualization: Real vs Diffusion vs Gaussian")
    print("=" * 80)
    
    # Load data
    print("\nLoading IMS data...")
    train_data, test_data, train_labels, test_labels = load_ims_data()
    
    # Load SMILE embeddings
    smile_embeddings, chemicals = load_smile_embeddings()
    print(f"Chemicals: {chemicals}")
    
    # Load encoder/generator
    print("\nLoading encoder and generator...")
    encoder = FlexibleNLayersEncoder(
        input_dim=train_data.shape[1],
        latent_dim=512,
        num_layers=9
    ).to(device)
    encoder.load_state_dict(torch.load('models/flexible_9layers_encoder.pt', map_location=device))
    encoder.eval()
    
    generator = FlexibleNLayersGenerator(
        latent_dim=512,
        output_dim=train_data.shape[1],
        num_layers=9
    ).to(device)
    generator.load_state_dict(torch.load('models/flexible_9layers_generator.pt', map_location=device))
    generator.eval()
    
    # Encode real data (sample 500 per class)
    print("\nEncoding real data to latent space...")
    samples_per_class = 500
    real_latents = []
    real_labels = []
    
    with torch.no_grad():
        for class_idx, chem in enumerate(chemicals):
            mask = test_labels == class_idx
            class_data = test_data[mask]
            
            # Sample randomly
            n_samples = min(samples_per_class, len(class_data))
            indices = np.random.choice(len(class_data), n_samples, replace=False)
            sampled = class_data[indices]
            
            # Encode to latent
            sampled_tensor = torch.FloatTensor(sampled).to(device)
            latent = encoder(sampled_tensor)
            
            real_latents.append(latent.cpu().numpy())
            real_labels.extend([class_idx] * n_samples)
    
    real_latents = np.vstack(real_latents)
    real_labels = np.array(real_labels)
    print(f"Real latents shape: {real_latents.shape}")
    
    # Load diffusion model
    print("\nLoading diffusion model...")
    from scripts.diffusion_ims_improved_separation import ClassConditionedDiffusion
    
    diffusion = ClassConditionedDiffusion(
        latent_dim=512,
        smile_dim=512,
        num_classes=8,
        timesteps=50,
        hidden_dim=512,
        num_layers=6
    ).to(device)
    
    checkpoint = torch.load('models/diffusion_separation_best.pt', map_location=device)
    diffusion.load_state_dict(checkpoint['model_state_dict'])
    diffusion.eval()
    
    # Generate diffusion samples (500 per class)
    print("\nGenerating diffusion samples...")
    diffusion_latents = []
    diffusion_labels = []
    
    with torch.no_grad():
        for class_idx, chem in enumerate(chemicals):
            smile_emb = smile_embeddings[chem].unsqueeze(0).repeat(samples_per_class, 1)
            class_labels = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
            
            # Start from random noise
            z_t = torch.randn(samples_per_class, 512, device=device)
            
            # Reverse diffusion
            for t in reversed(range(diffusion.timesteps)):
                t_batch = torch.full((samples_per_class,), t, dtype=torch.long, device=device)
                pred_z0, _ = diffusion(z_t, t_batch, smile_emb, class_labels)
                
                if t > 0:
                    alpha_t = diffusion.alphas[t]
                    alpha_prev = diffusion.alphas[t-1]
                    beta_t = diffusion.betas[t]
                    
                    z_t = (torch.sqrt(alpha_prev) * beta_t / (1 - alpha_t)) * pred_z0 + \
                          (torch.sqrt(1 - beta_t) * (1 - alpha_prev) / (1 - alpha_t)) * z_t
                    
                    if t > 1:
                        z_t += torch.sqrt(beta_t) * torch.randn_like(z_t)
                else:
                    z_t = pred_z0
            
            diffusion_latents.append(z_t.cpu().numpy())
            diffusion_labels.extend([class_idx] * samples_per_class)
    
    diffusion_latents = np.vstack(diffusion_latents)
    diffusion_labels = np.array(diffusion_labels)
    print(f"Diffusion latents shape: {diffusion_latents.shape}")
    
    # Generate Gaussian samples (500 per class)
    print("\nGenerating Gaussian samples...")
    gaussian_latents = []
    gaussian_labels = []
    
    for class_idx in range(len(chemicals)):
        # Sample from standard Gaussian
        samples = np.random.randn(samples_per_class, 512)
        gaussian_latents.append(samples)
        gaussian_labels.extend([class_idx] * samples_per_class)
    
    gaussian_latents = np.vstack(gaussian_latents)
    gaussian_labels = np.array(gaussian_labels)
    print(f"Gaussian latents shape: {gaussian_latents.shape}")
    
    # Combine all data for PCA
    print("\nComputing PCA on combined data...")
    all_latents = np.vstack([real_latents, diffusion_latents, gaussian_latents])
    
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_latents)
    
    # Split back
    n_real = len(real_latents)
    n_diffusion = len(diffusion_latents)
    
    real_pca = all_pca[:n_real]
    diffusion_pca = all_pca[n_real:n_real+n_diffusion]
    gaussian_pca = all_pca[n_real+n_diffusion:]
    
    print(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    
    # Create detailed plot
    print("\nCreating visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Define colors for each chemical
    colors = plt.cm.tab10(np.linspace(0, 1, 8))
    
    # Define markers for each source
    markers = {'real': 'o', 'diffusion': 's', 'gaussian': '^'}
    sizes = {'real': 60, 'diffusion': 60, 'gaussian': 60}
    alphas = {'real': 0.6, 'diffusion': 0.6, 'gaussian': 0.4}
    
    # Plot each chemical with each source
    for class_idx, chem in enumerate(chemicals):
        color = colors[class_idx]
        
        # Real data
        mask = real_labels == class_idx
        if mask.sum() > 0:
            ax.scatter(real_pca[mask, 0], real_pca[mask, 1],
                      c=[color], marker=markers['real'], s=sizes['real'],
                      alpha=alphas['real'], edgecolors='black', linewidth=0.5,
                      label=f'{chem} (Real)' if class_idx < 3 else None)
        
        # Diffusion samples
        mask = diffusion_labels == class_idx
        if mask.sum() > 0:
            ax.scatter(diffusion_pca[mask, 0], diffusion_pca[mask, 1],
                      c=[color], marker=markers['diffusion'], s=sizes['diffusion'],
                      alpha=alphas['diffusion'], edgecolors='black', linewidth=0.5,
                      label=f'{chem} (Diffusion)' if class_idx < 3 else None)
        
        # Gaussian samples
        mask = gaussian_labels == class_idx
        if mask.sum() > 0:
            ax.scatter(gaussian_pca[mask, 0], gaussian_pca[mask, 1],
                      c=[color], marker=markers['gaussian'], s=sizes['gaussian'],
                      alpha=alphas['gaussian'], edgecolors='black', linewidth=0.5,
                      label=f'{chem} (Gaussian)' if class_idx < 3 else None)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    
    # Chemical colors legend
    chemical_legend = [Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=colors[i], markersize=10, 
                             label=chem, markeredgecolor='black', markeredgewidth=0.5)
                      for i, chem in enumerate(chemicals)]
    
    # Source markers legend
    source_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='Real (Encoder)', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=10, label='Diffusion', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=10, label='Gaussian', markeredgecolor='black', markeredgewidth=0.5)
    ]
    
    # Add both legends
    first_legend = ax.legend(handles=chemical_legend, title='Chemicals',
                            loc='upper left', bbox_to_anchor=(1.02, 1), 
                            frameon=True, fontsize=11)
    ax.add_artist(first_legend)
    ax.legend(handles=source_legend, title='Source',
             loc='upper left', bbox_to_anchor=(1.02, 0.5),
             frameon=True, fontsize=11)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14, fontweight='bold')
    ax.set_title('Latent Space PCA: Real vs Diffusion vs Gaussian by Chemical', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f8f8')
    
    plt.tight_layout()
    
    # Save
    output_path = Path('images/detailed_pca_by_chemical.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also save metrics
    metrics = {
        'explained_variance_pc1': float(pca.explained_variance_ratio_[0]),
        'explained_variance_pc2': float(pca.explained_variance_ratio_[1]),
        'total_variance_explained': float(pca.explained_variance_ratio_[:2].sum()),
        'chemicals': chemicals,
        'samples_per_class_per_source': samples_per_class
    }
    
    metrics_path = Path('results/detailed_pca_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()
