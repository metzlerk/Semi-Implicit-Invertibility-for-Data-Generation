#!/usr/bin/env python3
"""
Plot synthetic spectra (from Cate's model) passed through encoder against ChemNet embeddings.
Uses the same PCA transformation from ChemNet baseline.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(IMAGES_DIR, exist_ok=True)

# Larger fonts for slide readability
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Load train data (for labels and to build PCA baseline)
train_path = os.path.join(DATA_DIR, 'train_data.feather')
train_df = pd.read_feather(train_path)
onehot_cols = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
labels = train_df[onehot_cols].values.argmax(axis=1)
class_names = onehot_cols

print("Building PCA transformation from ChemNet embeddings...")

# Load ChemNet embeddings mapping
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

# Build embedding dict
import ast
embedding_dict = {}
for _, row in smile_df.iterrows():
    if pd.notna(row.get('embedding')):
        try:
            embedding = np.array(ast.literal_eval(row['embedding']), dtype=np.float32)
            embedding_dict[row['Name']] = embedding
        except Exception:
            pass

# Assign ChemNet embedding per sample
emb_list = []
for lab_idx in labels:
    lab = class_names[lab_idx]
    full_name = label_mapping.get(lab)
    emb = embedding_dict.get(full_name)
    if emb is None:
        emb = np.zeros(512, dtype=np.float32)
    emb_list.append(emb)

E = np.stack(emb_list)  # [N, 512]

# Fit PCA on ChemNet embeddings (this will be the reference)
pca = PCA(n_components=2)
E_pca = pca.fit_transform(E)

print(f"PCA fitted on {E.shape[0]} ChemNet embeddings")
print(f"  PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}%")

# =============================================================================
# Load synthetic spectra and encode
# =============================================================================

print("\nLoading synthetic spectra from Cate's model...")
synthetic_path = '/scratch/cmdunham/test/synthetic_data/spectrum/universal_generator/nine_layer_trainable_bkg_bias_test_spectra_from_ChemNet.feather'

if not os.path.exists(synthetic_path):
    print(f"ERROR: Synthetic spectra file not found: {synthetic_path}")
    exit(1)

synthetic_df = pd.read_feather(synthetic_path)
print(f"Loaded {synthetic_df.shape[0]} synthetic spectra")

# Extract labels and features
synthetic_labels_str = synthetic_df['Label'].values
synthetic_features = synthetic_df.drop(columns=['Label', 'index']).values.astype(np.float32)

print(f"Synthetic features shape: {synthetic_features.shape}")

# Map string labels to indices
synthetic_labels = []
for label_str in synthetic_labels_str:
    if label_str in class_names:
        synthetic_labels.append(class_names.index(label_str))
    else:
        synthetic_labels.append(-1)  # Unknown label
synthetic_labels = np.array(synthetic_labels)

# Count valid labels
n_valid = np.sum(synthetic_labels >= 0)
print(f"Valid labels: {n_valid} / {len(synthetic_labels)}")

# =============================================================================
# Encoder + apply PCA
# =============================================================================

class BiasOnlyLayer(nn.Module):
    def __init__(self, bias_init=None, trainable=False):
        super().__init__()
        if bias_init is None:
            raise ValueError("BiasOnlyLayer requires bias_init")
        self.bias = nn.Parameter(bias_init.clone().detach(), requires_grad=trainable)

    def forward(self, x):
        return x + self.bias


class FlexibleNLayersEncoder(nn.Module):
    def __init__(self, input_size=1676, output_size=512, n_layers=9, init_style=None, bkg=None, trainable=False):
        super().__init__()
        if init_style == 'random':
            self.bias_layer = BiasOnlyLayer(bias_init=torch.randn(input_size), trainable=trainable)
        elif init_style == 'bkg':
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
        return self.encoder(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Load encoder
zero_bkg = torch.zeros(1676, device=device)
encoder = FlexibleNLayersEncoder(input_size=1676, output_size=512, n_layers=9,
                                 init_style='bkg', bkg=zero_bkg, trainable=False).to(device)

ckpt_path = os.path.join(MODELS_DIR, 'best_autoencoder.pth')
if not os.path.exists(ckpt_path):
    print(f"ERROR: Encoder checkpoint not found: {ckpt_path}")
    exit(1)

checkpoint = torch.load(ckpt_path, map_location=device)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
encoder.eval()
print(f"Loaded encoder from {ckpt_path}")

# Encode synthetic spectra
print("\nEncoding synthetic spectra...")
synthetic_latent_list = []
batch_size = 256
with torch.no_grad():
    for i in range(0, len(synthetic_features), batch_size):
        batch = torch.from_numpy(synthetic_features[i:i+batch_size]).to(device)
        latent = encoder(batch, use_bias=False).cpu().numpy()
        synthetic_latent_list.append(latent)

synthetic_latent = np.vstack(synthetic_latent_list)
print(f"Encoded shape: {synthetic_latent.shape}")

# Transform using ChemNet's PCA axes
synthetic_pca = pca.transform(synthetic_latent)
print(f"Transformed PCA shape: {synthetic_pca.shape}")

# =============================================================================
# Create comparison plot
# =============================================================================

print("\nCreating comparison plot...")

fig, ax = plt.subplots(figsize=(14, 10))

colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

# Plot ChemNet baseline (semi-transparent)
for i, cname in enumerate(class_names):
    mask = labels == i
    ax.scatter(E_pca[mask, 0], E_pca[mask, 1], s=20, alpha=0.4, color=colors[i],
               label=f"{cname} (ChemNet)", zorder=1)

# Plot synthetic spectra
for i, cname in enumerate(class_names):
    mask = synthetic_labels == i
    if np.any(mask):
        ax.scatter(synthetic_pca[mask, 0], synthetic_pca[mask, 1], s=30, alpha=0.8, color=colors[i],
                   marker='s', edgecolors='black', linewidth=0.5,
                   label=f"{cname} (Synthetic)", zorder=2)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('ChemNet vs Synthetic Spectra: PCA by Chemical Class')
ax.legend(markerscale=1.5, ncol=2, loc='best')
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = os.path.join(IMAGES_DIR, 'chemnet_vs_synthetic_pca_by_class.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nSaved {out_path}')

plt.close()

print("\nDone!")
