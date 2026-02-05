import os
import ast
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
os.makedirs(IMAGES_DIR, exist_ok=True)

# Larger fonts for slide readability
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Load train data (for labels)
train_path = os.path.join(DATA_DIR, 'train_data.feather')
train_df = pd.read_feather(train_path)
onehot_cols = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
labels = train_df[onehot_cols].values.argmax(axis=1)
class_names = onehot_cols

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
        # fallback: zeros of 512
        emb = np.zeros(512, dtype=np.float32)
    emb_list.append(emb)

E = np.stack(emb_list)  # [N, 512]

# PCA on ChemNet embeddings (baseline chart)
pca = PCA(n_components=2)
E_pca = pca.fit_transform(E)

plt.figure(figsize=(12, 9))
colors = plt.cm.tab10(np.linspace(0,1,len(class_names)))
for i, cname in enumerate(class_names):
    mask = labels == i
    plt.scatter(E_pca[mask,0], E_pca[mask,1], s=20, alpha=0.7, color=colors[i], label=cname)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('ChemNet Embeddings: PCA by Chemical Class (Train Set)')
plt.legend(markerscale=1.5, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
out_path = os.path.join(IMAGES_DIR, 'chemnet_train_pca_by_class.png')
plt.savefig(out_path, dpi=150)
print(f'Saved {out_path}')


# -----------------------------------------------------------------------------
# Overlay chart: add 20 IMS samples/class encoded via trained encoder
# -----------------------------------------------------------------------------

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
        # Optional bias layer support (kept for compatibility with checkpoint)
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


def sample_ims_by_class(df, onehot_cols, per_class=20, expected_features=1676):
    """Return dict: class_index -> np.array of IMS rows (per_class) with expected feature count."""
    samples = {}
    # Candidate feature columns = numeric, non-onehot, non-label
    candidate_cols = [c for c in df.columns if c not in onehot_cols and c != 'Label']
    # Keep only numeric columns
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    # Enforce expected feature size if larger
    if len(numeric_cols) >= expected_features:
        feature_cols = numeric_cols[:expected_features]
    else:
        feature_cols = numeric_cols
    if len(feature_cols) != expected_features:
        print(f"Warning: found {len(feature_cols)} IMS feature columns, expected {expected_features}. Overlay may be skipped.")
    for i, cname in enumerate(onehot_cols):
        class_df = df[df[cname] == 1]
        if len(class_df) == 0:
            samples[i] = np.empty((0, expected_features), dtype=np.float32)
            continue
        take = min(per_class, len(class_df))
        picked = class_df.sample(n=take, random_state=42)
        X = picked[feature_cols].values.astype(np.float32)
        # If columns less than expected, skip
        if X.shape[1] != expected_features:
            samples[i] = np.empty((0, expected_features), dtype=np.float32)
        else:
            samples[i] = X
    return samples


def load_trained_encoder(models_dir, device, input_size=1676, output_size=512, n_layers=9):
    # Instantiate with bias layer to match checkpoint keys
    zero_bkg = torch.zeros(input_size, device=device)
    enc = FlexibleNLayersEncoder(input_size=input_size, output_size=output_size, n_layers=n_layers,
                                 init_style='bkg', bkg=zero_bkg, trainable=False).to(device)
    ckpt_path = os.path.join(models_dir, 'best_autoencoder.pth')
    if not os.path.exists(ckpt_path):
        print(f"Warning: encoder checkpoint not found: {ckpt_path}. Skipping overlay plot.")
        return None
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get('encoder_state_dict')
    if state is None:
        print(f"Warning: 'encoder_state_dict' missing in checkpoint. Skipping overlay plot.")
        return None
    enc.load_state_dict(state)
    enc.eval()
    return enc


# Prepare overlay data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = load_trained_encoder(os.path.join(ROOT_DIR, 'models'), device)

if encoder is not None:
    ims_samples = sample_ims_by_class(train_df, onehot_cols, per_class=20, expected_features=1676)

    # Encode IMS samples per class
    encoded_per_class = {}
    with torch.no_grad():
        for i in range(len(class_names)):
            X = ims_samples.get(i)
            if X is None or X.size == 0:
                encoded_per_class[i] = np.empty((0, 512), dtype=np.float32)
                continue
            Xt = torch.from_numpy(X).to(device)
            Z = encoder(Xt, use_bias=False).cpu().numpy()
            encoded_per_class[i] = Z

    # Project both datasets onto same PCA axes (fit on ChemNet only)
    E_base = E  # ChemNet embeddings
    E_base_pca = pca.transform(E_base)

    # Build overlay arrays aligned with labels
    overlay_points = []
    overlay_labels = []
    for i in range(len(class_names)):
        Z = encoded_per_class[i]
        if Z.shape[0] == 0:
            continue
        Z_pca = pca.transform(Z)
        overlay_points.append(Z_pca)
        overlay_labels.extend([i] * Z_pca.shape[0])
    if len(overlay_points) > 0:
        overlay_points = np.vstack(overlay_points)
        overlay_labels = np.array(overlay_labels)

        # Plot combined figure
        plt.figure(figsize=(12, 9))
        colors = plt.cm.tab10(np.linspace(0,1,len(class_names)))
        # ChemNet baseline
        for i, cname in enumerate(class_names):
            mask = labels == i
            plt.scatter(E_base_pca[mask,0], E_base_pca[mask,1], s=24, alpha=0.6, color=colors[i], label=f"{cname} (ChemNet)")
        # Encoder-IMS overlays
        for i, cname in enumerate(class_names):
            mask = overlay_labels == i
            if np.any(mask):
                plt.scatter(overlay_points[mask,0], overlay_points[mask,1], s=40, alpha=0.9, color=colors[i], marker='x', label=f"{cname} (Encoder-IMS, 20)")

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title('ChemNet vs Encoder-IMS: PCA by Class (Train Set)')
        plt.legend(markerscale=1.5, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path2 = os.path.join(IMAGES_DIR, 'chemnet_vs_encoder_pca_by_class.png')
        plt.savefig(out_path2, dpi=150)
        print(f'Saved {out_path2}')
