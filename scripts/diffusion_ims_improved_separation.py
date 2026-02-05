"""
Improved Diffusion Model with Better Latent Space Separation
==============================================================

This script trains a unified diffusion model conditioned on both:
1. SMILE embeddings (512-dim)
2. Chemical class (one-hot, 8 classes)

The goal is to IMPROVE SEPARATION of different chemicals in the latent space
during the diffusion process.

Key improvements over per-class training:
1. Unified model learns to distinguish between classes
2. Explicit class conditioning helps maintain separation
3. Added separation loss to encourage inter-class distance
4. Monitor silhouette score during training

Architecture:
- Input: [noisy_latent, timestep, smile_embedding, class_onehot]
- Output: denoised_latent, predicted_class_logits (classification auxiliary)

Dataset: IMS Spectra (1676-dim) -> Latent space (512-dim) -> Conditioned diffusion
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.decomposition import PCA
import scipy.linalg
from tqdm import tqdm
import json
import time
from datetime import datetime
import math
import seaborn as sns
import ast
from collections import OrderedDict
import re

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

import wandb

# Initialize wandb
wandb.login(key="57680a36aa570ba8df25adbdd143df3d0bf6b6e8")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

latent_dim = 512
encoder_n_layers = 9
generator_n_layers = 9
encoder_lr = 1e-4
encoder_epochs = 100
encoder_batch_size = 256
use_background_bias = True
trainable_bias = True

# Diffusion hyperparameters
timesteps = 50  # INCREASED from 10 for better training
beta_start = 0.0001
beta_end = 0.02
hidden_dim = 512  # INCREASED from 256
embedding_dim = 256  # INCREASED from 128
num_layers = 6  # INCREASED from 4
learning_rate = 5e-5  # Slightly reduced for stability
max_epochs = 500  # REDUCED: better to train fewer epochs well
batch_size = 256
test_mode = '--test' in sys.argv

# Loss weights - ADJUSTED for better separation
generation_weight = 0.75  # Reduced from 0.9
classification_weight = 0.15  # Increased from 0.1
separation_weight = 0.10  # NEW: encourage class separation

print(f"Improved Diffusion Training Configuration:")
print(f"  Timesteps: {timesteps}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Num layers: {num_layers}")
print(f"  Max epochs: {max_epochs}")
print(f"  Loss weights - Gen: {generation_weight}, Class: {classification_weight}, Sep: {separation_weight}")

# =============================================================================
# CATE'S AUTOENCODER MODELS (unchanged)
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


def _load_state_dict_flex(model: nn.Module, checkpoint_obj):
    state_dict = None
    if isinstance(checkpoint_obj, dict):
        if 'state_dict' in checkpoint_obj:
            state_dict = checkpoint_obj['state_dict']
        elif 'model_state_dict' in checkpoint_obj:
            state_dict = checkpoint_obj['model_state_dict']
        elif 'encoder_state_dict' in checkpoint_obj:
            state_dict = checkpoint_obj['encoder_state_dict']
        elif 'generator_state_dict' in checkpoint_obj:
            state_dict = checkpoint_obj['generator_state_dict']
        else:
            if all(isinstance(k, str) for k in checkpoint_obj.keys()):
                state_dict = checkpoint_obj
    else:
        state_dict = checkpoint_obj

    if state_dict is None:
        raise RuntimeError("Unsupported checkpoint format for loading state dict")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[re.sub(r'^module\.', '', k)] = v
        model.load_state_dict(new_sd)
    return model


# =============================================================================
# DATA LOADING
# =============================================================================
def load_smile_embeddings():
    """Load SMILE embeddings for each chemical class"""
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
    
    print(f"\nLoaded SMILE embeddings: {list(label_embeddings.keys())}")
    return label_embeddings


def load_ims_data():
    """Load IMS spectra data"""
    print("\nLoading IMS data...")
    
    train_path = os.path.join(DATA_DIR, 'train_data.feather')
    test_path = os.path.join(DATA_DIR, 'test_data.feather')
    
    train_df = pd.read_feather(train_path)
    test_df = pd.read_feather(test_path)
    
    p_cols = [c for c in train_df.columns if c.startswith('p_')]
    n_cols = [c for c in train_df.columns if c.startswith('n_')]
    onehot_cols = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
    
    train_ims = np.concatenate([train_df[p_cols].values, train_df[n_cols].values], axis=1)
    test_ims = np.concatenate([test_df[p_cols].values, test_df[n_cols].values], axis=1)
    
    smile_embeddings = load_smile_embeddings()
    smile_embedding_dim = len(next(iter(smile_embeddings.values())))
    
    train_onehot = train_df[onehot_cols].values
    test_onehot = test_df[onehot_cols].values
    
    train_labels = train_onehot.argmax(axis=1)
    test_labels = test_onehot.argmax(axis=1)
    
    train_embeddings = np.zeros((len(train_df), smile_embedding_dim), dtype=np.float32)
    test_embeddings = np.zeros((len(test_df), smile_embedding_dim), dtype=np.float32)
    
    for idx, label in enumerate(onehot_cols):
        if label in smile_embeddings:
            train_mask = train_labels == idx
            test_mask = test_labels == idx
            train_embeddings[train_mask] = smile_embeddings[label]
            test_embeddings[test_mask] = smile_embeddings[label]
    
    # Normalize IMS
    ims_mean = train_ims.mean(axis=0)
    ims_std = train_ims.std(axis=0) + 1e-8
    
    train_ims_norm = (train_ims - ims_mean) / ims_std
    test_ims_norm = (test_ims - ims_mean) / ims_std
    
    train_bkg = train_ims_norm.mean(axis=0)
    
    print(f"  Loaded {len(train_df)} train, {len(test_df)} test samples")
    print(f"  Classes: {onehot_cols}")
    
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
        'num_ims_features': len(p_cols) + len(n_cols),
    }


# =============================================================================
# DIFFUSION SCHEDULE
# =============================================================================
betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32).to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)


def q_sample(x_start, t, noise=None):
    """Add noise to x_start at timestep t"""
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# =============================================================================
# IMPROVED DIFFUSION MODEL (with class conditioning)
# =============================================================================
class ClassConditionedDiffusion(nn.Module):
    """
    Diffusion model that takes:
    - noisy_latent: [batch, latent_dim]
    - t: [batch] timestep indices
    - smile_embedding: [batch, embedding_dim]
    - class_onehot: [batch, num_classes]
    
    Outputs:
    - pred_latent: [batch, latent_dim]
    - class_logits: [batch, num_classes]
    """
    
    def __init__(self, latent_dim=512, embedding_dim=512, num_classes=8, 
                 hidden_dim=512, num_layers=6, timesteps=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.timesteps = timesteps
        
        # Time embedding
        time_embed_dim = embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # SMILE embedding projection
        self.smile_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Class embedding projection
        self.class_proj = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection
        input_dim = latent_dim + time_embed_dim + hidden_dim + hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Main network
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
        
        # Output heads
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
        
        # Time embedding
        t_normalized = t.float().view(-1, 1) / self.timesteps
        t_emb = self.time_mlp(t_normalized)
        
        # SMILE embedding
        smile_emb = self.smile_proj(smile_embedding)
        
        # Class embedding
        class_emb = self.class_proj(class_onehot)
        
        # Concatenate inputs
        x = torch.cat([noisy_latent, t_emb, smile_emb, class_emb], dim=1)
        h = self.input_proj(x)
        
        # Process through layers with residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            h = h + layer(norm(h))
        
        # Output predictions
        pred_latent = self.latent_head(h)
        class_logits = self.class_head(h)
        
        return pred_latent, class_logits


# =============================================================================
# SEPARATION LOSS (to encourage inter-class distance)
# =============================================================================
def separation_loss(latent_codes, labels, num_classes=8):
    """
    Encourage latent codes from different classes to be well-separated.
    Uses a margin-based contrastive approach that's more stable.
    """
    # Compute class centroids
    centroids = []
    valid_classes = []
    
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() >= 2:  # Need at least 2 samples
            centroids.append(latent_codes[mask].mean(dim=0))
            valid_classes.append(c)
    
    if len(centroids) < 2:
        return torch.tensor(0.0, device=latent_codes.device)
    
    centroids = torch.stack(centroids)
    
    # Compute pairwise distances between centroids
    centroid_dists = torch.cdist(centroids, centroids)
    
    # Get off-diagonal elements (inter-class distances)
    mask = ~torch.eye(len(centroids), dtype=torch.bool, device=centroid_dists.device)
    inter_class_dists = centroid_dists[mask]
    
    # Margin-based loss: penalize if inter-class distance < margin
    margin = 10.0  # Desired minimum distance between class centroids
    loss = torch.relu(margin - inter_class_dists).mean()
    
    return loss


# =============================================================================
# TRAINING LOOP
# =============================================================================
print("\n" + "="*80)
print("LOADING DATA AND MODELS")
print("="*80)

data_dict = load_ims_data()
num_classes = data_dict['num_classes']
class_names = data_dict['class_names']

# Load encoder and generator
print("\nLoading encoder and generator...")
encoder = FlexibleNLayersEncoder(
    input_size=data_dict['num_ims_features'],
    output_size=latent_dim,
    n_layers=encoder_n_layers,
    init_style='bkg',
    bkg=torch.FloatTensor(data_dict['train_bkg']),
    trainable=trainable_bias
).to(device)

generator = FlexibleNLayersGenerator(
    input_size=latent_dim,
    output_size=data_dict['num_ims_features'],
    n_layers=generator_n_layers,
    init_style='bkg',
    bkg=torch.FloatTensor(data_dict['train_bkg']),
    trainable=trainable_bias
).to(device)

# Encode all data to latent space
print("\nEncoding data to latent space...")
encoder.eval()
with torch.no_grad():
    train_ims_tensor = torch.FloatTensor(data_dict['train_ims']).to(device)
    test_ims_tensor = torch.FloatTensor(data_dict['test_ims']).to(device)
    
    train_latent_list = []
    for i in range(0, len(train_ims_tensor), batch_size):
        batch = train_ims_tensor[i:i+batch_size]
        latent = encoder(batch, use_bias=use_background_bias)
        train_latent_list.append(latent.cpu())
    train_latent = torch.cat(train_latent_list, dim=0).numpy()
    
    test_latent_list = []
    for i in range(0, len(test_ims_tensor), batch_size):
        batch = test_ims_tensor[i:i+batch_size]
        latent = encoder(batch, use_bias=use_background_bias)
        test_latent_list.append(latent.cpu())
    test_latent = torch.cat(test_latent_list, dim=0).numpy()

print(f"  Train latent shape: {train_latent.shape}")
print(f"  Test latent shape: {test_latent.shape}")

# Create dataset
train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(train_latent),
    torch.FloatTensor(data_dict['train_embeddings']),
    torch.LongTensor(data_dict['train_labels']),
    torch.FloatTensor(data_dict['train_onehot'])
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize diffusion model
print("\nInitializing class-conditioned diffusion model...")
model = ClassConditionedDiffusion(
    latent_dim=latent_dim,
    embedding_dim=data_dict['train_embeddings'].shape[1],
    num_classes=num_classes,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    timesteps=timesteps
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

# Training
print("\n" + "="*80)
print("TRAINING DIFFUSION MODEL")
print("="*80)

wandb.init(
    project="ims-diffusion-separation",
    name=f"unified-diffusion-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        "timesteps": timesteps,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "generation_weight": generation_weight,
        "classification_weight": classification_weight,
        "separation_weight": separation_weight,
    }
)

loss_history = []
best_loss = float('inf')

for epoch in range(max_epochs):
    model.train()
    epoch_gen_loss = 0
    epoch_class_loss = 0
    epoch_sep_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
    for latent_batch, embedding_batch, label_batch, onehot_batch in pbar:
        latent_batch = latent_batch.to(device)
        embedding_batch = embedding_batch.to(device)
        label_batch = label_batch.to(device)
        onehot_batch = onehot_batch.to(device)
        
        optimizer.zero_grad()
        
        # Sample timesteps and z
        t = torch.randint(0, timesteps, (latent_batch.shape[0],), device=device)
        z = torch.randint(0, 2, (latent_batch.shape[0],), device=device)
        
        # Prepare inputs
        noisy_latent = torch.zeros_like(latent_batch)
        noisy_embedding = torch.zeros_like(embedding_batch)
        
        # Generation mode
        gen_mask = (z == 0)
        if gen_mask.any():
            noise_latent = torch.randn_like(latent_batch[gen_mask])
            noisy_latent[gen_mask] = q_sample(latent_batch[gen_mask], t[gen_mask], noise_latent)
            noisy_embedding[gen_mask] = embedding_batch[gen_mask]
        
        # Classification mode
        class_mask = (z == 1)
        if class_mask.any():
            noise_embedding = torch.randn_like(embedding_batch[class_mask])
            noisy_latent[class_mask] = latent_batch[class_mask]
            noisy_embedding[class_mask] = q_sample(embedding_batch[class_mask], t[class_mask], noise_embedding)
        
        # Forward pass
        pred_latent, class_logits = model(noisy_latent, t, noisy_embedding, onehot_batch)
        
        # Compute losses
        gen_loss = F.mse_loss(pred_latent[gen_mask], latent_batch[gen_mask]) if gen_mask.any() else torch.tensor(0.0, device=device)
        class_loss = F.cross_entropy(class_logits, label_batch) if True else torch.tensor(0.0, device=device)
        sep_loss = separation_loss(pred_latent, label_batch, num_classes)
        
        # Weighted combination
        loss = generation_weight * gen_loss + classification_weight * class_loss + separation_weight * sep_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_gen_loss += gen_loss.item() if isinstance(gen_loss, torch.Tensor) else gen_loss
        epoch_class_loss += class_loss.item() if isinstance(class_loss, torch.Tensor) else class_loss
        epoch_sep_loss += sep_loss.item() if isinstance(sep_loss, torch.Tensor) else sep_loss
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.6f}",
            'gen': f"{gen_loss.item():.6f}" if isinstance(gen_loss, torch.Tensor) else "0",
            'cls': f"{class_loss.item():.6f}" if isinstance(class_loss, torch.Tensor) else "0",
            'sep': f"{sep_loss.item():.6f}" if isinstance(sep_loss, torch.Tensor) else "0",
        })
    
    avg_gen_loss = epoch_gen_loss / max(num_batches, 1)
    avg_class_loss = epoch_class_loss / max(num_batches, 1)
    avg_sep_loss = epoch_sep_loss / max(num_batches, 1)
    avg_loss = generation_weight * avg_gen_loss + classification_weight * avg_class_loss + separation_weight * avg_sep_loss
    
    loss_history.append(avg_loss)
    scheduler.step()
    
    # Log to wandb
    wandb.log({
        "epoch": epoch,
        "loss/total": avg_loss,
        "loss/generation": avg_gen_loss,
        "loss/classification": avg_class_loss,
        "loss/separation": avg_sep_loss,
        "learning_rate": scheduler.get_last_lr()[0],
    })
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'diffusion_separation_best.pt'))
    
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.6f} (Gen={avg_gen_loss:.6f}, Class={avg_class_loss:.6f}, Sep={avg_sep_loss:.6f})")

# Save final model
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'diffusion_separation_final.pt'))
print(f"\nTraining complete! Best loss: {best_loss:.6f}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
results = {
    "model": "class_conditioned_diffusion_with_separation",
    "timestamp": datetime.now().isoformat(),
    "hyperparameters": {
        "timesteps": timesteps,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "generation_weight": generation_weight,
        "classification_weight": classification_weight,
        "separation_weight": separation_weight,
        "latent_dim": latent_dim,
        "num_classes": num_classes,
    },
    "best_loss": float(best_loss),
    "final_loss": float(loss_history[-1]) if loss_history else None,
}

with open(os.path.join(RESULTS_DIR, 'diffusion_separation_training_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Diffusion Training Loss (with Separation Component)')
plt.grid(True)
plt.savefig(os.path.join(IMAGES_DIR, 'diffusion_separation_training_loss.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nResults saved to {RESULTS_DIR}")
print(f"Model saved to {MODELS_DIR}")
wandb.finish()
