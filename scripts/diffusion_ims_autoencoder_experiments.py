"""
Diffusion Model for IMS Spectra Generation with Autoencoder Latent Space
=========================================================================

This script replaces PCA with Cate's trained encoder/generator for latent space diffusion.

Architecture:
1. Encoder: IMS spectra (1676-dim) → Latent space (512-dim)
2. Diffusion: Learn to denoise in latent space conditioned on SMILE embeddings
3. Generator: Latent space (512-dim) → IMS spectra (1676-dim)

Workflow:
- Train encoder/generator on IMS reconstruction task
- Encode all IMS data to latent space
- Train diffusion model in latent space
- Generate by sampling latent codes and decoding

Semi-Implicit Training with Dual Objectives:
- z=0: Generation - Input: [noisy_latent + clean_SMILE] -> Output: [clean_latent + clean_SMILE]
- z=1: Classification - Input: [clean_latent + noisy_SMILE] -> Output: [clean_latent + clean_SMILE]

Dataset: IMS Spectra with 8 chemical classes
- Positive mode: 838 features (p_184 to p_1021)
- Negative mode: 838 features (n_184 to n_1021)
- Total IMS features: 1676
- SMILE embeddings: 512-dimensional molecular embeddings
- Chemical classes: 8 (DEB, DEM, DMMP, DPM, DtBP, JP8, MES, TEPO)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy.stats import entropy
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

# Add ChemicalDataGeneration to path for functions module
CHEM_DATA_GEN_DIR = os.path.expanduser('~/ChemicalDataGeneration')
if CHEM_DATA_GEN_DIR not in sys.path:
    sys.path.insert(0, CHEM_DATA_GEN_DIR)

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

import wandb

# Initialize wandb
wandb.login(key="57680a36aa570ba8df25adbdd143df3d0bf6b6e8")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# Autoencoder hyperparameters (Cate's defaults)
latent_dim = 512  # Latent space dimension
encoder_n_layers = 9  # Number of layers in encoder
generator_n_layers = 9  # Number of layers in generator
encoder_lr = 1e-4
encoder_epochs = 100
encoder_batch_size = 256
use_background_bias = True  # Use Cate's background subtraction
trainable_bias = True

# Diffusion hyperparameters
timesteps = 10
beta_start = 0.0001
beta_end = 0.02
hidden_dim = 256
embedding_dim = 128
num_layers = 4
learning_rate = 1e-4
max_epochs = 1000
batch_size = 256
test_mode = '--test' in sys.argv  # Quick test with reduced data

# Optional: specify pretrained directory via CLI or environment
def _get_arg_value(flag):
    """Return value for a CLI flag, supporting --flag=value and --flag value."""
    for i, a in enumerate(sys.argv):
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

PRETRAINED_DIR = _get_arg_value('--pretrained-dir') or os.environ.get('PRETRAINED_MODELS_DIR')

# Loss weights (generation weighted more than classification)
generation_weight = 0.9  # Weight for latent reconstruction (prioritized)
classification_weight = 0.1  # Weight for SMILE embedding classification

print(f"Autoencoder Hyperparameters:")
print(f"  Latent dimension: {latent_dim}")
print(f"  Encoder layers: {encoder_n_layers}")
print(f"  Generator layers: {generator_n_layers}")
print(f"  Encoder learning rate: {encoder_lr}")
print(f"  Encoder epochs: {encoder_epochs}")
print(f"  Use background bias: {use_background_bias}")
print(f"  Trainable bias: {trainable_bias}")
print(f"\nDiffusion Hyperparameters:")
print(f"  Timesteps: {timesteps}")
print(f"  Hidden dimension: {hidden_dim}")
print(f"  Embedding dimension: {embedding_dim}")
print(f"  Number of layers: {num_layers}")
print(f"  Learning rate: {learning_rate}")
print(f"  Max epochs: {max_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Generation weight: {generation_weight}")
print(f"  Classification weight: {classification_weight}")
print(f"  Test mode: {test_mode}")

# =============================================================================
# CATE'S AUTOENCODER MODELS
# =============================================================================

class BiasOnlyLayer(nn.Module):
    """
    A PyTorch layer that adds a fixed or trainable bias to the input tensor.
    From Cate's implementation.
    """
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
    """
    Cate's encoder: IMS spectra (1676-dim) -> Latent space (512-dim)
    """
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
    """
    Cate's generator: Latent space (512-dim) -> IMS spectra (1676-dim)
    """
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
            if i < n_layers - 1:  # no activation after final layer
                layers.append(nn.LeakyReLU(inplace=True))
        self.generator = nn.Sequential(*layers)
    
    def forward(self, x, use_bias=False):
        x = self.generator(x)
        if use_bias and hasattr(self, 'bias_layer'):
            x = self.bias_layer(x)
        return x


# =============================================================================
# CHECKPOINT LOADING HELPERS
# =============================================================================
def _load_state_dict_flex(model: nn.Module, checkpoint_obj):
    """
    Load a state dict into model handling common checkpoint formats:
    - raw state_dict (tensor mapping)
    - {'state_dict': ...}
    - {'encoder_state_dict': ...} or {'generator_state_dict': ...}
    Also strips a leading 'module.' from keys if present.
    """
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
            # Assume it is directly a state_dict
            if all(isinstance(k, str) for k in checkpoint_obj.keys()):
                state_dict = checkpoint_obj
    else:
        state_dict = checkpoint_obj

    if state_dict is None:
        raise RuntimeError("Unsupported checkpoint format for loading state dict")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Try stripping leading 'module.' from keys
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[re.sub(r'^module\.', '', k)] = v
        model.load_state_dict(new_sd)
    return model

def _torch_load_weights_only(path, device):
    """Attempt to load checkpoint weights with PyTorch 2.6 safe loader."""
    try:
        # Import the actual functions module and create alias for checkpoint loading
        import models.functions as mf
        # Create a 'functions' module alias so torch.load can find the classes
        sys.modules['functions'] = mf
        
        # Register safe globals for PyTorch 2.6+ weights_only mode
        import torch.serialization as ts
        if hasattr(mf, 'FlexibleNLayersEncoder'):
            ts.add_safe_globals([mf.FlexibleNLayersEncoder])
        if hasattr(mf, 'FlexibleNLayersGenerator'):
            ts.add_safe_globals([mf.FlexibleNLayersGenerator])
    except Exception as e:
        print(f"Warning: Could not setup functions module alias: {e}")
        pass
    
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


# =============================================================================
# DATA LOADING
# =============================================================================
def load_smile_embeddings():
    """Load SMILE embeddings for each chemical class"""
    smile_path = os.path.join(DATA_DIR, 'name_smiles_embedding_file.csv')
    smile_df = pd.read_csv(smile_path)
    
    # Map abbreviated labels to full names
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
    
    # Create embedding dictionary
    embedding_dict = {}
    for _, row in smile_df.iterrows():
        if pd.notna(row['embedding']):
            embedding = np.array(ast.literal_eval(row['embedding']), dtype=np.float32)
            embedding_dict[row['Name']] = embedding
    
    # Map to abbreviated labels
    label_embeddings = {}
    for label, full_name in label_mapping.items():
        if full_name in embedding_dict:
            label_embeddings[label] = embedding_dict[full_name]
    
    print(f"\nLoaded SMILE embeddings:")
    print(f"  Embedding dimension: {len(next(iter(label_embeddings.values())))}")
    print(f"  Chemicals with embeddings: {list(label_embeddings.keys())}")
    
    return label_embeddings


def load_ims_data():
    """Load IMS spectra data from feather files"""
    print("\nLoading IMS spectra data...")
    
    train_path = os.path.join(DATA_DIR, 'train_data.feather')
    test_path = os.path.join(DATA_DIR, 'test_data.feather')
    
    train_df = pd.read_feather(train_path)
    test_df = pd.read_feather(test_path)
    
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    # Define column groups
    p_cols = [c for c in train_df.columns if c.startswith('p_')]
    n_cols = [c for c in train_df.columns if c.startswith('n_')]
    onehot_cols = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
    
    print(f"  Positive spectra features: {len(p_cols)}")
    print(f"  Negative spectra features: {len(n_cols)}")
    print(f"  Total IMS features: {len(p_cols) + len(n_cols)}")
    print(f"  Chemical classes: {len(onehot_cols)}")
    
    # Extract features
    train_ims = np.concatenate([
        train_df[p_cols].values,
        train_df[n_cols].values
    ], axis=1)
    
    test_ims = np.concatenate([
        test_df[p_cols].values,
        test_df[n_cols].values
    ], axis=1)
    
    # Load SMILE embeddings
    smile_embeddings = load_smile_embeddings()
    smile_embedding_dim = len(next(iter(smile_embeddings.values())))
    
    # Convert one-hot to SMILE embeddings
    train_onehot = train_df[onehot_cols].values
    test_onehot = test_df[onehot_cols].values
    
    train_labels = train_onehot.argmax(axis=1)
    test_labels = test_onehot.argmax(axis=1)
    
    # Create embedding matrices
    train_embeddings = np.zeros((len(train_df), smile_embedding_dim), dtype=np.float32)
    test_embeddings = np.zeros((len(test_df), smile_embedding_dim), dtype=np.float32)
    
    for idx, label in enumerate(onehot_cols):
        if label in smile_embeddings:
            train_mask = train_labels == idx
            test_mask = test_labels == idx
            train_embeddings[train_mask] = smile_embeddings[label]
            test_embeddings[test_mask] = smile_embeddings[label]
    
    print(f"\nSMILE embeddings created:")
    print(f"  Embedding dimension: {smile_embedding_dim}")
    print(f"  Train embeddings shape: {train_embeddings.shape}")
    print(f"  Test embeddings shape: {test_embeddings.shape}")
    
    # Normalize IMS data
    ims_mean = train_ims.mean(axis=0)
    ims_std = train_ims.std(axis=0) + 1e-8
    
    train_ims_norm = (train_ims - ims_mean) / ims_std
    test_ims_norm = (test_ims - ims_mean) / ims_std
    
    # Compute average background for bias layer
    train_bkg = train_ims_norm.mean(axis=0)
    
    print(f"\nNormalized IMS statistics:")
    print(f"  Train mean: {train_ims_norm.mean():.4f}, std: {train_ims_norm.std():.4f}")
    print(f"  Train range: [{train_ims_norm.min():.2f}, {train_ims_norm.max():.2f}]")
    print(f"  Background shape: {train_bkg.shape}")
    
    return {
        'train_ims': train_ims_norm,
        'train_embeddings': train_embeddings,
        'train_labels': train_labels,
        'test_ims': test_ims_norm,
        'test_embeddings': test_embeddings,
        'test_labels': test_labels,
        'ims_mean': ims_mean,
        'ims_std': ims_std,
        'train_bkg': train_bkg,
        'class_names': onehot_cols,
        'num_ims_features': len(p_cols) + len(n_cols),
        'num_embedding_dim': smile_embedding_dim,
        'smile_embeddings': smile_embeddings
    }


# =============================================================================
# AUTOENCODER TRAINING
# =============================================================================
def train_autoencoder(encoder, generator, train_ims, device, use_bias=True):
    """
    Train encoder and generator on IMS reconstruction task.
    """
    print("\n" + "="*80)
    print("TRAINING AUTOENCODER (Encoder + Generator)")
    print("="*80)
    
    encoder.train()
    generator.train()
    
    # Combine parameters for joint optimization
    params = list(encoder.parameters()) + list(generator.parameters())
    optimizer = torch.optim.AdamW(params, lr=encoder_lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Create dataset
    train_tensor = torch.FloatTensor(train_ims).to(device)
    dataset = torch.utils.data.TensorDataset(train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=encoder_batch_size, shuffle=True)
    
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(encoder_epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{encoder_epochs}")
        for batch_idx, (batch_ims,) in enumerate(pbar):
            optimizer.zero_grad()
            
            # Encode
            latent = encoder(batch_ims, use_bias=use_bias)
            
            # Decode
            reconstructed = generator(latent, use_bias=use_bias)
            
            # Reconstruction loss
            loss = criterion(reconstructed, batch_ims)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{encoder_epochs} - Avg Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, os.path.join(MODELS_DIR, 'best_autoencoder.pth'))
            print(f"  -> Saved best model (loss: {best_loss:.6f})")
    
    print(f"\nAutoencoder training complete!")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Final loss: {loss_history[-1]:.6f}")
    
    return encoder, generator, loss_history


# =============================================================================
# DIFFUSION NOISE SCHEDULE
# =============================================================================
def get_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear beta schedule"""
    return torch.linspace(beta_start, beta_end, timesteps)

betas = get_beta_schedule(timesteps, beta_start, beta_end).to(device)
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


def q_sample(x_start, t, noise=None):
    """
    Forward diffusion process: Add noise to x_0 to get x_t
    q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1).to(x_start.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1).to(x_start.device)
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# =============================================================================
# DIFFUSION MODEL (operates in latent space)
# =============================================================================
class DiffusionLatentModel(nn.Module):
    """
    Diffusion model operating in autoencoder latent space.
    
    Takes as input:
    - Noisy latent code (512-dim)
    - Timestep embedding
    - SMILE embedding (512-dim)
    
    Outputs:
    - Denoised latent code (512-dim)
    - Classification logits for SMILE embedding (512-dim)
    """
    
    def __init__(self, latent_dim=512, embedding_dim=512, hidden_dim=256, num_layers=4, timesteps=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
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
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection
        input_dim = latent_dim + time_embed_dim + hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Main network with residual connections
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
        
        self.smile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, noisy_latent, t, smile_embedding):
        """
        Args:
            noisy_latent: [batch, latent_dim] - Noisy latent code
            t: [batch] - Timestep indices
            smile_embedding: [batch, embedding_dim] - SMILE embeddings
        
        Returns:
            pred_latent: [batch, latent_dim] - Predicted clean latent
            pred_smile: [batch, embedding_dim] - Predicted SMILE embedding
        """
        batch_size = noisy_latent.shape[0]
        
        # Time embedding
        t_normalized = t.float().view(-1, 1) / self.timesteps
        t_emb = self.time_mlp(t_normalized)
        
        # SMILE embedding
        smile_emb = self.smile_proj(smile_embedding)
        
        # Concatenate inputs
        x = torch.cat([noisy_latent, t_emb, smile_emb], dim=1)
        h = self.input_proj(x)
        
        # Process through layers with residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            h = h + layer(norm(h))
        
        # Output predictions
        pred_latent = self.latent_head(h)
        pred_smile = self.smile_head(h)
        
        return pred_latent, pred_smile


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def p_sample(model, x_t, t, smile_embedding):
    """
    Single denoising step: p(x_{t-1} | x_t)
    """
    betas_t = betas[t].view(-1, 1).to(x_t.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1).to(x_t.device)
    sqrt_recip_alphas_t = sqrt_recip_alphas[t].view(-1, 1).to(x_t.device)
    
    # Predict clean latent
    model_output, _ = model(x_t, t, smile_embedding)
    
    # Compute mean
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
    
    if t[0] > 0:
        noise = torch.randn_like(x_t)
        posterior_variance_t = posterior_variance[t].view(-1, 1).to(x_t.device)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    else:
        return model_mean


def p_sample_loop(model, shape, smile_embedding, device):
    """
    Full denoising process: Generate samples from noise
    """
    batch_size = shape[0]
    
    # Start from pure noise
    latent = torch.randn(shape, device=device)
    
    for t in reversed(range(timesteps)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        latent = p_sample(model, latent, t_batch, smile_embedding)
    
    return latent


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

# Load data
data_dict = load_ims_data()

if test_mode:
    print("\n*** TEST MODE: Using only 5000 samples ***\n")
    n_samples_use = 5000
    indices = np.random.choice(len(data_dict['train_ims']), n_samples_use, replace=False)
    data_dict['train_ims'] = data_dict['train_ims'][indices]
    data_dict['train_embeddings'] = data_dict['train_embeddings'][indices]
    data_dict['train_labels'] = data_dict['train_labels'][indices]

num_ims_features = data_dict['num_ims_features']
num_embedding_dim = data_dict['num_embedding_dim']
num_classes = len(data_dict['class_names'])

print(f"\nData dimensions:")
print(f"  IMS features: {num_ims_features}")
print(f"  Embedding dimension: {num_embedding_dim}")
print(f"  Number of classes: {num_classes}")

# =============================================================================
# TRAIN OR LOAD AUTOENCODER
# =============================================================================
print("\n" + "="*80)
print("INITIALIZING AUTOENCODER")
print("="*80)

# Prepare background for bias layer
train_bkg_tensor = torch.FloatTensor(data_dict['train_bkg']).to(device)

# Initialize encoder and generator
encoder = FlexibleNLayersEncoder(
    input_size=num_ims_features,
    output_size=latent_dim,
    n_layers=encoder_n_layers,
    init_style='bkg' if use_background_bias else None,
    bkg=train_bkg_tensor if use_background_bias else None,
    trainable=trainable_bias
).to(device)

generator = FlexibleNLayersGenerator(
    input_size=latent_dim,
    output_size=num_ims_features,
    n_layers=generator_n_layers,
    init_style='bkg' if use_background_bias else None,
    bkg=train_bkg_tensor if use_background_bias else None,
    trainable=trainable_bias
).to(device)

# Try to load pre-trained model
loaded_pretrained = False

# Prefer scratch directory if provided
if PRETRAINED_DIR:
    enc_path = os.path.join(PRETRAINED_DIR, 'nine_layer_ims_to_chemnet_encoder.pth')
    gen_path = os.path.join(PRETRAINED_DIR, 'nine_layer_ChemNet_to_ims_generator_from_nine_layer__encoder.pth')
    if os.path.exists(enc_path) and os.path.exists(gen_path):
        print(f"\nLoading pre-trained autoencoder from {PRETRAINED_DIR}")
        enc_ckpt = _torch_load_weights_only(enc_path, device)
        gen_ckpt = _torch_load_weights_only(gen_path, device)
        _load_state_dict_flex(encoder, enc_ckpt)
        _load_state_dict_flex(generator, gen_ckpt)
        loaded_pretrained = True
        print("  Loaded encoder and generator weights from scratch directory")
    else:
        print(f"\nWARNING: Pretrained dir provided but expected files not found: {enc_path}, {gen_path}")

if not loaded_pretrained:
    # Fallback to local combined checkpoint
    autoencoder_path = os.path.join(MODELS_DIR, 'best_autoencoder.pth')
    if os.path.exists(autoencoder_path):
        print(f"\nLoading pre-trained autoencoder from {autoencoder_path}")
        checkpoint = _torch_load_weights_only(autoencoder_path, device)
        _load_state_dict_flex(encoder, checkpoint)
        _load_state_dict_flex(generator, checkpoint)
        # Optional epoch/loss info if present
        epoch_info = checkpoint.get('epoch') if isinstance(checkpoint, dict) else None
        loss_info = checkpoint.get('loss') if isinstance(checkpoint, dict) else None
        if epoch_info is not None and loss_info is not None:
            print(f"  Loaded model from epoch {epoch_info} with loss {loss_info:.6f}")
        loaded_pretrained = True

if not loaded_pretrained:
    print("\nNo pre-trained autoencoder found. Training from scratch...")
    encoder, generator, ae_loss_history = train_autoencoder(
        encoder, generator, data_dict['train_ims'], device, use_bias=use_background_bias
    )

# Count parameters
encoder_params = sum(p.numel() for p in encoder.parameters())
generator_params = sum(p.numel() for p in generator.parameters())
print(f"\nAutoencoder architecture:")
print(f"  Encoder parameters: {encoder_params:,}")
print(f"  Generator parameters: {generator_params:,}")
print(f"  Total autoencoder parameters: {encoder_params + generator_params:,}")

# =============================================================================
# ENCODE ALL DATA TO LATENT SPACE
# =============================================================================
print("\n" + "="*80)
print("ENCODING DATA TO LATENT SPACE")
print("="*80)

encoder.eval()
with torch.no_grad():
    train_ims_tensor = torch.FloatTensor(data_dict['train_ims']).to(device)
    test_ims_tensor = torch.FloatTensor(data_dict['test_ims']).to(device)
    
    # Encode in batches to avoid OOM
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

print(f"Encoded latent codes:")
print(f"  Train latent shape: {train_latent.shape}")
print(f"  Test latent shape: {test_latent.shape}")
print(f"  Train latent mean: {train_latent.mean():.4f}, std: {train_latent.std():.4f}")
print(f"  Train latent range: [{train_latent.min():.2f}, {train_latent.max():.2f}]")

# Verify reconstruction quality
with torch.no_grad():
    sample_indices = np.random.choice(len(test_ims_tensor), min(1000, len(test_ims_tensor)), replace=False)
    sample_ims = test_ims_tensor[sample_indices]
    sample_latent = torch.FloatTensor(test_latent[sample_indices]).to(device)
    reconstructed = generator(sample_latent, use_bias=use_background_bias)
    recon_mse = F.mse_loss(reconstructed, sample_ims).item()
    print(f"\nAutoencoder reconstruction quality:")
    print(f"  Test MSE: {recon_mse:.6f}")

# =============================================================================
# INITIALIZE DIFFUSION MODEL
# =============================================================================
print("\n" + "="*80)
print("INITIALIZING DIFFUSION MODEL")
print("="*80)

model = DiffusionLatentModel(
    latent_dim=latent_dim,
    embedding_dim=num_embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    timesteps=timesteps
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Diffusion model parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Create dataset with latent codes
train_dataset = torch.utils.data.TensorDataset(
    torch.FloatTensor(train_latent),
    torch.FloatTensor(data_dict['train_embeddings']),
    torch.LongTensor(data_dict['train_labels'])
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"Dataset prepared:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Batches per epoch: {len(train_loader)}")

# =============================================================================
# TRAIN DIFFUSION MODEL
# =============================================================================
print("\n" + "="*80)
print("STARTING DIFFUSION TRAINING (in latent space)")
print("="*80)

# Initialize wandb
wandb_run = wandb.init(
    entity="kjmetzler-worcester-polytechnic-institute",
    project="ims-spectra-diffusion-autoencoder",
    name=f"autoencoder_diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        "timesteps": timesteps,
        "beta_start": beta_start,
        "beta_end": beta_end,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "generation_weight": generation_weight,
        "classification_weight": classification_weight,
        "latent_dim": latent_dim,
        "encoder_n_layers": encoder_n_layers,
        "generator_n_layers": generator_n_layers,
        "use_background_bias": use_background_bias,
        "num_ims_features": num_ims_features,
        "num_embedding_dim": num_embedding_dim,
        "total_diffusion_params": total_params,
        "encoder_params": encoder_params,
        "generator_params": generator_params
    }
)

loss_history = []
gen_loss_history = []
class_loss_history = []
best_loss = float('inf')

for epoch in range(max_epochs):
    model.train()
    epoch_loss = 0
    epoch_gen_loss = 0
    epoch_class_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
    for batch_idx, (latent_batch, embedding_batch, label_batch) in enumerate(pbar):
        latent_batch = latent_batch.to(device)
        embedding_batch = embedding_batch.to(device)
        
        optimizer.zero_grad()
        
        # Sample timesteps
        t = torch.randint(0, timesteps, (latent_batch.shape[0],), device=device)
        
        # Sample z (0 = generation, 1 = classification)
        z = torch.randint(0, 2, (latent_batch.shape[0],), device=device)
        
        # Prepare inputs based on z
        noisy_latent = torch.zeros_like(latent_batch)
        noisy_embedding = torch.zeros_like(embedding_batch)
        
        # z=0: Generation mode - corrupt latent, keep SMILE clean
        gen_mask = (z == 0)
        if gen_mask.any():
            noise_latent = torch.randn_like(latent_batch[gen_mask])
            noisy_latent[gen_mask] = q_sample(latent_batch[gen_mask], t[gen_mask], noise_latent)
            noisy_embedding[gen_mask] = embedding_batch[gen_mask]
        
        # z=1: Classification mode - keep latent clean, corrupt SMILE
        class_mask = (z == 1)
        if class_mask.any():
            noise_embedding = torch.randn_like(embedding_batch[class_mask])
            noisy_latent[class_mask] = latent_batch[class_mask]
            noisy_embedding[class_mask] = q_sample(embedding_batch[class_mask], t[class_mask], noise_embedding)
        
        # Forward pass
        pred_latent, pred_embedding = model(noisy_latent, t, noisy_embedding)
        
        # Compute losses
        gen_loss = F.mse_loss(pred_latent[gen_mask], latent_batch[gen_mask]) if gen_mask.any() else 0
        class_loss = F.mse_loss(pred_embedding[class_mask], embedding_batch[class_mask]) if class_mask.any() else 0
        
        # Weighted combination
        if isinstance(gen_loss, torch.Tensor) and isinstance(class_loss, torch.Tensor):
            loss = generation_weight * gen_loss + classification_weight * class_loss
        elif isinstance(gen_loss, torch.Tensor):
            loss = gen_loss
        elif isinstance(class_loss, torch.Tensor):
            loss = class_loss
        else:
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        if isinstance(gen_loss, torch.Tensor):
            epoch_gen_loss += gen_loss.item()
        if isinstance(class_loss, torch.Tensor):
            epoch_class_loss += class_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'gen': f"{gen_loss.item() if isinstance(gen_loss, torch.Tensor) else 0:.4f}",
            'class': f"{class_loss.item() if isinstance(class_loss, torch.Tensor) else 0:.4f}"
        })
    
    # Average losses
    avg_loss = epoch_loss / num_batches
    avg_gen_loss = epoch_gen_loss / num_batches
    avg_class_loss = epoch_class_loss / num_batches
    
    loss_history.append(avg_loss)
    gen_loss_history.append(avg_gen_loss)
    class_loss_history.append(avg_class_loss)
    
    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "generation_loss": avg_gen_loss,
        "classification_loss": avg_class_loss
    })
    
    print(f"Epoch {epoch+1}/{max_epochs} - Loss: {avg_loss:.6f} | Gen: {avg_gen_loss:.6f} | Class: {avg_class_loss:.6f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'encoder_state_dict': encoder.state_dict(),
            'generator_state_dict': generator.state_dict()
        }, os.path.join(MODELS_DIR, 'best_diffusion_autoencoder_model.pth'))
        print(f"  -> Saved best model (loss: {best_loss:.6f})")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

# =============================================================================
# EVALUATION
# =============================================================================
print("\nEvaluating model...")

model.eval()
encoder.eval()
generator.eval()

n_test = min(1000, len(data_dict['test_ims']))

# Generate synthetic samples
print("\nGenerating synthetic samples...")
generated_latent_list = []
test_labels = data_dict['test_labels'][:n_test]

with torch.no_grad():
    for i in range(0, n_test, batch_size):
        end_idx = min(i + batch_size, n_test)
        batch_size_actual = end_idx - i
        
        batch_embeddings = torch.FloatTensor(data_dict['test_embeddings'][i:end_idx]).to(device)
        
        # Generate latent codes via diffusion
        generated_latent_batch = p_sample_loop(
            model,
            (batch_size_actual, latent_dim),
            batch_embeddings,
            device
        )
        
        generated_latent_list.append(generated_latent_batch.cpu())

generated_latent = torch.cat(generated_latent_list, dim=0).to(device)

# Decode to IMS spectra
with torch.no_grad():
    generated_ims_list = []
    for i in range(0, len(generated_latent), batch_size):
        batch_latent = generated_latent[i:i+batch_size]
        batch_ims = generator(batch_latent, use_bias=use_background_bias)
        generated_ims_list.append(batch_ims.cpu())
    
    generated_ims = torch.cat(generated_ims_list, dim=0).numpy()

print(f"Generated {len(generated_ims)} synthetic IMS spectra")

# Compute MSE
gen_mse = mean_squared_error(data_dict['test_ims'][:n_test].flatten(), generated_ims.flatten())
print(f"Generation MSE: {gen_mse:.6f}")

# Compute FID
print("\nComputing FID score...")
pca_for_fid = PCA(n_components=min(50, n_test, latent_dim))
real_latent_sample = test_latent[:n_test]
gen_latent_sample = generated_latent.cpu().numpy()

real_pca_features = pca_for_fid.fit_transform(real_latent_sample)
gen_pca_features = pca_for_fid.transform(gen_latent_sample)

def calculate_fid(real_features, generated_features):
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_gen = np.mean(generated_features, axis=0)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    diff = mu_real - mu_gen
    ssdiff = np.sum(diff ** 2)
    
    covmean = scipy.linalg.sqrtm(sigma_real.dot(sigma_gen))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    return fid

fid_score = calculate_fid(real_pca_features, gen_pca_features)
print(f"FID Score (latent space): {fid_score:.4f}")

# Per-class quality
print("\nComputing per-class quality...")
class_mses = []
for class_idx in range(num_classes):
    class_mask = test_labels == class_idx
    if class_mask.sum() > 0:
        class_real = data_dict['test_ims'][:n_test][class_mask]
        class_gen = generated_ims[class_mask]
        class_mse = mean_squared_error(class_real.flatten(), class_gen.flatten())
        class_mses.append(class_mse)
        print(f"  {data_dict['class_names'][class_idx]}: MSE = {class_mse:.6f}")
    else:
        class_mses.append(0)

# Log to wandb
wandb.log({
    "test_generation_mse": gen_mse,
    "test_fid_score": fid_score
})

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\nGenerating visualizations...")

# 1. Training loss
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(loss_history)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Total Loss')
axes[0].set_title('Total Training Loss')
axes[0].grid(True, alpha=0.3)

axes[1].plot(gen_loss_history)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Generation Loss')
axes[1].set_title('Generation Loss (Latent Space)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(class_loss_history)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Classification Loss')
axes[2].set_title('Classification Loss (SMILE)')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Diffusion Training with Autoencoder', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'autoencoder_diffusion_training_loss.png'), dpi=150)
wandb.log({"training_loss": wandb.Image(os.path.join(IMAGES_DIR, 'autoencoder_diffusion_training_loss.png'))})
plt.close()
print("  -> Saved training loss plot")

# 2. Generated vs Real Spectra
fig, axes = plt.subplots(4, 2, figsize=(15, 12))
for i in range(4):
    axes[i, 0].plot(data_dict['test_ims'][i])
    axes[i, 0].set_title(f'Real Spectrum {i+1} ({data_dict["class_names"][test_labels[i]]})')
    axes[i, 0].set_xlabel('Feature Index')
    axes[i, 0].set_ylabel('Intensity')
    axes[i, 0].grid(True, alpha=0.3)
    
    axes[i, 1].plot(generated_ims[i])
    axes[i, 1].set_title(f'Generated Spectrum {i+1}')
    axes[i, 1].set_xlabel('Feature Index')
    axes[i, 1].set_ylabel('Intensity')
    axes[i, 1].grid(True, alpha=0.3)

plt.suptitle('Real vs Generated IMS Spectra (Autoencoder + Diffusion)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'autoencoder_diffusion_spectra_comparison.png'), dpi=150)
wandb.log({"spectra_comparison": wandb.Image(os.path.join(IMAGES_DIR, 'autoencoder_diffusion_spectra_comparison.png'))})
plt.close()
print("  -> Saved spectra comparison")

# 3. Latent space visualization (PCA of latent codes)
pca_latent = PCA(n_components=2)
real_latent_pca = pca_latent.fit_transform(real_latent_sample)
gen_latent_pca = pca_latent.transform(gen_latent_sample)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter1 = axes[0].scatter(real_latent_pca[:, 0], real_latent_pca[:, 1], c=test_labels, 
                           cmap='tab10', alpha=0.6, s=20)
axes[0].set_title('Real Latent Codes (PCA)', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca_latent.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_latent.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Chemical Class')

scatter2 = axes[1].scatter(gen_latent_pca[:, 0], gen_latent_pca[:, 1], c=test_labels,
                           cmap='tab10', alpha=0.6, s=20)
axes[1].set_title('Generated Latent Codes (PCA)', fontsize=14, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca_latent.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_latent.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Chemical Class')

plt.suptitle('Latent Space: Real vs Generated', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'autoencoder_latent_space_comparison.png'), dpi=150)
wandb.log({"latent_comparison": wandb.Image(os.path.join(IMAGES_DIR, 'autoencoder_latent_space_comparison.png'))})
plt.close()
print("  -> Saved latent space comparison")

# 4. Per-class quality
plt.figure(figsize=(10, 6))
plt.bar(data_dict['class_names'], class_mses)
plt.xlabel('Chemical Class')
plt.ylabel('MSE')
plt.title('Generation Quality by Chemical Class')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'autoencoder_per_class_quality.png'), dpi=150)
wandb.log({"per_class_quality": wandb.Image(os.path.join(IMAGES_DIR, 'autoencoder_per_class_quality.png'))})
plt.close()
print("  -> Saved per-class quality plot")

# 5. IMS space PCA comparison
pca_ims = PCA(n_components=2)
real_ims_pca = pca_ims.fit_transform(data_dict['test_ims'][:n_test])
gen_ims_pca = pca_ims.transform(generated_ims)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter1 = axes[0].scatter(real_ims_pca[:, 0], real_ims_pca[:, 1], c=test_labels, 
                           cmap='tab10', alpha=0.6, s=20)
axes[0].set_title('Real IMS Spectra (PCA)', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca_ims.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_ims.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Chemical Class')

scatter2 = axes[1].scatter(gen_ims_pca[:, 0], gen_ims_pca[:, 1], c=test_labels,
                           cmap='tab10', alpha=0.6, s=20)
axes[1].set_title('Generated IMS Spectra (PCA)', fontsize=14, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca_ims.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_ims.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Chemical Class')

plt.suptitle('IMS Space: Real vs Generated', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'autoencoder_ims_pca_comparison.png'), dpi=150)
wandb.log({"ims_pca_comparison": wandb.Image(os.path.join(IMAGES_DIR, 'autoencoder_ims_pca_comparison.png'))})
plt.close()
print("  -> Saved IMS PCA comparison")

# Save results
print("\nSaving results...")
np.save(os.path.join(RESULTS_DIR, 'autoencoder_generated_ims.npy'), generated_ims)
np.save(os.path.join(RESULTS_DIR, 'autoencoder_train_latent.npy'), train_latent)
np.save(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'), test_latent)
np.save(os.path.join(RESULTS_DIR, 'autoencoder_generated_latent.npy'), gen_latent_sample)

results = {
    'training_complete': True,
    'final_loss': loss_history[-1],
    'best_loss': best_loss,
    'test_generation_mse': float(gen_mse),
    'test_fid_score': float(fid_score),
    'autoencoder_reconstruction_mse': float(recon_mse),
    'num_diffusion_parameters': total_params,
    'num_encoder_parameters': encoder_params,
    'num_generator_parameters': generator_params,
    'epochs_trained': max_epochs,
    'latent_dim': latent_dim,
    'encoder_n_layers': encoder_n_layers,
    'generator_n_layers': generator_n_layers,
    'use_background_bias': use_background_bias,
    'class_names': data_dict['class_names'],
    'per_class_mse': [float(m) for m in class_mses]
}

with open(os.path.join(RESULTS_DIR, 'autoencoder_diffusion_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print(f"Results:")
print(f"  Autoencoder reconstruction MSE: {recon_mse:.6f}")
print(f"  Final diffusion loss: {loss_history[-1]:.6f}")
print(f"  Best diffusion loss: {best_loss:.6f}")
print(f"  Test generation MSE: {gen_mse:.6f}")
print(f"  Test FID score (latent space): {fid_score:.4f}")
print(f"\nArchitecture:")
print(f"  Encoder: {num_ims_features} -> {latent_dim} ({encoder_params:,} params)")
print(f"  Generator: {latent_dim} -> {num_ims_features} ({generator_params:,} params)")
print(f"  Diffusion: {total_params:,} params")
print(f"\nOutputs saved to:")
print(f"  Models: {MODELS_DIR}")
print(f"  Images: {IMAGES_DIR}")
print(f"  Results: {RESULTS_DIR}")
print("="*80)

wandb.finish()
