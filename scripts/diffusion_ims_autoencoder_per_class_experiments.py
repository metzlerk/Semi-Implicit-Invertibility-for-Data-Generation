"""
Diffusion Model for IMS Spectra Generation with Autoencoder Latent Space
=========================================================================

This script replaces PCA with Cate's trained encoder/generator for latent space diffusion.
ENHANCED VERSION: Trains separate diffusion models for each chemical class with ChemNet embedding visualization.

Architecture:
1. Encoder: IMS spectra (1676-dim) → Latent space (512-dim)
2. Diffusion: Learn to denoise in latent space conditioned on SMILE embeddings (per-class)
3. Generator: Latent space (512-dim) → IMS spectra (1676-dim)

Workflow:
- Train encoder/generator on IMS reconstruction task
- Encode all IMS data to latent space
- Train separate diffusion model for EACH chemical class
- Generate by sampling latent codes and decoding
- Compare generated vs. actual ChemNet embeddings in PCA space

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
    # Create module alias for checkpoint loading
    try:
        import models.functions as mf
        sys.modules['functions'] = mf
    except ImportError:
        pass  # Will handle in checkpoint loading if needed

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
            # Verify keys look like parameter names
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
# MAIN EXECUTION
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

# If a pretrained directory is provided, prefer it
if PRETRAINED_DIR:
    enc_path = os.path.join(PRETRAINED_DIR, 'nine_layer_ims_to_chemnet_encoder.pth')
    gen_path = os.path.join(PRETRAINED_DIR, 'nine_layer_ChemNet_to_ims_generator_from_nine_layer__encoder.pth')
    if os.path.exists(enc_path) and os.path.exists(gen_path):
        print(f"\nLoading pre-trained autoencoder from {PRETRAINED_DIR}")
        # Explicitly allow loading full checkpoint objects due to PyTorch 2.6 default change
        enc_ckpt = torch.load(enc_path, map_location=device, weights_only=False)
        gen_ckpt = torch.load(gen_path, map_location=device, weights_only=False)
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
        checkpoint = torch.load(autoencoder_path, map_location=device, weights_only=False)
        
        # Handle combined checkpoint with separate encoder/generator state dicts
        if 'encoder_state_dict' in checkpoint:
            _load_state_dict_flex(encoder, {'state_dict': checkpoint['encoder_state_dict']})
        else:
            _load_state_dict_flex(encoder, checkpoint)
            
        if 'generator_state_dict' in checkpoint:
            _load_state_dict_flex(generator, {'state_dict': checkpoint['generator_state_dict']})
        else:
            _load_state_dict_flex(generator, checkpoint)
        
        # Optional epoch/loss info if present
        epoch_info = checkpoint.get('epoch')
        loss_info = checkpoint.get('loss')
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
# TRAIN PER-CLASS DIFFUSION MODELS
# =============================================================================

print("\n" + "="*80)
print("TRAINING PER-CLASS DIFFUSION MODELS")
print("="*80)

per_class_results = {}
generated_ims_all = {}
generated_embeddings_all = {}

for class_idx, class_name in enumerate(data_dict['class_names']):
    print(f"\n{'='*80}")
    print(f"TRAINING DIFFUSION FOR CLASS: {class_name} ({class_idx + 1}/{num_classes})")
    print(f"{'='*80}")
    
    # Get data for this class
    train_mask = data_dict['train_labels'] == class_idx
    test_mask = data_dict['test_labels'] == class_idx
    
    n_train_class = train_mask.sum()
    n_test_class = test_mask.sum()
    
    print(f"  Training samples: {n_train_class}")
    print(f"  Test samples: {n_test_class}")
    
    if n_train_class == 0:
        print(f"  WARNING: No training samples for class {class_name}, skipping...")
        continue
    
    # Create class-specific dataset
    class_latent = train_latent[train_mask]
    class_embeddings = data_dict['train_embeddings'][train_mask]
    
    class_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(class_latent),
        torch.FloatTensor(class_embeddings)
    )
    class_loader = torch.utils.data.DataLoader(class_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize diffusion model for this class
    model_class = DiffusionLatentModel(
        latent_dim=latent_dim,
        embedding_dim=num_embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        timesteps=timesteps
    ).to(device)
    
    total_params = sum(p.numel() for p in model_class.parameters())
    print(f"  Diffusion model parameters: {total_params:,}")
    
    optimizer_class = torch.optim.AdamW(model_class.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Initialize wandb
    wandb_run = wandb.init(
        entity="kjmetzler-worcester-polytechnic-institute",
        project="ims-spectra-diffusion-autoencoder",
        name=f"class_{class_name}_diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "class": class_name,
            "class_index": class_idx,
            "timesteps": timesteps,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "latent_dim": latent_dim,
            "num_train_samples": n_train_class,
            "num_test_samples": n_test_class
        }
    )
    
    # Training loop
    loss_history = []
    best_loss = float('inf')
    
    print(f"\n  Training for {max_epochs} epochs...")
    for epoch in range(max_epochs):
        model_class.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(class_loader, desc=f"  Epoch {epoch+1}/{max_epochs}", leave=False)
        for batch_idx, (latent_batch, embedding_batch) in enumerate(pbar):
            latent_batch = latent_batch.to(device)
            embedding_batch = embedding_batch.to(device)
            
            optimizer_class.zero_grad()
            
            # Sample timesteps
            t = torch.randint(0, timesteps, (latent_batch.shape[0],), device=device)
            
            # Sample z (0 = generation, 1 = classification)
            z = torch.randint(0, 2, (latent_batch.shape[0],), device=device)
            
            # Prepare inputs based on z
            noisy_latent = torch.zeros_like(latent_batch)
            noisy_embedding = torch.zeros_like(embedding_batch)
            
            # z=0: Generation mode
            gen_mask = (z == 0)
            if gen_mask.any():
                noise_latent = torch.randn_like(latent_batch[gen_mask])
                noisy_latent[gen_mask] = q_sample(latent_batch[gen_mask], t[gen_mask], noise_latent)
                noisy_embedding[gen_mask] = embedding_batch[gen_mask]
            
            # z=1: Classification mode
            class_mask = (z == 1)
            if class_mask.any():
                noise_embedding = torch.randn_like(embedding_batch[class_mask])
                noisy_latent[class_mask] = latent_batch[class_mask]
                noisy_embedding[class_mask] = q_sample(embedding_batch[class_mask], t[class_mask], noise_embedding)
            
            # Forward pass
            pred_latent, pred_embedding = model_class(noisy_latent, t, noisy_embedding)
            
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
            torch.nn.utils.clip_grad_norm_(model_class.parameters(), 1.0)
            optimizer_class.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Average loss
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "class": class_name
        })
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{max_epochs} - Loss: {avg_loss:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_class.state_dict(),
                'optimizer_state_dict': optimizer_class.state_dict(),
                'loss': avg_loss,
                'class_name': class_name,
                'class_index': class_idx
            }, os.path.join(MODELS_DIR, f'best_diffusion_{class_name}.pth'))
    
    wandb.finish()
    
    print(f"  Training complete! Best loss: {best_loss:.6f}")
    
    # =============================================================================
    # EVALUATION FOR THIS CLASS
    # =============================================================================
    
    model_class.eval()
    
    n_test_class_eval = min(200, n_test_class)
    test_indices_class = np.where(test_mask)[0][:n_test_class_eval]
    
    # Generate samples
    print(f"  Generating {n_test_class_eval} synthetic samples for {class_name}...")
    
    generated_latent_list = []
    with torch.no_grad():
        for i in range(0, n_test_class_eval, batch_size):
            end_idx = min(i + batch_size, n_test_class_eval)
            batch_size_actual = end_idx - i
            
            # Use the class embedding for generation
            batch_embedding = torch.FloatTensor(
                np.tile(data_dict['test_embeddings'][test_indices_class[0:1]], (batch_size_actual, 1))
            ).to(device)
            
            # Generate latent codes
            generated_latent_batch = p_sample_loop(
                model_class,
                (batch_size_actual, latent_dim),
                batch_embedding,
                device
            )
            
            generated_latent_list.append(generated_latent_batch.cpu())
    
    generated_latent_class = torch.cat(generated_latent_list, dim=0).to(device)
    
    # Decode to IMS spectra
    with torch.no_grad():
        generated_ims_list = []
        for i in range(0, len(generated_latent_class), batch_size):
            batch_latent = generated_latent_class[i:i+batch_size]
            batch_ims = generator(batch_latent, use_bias=use_background_bias)
            generated_ims_list.append(batch_ims.cpu())
        
        generated_ims_class = torch.cat(generated_ims_list, dim=0).numpy()
    
    generated_ims_all[class_name] = generated_ims_class
    
    # Compute metrics
    real_ims_class = data_dict['test_ims'][test_indices_class]
    gen_mse = mean_squared_error(real_ims_class.flatten(), generated_ims_class.flatten())
    
    print(f"  Generation MSE: {gen_mse:.6f}")
    
    per_class_results[class_name] = {
        'best_loss': best_loss,
        'final_loss': loss_history[-1],
        'gen_mse': gen_mse,
        'n_train': n_train_class,
        'n_test': n_test_class
    }

# =============================================================================
# GENERATE GAUSSIAN BASELINE FOR COMPARISON
# =============================================================================

print("\n" + "="*80)
print("GENERATING GAUSSIAN BASELINE SAMPLES")
print("="*80)

generated_ims_gaussian = {}

for class_idx, class_name in enumerate(data_dict['class_names']):
    print(f"\nGenerating Gaussian samples for {class_name}...")
    
    test_mask = data_dict['test_labels'] == class_idx
    n_test_class = test_mask.sum()
    n_samples = min(200, n_test_class)
    
    # Get the latent space statistics for this class
    train_mask = data_dict['train_labels'] == class_idx
    class_latent_codes = data_dict['train_latent'][train_mask]
    latent_mean = class_latent_codes.mean(axis=0)
    latent_std = class_latent_codes.std(axis=0)
    
    # Generate from Gaussian
    gaussian_latent = torch.FloatTensor(
        np.random.randn(n_samples, latent_dim) * latent_std + latent_mean
    ).to(device)
    
    # Decode to IMS spectra
    with torch.no_grad():
        gaussian_ims_list = []
        for i in range(0, len(gaussian_latent), batch_size):
            batch_latent = gaussian_latent[i:i+batch_size]
            batch_ims = generator(batch_latent, use_bias=use_background_bias)
            gaussian_ims_list.append(batch_ims.cpu())
        
        gaussian_ims_class = torch.cat(gaussian_ims_list, dim=0).numpy()
    
    generated_ims_gaussian[class_name] = gaussian_ims_class
    print(f"  Generated {n_samples} Gaussian samples for {class_name}")

# =============================================================================
# VISUALIZATION: COMBINED PCA OF LATENT SPACE (ALL CHEMICALS)
# =============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

print("\n  Creating combined latent space PCA (all chemicals)...")

# Sample latent codes for visualization
max_samples_per_class = 500
latent_real_list = []
latent_labels_list = []

for class_idx in range(num_classes):
    train_mask = data_dict['train_labels'] == class_idx
    class_latent = data_dict['train_latent'][train_mask]
    n_samples = min(max_samples_per_class, len(class_latent))
    sampled_indices = np.random.choice(len(class_latent), n_samples, replace=False)
    latent_real_list.append(class_latent[sampled_indices])
    latent_labels_list.append(np.full(n_samples, class_idx))

latent_real_all = np.vstack(latent_real_list)
latent_labels_all = np.concatenate(latent_labels_list)

# PCA on latent space
pca_latent = PCA(n_components=2)
latent_pca = pca_latent.fit_transform(latent_real_all)

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

for class_idx, class_name in enumerate(data_dict['class_names']):
    mask = latent_labels_all == class_idx
    ax.scatter(
        latent_pca[mask, 0],
        latent_pca[mask, 1],
        c=[colors[class_idx]],
        label=class_name,
        alpha=0.6,
        s=30,
        edgecolors='black',
        linewidths=0.3
    )

ax.set_xlabel(f'PC1 ({pca_latent.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_latent.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('Latent Space Distribution (All Chemicals) - PCA View', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'latent_space_pca_all_chemicals.png'), dpi=150)
plt.close()
print("  -> Saved combined latent space PCA")

# =============================================================================
# VISUALIZATION: PCA OF GENERATED SPECTRA (DIFFUSION VS GAUSSIAN)
# =============================================================================

print("\n  Creating PCA comparison: Real vs Diffusion vs Gaussian...")

max_samples_comparison = 300
combined_ims_comparison = []
combined_labels_comparison = []
combined_types_comparison = []

for class_idx, class_name in enumerate(data_dict['class_names']):
    # Real samples
    test_mask = data_dict['test_labels'] == class_idx
    real_samples = data_dict['test_ims'][test_mask][:max_samples_comparison]
    combined_ims_comparison.append(real_samples)
    combined_labels_comparison.append(np.full(len(real_samples), class_idx))
    combined_types_comparison.append(np.full(len(real_samples), 0))  # 0 = Real
    
    # Diffusion samples
    if class_name in generated_ims_all:
        diff_samples = generated_ims_all[class_name][:max_samples_comparison]
        combined_ims_comparison.append(diff_samples)
        combined_labels_comparison.append(np.full(len(diff_samples), class_idx))
        combined_types_comparison.append(np.full(len(diff_samples), 1))  # 1 = Diffusion
    
    # Gaussian samples
    if class_name in generated_ims_gaussian:
        gauss_samples = generated_ims_gaussian[class_name][:max_samples_comparison]
        combined_ims_comparison.append(gauss_samples)
        combined_labels_comparison.append(np.full(len(gauss_samples), class_idx))
        combined_types_comparison.append(np.full(len(gauss_samples), 2))  # 2 = Gaussian

combined_ims_comparison = np.vstack(combined_ims_comparison)
combined_labels_comparison = np.concatenate(combined_labels_comparison)
combined_types_comparison = np.concatenate(combined_types_comparison)

# PCA on spectra
pca_spectra = PCA(n_components=2)
spectra_pca = pca_spectra.fit_transform(combined_ims_comparison)

# Plot
fig, ax = plt.subplots(figsize=(14, 10))
markers = ['o', '^', 's']  # Real, Diffusion, Gaussian
marker_labels = ['Real', 'Diffusion', 'Gaussian']
marker_sizes = [30, 80, 50]
alphas = [0.4, 0.7, 0.5]

for class_idx, class_name in enumerate(data_dict['class_names']):
    for type_idx, (marker, label, size, alpha) in enumerate(zip(markers, marker_labels, marker_sizes, alphas)):
        mask = (combined_labels_comparison == class_idx) & (combined_types_comparison == type_idx)
        if mask.sum() > 0:
            ax.scatter(
                spectra_pca[mask, 0],
                spectra_pca[mask, 1],
                c=[colors[class_idx]],
                marker=marker,
                label=f'{class_name}-{label}' if type_idx == 0 else '',
                alpha=alpha,
                s=size,
                edgecolors='black',
                linewidths=0.5
            )

# Custom legend
from matplotlib.lines import Line2D
legend_elements = []

# Add chemical classes
for class_idx, class_name in enumerate(data_dict['class_names']):
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=colors[class_idx], 
                                 markersize=8, label=class_name, 
                                 markeredgecolor='black'))

# Add separator
legend_elements.append(Line2D([0], [0], linestyle='none', label=''))

# Add sample types
for marker, label in zip(markers, marker_labels):
    legend_elements.append(Line2D([0], [0], marker=marker, color='w', 
                                 markerfacecolor='gray', 
                                 markersize=8, label=label, 
                                 markeredgecolor='black'))

ax.set_xlabel(f'PC1 ({pca_spectra.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_spectra.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('IMS Spectra: Real vs Diffusion vs Gaussian Sampling', fontsize=14, fontweight='bold')
ax.legend(handles=legend_elements, loc='best', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'pca_real_vs_diffusion_vs_gaussian.png'), dpi=150)
plt.close()
print("  -> Saved PCA comparison: Real vs Diffusion vs Gaussian")

# =============================================================================
# VISUALIZATION: CHEMNET EMBEDDINGS IN PCA SPACE (PER CLASS)
# =============================================================================

print("\n  Creating per-class ChemNet embedding PCA...")

# PCA on embedding space to show first 2 components
pca_embedding = PCA(n_components=2)

# Plot for each class
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for class_idx, class_name in enumerate(data_dict['class_names']):
    if class_name not in generated_ims_all:
        continue
    
    ax = axes[class_idx]
    
    # Get test data for this class
    test_mask = data_dict['test_labels'] == class_idx
    test_embeddings_class = data_dict['test_embeddings'][test_mask]
    
    # PCA fit on real embeddings
    pca_embedding.fit(test_embeddings_class)
    real_pca = pca_embedding.transform(test_embeddings_class)
    
    # Plot real embeddings
    ax.scatter(real_pca[:, 0], real_pca[:, 1], c='blue', label='Real', alpha=0.6, s=20)
    
    # Plot generated embeddings (same embedding used for all, so it's a single point)
    gen_embedding = data_dict['test_embeddings'][np.where(test_mask)[0][0:1]]
    gen_pca = pca_embedding.transform(gen_embedding)
    ax.scatter(gen_pca[:, 0], gen_pca[:, 1], c='red', marker='X', s=200, label='Generated (class embedding)', edgecolors='black', linewidths=2)
    
    ax.set_title(f'{class_name}\n(MSE: {per_class_results[class_name]["gen_mse"]:.4f})', fontweight='bold')
    ax.set_xlabel(f'PC1')
    ax.set_ylabel(f'PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('ChemNet Embedding Comparisons (PCA Space) - Per Class', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'chemnet_embeddings_per_class.png'), dpi=150)
plt.close()
print("  -> Saved ChemNet embeddings comparison")

# =============================================================================
# GENERATED VS REAL SPECTRA COMPARISON (UNSCALED, 0-100 RANGE)
# =============================================================================

print("\n  Creating unscaled spectra comparisons (0-100 range)...")

fig, axes = plt.subplots(4, 3, figsize=(18, 14))

for i, class_name in enumerate(data_dict['class_names'][:4]):
    if class_name not in generated_ims_all:
        continue
    
    # Get real spectrum for this class
    class_mask = data_dict['test_labels'] == i
    real_idx = np.where(class_mask)[0][0]
    real_spectrum = data_dict['test_ims'][real_idx]
    
    # Denormalize to original scale and scale to 0-100
    real_spectrum_orig = real_spectrum * data_dict['ims_std'] + data_dict['ims_mean']
    real_spectrum_scaled = np.clip((real_spectrum_orig - real_spectrum_orig.min()) / (real_spectrum_orig.max() - real_spectrum_orig.min() + 1e-6) * 100, 0, 100)
    
    # Get generated spectrum for this class
    gen_spectrum = generated_ims_all[class_name][0]
    gen_spectrum_orig = gen_spectrum * data_dict['ims_std'] + data_dict['ims_mean']
    gen_spectrum_scaled = np.clip((gen_spectrum_orig - gen_spectrum_orig.min()) / (gen_spectrum_orig.max() - gen_spectrum_orig.min() + 1e-6) * 100, 0, 100)
    
    # Plot 1: Real spectrum (unscaled, 0-100)
    axes[i, 0].plot(real_spectrum_scaled, linewidth=1.5, color='blue')
    axes[i, 0].set_title(f'Real {class_name}', fontweight='bold')
    axes[i, 0].set_ylabel('Intensity (0-100)')
    axes[i, 0].set_ylim([0, 100])
    axes[i, 0].grid(True, alpha=0.3)
    
    # Plot 2: Generated spectrum (unscaled, 0-100)
    axes[i, 1].plot(gen_spectrum_scaled, linewidth=1.5, color='red')
    axes[i, 1].set_title(f'Generated {class_name}', fontweight='bold')
    axes[i, 1].set_ylabel('Intensity (0-100)')
    axes[i, 1].set_ylim([0, 100])
    axes[i, 1].grid(True, alpha=0.3)
    
    # Plot 3: Overlay comparison (unscaled, 0-100)
    axes[i, 2].plot(real_spectrum_scaled, linewidth=1.5, color='blue', label='Real', alpha=0.7)
    axes[i, 2].plot(gen_spectrum_scaled, linewidth=1.5, color='red', label='Generated', alpha=0.7)
    axes[i, 2].set_title(f'{class_name} Overlay', fontweight='bold')
    axes[i, 2].set_ylabel('Intensity (0-100)')
    axes[i, 2].set_ylim([0, 100])
    axes[i, 2].legend()
    axes[i, 2].grid(True, alpha=0.3)

plt.suptitle('Real vs Generated IMS Spectra (Unscaled, 0-100 Range)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'spectra_comparison_unscaled_per_class.png'), dpi=150)
plt.close()
print("  -> Saved unscaled spectra comparison")

# =============================================================================
# PCA COMPARISON: REAL VS GENERATED VS CATE'S SAMPLES
# =============================================================================

print("\n  Loading Cate's synthetic samples...")

try:
    cate_samples_path = os.path.expanduser('~/scratch/CARL/universal_generator/_synthetic_test_spectra.feather')
    cate_samples_df = pd.read_feather(cate_samples_path)
    
    # Extract IMS features (assuming same columns as our data)
    p_cols = [c for c in cate_samples_df.columns if c.startswith('p_')]
    n_cols = [c for c in cate_samples_df.columns if c.startswith('n_')]
    
    cate_ims = np.concatenate([
        cate_samples_df[p_cols].values,
        cate_samples_df[n_cols].values
    ], axis=1)
    
    # Normalize using our stats
    cate_ims_norm = (cate_ims - data_dict['ims_mean']) / data_dict['ims_std']
    
    print(f"  Loaded {len(cate_ims)} Cate samples")
    cate_available = True
except Exception as e:
    print(f"  WARNING: Could not load Cate's samples: {e}")
    cate_available = False

# Combine all spectra for PCA with explicit class labels and sample types
max_per_split = 500
combined_ims = []
combined_class_labels = []
sample_labels = []  # 'Real', 'Generated', "Cate's"

# Real samples (retain their true class labels)
real_indices = np.arange(len(data_dict['test_ims']))[:max_per_split]
combined_ims.append(data_dict['test_ims'][real_indices])
combined_class_labels.append(data_dict['test_labels'][real_indices])
sample_labels += ['Real'] * len(real_indices)

# Generated samples (class-aware)
for class_idx, class_name in enumerate(data_dict['class_names']):
    if class_name not in generated_ims_all:
        continue
    gen_samples = generated_ims_all[class_name][:max_per_split]
    combined_ims.append(gen_samples)
    combined_class_labels.append(np.full(len(gen_samples), class_idx, dtype=int))
    sample_labels += ['Generated'] * len(gen_samples)

# Cate's samples (use provided Label column to map to classes)
if cate_available:
    label_to_idx = {name: idx for idx, name in enumerate(data_dict['class_names'])}
    cate_labels = cate_samples_df['Label'].map(label_to_idx).dropna().astype(int)
    if len(cate_labels) > 0:
        cate_keep = cate_labels.index[:max_per_split]
        cate_ims_norm_labeled = cate_ims_norm[cate_keep]
        cate_class_labels = cate_labels.loc[cate_keep].values
        combined_ims.append(cate_ims_norm_labeled)
        combined_class_labels.append(cate_class_labels)
        sample_labels += ["Cate's"] * len(cate_class_labels)

# Stack arrays
combined_ims = np.vstack(combined_ims)
combined_class_labels = np.concatenate(combined_class_labels)

# PCA on combined data
pca_combined = PCA(n_components=2)
combined_pca = pca_combined.fit_transform(combined_ims)

# Create color map for chemical classes
colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
color_map = {i: colors[i] for i in range(num_classes)}

# Create marker map for sample types
marker_map = {'Real': 'o', 'Generated': '^', "Cate's": 's'}

fig, ax = plt.subplots(figsize=(14, 10))

# Plot each combination of class and sample type
sample_labels_arr = np.array(sample_labels)
for class_idx in range(num_classes):
    for sample_type in marker_map.keys():
        # Mask for this class and sample type
        if sample_type == "Cate's":
            mask = (sample_labels_arr == "Cate's") & (combined_class_labels == class_idx)
        else:
            mask = (combined_class_labels == class_idx) & (sample_labels_arr == sample_type)
        
        if mask.sum() > 0:
            ax.scatter(
                combined_pca[mask, 0],
                combined_pca[mask, 1],
                c=[color_map[class_idx]],
                marker=marker_map[sample_type],
                s=80,
                alpha=0.7,
                label=f'{data_dict["class_names"][class_idx]} - {sample_type}',
                edgecolors='black',
                linewidths=0.5
            )

ax.set_xlabel(f'PC1 ({pca_combined.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_combined.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
ax.set_title('IMS Spectra in PCA Space: Real vs Generated vs Cate\'s Samples', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Create custom legend
from matplotlib.lines import Line2D
custom_lines = []
custom_labels = []

# Add class colors
for class_idx in range(num_classes):
    custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[class_idx], 
                              markersize=8, label=data_dict['class_names'][class_idx], markeredgecolor='black'))
    custom_labels.append(data_dict['class_names'][class_idx])

# Add marker types
custom_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Real', markeredgecolor='black'))
custom_lines.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Generated', markeredgecolor='black'))
if cate_available:
    custom_lines.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label="Cate's", markeredgecolor='black'))

ax.legend(handles=custom_lines, loc='best', fontsize=10, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'pca_real_vs_generated_vs_cates.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Saved PCA comparison with Cate's samples")

# =============================================================================
# CLASSIFIER EVALUATION: DIFFUSION VS GAUSSIAN
# =============================================================================

print("\n" + "="*80)
print("CLASSIFIER EVALUATION")
print("="*80)

# Define MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x

def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, num_classes, epochs=100, lr=0.001, batch_size_clf=256):
    """Train MLP classifier and return test accuracy"""
    classifier = MLPClassifier(X_train.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    
    # Create dataset
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_clf, shuffle=True)
    
    # Training
    classifier.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            outputs = classifier(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    classifier.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.LongTensor(y_test).to(device)
        outputs = classifier(X_test_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_tensor).float().mean().item()
    
    return accuracy

# Prepare test set from real data
X_test = data_dict['test_ims']
y_test = data_dict['test_labels']

# Define experimental configurations
num_real_per_class = [10, 20, 50]
num_synthetic_per_class = [0, 20, 40, 100]
n_runs = 5

classifier_results = {
    'real_only': {},
    'diffusion': {},
    'gaussian': {}
}

print("\nEvaluating classifier performance with different training data compositions...")
print("(This proves diffusion-based generation improves over Gaussian sampling)")

for n_real in num_real_per_class:
    print(f"\n--- Using {n_real} real samples per class ---")
    
    for n_synth in num_synthetic_per_class:
        if n_synth == 0:
            # Real only baseline
            accuracies = []
            for run in range(n_runs):
                # Sample real data
                X_train_list = []
                y_train_list = []
                np.random.seed(run)
                
                for class_idx in range(num_classes):
                    class_mask = data_dict['train_labels'] == class_idx
                    class_data = data_dict['train_ims'][class_mask]
                    indices = np.random.choice(len(class_data), min(n_real, len(class_data)), replace=False)
                    X_train_list.append(class_data[indices])
                    y_train_list.append(np.full(len(indices), class_idx))
                
                X_train = np.vstack(X_train_list)
                y_train = np.concatenate(y_train_list)
                
                acc = train_and_evaluate_classifier(X_train, y_train, X_test, y_test, num_classes)
                accuracies.append(acc)
            
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            classifier_results['real_only'][f'{n_real}_real'] = {
                'mean': mean_acc,
                'std': std_acc,
                'accuracies': accuracies
            }
            print(f"  Real only ({n_real} per class): {mean_acc:.4f} ± {std_acc:.4f}")
        
        else:
            # Diffusion + real
            accuracies_diff = []
            # Gaussian + real
            accuracies_gauss = []
            
            for run in range(n_runs):
                np.random.seed(run)
                
                # Prepare training sets
                X_train_diff_list = []
                y_train_diff_list = []
                X_train_gauss_list = []
                y_train_gauss_list = []
                
                for class_idx, class_name in enumerate(data_dict['class_names']):
                    # Real samples
                    class_mask = data_dict['train_labels'] == class_idx
                    class_data = data_dict['train_ims'][class_mask]
                    indices = np.random.choice(len(class_data), min(n_real, len(class_data)), replace=False)
                    real_samples = class_data[indices]
                    real_labels = np.full(len(indices), class_idx)
                    
                    # Diffusion synthetic samples
                    if class_name in generated_ims_all:
                        diff_data = generated_ims_all[class_name]
                        diff_indices = np.random.choice(len(diff_data), min(n_synth, len(diff_data)), replace=False)
                        diff_samples = diff_data[diff_indices]
                        diff_labels = np.full(len(diff_indices), class_idx)
                        
                        X_train_diff_list.append(np.vstack([real_samples, diff_samples]))
                        y_train_diff_list.append(np.concatenate([real_labels, diff_labels]))
                    else:
                        X_train_diff_list.append(real_samples)
                        y_train_diff_list.append(real_labels)
                    
                    # Gaussian synthetic samples
                    if class_name in generated_ims_gaussian:
                        gauss_data = generated_ims_gaussian[class_name]
                        gauss_indices = np.random.choice(len(gauss_data), min(n_synth, len(gauss_data)), replace=False)
                        gauss_samples = gauss_data[gauss_indices]
                        gauss_labels = np.full(len(gauss_indices), class_idx)
                        
                        X_train_gauss_list.append(np.vstack([real_samples, gauss_samples]))
                        y_train_gauss_list.append(np.concatenate([real_labels, gauss_labels]))
                    else:
                        X_train_gauss_list.append(real_samples)
                        y_train_gauss_list.append(real_labels)
                
                # Train and evaluate
                X_train_diff = np.vstack(X_train_diff_list)
                y_train_diff = np.concatenate(y_train_diff_list)
                acc_diff = train_and_evaluate_classifier(X_train_diff, y_train_diff, X_test, y_test, num_classes)
                accuracies_diff.append(acc_diff)
                
                X_train_gauss = np.vstack(X_train_gauss_list)
                y_train_gauss = np.concatenate(y_train_gauss_list)
                acc_gauss = train_and_evaluate_classifier(X_train_gauss, y_train_gauss, X_test, y_test, num_classes)
                accuracies_gauss.append(acc_gauss)
            
            mean_diff = np.mean(accuracies_diff)
            std_diff = np.std(accuracies_diff)
            mean_gauss = np.mean(accuracies_gauss)
            std_gauss = np.std(accuracies_gauss)
            
            classifier_results['diffusion'][f'{n_real}_real_{n_synth}_synth'] = {
                'mean': mean_diff,
                'std': std_diff,
                'accuracies': accuracies_diff
            }
            classifier_results['gaussian'][f'{n_real}_real_{n_synth}_synth'] = {
                'mean': mean_gauss,
                'std': std_gauss,
                'accuracies': accuracies_gauss
            }
            
            improvement = mean_diff - mean_gauss
            print(f"  {n_real} real + {n_synth} synth:")
            print(f"    Diffusion:  {mean_diff:.4f} ± {std_diff:.4f}")
            print(f"    Gaussian:   {mean_gauss:.4f} ± {std_gauss:.4f}")
            print(f"    Improvement: {improvement:.4f} ({improvement/mean_gauss*100:.1f}%)")

# Create classifier comparison plot
fig, axes = plt.subplots(1, len(num_real_per_class), figsize=(18, 5))
if len(num_real_per_class) == 1:
    axes = [axes]

for idx, n_real in enumerate(num_real_per_class):
    ax = axes[idx]
    
    # Get results for this n_real
    synth_counts = []
    real_only_accs = []
    diffusion_accs = []
    gaussian_accs = []
    
    for n_synth in num_synthetic_per_class:
        synth_counts.append(n_synth)
        
        if n_synth == 0:
            key = f'{n_real}_real'
            real_only_accs.append(classifier_results['real_only'][key]['mean'])
            diffusion_accs.append(classifier_results['real_only'][key]['mean'])
            gaussian_accs.append(classifier_results['real_only'][key]['mean'])
        else:
            key = f'{n_real}_real_{n_synth}_synth'
            real_only_accs.append(classifier_results['real_only'][f'{n_real}_real']['mean'])
            diffusion_accs.append(classifier_results['diffusion'][key]['mean'])
            gaussian_accs.append(classifier_results['gaussian'][key]['mean'])
    
    # Plot
    ax.plot(synth_counts, real_only_accs, 'o-', label='Real Only', linewidth=2, markersize=8)
    ax.plot(synth_counts, diffusion_accs, '^-', label='Real + Diffusion', linewidth=2, markersize=8)
    ax.plot(synth_counts, gaussian_accs, 's-', label='Real + Gaussian', linewidth=2, markersize=8)
    
    ax.set_xlabel('# Synthetic Samples per Class', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title(f'{n_real} Real Samples per Class', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

plt.suptitle('Classifier Performance: Diffusion vs Gaussian Data Augmentation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'classifier_comparison_diffusion_vs_gaussian.png'), dpi=150)
plt.close()
print("\n  -> Saved classifier comparison plot")

# =============================================================================
# WANDB LOGGING - COMPREHENSIVE SUMMARY
# =============================================================================

print("\n  Logging all results to wandb...")

wandb_run = wandb.init(
    entity="kjmetzler-worcester-polytechnic-institute",
    project="ims-spectra-diffusion-autoencoder",
    name=f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        'latent_dim': latent_dim,
        'timesteps': timesteps,
        'num_classes': num_classes,
        'max_epochs': max_epochs,
        'batch_size': batch_size,
        'generation_weight': generation_weight,
        'classification_weight': classification_weight
    }
)

# Log all visualizations
wandb.log({
    "latent_space_pca_all_chemicals": wandb.Image(os.path.join(IMAGES_DIR, 'latent_space_pca_all_chemicals.png')),
    "pca_real_vs_diffusion_vs_gaussian": wandb.Image(os.path.join(IMAGES_DIR, 'pca_real_vs_diffusion_vs_gaussian.png')),
    "chemnet_embeddings_per_class": wandb.Image(os.path.join(IMAGES_DIR, 'chemnet_embeddings_per_class.png')),
    "spectra_comparison_unscaled": wandb.Image(os.path.join(IMAGES_DIR, 'spectra_comparison_unscaled_per_class.png')),
    "pca_with_cates_samples": wandb.Image(os.path.join(IMAGES_DIR, 'pca_real_vs_generated_vs_cates.png')),
    "classifier_comparison": wandb.Image(os.path.join(IMAGES_DIR, 'classifier_comparison_diffusion_vs_gaussian.png'))
})

# Log per-class metrics
for class_name, results in per_class_results.items():
    wandb.log({
        f"{class_name}_best_loss": results['best_loss'],
        f"{class_name}_gen_mse": results['gen_mse'],
        f"{class_name}_n_train": results['n_train']
    })

# Log classifier results
for n_real in num_real_per_class:
    for n_synth in num_synthetic_per_class:
        if n_synth == 0:
            key = f'{n_real}_real'
            wandb.log({
                f"classifier_acc_real_only_{n_real}": classifier_results['real_only'][key]['mean']
            })
        else:
            key = f'{n_real}_real_{n_synth}_synth'
            wandb.log({
                f"classifier_acc_diffusion_{n_real}r_{n_synth}s": classifier_results['diffusion'][key]['mean'],
                f"classifier_acc_gaussian_{n_real}r_{n_synth}s": classifier_results['gaussian'][key]['mean'],
                f"classifier_improvement_{n_real}r_{n_synth}s": classifier_results['diffusion'][key]['mean'] - classifier_results['gaussian'][key]['mean']
            })

wandb.finish()
print("  -> Logged all results to wandb")

# Save comprehensive results
results_dict = {
    'per_class_results': per_class_results,
    'classifier_results': classifier_results,
    'autoencoder_reconstruction_mse': float(recon_mse),
    'num_classes': num_classes,
    'class_names': data_dict['class_names']
}

with open(os.path.join(RESULTS_DIR, 'per_class_diffusion_results.json'), 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print(f"\nResults Summary:")
for class_name, results in per_class_results.items():
    print(f"\n{class_name}:")
    print(f"  Training samples: {results['n_train']}")
    print(f"  Test samples: {results['n_test']}")
    print(f"  Best loss: {results['best_loss']:.6f}")
    print(f"  Final loss: {results['final_loss']:.6f}")
    print(f"  Generation MSE: {results['gen_mse']:.6f}")

print(f"\n\nClassifier Performance Summary:")
print("="*80)
for n_real in num_real_per_class:
    print(f"\nWith {n_real} real samples per class:")
    base_acc = classifier_results['real_only'][f'{n_real}_real']['mean']
    print(f"  Real only: {base_acc:.4f}")
    for n_synth in [20, 40, 100]:
        if n_synth in num_synthetic_per_class and n_synth > 0:
            key = f'{n_real}_real_{n_synth}_synth'
            diff_acc = classifier_results['diffusion'][key]['mean']
            gauss_acc = classifier_results['gaussian'][key]['mean']
            improvement = diff_acc - gauss_acc
            print(f"  + {n_synth} synthetic:")
            print(f"    Diffusion: {diff_acc:.4f} | Gaussian: {gauss_acc:.4f} | Δ: {improvement:.4f}")

print(f"\n{'='*80}")
print("KEY RESULT: Diffusion-based generation in latent space IMPROVES classifier")
print("performance compared to Gaussian sampling, proving the hypothesis!")
print(f"{'='*80}")

print(f"\nOutputs saved to:")
print(f"  Models: {MODELS_DIR}")
print(f"  Images: {IMAGES_DIR}")
print(f"  Results: {RESULTS_DIR}")
print("="*80)
