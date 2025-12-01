"""
Bijection-based IMS Spectra Generation with Discrete Lattice
============================================================

This script implements an iterative fixed-point approach:
- X: Real IMS spectra (1676-dimensional)
- Y: Discrete lattice representation (same dimension, quantized to grid)
- ChemNet Embedding: 512-dimensional molecular embedding for conditioning

Key Concept: [X, Y, Embedding] is an ATTRACTING FIXED POINT
- From any starting point, iterate until convergence to the manifold
- The model learns to pull inputs toward the correct (X, Y) pair

Training:
1. Forward:  [X, 0, Emb] → iterate → [X, Y, Emb]  (Learn Y from X)
2. Inverse:  [0, Y, Emb] → iterate → [X, Y, Emb]  (Reconstruct X from Y)
3. Fixed-point: [X̃, Ỹ, Emb] → iterate → [X, Y, Emb]  (Attract from anywhere)

Generation:
- Pick a class, get its Y from the discrete lattice
- Start from [noise, Y, Emb] and iterate to fixed point
- Extract the converged X as the synthetic spectrum

Unlike diffusion: No noise schedule needed - just iterate until convergence!
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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
hidden_dim = 512
embedding_dim = 256
num_layers = 6
learning_rate = 1e-4
max_epochs = 1000
batch_size = 256
test_mode = '--test' in sys.argv

# Iterative fixed-point hyperparameters
num_iterations = 10  # Number of refinement steps during training
num_iterations_inference = 20  # More iterations at inference for better convergence
convergence_threshold = 1e-4  # Stop early if change is small

# Discrete lattice hyperparameters
lattice_levels = 256  # Number of discrete levels per dimension
lattice_min = -3.0  # Min value (after normalization)
lattice_max = 3.0   # Max value (after normalization)

# Loss weights
reconstruction_weight = 1.0
fixed_point_weight = 0.5  # Encourage convergence (output ≈ input when at fixed point)
lattice_consistency_weight = 0.1  # Keep Y on the lattice

print(f"Hyperparameters:")
print(f"  Hidden dimension: {hidden_dim}")
print(f"  Embedding dimension: {embedding_dim}")
print(f"  Number of layers: {num_layers}")
print(f"  Learning rate: {learning_rate}")
print(f"  Max epochs: {max_epochs}")
print(f"  Batch size: {batch_size}")
print(f"  Training iterations: {num_iterations}")
print(f"  Inference iterations: {num_iterations_inference}")
print(f"  Lattice levels: {lattice_levels}")
print(f"  Lattice range: [{lattice_min}, {lattice_max}]")
print(f"  Test mode: {test_mode}")

# =============================================================================
# DISCRETE LATTICE UTILITIES
# =============================================================================
class DiscreteLattice:
    """
    Discrete lattice for Y-space quantization.
    Maps continuous values to nearest lattice points.
    """
    
    def __init__(self, num_levels=256, min_val=-3.0, max_val=3.0):
        self.num_levels = num_levels
        self.min_val = min_val
        self.max_val = max_val
        self.step = (max_val - min_val) / (num_levels - 1)
        
        # Create lattice points
        self.lattice_points = torch.linspace(min_val, max_val, num_levels)
        
    def quantize(self, x):
        """Quantize continuous values to nearest lattice points"""
        # Clamp to range
        x_clamped = torch.clamp(x, self.min_val, self.max_val)
        # Find nearest lattice index
        indices = torch.round((x_clamped - self.min_val) / self.step).long()
        indices = torch.clamp(indices, 0, self.num_levels - 1)
        # Return quantized values
        return self.min_val + indices.float() * self.step
    
    def quantize_soft(self, x, temperature=0.1):
        """
        Soft quantization for training (differentiable).
        Uses straight-through estimator: quantize in forward, pass gradient in backward.
        """
        quantized = self.quantize(x)
        # Straight-through: use quantized value but pass gradient through x
        return x + (quantized - x).detach()
    
    def get_lattice_points(self, device='cpu'):
        """Get all lattice points as tensor"""
        return self.lattice_points.to(device)
    
    def random_lattice_point(self, shape, device='cpu'):
        """Sample random points from the lattice"""
        indices = torch.randint(0, self.num_levels, shape, device=device)
        return self.min_val + indices.float() * self.step


# Initialize global lattice
lattice = DiscreteLattice(lattice_levels, lattice_min, lattice_max)

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
    
    print(f"\nNormalized IMS statistics:")
    print(f"  Train mean: {train_ims_norm.mean():.4f}, std: {train_ims_norm.std():.4f}")
    print(f"  Train range: [{train_ims_norm.min():.2f}, {train_ims_norm.max():.2f}]")
    
    return {
        'train_ims': train_ims_norm,
        'train_embeddings': train_embeddings,
        'train_labels': train_labels,
        'test_ims': test_ims_norm,
        'test_embeddings': test_embeddings,
        'test_labels': test_labels,
        'ims_mean': ims_mean,
        'ims_std': ims_std,
        'class_names': onehot_cols,
        'num_ims_features': len(p_cols) + len(n_cols),
        'num_embedding_dim': smile_embedding_dim,
        'smile_embeddings': smile_embeddings
    }


# =============================================================================
# FIXED-POINT ITERATION NETWORK
# =============================================================================
class FixedPointNetwork(nn.Module):
    """
    Network that learns to map any input toward the fixed point [X, Y, Emb].
    
    Each forward pass is ONE iteration step.
    The network predicts a RESIDUAL/UPDATE to move closer to the fixed point.
    
    At the fixed point: f([X, Y, Emb]) = [X, Y, Emb] (identity)
    Away from fixed point: f pulls the input toward [X, Y, Emb]
    """
    
    def __init__(self, ims_dim, embedding_dim=512, hidden_dim=512, num_layers=6):
        super().__init__()
        self.ims_dim = ims_dim
        self.embedding_dim = embedding_dim
        
        # Input: [X_current, Y_current] with embedding as conditioning
        input_dim = ims_dim * 2
        output_dim = ims_dim * 2  # Residual for [X, Y]
        
        # Embedding projection (conditioning)
        self.embed_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection
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
        
        # Output projection - predicts RESIDUAL (update direction)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Bound the update magnitude
        )
        
        # Learnable step size (how much to move toward fixed point per iteration)
        self.step_size = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x_current, y_current, embedding):
        """
        One iteration step toward the fixed point.
        
        Args:
            x_current: [batch, ims_dim] - Current X estimate
            y_current: [batch, ims_dim] - Current Y (on lattice)
            embedding: [batch, embedding_dim] - ChemNet/SMILE embedding
        
        Returns:
            x_next: [batch, ims_dim] - Updated X
            y_next: [batch, ims_dim] - Updated Y
            residual_norm: scalar - Magnitude of update (for convergence check)
        """
        batch_size = x_current.shape[0]
        
        # Combine X and Y
        xy_current = torch.cat([x_current, y_current], dim=1)
        h = self.input_proj(xy_current)
        
        # Add embedding conditioning
        emb_h = self.embed_proj(embedding)
        h = h + emb_h
        
        # Process through layers with residual connections
        for layer, norm in zip(self.layers, self.layer_norms):
            h = h + layer(norm(h))
        
        # Predict residual (update direction)
        residual = self.output_proj(h)
        
        # Split residual into X and Y components
        residual_x = residual[:, :self.ims_dim]
        residual_y = residual[:, self.ims_dim:]
        
        # Apply update with learnable step size
        step = torch.sigmoid(self.step_size)  # Keep step size in (0, 1)
        x_next = x_current + step * residual_x
        y_next = y_current + step * residual_y
        
        # Compute residual norm for convergence checking
        residual_norm = torch.norm(residual, dim=1).mean()
        
        return x_next, y_next, residual_norm
    
    def iterate(self, x_init, y_init, embedding, num_steps, return_trajectory=False):
        """
        Run multiple iteration steps toward the fixed point.
        
        Args:
            x_init: Initial X (can be zeros or noise)
            y_init: Initial Y (should be on lattice)
            embedding: ChemNet embedding
            num_steps: Number of iterations
            return_trajectory: If True, return all intermediate states
        
        Returns:
            x_final, y_final: Converged values
            trajectory: (optional) List of (x, y) at each step
        """
        x_current = x_init
        y_current = y_init
        
        trajectory = [(x_current.clone(), y_current.clone())] if return_trajectory else None
        
        for step in range(num_steps):
            x_current, y_current, residual_norm = self.forward(x_current, y_current, embedding)
            
            if return_trajectory:
                trajectory.append((x_current.clone(), y_current.clone()))
            
            # Early stopping if converged
            if residual_norm < convergence_threshold:
                break
        
        if return_trajectory:
            return x_current, y_current, trajectory
        return x_current, y_current


# =============================================================================
# LATTICE Y ENCODER
# =============================================================================
class LatticeEncoder(nn.Module):
    """
    Learns to encode X into a discrete lattice Y.
    This creates the initial Y for each training sample.
    """
    
    def __init__(self, ims_dim, embedding_dim=512, hidden_dim=256):
        super().__init__()
        self.ims_dim = ims_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(ims_dim + embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, ims_dim),
            nn.Tanh()  # Output in [-1, 1], will be scaled to lattice range
        )
        
    def forward(self, x, embedding):
        """Encode X to continuous Y, then quantize to lattice"""
        combined = torch.cat([x, embedding], dim=1)
        y_continuous = self.encoder(combined)
        
        # Scale to lattice range
        y_scaled = y_continuous * (lattice_max - lattice_min) / 2 + (lattice_max + lattice_min) / 2
        
        # Quantize to discrete lattice (with straight-through gradient)
        y_quantized = lattice.quantize_soft(y_scaled)
        
        return y_quantized, y_continuous


# =============================================================================
# FID COMPUTATION
# =============================================================================
def calculate_fid_from_features(real_features, generated_features):
    """Calculate Frechet Inception Distance between real and generated features."""
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


# =============================================================================
# TRAINER
# =============================================================================
class FixedPointTrainer:
    """
    Trainer for the fixed-point iteration network.
    
    Training procedure:
    1. Encode X → Y (discrete lattice)
    2. Forward: [X, 0] → iterate → [X', Y'] should give [X, Y]
    3. Inverse: [0, Y] → iterate → [X', Y'] should give [X, Y]  
    4. Fixed-point: [X, Y] → iterate → [X', Y'] should stay at [X, Y]
    """
    
    def __init__(self, model, encoder, optimizer, device, ims_dim, embedding_dim,
                 num_iterations=10):
        self.model = model
        self.encoder = encoder
        self.optimizer = optimizer
        self.device = device
        self.ims_dim = ims_dim
        self.embedding_dim = embedding_dim
        self.num_iterations = num_iterations
        
        # Store learned Y values for each class (for generation)
        self.class_Y = {}
        
    def train_step(self, x_batch, embedding_batch, labels_batch):
        """
        Training step with multiple objectives.
        """
        self.model.train()
        self.encoder.train()
        self.optimizer.zero_grad()
        
        batch_size = x_batch.shape[0]
        zeros = torch.zeros_like(x_batch)
        
        total_loss = 0
        loss_dict = {
            'encode': 0, 
            'forward_x': 0, 
            'forward_y': 0,
            'inverse_x': 0, 
            'inverse_y': 0,
            'fixed_point': 0,
            'lattice': 0
        }
        
        # ====================================================================
        # STEP 1: Encode X → Y (learn the discrete lattice mapping)
        # ====================================================================
        y_target, y_continuous = self.encoder(x_batch, embedding_batch)
        
        # Lattice consistency loss (encourage Y to stay on lattice)
        y_quantized = lattice.quantize(y_continuous * (lattice_max - lattice_min) / 2 + (lattice_max + lattice_min) / 2)
        lattice_loss = F.mse_loss(y_continuous * (lattice_max - lattice_min) / 2 + (lattice_max + lattice_min) / 2, y_quantized)
        loss_dict['lattice'] = lattice_loss.item()
        total_loss += lattice_consistency_weight * lattice_loss
        
        # ====================================================================
        # STEP 2: Forward - [X, 0] → iterate → [X, Y]
        # Start with correct X, zeros for Y, should learn Y
        # ====================================================================
        x_fwd, y_fwd = self.model.iterate(x_batch, zeros, embedding_batch, self.num_iterations)
        
        forward_x_loss = F.mse_loss(x_fwd, x_batch)
        forward_y_loss = F.mse_loss(y_fwd, y_target)
        
        loss_dict['forward_x'] = forward_x_loss.item()
        loss_dict['forward_y'] = forward_y_loss.item()
        
        total_loss += reconstruction_weight * (forward_x_loss + forward_y_loss)
        
        # ====================================================================
        # STEP 3: Inverse - [0, Y] → iterate → [X, Y]
        # Start with zeros for X, correct Y, should reconstruct X
        # ====================================================================
        x_inv, y_inv = self.model.iterate(zeros, y_target.detach(), embedding_batch, self.num_iterations)
        
        inverse_x_loss = F.mse_loss(x_inv, x_batch)
        inverse_y_loss = F.mse_loss(y_inv, y_target.detach())
        
        loss_dict['inverse_x'] = inverse_x_loss.item()
        loss_dict['inverse_y'] = inverse_y_loss.item()
        
        total_loss += reconstruction_weight * (inverse_x_loss + 0.1 * inverse_y_loss)
        
        # ====================================================================
        # STEP 4: Fixed-point - [X, Y] → iterate → [X, Y]
        # At the fixed point, iteration should be identity (stay put)
        # ====================================================================
        x_fp, y_fp = self.model.iterate(x_batch, y_target.detach(), embedding_batch, self.num_iterations)
        
        fp_loss = F.mse_loss(x_fp, x_batch) + F.mse_loss(y_fp, y_target.detach())
        loss_dict['fixed_point'] = fp_loss.item()
        
        total_loss += fixed_point_weight * fp_loss
        
        # ====================================================================
        # STEP 5: Random start - [noise, Y] → iterate → [X, Y]
        # Test attraction from random X
        # ====================================================================
        noise = torch.randn_like(x_batch) * 0.5
        x_rand, y_rand = self.model.iterate(noise, y_target.detach(), embedding_batch, self.num_iterations)
        
        rand_loss = F.mse_loss(x_rand, x_batch)
        total_loss += 0.3 * rand_loss  # Lower weight for random start
        
        # Backprop
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.optimizer.step()
        
        return total_loss.item(), loss_dict, y_target.detach()
    
    def generate(self, y_samples, embeddings, num_iterations=None):
        """
        Generate X from Y samples using iterative fixed-point approach.
        """
        if num_iterations is None:
            num_iterations = num_iterations_inference
            
        self.model.eval()
        with torch.no_grad():
            # Start from zeros (or small noise) for X
            x_init = torch.zeros(y_samples.shape[0], self.ims_dim, device=self.device)
            
            # Iterate to fixed point
            x_generated, y_final = self.model.iterate(x_init, y_samples, embeddings, num_iterations)
            
        return x_generated
    
    def encode_to_lattice(self, x_batch, embeddings):
        """Encode X to discrete lattice Y"""
        self.encoder.eval()
        with torch.no_grad():
            y_quantized, _ = self.encoder(x_batch, embeddings)
        return y_quantized


# =============================================================================
# DATASET AND DATALOADER
# =============================================================================
class FixedPointDataset(torch.utils.data.Dataset):
    def __init__(self, ims_data, embeddings, labels):
        self.ims_data = torch.FloatTensor(ims_data)
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.ims_data)
    
    def __getitem__(self, idx):
        return self.ims_data[idx], self.embeddings[idx], self.labels[idx]


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
print(f"  Lattice: {lattice_levels} levels in [{lattice_min}, {lattice_max}]")

# Create dataset and loader
train_dataset = FixedPointDataset(
    data_dict['train_ims'],
    data_dict['train_embeddings'],
    data_dict['train_labels']
)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"\nDataset prepared:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Batches per epoch: {len(train_loader)}")

# =============================================================================
# TRAINING
# =============================================================================
print("\n" + "="*80)
print("STARTING FIXED-POINT BIJECTION TRAINING")
print("="*80)

# Initialize models
model = FixedPointNetwork(
    ims_dim=num_ims_features,
    embedding_dim=num_embedding_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers
).to(device)

encoder = LatticeEncoder(
    ims_dim=num_ims_features,
    embedding_dim=num_embedding_dim,
    hidden_dim=hidden_dim // 2
).to(device)

total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in encoder.parameters())
print(f"\nModel architecture:")
print(f"  Fixed-point network params: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Lattice encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
print(f"  Total parameters: {total_params:,}")

# Initialize optimizer (joint for both networks)
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(encoder.parameters()), 
    lr=learning_rate, 
    weight_decay=1e-5
)

# Initialize trainer
trainer = FixedPointTrainer(
    model, encoder, optimizer, device,
    num_ims_features, num_embedding_dim,
    num_iterations
)

# Initialize wandb
wandb_run = wandb.init(
    entity="kjmetzler-worcester-polytechnic-institute",
    project="ims-spectra-bijection",
    name=f"fixed_point_lattice_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "num_iterations": num_iterations,
        "num_iterations_inference": num_iterations_inference,
        "lattice_levels": lattice_levels,
        "lattice_range": [lattice_min, lattice_max],
        "reconstruction_weight": reconstruction_weight,
        "fixed_point_weight": fixed_point_weight,
        "num_ims_features": num_ims_features,
        "num_embedding_dim": num_embedding_dim,
        "total_params": total_params
    }
)

# Training loop
print(f"\nTraining for {max_epochs} epochs...")
loss_history = []
best_loss = float('inf')

for epoch in range(max_epochs):
    epoch_loss = 0
    epoch_losses = {k: 0 for k in ['forward_x', 'forward_y', 'inverse_x', 'inverse_y', 'fixed_point', 'lattice']}
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
    for batch_idx, (x_batch, emb_batch, label_batch) in enumerate(pbar):
        x_batch = x_batch.to(device)
        emb_batch = emb_batch.to(device)
        label_batch = label_batch.to(device)
        
        # Training step
        batch_loss, loss_dict, _ = trainer.train_step(x_batch, emb_batch, label_batch)
        
        epoch_loss += batch_loss
        for k in epoch_losses:
            epoch_losses[k] += loss_dict.get(k, 0)
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{batch_loss:.4f}",
            'fwd': f"{loss_dict['forward_x']:.4f}",
            'inv': f"{loss_dict['inverse_x']:.4f}",
            'fp': f"{loss_dict['fixed_point']:.4f}"
        })
    
    # Average losses
    avg_loss = epoch_loss / num_batches
    avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
    
    loss_history.append(avg_loss)
    
    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        **{f"loss_{k}": v for k, v in avg_losses.items()}
    })
    
    # Print epoch summary
    print(f"Epoch {epoch+1}/{max_epochs} - Loss: {avg_loss:.6f} | "
          f"Fwd_X: {avg_losses['forward_x']:.6f} | Inv_X: {avg_losses['inverse_x']:.6f} | "
          f"FP: {avg_losses['fixed_point']:.6f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'model_config': {
                'ims_dim': num_ims_features,
                'embedding_dim': num_embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'lattice_levels': lattice_levels,
                'lattice_range': [lattice_min, lattice_max]
            }
        }, os.path.join(MODELS_DIR, 'best_fixed_point_model.pth'))
        print(f"  -> Saved best model (loss: {best_loss:.6f})")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

# =============================================================================
# COMPUTE LATTICE Y FOR ALL TRAINING DATA
# =============================================================================
print("\nComputing discrete lattice Y for all training data...")

model.eval()
encoder.eval()
all_train_Y = []
all_train_labels = []

with torch.no_grad():
    for x_batch, emb_batch, label_batch in tqdm(train_loader, desc="Encoding to lattice"):
        x_batch = x_batch.to(device)
        emb_batch = emb_batch.to(device)
        
        y_batch = trainer.encode_to_lattice(x_batch, emb_batch)
        all_train_Y.append(y_batch.cpu().numpy())
        all_train_labels.append(label_batch.numpy())

all_train_Y = np.vstack(all_train_Y)
all_train_labels = np.concatenate(all_train_labels)

# Store per-class Y distributions (for generation)
class_Y_means = {}
class_Y_stds = {}
for c in range(num_classes):
    mask = all_train_labels == c
    if mask.sum() > 0:
        class_Y_means[c] = all_train_Y[mask].mean(axis=0)
        class_Y_stds[c] = all_train_Y[mask].std(axis=0)

print(f"Computed lattice Y for {len(all_train_Y)} samples")

# Verify Y is on lattice
y_on_lattice = lattice.quantize(torch.tensor(all_train_Y)).numpy()
lattice_error = np.abs(all_train_Y - y_on_lattice).mean()
print(f"Average distance from lattice: {lattice_error:.6f}")

# =============================================================================
# EVALUATION
# =============================================================================
print("\nEvaluating model...")

# Prepare test data
test_ims = data_dict['test_ims']
test_embeddings = data_dict['test_embeddings']
test_labels = data_dict['test_labels']

n_test = min(1000, len(test_ims))

# Test reconstruction (X → Y → X')
print("\nTesting reconstruction (X → Y, then [0, Y] → X')...")
with torch.no_grad():
    test_x = torch.FloatTensor(test_ims[:n_test]).to(device)
    test_emb = torch.FloatTensor(test_embeddings[:n_test]).to(device)
    
    # Encode to lattice
    test_y = trainer.encode_to_lattice(test_x, test_emb)
    
    # Reconstruct from Y
    x_recon = trainer.generate(test_y, test_emb, num_iterations_inference)
    
    recon_mse = F.mse_loss(x_recon, test_x).item()
    print(f"  Reconstruction MSE: {recon_mse:.6f}")

# Test generation (sample Y → X)
print("\nTesting generation (class Y → X)...")
with torch.no_grad():
    generated_ims_list = []
    
    for i in range(n_test):
        class_idx = test_labels[i]
        
        # Get class mean Y (quantized to lattice)
        y_mean = torch.FloatTensor(class_Y_means[class_idx]).unsqueeze(0).to(device)
        y_sample = lattice.quantize(y_mean + torch.randn_like(y_mean) * 0.1)
        
        emb_sample = torch.FloatTensor(test_embeddings[i:i+1]).to(device)
        
        x_gen = trainer.generate(y_sample, emb_sample, num_iterations_inference)
        generated_ims_list.append(x_gen.cpu().numpy())
    
    generated_ims = np.vstack(generated_ims_list)
    
    # Compute MSE
    gen_mse = mean_squared_error(test_ims[:n_test].flatten(), generated_ims.flatten())
    print(f"  Generation MSE: {gen_mse:.6f}")
    
    # Compute FID
    print("\n  Computing FID score...")
    pca_for_fid = PCA(n_components=min(50, n_test, num_ims_features))
    real_pca_features = pca_for_fid.fit_transform(test_ims[:n_test])
    gen_pca_features = pca_for_fid.transform(generated_ims)
    fid_score = calculate_fid_from_features(real_pca_features, gen_pca_features)
    print(f"  FID Score: {fid_score:.4f}")

# Test convergence visualization
print("\nTesting convergence trajectory...")
with torch.no_grad():
    # Pick one sample
    sample_x = torch.FloatTensor(test_ims[0:1]).to(device)
    sample_emb = torch.FloatTensor(test_embeddings[0:1]).to(device)
    sample_y = trainer.encode_to_lattice(sample_x, sample_emb)
    
    # Start from zeros and track trajectory
    x_init = torch.zeros_like(sample_x)
    x_traj, y_traj, trajectory = model.iterate(x_init, sample_y, sample_emb, 
                                                num_iterations_inference, return_trajectory=True)
    
    # Compute MSE at each step
    convergence_curve = []
    for x_t, y_t in trajectory:
        mse_t = F.mse_loss(x_t, sample_x).item()
        convergence_curve.append(mse_t)
    
    print(f"  Convergence: MSE {convergence_curve[0]:.4f} → {convergence_curve[-1]:.4f}")

# Per-class quality
print("\nTesting per-class quality...")
class_mses = []
for class_idx in range(num_classes):
    class_mask = test_labels[:n_test] == class_idx
    if class_mask.sum() > 0:
        class_real = test_ims[:n_test][class_mask]
        class_gen = generated_ims[class_mask]
        class_mse = mean_squared_error(class_real.flatten(), class_gen.flatten())
        class_mses.append(class_mse)
        print(f"  {data_dict['class_names'][class_idx]}: MSE = {class_mse:.6f}")
    else:
        class_mses.append(0)

# Log final metrics
wandb.log({
    "test_reconstruction_mse": recon_mse,
    "test_generation_mse": gen_mse,
    "test_fid_score": fid_score
})

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\nGenerating visualizations...")

# 1. Loss curves
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Fixed-Point Bijection Training Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fixed_point_training_loss.png'), dpi=150)
plt.close()
print("  -> Saved training loss plot")

# 2. Convergence curve
plt.figure(figsize=(10, 6))
plt.plot(convergence_curve, 'b-o', markersize=4)
plt.xlabel('Iteration')
plt.ylabel('MSE to Target')
plt.title('Fixed-Point Convergence (Starting from Zeros)')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fixed_point_convergence.png'), dpi=150)
wandb.log({"convergence_curve": wandb.Image(os.path.join(IMAGES_DIR, 'fixed_point_convergence.png'))})
plt.close()
print("  -> Saved convergence plot")

# 3. Generated vs Real Spectra
fig, axes = plt.subplots(4, 2, figsize=(15, 12))
for i in range(4):
    axes[i, 0].plot(test_ims[i])
    axes[i, 0].set_title(f'Real Spectrum {i+1} ({data_dict["class_names"][test_labels[i]]})')
    axes[i, 0].set_xlabel('Feature Index')
    axes[i, 0].set_ylabel('Intensity')
    axes[i, 0].grid(True, alpha=0.3)
    
    axes[i, 1].plot(generated_ims[i])
    axes[i, 1].set_title(f'Generated Spectrum {i+1} (Fixed-Point)')
    axes[i, 1].set_xlabel('Feature Index')
    axes[i, 1].set_ylabel('Intensity')
    axes[i, 1].grid(True, alpha=0.3)

plt.suptitle('Real vs Generated IMS Spectra (Fixed-Point Bijection)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fixed_point_spectra_comparison.png'), dpi=150)
wandb.log({"spectra_comparison": wandb.Image(os.path.join(IMAGES_DIR, 'fixed_point_spectra_comparison.png'))})
plt.close()
print("  -> Saved spectra comparison")

# 4. Lattice Y visualization (PCA)
pca_y = PCA(n_components=2)
y_pca = pca_y.fit_transform(all_train_Y[:5000])

plt.figure(figsize=(10, 8))
scatter = plt.scatter(y_pca[:, 0], y_pca[:, 1], c=all_train_labels[:5000], 
                      cmap='tab10', alpha=0.5, s=10)
plt.colorbar(scatter, label='Chemical Class')
plt.xlabel(f'PC1 ({pca_y.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_y.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Discrete Lattice Y Space (PCA)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fixed_point_lattice_space.png'), dpi=150)
wandb.log({"lattice_space": wandb.Image(os.path.join(IMAGES_DIR, 'fixed_point_lattice_space.png'))})
plt.close()
print("  -> Saved lattice space visualization")

# 5. Per-class quality
plt.figure(figsize=(10, 6))
plt.bar(data_dict['class_names'], class_mses)
plt.xlabel('Chemical Class')
plt.ylabel('MSE')
plt.title('Generation Quality by Chemical Class (Fixed-Point)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fixed_point_per_class_quality.png'), dpi=150)
wandb.log({"per_class_quality": wandb.Image(os.path.join(IMAGES_DIR, 'fixed_point_per_class_quality.png'))})
plt.close()
print("  -> Saved per-class quality plot")

# 6. PCA comparison of real vs generated
pca_compare = PCA(n_components=2)
real_pca = pca_compare.fit_transform(test_ims[:n_test])
gen_pca = pca_compare.transform(generated_ims)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

scatter1 = axes[0].scatter(real_pca[:, 0], real_pca[:, 1], c=test_labels[:n_test], 
                           cmap='tab10', alpha=0.6, s=20)
axes[0].set_title('Real IMS Spectra (PCA)', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca_compare.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_compare.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Chemical Class')

scatter2 = axes[1].scatter(gen_pca[:, 0], gen_pca[:, 1], c=test_labels[:n_test],
                           cmap='tab10', alpha=0.6, s=20)
axes[1].set_title('Generated IMS Spectra (PCA)', fontsize=14, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca_compare.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca_compare.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Chemical Class')

plt.suptitle('PCA Comparison: Real vs Generated (Fixed-Point)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fixed_point_pca_comparison.png'), dpi=150)
wandb.log({"pca_comparison": wandb.Image(os.path.join(IMAGES_DIR, 'fixed_point_pca_comparison.png'))})
plt.close()
print("  -> Saved PCA comparison")

# Save results
print("\nSaving results...")
np.save(os.path.join(RESULTS_DIR, 'fixed_point_generated_ims.npy'), generated_ims)
np.save(os.path.join(RESULTS_DIR, 'fixed_point_lattice_Y.npy'), all_train_Y)
np.save(os.path.join(RESULTS_DIR, 'fixed_point_labels.npy'), test_labels[:n_test])

results = {
    'training_complete': True,
    'final_loss': loss_history[-1],
    'best_loss': best_loss,
    'test_reconstruction_mse': float(recon_mse),
    'test_generation_mse': float(gen_mse),
    'test_fid_score': float(fid_score),
    'num_parameters': total_params,
    'epochs_trained': max_epochs,
    'num_iterations': num_iterations,
    'num_iterations_inference': num_iterations_inference,
    'lattice_levels': lattice_levels,
    'lattice_range': [lattice_min, lattice_max],
    'class_names': data_dict['class_names'],
    'per_class_mse': [float(m) for m in class_mses],
    'convergence_curve': convergence_curve
}

with open(os.path.join(RESULTS_DIR, 'fixed_point_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print(f"Results:")
print(f"  Final training loss: {loss_history[-1]:.6f}")
print(f"  Best training loss: {best_loss:.6f}")
print(f"  Test reconstruction MSE: {recon_mse:.6f}")
print(f"  Test generation MSE: {gen_mse:.6f}")
print(f"  Test FID score: {fid_score:.4f}")
print(f"  Convergence: {convergence_curve[0]:.4f} → {convergence_curve[-1]:.4f} over {len(convergence_curve)} steps")
print(f"\nOutputs saved to:")
print(f"  Models: {MODELS_DIR}")
print(f"  Images: {IMAGES_DIR}")
print(f"  Results: {RESULTS_DIR}")
print("="*80)

wandb.finish()
