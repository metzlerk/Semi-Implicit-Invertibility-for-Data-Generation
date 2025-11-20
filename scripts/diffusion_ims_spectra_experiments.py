"""
Diffusion Model for IMS Spectra Generation and Classification
==============================================================

Semi-Implicit Training with Dual Objectives:
- z=0: Generation - Input: [noisy_IMS + clean_onehot] → Output: [clean_IMS + clean_onehot]
- z=1: Classification - Input: [clean_IMS + noisy_onehot] → Output: [clean_IMS + clean_onehot]

Dataset: IMS Spectra with 8 chemical classes
- Positive mode: 838 features (p_184 to p_1021)
- Negative mode: 838 features (n_184 to n_1021)  
- Total IMS features: 1676
- One-hot labels: 8 classes (DEB, DEM, DMMP, DPM, DtBP, JP8, MES, TEPO)
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
timesteps = 100
beta_start = 0.0001
beta_end = 0.02
hidden_dim = 256
embedding_dim = 128
num_layers = 4
learning_rate = 1e-4
max_epochs = 300
batch_size = 256
test_mode = '--test' in sys.argv  # Quick test with reduced data

# Loss weights (generation weighted more than classification)
generation_weight = 0.9  # Weight for IMS reconstruction (prioritized)
classification_weight = 0.1  # Weight for SMILE embedding classification

print(f"Hyperparameters:")
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
    index_cols = ['Unnamed: 0', 'index', 'Label']
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
    embedding_dim = len(next(iter(smile_embeddings.values())))
    
    # Convert one-hot to SMILE embeddings
    train_onehot = train_df[onehot_cols].values
    test_onehot = test_df[onehot_cols].values
    
    train_labels = train_onehot.argmax(axis=1)
    test_labels = test_onehot.argmax(axis=1)
    
    # Create embedding matrices
    train_embeddings = np.zeros((len(train_df), embedding_dim), dtype=np.float32)
    test_embeddings = np.zeros((len(test_df), embedding_dim), dtype=np.float32)
    
    for idx, label in enumerate(onehot_cols):
        if label in smile_embeddings:
            train_mask = train_labels == idx
            test_mask = test_labels == idx
            train_embeddings[train_mask] = smile_embeddings[label]
            test_embeddings[test_mask] = smile_embeddings[label]
    
    print(f"\nSMILE embeddings created:")
    print(f"  Embedding dimension: {embedding_dim}")
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
        'num_embedding_dim': embedding_dim,
        'smile_embeddings': smile_embeddings
    }


def prepare_diffusion_data(data, n_samples=None):
    """
    Prepare data for semi-implicit diffusion training
    
    Two training modes:
    - z=0 (Generation): [noisy_IMS + clean_embedding] → [clean_IMS + clean_embedding]
    - z=1 (Classification): [clean_IMS + noisy_embedding] → [clean_IMS + clean_embedding]
    """
    print("\nPreparing data for semi-implicit diffusion...")
    
    train_ims = data['train_ims']
    train_embeddings = data['train_embeddings']
    train_labels = data['train_labels']
    
    if n_samples is not None:
        indices = np.random.choice(len(train_ims), n_samples, replace=False)
        train_ims = train_ims[indices]
        train_embeddings = train_embeddings[indices]
        train_labels = train_labels[indices]
    
    n_per_mode = len(train_ims)
    
    # Mode 0: Generation (noisy IMS + clean embedding → clean IMS + clean embedding)
    gen_inputs_ims = train_ims.copy()  # Will be noised during training
    gen_inputs_embedding = train_embeddings.copy()
    gen_targets_ims = train_ims.copy()
    gen_targets_embedding = train_embeddings.copy()
    gen_z = np.zeros((n_per_mode, 1))
    
    # Mode 1: Classification (clean IMS + noisy embedding → clean IMS + clean embedding)
    class_inputs_ims = train_ims.copy()
    class_inputs_embedding = train_embeddings.copy()  # Will be noised during training
    class_targets_ims = train_ims.copy()
    class_targets_embedding = train_embeddings.copy()
    class_z = np.ones((n_per_mode, 1))
    
    # Stack both modes
    input_ims = np.vstack([gen_inputs_ims, class_inputs_ims])
    input_embedding = np.vstack([gen_inputs_embedding, class_inputs_embedding])
    target_ims = np.vstack([gen_targets_ims, class_targets_ims])
    target_embedding = np.vstack([gen_targets_embedding, class_targets_embedding])
    z_coords = np.vstack([gen_z, class_z])
    labels = np.concatenate([train_labels, train_labels])
    
    # Concatenate IMS + embedding + z for full input/target vectors
    input_data = np.concatenate([input_ims, input_embedding, z_coords], axis=1)
    target_data = np.concatenate([target_ims, target_embedding, z_coords], axis=1)
    
    print(f"  Generation samples (z=0): {n_per_mode}")
    print(f"  Classification samples (z=1): {n_per_mode}")
    print(f"  Total samples: {len(input_data)}")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Target shape: {target_data.shape}")
    print(f"  Features breakdown: {data['num_ims_features']} IMS + {data['num_embedding_dim']} embedding + 1 z")
    
    return input_data, target_data, labels, data['num_ims_features'], data['num_embedding_dim']


# Load data
data_dict = load_ims_data()
n_samples_use = 5000 if test_mode else None
if test_mode:
    print("\n*** TEST MODE: Using only 5000 samples ***\n")
input_data, target_data, train_labels, num_ims_features, num_embedding_dim = prepare_diffusion_data(data_dict, n_samples=n_samples_use)

feature_dim = input_data.shape[1]  # IMS + embedding + z
print(f"\nTotal feature dimension: {feature_dim}")

# =============================================================================
# TIME EMBEDDING
# =============================================================================
class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# =============================================================================
# U-NET MODEL
# =============================================================================
class IMSUNet(nn.Module):
    """U-Net for IMS spectra denoising and classification"""
    
    def __init__(self, input_dim, hidden_dim=256, time_dim=128, condition_dim=None, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.condition_dim = condition_dim if condition_dim else input_dim
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding (for z-coordinate conditioning)
        self.condition_embed = nn.Sequential(
            nn.Linear(self.condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            ))
        
        # Middle layer
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU()
        )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU()
            ))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, t, condition=None):
        """
        Args:
            x: Noisy spectra + onehot [batch_size, feature_dim]
            t: Time steps [batch_size]
            condition: Condition vector [batch_size, condition_dim]
        """
        # Embed time
        t_embed = self.time_embed(t)
        t_embed = self.time_mlp(t_embed)
        
        # Embed condition
        if condition is not None:
            c_embed = self.condition_embed(condition)
        else:
            c_embed = torch.zeros_like(t_embed)
        
        # Project input
        h = self.input_proj(x)
        
        # Add time and condition embeddings
        h = h + t_embed + c_embed
        
        # Encoder with skip connections
        skip_connections = []
        for layer in self.encoder_layers:
            h = layer(h)
            skip_connections.append(h)
        
        # Middle
        h = self.middle(h)
        
        # Decoder with skip connections
        for layer, skip in zip(self.decoder_layers, reversed(skip_connections)):
            h = torch.cat([h, skip], dim=1)
            h = layer(h)
        
        # Output
        return self.output_proj(h)


# =============================================================================
# GAUSSIAN DIFFUSION
# =============================================================================
class GaussianDiffusion:
    """Gaussian diffusion process"""
    
    def __init__(self, timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """Add noise to data at timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Move t to CPU for indexing, then move result to device
        t_cpu = t.cpu()
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_cpu].to(x_start.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_cpu].to(x_start.device)
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x_t, t, condition=None):
        """Sample x_{t-1} from p(x_{t-1} | x_t)"""
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # Predict noise
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        predicted_noise = model(x_t, t_tensor, condition)
        
        # Calculate mean
        alpha = self.alphas[t].to(device)
        alpha_cumprod = self.alphas_cumprod[t].to(device)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].to(device)
        
        mean = (x_t - (1 - alpha) / sqrt_one_minus_alpha_cumprod * predicted_noise) / torch.sqrt(alpha)
        
        if t == 0:
            return mean
        else:
            variance = self.posterior_variance[t].to(device)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise
    
    def p_sample_loop(self, model, shape, condition=None):
        """Generate samples by iterating the reverse process"""
        device = next(model.parameters()).device
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            x_t = self.p_sample(model, x_t, t, condition)
        
        return x_t
    
    def sample(self, model, shape, condition=None):
        return self.p_sample_loop(model, shape, condition)


# =============================================================================
# SEMI-IMPLICIT TRAINER
# =============================================================================
class SemiImplicitIMSTrainer:
    """
    Semi-implicit training for IMS spectra generation and classification
    
    z=0 (Generation): [noisy_IMS + clean_embedding] → [clean_IMS + clean_embedding]
    z=1 (Classification): [clean_IMS + noisy_embedding] → [clean_IMS + clean_embedding]
    """
    
    def __init__(self, model, diffusion, optimizer, device, num_ims_features, num_embedding_dim,
                 generation_weight=0.7, classification_weight=0.3):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.device = device
        self.num_ims_features = num_ims_features
        self.num_embedding_dim = num_embedding_dim
        self.generation_weight = generation_weight
        self.classification_weight = classification_weight
        
    def train_step(self, input_batch, target_batch):
        """
        Training step that handles both generation and classification
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = input_batch.shape[0]
        
        # Extract components from input
        # Input structure: [IMS (1676) | embedding (512) | z (1)]
        z_values = input_batch[:, -1]
        gen_mask = z_values == 0
        class_mask = z_values == 1
        
        total_loss = 0
        loss_dict = {'generation': 0, 'classification': 0, 'ims_recon': 0, 'embedding_recon': 0}
        
        # ====================================================================
        # MODE 0: GENERATION (noisy IMS + clean onehot → clean IMS + clean onehot)
        # ====================================================================
        if gen_mask.sum() > 0:
            gen_inputs = input_batch[gen_mask]
            gen_targets = target_batch[gen_mask]
            
            # Random timesteps
            t_gen = torch.randint(0, self.diffusion.timesteps, (gen_inputs.shape[0],), device=self.device).long()
            
            # Extract IMS and embedding from targets
            target_ims = gen_targets[:, :self.num_ims_features]
            target_embedding = gen_targets[:, self.num_ims_features:self.num_ims_features+self.num_embedding_dim]
            
            # Create noise for IMS only
            noise = torch.randn_like(target_ims)
            
            # Noise the IMS spectra
            noisy_ims = self.diffusion.q_sample(target_ims, t_gen, noise)
            
            # Reconstruct full noisy input: [noisy_IMS | clean_embedding | z=0]
            noisy_input = torch.cat([
                noisy_ims,
                target_embedding,  # Keep embedding clean
                torch.zeros((gen_inputs.shape[0], 1), device=self.device)  # z=0
            ], dim=1)
            
            # Condition on the input
            condition = noisy_input.clone()
            
            # Predict the noise
            predicted_noise_full = self.model(noisy_input, t_gen, condition)
            
            # Extract predicted IMS noise (first num_ims_features dimensions)
            predicted_noise_ims = predicted_noise_full[:, :self.num_ims_features]
            
            # Generation loss (IMS reconstruction)
            gen_loss = F.mse_loss(predicted_noise_ims, noise)
            total_loss += self.generation_weight * gen_loss
            loss_dict['generation'] = gen_loss.item()
            loss_dict['ims_recon'] += gen_loss.item()
        
        # ====================================================================
        # MODE 1: CLASSIFICATION (clean IMS + noisy onehot → clean IMS + clean onehot)
        # ====================================================================
        if class_mask.sum() > 0:
            class_inputs = input_batch[class_mask]
            class_targets = target_batch[class_mask]
            
            # Random timesteps
            t_class = torch.randint(0, self.diffusion.timesteps, (class_inputs.shape[0],), device=self.device).long()
            
            # Extract IMS and embedding from targets
            target_ims = class_targets[:, :self.num_ims_features]
            target_embedding = class_targets[:, self.num_ims_features:self.num_ims_features+self.num_embedding_dim]
            
            # Create Gaussian noise for embedding (classification task)
            embedding_noise_scale = 0.5  # Control noise level
            embedding_noise = torch.randn_like(target_embedding) * embedding_noise_scale
            
            # Add noise to embedding using diffusion schedule
            # Move t_class to CPU for indexing, then move result to device
            t_class_cpu = t_class.cpu()
            alpha_t = self.diffusion.sqrt_alphas_cumprod[t_class_cpu].to(self.device).view(-1, 1)
            sigma_t = self.diffusion.sqrt_one_minus_alphas_cumprod[t_class_cpu].to(self.device).view(-1, 1)
            noisy_embedding = alpha_t * target_embedding + sigma_t * embedding_noise
            
            # Reconstruct full noisy input: [clean_IMS | noisy_embedding | z=1]
            noisy_input = torch.cat([
                target_ims,  # Keep IMS clean
                noisy_embedding,
                torch.ones((class_inputs.shape[0], 1), device=self.device)  # z=1
            ], dim=1)
            
            # Condition on the input
            condition = noisy_input.clone()
            
            # Predict the noise for entire vector
            predicted_noise_full = self.model(noisy_input, t_class, condition)
            
            # Extract predicted embedding noise (embedding dimensions)
            predicted_noise_embedding = predicted_noise_full[:, self.num_ims_features:self.num_ims_features+self.num_embedding_dim]
            
            # Classification loss (embedding reconstruction)
            class_loss = F.mse_loss(predicted_noise_embedding, embedding_noise)
            total_loss += self.classification_weight * class_loss
            loss_dict['classification'] = class_loss.item()
            loss_dict['onehot_recon'] += class_loss.item()
        
        # Backpropagation
        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return total_loss.item(), loss_dict


# =============================================================================
# DATASET AND DATALOADER
# =============================================================================
class IMSDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, target_data, labels):
        self.input_data = torch.FloatTensor(input_data)
        self.target_data = torch.FloatTensor(target_data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx], self.labels[idx]


train_dataset = IMSDataset(input_data, target_data, train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"\nDataset prepared:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Batches per epoch: {len(train_loader)}")

# =============================================================================
# TRAINING
# =============================================================================
print("\n" + "="*80)
print("STARTING IMS SPECTRA DIFFUSION TRAINING")
print("="*80)

# Initialize model
model = IMSUNet(
    input_dim=feature_dim,
    hidden_dim=hidden_dim,
    time_dim=embedding_dim,
    condition_dim=feature_dim,
    num_layers=num_layers
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel architecture:")
print(f"  Total parameters: {total_params:,}")

# Initialize diffusion
diffusion = GaussianDiffusion(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Initialize trainer
trainer = SemiImplicitIMSTrainer(
    model, diffusion, optimizer, device,
    num_ims_features, num_embedding_dim,
    generation_weight, classification_weight
)

# Initialize wandb
wandb_run = wandb.init(
    entity="kjmetzler-worcester-polytechnic-institute",
    project="ims-spectra-diffusion",
    name=f"semi_implicit_dual_mode_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
        "timesteps": timesteps,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "generation_weight": generation_weight,
        "classification_weight": classification_weight,
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
    epoch_gen_loss = 0
    epoch_class_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
    for batch_idx, (input_batch, target_batch, labels_batch) in enumerate(pbar):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        
        # Training step
        batch_loss, loss_dict = trainer.train_step(input_batch, target_batch)
        
        epoch_loss += batch_loss
        epoch_gen_loss += loss_dict['generation']
        epoch_class_loss += loss_dict['classification']
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{batch_loss:.4f}",
            'gen': f"{loss_dict['generation']:.4f}",
            'class': f"{loss_dict['classification']:.4f}"
        })
    
    # Average losses
    avg_loss = epoch_loss / num_batches
    avg_gen_loss = epoch_gen_loss / num_batches
    avg_class_loss = epoch_class_loss / num_batches
    
    loss_history.append(avg_loss)
    
    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "generation_loss": avg_gen_loss,
        "classification_loss": avg_class_loss
    })
    
    # Print epoch summary
    print(f"Epoch {epoch+1}/{max_epochs} - Loss: {avg_loss:.6f} | Gen: {avg_gen_loss:.6f} | Class: {avg_class_loss:.6f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'model_config': {
                'feature_dim': feature_dim,
                'hidden_dim': hidden_dim,
                'embedding_dim': embedding_dim,
                'num_layers': num_layers,
                'num_ims_features': num_ims_features,
                'num_embedding_dim': num_embedding_dim
            }
        }, os.path.join(MODELS_DIR, 'best_ims_model.pth'))
        print(f"  → Saved best model (loss: {best_loss:.6f})")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)

# =============================================================================
# EVALUATION
# =============================================================================
print("\nEvaluating model...")
model.eval()

# Prepare test data
test_input_ims = data_dict['test_ims']
test_input_embeddings = data_dict['test_embeddings']
test_labels = data_dict['test_labels']

# Test generation (z=0)
print("\nTesting Generation (z=0)...")
with torch.no_grad():
    n_test = min(1000, len(test_input_ims))
    test_ims_batch = torch.FloatTensor(test_input_ims[:n_test]).to(device)
    test_embedding_batch = torch.FloatTensor(test_input_embeddings[:n_test]).to(device)
    
    # Generate samples: start with noise for IMS, clean embedding, z=0
    gen_condition = torch.cat([
        torch.randn_like(test_ims_batch),  # noisy IMS
        test_embedding_batch,  # clean embedding
        torch.zeros((n_test, 1), device=device)  # z=0
    ], dim=1)
    
    generated = diffusion.sample(model, shape=(n_test, feature_dim), condition=gen_condition)
    generated_ims = generated[:, :num_ims_features].cpu().numpy()
    
    # Compute MSE for IMS reconstruction
    gen_mse = mean_squared_error(test_input_ims[:n_test].flatten(), generated_ims.flatten())
    print(f"  Generation MSE: {gen_mse:.6f}")

# Test classification (z=1)
print("\nTesting Classification (z=1)...")
with torch.no_grad():
    # Classification: clean IMS + noisy embedding, z=1
    class_condition = torch.cat([
        test_ims_batch,  # clean IMS
        torch.randn_like(test_embedding_batch) * 0.5,  # noisy embedding
        torch.ones((n_test, 1), device=device)  # z=1
    ], dim=1)
    
    classified = diffusion.sample(model, shape=(n_test, feature_dim), condition=class_condition)
    classified_embeddings = classified[:, num_ims_features:num_ims_features+num_embedding_dim].cpu().numpy()
    
    # Get predicted classes by finding closest SMILE embedding
    smile_embeddings_array = np.array([data_dict['smile_embeddings'][label] for label in data_dict['class_names']])
    pred_classes = []
    for emb in classified_embeddings:
        distances = np.linalg.norm(smile_embeddings_array - emb, axis=1)
        pred_classes.append(np.argmin(distances))
    pred_classes = np.array(pred_classes)
    true_classes = test_labels[:n_test]
    
    # Compute accuracy
    accuracy = accuracy_score(true_classes, pred_classes)
    print(f"  Classification Accuracy: {accuracy*100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=data_dict['class_names'],
                yticklabels=data_dict['class_names'])
    plt.title('Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"  → Saved confusion matrix")

# Log final metrics
wandb.log({
    "test_generation_mse": gen_mse,
    "test_classification_accuracy": accuracy,
    "confusion_matrix": wandb.Image(os.path.join(IMAGES_DIR, 'confusion_matrix.png'))
})

# Create wandb Table with generated spectra samples
print("\nLogging generated spectra to wandb...")
spectra_table = wandb.Table(columns=["class_name", "spectrum_plot", "real_vs_generated"])
for i in range(min(8, n_test)):  # Log one sample per class
    class_name = data_dict["class_names"][test_labels[i]]
    
    # Create individual spectrum plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(generated_ims[i], label='Generated', linewidth=1.5)
    ax.plot(test_input_ims[i], label='Real', linewidth=1.5, alpha=0.7)
    ax.set_title(f'{class_name} Spectrum')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Intensity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    spectra_table.add_data(class_name, wandb.Image(fig), i)
    plt.close(fig)

wandb.log({"generated_spectra_samples": spectra_table})
print("  → Logged spectra samples to wandb")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\nGenerating visualizations...")

# 1. Loss curves
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'training_loss.png'), dpi=150)
plt.close()
print("  → Saved training loss plot")

# 2. Generated vs Real Spectra (sample comparison)
fig, axes = plt.subplots(4, 2, figsize=(15, 12))
for i in range(4):
    # Real spectrum
    axes[i, 0].plot(test_input_ims[i])
    axes[i, 0].set_title(f'Real Spectrum {i+1} ({data_dict["class_names"][test_labels[i]]})')
    axes[i, 0].set_xlabel('Feature Index')
    axes[i, 0].set_ylabel('Intensity')
    axes[i, 0].grid(True, alpha=0.3)
    
    # Generated spectrum
    axes[i, 1].plot(generated_ims[i])
    axes[i, 1].set_title(f'Generated Spectrum {i+1}')
    axes[i, 1].set_xlabel('Feature Index')
    axes[i, 1].set_ylabel('Intensity')
    axes[i, 1].grid(True, alpha=0.3)

plt.suptitle('Real vs Generated IMS Spectra', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'spectra_comparison.png'), dpi=150)
wandb.log({"spectra_comparison": wandb.Image(os.path.join(IMAGES_DIR, 'spectra_comparison.png'))})
plt.close()
print("  → Saved spectra comparison")

# 3. PCA visualization
pca = PCA(n_components=2)
real_pca = pca.fit_transform(test_input_ims[:n_test])
gen_pca = pca.transform(generated_ims)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Real data
scatter1 = axes[0].scatter(real_pca[:, 0], real_pca[:, 1], c=test_labels[:n_test], 
                           cmap='tab10', alpha=0.6, s=20)
axes[0].set_title('Real IMS Spectra (PCA)', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Chemical Class')

# Generated data
scatter2 = axes[1].scatter(gen_pca[:, 0], gen_pca[:, 1], c=test_labels[:n_test],
                           cmap='tab10', alpha=0.6, s=20)
axes[1].set_title('Generated IMS Spectra (PCA)', fontsize=14, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Chemical Class')

plt.suptitle('PCA Comparison: Real vs Generated Spectra', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'pca_comparison.png'), dpi=150)
wandb.log({"pca_comparison": wandb.Image(os.path.join(IMAGES_DIR, 'pca_comparison.png'))})
plt.close()
print("  → Saved PCA comparison")

# 4. Per-class generation quality
class_mses = []
for class_idx in range(len(data_dict['class_names'])):
    class_mask = test_labels[:n_test] == class_idx
    if class_mask.sum() > 0:
        class_real = test_input_ims[:n_test][class_mask]
        class_gen = generated_ims[class_mask]
        class_mse = mean_squared_error(class_real.flatten(), class_gen.flatten())
        class_mses.append(class_mse)
    else:
        class_mses.append(0)

plt.figure(figsize=(10, 6))
plt.bar(data_dict['class_names'], class_mses)
plt.xlabel('Chemical Class')
plt.ylabel('MSE')
plt.title('Generation Quality by Chemical Class')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'per_class_generation_quality.png'), dpi=150)
wandb.log({"per_class_quality": wandb.Image(os.path.join(IMAGES_DIR, 'per_class_generation_quality.png'))})
plt.close()
print("  → Saved per-class quality plot")

# Save generated spectra for testing
print("\nSaving generated spectra...")
np.save(os.path.join(RESULTS_DIR, 'generated_ims_test.npy'), generated_ims)
np.save(os.path.join(RESULTS_DIR, 'generated_labels_test.npy'), test_labels[:n_test])
print(f"  → Saved {n_test} generated test spectra")

# Save results
results = {
    'training_complete': True,
    'final_loss': loss_history[-1],
    'best_loss': best_loss,
    'test_generation_mse': float(gen_mse),
    'test_classification_accuracy': float(accuracy),
    'num_parameters': total_params,
    'epochs_trained': max_epochs,
    'class_names': data_dict['class_names'],
    'per_class_mse': [float(m) for m in class_mses]
}

with open(os.path.join(RESULTS_DIR, 'ims_diffusion_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*80)
print("EXPERIMENT COMPLETE!")
print("="*80)
print(f"Results:")
print(f"  Final training loss: {loss_history[-1]:.6f}")
print(f"  Best training loss: {best_loss:.6f}")
print(f"  Test generation MSE: {gen_mse:.6f}")
print(f"  Test classification accuracy: {accuracy*100:.2f}%")
print(f"\nOutputs saved to:")
print(f"  Models: {MODELS_DIR}")
print(f"  Images: {IMAGES_DIR}")
print(f"  Results: {RESULTS_DIR}")
print("="*80)

wandb.finish()
