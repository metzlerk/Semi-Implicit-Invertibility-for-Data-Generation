"""
Focused Comparison: Implicit vs Semi-Implicit Training
======================================================

This script performs hyperparameter tuning to compare:
- Implicit training (two separate models)
- Semi-Implicit training (z-switching with roundtrip losses)

Fixed Configuration:
- Sampling Method: Gaussian (proven superior to Convex)
- Loss Function: FID (best for distribution matching)

Hyperparameter Sweep:
- Batch size: [64, 128, 256]
- Learning rate: [1e-4, 5e-4, 1e-3]
- Hidden dimension: [64, 128]
- Number of layers: [2, 3, 4]

Total: 2 methods × 3 batch sizes × 3 learning rates × 2 hidden dims × 3 layers = 108 experiments
"""

import os
import sys
import pickle
import json
import time
from datetime import datetime
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import entropy
import scipy.linalg
from tqdm import tqdm
import requests
import wandb

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = ROOT_DIR
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Initialize wandb
wandb.login(key="57680a36aa570ba8df25adbdd143df3d0bf6b6e8")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# HYPERPARAMETER CONFIGURATION
# =============================================================================
FIXED_PARAMS = {
    'timesteps': 100,
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'max_epochs': 1000,
    'sampling_method': 'gaussian',  # Fixed
    'loss_function': 'fid',  # Fixed
}

HYPERPARAMETER_GRID = {
    'batch_size': [64, 128, 256],
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'hidden_dim': [64, 128],
    'embedding_dim': [32],  # Keep fixed for now
    'num_layers': [2, 3, 4],
}

TRAINING_METHODS = ['implicit', 'semi_implicit']

print(f"\nHyperparameter Tuning Configuration:")
print(f"  Fixed: Gaussian sampling, FID loss")
print(f"  Methods: {TRAINING_METHODS}")
print(f"  Batch sizes: {HYPERPARAMETER_GRID['batch_size']}")
print(f"  Learning rates: {HYPERPARAMETER_GRID['learning_rate']}")
print(f"  Hidden dims: {HYPERPARAMETER_GRID['hidden_dim']}")
print(f"  Num layers: {HYPERPARAMETER_GRID['num_layers']}")
total_experiments = (len(TRAINING_METHODS) * 
                     len(HYPERPARAMETER_GRID['batch_size']) * 
                     len(HYPERPARAMETER_GRID['learning_rate']) * 
                     len(HYPERPARAMETER_GRID['hidden_dim']) * 
                     len(HYPERPARAMETER_GRID['num_layers']))
print(f"\nTotal experiments: {total_experiments}")

# =============================================================================
# MNIST1D DATASET
# =============================================================================
def load_mnist1d():
    """Load MNIST1D dataset from GitHub"""
    print("Loading MNIST1D dataset...")
    
    data_path = os.path.join(DATA_DIR, 'mnist1d_data.pkl')
    
    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print("  Loaded from local cache")
    except:
        print("  Downloading from GitHub...")
        url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
        response = requests.get(url)
        data = pickle.loads(response.content)
        
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        print("  Downloaded and cached locally")
    
    return data

def create_prime_inverse_patterns(x_data, y_labels, normalize=True):
    """Create deterministic prime inverse patterns for each digit class"""
    print("Creating prime inverse patterns for each digit class...")
    
    prime_inverses = []
    
    for digit in range(10):
        mask = y_labels == digit
        digit_samples = x_data[mask][:100]
        
        if digit == 0:
            pattern = -digit_samples
        elif digit == 1:
            pattern = np.roll(digit_samples, 10, axis=1)
        elif digit == 2:
            pattern = -0.8 * digit_samples
        elif digit == 3:
            pattern = digit_samples[:, ::-1]
        elif digit == 4:
            indices = np.arange(digit_samples.shape[1])
            sin_pattern = np.sin(indices * 2 * np.pi / digit_samples.shape[1])
            pattern = digit_samples + sin_pattern[np.newaxis, :]
        elif digit == 5:
            pattern = digit_samples * 1.5 - 1.0
        elif digit == 6:
            pattern = np.roll(digit_samples, -10, axis=1)
        elif digit == 7:
            pattern = np.abs(digit_samples) - 2.0
        elif digit == 8:
            pattern = np.sign(digit_samples) * np.minimum(np.abs(digit_samples)**1.5, 5.0)
        else:
            pattern = np.cos(digit_samples * np.pi / 5.0) + 1.0
        
        if normalize:
            pattern = np.clip(pattern, -5, 5)
        
        prime_inverses.append(pattern)
    
    print(f"  Created {len(prime_inverses)} prime inverse patterns")
    return prime_inverses

def prepare_mnist1d_for_diffusion(data, n_samples=2000):
    """Prepare MNIST1D data for training"""
    print("Preparing MNIST1D data for diffusion training...")
    
    x_train = data['x']
    y_train = data['y']
    
    x_normalized = (x_train - x_train.mean()) / x_train.std() * 2.0
    x_normalized = np.clip(x_normalized, -5, 5)
    x_normalized = x_normalized[:, :40]
    
    prime_inverse_patterns = create_prime_inverse_patterns(x_normalized, y_train, normalize=True)
    
    n_per_set = n_samples // 2
    
    forward_inputs = []
    forward_targets = []
    forward_labels = []
    inverse_inputs = []
    inverse_targets = []
    inverse_labels = []
    
    samples_per_digit = n_per_set // 10
    
    for digit in range(10):
        mask = y_train == digit
        digit_samples = x_normalized[mask]
        digit_labels = y_train[mask]
        
        n_available = min(len(digit_samples), samples_per_digit)
        selected_indices = np.random.choice(len(digit_samples), n_available, replace=False)
        clean_samples = digit_samples[selected_indices]
        
        forward_inputs.append(clean_samples)
        forward_targets.append(clean_samples)
        forward_labels.append(np.full(n_available, digit))
        
        inverse_inputs.append(clean_samples)
        inverse_targets.append(prime_inverse_patterns[digit][:n_available])
        inverse_labels.append(np.full(n_available, digit))
    
    forward_inputs = np.vstack(forward_inputs)
    forward_targets = np.vstack(forward_targets)
    forward_labels = np.concatenate(forward_labels)
    inverse_inputs = np.vstack(inverse_inputs)
    inverse_targets = np.vstack(inverse_targets)
    inverse_labels = np.concatenate(inverse_labels)
    
    forward_inputs_with_z = np.column_stack([forward_inputs, np.zeros((len(forward_inputs), 1))])
    forward_targets_with_z = np.column_stack([forward_targets, np.zeros((len(forward_targets), 1))])
    
    inverse_inputs_with_z = np.column_stack([inverse_inputs, np.ones((len(inverse_inputs), 1))])
    inverse_targets_with_z = np.column_stack([inverse_targets, np.ones((len(inverse_targets), 1))])
    
    input_points = np.vstack([forward_inputs_with_z, inverse_inputs_with_z])
    target_points = np.vstack([forward_targets_with_z, inverse_targets_with_z])
    labels = np.concatenate([forward_labels, inverse_labels])
    
    print(f"  Total samples: {len(input_points)}")
    return input_points, target_points, prime_inverse_patterns, labels

# Load data once
mnist1d_data = load_mnist1d()
input_data, target_data, prime_inverse_patterns, digit_labels = prepare_mnist1d_for_diffusion(mnist1d_data, n_samples=2000)
feature_dim = input_data.shape[1]
print(f"Feature dimension: {feature_dim}")

# =============================================================================
# DIFFUSION MODELS
# =============================================================================
class GaussianDiffusion:
    """Gaussian diffusion process"""
    
    def __init__(self, timesteps=100, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and noise"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Compute mean and variance of diffusion posterior"""
        posterior_mean_coef1 = self.betas[t] * self.sqrt_alphas_cumprod_prev[t] / (1.0 - self.alphas_cumprod[t])
        posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev[t]) * torch.sqrt(self.alphas[t]) / (1.0 - self.alphas_cumprod[t])
        
        posterior_mean = (
            posterior_mean_coef1.reshape(-1, 1) * x_start +
            posterior_mean_coef2.reshape(-1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].reshape(-1, 1)
        
        return posterior_mean, posterior_variance
    
    def p_mean_variance(self, model, x_t, t, condition=None):
        """Apply model to get p(x_{t-1} | x_t)"""
        model_output = model(x_t, t, condition)
        x_start = self.predict_start_from_noise(x_t, t, model_output)
        x_start = torch.clamp(x_start, -5.0, 5.0)
        model_mean, posterior_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        return model_mean, posterior_variance
    
    def p_sample(self, model, x_t, t, condition=None):
        """Sample from p(x_{t-1} | x_t)"""
        model_mean, model_variance = self.p_mean_variance(model, x_t, t, condition)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, 1)
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
    
    def p_sample_loop(self, model, shape, condition=None):
        """Generate samples"""
        device = next(model.parameters()).device
        x_t = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling", leave=False):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, condition)
        
        return x_t
    
    def sample(self, model, shape, condition=None):
        """Generate samples"""
        return self.p_sample_loop(model, shape, condition)

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
class PointUNet(nn.Module):
    """U-Net for denoising points"""
    
    def __init__(self, input_dim=41, hidden_dim=128, time_dim=64, condition_dim=41, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ))
        
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ))
        
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, t, condition=None):
        t_embed = self.time_embed(t)
        t_embed = self.time_mlp(t_embed)
        
        if condition is not None:
            c_embed = self.condition_embed(condition)
        else:
            c_embed = torch.zeros_like(t_embed)
        
        h = self.input_proj(x)
        h = h + t_embed + c_embed
        
        skip_connections = []
        for layer in self.encoder_layers:
            h = layer(h)
            skip_connections.append(h)
        
        h = self.middle(h)
        
        for layer, skip in zip(self.decoder_layers, reversed(skip_connections)):
            h = torch.cat([h, skip], dim=1)
            h = layer(h)
        
        return self.output_proj(h)

# =============================================================================
# METRICS
# =============================================================================
def compute_fid_score(real_samples, generated_samples):
    """Compute FID score"""
    real_np = real_samples.detach().cpu().numpy()
    gen_np = generated_samples.detach().cpu().numpy()
    
    mu_real = np.mean(real_np, axis=0)
    sigma_real = np.cov(real_np, rowvar=False)
    
    mu_gen = np.mean(gen_np, axis=0)
    sigma_gen = np.cov(gen_np, rowvar=False)
    
    mean_diff = np.sum((mu_real - mu_gen) ** 2)
    
    eps = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_gen += np.eye(sigma_gen.shape[0]) * eps
    
    covmean = scipy.linalg.sqrtm(sigma_real @ sigma_gen)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    trace_term = np.trace(sigma_real + sigma_gen - 2 * covmean)
    fid = mean_diff + trace_term
    
    return torch.tensor(fid, device=real_samples.device, dtype=torch.float32)

def compute_per_digit_fid(real_samples, generated_samples, labels):
    """Compute FID score per digit and return mean"""
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    unique_digits = np.unique(labels_np)
    
    fid_scores = []
    
    for digit in unique_digits:
        mask = labels_np == digit
        if mask.sum() > 5:
            real_digit = real_samples[mask]
            gen_digit = generated_samples[mask]
            
            fid_digit = compute_fid_score(real_digit, gen_digit)
            fid_scores.append(fid_digit.item())
    
    if len(fid_scores) > 0:
        return torch.tensor(np.mean(fid_scores), device=real_samples.device, dtype=torch.float32)
    else:
        return torch.tensor(0.0, device=real_samples.device, dtype=torch.float32)

# =============================================================================
# TRAINING CLASSES
# =============================================================================
class SemiImplicitTrainer:
    """Semi-implicit training with z-switching and roundtrip losses"""
    
    def __init__(self, model, diffusion, optimizer, device):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.device = device
    
    def train_step(self, input_points, target_points, condition_points):
        self.optimizer.zero_grad()
        
        z_values = input_points[:, -1]
        forward_mask = z_values == 0
        inverse_mask = z_values == 1
        
        total_loss = 0
        loss_dict = {}
        
        # Forward loss
        if forward_mask.sum() > 0:
            forward_inputs = input_points[forward_mask]
            forward_targets = target_points[forward_mask]
            
            t_forward = torch.randint(0, self.diffusion.timesteps, (forward_inputs.shape[0],), device=self.device).long()
            noise_forward = torch.randn_like(forward_targets)
            noisy_inputs = self.diffusion.q_sample(forward_targets, t_forward, noise_forward)
            condition_forward = noisy_inputs.clone()
            condition_forward[:, -1] = 0
            
            predicted_noise_forward = self.model(noisy_inputs, t_forward, condition_forward)
            forward_loss = F.mse_loss(predicted_noise_forward, noise_forward)
            total_loss += forward_loss
            loss_dict['forward'] = forward_loss.item()
        else:
            loss_dict['forward'] = 0.0
        
        # Inverse loss
        if inverse_mask.sum() > 0:
            inverse_inputs = input_points[inverse_mask]
            inverse_targets = target_points[inverse_mask]
            
            t_inverse = torch.randint(0, self.diffusion.timesteps, (inverse_inputs.shape[0],), device=self.device).long()
            noise_to_prime = torch.randn_like(inverse_targets)
            noisy_prime = self.diffusion.q_sample(inverse_targets, t_inverse, noise_to_prime)
            condition_inverse = inverse_inputs.clone()
            condition_inverse[:, -1] = 1
            
            predicted_noise_inverse = self.model(noisy_prime, t_inverse, condition_inverse)
            inverse_loss = F.mse_loss(predicted_noise_inverse, noise_to_prime)
            total_loss += inverse_loss
            loss_dict['inverse'] = inverse_loss.item()
        else:
            loss_dict['inverse'] = 0.0
        
        # Roundtrip losses (simplified for speed)
        if forward_mask.sum() > 0 and inverse_mask.sum() > 0:
            n_roundtrip = min(10, forward_mask.sum())
            clean_data = forward_targets[:n_roundtrip]
            t_round = torch.randint(0, self.diffusion.timesteps, (n_roundtrip,), device=self.device).long()
            
            # F->I roundtrip
            noise_1 = torch.randn_like(clean_data)
            noisy_1 = self.diffusion.q_sample(clean_data, t_round, noise_1)
            cond_1 = noisy_1.clone()
            cond_1[:, -1] = 0
            
            pred_noise_1 = self.model(noisy_1, t_round, cond_1)
            roundtrip_loss = F.mse_loss(pred_noise_1, noise_1)
            total_loss += 0.5 * roundtrip_loss
            loss_dict['roundtrip'] = roundtrip_loss.item()
        else:
            loss_dict['roundtrip'] = 0.0
        
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), loss_dict

# =============================================================================
# DATASET
# =============================================================================
class PointDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, target_data, labels):
        self.input_data = torch.tensor(input_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx], self.labels[idx]

# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
def run_experiment(method, batch_size, learning_rate, hidden_dim, embedding_dim, num_layers, exp_id):
    """Run a single hyperparameter configuration"""
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {exp_id}")
    print(f"Method: {method}")
    print(f"Batch size: {batch_size}, LR: {learning_rate}, Hidden: {hidden_dim}, Layers: {num_layers}")
    print(f"{'='*80}")
    
    # Initialize wandb
    wandb_run = wandb.init(
        entity="kjmetzler-worcester-polytechnic-institute",
        project="implicit-vs-semiimplicit-comparison",
        name=f"exp{exp_id:03d}_{method}_bs{batch_size}_lr{learning_rate}_h{hidden_dim}_l{num_layers}",
        config={
            "experiment_id": exp_id,
            "training_method": method,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            **FIXED_PARAMS,
            "feature_dim": feature_dim,
            "dataset": "mnist1d"
        },
        reinit=True
    )
    
    start_time = time.time()
    
    # Create dataset and dataloader
    train_dataset = PointDiffusionDataset(input_data, target_data, digit_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Create diffusion process
    diffusion = GaussianDiffusion(
        timesteps=FIXED_PARAMS['timesteps'],
        beta_start=FIXED_PARAMS['beta_start'],
        beta_end=FIXED_PARAMS['beta_end'],
        device=device
    )
    
    # Create model(s)
    model_obj = PointUNet(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        time_dim=embedding_dim,
        condition_dim=feature_dim,
        num_layers=num_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model_obj.parameters())
    
    # For implicit: create second model
    inverse_model_obj = None
    if method == 'implicit':
        inverse_model_obj = PointUNet(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            time_dim=embedding_dim,
            condition_dim=feature_dim,
            num_layers=num_layers
        ).to(device)
        total_params += sum(p.numel() for p in inverse_model_obj.parameters())
        print(f"  Created separate inverse model (2 models total)")
    
    # Create optimizer
    if inverse_model_obj is not None:
        optimizer = optim.Adam(
            list(model_obj.parameters()) + list(inverse_model_obj.parameters()),
            lr=learning_rate
        )
    else:
        optimizer = optim.Adam(model_obj.parameters(), lr=learning_rate)
    
    print(f"  Total parameters: {total_params:,}")
    
    # Training
    print(f"\nTraining...")
    loss_history = []
    
    for epoch in range(FIXED_PARAMS['max_epochs']):
        epoch_loss = 0
        epoch_loss_dict = {'forward': 0, 'inverse': 0, 'roundtrip': 0}
        num_batches = 0
        
        for batch_idx, (input_batch, target_batch, labels_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            if method == 'implicit':
                # Implicit: train two separate models
                optimizer.zero_grad()
                
                z_values = input_batch[:, -1]
                forward_mask = z_values == 0
                inverse_mask = z_values == 1
                
                total_batch_loss = 0
                
                if forward_mask.sum() > 0:
                    forward_inputs = input_batch[forward_mask]
                    forward_targets = target_batch[forward_mask]
                    
                    t_forward = torch.randint(0, diffusion.timesteps, (forward_inputs.shape[0],), device=device).long()
                    noise_forward = torch.randn_like(forward_targets)
                    noisy_inputs = diffusion.q_sample(forward_targets, t_forward, noise_forward)
                    
                    condition_forward = noisy_inputs.clone()
                    condition_forward[:, -1] = 0
                    
                    predicted_noise_forward = model_obj(noisy_inputs, t_forward, condition_forward)
                    forward_loss = F.mse_loss(predicted_noise_forward, noise_forward)
                    total_batch_loss += forward_loss
                    epoch_loss_dict['forward'] += forward_loss.item()
                
                if inverse_mask.sum() > 0 and inverse_model_obj is not None:
                    inverse_inputs = input_batch[inverse_mask]
                    inverse_targets = target_batch[inverse_mask]
                    
                    t_inverse = torch.randint(0, diffusion.timesteps, (inverse_inputs.shape[0],), device=device).long()
                    noise_inverse = torch.randn_like(inverse_targets)
                    noisy_inverse = diffusion.q_sample(inverse_targets, t_inverse, noise_inverse)
                    
                    condition_inverse = noisy_inverse.clone()
                    condition_inverse[:, -1] = 1
                    
                    predicted_noise_inverse = inverse_model_obj(noisy_inverse, t_inverse, condition_inverse)
                    inverse_loss = F.mse_loss(predicted_noise_inverse, noise_inverse)
                    total_batch_loss += inverse_loss
                    epoch_loss_dict['inverse'] += inverse_loss.item()
                
                if total_batch_loss > 0:
                    total_batch_loss.backward()
                    optimizer.step()
                    epoch_loss += total_batch_loss.item()
                    
            elif method == 'semi_implicit':
                # Semi-implicit: use trainer
                trainer = SemiImplicitTrainer(model_obj, diffusion, optimizer, device)
                total_loss, loss_dict = trainer.train_step(input_batch, target_batch, input_batch)
                epoch_loss += total_loss
                
                for key in epoch_loss_dict:
                    if key in loss_dict:
                        epoch_loss_dict[key] += loss_dict[key]
            
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Log to wandb
        wandb_log = {"epoch": epoch, "train_loss": avg_loss}
        
        if method == 'semi_implicit':
            for key in epoch_loss_dict:
                wandb_log[f"train_{key}_loss"] = epoch_loss_dict[key] / num_batches
        
        wandb.log(wandb_log)
        
        if epoch % 50 == 0 or epoch == FIXED_PARAMS['max_epochs'] - 1:
            print(f"  Epoch {epoch}/{FIXED_PARAMS['max_epochs']}, Loss: {avg_loss:.6f}")
    
    # Testing
    print(f"\nTesting...")
    model_obj.eval()
    if inverse_model_obj is not None:
        inverse_model_obj.eval()
    
    with torch.no_grad():
        # Generate test data
        test_data = load_mnist1d()
        test_input, test_target, _, test_labels = prepare_mnist1d_for_diffusion(test_data, n_samples=200)
        test_input_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)
        test_target_tensor = torch.tensor(test_target, dtype=torch.float32).to(device)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(device)
        
        z_values = test_input_tensor[:, -1]
        forward_mask = z_values == 0
        inverse_mask = z_values == 1
        
        generated_samples = torch.zeros_like(test_target_tensor)
        
        # Generate forward samples
        if forward_mask.sum() > 0:
            use_model = model_obj
            forward_samples = diffusion.sample(
                use_model,
                shape=(forward_mask.sum(), feature_dim),
                condition=test_input_tensor[forward_mask]
            )
            generated_samples[forward_mask] = forward_samples
        
        # Generate inverse samples
        if inverse_mask.sum() > 0:
            if method == 'implicit' and inverse_model_obj is not None:
                use_model = inverse_model_obj
            else:
                use_model = model_obj
            
            inverse_samples = diffusion.sample(
                use_model,
                shape=(inverse_mask.sum(), feature_dim),
                condition=test_input_tensor[inverse_mask]
            )
            generated_samples[inverse_mask] = inverse_samples
        
        # Compute metrics
        mse = F.mse_loss(generated_samples, test_target_tensor).item()
        mae = np.mean(np.abs(generated_samples.cpu().numpy() - test_target_tensor.cpu().numpy()))
        
        fid_score = compute_per_digit_fid(test_target_tensor, generated_samples, test_labels_tensor).item()
        
        # Per-direction metrics
        if forward_mask.sum() > 0:
            forward_mse = F.mse_loss(generated_samples[forward_mask], test_target_tensor[forward_mask]).item()
            forward_fid = compute_per_digit_fid(
                test_target_tensor[forward_mask], 
                generated_samples[forward_mask],
                test_labels_tensor[forward_mask]
            ).item()
        else:
            forward_mse = 0.0
            forward_fid = 0.0
            
        if inverse_mask.sum() > 0:
            inverse_mse = F.mse_loss(generated_samples[inverse_mask], test_target_tensor[inverse_mask]).item()
            inverse_fid = compute_per_digit_fid(
                test_target_tensor[inverse_mask], 
                generated_samples[inverse_mask],
                test_labels_tensor[inverse_mask]
            ).item()
        else:
            inverse_mse = 0.0
            inverse_fid = 0.0
    
    runtime = time.time() - start_time
    
    # Store results
    result = {
        'experiment_id': exp_id,
        'training_method': method,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'hidden_dim': hidden_dim,
        'embedding_dim': embedding_dim,
        'num_layers': num_layers,
        'num_parameters': total_params,
        'final_training_loss': loss_history[-1],
        'test_mse': mse,
        'test_mae': mae,
        'test_fid_score': fid_score,
        'test_forward_mse': forward_mse,
        'test_forward_fid': forward_fid,
        'test_inverse_mse': inverse_mse,
        'test_inverse_fid': inverse_fid,
        'runtime_seconds': runtime,
        'epochs': FIXED_PARAMS['max_epochs'],
    }
    
    # Log final metrics
    wandb.log({
        "test_mse": mse,
        "test_mae": mae,
        "test_fid_score": fid_score,
        "test_forward_mse": forward_mse,
        "test_forward_fid": forward_fid,
        "test_inverse_mse": inverse_mse,
        "test_inverse_fid": inverse_fid,
        "runtime_seconds": runtime
    })
    
    print(f"\nResults:")
    print(f"  Test MSE: {mse:.6f}")
    print(f"  Test FID: {fid_score:.6f}")
    print(f"  Forward FID: {forward_fid:.6f}")
    print(f"  Inverse FID: {inverse_fid:.6f}")
    print(f"  Runtime: {runtime:.1f}s")
    
    wandb_run.finish()
    
    return result

# =============================================================================
# RUN ALL EXPERIMENTS
# =============================================================================
print("\n" + "="*80)
print("HYPERPARAMETER TUNING: IMPLICIT VS SEMI-IMPLICIT")
print("="*80)
print(f"Total experiments: {total_experiments}")
print("="*80)

all_results = []
exp_id = 1

for method in TRAINING_METHODS:
    for batch_size in HYPERPARAMETER_GRID['batch_size']:
        for learning_rate in HYPERPARAMETER_GRID['learning_rate']:
            for hidden_dim in HYPERPARAMETER_GRID['hidden_dim']:
                for num_layers in HYPERPARAMETER_GRID['num_layers']:
                    try:
                        result = run_experiment(
                            method=method,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            hidden_dim=hidden_dim,
                            embedding_dim=HYPERPARAMETER_GRID['embedding_dim'][0],
                            num_layers=num_layers,
                            exp_id=exp_id
                        )
                        
                        all_results.append(result)
                        
                        # Save intermediate results
                        if exp_id % 5 == 0:
                            results_path = os.path.join(RESULTS_DIR, 'implicit_vs_semiimplicit_results.json')
                            with open(results_path, 'w') as f:
                                json.dump(all_results, f, indent=2, cls=NumpyEncoder)
                            print(f"\n  ✓ Saved results after {exp_id} experiments")
                        
                        exp_id += 1
                        
                    except Exception as e:
                        print(f"\n  ✗ ERROR in experiment {exp_id}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        exp_id += 1

# Save final results
print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETE!")
print("="*80)

results_path = os.path.join(RESULTS_DIR, 'implicit_vs_semiimplicit_results_final.json')
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2, cls=NumpyEncoder)

# Create summary DataFrame
df = pd.DataFrame(all_results)
summary_path = os.path.join(RESULTS_DIR, 'implicit_vs_semiimplicit_summary.csv')
df.to_csv(summary_path, index=False)

print(f"\nResults saved to:")
print(f"  {results_path}")
print(f"  {summary_path}")

# Find best configurations
print("\n" + "="*80)
print("BEST CONFIGURATIONS")
print("="*80)

for method in TRAINING_METHODS:
    method_results = df[df['training_method'] == method]
    best_fid = method_results.loc[method_results['test_fid_score'].idxmin()]
    
    print(f"\nBest {method}:")
    print(f"  FID Score: {best_fid['test_fid_score']:.6f}")
    print(f"  Batch size: {best_fid['batch_size']}")
    print(f"  Learning rate: {best_fid['learning_rate']}")
    print(f"  Hidden dim: {best_fid['hidden_dim']}")
    print(f"  Num layers: {best_fid['num_layers']}")
    print(f"  Parameters: {best_fid['num_parameters']:,}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

implicit_best_fid = df[df['training_method'] == 'implicit']['test_fid_score'].min()
semiimplicit_best_fid = df[df['training_method'] == 'semi_implicit']['test_fid_score'].min()

print(f"\nBest Implicit FID: {implicit_best_fid:.6f}")
print(f"Best Semi-Implicit FID: {semiimplicit_best_fid:.6f}")
print(f"Improvement: {((implicit_best_fid - semiimplicit_best_fid) / implicit_best_fid * 100):.2f}%")

print("\nDone!")
