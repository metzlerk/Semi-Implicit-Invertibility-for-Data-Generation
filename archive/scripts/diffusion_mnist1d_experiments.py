"""
Diffusion Model Experiments on MNIST1D Dataset
Compares 8 configurations (removing architecture dimension):
- Loss Function: MSE vs KL Divergence
- Sampling Method: Gaussian vs Convex
- Training Method: Implicit vs Semi-Implicit

Semi-Implicit Training:
- z=0: Forward direction (noisy -> clean)
- z=1: Inverse direction (clean -> prime_inverse)
- Four losses: forward, inverse, roundtrip_forward_inverse, roundtrip_inverse_forward
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import entropy
import scipy.linalg
from tqdm import tqdm
import json
import time
from datetime import datetime
import math
import requests

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
import pickle
import wandb

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
# HYPERPARAMETERS
# =============================================================================
timesteps = 100
beta_start = 0.0001
beta_end = 0.02
hidden_dim = 64
embedding_dim = 32
num_layers = 2
learning_rate = 1e-3
max_epochs = 1000
batch_size = 128

print(f"Hyperparameters:")
print(f"  Timesteps: {timesteps}")
print(f"  Hidden dimension: {hidden_dim}")
print(f"  Embedding dimension: {embedding_dim}")
print(f"  Number of layers: {num_layers}")
print(f"  Learning rate: {learning_rate}")
print(f"  Max epochs: {max_epochs}")
print(f"  Batch size: {batch_size}")

# =============================================================================
# MNIST1D DATASET
# =============================================================================
def load_mnist1d():
    """Load MNIST1D dataset from GitHub"""
    print("Loading MNIST1D dataset...")
    
    data_path = os.path.join(DATA_DIR, 'mnist1d_data.pkl')
    
    try:
        # Try to load from local file first
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print("  Loaded from local cache")
    except:
        print("  Downloading from GitHub...")
        # Download the dataset
        url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
        response = requests.get(url)
        data = pickle.loads(response.content)
        
        # Save locally for future use
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        print("  Downloaded and cached locally")
    
    return data

def create_prime_inverse_patterns(x_data, y_labels, normalize=True):
    """
    Create deterministic "prime inverse" patterns for each digit class (0-9)
    These are fixed transformations, one per digit class
    
    Similar to how sin(x) -> cos(x)+1 in the original experiment
    """
    print("Creating prime inverse patterns for each digit class...")
    
    prime_inverses = []
    
    for digit in range(10):
        mask = y_labels == digit
        digit_samples = x_data[mask][:100]  # Use first 100 samples of each digit
        
        # Create different deterministic patterns for each digit
        if digit == 0:
            # Inversion
            pattern = -digit_samples
        elif digit == 1:
            # Phase shift
            pattern = np.roll(digit_samples, 10, axis=1)
        elif digit == 2:
            # Inversion + scale
            pattern = -0.8 * digit_samples
        elif digit == 3:
            # Reverse
            pattern = digit_samples[:, ::-1]
        elif digit == 4:
            # Add sinusoidal pattern
            indices = np.arange(digit_samples.shape[1])
            sin_pattern = np.sin(indices * 2 * np.pi / digit_samples.shape[1])
            pattern = digit_samples + sin_pattern[np.newaxis, :]
        elif digit == 5:
            # Scale and shift
            pattern = digit_samples * 1.5 - 1.0
        elif digit == 6:
            # Phase shift opposite direction
            pattern = np.roll(digit_samples, -10, axis=1)
        elif digit == 7:
            # Absolute value transformation
            pattern = np.abs(digit_samples) - 2.0
        elif digit == 8:
            # Squared transformation (clipped)
            pattern = np.sign(digit_samples) * np.minimum(np.abs(digit_samples)**1.5, 5.0)
        else:  # digit == 9
            # Cosine-like transformation
            pattern = np.cos(digit_samples * np.pi / 5.0) + 1.0
        
        # Normalize to [-5, 5] range if requested
        if normalize:
            pattern = np.clip(pattern, -5, 5)
        
        prime_inverses.append(pattern)
    
    print(f"  Created {len(prime_inverses)} prime inverse patterns (one per digit)")
    return prime_inverses


def prepare_mnist1d_for_diffusion(data, n_samples=2000):
    """
    Prepare MNIST1D data for semi-implicit training
    
    Semi-Implicit Training Setup:
    - Set 1 (Forward, z=0): Noisy data -> Clean digits
    - Set 2 (Inverse, z=1): Clean digits -> Prime inverse (deterministic noise)
    
    The z-coordinate indicates direction:
    - z=0: Model should denoise (forward)
    - z=1: Model should add noise to prime inverse pattern (inverse)
    
    Returns also the digit labels for same-class convex combinations
    """
    print("Preparing MNIST1D data for semi-implicit diffusion training...")
    
    x_train = data['x']
    y_train = data['y']
    
    # Normalize data to [-5, 5] range to match original experiment
    x_normalized = (x_train - x_train.mean()) / x_train.std() * 2.0
    x_normalized = np.clip(x_normalized, -5, 5)
    
    # Use first 40 features
    x_normalized = x_normalized[:, :40]
    
    # Create prime inverse patterns for each digit
    prime_inverse_patterns = create_prime_inverse_patterns(x_normalized, y_train, normalize=True)
    
    # Prepare data for both forward and inverse directions
    n_per_set = n_samples // 2
    
    # Sample balanced across all digits
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
        
        # Select samples for this digit
        n_available = min(len(digit_samples), samples_per_digit)
        selected_indices = np.random.choice(len(digit_samples), n_available, replace=False)
        clean_samples = digit_samples[selected_indices]
        
        # Set 1 (Forward, z=0): Noisy -> Clean
        # We'll add noise during training, so inputs will be created then
        # For now, store clean data which will be noised during training
        forward_inputs.append(clean_samples)
        forward_targets.append(clean_samples)  # Target is clean
        forward_labels.append(np.full(n_available, digit))
        
        # Set 2 (Inverse, z=1): Clean -> Prime Inverse
        inverse_inputs.append(clean_samples)
        # Target is the prime inverse pattern for this digit
        inverse_targets.append(prime_inverse_patterns[digit][:n_available])
        inverse_labels.append(np.full(n_available, digit))
    
    # Stack all samples
    forward_inputs = np.vstack(forward_inputs)
    forward_targets = np.vstack(forward_targets)
    forward_labels = np.concatenate(forward_labels)
    inverse_inputs = np.vstack(inverse_inputs)
    inverse_targets = np.vstack(inverse_targets)
    inverse_labels = np.concatenate(inverse_labels)
    
    # Add z-coordinate
    # Forward set: z=0
    forward_inputs_with_z = np.column_stack([forward_inputs, np.zeros((len(forward_inputs), 1))])
    forward_targets_with_z = np.column_stack([forward_targets, np.zeros((len(forward_targets), 1))])
    
    # Inverse set: z=1
    inverse_inputs_with_z = np.column_stack([inverse_inputs, np.ones((len(inverse_inputs), 1))])
    inverse_targets_with_z = np.column_stack([inverse_targets, np.ones((len(inverse_targets), 1))])
    
    # Combine both sets
    input_points = np.vstack([forward_inputs_with_z, inverse_inputs_with_z])
    target_points = np.vstack([forward_targets_with_z, inverse_targets_with_z])
    labels = np.concatenate([forward_labels, inverse_labels])
    
    print(f"  Set 1 (Forward, z=0): {len(forward_inputs)} noisy->clean samples")
    print(f"  Set 2 (Inverse, z=1): {len(inverse_inputs)} clean->prime_inverse samples")
    print(f"  Total samples: {len(input_points)}")
    print(f"  Input shape: {input_points.shape}")
    print(f"  Target shape: {target_points.shape}")
    
    return input_points, target_points, prime_inverse_patterns, labels

# Load and prepare data
mnist1d_data = load_mnist1d()
input_data, target_data, prime_inverse_patterns, digit_labels = prepare_mnist1d_for_diffusion(mnist1d_data, n_samples=2000)

# Feature dimension (40 features + 1 label)
feature_dim = input_data.shape[1]
print(f"Feature dimension: {feature_dim}")

# =============================================================================
# GAUSSIAN DIFFUSION
# =============================================================================
class GaussianDiffusion:
    """Gaussian diffusion process with linear schedule for beta values"""
    
    def __init__(self, timesteps=100, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0) - the forward diffusion process"""
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
        """Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)"""
        posterior_mean_coef1 = self.betas[t] * self.sqrt_alphas_cumprod_prev[t] / (1.0 - self.alphas_cumprod[t])
        posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev[t]) * torch.sqrt(self.alphas[t]) / (1.0 - self.alphas_cumprod[t])
        
        posterior_mean = (
            posterior_mean_coef1.reshape(-1, 1) * x_start +
            posterior_mean_coef2.reshape(-1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].reshape(-1, 1)
        
        return posterior_mean, posterior_variance
    
    def p_mean_variance(self, model, x_t, t, condition=None):
        """Apply the model to get p(x_{t-1} | x_t)"""
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
        """Generate samples by iterating the reverse process"""
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
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ))
        
        # Middle layer
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x, t, condition=None):
        """
        Args:
            x: Noisy points [batch_size, feature_dim]
            t: Time steps [batch_size]
            condition: Condition points [batch_size, feature_dim]
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
# CONVEX COMBINATION SAMPLING
# =============================================================================
class ConvexCombinationDiffusion:
    """Diffusion process using convex combinations instead of Gaussian noise
    
    For each sample, takes convex combination with another sample of the SAME CLASS
    """
    
    def __init__(self, base_diffusion, labels=None):
        self.timesteps = base_diffusion.timesteps
        self.device = base_diffusion.device
        self.betas = base_diffusion.betas
        self.alphas = base_diffusion.alphas
        self.alphas_cumprod = base_diffusion.alphas_cumprod
        self.alphas_cumprod_prev = base_diffusion.alphas_cumprod_prev
        self.sqrt_alphas_cumprod = base_diffusion.sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = base_diffusion.sqrt_one_minus_alphas_cumprod
        self.sqrt_alphas_cumprod_prev = base_diffusion.sqrt_alphas_cumprod_prev
        self.posterior_variance = base_diffusion.posterior_variance
        self.labels = labels  # Store labels for same-class sampling
    
    def q_sample(self, x_start, t, noise=None, batch_labels=None):
        """Sample using convex combinations of SAME CLASS instead of Gaussian noise"""
        batch_size = x_start.shape[0]
        
        # Create same-class convex combinations
        if batch_labels is not None:
            # For each sample, find another sample with the same label
            x_shuffled = torch.zeros_like(x_start)
            for i in range(batch_size):
                label = batch_labels[i].item()
                # Find all samples with same label
                same_class_mask = (batch_labels == label)
                same_class_indices = torch.where(same_class_mask)[0]
                
                if len(same_class_indices) > 1:
                    # Remove current index
                    same_class_indices = same_class_indices[same_class_indices != i]
                    if len(same_class_indices) > 0:
                        # Randomly select one
                        pair_idx = same_class_indices[torch.randint(0, len(same_class_indices), (1,))].item()
                        x_shuffled[i] = x_start[pair_idx]
                    else:
                        # Fallback: use random permutation if no other same-class sample
                        x_shuffled[i] = x_start[torch.randperm(batch_size)[0]]
                else:
                    # Only one sample of this class, use random permutation
                    x_shuffled[i] = x_start[torch.randperm(batch_size)[0]]
        else:
            # Fallback to random permutation if no labels provided
            indices = torch.randperm(batch_size, device=self.device)
            x_shuffled = x_start[indices]
        
        # Random convex combination weight
        alpha = torch.rand(batch_size, 1, device=self.device)
        convex_combo = alpha * x_start + (1 - alpha) * x_shuffled
        
        # Scale based on timestep
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * convex_combo
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Same as Gaussian diffusion"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Same as Gaussian diffusion"""
        posterior_mean_coef1 = self.betas[t] * self.sqrt_alphas_cumprod_prev[t] / (1.0 - self.alphas_cumprod[t])
        posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev[t]) * torch.sqrt(self.alphas[t]) / (1.0 - self.alphas_cumprod[t])
        
        posterior_mean = (
            posterior_mean_coef1.reshape(-1, 1) * x_start +
            posterior_mean_coef2.reshape(-1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].reshape(-1, 1)
        
        return posterior_mean, posterior_variance
    
    def p_mean_variance(self, model, x_t, t, condition=None):
        """Same as Gaussian diffusion"""
        model_output = model(x_t, t, condition)
        x_start = self.predict_start_from_noise(x_t, t, model_output)
        x_start = torch.clamp(x_start, -5.0, 5.0)
        model_mean, posterior_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        return model_mean, posterior_variance
    
    def p_sample(self, model, x_t, t, condition=None, batch_labels=None):
        """Sample using convex combinations of same class"""
        model_mean, model_variance = self.p_mean_variance(model, x_t, t, condition)
        
        batch_size = x_t.shape[0]
        
        # Create same-class convex combinations for noise
        if batch_labels is not None:
            mean_shuffled = torch.zeros_like(model_mean)
            for i in range(batch_size):
                label = batch_labels[i].item()
                same_class_mask = (batch_labels == label)
                same_class_indices = torch.where(same_class_mask)[0]
                
                if len(same_class_indices) > 1:
                    same_class_indices = same_class_indices[same_class_indices != i]
                    if len(same_class_indices) > 0:
                        pair_idx = same_class_indices[torch.randint(0, len(same_class_indices), (1,))].item()
                        mean_shuffled[i] = model_mean[pair_idx]
                    else:
                        mean_shuffled[i] = model_mean[torch.randperm(batch_size)[0]]
                else:
                    mean_shuffled[i] = model_mean[torch.randperm(batch_size)[0]]
        else:
            indices = torch.randperm(batch_size, device=self.device)
            mean_shuffled = model_mean[indices]
        
        alpha = torch.rand(batch_size, 1, device=self.device)
        noise_replacement = alpha * model_mean + (1 - alpha) * mean_shuffled - model_mean
        
        nonzero_mask = (t != 0).float().reshape(-1, 1)
        
        return model_mean + nonzero_mask * torch.sqrt(model_variance) * noise_replacement
    
    def p_sample_loop(self, model, shape, condition=None, batch_labels=None):
        """Generate samples using convex combination sampling"""
        device = next(model.parameters()).device
        
        # Start with convex combinations
        x_base = torch.randn(shape, device=device)
        
        if batch_labels is not None:
            x_shuffled = torch.zeros_like(x_base)
            for i in range(shape[0]):
                label = batch_labels[i].item()
                same_class_mask = (batch_labels == label)
                same_class_indices = torch.where(same_class_mask)[0]
                
                if len(same_class_indices) > 1:
                    same_class_indices = same_class_indices[same_class_indices != i]
                    if len(same_class_indices) > 0:
                        pair_idx = same_class_indices[torch.randint(0, len(same_class_indices), (1,))].item()
                        x_shuffled[i] = x_base[pair_idx]
                    else:
                        x_shuffled[i] = x_base[torch.randperm(shape[0])[0]]
                else:
                    x_shuffled[i] = x_base[torch.randperm(shape[0])[0]]
        else:
            indices = torch.randperm(shape[0], device=device)
            x_shuffled = x_base[indices]
        
        alpha = torch.rand(shape[0], 1, device=device)
        x_t = alpha * x_base + (1 - alpha) * x_shuffled
        
        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling (Convex)", leave=False):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, condition, batch_labels)
        
        return x_t
    
    def sample(self, model, shape, condition=None, batch_labels=None):
        """Generate samples using convex combinations"""
        return self.p_sample_loop(model, shape, condition, batch_labels)

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
def compute_fid_score(real_samples, generated_samples):
    """
    Compute Fréchet Inception Distance (FID) between real and generated samples
    FID measures the distance between two multivariate Gaussians fitted to 
    real and generated feature distributions
    
    FID = ||mu_real - mu_gen||^2 + Tr(Sigma_real + Sigma_gen - 2*sqrt(Sigma_real @ Sigma_gen))
    """
    real_np = real_samples.detach().cpu().numpy()
    gen_np = generated_samples.detach().cpu().numpy()
    
    # Calculate mean and covariance for real samples
    mu_real = np.mean(real_np, axis=0)
    sigma_real = np.cov(real_np, rowvar=False)
    
    # Calculate mean and covariance for generated samples
    mu_gen = np.mean(gen_np, axis=0)
    sigma_gen = np.cov(gen_np, rowvar=False)
    
    # Calculate mean difference
    mean_diff = np.sum((mu_real - mu_gen) ** 2)
    
    # Calculate covariance term
    # Add small epsilon to diagonal for numerical stability
    eps = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_gen += np.eye(sigma_gen.shape[0]) * eps
    
    # Compute sqrt of product of covariances
    covmean = scipy.linalg.sqrtm(sigma_real @ sigma_gen)
    
    # Handle numerical errors (complex numbers due to floating point errors)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate trace term
    trace_term = np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    fid = mean_diff + trace_term
    
    return torch.tensor(fid, device=real_samples.device, dtype=torch.float32)

def compute_kl_divergence_loss(predicted_samples, target_samples, n_bins=50):
    """
    Compute KL divergence between predicted and target distributions
    """
    pred_np = predicted_samples.detach().cpu().numpy()
    target_np = target_samples.detach().cpu().numpy()
    
    total_kl = 0.0
    
    # Compute KL divergence for each dimension
    for dim in range(predicted_samples.shape[1]):
        pred_hist, bins = np.histogram(pred_np[:, dim], bins=n_bins, density=True)
        target_hist, _ = np.histogram(target_np[:, dim], bins=bins, density=True)
        
        pred_hist = pred_hist + 1e-10
        target_hist = target_hist + 1e-10
        
        pred_hist = pred_hist / pred_hist.sum()
        target_hist = target_hist / target_hist.sum()
        
        kl_div = entropy(target_hist, pred_hist)
        total_kl += kl_div
    return torch.tensor(total_kl, device=predicted_samples.device, dtype=torch.float32)


def compute_per_digit_kl_divergence(predicted_samples, target_samples, labels, n_bins=50):
    """
    Compute KL divergence separately for each digit class and return mean
    This avoids mixing distributions across different digits
    """
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    unique_digits = np.unique(labels_np)
    
    kl_scores = []
    
    for digit in unique_digits:
        mask = labels_np == digit
        if mask.sum() > 1:  # Need at least 2 samples
            pred_digit = predicted_samples[mask]
            target_digit = target_samples[mask]
            
            kl_digit = compute_kl_divergence_loss(pred_digit, target_digit, n_bins)
            kl_scores.append(kl_digit.item())
    
    # Return mean KL across all digits
    if len(kl_scores) > 0:
        return torch.tensor(np.mean(kl_scores), device=predicted_samples.device, dtype=torch.float32)
    else:
        return torch.tensor(0.0, device=predicted_samples.device, dtype=torch.float32)


def compute_per_digit_fid(real_samples, generated_samples, labels):
    """
    Compute FID score separately for each digit class and return mean
    This avoids mixing distributions across different digits
    """
    labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    unique_digits = np.unique(labels_np)
    
    fid_scores = []
    
    for digit in unique_digits:
        mask = labels_np == digit
        if mask.sum() > 5:  # Need enough samples for covariance
            real_digit = real_samples[mask]
            gen_digit = generated_samples[mask]
            
            fid_digit = compute_fid_score(real_digit, gen_digit)
            fid_scores.append(fid_digit.item())
    
    # Return mean FID across all digits
    if len(fid_scores) > 0:
        return torch.tensor(np.mean(fid_scores), device=real_samples.device, dtype=torch.float32)
    else:
        return torch.tensor(0.0, device=real_samples.device, dtype=torch.float32)


def compute_distribution_loss(model, diffusion, input_points, target_points, condition_points, 
                              loss_type='mse', n_samples_for_dist=100):
    """Compute loss based on loss_type"""
    if loss_type == 'mse':
        t = torch.randint(0, diffusion.timesteps, (input_points.shape[0],), device=input_points.device).long()
        noise = torch.randn_like(target_points)
        noisy_points = diffusion.q_sample(target_points, t, noise)
        predicted_noise = model(noisy_points, t, condition_points)
        return F.mse_loss(predicted_noise, noise)
    
    elif loss_type == 'kl_divergence':
        with torch.no_grad():
            generated_samples = diffusion.sample(
                model, 
                shape=(n_samples_for_dist, target_points.shape[1]), 
                condition=condition_points[:n_samples_for_dist] if condition_points is not None else None
            )
        
        kl_loss = compute_kl_divergence_loss(generated_samples, target_points[:n_samples_for_dist])
        
        # Add MSE component for gradient flow
        t = torch.randint(0, diffusion.timesteps, (input_points.shape[0],), device=input_points.device).long()
        noise = torch.randn_like(target_points)
        noisy_points = diffusion.q_sample(target_points, t, noise)
        predicted_noise = model(noisy_points, t, condition_points)
        mse_component = F.mse_loss(predicted_noise, noise)
        
        return 0.1 * mse_component + kl_loss
    
    elif loss_type == 'fid':
        with torch.no_grad():
            generated_samples = diffusion.sample(
                model, 
                shape=(n_samples_for_dist, target_points.shape[1]), 
                condition=condition_points[:n_samples_for_dist] if condition_points is not None else None
            )
        
        fid_loss = compute_fid_score(target_points[:n_samples_for_dist], generated_samples)
        
        # Add MSE component for gradient flow
        t = torch.randint(0, diffusion.timesteps, (input_points.shape[0],), device=input_points.device).long()
        noise = torch.randn_like(target_points)
        noisy_points = diffusion.q_sample(target_points, t, noise)
        predicted_noise = model(noisy_points, t, condition_points)
        mse_component = F.mse_loss(predicted_noise, noise)
        
        return 0.1 * mse_component + 0.01 * fid_loss  # Scale FID since it can be large
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

# =============================================================================
# SEMI-IMPLICIT TRAINING
# =============================================================================
class SemiImplicitTrainer:
    """
    Semi-implicit training with bidirectional learning
    
    z=0: Forward direction (noisy -> clean)
    z=1: Inverse direction (clean -> prime_inverse)
    
    Four losses:
    1. Forward loss: MSE(model(noisy, t, z=0), clean)
    2. Inverse loss: MSE(model(clean, t, z=1), prime_inverse)
    3. Roundtrip forward->inverse: MSE(model(model(clean, t, z=0), t, z=1), prime_inverse)
    4. Roundtrip inverse->forward: MSE(model(model(clean, t, z=1), t, z=0), clean)
    """
    
    def __init__(self, model, diffusion, optimizer, device):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.device = device
    
    def train_step(self, input_points, target_points, condition_points):
        """
        Train with bidirectional self-inverse constraint
        
        Input/target points contain both forward (z=0) and inverse (z=1) samples
        """
        self.optimizer.zero_grad()
        
        # Separate forward and inverse samples based on z-coordinate
        z_values = input_points[:, -1]
        forward_mask = z_values == 0
        inverse_mask = z_values == 1
        
        # Forward samples (z=0): noisy -> clean
        forward_inputs = input_points[forward_mask]
        forward_targets = target_points[forward_mask]
        
        # Inverse samples (z=1): clean -> prime_inverse
        inverse_inputs = input_points[inverse_mask]
        inverse_targets = target_points[inverse_mask]
        
        total_loss = 0
        loss_dict = {}
        
        # =====================================================================
        # LOSS 1: Forward loss - denoise noisy data to clean
        # =====================================================================
        if forward_inputs.shape[0] > 0:
            t_forward = torch.randint(0, self.diffusion.timesteps, (forward_inputs.shape[0],), device=self.device).long()
            noise_forward = torch.randn_like(forward_targets)
            
            # Add noise to clean targets to create noisy inputs
            noisy_inputs = self.diffusion.q_sample(forward_targets, t_forward, noise_forward)
            
            # Create condition with z=0 (forward direction)
            condition_forward = noisy_inputs.clone()
            condition_forward[:, -1] = 0  # Ensure z=0
            
            # Model predicts noise
            predicted_noise_forward = self.model(noisy_inputs, t_forward, condition_forward)
            
            # Forward loss
            forward_loss = F.mse_loss(predicted_noise_forward, noise_forward)
            total_loss += forward_loss
            loss_dict['forward'] = forward_loss.item()
        else:
            loss_dict['forward'] = 0.0
        
        # =====================================================================
        # LOSS 2: Inverse loss - map clean to prime_inverse
        # =====================================================================
        if inverse_inputs.shape[0] > 0:
            t_inverse = torch.randint(0, self.diffusion.timesteps, (inverse_inputs.shape[0],), device=self.device).long()
            
            # For inverse: we want model to learn clean -> prime_inverse
            # Treat prime_inverse as the "noisy" version we're trying to reach
            noise_to_prime = torch.randn_like(inverse_targets)
            noisy_prime = self.diffusion.q_sample(inverse_targets, t_inverse, noise_to_prime)
            
            # Create condition with z=1 (inverse direction)
            condition_inverse = inverse_inputs.clone()
            condition_inverse[:, -1] = 1  # Ensure z=1
            
            # Model predicts noise to get from current state to prime_inverse
            predicted_noise_inverse = self.model(noisy_prime, t_inverse, condition_inverse)
            
            # Inverse loss
            inverse_loss = F.mse_loss(predicted_noise_inverse, noise_to_prime)
            total_loss += inverse_loss
            loss_dict['inverse'] = inverse_loss.item()
        else:
            loss_dict['inverse'] = 0.0
        
        # =====================================================================
        # LOSS 3: Roundtrip forward->inverse
        # Start with clean, denoise (should stay clean), then noise to prime_inverse
        # =====================================================================
        if forward_inputs.shape[0] > 0:
            t_round = torch.randint(0, self.diffusion.timesteps, (forward_inputs.shape[0],), device=self.device).long()
            
            # Start with clean data
            clean_data = forward_targets
            
            # Step 1: Add noise and denoise (forward direction, z=0)
            noise_1 = torch.randn_like(clean_data)
            noisy_1 = self.diffusion.q_sample(clean_data, t_round, noise_1)
            condition_1 = noisy_1.clone()
            condition_1[:, -1] = 0  # z=0 for forward
            
            predicted_noise_1 = self.model(noisy_1, t_round, condition_1)
            denoised = self.diffusion.predict_start_from_noise(noisy_1, t_round, predicted_noise_1)
            denoised = torch.clamp(denoised, -5.0, 5.0)
            
            # Step 2: Now apply inverse direction (z=1) to get to prime_inverse
            noise_2 = torch.randn_like(denoised)
            # Need to match with corresponding prime_inverse targets
            # For simplicity, use the inverse targets from inverse_mask samples
            # This is an approximation since we're mixing samples
            if inverse_inputs.shape[0] > 0:
                # Use first N inverse targets as reference
                n_roundtrip = min(denoised.shape[0], inverse_targets.shape[0])
                target_prime = inverse_targets[:n_roundtrip]
                denoised_subset = denoised[:n_roundtrip]
                
                noisy_2 = self.diffusion.q_sample(target_prime, t_round[:n_roundtrip], noise_2[:n_roundtrip])
                condition_2 = denoised_subset.clone()
                condition_2[:, -1] = 1  # z=1 for inverse
                
                predicted_noise_2 = self.model(noisy_2, t_round[:n_roundtrip], condition_2)
                
                roundtrip_fi_loss = F.mse_loss(predicted_noise_2, noise_2[:n_roundtrip])
                total_loss += 0.5 * roundtrip_fi_loss
                loss_dict['roundtrip_fi'] = roundtrip_fi_loss.item()
            else:
                loss_dict['roundtrip_fi'] = 0.0
        else:
            loss_dict['roundtrip_fi'] = 0.0
        
        # =====================================================================
        # LOSS 4: Roundtrip inverse->forward
        # Start with clean, noise to prime_inverse, then denoise back to clean
        # =====================================================================
        if inverse_inputs.shape[0] > 0:
            t_round_inv = torch.randint(0, self.diffusion.timesteps, (inverse_inputs.shape[0],), device=self.device).long()
            
            # Start with clean data
            clean_data_inv = inverse_inputs[:, :-1]  # Remove z coordinate
            clean_data_inv_with_z = inverse_inputs.clone()
            
            # Step 1: Apply inverse to get to prime_inverse (z=1)
            target_prime_inv = inverse_targets
            noise_inv_1 = torch.randn_like(target_prime_inv)
            noisy_inv_1 = self.diffusion.q_sample(target_prime_inv, t_round_inv, noise_inv_1)
            
            condition_inv_1 = clean_data_inv_with_z.clone()
            condition_inv_1[:, -1] = 1  # z=1
            
            predicted_noise_inv_1 = self.model(noisy_inv_1, t_round_inv, condition_inv_1)
            to_prime = self.diffusion.predict_start_from_noise(noisy_inv_1, t_round_inv, predicted_noise_inv_1)
            to_prime = torch.clamp(to_prime, -5.0, 5.0)
            
            # Step 2: Now denoise back to clean (z=0)
            noise_inv_2 = torch.randn_like(clean_data_inv_with_z)
            # Target should be clean data
            noisy_inv_2 = self.diffusion.q_sample(clean_data_inv_with_z, t_round_inv, noise_inv_2)
            
            condition_inv_2 = to_prime.clone()
            condition_inv_2[:, -1] = 0  # z=0 for forward
            
            predicted_noise_inv_2 = self.model(noisy_inv_2, t_round_inv, condition_inv_2)
            
            roundtrip_if_loss = F.mse_loss(predicted_noise_inv_2, noise_inv_2)
            total_loss += 0.5 * roundtrip_if_loss
            loss_dict['roundtrip_if'] = roundtrip_if_loss.item()
        else:
            loss_dict['roundtrip_if'] = 0.0
        
        # Backward and optimize
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), loss_dict

# =============================================================================
# SEMI-EXPLICIT TRAINING (SVD-based inversion)
# =============================================================================
class SemiExplicitTrainer:
    """
    Semi-explicit training using SVD-based pseudo-inverse
    
    TRAINING: Only trains forward model (noisy -> clean) on forward samples
    INFERENCE: Uses SVD pseudo-inverse to invert the model for inverse direction
    
    Instead of training a separate inverse model or using a z-switch,
    we compute the pseudo-inverse of the forward network using SVD.
    
    For a linear layer: W_inv ≈ (W^T W)^{-1} W^T
    """
    
    def __init__(self, model, diffusion, optimizer, device):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.device = device
    
    def compute_layer_pseudoinverse(self, layer):
        """Compute pseudo-inverse of a linear layer using SVD"""
        if isinstance(layer, nn.Linear):
            W = layer.weight.data  # Shape: (out_features, in_features)
            # Compute SVD: W = U @ S @ V^T
            U, S, Vt = torch.svd(W)
            # Pseudo-inverse: W^+ = V @ S^{-1} @ U^T
            # Add small epsilon for numerical stability
            S_inv = torch.where(S > 1e-5, 1.0 / S, torch.zeros_like(S))
            W_pinv = Vt.t() @ torch.diag(S_inv) @ U.t()
            return W_pinv
        return None
    
    def apply_inverse_pass(self, output, model):
        """
        Apply approximate inverse pass through network using SVD
        This is a simplified version - in practice, we'd need to handle
        nonlinearities and composite layers more carefully
        """
        # For simplicity, we'll use the model's forward pass with z=1
        # and approximate the inverse through the output projection layer
        
        # Get the output projection layer
        output_layer = model.output_proj
        W_inv = self.compute_layer_pseudoinverse(output_layer)
        
        # Apply pseudo-inverse
        if W_inv is not None:
            # Simple linear approximation of inverse
            approx_inverse = F.linear(output, W_inv)
            return approx_inverse
        else:
            # Fallback: return output
            return output
    
    def train_step(self, input_points, target_points, condition_points):
        """
        Train ONLY on forward samples (standard denoising)
        Inverse capability comes from SVD inversion at test time
        """
        self.optimizer.zero_grad()
        
        # Only train on forward samples (z=0)
        z_values = input_points[:, -1]
        forward_mask = z_values == 0
        
        loss_dict = {}
        
        # Forward loss - train the denoising model
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
            
            forward_loss.backward()
            self.optimizer.step()
            
            loss_dict['forward'] = forward_loss.item()
            loss_dict['inverse'] = 0.0  # No inverse training
            
            return forward_loss.item(), loss_dict
        else:
            loss_dict['forward'] = 0.0
            loss_dict['inverse'] = 0.0
            return 0.0, loss_dict

# =============================================================================
# EXPLICIT TRAINING (Mathematical network inversion)
# =============================================================================
class ExplicitTrainer:
    """
    Explicit training with mathematical network inversion
    
    TRAINING: Only trains forward model (noisy -> clean) on forward samples
    INFERENCE: Uses mathematical inversion to invert the model for inverse direction
    
    For networks with invertible activation functions and square weight matrices,
    we can compute exact inverses. This uses the chain rule for composite functions.
    
    For f(x) = σ(Wx + b), the inverse is: f^{-1}(y) = W^{-1}(σ^{-1}(y) - b)
    """
    
    def __init__(self, model, diffusion, optimizer, device):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.device = device
    
    def invert_activation(self, y, activation_type='relu'):
        """
        Invert activation function where possible
        ReLU: inverse is identity for y>0, undefined for y<=0 (we use identity)
        """
        if activation_type == 'relu':
            # ReLU inverse is identity (with assumption that input was positive)
            return y
        elif activation_type == 'tanh':
            # atanh
            return torch.atanh(torch.clamp(y, -0.99, 0.99))
        elif activation_type == 'sigmoid':
            # logit
            y_clamped = torch.clamp(y, 0.01, 0.99)
            return torch.log(y_clamped / (1 - y_clamped))
        else:
            return y
    
    def explicit_inverse_layer(self, layer, y):
        """
        Compute explicit inverse of a linear layer
        For W @ x + b = y, solve for x: x = W^{-1} @ (y - b)
        """
        if isinstance(layer, nn.Linear):
            W = layer.weight.data
            b = layer.bias.data if layer.bias is not None else torch.zeros(W.shape[0], device=self.device)
            
            # Check if matrix is square and invertible
            if W.shape[0] == W.shape[1]:
                try:
                    # Compute inverse
                    W_inv = torch.inverse(W + 1e-5 * torch.eye(W.shape[0], device=self.device))
                    # Apply inverse: x = W^{-1} @ (y - b)
                    y_centered = y - b
                    x = F.linear(y_centered, W_inv)
                    return x
                except:
                    # If inversion fails, use pseudo-inverse
                    W_pinv = torch.pinverse(W)
                    y_centered = y - b
                    x = F.linear(y_centered, W_pinv)
                    return x
            else:
                # Use pseudo-inverse for non-square matrices
                W_pinv = torch.pinverse(W)
                y_centered = y - b
                x = F.linear(y_centered, W_pinv)
                return x
        return y
    
    def apply_network_inverse(self, output, model):
        """
        Apply mathematical inverse through the network
        
        This is approximate since we can't perfectly invert complex architectures,
        but we try to invert the key transformation layers.
        """
        # Start from output and work backwards through output_proj
        x = output
        
        # Invert output projection
        x = self.explicit_inverse_layer(model.output_proj, x)
        
        # For U-Net, we'd need to invert the decoder, middle, and encoder
        # This is simplified - just inverting the output layer as demonstration
        
        return x
    
    def train_step(self, input_points, target_points, condition_points):
        """
        Train ONLY on forward samples (standard denoising)
        Inverse capability comes from mathematical inversion at test time
        """
        self.optimizer.zero_grad()
        
        # Only train on forward samples (z=0)
        z_values = input_points[:, -1]
        forward_mask = z_values == 0
        
        loss_dict = {}
        
        # Forward loss (standard denoising)
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
            
            forward_loss.backward()
            self.optimizer.step()
            
            loss_dict['forward'] = forward_loss.item()
            loss_dict['inverse'] = 0.0  # No inverse training
            
            return forward_loss.item(), loss_dict
        else:
            loss_dict['forward'] = 0.0
            loss_dict['inverse'] = 0.0
            return 0.0, loss_dict

# =============================================================================
# DATASET AND DATALOADER
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

train_dataset = PointDiffusionDataset(input_data, target_data, digit_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f"\nDataset prepared:")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Batches per epoch: {len(train_loader)}")

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================
EXPERIMENT_CONFIG = {
    'loss': ['fid', 'kl_divergence'],
    'sampling': ['gaussian', 'convex'],
    'training': ['implicit', 'semi_implicit', 'semi_explicit', 'explicit']
}

experiment_results = []
experiment_metadata = {
    'start_time': datetime.now().isoformat(),
    'dataset': 'mnist1d',
    'n_samples': len(train_dataset),
    'max_epochs': max_epochs,
    'batch_size': batch_size,
    'timesteps': timesteps,
    'feature_dim': feature_dim,
    'device': str(device)
}

print("\n" + "="*80)
print("EXPERIMENTAL COMPARISON FRAMEWORK")
print("="*80)
print(f"Configuration:")
print(f"  Loss functions: {EXPERIMENT_CONFIG['loss']}")
print(f"  Sampling methods: {EXPERIMENT_CONFIG['sampling']}")
print(f"  Training methods: {EXPERIMENT_CONFIG['training']}")
print(f"\nTotal experiments: 16 (2 loss × 2 sampling × 4 training)")
print(f"Architecture: Iterative (single shared U-Net)")
print(f"Epochs per experiment: {max_epochs}")
print(f"Dataset: MNIST1D with {len(train_dataset)} samples")
print("="*80)

# =============================================================================
# VISUALIZATION AND MODEL SAVING FUNCTIONS
# =============================================================================
def generate_digit_samples(model, inverse_model, diffusion, sampling_type, training_type, device, feature_dim):
    """
    Generate one sample for each digit (0-9) for visual comparison
    Returns dict mapping digit -> generated sample
    """
    digit_samples = {}
    
    model.eval()
    if inverse_model is not None:
        inverse_model.eval()
    
    with torch.no_grad():
        for digit in range(10):
            # Create a condition vector for this digit
            # Use the mean of all samples of this digit from the training data
            digit_mask = digit_labels == digit
            digit_data = input_data[digit_mask]
            
            if len(digit_data) > 0:
                # Use mean as condition
                condition = torch.tensor(digit_data.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)
                
                # Generate sample (forward direction, z=0)
                if training_type == 'implicit' and inverse_model is not None:
                    use_model = model
                else:
                    use_model = model
                
                # Create a dummy label tensor for this digit
                label_tensor = torch.tensor([digit], dtype=torch.long).to(device)
                
                if sampling_type == 'convex':
                    sample = diffusion.sample(
                        use_model,
                        shape=(1, feature_dim),
                        condition=condition,
                        batch_labels=label_tensor
                    )
                else:
                    sample = diffusion.sample(
                        use_model,
                        shape=(1, feature_dim),
                        condition=condition
                    )
                
                digit_samples[digit] = sample[0].cpu().numpy()
    
    return digit_samples


def plot_digit_samples(digit_samples, exp_id, loss_type, sampling_type, training_type):
    """
    Plot generated samples for each digit
    """
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for digit in range(10):
        if digit in digit_samples:
            sample = digit_samples[digit][:-1]  # Remove z-coordinate
            axes[digit].plot(sample)
            axes[digit].set_title(f'Digit {digit}')
            axes[digit].set_ylim(-5, 5)
            axes[digit].grid(True, alpha=0.3)
    
    plt.suptitle(f'Exp {exp_id}: {training_type}_{sampling_type}_{loss_type} - Generated Samples per Digit', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'exp{exp_id:02d}_digit_samples_{training_type}_{sampling_type}_{loss_type}.png'
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved digit samples visualization: {filename}")
    return filepath


def plot_pca_comparison(original_data, synthetic_data, original_labels, exp_id, loss_type, sampling_type, training_type):
    """
    Plot PCA comparison of original vs synthetic data distributions
    """
    # Remove z-coordinate if present
    if original_data.shape[1] > 40:
        original_data_clean = original_data[:, :-1]
    else:
        original_data_clean = original_data
        
    if synthetic_data.shape[1] > 40:
        synthetic_data_clean = synthetic_data[:, :-1]
    else:
        synthetic_data_clean = synthetic_data
    
    # Fit PCA on original data
    pca = PCA(n_components=2)
    original_pca = pca.fit_transform(original_data_clean)
    synthetic_pca = pca.transform(synthetic_data_clean)
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot original data
    scatter1 = axes[0].scatter(original_pca[:, 0], original_pca[:, 1], 
                               c=original_labels, cmap='tab10', alpha=0.6, s=20)
    axes[0].set_title('Original Data (First 2 PCs)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Digit')
    
    # Plot synthetic data
    scatter2 = axes[1].scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], 
                               c=original_labels[:len(synthetic_pca)], cmap='tab10', alpha=0.6, s=20)
    axes[1].set_title('Synthetic Data (First 2 PCs)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Digit')
    
    plt.suptitle(f'Exp {exp_id}: {training_type}_{sampling_type}_{loss_type} - PCA Distribution Comparison', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    filename = f'exp{exp_id:02d}_pca_comparison_{training_type}_{sampling_type}_{loss_type}.png'
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved PCA comparison: {filename}")
    return filepath


def save_models(model, inverse_model, exp_id, loss_type, sampling_type, training_type):
    """
    Save trained model(s) to disk
    """
    model_info = {
        'experiment_id': exp_id,
        'loss_type': loss_type,
        'sampling_type': sampling_type,
        'training_type': training_type,
        'feature_dim': feature_dim,
        'hidden_dim': hidden_dim,
        'embedding_dim': embedding_dim,
        'num_layers': num_layers,
        'timesteps': timesteps
    }
    
    # Save forward model
    forward_filename = f'exp{exp_id:02d}_model_{training_type}_{sampling_type}_{loss_type}.pth'
    forward_path = os.path.join(MODELS_DIR, forward_filename)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_info': model_info,
        'model_type': 'forward' if inverse_model is not None else 'unified'
    }, forward_path)
    
    print(f"  ✓ Saved model: {forward_filename}")
    
    # Save inverse model if it exists (implicit training)
    if inverse_model is not None:
        inverse_filename = f'exp{exp_id:02d}_inverse_model_{training_type}_{sampling_type}_{loss_type}.pth'
        inverse_path = os.path.join(MODELS_DIR, inverse_filename)
        
        torch.save({
            'model_state_dict': inverse_model.state_dict(),
            'model_info': model_info,
            'model_type': 'inverse'
        }, inverse_path)
        
        print(f"  ✓ Saved inverse model: {inverse_filename}")
        return forward_path, inverse_path
    
    return forward_path, None


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
def run_single_experiment(loss_type, sampling_type, training_type, exp_id):
    """Run a single experiment configuration"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT {exp_id}/16")
    print(f"Loss: {loss_type}, Sampling: {sampling_type}, Training: {training_type}")
    print(f"{'='*80}")
    
    # Initialize wandb for this experiment
    wandb_run = wandb.init(
        entity="kjmetzler-worcester-polytechnic-institute",
        project="mnist1d-diffusion-experiments",
        name=f"exp{exp_id}_{loss_type}_{sampling_type}_{training_type}",
        config={
            "experiment_id": exp_id,
            "loss_function": loss_type,
            "sampling_method": sampling_type,
            "training_method": training_type,
            "architecture": "iterative",
            "timesteps": timesteps,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "feature_dim": feature_dim,
            "dataset": "mnist1d"
        },
        reinit=True
    )
    
    start_time = time.time()
    
    # Create model (always iterative architecture)
    model_obj = PointUNet(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        time_dim=embedding_dim,
        condition_dim=feature_dim,
        num_layers=num_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model_obj.parameters())
    
    # Create diffusion process based on sampling type
    if sampling_type == 'gaussian':
        diff_process = GaussianDiffusion(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )
    else:  # convex
        base_diff = GaussianDiffusion(
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )
        diff_process = ConvexCombinationDiffusion(base_diff, labels=digit_labels)
    
    # For implicit training, create a second model for inverse
    inverse_model_obj = None
    if training_type == 'implicit':
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
    
    # Training
    print(f"\nTraining model...")
    loss_history = []
    
    for epoch in range(max_epochs):
        epoch_loss = 0
        epoch_loss_dict = {'forward': 0, 'inverse': 0, 'roundtrip_fi': 0, 'roundtrip_if': 0}
        num_batches = 0
        
        for batch_idx, (input_batch, target_batch, labels_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            if training_type == 'implicit':
                # Implicit training: Train two separate models
                # Model 1: Forward (noisy -> clean)
                # Model 2: Inverse (clean -> prime)
                optimizer.zero_grad()
                
                z_values = input_batch[:, -1]
                forward_mask = z_values == 0
                inverse_mask = z_values == 1
                
                total_batch_loss = 0
                
                # Train forward model
                if forward_mask.sum() > 0:
                    forward_inputs = input_batch[forward_mask]
                    forward_targets = target_batch[forward_mask]
                    forward_labels = labels_batch[forward_mask]
                    
                    t_forward = torch.randint(0, diff_process.timesteps, (forward_inputs.shape[0],), device=device).long()
                    noise_forward = torch.randn_like(forward_targets)
                    
                    # Pass labels for convex sampling
                    if sampling_type == 'convex':
                        noisy_inputs = diff_process.q_sample(forward_targets, t_forward, noise_forward, forward_labels)
                    else:
                        noisy_inputs = diff_process.q_sample(forward_targets, t_forward, noise_forward)
                    
                    condition_forward = noisy_inputs.clone()
                    condition_forward[:, -1] = 0
                    
                    predicted_noise_forward = model_obj(noisy_inputs, t_forward, condition_forward)
                    forward_loss = F.mse_loss(predicted_noise_forward, noise_forward)
                    total_batch_loss += forward_loss
                    epoch_loss_dict['forward'] += forward_loss.item()
                
                # Train inverse model
                if inverse_mask.sum() > 0 and inverse_model_obj is not None:
                    inverse_inputs = input_batch[inverse_mask]
                    inverse_targets = target_batch[inverse_mask]
                    inverse_labels = labels_batch[inverse_mask]
                    
                    t_inverse = torch.randint(0, diff_process.timesteps, (inverse_inputs.shape[0],), device=device).long()
                    noise_inverse = torch.randn_like(inverse_targets)
                    
                    if sampling_type == 'convex':
                        noisy_inverse = diff_process.q_sample(inverse_targets, t_inverse, noise_inverse, inverse_labels)
                    else:
                        noisy_inverse = diff_process.q_sample(inverse_targets, t_inverse, noise_inverse)
                    
                    condition_inverse = noisy_inverse.clone()
                    condition_inverse[:, -1] = 1
                    
                    predicted_noise_inverse = inverse_model_obj(noisy_inverse, t_inverse, condition_inverse)
                    inverse_loss = F.mse_loss(predicted_noise_inverse, noise_inverse)
                    total_batch_loss += inverse_loss
                    epoch_loss_dict['inverse'] += inverse_loss.item()
                
                # Backward and optimize both models
                if total_batch_loss > 0:
                    total_batch_loss.backward()
                    optimizer.step()
                    epoch_loss += total_batch_loss.item()
                
            elif training_type == 'semi_implicit':
                # Semi-implicit training with all four losses
                trainer = SemiImplicitTrainer(model_obj, diff_process, optimizer, device)
                total_loss, loss_dict = trainer.train_step(
                    input_batch, target_batch, input_batch
                )
                epoch_loss += total_loss
                
                for key in epoch_loss_dict:
                    if key in loss_dict:
                        epoch_loss_dict[key] += loss_dict[key]
            
            elif training_type == 'semi_explicit':
                # Semi-explicit training with SVD-based inversion
                trainer = SemiExplicitTrainer(model_obj, diff_process, optimizer, device)
                total_loss, loss_dict = trainer.train_step(
                    input_batch, target_batch, input_batch
                )
                epoch_loss += total_loss
                
                for key in epoch_loss_dict:
                    if key in loss_dict:
                        epoch_loss_dict[key] += loss_dict[key]
            
            elif training_type == 'explicit':
                # Explicit training with mathematical network inversion
                trainer = ExplicitTrainer(model_obj, diff_process, optimizer, device)
                total_loss, loss_dict = trainer.train_step(
                    input_batch, target_batch, input_batch
                )
                epoch_loss += total_loss
                
                for key in epoch_loss_dict:
                    if key in loss_dict:
                        epoch_loss_dict[key] += loss_dict[key]
            
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)
        
        # Log to wandb
        wandb_log = {"epoch": epoch, "train_loss": avg_loss}
        
        if training_type == 'semi_implicit':
            for key in epoch_loss_dict:
                wandb_log[f"train_{key}_loss"] = epoch_loss_dict[key] / num_batches
        
        wandb.log(wandb_log)
        
        if epoch % 10 == 0 or epoch == max_epochs - 1:
            print(f"  Epoch {epoch}/{max_epochs}, Loss: {avg_loss:.6f}")
    
    # Testing
    print(f"\nTesting model...")
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
        
        # CONSISTENT TESTING FOR ALL METHODS:
        # Separate forward and inverse samples based on z-coordinate
        z_values = test_input_tensor[:, -1]
        forward_mask = z_values == 0
        inverse_mask = z_values == 1
        
        generated_samples = torch.zeros_like(test_target_tensor)
        
        # Generate forward samples (z=0: noisy -> clean)
        if forward_mask.sum() > 0:
            if training_type == 'implicit' and inverse_model_obj is not None:
                # Use forward model for implicit method
                use_model = model_obj
            else:
                # Use single model for all other methods
                use_model = model_obj
            
            if sampling_type == 'convex':
                forward_samples = diff_process.sample(
                    use_model,
                    shape=(forward_mask.sum(), feature_dim),
                    condition=test_input_tensor[forward_mask],
                    batch_labels=test_labels_tensor[forward_mask]
                )
            else:
                forward_samples = diff_process.sample(
                    use_model,
                    shape=(forward_mask.sum(), feature_dim),
                    condition=test_input_tensor[forward_mask]
                )
            generated_samples[forward_mask] = forward_samples
        
        # Generate inverse samples (z=1: clean -> prime)
        if inverse_mask.sum() > 0:
            if training_type == 'implicit' and inverse_model_obj is not None:
                # IMPLICIT: Use dedicated inverse model
                use_model = inverse_model_obj
                
                if sampling_type == 'convex':
                    inverse_samples = diff_process.sample(
                        use_model,
                        shape=(inverse_mask.sum(), feature_dim),
                        condition=test_input_tensor[inverse_mask],
                        batch_labels=test_labels_tensor[inverse_mask]
                    )
                else:
                    inverse_samples = diff_process.sample(
                        use_model,
                        shape=(inverse_mask.sum(), feature_dim),
                        condition=test_input_tensor[inverse_mask]
                    )
                    
            elif training_type == 'semi_implicit':
                # SEMI-IMPLICIT: Use same model with z=1
                use_model = model_obj
                
                if sampling_type == 'convex':
                    inverse_samples = diff_process.sample(
                        use_model,
                        shape=(inverse_mask.sum(), feature_dim),
                        condition=test_input_tensor[inverse_mask],
                        batch_labels=test_labels_tensor[inverse_mask]
                    )
                else:
                    inverse_samples = diff_process.sample(
                        use_model,
                        shape=(inverse_mask.sum(), feature_dim),
                        condition=test_input_tensor[inverse_mask]
                    )
                    
            elif training_type == 'semi_explicit':
                # SEMI-EXPLICIT: Denoise clean samples, then apply SVD pseudo-inverse
                clean_inputs = test_input_tensor[inverse_mask][:, :-1]  # Remove z-coordinate
                
                # Add noise and denoise to get denoised representation
                t_test = torch.full((inverse_mask.sum(),), timesteps-1, device=device, dtype=torch.long)
                noise_test = torch.randn_like(clean_inputs)
                
                # Create noisy version
                noisy_test = clean_inputs + 0.1 * noise_test  # Small noise for stability
                noisy_test_with_z = torch.cat([noisy_test, torch.zeros(noisy_test.shape[0], 1, device=device)], dim=1)
                
                # Denoise using forward model
                predicted_noise = model_obj(noisy_test_with_z, t_test, noisy_test_with_z)
                denoised = noisy_test_with_z - predicted_noise
                
                # Apply SVD pseudo-inverse using the trainer's method
                trainer = SemiExplicitTrainer(model_obj, diff_process, None, device)
                inverse_samples = trainer.apply_inverse_pass(denoised, model_obj)
                
            elif training_type == 'explicit':
                # EXPLICIT: Denoise clean samples, then apply mathematical inversion
                clean_inputs = test_input_tensor[inverse_mask][:, :-1]  # Remove z-coordinate
                
                # Add noise and denoise to get denoised representation
                t_test = torch.full((inverse_mask.sum(),), timesteps-1, device=device, dtype=torch.long)
                noise_test = torch.randn_like(clean_inputs)
                
                # Create noisy version
                noisy_test = clean_inputs + 0.1 * noise_test  # Small noise for stability
                noisy_test_with_z = torch.cat([noisy_test, torch.zeros(noisy_test.shape[0], 1, device=device)], dim=1)
                
                # Denoise using forward model
                predicted_noise = model_obj(noisy_test_with_z, t_test, noisy_test_with_z)
                denoised = noisy_test_with_z - predicted_noise
                
                # Apply explicit mathematical inversion using the trainer's method
                trainer = ExplicitTrainer(model_obj, diff_process, None, device)
                inverse_samples = trainer.apply_network_inverse(denoised, model_obj)
            else:
                inverse_samples = torch.zeros((inverse_mask.sum(), feature_dim), device=device)
                
            generated_samples[inverse_mask] = inverse_samples
        
        # =====================================================================
        # ROUNDTRIP METRICS
        # =====================================================================
        # Compute roundtrip errors: forward->inverse->forward and inverse->forward->inverse
        roundtrip_fi_mse = 0.0  # Forward -> Inverse -> Forward
        roundtrip_if_mse = 0.0  # Inverse -> Forward -> Inverse
        
        # Use a subset for roundtrip testing (to save time)
        n_roundtrip_samples = min(20, forward_mask.sum())
        
        if n_roundtrip_samples > 0:
            # Forward -> Inverse -> Forward
            # Start with clean samples, denoise them, invert, then denoise back
            roundtrip_indices = torch.where(forward_mask)[0][:n_roundtrip_samples]
            clean_samples = test_target_tensor[roundtrip_indices]
            
            # Step 1: Denoise (should stay clean)
            denoised_step1 = generated_samples[roundtrip_indices]
            
            # Step 2: Apply inverse transformation
            if training_type == 'semi_explicit':
                trainer = SemiExplicitTrainer(model_obj, diff_process, None, device)
                inverted = trainer.apply_inverse_pass(denoised_step1, model_obj)
            elif training_type == 'explicit':
                trainer = ExplicitTrainer(model_obj, diff_process, None, device)
                inverted = trainer.apply_network_inverse(denoised_step1, model_obj)
            elif training_type == 'semi_implicit':
                # For semi-implicit, change z coordinate and sample
                inverted_cond = denoised_step1.clone()
                inverted_cond[:, -1] = 1
                inverted = diff_process.sample(
                    model_obj, 
                    shape=(n_roundtrip_samples, feature_dim),
                    condition=inverted_cond
                )
            elif training_type == 'implicit' and inverse_model_obj is not None:
                inverted = diff_process.sample(
                    inverse_model_obj,
                    shape=(n_roundtrip_samples, feature_dim),
                    condition=denoised_step1
                )
            else:
                inverted = denoised_step1
            
            # Step 3: Denoise back to clean
            # Use forward model to denoise the inverted samples
            t_rt = torch.full((n_roundtrip_samples,), timesteps//2, device=device, dtype=torch.long)
            noise_rt = torch.randn_like(inverted)
            noisy_rt = diff_process.q_sample(inverted, t_rt, noise_rt)
            
            if training_type == 'implicit' and inverse_model_obj is not None:
                use_model = model_obj
            else:
                use_model = model_obj
                
            predicted_noise_rt = use_model(noisy_rt, t_rt, noisy_rt)
            roundtrip_back = diff_process.predict_start_from_noise(noisy_rt, t_rt, predicted_noise_rt)
            
            # Compute roundtrip error
            roundtrip_fi_mse = F.mse_loss(roundtrip_back, clean_samples).item()
        
        # Inverse -> Forward -> Inverse (if we have inverse samples)
        n_inverse_roundtrip = min(20, inverse_mask.sum())
        if n_inverse_roundtrip > 0:
            roundtrip_inv_indices = torch.where(inverse_mask)[0][:n_inverse_roundtrip]
            clean_samples_inv = test_input_tensor[roundtrip_inv_indices][:, :-1]  # Original clean
            prime_targets = test_target_tensor[roundtrip_inv_indices]  # Target prime inverse
            
            # Step 1: Get the generated inverse samples
            generated_prime = generated_samples[roundtrip_inv_indices]
            
            # Step 2: Apply forward (denoise)
            t_rt_inv = torch.full((n_inverse_roundtrip,), timesteps//2, device=device, dtype=torch.long)
            noise_rt_inv = torch.randn_like(generated_prime)
            noisy_rt_inv = diff_process.q_sample(generated_prime, t_rt_inv, noise_rt_inv)
            
            predicted_noise_rt_inv = model_obj(noisy_rt_inv, t_rt_inv, noisy_rt_inv)
            denoised_inv = diff_process.predict_start_from_noise(noisy_rt_inv, t_rt_inv, predicted_noise_rt_inv)
            
            # Step 3: Apply inverse again
            if training_type == 'semi_explicit':
                trainer = SemiExplicitTrainer(model_obj, diff_process, None, device)
                roundtrip_prime = trainer.apply_inverse_pass(denoised_inv, model_obj)
            elif training_type == 'explicit':
                trainer = ExplicitTrainer(model_obj, diff_process, None, device)
                roundtrip_prime = trainer.apply_network_inverse(denoised_inv, model_obj)
            elif training_type == 'semi_implicit':
                inverted_cond_inv = denoised_inv.clone()
                inverted_cond_inv[:, -1] = 1
                roundtrip_prime = diff_process.sample(
                    model_obj,
                    shape=(n_inverse_roundtrip, feature_dim),
                    condition=inverted_cond_inv
                )
            elif training_type == 'implicit' and inverse_model_obj is not None:
                roundtrip_prime = diff_process.sample(
                    inverse_model_obj,
                    shape=(n_inverse_roundtrip, feature_dim),
                    condition=denoised_inv
                )
            else:
                roundtrip_prime = denoised_inv
            
            # Compute roundtrip error
            roundtrip_if_mse = F.mse_loss(roundtrip_prime, prime_targets).item()
        
        # Calculate overall metrics (combined forward + inverse)
        mse = F.mse_loss(generated_samples, test_target_tensor).item()
        
        gen_np = generated_samples.cpu().numpy()
        target_np = test_target_tensor.cpu().numpy()
        
        # Mean absolute error
        mae = np.mean(np.abs(gen_np - target_np))
        
        # Distribution similarity metrics - COMPUTED PER-DIGIT
        # This avoids mixing distributions across different digit classes
        kl_div = compute_per_digit_kl_divergence(generated_samples, test_target_tensor, test_labels_tensor).item()
        fid_score = compute_per_digit_fid(test_target_tensor, generated_samples, test_labels_tensor).item()
        
        # Calculate metrics separately for forward and inverse directions
        if forward_mask.sum() > 0:
            forward_mse = F.mse_loss(generated_samples[forward_mask], test_target_tensor[forward_mask]).item()
            # Per-digit FID for forward direction only
            forward_fid = compute_per_digit_fid(
                test_target_tensor[forward_mask], 
                generated_samples[forward_mask],
                test_labels_tensor[forward_mask]
            ).item()
            # Per-digit KL for forward direction only
            forward_kl = compute_per_digit_kl_divergence(
                generated_samples[forward_mask],
                test_target_tensor[forward_mask],
                test_labels_tensor[forward_mask]
            ).item()
        else:
            forward_mse = 0.0
            forward_fid = 0.0
            forward_kl = 0.0
            
        if inverse_mask.sum() > 0:
            inverse_mse = F.mse_loss(generated_samples[inverse_mask], test_target_tensor[inverse_mask]).item()
            # Per-digit FID for inverse direction only
            inverse_fid = compute_per_digit_fid(
                test_target_tensor[inverse_mask], 
                generated_samples[inverse_mask],
                test_labels_tensor[inverse_mask]
            ).item()
            # Per-digit KL for inverse direction only
            inverse_kl = compute_per_digit_kl_divergence(
                generated_samples[inverse_mask],
                test_target_tensor[inverse_mask],
                test_labels_tensor[inverse_mask]
            ).item()
        else:
            inverse_mse = 0.0
            inverse_fid = 0.0
            inverse_kl = 0.0
    
    runtime = time.time() - start_time
    
    # Store results
    result = {
        'experiment_id': exp_id,
        'architecture': 'iterative',  # Fixed to iterative
        'loss_function': loss_type,
        'sampling_method': sampling_type,
        'training_method': training_type,
        'num_parameters': total_params,
        'feature_dim': feature_dim,
        'final_training_loss': loss_history[-1],
        'test_mse': mse,
        'test_mae': mae,
        'test_kl_divergence': kl_div,  # Per-digit mean
        'test_fid_score': fid_score,    # Per-digit mean
        'test_forward_mse': forward_mse,
        'test_forward_fid': forward_fid,  # Per-digit mean for forward
        'test_forward_kl': forward_kl,    # Per-digit mean for forward
        'test_inverse_mse': inverse_mse,
        'test_inverse_fid': inverse_fid,  # Per-digit mean for inverse
        'test_inverse_kl': inverse_kl,    # Per-digit mean for inverse
        'test_roundtrip_fi_mse': roundtrip_fi_mse,  # Forward->Inverse->Forward
        'test_roundtrip_if_mse': roundtrip_if_mse,  # Inverse->Forward->Inverse
        'runtime_seconds': runtime,
        'epochs': max_epochs,
        'convergence_rate': (loss_history[0] - loss_history[-1]) / max_epochs if len(loss_history) > 0 else 0
    }
    
    # Log final test metrics to wandb
    wandb.log({
        "test_mse": mse,
        "test_mae": mae,
        "test_kl_divergence": kl_div,
        "test_fid_score": fid_score,
        "test_forward_mse": forward_mse,
        "test_forward_fid": forward_fid,
        "test_forward_kl": forward_kl,
        "test_inverse_mse": inverse_mse,
        "test_inverse_fid": inverse_fid,
        "test_inverse_kl": inverse_kl,
        "test_roundtrip_fi_mse": roundtrip_fi_mse,
        "test_roundtrip_if_mse": roundtrip_if_mse,
        "runtime_seconds": runtime
    })
    
    print(f"\nResults:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Test MSE: {mse:.6f}")
    print(f"  Test MAE: {mae:.6f}")
    print(f"  Test KL Divergence (per-digit mean): {kl_div:.6f}")
    print(f"  Test FID Score (per-digit mean): {fid_score:.6f}")
    print(f"  Test Forward MSE: {forward_mse:.6f}")
    print(f"  Test Forward FID (per-digit): {forward_fid:.6f}")
    print(f"  Test Forward KL (per-digit): {forward_kl:.6f}")
    print(f"  Test Inverse MSE: {inverse_mse:.6f}")
    print(f"  Test Inverse FID (per-digit): {inverse_fid:.6f}")
    print(f"  Test Inverse KL (per-digit): {inverse_kl:.6f}")
    print(f"  Test Roundtrip F->I->F MSE: {roundtrip_fi_mse:.6f}")
    print(f"  Test Roundtrip I->F->I MSE: {roundtrip_if_mse:.6f}")
    print(f"  Runtime: {runtime:.1f}s")
    
    # ==========================================================================
    # SAVE MODELS AND GENERATE VISUALIZATIONS
    # ==========================================================================
    print(f"\nGenerating visualizations and saving models...")
    
    # 1. Save trained models
    forward_model_path, inverse_model_path = save_models(
        model_obj, inverse_model_obj, exp_id, loss_type, sampling_type, training_type
    )
    result['forward_model_path'] = forward_model_path
    result['inverse_model_path'] = inverse_model_path
    
    # 2. Generate and plot digit samples
    digit_samples = generate_digit_samples(
        model_obj, inverse_model_obj, diff_process, sampling_type, training_type, device, feature_dim
    )
    digit_samples_path = plot_digit_samples(
        digit_samples, exp_id, loss_type, sampling_type, training_type
    )
    result['digit_samples_plot'] = digit_samples_path
    
    # 3. Generate PCA comparison plot
    pca_plot_path = plot_pca_comparison(
        test_target_tensor.cpu().numpy(),
        generated_samples.cpu().numpy(),
        test_labels_tensor.cpu().numpy(),
        exp_id, loss_type, sampling_type, training_type
    )
    result['pca_comparison_plot'] = pca_plot_path
    
    # Upload images to wandb
    print(f"\nUploading visualizations to Weights & Biases...")
    wandb.log({
        "forward_model_path": forward_model_path,
        "digit_samples": wandb.Image(digit_samples_path, caption=f"Exp {exp_id}: Generated samples per digit"),
        "pca_comparison": wandb.Image(pca_plot_path, caption=f"Exp {exp_id}: PCA distribution comparison")
    })
    print(f"  ✓ Uploaded images to wandb")
    
    # Finish wandb run
    wandb_run.finish()
    
    return result, loss_history

# =============================================================================
# RUN ALL EXPERIMENTS
# =============================================================================
print("\n" + "="*80)
print("STARTING COMPREHENSIVE EXPERIMENTAL SWEEP")
print("="*80)
print(f"Total experiments: 16")
print(f"Dataset: MNIST1D")
print("="*80)

all_results = []
all_loss_histories = {}

exp_id = 1
for loss in EXPERIMENT_CONFIG['loss']:
    for sampling in EXPERIMENT_CONFIG['sampling']:
        for training in EXPERIMENT_CONFIG['training']:
            try:
                result, loss_hist = run_single_experiment(
                    loss, sampling, training, exp_id
                )
                
                all_results.append(result)
                all_loss_histories[exp_id] = loss_hist
                
                # Save intermediate results
                if exp_id % 2 == 0:
                    partial_path = os.path.join(RESULTS_DIR, 'mnist1d_experiment_results_partial.json')
                    with open(partial_path, 'w') as f:
                        json.dump({
                            'metadata': experiment_metadata,
                            'results': all_results
                        }, f, indent=2, cls=NumpyEncoder)
                    print(f"\n  ✓ Saved intermediate results after {exp_id} experiments")
                
            except Exception as e:
                print(f"\n  ✗ ERROR in experiment {exp_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                all_results.append({
                    'experiment_id': exp_id,
                    'architecture': 'iterative',
                    'loss_function': loss,
                    'sampling_method': sampling,
                    'training_method': training,
                    'error': str(e),
                    'status': 'failed'
                })
            
            exp_id += 1

experiment_metadata['end_time'] = datetime.now().isoformat()
experiment_metadata['total_experiments'] = len(all_results)

print("\n" + "="*80)
print("EXPERIMENTAL SWEEP COMPLETE!")
print("="*80)
print(f"Completed: {len([r for r in all_results if 'error' not in r])}/{len(all_results)} experiments")
print(f"Failed: {len([r for r in all_results if 'error' in r])}/{len(all_results)} experiments")
print("="*80)

# =============================================================================
# SAVE RESULTS
# =============================================================================
results_filename = os.path.join(RESULTS_DIR, f"mnist1d_diffusion_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

final_results = {
    'metadata': experiment_metadata,
    'results': all_results,
    'summary': {
        'total_experiments': len(all_results),
        'successful': len([r for r in all_results if 'error' not in r]),
        'failed': len([r for r in all_results if 'error' in r]),
        'configurations': EXPERIMENT_CONFIG
    }
}

with open(results_filename, 'w') as f:
    json.dump(final_results, f, indent=2, cls=NumpyEncoder)

print(f"\n✓ Complete results saved to: {results_filename}")

# =============================================================================
# CREATE SUMMARY REPORT
# =============================================================================
report_filename = os.path.join(RESULTS_DIR, f"mnist1d_experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

successful_results = [r for r in all_results if 'error' not in r]

with open(report_filename, 'w') as f:
    f.write("# MNIST1D Diffusion Model Experimental Results\n\n")
    f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Dataset**: MNIST1D (digits 0-9)\n")
    f.write(f"**Total Experiments**: {len(all_results)}\n")
    f.write(f"**Successful**: {len(successful_results)}\n")
    f.write(f"**Failed**: {len(all_results) - len(successful_results)}\n\n")
    
    f.write("## Configuration\n\n")
    f.write(f"- Architecture: Iterative (single U-Net)\n")
    f.write(f"- Loss Functions: {', '.join(EXPERIMENT_CONFIG['loss'])}\n")
    f.write(f"- Sampling Methods: {', '.join(EXPERIMENT_CONFIG['sampling'])}\n")
    f.write(f"- Training Methods: {', '.join(EXPERIMENT_CONFIG['training'])}\n")
    f.write(f"- Epochs: {max_epochs}\n")
    f.write(f"- Batch Size: {batch_size}\n")
    f.write(f"- Feature Dimension: {feature_dim}\n\n")
    
    if successful_results:
        f.write("## Results Summary\n\n")
        f.write("| Exp | Loss | Sampling | Training | Test MSE | Test KL | Runtime (s) |\n")
        f.write("|-----|------|----------|----------|----------|---------|-------------|\n")
        
        for r in successful_results:
            f.write(f"| {r['experiment_id']} | {r['loss_function']} | {r['sampling_method']} | ")
            f.write(f"{r['training_method']} | {r['test_mse']:.6f} | ")
            f.write(f"{r['test_kl_divergence']:.6f} | {r['runtime_seconds']:.1f} |\n")
        
        f.write("\n## Best Configurations\n\n")
        
        best_mse = min(successful_results, key=lambda x: x['test_mse'])
        f.write(f"**Best MSE**: Experiment {best_mse['experiment_id']}\n")
        f.write(f"- Configuration: {best_mse['loss_function']}, {best_mse['sampling_method']}, {best_mse['training_method']}\n")
        f.write(f"- Test MSE: {best_mse['test_mse']:.6f}\n\n")
        
        best_kl = min(successful_results, key=lambda x: x['test_kl_divergence'])
        f.write(f"**Best KL Divergence**: Experiment {best_kl['experiment_id']}\n")
        f.write(f"- Configuration: {best_kl['loss_function']}, {best_kl['sampling_method']}, {best_kl['training_method']}\n")
        f.write(f"- Test KL: {best_kl['test_kl_divergence']:.6f}\n\n")

print(f"✓ Report saved to: {report_filename}")
print("\nExperiment complete!")
