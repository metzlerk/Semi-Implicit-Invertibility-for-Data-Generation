"""
Generate synthetic MNIST1D samples from a saved model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle
import requests
import os


class TimeEmbedding(torch.nn.Module):
    """Sinusoidal time embedding"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PointUNet(torch.nn.Module):
    """U-Net for denoising points"""
    
    def __init__(self, input_dim=41, hidden_dim=128, time_dim=64, condition_dim=41, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding
        self.condition_embed = torch.nn.Sequential(
            torch.nn.Linear(condition_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        
        # Encoder layers
        self.encoder_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.encoder_layers.append(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            ))
        
        # Middle layer
        self.middle = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU()
        )
        
        # Decoder layers
        self.decoder_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 2, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU()
            ))
        
        # Output projection
        self.output_proj = torch.nn.Linear(hidden_dim, input_dim)
        
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


class GaussianDiffusion:
    """Gaussian diffusion process."""
    
    def __init__(self, timesteps=100, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def p_sample(self, model, x_t, t, condition, device):
        """Sample x_{t-1} from p(x_{t-1} | x_t)."""
        batch_size = x_t.shape[0]
        
        # Predict noise
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        predicted_noise = model(x_t, t_tensor, condition)
        
        # Calculate mean
        alpha = self.alphas[t].to(device)
        alpha_cumprod = self.alphas_cumprod[t].to(device)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].to(device)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].to(device)
        
        mean = (x_t - (1 - alpha) / sqrt_one_minus_alpha_cumprod * predicted_noise) / torch.sqrt(alpha)
        
        if t == 0:
            return mean
        else:
            variance = self.posterior_variance[t].to(device)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise
    
    def p_sample_loop(self, model, shape, condition, device):
        """Generate samples by iterating the reverse process."""
        batch_size = shape[0]
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for t in reversed(range(self.timesteps)):
            x_t = self.p_sample(model, x_t, t, condition, device)
        
        return x_t


def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a saved model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model info (structure changed in newer version)
    if 'model_info' in checkpoint:
        info = checkpoint['model_info']
        input_dim = info.get('feature_dim', 41)
        hidden_dim = info.get('hidden_dim', 64)
        embedding_dim = info.get('embedding_dim', 32)
        num_layers = info.get('num_layers', 2)
        timesteps = info.get('timesteps', 100)
        training_method = info.get('training_type', 'unknown')
        sampling_method = info.get('sampling_type', 'unknown')
        loss_type = info.get('loss_type', 'unknown')
    else:
        # Old format
        input_dim = checkpoint.get('input_dim', 41)
        hidden_dim = checkpoint.get('hidden_dim', 64)
        embedding_dim = checkpoint.get('embedding_dim', 32)
        num_layers = checkpoint.get('num_layers', 2)
        timesteps = checkpoint.get('timesteps', 100)
        training_method = checkpoint.get('training_method', 'unknown')
        sampling_method = checkpoint.get('sampling_method', 'unknown')
        loss_type = checkpoint.get('loss_type', 'unknown')
    
    # Create model with saved architecture
    model = PointUNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        time_dim=embedding_dim,
        condition_dim=input_dim,
        num_layers=num_layers
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Architecture: input_dim={input_dim}, hidden_dim={hidden_dim}, "
          f"embedding_dim={embedding_dim}, num_layers={num_layers}")
    print(f"Training method: {training_method}")
    print(f"Sampling method: {sampling_method}")
    print(f"Loss type: {loss_type}")
    print(f"Timesteps: {timesteps}")
    
    return model, {'timesteps': timesteps, 'training_method': training_method, 
                   'sampling_method': sampling_method, 'loss_type': loss_type,
                   'input_dim': input_dim}


def generate_samples(model, num_samples, z_value, timesteps, input_dim, device):
    """Generate synthetic samples."""
    diffusion = GaussianDiffusion(timesteps=timesteps)
    
    with torch.no_grad():
        # Create condition tensor (z-coordinate embedded as condition)
        # For semi-implicit models, z=0 means forward (noisy->clean), z=1 means inverse
        condition = torch.full((num_samples, input_dim), z_value, device=device, dtype=torch.float32)
        
        # Generate samples
        samples = diffusion.p_sample_loop(
            model,
            shape=(num_samples, input_dim),
            condition=condition,
            device=device
        )
    
    return samples.cpu().numpy()


def plot_samples(samples, labels=None, save_path=None):
    """Plot generated samples."""
    num_samples = min(10, len(samples))
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].plot(samples[i])
        if labels is not None:
            axes[i].set_title(f"Digit {labels[i]}")
        else:
            axes[i].set_title(f"Sample {i+1}")
        axes[i].set_xlabel("Feature Index")
        axes[i].set_ylabel("Value")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def load_mnist1d():
    """Load MNIST1D dataset from GitHub"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    data_path = os.path.join(root_dir, 'mnist1d_data.pkl')
    
    try:
        # Try to load from local file first
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print("Loaded MNIST1D from local cache")
    except:
        print("Downloading MNIST1D from GitHub...")
        # Download the dataset
        url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
        response = requests.get(url)
        data = pickle.loads(response.content)
        
        # Save locally for future use
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        print("Downloaded and cached locally")
    
    return data


def compare_with_real_data(synthetic_samples, save_path=None):
    """Compare synthetic samples with real MNIST1D data."""
    # Load real data
    data = load_mnist1d()
    real_samples = data['x'][:10]  # First 10 real samples
    real_labels = data['y'][:10]
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    
    # Plot real samples (top 2 rows)
    for i in range(10):
        row = i // 5
        col = i % 5
        axes[row, col].plot(real_samples[i], color='blue', alpha=0.7)
        axes[row, col].set_title(f"Real: Digit {real_labels[i]}")
        axes[row, col].set_xlabel("Feature")
        axes[row, col].set_ylabel("Value")
    
    # Plot synthetic samples (bottom 2 rows)
    for i in range(10):
        row = 2 + i // 5
        col = i % 5
        axes[row, col].plot(synthetic_samples[i], color='red', alpha=0.7)
        axes[row, col].set_title(f"Synthetic: Sample {i+1}")
        axes[row, col].set_xlabel("Feature")
        axes[row, col].set_ylabel("Value")
    
    plt.suptitle("Real MNIST1D (Blue) vs Synthetic (Red)", fontsize=16, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    # Configuration
    model_path = "models/exp02_model_semi_implicit_gaussian_fid.pth"  # Gaussian semi-implicit with FID loss
    num_samples = 100
    z_value = 0.0  # 0.0 for forward direction, 1.0 for inverse
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Generating {num_samples} samples from {model_path}")
    print(f"Z-coordinate: {z_value} (0.0=forward, 1.0=inverse)")
    print()
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    timesteps = checkpoint['timesteps']
    input_dim = checkpoint['input_dim']
    
    print()
    print("Generating samples...")
    
    # Generate samples
    samples = generate_samples(model, num_samples, z_value, timesteps, input_dim, device)
    
    print(f"Generated {len(samples)} samples")
    print(f"Sample shape: {samples.shape}")
    print(f"Sample statistics:")
    print(f"  Mean: {samples.mean():.4f}")
    print(f"  Std: {samples.std():.4f}")
    print(f"  Min: {samples.min():.4f}")
    print(f"  Max: {samples.max():.4f}")
    print()
    
    # Create output directory
    output_dir = Path("generated_samples")
    output_dir.mkdir(exist_ok=True)
    
    # Save samples
    np.save(output_dir / "synthetic_samples.npy", samples)
    print(f"Saved samples to {output_dir / 'synthetic_samples.npy'}")
    
    # Plot first 10 samples
    plot_samples(samples[:10], save_path=output_dir / "sample_visualization.png")
    
    # Compare with real data
    compare_with_real_data(samples[:10], save_path=output_dir / "real_vs_synthetic.png")
    
    print()
    print("Done! Check the 'generated_samples' directory for outputs.")


if __name__ == "__main__":
    main()
