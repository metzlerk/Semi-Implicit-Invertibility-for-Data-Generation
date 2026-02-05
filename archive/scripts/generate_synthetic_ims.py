"""
Generate synthetic IMS spectra from trained diffusion model
Saves synthetic data in multiple formats for testing
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm

# Import model architecture from training script
import sys
sys.path.append(os.path.dirname(__file__))

# Model architecture definitions (copied from training script)
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

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

class GaussianDiffusion:
    def __init__(self, timesteps=100, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    
    def p_sample(self, model, x, t, condition=None):
        """Sample from p(x_{t-1} | x_t)"""
        betas_t = self.betas[t.cpu()].to(x.device).view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t.cpu()].to(x.device).view(-1, 1)
        sqrt_alphas_t = torch.sqrt(self.alphas[t.cpu()]).to(x.device).view(-1, 1)
        
        # Predict noise
        predicted_noise = model(x, t, condition)
        
        # Compute mean
        model_mean = (x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t) / sqrt_alphas_t
        
        if t[0] == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            # Posterior variance
            posterior_variance = betas_t
            return model_mean + torch.sqrt(posterior_variance) * noise
    
    def p_sample_loop(self, model, shape, condition=None, device='cuda'):
        """Generate samples by denoising from pure noise"""
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, condition)
        
        return x

def load_model_and_data():
    """Load trained model and data statistics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = 'models/best_ims_model.pth'
    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    model_config = checkpoint['model_config']
    
    print(f"\nModel Config:")
    print(f"  Feature dimension: {model_config['feature_dim']}")
    print(f"  Hidden dimension: {model_config['hidden_dim']}")
    print(f"  Embedding dimension: {model_config['embedding_dim']}")
    print(f"  Number of layers: {model_config['num_layers']}")
    
    print(f"\nData Info:")
    print(f"  IMS features: {model_config['num_ims_features']}")
    print(f"  Classes: {model_config['num_classes']}")
    
    # Load class names from results
    results_path = 'results/ims_diffusion_results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    class_names = results['class_names']
    print(f"  Class names: {class_names}")
    
    # Create data_info dict
    data_info = {
        'num_ims_features': model_config['num_ims_features'],
        'num_classes': model_config['num_classes'],
        'class_names': class_names
    }
    
    # Initialize model
    model = IMSUNet(
        input_dim=model_config['feature_dim'],
        hidden_dim=model_config['hidden_dim'],
        time_dim=model_config['embedding_dim'],
        num_layers=model_config['num_layers']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize diffusion (use same timesteps as training)
    diffusion = GaussianDiffusion(timesteps=100)
    
    return model, diffusion, data_info, device

def generate_synthetic_spectra(model, diffusion, data_info, device, 
                               num_samples_per_class=100, output_dir='synthetic_ims_data'):
    """Generate synthetic IMS spectra for all classes"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_ims_features = data_info['num_ims_features']
    num_classes = data_info['num_classes']
    class_names = data_info['class_names']
    
    all_synthetic_ims = []
    all_labels = []
    all_class_names = []
    
    print(f"\nGenerating {num_samples_per_class} samples per class...")
    
    with torch.no_grad():
        for class_idx in tqdm(range(num_classes), desc="Classes"):
            class_name = class_names[class_idx]
            
            # Generate synthetic spectra for this class
            for batch_start in range(0, num_samples_per_class, 100):
                batch_size = min(100, num_samples_per_class - batch_start)
                
                # Create one-hot encoding for this class
                onehot = torch.zeros(batch_size, num_classes, device=device)
                onehot[:, class_idx] = 1.0
                
                # Create condition: [zeros_for_IMS | onehot | z=0]
                condition = torch.cat([
                    torch.zeros(batch_size, num_ims_features, device=device),
                    onehot,
                    torch.zeros(batch_size, 1, device=device)  # z=0 for generation
                ], dim=1)
                
                # Generate samples
                generated = diffusion.p_sample_loop(
                    model, 
                    (batch_size, num_ims_features + num_classes + 1),
                    condition=condition,
                    device=device
                )
                
                # Extract IMS spectra (first num_ims_features dimensions)
                generated_ims = generated[:, :num_ims_features].cpu().numpy()
                
                all_synthetic_ims.append(generated_ims)
                all_labels.extend([class_idx] * batch_size)
                all_class_names.extend([class_name] * batch_size)
    
    # Concatenate all samples
    all_synthetic_ims = np.vstack(all_synthetic_ims)
    all_labels = np.array(all_labels)
    
    print(f"\nGenerated {len(all_synthetic_ims)} total synthetic spectra")
    print(f"Shape: {all_synthetic_ims.shape}")
    
    # Save in multiple formats
    print(f"\nSaving to {output_dir}/...")
    
    # 1. NumPy array (.npy) - fastest for Python
    np.save(os.path.join(output_dir, 'synthetic_ims_spectra.npy'), all_synthetic_ims)
    np.save(os.path.join(output_dir, 'synthetic_labels.npy'), all_labels)
    print("  ✓ Saved .npy files")
    
    # 2. Pandas DataFrame (.feather) - compatible with original data format
    # Split into positive and negative ion modes
    n_features_per_mode = num_ims_features // 2
    
    df_data = {}
    for i in range(n_features_per_mode):
        df_data[f'p_{i+184}'] = all_synthetic_ims[:, i]
    for i in range(n_features_per_mode):
        df_data[f'n_{i+184}'] = all_synthetic_ims[:, n_features_per_mode + i]
    
    # Add one-hot encodings
    for class_idx, class_name in enumerate(class_names):
        df_data[class_name] = (all_labels == class_idx).astype(int)
    
    df = pd.DataFrame(df_data)
    df.to_feather(os.path.join(output_dir, 'synthetic_ims_spectra.feather'))
    print("  ✓ Saved .feather file")
    
    # 3. CSV (.csv) - human-readable, widely compatible
    df_csv = df.copy()
    df_csv['class_name'] = [class_names[label] for label in all_labels]
    df_csv.to_csv(os.path.join(output_dir, 'synthetic_ims_spectra.csv'), index=False)
    print("  ✓ Saved .csv file")
    
    # 4. Metadata JSON
    metadata = {
        'num_samples': len(all_synthetic_ims),
        'num_samples_per_class': num_samples_per_class,
        'num_features': num_ims_features,
        'num_classes': num_classes,
        'class_names': class_names,
        'class_distribution': {
            class_names[i]: int(np.sum(all_labels == i)) 
            for i in range(num_classes)
        },
        'feature_ranges': {
            'min': float(all_synthetic_ims.min()),
            'max': float(all_synthetic_ims.max()),
            'mean': float(all_synthetic_ims.mean()),
            'std': float(all_synthetic_ims.std())
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ Saved metadata.json")
    
    print("\n" + "="*60)
    print("SYNTHETIC DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"\nGenerated files in {output_dir}/:")
    print(f"  - synthetic_ims_spectra.npy (NumPy array)")
    print(f"  - synthetic_labels.npy (NumPy array)")
    print(f"  - synthetic_ims_spectra.feather (Pandas DataFrame)")
    print(f"  - synthetic_ims_spectra.csv (CSV)")
    print(f"  - metadata.json (Dataset info)")
    print("\nUsage examples:")
    print("  # NumPy:")
    print(f"  spectra = np.load('{output_dir}/synthetic_ims_spectra.npy')")
    print(f"  labels = np.load('{output_dir}/synthetic_labels.npy')")
    print("\n  # Pandas:")
    print(f"  df = pd.read_feather('{output_dir}/synthetic_ims_spectra.feather')")
    print("="*60)
    
    return all_synthetic_ims, all_labels

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic IMS spectra')
    parser.add_argument('--samples-per-class', type=int, default=100,
                       help='Number of samples to generate per chemical class (default: 100)')
    parser.add_argument('--output-dir', type=str, default='synthetic_ims_data',
                       help='Output directory for synthetic data (default: synthetic_ims_data)')
    args = parser.parse_args()
    
    # Load model
    model, diffusion, data_info, device = load_model_and_data()
    
    # Generate synthetic data
    synthetic_ims, labels = generate_synthetic_spectra(
        model, diffusion, data_info, device,
        num_samples_per_class=args.samples_per_class,
        output_dir=args.output_dir
    )
