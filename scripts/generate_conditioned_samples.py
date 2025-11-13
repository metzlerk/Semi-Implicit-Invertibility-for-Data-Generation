"""
Generate digit-conditioned samples from saved models using the same method as training.
This uses the digit mean as conditioning and labels for convex sampling.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os
import sys

# Reuse classes from the training script
sys.path.append(str(Path(__file__).parent))


def load_mnist1d():
    """Load MNIST1D dataset"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    data_path = os.path.join(root_dir, 'mnist1d_data.pkl')
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


# Import necessary classes
exec(open('/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/scripts/diffusion_mnist1d_experiments.py').read().split('# EXPERIMENT CONFIGURATION')[0])

def generate_conditioned_digit_samples(model_path, output_dir, device='cpu'):
    """Generate one sample per digit using proper conditioning"""
    
    # Load the saved model
    checkpoint = torch.load(model_path, map_location=device)
    model_info = checkpoint['model_info']
    
    print(f"Loading model: {model_path}")
    print(f"  Training: {model_info['training_type']}")
    print(f"  Sampling: {model_info['sampling_type']}")
    print(f"  Loss: {model_info['loss_type']}")
    print()
    
    # Create model
    model = PointUNet(
        input_dim=model_info['feature_dim'],
        hidden_dim=model_info['hidden_dim'],
        time_dim=model_info['embedding_dim'],
        condition_dim=model_info['feature_dim'],
        num_layers=model_info['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create diffusion process
    if model_info['sampling_type'] == 'gaussian':
        diffusion = GaussianDiffusion(timesteps=model_info['timesteps'])
    else:
        diffusion = ConvexCombinationDiffusion(timesteps=model_info['timesteps'])
    
    # Load MNIST1D data to get digit means
    data = load_mnist1d()
    x_data = data['x']
    y_labels = data['y']
    
    # Generate samples
    digit_samples = {}
    
    print("Generating digit-conditioned samples...")
    with torch.no_grad():
        for digit in range(10):
            # Get mean of this digit class
            digit_mask = y_labels == digit
            digit_data = x_data[digit_mask]
            digit_mean = digit_data.mean(axis=0)
            
            # Create condition (z=0 for forward)
            condition = torch.tensor(digit_mean, dtype=torch.float32).unsqueeze(0).to(device)
            condition = torch.cat([condition, torch.zeros((1, 1), device=device)], dim=1)  # Add z=0
            
            # Create label tensor
            label_tensor = torch.tensor([digit], dtype=torch.long).to(device)
            
            # Generate sample
            if model_info['sampling_type'] == 'convex':
                # Convex sampling uses labels to ensure same-class mixing
                sample = diffusion.sample(
                    model,
                    shape=(1, model_info['feature_dim']),
                    condition=condition,
                    batch_labels=label_tensor
                )
            else:
                # Gaussian sampling doesn't use labels
                sample = diffusion.sample(
                    model,
                    shape=(1, model_info['feature_dim']),
                    condition=condition
                )
            
            digit_samples[digit] = sample[0, :-1].cpu().numpy()  # Remove z-coordinate
            print(f"  Generated digit {digit}")
    
    print()
    
    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for digit in range(10):
        axes[digit].plot(digit_samples[digit], linewidth=2)
        axes[digit].set_title(f"Digit {digit}", fontsize=14, fontweight='bold')
        axes[digit].set_xlabel("Feature Index")
        axes[digit].set_ylabel("Value")
        axes[digit].grid(True, alpha=0.3)
    
    model_name = f"{model_info['training_type']}_{model_info['sampling_type']}_{model_info['loss_type']}"
    plt.suptitle(f"Generated MNIST1D Samples (Conditioned on Digit Mean)\\nModel: {model_name}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"conditioned_digits_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Save individual samples
    for digit, sample in digit_samples.items():
        np.save(output_dir / f"digit_{digit}_{model_name}.npy", sample)
    
    # Compare with real digits
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    
    for digit in range(10):
        col = digit % 5
        row_real = (digit // 5) * 2
        row_syn = row_real + 1
        
        # Real sample
        real_idx = np.where(y_labels == digit)[0][0]
        real_sample = x_data[real_idx]
        axes[row_real, col].plot(real_sample, color='blue', linewidth=2, alpha=0.7)
        axes[row_real, col].set_title(f"Real Digit {digit}", fontsize=12, fontweight='bold')
        axes[row_real, col].grid(True, alpha=0.3)
        
        # Synthetic sample
        axes[row_syn, col].plot(digit_samples[digit], color='red', linewidth=2, alpha=0.7)
        axes[row_syn, col].set_title(f"Synthetic Digit {digit}", fontsize=12, fontweight='bold')
        axes[row_syn, col].set_xlabel("Feature Index")
        axes[row_syn, col].grid(True, alpha=0.3)
    
    plt.suptitle(f"Real vs Synthetic MNIST1D (Digit-Conditioned)\\nModel: {model_name}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"real_vs_synthetic_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return digit_samples


def main():
    output_dir = Path("conditioned_samples")
    output_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Generate from best models (Gaussian FID and Gaussian KL)
    models = [
        "models/exp02_model_semi_implicit_gaussian_fid.pth",
        "models/exp10_model_semi_implicit_gaussian_kl_divergence.pth"
    ]
    
    for model_path in models:
        if os.path.exists(model_path):
            print("="*80)
            generate_conditioned_digit_samples(model_path, output_dir, device)
            print()
        else:
            print(f"Model not found: {model_path}")
    
    print("="*80)
    print(f"Done! Check '{output_dir}/' for digit-conditioned samples")
    print("="*80)


if __name__ == "__main__":
    main()
