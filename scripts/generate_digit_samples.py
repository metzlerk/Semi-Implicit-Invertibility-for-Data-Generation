"""
Generate one sample of each digit (0-9) from a saved model.
Uses a simple KNN classifier to identify which digit each generated sample represents.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import requests
import os
import sys

# Import from the other script
sys.path.append(str(Path(__file__).parent))
from generate_samples_from_saved_model import (
    PointUNet, TimeEmbedding, GaussianDiffusion, 
    load_model, generate_samples, load_mnist1d
)


def classify_samples(samples, real_data, real_labels, k=5):
    """
    Classify generated samples using k-NN on real data.
    
    Args:
        samples: Generated samples [n_samples, feature_dim]
        real_data: Real MNIST1D data [n_real, feature_dim]
        real_labels: Labels for real data [n_real]
        k: Number of neighbors to consider
    
    Returns:
        predicted_labels: Predicted digit for each sample
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    # Train a simple k-NN classifier on real data
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(real_data, real_labels)
    
    # Predict labels for generated samples
    predicted_labels = knn.predict(samples)
    
    return predicted_labels


def get_one_of_each_digit(samples, labels, target_digits=range(10)):
    """
    Get one sample of each digit.
    
    Args:
        samples: Generated samples
        labels: Predicted labels for samples
        target_digits: Which digits to extract (default 0-9)
    
    Returns:
        digit_samples: Dictionary mapping digit -> sample
        digit_indices: Dictionary mapping digit -> index in original samples
    """
    digit_samples = {}
    digit_indices = {}
    
    for digit in target_digits:
        # Find indices where this digit was generated
        indices = np.where(labels == digit)[0]
        
        if len(indices) > 0:
            # Take the first occurrence
            idx = indices[0]
            digit_samples[digit] = samples[idx]
            digit_indices[digit] = idx
        else:
            print(f"Warning: No samples classified as digit {digit}")
            digit_samples[digit] = None
            digit_indices[digit] = None
    
    return digit_samples, digit_indices


def plot_digit_grid(digit_samples, model_name, save_path=None):
    """
    Plot one sample of each digit in a 2x5 grid.
    
    Args:
        digit_samples: Dictionary mapping digit -> sample
        model_name: Name of the model for title
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for digit in range(10):
        ax = axes[digit]
        sample = digit_samples.get(digit)
        
        if sample is not None:
            ax.plot(sample, linewidth=2)
            ax.set_title(f"Digit {digit}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"Digit {digit}\nNot Generated", 
                   ha='center', va='center', fontsize=12, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
    
    plt.suptitle(f"Generated MNIST1D Samples (One Per Digit)\nModel: {model_name}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved digit grid to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_with_real_digits(digit_samples, real_data, real_labels, model_name, save_path=None):
    """
    Compare generated samples with real samples of the same digits.
    """
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    
    for digit in range(10):
        col = digit % 5
        
        # Real sample (top row)
        real_idx = np.where(real_labels == digit)[0][0]
        real_sample = real_data[real_idx]
        axes[col // 5 * 2, col].plot(real_sample, color='blue', linewidth=2, alpha=0.7)
        axes[col // 5 * 2, col].set_title(f"Real Digit {digit}", fontsize=12, fontweight='bold')
        axes[col // 5 * 2, col].set_ylabel("Value")
        axes[col // 5 * 2, col].grid(True, alpha=0.3)
        
        # Generated sample (bottom row)
        synthetic_sample = digit_samples.get(digit)
        if synthetic_sample is not None:
            axes[col // 5 * 2 + 1, col].plot(synthetic_sample, color='red', linewidth=2, alpha=0.7)
            axes[col // 5 * 2 + 1, col].set_title(f"Synthetic Digit {digit}", fontsize=12, fontweight='bold')
        else:
            axes[col // 5 * 2 + 1, col].text(0.5, 0.5, "Not Generated", 
                                             ha='center', va='center', fontsize=10, color='red')
            axes[col // 5 * 2 + 1, col].set_xlim(0, 1)
            axes[col // 5 * 2 + 1, col].set_ylim(0, 1)
        
        axes[col // 5 * 2 + 1, col].set_xlabel("Feature Index")
        axes[col // 5 * 2 + 1, col].set_ylabel("Value")
        axes[col // 5 * 2 + 1, col].grid(True, alpha=0.3)
    
    plt.suptitle(f"Real vs Synthetic MNIST1D Samples\nModel: {model_name}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    # Configuration
    model_path = "models/exp02_model_semi_implicit_gaussian_fid.pth"
    num_samples = 500  # Generate many samples to ensure we get all digits
    z_value = 0.0  # Forward direction
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path("generated_samples")
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("Generating One Sample Per Digit (0-9)")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Generating {num_samples} samples to find all digits...")
    print()
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    timesteps = checkpoint['timesteps']
    input_dim = checkpoint['input_dim']
    model_name = f"{checkpoint['training_method']}_{checkpoint['sampling_method']}_{checkpoint['loss_type']}"
    
    print()
    print("Generating samples...")
    
    # Generate samples
    samples = generate_samples(model, num_samples, z_value, timesteps, input_dim, device)
    
    print(f"Generated {len(samples)} samples")
    print()
    
    # Load real MNIST1D data for classification
    print("Loading real MNIST1D data for classification...")
    data = load_mnist1d()
    real_data = data['x']
    real_labels = data['y']
    print(f"Loaded {len(real_data)} real samples")
    print()
    
    # Classify generated samples
    print("Classifying generated samples using k-NN...")
    predicted_labels = classify_samples(samples, real_data, real_labels, k=5)
    
    # Count how many of each digit were generated
    print("\nDigit distribution in generated samples:")
    for digit in range(10):
        count = np.sum(predicted_labels == digit)
        percentage = 100 * count / len(predicted_labels)
        print(f"  Digit {digit}: {count:3d} samples ({percentage:5.1f}%)")
    
    print()
    
    # Get one sample of each digit
    digit_samples, digit_indices = get_one_of_each_digit(samples, predicted_labels)
    
    # Check if we got all digits
    missing_digits = [d for d in range(10) if digit_samples[d] is None]
    if missing_digits:
        print(f"WARNING: Missing digits: {missing_digits}")
        print("Try generating more samples or the model may not generate these digits well.")
    else:
        print("✓ Successfully found at least one sample of each digit (0-9)")
    
    print()
    
    # Save individual digit samples
    for digit, sample in digit_samples.items():
        if sample is not None:
            np.save(output_dir / f"digit_{digit}_sample.npy", sample)
    
    print(f"Saved individual digit samples to {output_dir}/digit_X_sample.npy")
    
    # Plot grid of one sample per digit
    plot_digit_grid(digit_samples, model_name, save_path=output_dir / "digits_0_to_9_grid.png")
    
    # Compare with real samples
    compare_with_real_digits(digit_samples, real_data, real_labels, model_name, 
                            save_path=output_dir / "real_vs_synthetic_by_digit.png")
    
    print()
    print("="*80)
    print(f"Done! Check the '{output_dir}' directory for:")
    print("  - digits_0_to_9_grid.png: One sample of each digit in order")
    print("  - real_vs_synthetic_by_digit.png: Comparison with real samples")
    print("  - digit_X_sample.npy: Individual samples for each digit")
    print("="*80)
    
    # Answer the user's question about conditioning
    print()
    print("NOTE: The digit labels (0-9) were NOT used during training/generation.")
    print("The model generates samples from the learned distribution without")
    print("explicit digit conditioning. We used k-NN classification to identify")
    print("which digit each generated sample most closely resembles.")


if __name__ == "__main__":
    main()
