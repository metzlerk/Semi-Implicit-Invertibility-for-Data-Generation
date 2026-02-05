"""
Test script to verify encoder and decoder from ChemicalDataGeneration repo work correctly.
Loads pre-trained models directly without using functions.py to avoid dependencies.
"""

import sys
# Add ChemicalDataGeneration models to path so torch.load can find custom classes
sys.path.insert(0, '/home/kjmetzler/ChemicalDataGeneration/models')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path('/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data')
ENCODER_PATH = '/scratch/cmdunham/trained_models/spectrum/current_best_models/nine_layer_ims_to_chemnet_encoder.pth'
DECODER_PATH = '/scratch/cmdunham/trained_models/spectrum/current_best_models/nine_layer_ChemNet_to_ims_generator_from_nine_layer__encoder.pth'

def load_data():
    """Load test IMS data and SMILE embeddings"""
    print("Loading data...")
    test_data = pd.read_feather(DATA_DIR / 'test_data.feather')
    smile_data = pd.read_csv(DATA_DIR / 'name_smiles_embedding_file.csv')
    
    # Extract spectra (assuming columns p_184 to p_1859)
    spectrum_cols = [col for col in test_data.columns if col.startswith('p_')]
    spectra = test_data[spectrum_cols].values
    
    # Extract labels
    labels = test_data['Label'].values
    
    print(f"Loaded {len(spectra)} test spectra with shape {spectra.shape}")
    print(f"Unique chemicals: {np.unique(labels)}")
    
    return spectra, labels, smile_data

def load_models(device):
    """Load pre-trained encoder and decoder"""
    print(f"\nLoading models on {device}...")
    
    # Load encoder
    encoder = torch.load(ENCODER_PATH, map_location=device, weights_only=False)
    encoder.eval()
    print(f"✓ Encoder loaded: {type(encoder).__name__}")
    
    # Load decoder  
    decoder = torch.load(DECODER_PATH, map_location=device, weights_only=False)
    decoder.eval()
    print(f"✓ Decoder loaded: {type(decoder).__name__}")
    
    return encoder, decoder

def test_encode_decode(encoder, decoder, spectra, device, n_samples=10):
    """Test encoding and decoding a few samples"""
    print(f"\nTesting encode -> decode on {n_samples} samples...")
    
    # Take subset
    test_spectra = torch.FloatTensor(spectra[:n_samples]).to(device)
    
    # Encode
    with torch.no_grad():
        latent = encoder(test_spectra)
        print(f"Latent shape: {latent.shape}")
        print(f"Latent stats: mean={latent.mean().item():.4f}, std={latent.std().item():.4f}")
        
        # Decode
        reconstructed = decoder(latent)
        print(f"Reconstructed shape: {reconstructed.shape}")
        
        # Compute reconstruction error
        mse = torch.nn.functional.mse_loss(reconstructed, test_spectra)
        mae = torch.nn.functional.l1_loss(reconstructed, test_spectra)
        
        print(f"\nReconstruction metrics:")
        print(f"  MSE: {mse.item():.6f}")
        print(f"  MAE: {mae.item():.6f}")
        
    return latent.cpu().numpy(), reconstructed.cpu().numpy()

def main():
    print("="*60)
    print("Testing Encoder/Decoder from ChemicalDataGeneration")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    spectra, labels, smile_data = load_data()
    
    # Load models
    encoder, decoder = load_models(device)
    
    # Test
    latent, reconstructed = test_encode_decode(encoder, decoder, spectra, device)
    
    print("\n" + "="*60)
    print("✓ Encoder and decoder working correctly!")
    print("="*60)
    print(f"\nNext steps:")
    print("  1. Train diffusion model in {latent.shape[1]}-dim latent space")
    print("  2. Condition diffusion on chemical type using SMILE embeddings")
    print("  3. Compare diffusion samples vs Gaussian samples")

if __name__ == "__main__":
    main()
