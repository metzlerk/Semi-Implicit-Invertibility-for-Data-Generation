#!/usr/bin/env python3
"""
Continue Training Autoencoder for More Epochs
==============================================

Loads the best saved autoencoder checkpoint and continues training for additional epochs
to bring the learned latent space closer to the ChemNet embedding distribution.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
import ast

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

latent_dim = 512
encoder_n_layers = 9
generator_n_layers = 9
encoder_lr = 1e-4
additional_epochs = 200  # Continue training for 200 more epochs
encoder_batch_size = 256
use_background_bias = True
trainable_bias = True

# =============================================================================
# ARCHITECTURE (copied from main training script)
# =============================================================================

class BiasOnlyLayer(nn.Module):
    def __init__(self, bias_init=None, trainable=False):
        super().__init__()
        if bias_init is None:
            raise ValueError("BiasOnlyLayer requires bias_init")
        self.bias = nn.Parameter(bias_init.clone().detach(), requires_grad=trainable)

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
# DATA LOADING (same as main script)
# =============================================================================

def load_data():
    """Load and prepare IMS data with labels"""
    print("\nLoading IMS data...")
    
    # Load data
    train_path = os.path.join(DATA_DIR, 'train_data.feather')
    test_path = os.path.join(DATA_DIR, 'test_data.feather')
    
    train_df = pd.read_feather(train_path)
    test_df = pd.read_feather(test_path)
    
    # Extract labels
    onehot_cols = ['DEB','DEM','DMMP','DPM','DtBP','JP8','MES','TEPO']
    train_labels = train_df[onehot_cols].values.argmax(axis=1)
    test_labels = test_df[onehot_cols].values.argmax(axis=1)
    
    # Extract IMS features (positive and negative modes)
    p_cols = [f'p_{i}' for i in range(184, 1022)]
    n_cols = [f'n_{i}' for i in range(184, 1022)]
    
    train_ims = train_df[p_cols + n_cols].values.astype(np.float32)
    test_ims = test_df[p_cols + n_cols].values.astype(np.float32)
    
    print(f"  Train IMS shape: {train_ims.shape}")
    print(f"  Test IMS shape: {test_ims.shape}")
    
    # Normalize
    ims_mean = train_ims.mean(axis=0)
    ims_std = train_ims.std(axis=0) + 1e-8
    
    train_ims_norm = (train_ims - ims_mean) / ims_std
    test_ims_norm = (test_ims - ims_mean) / ims_std
    
    # Background
    train_bkg = train_ims_norm.mean(axis=0)
    
    print(f"  Normalized train mean: {train_ims_norm.mean():.4f}, std: {train_ims_norm.std():.4f}")
    
    return train_ims_norm, test_ims_norm, train_bkg, onehot_cols


# =============================================================================
# CONTINUED TRAINING
# =============================================================================

def continue_training(encoder, generator, train_ims, device, start_epoch, use_bias=True):
    """
    Continue training encoder and generator from saved checkpoint.
    """
    print("\n" + "="*80)
    print(f"CONTINUING AUTOENCODER TRAINING FOR {additional_epochs} MORE EPOCHS")
    print("="*80)
    
    encoder.train()
    generator.train()
    
    # Parameters and optimizer
    params = list(encoder.parameters()) + list(generator.parameters())
    optimizer = torch.optim.AdamW(params, lr=encoder_lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Create dataset
    train_tensor = torch.FloatTensor(train_ims).to(device)
    dataset = torch.utils.data.TensorDataset(train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=encoder_batch_size, shuffle=True)
    
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(start_epoch, start_epoch + additional_epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch + additional_epochs}")
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
        
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}")
        
        # Save best model (overwrite previous)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, os.path.join(MODELS_DIR, 'best_autoencoder.pth'))
            print(f"  -> Saved best model (loss: {best_loss:.6f})")
    
    print(f"\nContinued training complete!")
    print(f"  Best loss: {best_loss:.6f}")
    print(f"  Final loss: {loss_history[-1]:.6f}")
    
    return encoder, generator, loss_history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Load data
    train_ims, test_ims, train_bkg, class_names = load_data()
    
    # Initialize models
    print("\nInitializing encoder and generator...")
    
    train_bkg_tensor = torch.FloatTensor(train_bkg).to(device)
    
    encoder = FlexibleNLayersEncoder(
        input_size=1676,
        output_size=latent_dim,
        n_layers=encoder_n_layers,
        init_style='bkg' if use_background_bias else None,
        bkg=train_bkg_tensor if use_background_bias else None,
        trainable=trainable_bias
    ).to(device)
    
    generator = FlexibleNLayersGenerator(
        input_size=latent_dim,
        output_size=1676,
        n_layers=generator_n_layers,
        init_style='bkg' if use_background_bias else None,
        bkg=train_bkg_tensor if use_background_bias else None,
        trainable=trainable_bias
    ).to(device)
    
    # Load checkpoint
    autoencoder_path = os.path.join(MODELS_DIR, 'best_autoencoder.pth')
    if not os.path.exists(autoencoder_path):
        print(f"ERROR: Checkpoint not found: {autoencoder_path}")
        sys.exit(1)
    
    print(f"\nLoading checkpoint from {autoencoder_path}")
    checkpoint = torch.load(autoencoder_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss_at_load = checkpoint['loss']
    
    print(f"  Loaded from epoch {checkpoint['epoch']} with loss {best_loss_at_load:.6f}")
    print(f"  Resuming from epoch {start_epoch}")
    
    # Continue training
    encoder, generator, loss_history = continue_training(
        encoder, generator, train_ims, device, start_epoch, use_bias=use_background_bias
    )
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Model saved to: {autoencoder_path}")
    print(f"\nTo regenerate plots with improved model, run:")
    print(f"  python scripts/plot_synthetic_spectra_pca_by_class.py")
