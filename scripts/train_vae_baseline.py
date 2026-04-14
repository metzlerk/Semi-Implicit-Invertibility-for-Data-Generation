"""
Train a VAE Baseline for Comparison
====================================

Standard Variational Autoencoder with Gaussian prior regularization (KL divergence).
This serves as a baseline to compare against geometry-preserving latent diffusion.

Architecture matches decoupled autoencoder but adds KL regularization.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

# Paths
ROOT_DIR = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation'
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
IMAGES_DIR = os.path.join(ROOT_DIR, 'images')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
LATENT_DIM = 512
INPUT_DIM = 1676
N_LAYERS = 9
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 500
PATIENCE = 50
KL_WEIGHT = 0.0001  # Beta for KL divergence loss

# Encoder (IMS -> latent)
class VAEEncoder(nn.Module):
    def __init__(self, input_dim=1676, latent_dim=512, n_layers=9):
        super().__init__()
        
        layer_sizes = np.linspace(input_dim, latent_dim, n_layers + 1).astype(int)
        
        layers = []
        for i in range(n_layers - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        
        self.encoder = nn.Sequential(*layers)
        
        # Split into mean and log variance
        self.fc_mu = nn.Linear(layer_sizes[-2], latent_dim)
        self.fc_logvar = nn.Linear(layer_sizes[-2], latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder (latent -> IMS)
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=512, output_dim=1676, n_layers=9):
        super().__init__()
        
        layer_sizes = np.linspace(latent_dim, output_dim, n_layers + 1).astype(int)
        
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < n_layers - 1:
                layers.append(nn.LeakyReLU(0.1, inplace=True))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder(z)

# Full VAE
class VAE(nn.Module):
    def __init__(self, input_dim=1676, latent_dim=512, n_layers=9):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim, n_layers)
        self.decoder = VAEDecoder(latent_dim, input_dim, n_layers)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def encode(self, x):
        """Get latent representation (mean)"""
        mu, logvar = self.encoder(x)
        return mu
    
    def decode(self, z):
        """Decode latent to IMS"""
        return self.decoder(z)

# Loss function
def vae_loss(recon_x, x, mu, logvar, kl_weight=0.0001):
    """
    VAE loss = reconstruction loss + KL divergence
    
    Args:
        recon_x: reconstructed data
        x: original data
        mu: mean from encoder
        logvar: log variance from encoder
        kl_weight: weight for KL divergence term (beta-VAE style)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = recon_loss + kl_weight * kld
    
    return total_loss, recon_loss, kld

# Data loading
def load_data():
    """Load IMS spectra data"""
    print("Loading IMS data...")
    train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))
    test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
    
    # Extract features and labels
    feature_cols = [col for col in train_df.columns if col not in ['Label', 'index']]
    
    train_data = train_df[feature_cols].values.astype(np.float32)
    train_labels = train_df['Label'].values
    
    test_data = test_df[feature_cols].values.astype(np.float32)
    test_labels = test_df['Label'].values
    
    print(f"Train shape: {train_data.shape}, labels: {train_labels.shape}")
    print(f"Test shape: {test_data.shape}, labels: {test_labels.shape}")
    print(f"Unique chemicals: {np.unique(train_labels)}")
    
    return train_data, train_labels, test_data, test_labels

# Training
def train_vae():
    """Train VAE model"""
    
    # Load data
    train_data, train_labels, test_data, test_labels = load_data()
    
    # Convert to tensors
    train_tensor = torch.FloatTensor(train_data).to(device)
    test_tensor = torch.FloatTensor(test_data).to(device)
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM, n_layers=N_LAYERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []
    
    print("\nTraining VAE...")
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kld = 0
        
        for batch in train_loader:
            x = batch[0]
            
            # Forward pass
            recon, mu, logvar = model(x)
            loss, recon_loss, kld = vae_loss(recon, x, mu, logvar, kl_weight=KL_WEIGHT)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kld += kld.item()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_recon, test_mu, test_logvar = model(test_tensor)
            test_loss, test_recon_loss, test_kld = vae_loss(test_recon, test_tensor, test_mu, test_logvar, kl_weight=KL_WEIGHT)
        
        epoch_loss /= len(train_loader)
        epoch_recon /= len(train_loader)
        epoch_kld /= len(train_loader)
        
        train_losses.append(epoch_loss)
        test_losses.append(test_loss.item())
        
        # Learning rate schedule
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{MAX_EPOCHS} | "
                  f"Train Loss: {epoch_loss:.4f} (Recon: {epoch_recon:.4f}, KLD: {epoch_kld:.4f}) | "
                  f"Test Loss: {test_loss.item():.4f}")
        
        # Early stopping
        if test_loss.item() < best_loss:
            best_loss = test_loss.item()
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'test_loss': test_loss.item(),
            }, os.path.join(MODELS_DIR, 'vae_baseline_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('VAE Training Loss')
    plt.savefig(os.path.join(IMAGES_DIR, 'vae_training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Load best model
    checkpoint = torch.load(os.path.join(MODELS_DIR, 'vae_baseline_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save latent representations
    model.eval()
    with torch.no_grad():
        train_latent = model.encode(train_tensor).cpu().numpy()
        test_latent = model.encode(test_tensor).cpu().numpy()
    
    np.save(os.path.join(RESULTS_DIR, 'vae_train_latent.npy'), train_latent)
    np.save(os.path.join(RESULTS_DIR, 'vae_test_latent.npy'), test_latent)
    
    print(f"\nBest test loss: {best_loss:.4f}")
    print(f"Saved latents: {train_latent.shape}, {test_latent.shape}")
    
    return model

if __name__ == '__main__':
    model = train_vae()
    print("\nVAE training complete!")
