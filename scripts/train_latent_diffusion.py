"""
Diffusion Training in IMS Latent Space
=======================================

Train a class-conditioned diffusion model to generate latent codes that can be 
decoded back to IMS spectra. Uses pre-computed latent codes from encoder.

Architecture:
- Input: [noisy_latent (512-dim), timestep, SMILE_embedding (512-dim), class_onehot (8-dim)]  
- Output: predicted noise (512-dim)

Goals:
1. Train diffusion to generate chemically-distinct latent points
2. Compare diffusion samples vs Gaussian samples (both → decoder → IMS)
3. Visualize PCA in latent space and IMS space
4. Compare to actual SMILE embedding structure
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import ast
import wandb

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
SMILE_DIM = 512
NUM_CLASSES = 8
TIMESTEPS = 50
HIDDEN_DIM = 512
NUM_LAYERS = 6
LEARNING_RATE = 5e-5
BATCH_SIZE = 256
MAX_EPOCHS = 1000  # Increased for better separation
BETA_START = 0.001   # Increased 10x for higher-variance separated latents
BETA_END = 0.2       # Increased 10x for higher-variance separated latents

# Loss weights
NOISE_WEIGHT = 0.8      # Weight for noise prediction loss
SEPARATION_WEIGHT = 0.2  # Weight for inter-class separation loss
SEPARATION_MARGIN = 5.0  # Minimum distance between class centroids

# Early stopping
PATIENCE = 100  # Stop if no improvement for 100 epochs

wandb.login(key="57680a36aa570ba8df25adbdd143df3d0bf6b6e8")

# =============================================================================
# DIFFUSION MODEL
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ClassConditionedDiffusion(nn.Module):
    def __init__(self, latent_dim=512, smile_dim=512, num_classes=8, 
                 timesteps=50, hidden_dim=512, num_layers=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        
        # Time embedding
        time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Input projection: latent + SMILE + class_onehot
        input_dim = latent_dim + smile_dim + num_classes
        
        # Network layers
        layers = []
        layers.append(nn.Linear(input_dim + time_dim, hidden_dim))
        layers.append(nn.SiLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t, smile_emb, class_onehot):
        """
        Args:
            x: (batch, latent_dim) - noisy latent
            t: (batch,) - timesteps
            smile_emb: (batch, smile_dim) - SMILE embeddings
            class_onehot: (batch, num_classes) - one-hot class labels
        Returns:
            predicted_noise: (batch, latent_dim)
        """
        # Get time embedding
        t_emb = self.time_mlp(t)
        
        # Concatenate all inputs
        x_in = torch.cat([x, smile_emb, class_onehot], dim=1)
        x_in = torch.cat([x_in, t_emb], dim=1)
        
        # Predict noise
        return self.net(x_in)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_smile_embeddings():
    """Load SMILE embeddings"""
    smile_path = os.path.join(DATA_DIR, 'name_smiles_embedding_file.csv')
    smile_df = pd.read_csv(smile_path)
    
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
    
    embedding_dict = {}
    for _, row in smile_df.iterrows():
        if pd.notna(row['embedding']):
            embedding = np.array(ast.literal_eval(row['embedding']), dtype=np.float32)
            embedding_dict[row['Name']] = embedding
    
    label_embeddings = {}
    for label, full_name in label_mapping.items():
        if full_name in embedding_dict:
            label_embeddings[label] = embedding_dict[full_name]
    
    return label_embeddings


def load_precomputed_latents():
    """Load pre-computed latent codes and labels"""
    print("Loading pre-computed latent codes...")
    
    # Load latents (use separated version if available, otherwise fall back to original)
    separated_train = os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy')
    separated_test = os.path.join(RESULTS_DIR, 'autoencoder_test_latent_separated.npy')
    
    if os.path.exists(separated_train):
        print("  → Using SEPARATED latents from fine-tuned encoder")
        train_latent = np.load(separated_train)
        test_latent = np.load(separated_test)
    else:
        print("  → Using original latents")
        train_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent.npy'))
        test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'))
    
    # Load labels
    train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))
    test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
    
    train_labels = train_df['Label'].values
    test_labels = test_df['Label'].values
    
    print(f"Train latents: {train_latent.shape}")
    print(f"Test latents: {test_latent.shape}")
    print(f"Unique chemicals: {np.unique(train_labels)}")
    
    return train_latent, train_labels, test_latent, test_labels


def create_dataloaders(train_latent, train_labels, smile_embeddings, batch_size=256):
    """Create PyTorch dataloaders"""
    
    # Get unique classes and create mapping
    unique_classes = sorted(np.unique(train_labels))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    
    # Convert to tensors
    latent_tensor = torch.FloatTensor(train_latent)
    label_indices = torch.LongTensor([class_to_idx[l] for l in train_labels])
    
    # Create SMILE embedding tensor
    smile_tensor = torch.FloatTensor(np.array([smile_embeddings[l] for l in train_labels]))
    
    # Create one-hot encodings
    onehot = torch.zeros(len(train_labels), NUM_CLASSES)
    onehot.scatter_(1, label_indices.unsqueeze(1), 1)
    
    dataset = torch.utils.data.TensorDataset(latent_tensor, smile_tensor, onehot, label_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return loader, unique_classes, class_to_idx


# =============================================================================
# DIFFUSION PROCESS
# =============================================================================

def get_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear beta schedule"""
    return torch.linspace(beta_start, beta_end, timesteps)


def forward_diffusion(x_0, t, betas):
    """
    Add noise to x_0 according to timestep t
    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    """
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    alpha_bar_t = alpha_bars[t].reshape(-1, 1)
    noise = torch.randn_like(x_0)
    
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    return x_t, noise


def sample_diffusion(model, smile_emb, class_onehot, betas, device, n_samples=100):
    """
    Reverse diffusion sampling (DDPM)
    """
    model.eval()
    
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # Start from pure noise
    x_t = torch.randn(n_samples, LATENT_DIM).to(device)
    
    with torch.no_grad():
        for t in reversed(range(len(betas))):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(x_t, t_tensor, smile_emb, class_onehot)
            
            # Compute x_{t-1}
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]
            
            if t > 0:
                alpha_bar_prev = alpha_bars[t-1]
                beta_t = betas[t]
                
                # DDPM sampling
                x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
                dir_xt = torch.sqrt(1 - alpha_bar_prev) * predicted_noise
                
                x_t = torch.sqrt(alpha_bar_prev) * x_0_pred + dir_xt
                
                # Add noise (except at last step)
                noise = torch.randn_like(x_t) * torch.sqrt(beta_t)
                x_t = x_t + noise
            else:
                # Final denoising step
                x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
    
    return x_t


# =============================================================================
# TRAINING
# =============================================================================

def compute_separation_loss(x_pred, labels, margin=5.0):
    """
    Encourage different classes to be far apart in latent space.
    Computes margin-based contrastive loss on predicted clean latents.
    """
    unique_labels = torch.unique(labels)
    if len(unique_labels) < 2:
        return torch.tensor(0.0, device=x_pred.device)
    
    # Compute class centroids
    centroids = []
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 0:
            centroids.append(x_pred[mask].mean(dim=0))
    
    if len(centroids) < 2:
        return torch.tensor(0.0, device=x_pred.device)
    
    centroids = torch.stack(centroids)
    
    # Compute pairwise distances between centroids
    n_classes = len(centroids)
    distances = torch.cdist(centroids, centroids, p=2)
    
    # Mask out diagonal (distance to self)
    mask = ~torch.eye(n_classes, dtype=torch.bool, device=distances.device)
    inter_class_dists = distances[mask]
    
    # Penalize distances below margin (encourage separation)
    loss = torch.relu(margin - inter_class_dists).mean()
    
    return loss


def train_diffusion():
    """Train class-conditioned diffusion model with separation loss"""
    
    # Load data
    train_latent, train_labels, test_latent, test_labels = load_precomputed_latents()
    smile_embeddings = load_smile_embeddings()
    
    # Create dataloaders
    train_loader, unique_classes, class_to_idx = create_dataloaders(
        train_latent, train_labels, smile_embeddings, BATCH_SIZE
    )
    
    # Initialize model
    model = ClassConditionedDiffusion(
        latent_dim=LATENT_DIM,
        smile_dim=SMILE_DIM,
        num_classes=NUM_CLASSES,
        timesteps=TIMESTEPS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)
    
    # Beta schedule
    betas = get_beta_schedule(TIMESTEPS, BETA_START, BETA_END).to(device)
    
    # Initialize wandb
    wandb.init(
        project="ims-latent-diffusion",
        config={
            "latent_dim": LATENT_DIM,
            "timesteps": TIMESTEPS,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "noise_weight": NOISE_WEIGHT,
            "separation_weight": SEPARATION_WEIGHT,
            "separation_margin": SEPARATION_MARGIN,
        }
    )
    
    # Training loop
    print("\nStarting training with separation loss...")
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_noise_loss = 0
        epoch_sep_loss = 0
        
        for batch_latent, batch_smile, batch_onehot, batch_idx in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}"):
            batch_latent = batch_latent.to(device)
            batch_smile = batch_smile.to(device)
            batch_onehot = batch_onehot.to(device)
            batch_idx = batch_idx.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, TIMESTEPS, (len(batch_latent),), device=device)
            
            # Forward diffusion
            x_noisy, noise = forward_diffusion(batch_latent, t, betas)
            
            # Predict noise
            predicted_noise = model(x_noisy, t, batch_smile, batch_onehot)
            
            # Noise prediction loss
            noise_loss = nn.functional.mse_loss(predicted_noise, noise)
            
            # Predict clean latent (for separation loss)
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)
            alpha_bar_t = alpha_bars[t].reshape(-1, 1)
            x_pred = (x_noisy - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            
            # Separation loss
            sep_loss = compute_separation_loss(x_pred, batch_idx, margin=SEPARATION_MARGIN)
            
            # Combined loss
            loss = NOISE_WEIGHT * noise_loss + SEPARATION_WEIGHT * sep_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_noise_loss += noise_loss.item()
            epoch_sep_loss += sep_loss.item()
        
        avg_noise_loss = epoch_noise_loss / len(train_loader)
        avg_sep_loss = epoch_sep_loss / len(train_loader)
        avg_total_loss = NOISE_WEIGHT * avg_noise_loss + SEPARATION_WEIGHT * avg_sep_loss
        scheduler.step()
        
        # Log
        wandb.log({
            "epoch": epoch + 1,
            "total_loss": avg_total_loss,
            "noise_loss": avg_noise_loss,
            "separation_loss": avg_sep_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f"Epoch {epoch+1}: Total={avg_total_loss:.6f}, Noise={avg_noise_loss:.6f}, Sep={avg_sep_loss:.6f}")
        
        # Save best model and check early stopping
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(MODELS_DIR, 'diffusion_latent_separated_best.pt'))
            print(f"  ✓ Saved best model (loss: {best_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"\n⚠ Early stopping: No improvement for {PATIENCE} epochs")
                break
        
        # Save checkpoint every 100 epochs
        if (epoch + 1) % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total_loss,
            }, os.path.join(MODELS_DIR, f'diffusion_latent_separated_epoch_{epoch+1}.pt'))
    
    wandb.finish()
    print("\n✓ Training complete!")


if __name__ == "__main__":
    train_diffusion()
