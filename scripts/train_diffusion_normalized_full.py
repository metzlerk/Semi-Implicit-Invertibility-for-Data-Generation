#!/usr/bin/env python3
"""
Train class-conditioned diffusion on normalized latent space
Key fix: Normalize latents before training, denormalize after sampling
This allows diffusion to work properly with high-variance separated latents
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Paths
DATA_DIR = 'Data'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'

# Hyperparameters
LATENT_DIM = 512
SMILE_DIM = 512
NUM_CLASSES = 8
TIMESTEPS = 1000  # Increased from 50 for better manifold preservation
HIDDEN_DIM = 512
NUM_LAYERS = 6
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
MAX_EPOCHS = 1000
SAVE_INTERVAL = 100

# Beta schedule - cosine schedule for better manifold preservation
BETA_START = 0.0001
BETA_END = 0.02

# =============================================================================
# MODEL
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
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ClassConditionedDiffusion(nn.Module):
    def __init__(self, latent_dim=512, smile_dim=512, num_classes=8, 
                 timesteps=50, hidden_dim=512, num_layers=6,
                 beta_start=0.00002, beta_end=0.005):
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
        
        # Cosine noise schedule for better manifold preservation
        steps = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999).float()
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
    
    def forward(self, x, t, smile_emb, class_onehot):
        t_emb = self.time_mlp(t)
        x_in = torch.cat([x, smile_emb, class_onehot], dim=1)
        x_in = torch.cat([x_in, t_emb], dim=1)
        return self.net(x_in)

# =============================================================================
# DATA LOADING
# =============================================================================

def load_smile_embeddings():
    """Load SMILE embeddings"""
    import ast
    
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
            label_embeddings[label] = torch.FloatTensor(embedding_dict[full_name])
    
    return label_embeddings

print("Loading data...")

# Load latents
train_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy'))
test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent_separated.npy'))

# Load data for labels
train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))
test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))

label_columns = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
train_labels = train_df[label_columns].values
test_labels = test_df[label_columns].values

# Load SMILE embeddings
smile_dict = load_smile_embeddings()

print(f"Train samples: {len(train_latent)}")
print(f"Test samples: {len(test_latent)}")
print(f"Latent dim: {train_latent.shape[1]}")

# NORMALIZE LATENTS - USE TRAIN ONLY TO AVOID DATA SNOOPING
print("\nNormalizing latents...")
DATA_MEAN = train_latent.mean()
DATA_STD = train_latent.std()

print(f"  Train original mean: {DATA_MEAN:.4f}, std: {DATA_STD:.4f}")

train_latent_norm = (train_latent - DATA_MEAN) / DATA_STD
test_latent_norm = (test_latent - DATA_MEAN) / DATA_STD  # Use train stats

print(f"  Normalized train mean: {train_latent_norm.mean():.4f}, std: {train_latent_norm.std():.4f}")
print(f"  Normalized test mean: {test_latent_norm.mean():.4f}, std: {test_latent_norm.std():.4f}")

# =============================================================================
# DATASET
# =============================================================================

class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, latents, labels, smile_dict, label_columns):
        self.latents = torch.FloatTensor(latents)
        self.labels = torch.FloatTensor(labels)
        self.label_columns = label_columns
        self.smile_dict = smile_dict
        
        # Precompute class indices and SMILE embeddings
        self.class_indices = torch.argmax(self.labels, dim=1)
        self.smile_embs = []
        for i in range(len(self.latents)):
            class_idx = self.class_indices[i].item()
            chemical_name = label_columns[class_idx]
            self.smile_embs.append(smile_dict[chemical_name])
        self.smile_embs = torch.stack(self.smile_embs)
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return self.latents[idx], self.smile_embs[idx], self.labels[idx]

train_dataset = LatentDataset(train_latent_norm, train_labels, smile_dict, label_columns)
test_dataset = LatentDataset(test_latent_norm, test_labels, smile_dict, label_columns)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

# =============================================================================
# TRAINING
# =============================================================================

def forward_diffusion(x_0, t, model):
    """Forward diffusion: q(x_t | x_0)"""
    sqrt_alpha_bar = torch.sqrt(model.alphas_cumprod[t]).reshape(-1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - model.alphas_cumprod[t]).reshape(-1, 1)
    noise = torch.randn_like(x_0)
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t, x_0  # Return x_0 for x_0-prediction training

@torch.no_grad()
def sample_ddpm(model, smile_embs, class_onehots, n_samples=None):
    """DDIM sampling with x_0 prediction - better for structure"""
    model.eval()
    if n_samples is None:
        n_samples = len(smile_embs)
    
    x_t = torch.randn(n_samples, LATENT_DIM, device=device)
    
    # Use DDIM with 100 steps through 1000 timesteps
    ddim_steps = 100
    step_size = model.timesteps // ddim_steps
    timesteps = list(range(0, model.timesteps, step_size))
    
    for i in reversed(range(len(timesteps))):
        t = timesteps[i]
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
        
        # Model directly predicts x_0
        x_0_pred = model(x_t, t_batch, smile_embs, class_onehots)
        x_0_pred = torch.clamp(x_0_pred, -5, 5)
        
        if i > 0:
            t_prev = timesteps[i-1]
            alpha_bar_t = model.alphas_cumprod[t]
            alpha_bar_prev = model.alphas_cumprod[t_prev]
            
            # DDIM update
            predicted_noise = (x_t - torch.sqrt(alpha_bar_t) * x_0_pred) / torch.sqrt(1 - alpha_bar_t)
            x_t = torch.sqrt(alpha_bar_prev) * x_0_pred + torch.sqrt(1 - alpha_bar_prev) * predicted_noise
        else:
            x_t = x_0_pred
    
    return x_t

print("\n" + "="*80)
print(f"TRAINING NORMALIZED CLASS-CONDITIONED DIFFUSION")
print(f"Timesteps: {TIMESTEPS} | Cosine schedule | X_0 PREDICTION")
print("="*80 + "\n")

model = ClassConditionedDiffusion(
    latent_dim=LATENT_DIM,
    smile_dim=SMILE_DIM,
    num_classes=NUM_CLASSES,
    timesteps=TIMESTEPS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    beta_start=BETA_START,
    beta_end=BETA_END
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)

best_loss = float('inf')

for epoch in range(MAX_EPOCHS):
    model.train()
    epoch_loss = 0
    n_batches = 0
    
    for latent, smile_emb, class_onehot in train_loader:
        latent = latent.to(device)
        smile_emb = smile_emb.to(device)
        class_onehot = class_onehot.to(device)
        
        t = torch.randint(0, TIMESTEPS, (len(latent),), device=device)
        # Forward diffusion returns (x_t, x_0) for x_0-prediction training
        x_noisy, target_x0 = forward_diffusion(latent, t, model)
        
        # Model predicts x_0 directly
        predicted_x0 = model(x_noisy, t, smile_emb, class_onehot)
        
        # Loss: MSE between predicted x_0 and actual x_0
        loss = nn.functional.mse_loss(predicted_x0, target_x0)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    scheduler.step()
    avg_loss = epoch_loss / n_batches
    
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{MAX_EPOCHS}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'data_mean': DATA_MEAN,
            'data_std': DATA_STD,
            'epoch': epoch,
            'loss': avg_loss
        }, os.path.join(MODELS_DIR, 'diffusion_latent_normalized_best.pt'))
    
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'data_mean': DATA_MEAN,
            'data_std': DATA_STD,
            'epoch': epoch,
            'loss': avg_loss
        }, os.path.join(MODELS_DIR, f'diffusion_latent_normalized_epoch_{epoch+1}.pt'))

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Best loss: {best_loss:.4f}")
print(f"Model saved to: {os.path.join(MODELS_DIR, 'diffusion_latent_normalized_best.pt')}")
print(f"Normalization: mean={DATA_MEAN:.4f}, std={DATA_STD:.4f}")
print("\nTo generate samples, denormalize: samples_real = samples_norm * DATA_STD + DATA_MEAN")
