"""
Train diffusion model on NORMALIZED latents (matching original approach)
This version is for the beta ablation study where everything is identical
except for beta_end (0.02 vs 0.2)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import ast
import wandb
import argparse

# Paths
ROOT_DIR = '/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation'
DATA_DIR = os.path.join(ROOT_DIR, 'Data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# MODEL ARCHITECTURE (must match original)
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
        
        # Input projection: noisy_latent + SMILE + class_onehot
        input_dim = latent_dim + smile_dim + num_classes
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer-style layers
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
            ])
        self.layers = nn.Sequential(*layers)
        
        # Output projection (predict noise)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x_t, t, smile_emb, class_onehot):
        """
        x_t: [B, latent_dim] - noisy latent
        t: [B] - timestep
        smile_emb: [B, smile_dim] - SMILE embedding
        class_onehot: [B, num_classes] - one-hot class label
        """
        # Time embedding
        t_emb = self.time_mlp(t.float())
        
        # Concatenate all inputs
        x = torch.cat([x_t, smile_emb, class_onehot], dim=-1)
        x = self.input_proj(x)
        
        # Add time embedding
        x = x + t_emb
        
        # Process through layers
        x = self.layers(x)
        
        # Predict noise
        noise_pred = self.output_proj(x)
        
        return noise_pred


# =============================================================================
# DATA LOADING
# =============================================================================

def load_smile_embeddings():
    """Load pre-computed ChemNet embeddings for each chemical"""
    smile_df = pd.read_csv(os.path.join(DATA_DIR, 'name_smiles_embedding_file.csv'))

    label_candidates = {
        'DEB': ['DEB', '1,2,3,4-Diepoxybutane', 'Diethylene glycol dibutyl ether'],
        'DEM': ['DEM', 'Diethyl Malonate', 'Diethylene glycol diethyl ether'],
        'DMMP': ['DMMP', 'Dimethyl methylphosphonate'],
        'DPM': ['DPM', 'Oxybispropanol'],
        'DtBP': ['DtBP', 'Di-tert-butyl peroxide'],
        'JP8': ['JP8'],
        'MES': ['MES', '2-(N-morpholino)ethanesulfonic acid'],
        'TEPO': ['TEPO', 'Triethyl phosphate'],
    }

    embedding_dict = {}
    for _, row in smile_df.iterrows():
        if pd.notna(row['embedding']):
            embedding = np.array(ast.literal_eval(row['embedding']), dtype=np.float32)
            embedding_dict[row['Name']] = embedding
    
    label_embeddings = {}
    missing = []
    for label, candidates in label_candidates.items():
        for name in candidates:
            if name in embedding_dict:
                label_embeddings[label] = embedding_dict[name]
                break
        else:
            missing.append(label)

    if missing:
        raise ValueError(
            "Missing ChemNet embeddings for labels: "
            + ", ".join(sorted(missing))
            + ". Check name_smiles_embedding_file.csv."
        )
    
    return label_embeddings


def load_precomputed_latents():
    """Load ORIGINAL (un-separated) latent codes"""
    print("Loading ORIGINAL latent codes...")
    
    # Load ORIGINAL latents (not separated)
    train_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent.npy'))
    test_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_test_latent.npy'))
    
    # Load labels
    train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))
    test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
    
    train_labels = train_df['Label'].values
    test_labels = test_df['Label'].values
    
    print(f"Train latents: {train_latent.shape}, mean={train_latent.mean():.4f}, std={train_latent.std():.4f}")
    print(f"Test latents: {test_latent.shape}, mean={test_latent.mean():.4f}, std={test_latent.std():.4f}")
    print(f"Unique chemicals: {np.unique(train_labels)}")
    
    return train_latent, train_labels, test_latent, test_labels


def normalize_data(train_latent, test_latent):
    """Normalize latents to zero mean and unit variance"""
    data_mean = train_latent.mean()
    data_std = train_latent.std()
    
    train_normalized = (train_latent - data_mean) / data_std
    test_normalized = (test_latent - data_mean) / data_std
    
    print(f"\nNormalization:")
    print(f"  Data mean: {data_mean:.6f}")
    print(f"  Data std: {data_std:.6f}")
    print(f"  After normalization: mean={train_normalized.mean():.6f}, std={train_normalized.std():.6f}")
    
    return train_normalized, test_normalized, data_mean, data_std


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
    num_classes = len(unique_classes)
    onehot = torch.zeros(len(train_labels), num_classes)
    onehot.scatter_(1, label_indices.unsqueeze(1), 1)
    
    dataset = torch.utils.data.TensorDataset(latent_tensor, smile_tensor, onehot, label_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return loader, unique_classes, class_to_idx


# =============================================================================
# DIFFUSION PROCESS
# =============================================================================

def get_beta_schedule(timesteps, beta_start=0.001, beta_end=0.02):
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


def compute_inter_class_dists(x_pred, labels):
    unique_labels = torch.unique(labels)
    if len(unique_labels) < 2:
        return None

    centroids = []
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 0:
            centroids.append(x_pred[mask].mean(dim=0))

    if len(centroids) < 2:
        return None

    centroids = torch.stack(centroids)
    n_classes = len(centroids)
    distances = torch.cdist(centroids, centroids, p=2)
    mask = ~torch.eye(n_classes, dtype=torch.bool, device=distances.device)
    return distances[mask]


def compute_separation_loss(inter_class_dists, margin):
    if inter_class_dists is None or inter_class_dists.numel() == 0:
        device = margin.device if torch.is_tensor(margin) else 'cpu'
        return torch.tensor(0.0, device=device)
    margin_tensor = margin if torch.is_tensor(margin) else torch.tensor(margin, device=inter_class_dists.device)
    return torch.relu(margin_tensor - inter_class_dists).mean()


def compute_sliced_wasserstein_loss(x_pred, x_target, num_projections=64):
    if num_projections <= 0:
        return torch.tensor(0.0, device=x_pred.device)
    projections = torch.randn(num_projections, x_pred.shape[1], device=x_pred.device)
    projections = projections / (projections.norm(dim=1, keepdim=True) + 1e-8)
    proj_pred = x_pred @ projections.T
    proj_target = x_target @ projections.T
    proj_pred, _ = torch.sort(proj_pred, dim=0)
    proj_target, _ = torch.sort(proj_target, dim=0)
    return torch.mean((proj_pred - proj_target) ** 2)


def compute_local_alignment_loss(x_pred, x_target, k=5):
    n_samples = x_pred.shape[0]
    if k <= 0 or n_samples < 2:
        return torch.tensor(0.0, device=x_pred.device)
    k = min(k, n_samples - 1)
    with torch.no_grad():
        target_dist = torch.cdist(x_target, x_target, p=2)
        _, nn_idx = torch.topk(target_dist, k + 1, largest=False)
        nn_idx = nn_idx[:, 1:]
        target_nn = target_dist.gather(1, nn_idx)
    pred_dist = torch.cdist(x_pred, x_pred, p=2)
    pred_nn = pred_dist.gather(1, nn_idx)
    return torch.mean((pred_nn - target_nn) ** 2)


# =============================================================================
# TRAINING
# =============================================================================

def train_diffusion(args):
    """Train class-conditioned diffusion model"""
    
    # Hyperparameters (matching original)
    LATENT_DIM = 512
    SMILE_DIM = 512
    NUM_CLASSES = 8
    TIMESTEPS = 50
    HIDDEN_DIM = 512
    NUM_LAYERS = 6
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 256
    MAX_EPOCHS = 1000
    BETA_START = 0.001
    
    # Loss weights
    NOISE_WEIGHT = args.noise_weight
    SEPARATION_WEIGHT = args.separation_weight
    SEPARATION_MARGIN = args.separation_margin
    
    # Early stopping
    PATIENCE = 100
    
    print(f"\n{'='*80}")
    print(f"Training diffusion model with NORMALIZED latents")
    print(f"Beta schedule: [{BETA_START}, {args.beta_end}]")
    print(f"{'='*80}\n")
    
    # Load data
    train_latent, train_labels, test_latent, test_labels = load_precomputed_latents()
    
    # Normalize data
    train_latent_norm, test_latent_norm, data_mean, data_std = normalize_data(train_latent, test_latent)
    
    smile_embeddings = load_smile_embeddings()
    
    # Create dataloaders
    train_loader, unique_classes, class_to_idx = create_dataloaders(
        train_latent_norm, train_labels, smile_embeddings, BATCH_SIZE
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
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_EPOCHS)
    
    # Beta schedule
    betas = get_beta_schedule(TIMESTEPS, BETA_START, args.beta_end).to(device)
    
    # Initialize wandb
    wandb.login(key="57680a36aa570ba8df25adbdd143df3d0bf6b6e8")
    wandb.init(
        project="ims-latent-diffusion-ablation",
        name=f"normalized_beta{args.beta_end:.2f}{f'_{args.model_tag}' if args.model_tag else ''}",
        config={
            "beta_end": args.beta_end,
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
            "margin_mode": args.margin_mode,
            "margin_quantile": args.margin_quantile,
            "margin_min": args.margin_min,
            "margin_max": args.margin_max,
            "swd_weight": args.swd_weight,
            "swd_projections": args.swd_projections,
            "local_align_weight": args.local_align_weight,
            "local_align_k": args.local_align_k,
            "normalized": True,
            "data_mean": data_mean,
            "data_std": data_std,
        }
    )
    
    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_noise_loss = 0
        epoch_sep_loss = 0
        epoch_swd_loss = 0
        epoch_local_align_loss = 0
        last_margin_value = SEPARATION_MARGIN
        
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
            
            # Separation loss (with adaptive margin if requested)
            inter_class_dists = compute_inter_class_dists(x_pred, batch_idx)
            if inter_class_dists is not None and args.margin_mode == "adaptive-percentile":
                margin_value = torch.quantile(inter_class_dists, args.margin_quantile)
                margin_value = torch.clamp(margin_value, args.margin_min, args.margin_max)
            else:
                margin_value = torch.tensor(SEPARATION_MARGIN, device=x_pred.device)
            sep_loss = compute_separation_loss(inter_class_dists, margin_value)
            last_margin_value = float(margin_value.detach().cpu())

            # Optional geometry regularizers
            swd_loss = (
                compute_sliced_wasserstein_loss(
                    x_pred, batch_latent, num_projections=args.swd_projections
                )
                if args.swd_weight > 0
                else torch.tensor(0.0, device=x_pred.device)
            )
            local_align_loss = (
                compute_local_alignment_loss(x_pred, batch_latent, k=args.local_align_k)
                if args.local_align_weight > 0
                else torch.tensor(0.0, device=x_pred.device)
            )
            
            # Combined loss
            loss = (
                NOISE_WEIGHT * noise_loss
                + SEPARATION_WEIGHT * sep_loss
                + args.swd_weight * swd_loss
                + args.local_align_weight * local_align_loss
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_noise_loss += noise_loss.item()
            epoch_sep_loss += sep_loss.item()
            epoch_swd_loss += swd_loss.item()
            epoch_local_align_loss += local_align_loss.item()
        
        avg_noise_loss = epoch_noise_loss / len(train_loader)
        avg_sep_loss = epoch_sep_loss / len(train_loader)
        avg_swd_loss = (
            epoch_swd_loss / len(train_loader) if args.swd_weight > 0 else 0.0
        )
        avg_local_align_loss = (
            epoch_local_align_loss / len(train_loader) if args.local_align_weight > 0 else 0.0
        )
        avg_total_loss = (
            NOISE_WEIGHT * avg_noise_loss
            + SEPARATION_WEIGHT * avg_sep_loss
            + args.swd_weight * avg_swd_loss
            + args.local_align_weight * avg_local_align_loss
        )
        scheduler.step()
        
        # Log
        wandb.log({
            "epoch": epoch + 1,
            "total_loss": avg_total_loss,
            "noise_loss": avg_noise_loss,
            "separation_loss": avg_sep_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "swd_loss": avg_swd_loss,
            "local_align_loss": avg_local_align_loss,
            "margin_value": last_margin_value,
        })
        
        print(f"Epoch {epoch+1}: Total={avg_total_loss:.6f}, Noise={avg_noise_loss:.6f}, Sep={avg_sep_loss:.6f}")
        
        # Save best model
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            epochs_without_improvement = 0
            tag = f"_{args.model_tag}" if args.model_tag else ""
            model_name = f'diffusion_normalized_beta{args.beta_end:.2f}{tag}_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'beta_end': args.beta_end,
                'data_mean': data_mean,
                'data_std': data_std,
            }, os.path.join(MODELS_DIR, model_name))
            print(f"  ✓ Saved best model to {model_name} (loss: {best_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly stopping: No improvement for {PATIENCE} epochs")
                break
    
    wandb.finish()
    print("\n✓ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta_end', type=float, required=True, help='Beta end value (e.g., 0.02 or 0.2)')
    parser.add_argument('--noise-weight', type=float, default=0.8, help='Weight for noise prediction loss.')
    parser.add_argument('--separation-weight', type=float, default=0.2, help='Weight for separation loss.')
    parser.add_argument('--separation-margin', type=float, default=5.0, help='Base separation margin.')
    parser.add_argument(
        '--margin-mode',
        choices=['fixed', 'adaptive-percentile'],
        default='fixed',
        help='Margin selection strategy.',
    )
    parser.add_argument(
        '--margin-quantile',
        type=float,
        default=0.2,
        help='Quantile for adaptive-percentile margin.',
    )
    parser.add_argument('--margin-min', type=float, default=0.0, help='Minimum adaptive margin.')
    parser.add_argument('--margin-max', type=float, default=10.0, help='Maximum adaptive margin.')
    parser.add_argument('--swd-weight', type=float, default=0.0, help='Weight for sliced Wasserstein loss.')
    parser.add_argument(
        '--swd-projections',
        type=int,
        default=64,
        help='Number of random projections for sliced Wasserstein loss.',
    )
    parser.add_argument(
        '--local-align-weight',
        type=float,
        default=0.0,
        help='Weight for local manifold alignment loss.',
    )
    parser.add_argument(
        '--local-align-k',
        type=int,
        default=5,
        help='Number of neighbors for local alignment loss.',
    )
    parser.add_argument(
        '--model-tag',
        default='',
        help='Optional tag appended to saved model filenames.',
    )
    args = parser.parse_args()
    
    train_diffusion(args)
