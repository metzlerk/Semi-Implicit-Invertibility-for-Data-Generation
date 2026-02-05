#!/usr/bin/env python3
"""
Fine-tune the pre-trained encoder to improve chemical class separation in latent space.

Strategy:
- Load pre-trained encoder/generator
- Add separation loss (margin-based contrastive loss) to encoder output
- Fine-tune encoder while keeping generator frozen initially
- Then jointly fine-tune both with reconstruction + separation loss
"""

import os
os.environ['WANDB_MODE'] = 'disabled'

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import OrderedDict

# Paths
DATA_DIR = 'Data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'

# Training hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
EPOCHS_PHASE1 = 50  # Encoder-only with separation loss
EPOCHS_PHASE2 = 50  # Joint with both losses
SEPARATION_MARGIN = 10.0  # Margin for contrastive loss
SEPARATION_WEIGHT = 0.5  # Weight for separation loss vs reconstruction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# =============================================================================
# MODEL ARCHITECTURE (from original training script)
# =============================================================================

class BiasOnlyLayer(nn.Module):
    def __init__(self, bias_init, trainable=True):
        super().__init__()
        if trainable:
            self.bias = nn.Parameter(bias_init)
        else:
            self.register_buffer('bias', bias_init)
    
    def forward(self, x):
        return x + self.bias

class FlexibleNLayersEncoder(nn.Module):
    """Encoder: IMS spectra (1676-dim) -> Latent space (512-dim)"""
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
    """Generator: Latent space (512-dim) -> IMS spectra (1676-dim)"""
    def __init__(self, input_size=512, output_size=1676, n_layers=9, init_style=None, bkg=None, trainable=False):
        super().__init__()
        if init_style == 'bkg':
            if bkg is None:
                raise ValueError("Must provide `bkg` tensor when init_style='bkg'")
            self.bias_layer = BiasOnlyLayer(bias_init=bkg, trainable=trainable)
        
        # Use np.linspace to match original training (produces 512->641->770->...->1676)
        layers = []
        layer_sizes = np.linspace(input_size, output_size, n_layers + 1, dtype=int)
        for i in range(n_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < n_layers - 1:
                layers.append(nn.LeakyReLU(inplace=True))
        
        self.generator = nn.Sequential(*layers)
    
    def forward(self, x, use_bias=False):
        x = self.generator(x)
        if use_bias and hasattr(self, 'bias_layer'):
            x = self.bias_layer(x)
        return x

# =============================================================================
# SEPARATION LOSS
# =============================================================================

def compute_separation_loss(latents, labels, margin=10.0):
    """
    Contrastive loss to push different chemical classes apart in latent space.
    Encourages inter-class distances > margin while minimizing intra-class variance.
    """
    unique_labels = torch.unique(labels)
    if len(unique_labels) < 2:
        return torch.tensor(0.0, device=latents.device)
    
    # Compute class centroids
    centroids = []
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 0:
            centroids.append(latents[mask].mean(dim=0))
    
    if len(centroids) < 2:
        return torch.tensor(0.0, device=latents.device)
    
    centroids = torch.stack(centroids)
    
    # Inter-class separation: push centroids apart
    n_classes = len(centroids)
    distances = torch.cdist(centroids, centroids, p=2)
    mask = ~torch.eye(n_classes, dtype=torch.bool, device=distances.device)
    inter_class_dists = distances[mask]
    
    # Penalize distances below margin
    separation_loss = torch.relu(margin - inter_class_dists).mean()
    
    # Intra-class compactness: pull samples toward their centroid
    compactness_loss = 0.0
    for i, label in enumerate(unique_labels):
        mask = labels == label
        if mask.sum() > 1:
            class_samples = latents[mask]
            centroid = centroids[i]
            distances = torch.norm(class_samples - centroid, dim=1)
            compactness_loss += distances.mean()
    
    compactness_loss /= len(unique_labels)
    
    # Combined: maximize separation, minimize compactness
    return separation_loss + 0.1 * compactness_loss

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load IMS data with labels"""
    print("Loading IMS data...")
    
    train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))
    test_df = pd.read_feather(os.path.join(DATA_DIR, 'test_data.feather'))
    
    # Extract labels
    onehot_cols = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
    train_labels = train_df[onehot_cols].values.argmax(axis=1)
    test_labels = test_df[onehot_cols].values.argmax(axis=1)
    
    # Extract IMS features
    p_cols = [c for c in train_df.columns if c.startswith('p_')]
    n_cols = [c for c in train_df.columns if c.startswith('n_')]
    
    train_ims = train_df[p_cols + n_cols].values.astype(np.float32)
    test_ims = test_df[p_cols + n_cols].values.astype(np.float32)
    
    print(f"  Train: {len(train_ims)} samples")
    print(f"  Test: {len(test_ims)} samples")
    print(f"  Features: {train_ims.shape[1]}")
    print(f"  Classes: {len(onehot_cols)}")
    
    return train_ims, train_labels, test_ims, test_labels

# =============================================================================
# FINE-TUNING
# =============================================================================

def evaluate_separation(encoder, data_loader, device, use_bias=True):
    """Compute silhouette score to measure separation quality"""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    
    encoder.eval()
    all_latents = []
    all_labels = []
    
    with torch.no_grad():
        for batch_ims, batch_labels in data_loader:
            batch_ims = batch_ims.to(device)
            latents = encoder(batch_ims, use_bias=use_bias)
            all_latents.append(latents.cpu().numpy())
            all_labels.append(batch_labels.numpy())
    
    all_latents = np.vstack(all_latents)
    all_labels = np.concatenate(all_labels)
    
    # Compute PCA for visualization
    pca = PCA(n_components=2)
    latents_pca = pca.fit_transform(all_latents)
    
    # Silhouette score (higher is better, range [-1, 1])
    sil_score = silhouette_score(latents_pca, all_labels)
    
    return sil_score, pca.explained_variance_ratio_[:2].sum()

def phase1_encoder_separation(encoder, train_ims, train_labels, test_ims, test_labels, use_bias=True):
    """Phase 1: Fine-tune encoder with separation loss only (generator frozen)"""
    print("\n" + "="*80)
    print("PHASE 1: Fine-tuning Encoder with Separation Loss")
    print("="*80)
    
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_ims),
        torch.LongTensor(train_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_ims),
        torch.LongTensor(test_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initial evaluation
    init_sil, init_var = evaluate_separation(encoder, test_loader, device, use_bias)
    print(f"\nInitial test silhouette: {init_sil:.4f} (var: {init_var:.1%})")
    
    best_sil = init_sil
    
    for epoch in range(EPOCHS_PHASE1):
        encoder.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_PHASE1}")
        for batch_ims, batch_labels in pbar:
            batch_ims = batch_ims.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Encode
            latents = encoder(batch_ims, use_bias=use_bias)
            
            # Separation loss
            loss = compute_separation_loss(latents, batch_labels, margin=SEPARATION_MARGIN)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            sil_score, var_explained = evaluate_separation(encoder, test_loader, device, use_bias)
            print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, sil={sil_score:.4f}, var={var_explained:.1%}")
            
            if sil_score > best_sil:
                best_sil = sil_score
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'epoch': epoch,
                    'silhouette': sil_score
                }, os.path.join(MODELS_DIR, 'encoder_separated_phase1.pth'))
                print(f"  ✓ Saved best encoder (sil: {best_sil:.4f})")
    
    final_sil, final_var = evaluate_separation(encoder, test_loader, device, use_bias)
    print(f"\nPhase 1 complete!")
    print(f"  Initial silhouette: {init_sil:.4f}")
    print(f"  Final silhouette: {final_sil:.4f}")
    print(f"  Improvement: {final_sil - init_sil:+.4f}")
    
    return encoder

def phase2_joint_training(encoder, generator, train_ims, train_labels, test_ims, test_labels, use_bias=True):
    """Phase 2: Joint fine-tuning with reconstruction + separation loss"""
    print("\n" + "="*80)
    print("PHASE 2: Joint Fine-tuning (Reconstruction + Separation)")
    print("="*80)
    
    encoder.train()
    generator.train()
    
    params = list(encoder.parameters()) + list(generator.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-5)
    recon_criterion = nn.MSELoss()
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_ims),
        torch.LongTensor(train_labels)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_ims),
        torch.LongTensor(test_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    best_combined_metric = -float('inf')
    
    for epoch in range(EPOCHS_PHASE2):
        encoder.train()
        generator.train()
        
        epoch_recon_loss = 0
        epoch_sep_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS_PHASE2}")
        for batch_ims, batch_labels in pbar:
            batch_ims = batch_ims.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Encode
            latents = encoder(batch_ims, use_bias=use_bias)
            
            # Decode
            reconstructed = generator(latents, use_bias=use_bias)
            
            # Reconstruction loss
            recon_loss = recon_criterion(reconstructed, batch_ims)
            
            # Separation loss
            sep_loss = compute_separation_loss(latents, batch_labels, margin=SEPARATION_MARGIN)
            
            # Combined loss
            loss = recon_loss + SEPARATION_WEIGHT * sep_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            epoch_recon_loss += recon_loss.item()
            epoch_sep_loss += sep_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'recon': f"{recon_loss.item():.4f}",
                'sep': f"{sep_loss.item():.4f}"
            })
        
        avg_recon = epoch_recon_loss / num_batches
        avg_sep = epoch_sep_loss / num_batches
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            sil_score, var_explained = evaluate_separation(encoder, test_loader, device, use_bias)
            combined_metric = sil_score - 0.1 * avg_recon  # Balance separation and reconstruction
            
            print(f"Epoch {epoch+1}: recon={avg_recon:.4f}, sep={avg_sep:.4f}, sil={sil_score:.4f}, var={var_explained:.1%}")
            
            if combined_metric > best_combined_metric:
                best_combined_metric = combined_metric
                torch.save({
                    'encoder_state_dict': encoder.state_dict(),
                    'generator_state_dict': generator.state_dict(),
                    'epoch': epoch,
                    'silhouette': sil_score,
                    'recon_loss': avg_recon
                }, os.path.join(MODELS_DIR, 'autoencoder_separated.pth'))
                print(f"  ✓ Saved best model (metric: {combined_metric:.4f})")
    
    final_sil, final_var = evaluate_separation(encoder, test_loader, device, use_bias)
    print(f"\nPhase 2 complete!")
    print(f"  Final silhouette: {final_sil:.4f}")
    print(f"  Final reconstruction: {avg_recon:.4f}")
    
    return encoder, generator

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("ENCODER FINE-TUNING FOR IMPROVED CHEMICAL SEPARATION")
    print("="*80)
    
    # Load data
    train_ims, train_labels, test_ims, test_labels = load_data()
    
    # Initialize models
    print("\nInitializing models...")
    encoder = FlexibleNLayersEncoder(
        input_size=1676,
        output_size=512,
        n_layers=9,
        init_style='bkg',
        bkg=torch.zeros(1676),
        trainable=True
    ).to(device)
    
    generator = FlexibleNLayersGenerator(
        input_size=512,
        output_size=1676,
        n_layers=9,
        init_style='bkg',
        bkg=torch.zeros(1676),
        trainable=True
    ).to(device)
    
    # Load pre-trained weights
    checkpoint_path = os.path.join(MODELS_DIR, 'best_autoencoder.pth')
    if os.path.exists(checkpoint_path):
        print(f"Loading pre-trained autoencoder from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load with strict=False to handle minor mismatches
        missing_keys_enc = encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        missing_keys_gen = generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
        
        print("  ✓ Loaded encoder")
        if missing_keys_enc.missing_keys:
            print(f"    Missing keys: {missing_keys_enc.missing_keys[:5]}")
        if missing_keys_enc.unexpected_keys:
            print(f"    Unexpected keys: {missing_keys_enc.unexpected_keys[:5]}")
        
        print("  ✓ Loaded generator")
        if missing_keys_gen.missing_keys:
            print(f"    Missing keys: {missing_keys_gen.missing_keys[:5]}")
        if missing_keys_gen.unexpected_keys:
            print(f"    Unexpected keys: {missing_keys_gen.unexpected_keys[:5]}")
    else:
        print(f"WARNING: No pre-trained model found at {checkpoint_path}")
        print("Training from scratch...")
    
    # =========================================================================
    # VALIDATION TESTS - Fail fast if there are issues
    # =========================================================================
    print("\n" + "="*80)
    print("RUNNING VALIDATION TESTS")
    print("="*80)
    
    # Test 1: Data shapes
    print("\n1. Testing data shapes...")
    assert train_ims.shape[1] == 1676, f"Expected 1676 features, got {train_ims.shape[1]}"
    assert len(train_labels) == len(train_ims), f"Label/data mismatch: {len(train_labels)} vs {len(train_ims)}"
    assert train_labels.min() >= 0 and train_labels.max() <= 7, f"Invalid labels range: {train_labels.min()}-{train_labels.max()}"
    print(f"   ✓ Train data: {train_ims.shape}, labels: {len(train_labels)}")
    print(f"   ✓ Test data: {test_ims.shape}, labels: {len(test_labels)}")
    
    # Test 2: Model forward passes
    print("\n2. Testing model forward passes...")
    test_batch = torch.FloatTensor(train_ims[:8]).to(device)
    test_labels_batch = torch.LongTensor(train_labels[:8]).to(device)
    
    encoder.eval()
    generator.eval()
    
    with torch.no_grad():
        # Encoder
        latent = encoder(test_batch, use_bias=True)
        assert latent.shape == (8, 512), f"Expected (8, 512) latent, got {latent.shape}"
        assert not torch.isnan(latent).any(), "NaN in encoder output"
        print(f"   ✓ Encoder output: {latent.shape}, range: [{latent.min():.2f}, {latent.max():.2f}]")
        
        # Generator
        reconstructed = generator(latent, use_bias=True)
        assert reconstructed.shape == (8, 1676), f"Expected (8, 1676) reconstruction, got {reconstructed.shape}"
        assert not torch.isnan(reconstructed).any(), "NaN in generator output"
        print(f"   ✓ Generator output: {reconstructed.shape}, range: [{reconstructed.min():.2f}, {reconstructed.max():.2f}]")
        
        # Reconstruction error
        recon_error = torch.nn.functional.mse_loss(reconstructed, test_batch)
        print(f"   ✓ Reconstruction MSE: {recon_error.item():.6f}")
        # Note: High error is expected if starting from untrained checkpoint
        if recon_error.item() > 1e6:
            print(f"   ⚠ Warning: Very high reconstruction error - may indicate model not trained")
    
    # Test 3: Separation loss
    print("\n3. Testing separation loss...")
    encoder.train()
    latent_train = encoder(test_batch, use_bias=True)
    sep_loss = compute_separation_loss(latent_train, test_labels_batch, margin=SEPARATION_MARGIN)
    assert not torch.isnan(sep_loss), "NaN in separation loss"
    assert sep_loss.item() >= 0, f"Negative separation loss: {sep_loss.item()}"
    print(f"   ✓ Separation loss: {sep_loss.item():.4f}")
    
    # Test 4: Optimizer
    print("\n4. Testing optimizer...")
    test_optimizer = torch.optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)
    test_optimizer.zero_grad()
    sep_loss.backward()
    total_grad_norm = 0
    for p in encoder.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item()
    print(f"   ✓ Gradients computed, total norm: {total_grad_norm:.2f}")
    assert total_grad_norm > 0, "No gradients computed"
    
    # Test 5: Check GPU memory
    if torch.cuda.is_available():
        print("\n5. GPU memory check...")
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"   ✓ GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    print("\n" + "="*80)
    print("✓ ALL VALIDATION TESTS PASSED - Starting training...")
    print("="*80)
    
    # Phase 1: Encoder separation
    encoder = phase1_encoder_separation(
        encoder, train_ims, train_labels, test_ims, test_labels, use_bias=True
    )
    
    # Phase 2: Joint fine-tuning
    encoder, generator = phase2_joint_training(
        encoder, generator, train_ims, train_labels, test_ims, test_labels, use_bias=True
    )
    
    # Save final latent representations
    print("\nSaving updated latent representations...")
    encoder.eval()
    
    with torch.no_grad():
        train_tensor = torch.FloatTensor(train_ims).to(device)
        test_tensor = torch.FloatTensor(test_ims).to(device)
        
        train_latent = []
        for i in range(0, len(train_tensor), BATCH_SIZE):
            batch = train_tensor[i:i+BATCH_SIZE]
            latent = encoder(batch, use_bias=True)
            train_latent.append(latent.cpu().numpy())
        
        test_latent = []
        for i in range(0, len(test_tensor), BATCH_SIZE):
            batch = test_tensor[i:i+BATCH_SIZE]
            latent = encoder(batch, use_bias=True)
            test_latent.append(latent.cpu().numpy())
        
        train_latent = np.vstack(train_latent)
        test_latent = np.vstack(test_latent)
    
    np.save(os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy'), train_latent)
    np.save(os.path.join(RESULTS_DIR, 'autoencoder_test_latent_separated.npy'), test_latent)
    
    print("\n" + "="*80)
    print("✓ FINE-TUNING COMPLETE!")
    print("="*80)
    print(f"\nSaved files:")
    print(f"  - {MODELS_DIR}/encoder_separated_phase1.pth (encoder only)")
    print(f"  - {MODELS_DIR}/autoencoder_separated.pth (encoder + generator)")
    print(f"  - {RESULTS_DIR}/autoencoder_train_latent_separated.npy")
    print(f"  - {RESULTS_DIR}/autoencoder_test_latent_separated.npy")
    print("\nNext steps:")
    print("  1. Retrain diffusion model with new separated latents")
    print("  2. Or swap latent files and regenerate PCA plots")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
