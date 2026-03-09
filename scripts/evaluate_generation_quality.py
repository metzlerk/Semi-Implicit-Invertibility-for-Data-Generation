"""
Comprehensive evaluation of generated latent quality for beta ablation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_latent_diffusion import (
    ConditionalDiffusionModel, 
    load_precomputed_latents,
    sample_diffusion
)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Directories
MODELS_DIR = ROOT_DIR / 'models'
RESULTS_DIR = ROOT_DIR / 'results'
IMAGES_DIR = ROOT_DIR / 'images'

# Load real data
train_latent, train_labels, test_latent, test_labels, chemical_to_idx = load_precomputed_latents()
print(f"\nReal data loaded:")
print(f"  Train: {train_latent.shape}, std={train_latent.std():.3f}")
print(f"  Test: {test_latent.shape}, std={test_latent.std():.3f}")

# Calculate real data statistics per chemical
chemicals = sorted(chemical_to_idx.keys(), key=lambda x: chemical_to_idx[x])
print(f"\nReal data statistics by chemical:")
print(f"{'Chemical':<10} {'Count':>6} {'Mean':>10} {'Std':>10} {'L2 Norm':>10}")
print("-" * 56)

real_stats = {}
for chem in chemicals:
    idx = chemical_to_idx[chem]
    mask = train_labels == idx
    chem_latents = train_latent[mask]
    l2_norms = np.linalg.norm(chem_latents, axis=1)
    
    real_stats[chem] = {
        'count': len(chem_latents),
        'mean': float(chem_latents.mean()),
        'std': float(chem_latents.std()),
        'l2_mean': float(l2_norms.mean()),
        'l2_std': float(l2_norms.std())
    }
    
    print(f"{chem:<10} {real_stats[chem]['count']:>6} {real_stats[chem]['mean']:>10.4f} "
          f"{real_stats[chem]['std']:>10.4f} {real_stats[chem]['l2_mean']:>10.2f}")

# Calculate inter-class separation for real data
real_centroids = []
for chem in chemicals:
    idx = chemical_to_idx[chem]
    mask = train_labels == idx
    centroid = train_latent[mask].mean(axis=0)
    real_centroids.append(centroid)

real_centroids = np.array(real_centroids)
real_dists = []
for i in range(len(real_centroids)):
    for j in range(i+1, len(real_centroids)):
        dist = np.linalg.norm(real_centroids[i] - real_centroids[j])
        real_dists.append(dist)
real_separation = np.mean(real_dists)
print(f"\nReal inter-class separation: {real_separation:.3f} ± {np.std(real_dists):.3f}")

# Models to compare
models = [
    ('diffusion_latent_separated_beta0.02_best.pt', 0.02, 'Weak'),
    ('diffusion_latent_separated_beta0.20_best.pt', 0.2, 'Strong')
]

# Generate samples and evaluate
n_samples = 500
results = {}

for model_name, beta_end, label in models:
    print(f"\n{'='*80}")
    print(f"Evaluating: {label} (β_end={beta_end})")
    print(f"{'='*80}")
    
    model_path = MODELS_DIR / model_name
    checkpoint = torch.load(model_path, weights_only=False)
    
    model = ConditionalDiffusionModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate samples for each chemical
    gen_stats = {}
    all_gen_latents = []
    all_gen_labels = []
    gen_centroids = []
    
    for chem in chemicals:
        idx = chemical_to_idx[chem]
        class_label = torch.tensor([idx], device=device)
        
        # Generate samples
        samples = []
        for _ in range(n_samples // 10):  # Batch of 10
            batch_labels = class_label.repeat(10)
            batch_samples = sample_diffusion(model, batch_labels, beta_end=beta_end)
            samples.append(batch_samples.cpu().numpy())
        
        samples = np.concatenate(samples, axis=0)
        all_gen_latents.append(samples)
        all_gen_labels.extend([idx] * len(samples))
        
        # Calculate statistics
        l2_norms = np.linalg.norm(samples, axis=1)
        gen_stats[chem] = {
            'mean': float(samples.mean()),
            'std': float(samples.std()),
            'l2_mean': float(l2_norms.mean()),
            'l2_std': float(l2_norms.std())
        }
        
        centroid = samples.mean(axis=0)
        gen_centroids.append(centroid)
    
    all_gen_latents = np.concatenate(all_gen_latents, axis=0)
    all_gen_labels = np.array(all_gen_labels)
    gen_centroids = np.array(gen_centroids)
    
    # Calculate inter-class separation
    gen_dists = []
    for i in range(len(gen_centroids)):
        for j in range(i+1, len(gen_centroids)):
            dist = np.linalg.norm(gen_centroids[i] - gen_centroids[j])
            gen_dists.append(dist)
    gen_separation = np.mean(gen_dists)
    
    # Print statistics
    print(f"\nGenerated data statistics by chemical:")
    print(f"{'Chemical':<10} {'Mean':>10} {'Std':>10} {'L2 Norm':>10} {'vs Real':>10}")
    print("-" * 60)
    
    for chem in chemicals:
        real_l2 = real_stats[chem]['l2_mean']
        gen_l2 = gen_stats[chem]['l2_mean']
        ratio = gen_l2 / real_l2 if real_l2 > 0 else 0
        
        print(f"{chem:<10} {gen_stats[chem]['mean']:>10.4f} {gen_stats[chem]['std']:>10.4f} "
              f"{gen_stats[chem]['l2_mean']:>10.2f} {ratio:>9.2%}")
    
    print(f"\nInter-class separation: {gen_separation:.3f} ± {np.std(gen_dists):.3f}")
    print(f"  vs Real: {gen_separation/real_separation:.2%}")
    
    # Store results
    results[label] = {
        'beta_end': beta_end,
        'gen_stats': gen_stats,
        'separation': gen_separation,
        'separation_std': float(np.std(gen_dists)),
        'latents': all_gen_latents,
        'labels': all_gen_labels
    }

# Summary comparison
print(f"\n{'='*80}")
print(f"SUMMARY: Quality Comparison")
print(f"{'='*80}")

print(f"\n{'Metric':<30} {'Real':<15} {'Weak (0.02)':<15} {'Strong (0.2)':<15}")
print("-" * 75)

# Overall L2 norm comparison (averaged across chemicals)
real_l2_avg = np.mean([real_stats[c]['l2_mean'] for c in chemicals])
weak_l2_avg = np.mean([results['Weak']['gen_stats'][c]['l2_mean'] for c in chemicals])
strong_l2_avg = np.mean([results['Strong']['gen_stats'][c]['l2_mean'] for c in chemicals])

print(f"{'Avg L2 Norm':<30} {real_l2_avg:<15.2f} {weak_l2_avg:<15.2f} {strong_l2_avg:<15.2f}")
print(f"{'  Ratio vs Real':<30} {'1.00':<15} {weak_l2_avg/real_l2_avg:<15.2f} {strong_l2_avg/real_l2_avg:<15.2f}")

# Separation
print(f"{'Inter-class Separation':<30} {real_separation:<15.2f} {results['Weak']['separation']:<15.2f} {results['Strong']['separation']:<15.2f}")
print(f"{'  Ratio vs Real':<30} {'1.00':<15} {results['Weak']['separation']/real_separation:<15.2f} {results['Strong']['separation']/real_separation:<15.2f}")

print(f"\n{'='*80}")
print(f"CONCLUSION:")
print(f"{'='*80}")
if weak_l2_avg / real_l2_avg < 0.5:
    print(f"❌ Weak diffusion (β=0.02): L2 norm {weak_l2_avg/real_l2_avg:.1%} of real - TOO SMALL")
else:
    print(f"✓ Weak diffusion (β=0.02): L2 norm {weak_l2_avg/real_l2_avg:.1%} of real")

if 0.8 <= strong_l2_avg / real_l2_avg <= 1.2:
    print(f"✓ Strong diffusion (β=0.2): L2 norm {strong_l2_avg/real_l2_avg:.1%} of real - GOOD MATCH")
else:
    print(f"⚠ Strong diffusion (β=0.2): L2 norm {strong_l2_avg/real_l2_avg:.1%} of real")

if results['Strong']['separation'] / real_separation > 2.0:
    print(f"⚠ Strong diffusion over-separates classes ({results['Strong']['separation']/real_separation:.1f}x real)")
elif results['Weak']['separation'] / real_separation < 0.5:
    print(f"❌ Weak diffusion under-separates classes ({results['Weak']['separation']/real_separation:.1f}x real)")

print(f"\n✓ Evaluation complete!")
