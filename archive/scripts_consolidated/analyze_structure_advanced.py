#!/usr/bin/env python3
"""
Advanced Structure Analysis
============================
Once samples are generated, perform deeper analysis:
- MMD (Maximum Mean Discrepancy) between real and generated
- Coverage and precision metrics
- Manifold alignment analysis
- Cross-distance comparisons
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import seaborn as sns

RESULTS_DIR = 'results'
DATA_DIR = 'Data'
IMAGES_DIR = 'images'

def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """Compute Maximum Mean Discrepancy between two distributions"""
    n, m = len(X), len(Y)
    
    if gamma is None:
        # Use median heuristic
        all_data = np.vstack([X[:1000], Y[:1000]])
        dists = cdist(all_data, all_data, 'euclidean')
        gamma = 1.0 / np.median(dists[dists > 0])
    
    # Compute kernel matrices
    XX = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
    YY = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
    XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
    
    mmd = XX.sum() / (n * n) + YY.sum() / (m * m) - 2 * XY.sum() / (n * m)
    
    return mmd

def compute_coverage_precision(real, fake, k=5):
    """
    Compute coverage and precision metrics
    Coverage: % of real manifold covered by generated samples
    Precision: % of generated samples that are near real manifold
    """
    # Build k-NN on real data
    nbrs_real = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(real)
    nbrs_fake = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(fake)
    
    # Coverage: For each real point, check if any fake point is in its k-NN
    dists_real_to_fake, _ = nbrs_fake.kneighbors(real)
    coverage = (dists_real_to_fake[:, 0] < np.percentile(dists_real_to_fake, 50)).mean()
    
    # Precision: For each fake point, check if it's close to a real point
    dists_fake_to_real, _ = nbrs_real.kneighbors(fake)
    precision = (dists_fake_to_real[:, 0] < np.percentile(dists_fake_to_real, 50)).mean()
    
    return coverage, precision

def analyze_structure_preservation(real_latents, gen_latents, real_labels, gen_labels, chemical_names):
    """Comprehensive structure preservation analysis"""
    
    print("="*80)
    print("ADVANCED STRUCTURE ANALYSIS")
    print("="*80)
    
    results = {}
    
    # Global metrics
    print("\n1. Global MMD (Maximum Mean Discrepancy)")
    print("-"*80)
    
    # Use subset for speed
    n_sample = min(5000, len(real_latents), len(gen_latents))
    real_subset = real_latents[np.random.choice(len(real_latents), n_sample, replace=False)]
    gen_subset = gen_latents[np.random.choice(len(gen_latents), n_sample, replace=False)]
    
    mmd_global = compute_mmd(real_subset, gen_subset)
    print(f"Global MMD: {mmd_global:.6f}")
    print("(Lower is better; <0.01 is excellent)")
    results['mmd_global'] = mmd_global
    
    # Coverage and precision
    print("\n2. Coverage and Precision")
    print("-"*80)
    coverage, precision = compute_coverage_precision(real_subset, gen_subset, k=5)
    print(f"Coverage:  {coverage:.4f} (% of real manifold covered)")
    print(f"Precision: {precision:.4f} (% of generated on-manifold)")
    results['coverage'] = coverage
    results['precision'] = precision
    
    # Per-class analysis
    print("\n3. Per-Class Structure Metrics")
    print("-"*80)
    print(f"{'Chemical':<10} {'MMD':<12} {'Coverage':<12} {'Precision':<12}")
    print("-"*80)
    
    per_class_results = {}
    
    for class_idx, chemical in enumerate(chemical_names):
        # Get class-specific data
        real_mask = real_labels[:, class_idx] == 1
        gen_mask = gen_labels[:, class_idx] == 1
        
        real_class = real_latents[real_mask]
        gen_class = gen_latents[gen_mask]
        
        # Sample if needed
        n_class_sample = min(1000, len(real_class), len(gen_class))
        if len(real_class) > n_class_sample:
            real_class = real_class[np.random.choice(len(real_class), n_class_sample, replace=False)]
        if len(gen_class) > n_class_sample:
            gen_class = gen_class[np.random.choice(len(gen_class), n_class_sample, replace=False)]
        
        # Compute metrics
        mmd_class = compute_mmd(real_class, gen_class)
        cov_class, prec_class = compute_coverage_precision(real_class, gen_class, k=5)
        
        print(f"{chemical:<10} {mmd_class:>11.6f} {cov_class:>11.4f} {prec_class:>11.4f}")
        
        per_class_results[chemical] = {
            'mmd': mmd_class,
            'coverage': cov_class,
            'precision': prec_class
        }
    
    results['per_class'] = per_class_results
    
    # Manifold distance analysis
    print("\n4. Manifold Distance Analysis")
    print("-"*80)
    
    # For each generated point, find distance to nearest real point
    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(real_subset)
    dists_gen_to_real, _ = nbrs.kneighbors(gen_subset)
    
    # For each real point, find distance to nearest other real point (baseline)
    nbrs_baseline = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(real_subset)
    dists_real_to_real, _ = nbrs_baseline.kneighbors(real_subset)
    dists_real_to_real = dists_real_to_real[:, 1]  # Skip self (0th neighbor)
    
    print(f"Real-to-Real nearest neighbor: {dists_real_to_real.mean():.2f} ± {dists_real_to_real.std():.2f}")
    print(f"Generated-to-Real nearest:     {dists_gen_to_real.mean():.2f} ± {dists_gen_to_real.std():.2f}")
    print(f"Ratio (Gen/Real):              {dists_gen_to_real.mean() / dists_real_to_real.mean():.3f}")
    print("(Ratio near 1.0 means generated samples lie on the same manifold)")
    
    results['real_to_real_dist'] = dists_real_to_real.mean()
    results['gen_to_real_dist'] = dists_gen_to_real.mean()
    results['manifold_ratio'] = dists_gen_to_real.mean() / dists_real_to_real.mean()
    
    # Visualization
    print("\n5. Creating visualizations...")
    
    # Distance distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.hist(dists_real_to_real.flatten(), bins=50, alpha=0.6, label='Real-to-Real', color='blue', density=True)
    ax.hist(dists_gen_to_real.flatten(), bins=50, alpha=0.6, label='Generated-to-Real', color='red', density=True)
    ax.set_xlabel('Distance to Nearest Neighbor', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Manifold Distance Distributions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Per-class metrics
    ax = axes[1]
    x = np.arange(len(chemical_names))
    width = 0.25
    
    mmds = [per_class_results[c]['mmd'] * 100 for c in chemical_names]  # Scale for vis
    covs = [per_class_results[c]['coverage'] * 100 for c in chemical_names]
    precs = [per_class_results[c]['precision'] * 100 for c in chemical_names]
    
    ax.bar(x - width, mmds, width, label='MMD (×100)', alpha=0.8)
    ax.bar(x, covs, width, label='Coverage (%)', alpha=0.8)
    ax.bar(x + width, precs, width, label='Precision (%)', alpha=0.8)
    
    ax.set_xlabel('Chemical', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Structure Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(chemical_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'normalized_diffusion_advanced_metrics.png'), 
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(IMAGES_DIR, 'normalized_diffusion_advanced_metrics.png')}")
    plt.close()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Global MMD: {mmd_global:.6f} {'(EXCELLENT)' if mmd_global < 0.01 else '(GOOD)' if mmd_global < 0.05 else '(NEEDS IMPROVEMENT)'}")
    print(f"✓ Coverage:   {coverage:.3f} {'(EXCELLENT)' if coverage > 0.9 else '(GOOD)' if coverage > 0.7 else '(NEEDS IMPROVEMENT)'}")
    print(f"✓ Precision:  {precision:.3f} {'(EXCELLENT)' if precision > 0.9 else '(GOOD)' if precision > 0.7 else '(NEEDS IMPROVEMENT)'}")
    print(f"✓ Manifold Ratio: {results['manifold_ratio']:.3f} {'(ON MANIFOLD)' if 0.8 < results['manifold_ratio'] < 1.5 else '(OFF MANIFOLD)'}")
    
    return results

if __name__ == "__main__":
    print("Checking for generated samples...")
    
    # Check if samples exist
    generated_file = os.path.join(RESULTS_DIR, 'normalized_diffusion_samples.npy')
    if not os.path.exists(generated_file):
        print(f"ERROR: Generated samples not found at {generated_file}")
        print("Run evaluate_normalized_structure.py first to generate samples.")
        exit(1)
    
    # Load data
    print("\nLoading data...")
    train_latent = np.load(os.path.join(RESULTS_DIR, 'autoencoder_train_latent_separated.npy'))
    train_df = pd.read_feather(os.path.join(DATA_DIR, 'train_data.feather'))
    
    label_columns = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
    train_labels = train_df[label_columns].values
    
    generated_latents = np.load(generated_file)
    generated_labels = np.load(os.path.join(RESULTS_DIR, 'normalized_diffusion_labels.npy'))
    
    print(f"Real samples: {len(train_latent)}")
    print(f"Generated samples: {len(generated_latents)}")
    
    # Run analysis
    results = analyze_structure_preservation(
        train_latent, generated_latents,
        train_labels, generated_labels,
        label_columns
    )
    
    # Save results
    results_simple = {
        'mmd_global': float(results['mmd_global']),
        'coverage': float(results['coverage']),
        'precision': float(results['precision']),
        'real_to_real_dist': float(results['real_to_real_dist']),
        'gen_to_real_dist': float(results['gen_to_real_dist']),
        'manifold_ratio': float(results['manifold_ratio'])
    }
    
    import json
    with open(os.path.join(RESULTS_DIR, 'normalized_diffusion_structure_metrics.json'), 'w') as f:
        json.dump(results_simple, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {os.path.join(RESULTS_DIR, 'normalized_diffusion_structure_metrics.json')}")
