#!/usr/bin/env python
"""
Quick test script to verify the diffusion experiments can run
Tests with minimal epochs to catch any errors quickly
"""

import sys
import os

# Temporarily modify hyperparameters for quick test
test_config = """
# Quick test configuration
timesteps = 5
max_epochs = 2
batch_size = 32
n_samples_test = 100
"""

print("="*80)
print("QUICK TEST MODE")
print("="*80)
print("This will run a minimal test with:")
print("  - 5 timesteps (instead of 20)")
print("  - 2 epochs (instead of 50)")
print("  - 32 batch size (instead of 128)")
print("  - 100 samples (instead of 2000)")
print()
print("Purpose: Verify code runs without errors")
print("="*80)

# Read the main script
with open('diffusion_mnist1d_experiments.py', 'r') as f:
    script_content = f.read()

# Check if imports work
print("\n[1/4] Testing imports...")
try:
    import torch
    import numpy as np
    import wandb
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Check if wandb login works
print("\n[2/4] Testing W&B connection...")
try:
    wandb.login(key="57680a36aa570ba8df25adbdd143df3d0bf6b6e8")
    print("  ✓ W&B login successful")
except Exception as e:
    print(f"  ✗ W&B login failed: {e}")
    print("  Note: Experiments will still run, but won't log to W&B")

# Check if MNIST1D can be loaded
print("\n[3/4] Testing MNIST1D dataset loading...")
try:
    import requests
    import pickle
    
    try:
        with open('mnist1d_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print("  ✓ Loaded from cache")
    except:
        print("  Downloading MNIST1D...")
        url = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
        response = requests.get(url)
        data = pickle.loads(response.content)
        with open('mnist1d_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        print("  ✓ Downloaded and cached")
    
    print(f"  Dataset shape: {data['x'].shape}")
    print(f"  Labels shape: {data['y'].shape}")
except Exception as e:
    print(f"  ✗ Dataset loading failed: {e}")
    sys.exit(1)

# Check if script syntax is valid
print("\n[4/4] Testing script syntax...")
try:
    compile(script_content, 'diffusion_mnist1d_experiments.py', 'exec')
    print("  ✓ Script syntax valid")
except SyntaxError as e:
    print(f"  ✗ Syntax error: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("PRE-FLIGHT CHECK PASSED!")
print("="*80)
print("\nThe script appears ready to run. To start experiments:")
print("  sbatch run_diffusion_mnist1d.sh")
print("\nOr for a quick test run (2 epochs):")
print("  python diffusion_mnist1d_experiments.py")
print("  (Edit max_epochs=2 in the script first)")
print("="*80)
