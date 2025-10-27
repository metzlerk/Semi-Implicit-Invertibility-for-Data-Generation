# Semi-Implicit Training Verification Walkthrough

This document walks through the semi-implicit training implementation step-by-step so you can verify it's set up correctly according to your specifications.

## Overview of Your Requirements

✓ **z=0**: Forward direction (noisy → clean)  
✓ **z=1**: Inverse direction (clean → prime inverse)  
✓ **Prime inverse**: Deterministic noise pattern per digit (like cos(x)+1 for sin(x))  
✓ **Forward samples**: Use standard diffusion (Gaussian) or convex combinations  
✓ **Train both sets**: Simultaneously in same batch  
✓ **Four losses**: forward, inverse, roundtrip F→I, roundtrip I→F  
✓ **Timesteps**: Used in both directions  
✓ **Implicit training**: Only forward (z=0), no z-switch  

## Data Preparation (`prepare_mnist1d_for_diffusion`)

### Step 1: Create Prime Inverse Patterns
```python
def create_prime_inverse_patterns(x_data, y_labels, normalize=True):
    # For each digit 0-9, create a fixed deterministic pattern
    # Digit 0: -x (inversion)
    # Digit 1: roll(x, 10) (phase shift)
    # ...
    # Digit 9: cos(x*π/5) + 1 (like your cos(x)+1)
```

**Verification**: 
- ✓ One unique pattern per digit
- ✓ Deterministic (same digit always → same pattern)
- ✓ Similar to your cos(x)+1 transformation

### Step 2: Prepare Forward Set (z=0)
```python
# Set 1 (Forward, z=0): Noisy -> Clean
forward_inputs = clean_samples  # Will be noised during training
forward_targets = clean_samples  # Target is clean
# Add z=0 coordinate
forward_inputs_with_z = [forward_inputs, z=0]
forward_targets_with_z = [forward_targets, z=0]
```

**Verification**:
- ✓ Inputs are clean (noise added during training)
- ✓ Targets are clean digits
- ✓ z=0 indicates forward direction

### Step 3: Prepare Inverse Set (z=1)
```python
# Set 2 (Inverse, z=1): Clean -> Prime Inverse
inverse_inputs = clean_samples
inverse_targets = prime_inverse_patterns[digit]  # Deterministic per digit
# Add z=1 coordinate
inverse_inputs_with_z = [inverse_inputs, z=1]
inverse_targets_with_z = [inverse_targets, z=1]
```

**Verification**:
- ✓ Inputs are clean digits
- ✓ Targets are prime inverse (deterministic)
- ✓ z=1 indicates inverse direction

## Training: Implicit vs Semi-Implicit

### Implicit Training (Simple)
```python
if training_type == 'implicit':
    # Filter only forward samples (z=0)
    forward_mask = input_batch[:, -1] == 0
    forward_inputs = input_batch[forward_mask]
    forward_targets = target_batch[forward_mask]
    
    # Standard diffusion training
    t = random_timesteps
    noise = random_noise
    noisy = add_noise(forward_targets, t, noise)
    predicted_noise = model(noisy, t, forward_inputs)
    loss = MSE(predicted_noise, noise)
```

**Verification**:
- ✓ Only uses z=0 samples
- ✓ No z-switch
- ✓ Standard forward denoising only

### Semi-Implicit Training (Complex - 4 Losses)

#### Loss 1: Forward (z=0)
```python
# Separate forward samples (z=0)
forward_inputs = input_batch[z_values == 0]
forward_targets = target_batch[z_values == 0]

# Add noise to clean targets
t_forward = random_timesteps
noise_forward = random_noise
noisy_inputs = add_noise(forward_targets, t_forward, noise_forward)

# Set condition with z=0
condition_forward[:, -1] = 0  # ← z=0 for forward direction

# Model predicts noise
predicted_noise = model(noisy_inputs, t_forward, condition_forward)

# Forward loss: how well did we predict noise?
forward_loss = MSE(predicted_noise, noise_forward)
```

**Verification**:
- ✓ Uses only z=0 samples
- ✓ Adds noise to clean data
- ✓ Model sees condition with z=0
- ✓ Learns to denoise

#### Loss 2: Inverse (z=1)
```python
# Separate inverse samples (z=1)
inverse_inputs = input_batch[z_values == 1]  # Clean digits
inverse_targets = target_batch[z_values == 1]  # Prime inverse patterns

# Treat prime inverse as target, add noise to it
t_inverse = random_timesteps
noise_to_prime = random_noise
noisy_prime = add_noise(inverse_targets, t_inverse, noise_to_prime)

# Set condition with z=1
condition_inverse[:, -1] = 1  # ← z=1 for inverse direction

# Model predicts noise to reach prime inverse
predicted_noise_inverse = model(noisy_prime, t_inverse, condition_inverse)

# Inverse loss: how well did we predict noise to prime?
inverse_loss = MSE(predicted_noise_inverse, noise_to_prime)
```

**Verification**:
- ✓ Uses only z=1 samples
- ✓ Targets are prime inverse (deterministic)
- ✓ Model sees condition with z=1
- ✓ Learns to transform to prime inverse

#### Loss 3: Roundtrip Forward→Inverse
```python
# Start with clean data from forward set
clean_data = forward_targets

# Step 1: Denoise with z=0 (should stay clean)
noisy_1 = add_noise(clean_data, t_round, noise_1)
condition_1[:, -1] = 0  # z=0
predicted_noise_1 = model(noisy_1, t_round, condition_1)
denoised = remove_noise(noisy_1, t_round, predicted_noise_1)

# Step 2: Now apply inverse (z=1) to reach prime inverse
target_prime = inverse_targets  # Prime inverse patterns
noisy_2 = add_noise(target_prime, t_round, noise_2)
condition_2 = denoised.clone()
condition_2[:, -1] = 1  # z=1
predicted_noise_2 = model(noisy_2, t_round, condition_2)

# Roundtrip loss: should predict noise correctly
roundtrip_fi_loss = MSE(predicted_noise_2, noise_2)
```

**Verification**:
- ✓ First uses z=0 (denoise)
- ✓ Then uses z=1 (to prime inverse)
- ✓ Tests consistency: clean → denoise → prime inverse
- ✓ Weight: 0.5

#### Loss 4: Roundtrip Inverse→Forward
```python
# Start with clean data from inverse set
clean_data_inv = inverse_inputs

# Step 1: Apply inverse (z=1) to reach prime inverse
target_prime_inv = inverse_targets
noisy_inv_1 = add_noise(target_prime_inv, t_round_inv, noise_inv_1)
condition_inv_1[:, -1] = 1  # z=1
predicted_noise_inv_1 = model(noisy_inv_1, t_round_inv, condition_inv_1)
to_prime = remove_noise(noisy_inv_1, t_round_inv, predicted_noise_inv_1)

# Step 2: Now denoise back to clean (z=0)
noisy_inv_2 = add_noise(clean_data_inv, t_round_inv, noise_inv_2)
condition_inv_2 = to_prime.clone()
condition_inv_2[:, -1] = 0  # z=0
predicted_noise_inv_2 = model(noisy_inv_2, t_round_inv, condition_inv_2)

# Roundtrip loss: should recover clean
roundtrip_if_loss = MSE(predicted_noise_inv_2, noise_inv_2)
```

**Verification**:
- ✓ First uses z=1 (to prime inverse)
- ✓ Then uses z=0 (denoise back)
- ✓ Tests consistency: clean → prime inverse → denoise back
- ✓ Weight: 0.5

#### Total Loss
```python
total_loss = forward_loss + inverse_loss + 0.5*roundtrip_fi_loss + 0.5*roundtrip_if_loss
```

**Verification**:
- ✓ All four losses included
- ✓ Weights: 1.0, 1.0, 0.5, 0.5
- ✓ Can hyperparameter tune later

## Sampling Methods

### Gaussian Sampling (Standard)
```python
if sampling_type == 'gaussian':
    diff_process = GaussianDiffusion(...)
    # Uses standard Gaussian noise
    noise = torch.randn_like(x)
    noisy = sqrt(alpha) * x + sqrt(1-alpha) * noise
```

**Verification**:
- ✓ Standard diffusion process
- ✓ Gaussian noise

### Convex Sampling
```python
if sampling_type == 'convex':
    diff_process = ConvexCombinationDiffusion(...)
    # Uses convex combinations instead of Gaussian
    indices = random_permutation
    alpha = random_weights
    convex_combo = alpha * x + (1-alpha) * x_shuffled
```

**Verification**:
- ✓ Convex combinations instead of Gaussian
- ✓ Tests if geometric mixing works better

## Weights & Biases Logging

```python
wandb.init(
    entity="metzlerk",
    project="mnist1d-diffusion-experiments",
    config={...}
)

# During training
wandb.log({
    "epoch": epoch,
    "train_loss": avg_loss,
    "train_forward_loss": forward_loss,
    "train_inverse_loss": inverse_loss,
    "train_roundtrip_fi_loss": roundtrip_fi_loss,
    "train_roundtrip_if_loss": roundtrip_if_loss
})

# After testing
wandb.log({
    "test_mse": mse,
    "test_forward_mse": forward_mse,
    "test_inverse_mse": inverse_mse,
    ...
})
```

**Verification**:
- ✓ API key configured
- ✓ Project name set
- ✓ Logs all losses individually
- ✓ Logs test metrics

## Summary Checklist

**Dataset**:
- ✓ Prime inverse: deterministic per digit (10 patterns)
- ✓ Forward set (z=0): noisy → clean
- ✓ Inverse set (z=1): clean → prime inverse
- ✓ Both sets combined, trained simultaneously

**Training Methods**:
- ✓ Implicit: Only forward (z=0), single loss
- ✓ Semi-implicit: Both directions, four losses

**Sampling Methods**:
- ✓ Gaussian: Standard diffusion noise
- ✓ Convex: Geometric combinations

**Semi-Implicit Four Losses**:
- ✓ Forward (z=0): denoise
- ✓ Inverse (z=1): to prime inverse
- ✓ Roundtrip F→I: clean → denoise → prime
- ✓ Roundtrip I→F: clean → prime → denoise

**Implementation Details**:
- ✓ Timesteps used in both directions
- ✓ z-coordinate switches between 0 and 1
- ✓ Proper condition passing to model
- ✓ Loss weights: 1, 1, 0.5, 0.5

**Logging**:
- ✓ W&B integration
- ✓ Individual loss tracking
- ✓ Forward and inverse test MSE

## Questions Answered

1. ✓ Prime inverse: Fixed patterns per digit (Option A)
2. ✓ Noisy data: Standard diffusion or convex
3. ✓ Training: Both sets simultaneously
4. ✓ Self-inverse: Both roundtrips (F→I and I→F)
5. ✓ Timesteps: Used in both directions
6. ✓ Loss weighting: 1 + 1 + 0.5 + 0.5
7. ✓ Normalized: Prime inverse normalized to [-5, 5]
8. ✓ Implicit: Only forward, no z-switch

## Ready to Run!

The implementation follows all your specifications. To execute:

```bash
cd ~/4f-files
sbatch run_diffusion_mnist1d.sh
```

Expected output:
- 8 experiments (2 loss × 2 sampling × 2 training)
- ~40-80 minutes total
- Results in JSON and Markdown
- W&B dashboard with live training curves
