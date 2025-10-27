# Semi-Implicit Training Explanation (CORRECTED)

## Overview
Semi-implicit training is a bidirectional learning technique where the model learns both forward and inverse transformations using a directional indicator (z-coordinate).

**Key Concept**: The z-coordinate tells the model which direction to go:
- **z=0**: Forward direction (denoise noisy data → clean data)
- **z=1**: Inverse direction (clean data → prime inverse pattern)

The model learns to be **bidirectionally consistent** across four losses.

## Dataset Structure

### Set 1: Forward Direction (z=0)
- **Input**: Noisy MNIST1D data + z=0 indicator
- **Target**: Clean MNIST1D digits (0-9)
- **What model learns**: "When z=0, denoise to recover clean digits"

### Set 2: Inverse Direction (z=1)
- **Input**: Clean MNIST1D digits (0-9) + z=1 indicator
- **Target**: Prime inverse patterns (deterministic noise, one pattern per digit)
- **What model learns**: "When z=1, transform to the prime inverse pattern"

### Prime Inverse Patterns
Each digit class (0-9) has a fixed deterministic noise pattern:
- Digit 0 → inversion: `-x`
- Digit 1 → phase shift: `roll(x, 10)`
- Digit 2 → inversion + scale: `-0.8*x`
- Digit 3 → reverse: `x[::-1]`
- Digit 4 → add sine wave
- Digit 5 → scale and shift: `1.5*x - 1`
- Digit 6 → negative phase shift
- Digit 7 → absolute transformation
- Digit 8 → power transformation
- Digit 9 → cosine-like: `cos(x*π/5) + 1`

Similar to original experiment: `sin(x) → cos(x) + 1`

## Step-by-Step Process

### Standard Implicit Training (for comparison)
In standard diffusion model training:
1. Take clean target data `x_0`
2. Add noise at timestep `t` → get `x_t` (noisy data)
3. Model predicts the noise: `predicted_noise = model(x_t, t, condition_z0)`
4. Loss: MSE between `predicted_noise` and actual `noise`
5. **Only trains forward direction (z=0)**

This teaches the model to predict what noise was added.

### Semi-Implicit Training (our approach)

Training uses **four losses** to ensure bidirectional consistency:

#### Loss 1: Forward Loss (Denoising)
For samples with z=0:

1. **Start with clean digits**
   ```
   clean_data = [batch_size, 40 features]
   ```

2. **Add noise at timestep t**
   ```
   t = random_int(0, timesteps, size=batch_size)
   noise = random_normal(like=clean_data)
   noisy_data = sqrt(alpha_t) * clean_data + sqrt(1-alpha_t) * noise
   ```

3. **Create condition with z=0**
   ```
   condition = [noisy_data, z=0]
   ```

4. **Model predicts the noise**
   ```
   predicted_noise = model(noisy_data, t, condition_with_z0)
   ```

5. **Calculate forward loss**
   ```
   forward_loss = MSE(predicted_noise, actual_noise)
   ```

#### Loss 2: Inverse Loss (To Prime Inverse)
For samples with z=1:

1. **Start with clean digits**
   ```
   clean_data = [batch_size, 40 features]
   prime_inverse = [batch_size, 40 features]  # Deterministic pattern for each digit
   ```

2. **Add noise toward prime inverse**
   ```
   t = random_int(0, timesteps, size=batch_size)
   noise = random_normal(like=prime_inverse)
   noisy_prime = sqrt(alpha_t) * prime_inverse + sqrt(1-alpha_t) * noise
   ```

3. **Create condition with z=1**
   ```
   condition = [clean_data, z=1]
   ```

4. **Model predicts noise to reach prime inverse**
   ```
   predicted_noise = model(noisy_prime, t, condition_with_z1)
   ```

5. **Calculate inverse loss**
   ```
   inverse_loss = MSE(predicted_noise, actual_noise)
   ```

#### Loss 3: Roundtrip Forward → Inverse
Ensures: clean → denoise (z=0) → still clean → add noise (z=1) → prime inverse

1. **Start with clean data**
2. **Add noise and denoise with z=0** (should recover clean)
3. **Take result and add noise with z=1** (should reach prime inverse)
4. **Loss**: MSE between predicted and actual noise in step 3

#### Loss 4: Roundtrip Inverse → Forward
Ensures: clean → to prime inverse (z=1) → denoise back (z=0) → clean

1. **Start with clean data**
2. **Transform to prime inverse with z=1**
3. **Denoise back to clean with z=0**
4. **Loss**: MSE between predicted and actual noise in step 3

#### Combined Loss
```
total_loss = forward_loss + inverse_loss + 0.5 * roundtrip_fi + 0.5 * roundtrip_if
```

Weights:
- Forward and inverse: 1.0 each (primary objectives)
- Roundtrips: 0.5 each (consistency regularization)

## Why Does This Work?

### Intuition
- **Forward loss**: Teaches model to denoise (z=0 direction)
- **Inverse loss**: Teaches model to map to prime inverse (z=1 direction)
- **Roundtrip losses**: Enforce bidirectional consistency

### Key Insight: Prime Inverse
Unlike standard diffusion which can only go from random noise → clean, semi-implicit training creates a **deterministic inverse mapping**. 

- Random noise → clean: Not reliably reversible
- Clean → prime inverse: Deterministic and reversible!

The model learns:
1. `model(noisy, t, z=0)` → denoise to clean
2. `model(clean, t, z=1)` → transform to prime inverse
3. These operations should be consistent in roundtrips

### Mathematical Property
For a good semi-implicit model:
- `denoise(clean, z=0) ≈ clean` (denoising clean data shouldn't change it)
- `noise_to_prime(clean, z=1) = prime_inverse` (reaches deterministic target)
- `denoise(noise_to_prime(clean), z=0) ≈ clean` (can reverse the operation)
- `noise_to_prime(denoise(noisy), z=1) ≈ prime_inverse` (consistent at different noise levels)

### Benefits
1. **Bidirectional learning**: Model understands both forward and inverse transformations
2. **Deterministic inverse**: Unlike random Gaussian noise, prime inverse is predictable
3. **Better generalization**: Roundtrip losses ensure consistency
4. **Interpretable**: z-coordinate clearly indicates direction

## Data Setup for MNIST1D

To match the original sine/cosine experiment structure, we setup two distinct transformation sets:

### Set 1: Digits 0-4 (labeled with z=0)
- **Input**: Original MNIST1D signals (40 features) + z=0 label → shape (N, 41)
- **Target**: Phase-shifted version of the signal + z=0 label → shape (N, 41)
- **Transformation**: `target = roll(input, shift=5)` (circular shift)

### Set 2: Digits 5-9 (labeled with z=1)
- **Input**: Original MNIST1D signals (40 features) + z=1 label → shape (N, 41)
- **Target**: Inverted and scaled version + z=1 label → shape (N, 41)
- **Transformation**: `target = -0.8 * input` (inversion + scaling)

### Why This Setup?
1. **Two distinct transformations**: Like sine→sin(x) and sin(x)→cos(x)+1 in original
2. **Conditional generation**: The z-coordinate (last feature) tells the model which transformation to apply
3. **Semi-implicit training compatibility**: We have clear input-target pairs for both forward and inverse passes

## Key Differences from Original Notebook

1. **Removed Architecture Comparison**: Only using iterative (single U-Net), not variable (multiple U-Nets)
   - Reduces experiments from 16 to 8
   - Simplifies code and training time

2. **MNIST1D Dataset**: Real-world sequential data instead of synthetic sine/cosine
   - 40 features per sample instead of 3D points
   - More complex patterns to learn
   - Tests if diffusion + semi-implicit works on realistic data

3. **Feature Dimension**: 41 instead of 3
   - Model input_dim and condition_dim are 41
   - All other architecture remains the same

## Verification Checklist

✓ **Forward diffusion** adds noise correctly using q_sample
✓ **Model prediction** happens with noisy input, timestep, and condition
✓ **Forward loss** compares predicted vs actual noise
✓ **Predicted clean** is estimated without gradients (detached)
✓ **Re-noising** happens at potentially different timestep
✓ **Second denoising** processes the re-noised data
✓ **Inverse loss** compares recovered_clean to original predicted_clean (detached)
✓ **Combined loss** uses weighted sum: forward + 0.5 * inverse
✓ **Two dataset splits** with different transformations based on z-coordinate

## Running the Experiment

1. **On login node** (for testing):
   ```bash
   cd 4f-files
   python diffusion_mnist1d_experiments.py
   ```

2. **On compute node** (recommended):
   ```bash
   cd 4f-files
   sbatch run_diffusion_mnist1d.sh
   ```

## Expected Output

The script will:
1. Download MNIST1D dataset (or load from cache)
2. Prepare data with two transformation sets
3. Run 8 experiments (2 loss × 2 sampling × 2 training)
4. Save results to:
   - `mnist1d_diffusion_experiments_YYYYMMDD_HHMMSS.json`
   - `mnist1d_experiment_report_YYYYMMDD_HHMMSS.md`
5. Print summary statistics for each experiment

Each experiment takes approximately 5-10 minutes, total runtime ~40-80 minutes.
