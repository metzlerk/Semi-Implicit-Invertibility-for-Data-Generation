# Corrected Semi-Implicit Training Implementation

## What Changed

The initial implementation misunderstood the semi-implicit training concept. Here's what was corrected:

### Original (Incorrect) Understanding
- Two transformation sets with different z-coordinates
- Both sets learned input→target mappings
- Semi-implicit added roundtrip consistency

### Corrected Understanding
- **z-coordinate is a directional indicator**, not just a label
- **z=0**: Forward direction (noisy → clean)
- **z=1**: Inverse direction (clean → prime inverse)
- Prime inverse is a **deterministic noise pattern per digit class**

## Key Concepts

### 1. Prime Inverse
A deterministic "noise-like" pattern for each digit class (0-9), similar to how `sin(x) → cos(x)+1` in the original experiment.

**Examples**:
- Digit 0: Inversion `-x`
- Digit 1: Phase shift `roll(x, 10)`
- Digit 9: Cosine transform `cos(x*π/5) + 1`

Each digit always maps to its specific prime inverse pattern.

### 2. Directional Training
The model learns TWO directions simultaneously:

**Forward (z=0)**:
```
noisy_data + z=0 → model → clean_digit
```

**Inverse (z=1)**:
```
clean_digit + z=1 → model → prime_inverse_pattern
```

### 3. Four Losses

#### Loss 1: Forward (Denoise)
```python
# Noisy data with z=0 should denoise to clean
forward_loss = MSE(model(noisy, t, z=0), clean)
```

#### Loss 2: Inverse (To Prime)
```python
# Clean data with z=1 should transform to prime inverse
inverse_loss = MSE(model(clean, t, z=1), prime_inverse)
```

#### Loss 3: Roundtrip Forward→Inverse
```python
# Clean → denoise(z=0) → result → noise to prime(z=1) → should reach prime inverse
denoised = model(add_noise(clean), t, z=0)
to_prime = model(denoised, t, z=1)
roundtrip_fi_loss = MSE(to_prime, prime_inverse)
```

#### Loss 4: Roundtrip Inverse→Forward
```python
# Clean → to prime(z=1) → result → denoise(z=0) → should recover clean
to_prime = model(clean, t, z=1)
recovered = model(add_noise(to_prime), t, z=0)
roundtrip_if_loss = MSE(recovered, clean)
```

#### Total Loss
```python
total_loss = forward_loss + inverse_loss + 0.5*roundtrip_fi_loss + 0.5*roundtrip_if_loss
```

## Implicit vs Semi-Implicit

### Implicit Training (Standard)
- **Only uses forward direction** (z=0)
- Single loss: MSE(model(noisy, t, condition), clean)
- No z-switch, no inverse mapping
- Model learns: noisy → clean

### Semi-Implicit Training
- **Uses both directions** (z=0 and z=1)
- Four losses (forward, inverse, two roundtrips)
- Model learns: 
  - noisy → clean (z=0)
  - clean → prime inverse (z=1)
  - Bidirectional consistency

## Why Prime Inverse Instead of Random Noise?

**Problem with random noise**:
```
clean → add Gaussian noise → random_noise₁
clean → add Gaussian noise → random_noise₂  (different!)
```
Cannot reliably reverse: which noise should we go to?

**Solution with prime inverse**:
```
clean_digit_0 → always goes to → prime_inverse_0 (deterministic!)
clean_digit_1 → always goes to → prime_inverse_1 (deterministic!)
```
Can reliably reverse because target is predictable.

## Dataset Structure

```
Total dataset: 2000 samples

Set 1 (1000 samples, z=0):
  Input: Clean MNIST1D digits (will be noised during training) + z=0
  Target: Clean MNIST1D digits (same as input before noising)
  Purpose: Learn to denoise

Set 2 (1000 samples, z=1):
  Input: Clean MNIST1D digits + z=1
  Target: Prime inverse patterns (deterministic per digit class)
  Purpose: Learn to transform to prime inverse
```

During training:
- Set 1 inputs get noise added: `noisy = add_noise(clean)`
- Set 2 inputs stay clean, but target is prime inverse

## Weights & Biases Integration

Each experiment logs:
- Training loss per epoch
- Individual losses (forward, inverse, roundtrip_fi, roundtrip_if) for semi-implicit
- Test metrics (MSE, MAE, KL divergence)
- Separate forward and inverse test MSE
- Runtime and convergence rate

Project: `mnist1d-diffusion-experiments`
Entity: `metzlerk`

## Verification Checklist

✓ Forward samples (z=0) train on noisy→clean with forward_loss
✓ Inverse samples (z=1) train on clean→prime_inverse with inverse_loss
✓ Roundtrip F→I: clean → denoise → to_prime → loss
✓ Roundtrip I→F: clean → to_prime → denoise → loss
✓ Implicit training uses only z=0 samples (forward only)
✓ Prime inverse is deterministic (one pattern per digit)
✓ Both sets trained simultaneously in same batch
✓ Timesteps used in both directions
✓ Weights & Biases logging integrated

## Running the Experiments

```bash
cd ~/4f-files
sbatch run_diffusion_mnist1d.sh
```

Or for testing:
```bash
python diffusion_mnist1d_experiments.py
```

Expected runtime: ~40-80 minutes for all 8 experiments

## Output Files

1. `mnist1d_diffusion_experiments_YYYYMMDD_HHMMSS.json` - Complete results
2. `mnist1d_experiment_report_YYYYMMDD_HHMMSS.md` - Summary report
3. Weights & Biases dashboard - Interactive visualizations

## Next Steps

After experiments complete:
1. Check W&B dashboard for training curves
2. Compare MSE between implicit and semi-implicit
3. Analyze forward vs inverse test MSE
4. Check if roundtrip losses converge
5. Verify bidirectional consistency
