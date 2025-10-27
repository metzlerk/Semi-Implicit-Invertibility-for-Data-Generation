# Semi-Implicit Training Visual Walkthrough

## Complete Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      SEMI-IMPLICIT TRAINING STEP                        │
└─────────────────────────────────────────────────────────────────────────┘

INPUTS:
  • input_points (batch_size, 41)  - Original MNIST1D data
  • target_points (batch_size, 41) - Transformed target data  
  • condition_points (batch_size, 41) - Same as input_points

═══════════════════════════════════════════════════════════════════════════
PHASE 1: FORWARD DENOISING (Standard Diffusion Training)
═══════════════════════════════════════════════════════════════════════════

Step 1.1: Sample random timesteps
┌─────────────────────┐
│ t ~ Uniform(0, 20)  │  → [batch_size] tensor of integers
└─────────────────────┘
    Example: [5, 12, 3, 18, 7, ...]

Step 1.2: Add noise to target points
┌─────────────────────────────────────────────────────────────┐
│ noise = N(0, I)  [batch_size, 41]                          │
│                                                              │
│ noisy_points = √(α_t) * target_points                      │
│              + √(1-α_t) * noise                            │
└─────────────────────────────────────────────────────────────┘
    This is the forward diffusion process: q(x_t | x_0)
    α_t controls how much signal vs noise

Step 1.3: Model predicts the noise
┌──────────────────────────────────────────────────────────────┐
│ predicted_noise = model(noisy_points, t, condition_points) │
│                         ↓                                    │
│                    [batch_size, 41]                         │
└──────────────────────────────────────────────────────────────┘
    Model takes:
      - noisy_points: data with noise
      - t: how much noise was added
      - condition_points: context for conditional generation

Step 1.4: Calculate forward loss
┌─────────────────────────────────────────────────────┐
│ forward_loss = MSE(predicted_noise, actual_noise) │
└─────────────────────────────────────────────────────┘
    ✓ Gradients WILL flow through this
    ✓ This is the standard diffusion training objective

═══════════════════════════════════════════════════════════════════════════
PHASE 2: INVERSE CONSTRAINT (Self-Inverse Property)
═══════════════════════════════════════════════════════════════════════════

Step 2.1: Estimate clean data (NO GRADIENTS!)
┌────────────────────────────────────────────────────────────────┐
│ with torch.no_grad():                                         │
│   predicted_clean = (noisy_points - √(1-α_t) * pred_noise)  │
│                   / √(α_t)                                   │
│   predicted_clean = clamp(predicted_clean, -5, 5)           │
└────────────────────────────────────────────────────────────────┘
    ✗ No gradients - this is just using the model's prediction
    ✓ This is what the model "thinks" the clean data looks like
    
    Intuition: We reverse the noise addition formula to get clean estimate

Step 2.2: Add noise AGAIN (at different timestep!)
┌──────────────────────────────────────────────────────────────┐
│ t_inverse ~ Uniform(0, 20)  [batch_size]                   │
│ noise_inverse = N(0, I)  [batch_size, 41]                  │
│                                                              │
│ renoised_points = √(α_t_inv) * predicted_clean            │
│                 + √(1-α_t_inv) * noise_inverse            │
└──────────────────────────────────────────────────────────────┘
    NOTE: t_inverse is DIFFERENT from t
    We're testing if model can handle noise at various levels

Step 2.3: Denoise the renoised data
┌────────────────────────────────────────────────────────────────────┐
│ pred_noise_inv = model(renoised_points, t_inverse, condition)   │
│                                                                    │
│ recovered_clean = (renoised_points - √(1-α_t_inv)*pred_noise_inv)│
│                 / √(α_t_inv)                                     │
│ recovered_clean = clamp(recovered_clean, -5, 5)                 │
└────────────────────────────────────────────────────────────────────┘
    ✓ Gradients WILL flow through this
    ✓ Model processes renoised data to recover clean version

Step 2.4: Calculate inverse loss
┌────────────────────────────────────────────────────────────────┐
│ inverse_loss = MSE(recovered_clean, predicted_clean.detach())│
└────────────────────────────────────────────────────────────────┘
    ✓ Gradients flow through recovered_clean
    ✗ No gradients through predicted_clean (.detach())
    
    This enforces: denoise(noise(x)) ≈ x

═══════════════════════════════════════════════════════════════════════════
PHASE 3: OPTIMIZATION
═══════════════════════════════════════════════════════════════════════════

Step 3.1: Combine losses
┌────────────────────────────────────────────────┐
│ total_loss = forward_loss + 0.5 * inverse_loss│
└────────────────────────────────────────────────┘
    forward_loss: teaches noise prediction
    inverse_loss: enforces self-consistency
    0.5 weight: balances the two objectives

Step 3.2: Backpropagate and update
┌────────────────────────────┐
│ total_loss.backward()     │
│ optimizer.step()          │
└────────────────────────────┘
```

## Data Flow Example with Numbers

Let's trace one batch through the system:

```
INITIAL DATA:
  batch_size = 4
  feature_dim = 41
  
  target_points.shape = (4, 41)
  Example values: [[0.5, -0.3, ..., 0.0],  ← Sample 1, z=0
                   [0.2,  0.8, ..., 0.0],  ← Sample 2, z=0
                   [-0.1, 0.4, ..., 1.0],  ← Sample 3, z=1
                   [0.9, -0.2, ..., 1.0]]  ← Sample 4, z=1

FORWARD PASS:
  t = [5, 12, 18, 3]  ← Random timesteps
  
  For sample 0 (t=5):
    √(α_5) ≈ 0.95
    √(1-α_5) ≈ 0.31
    
    noisy_points[0] = 0.95 * target_points[0] + 0.31 * noise[0]
    
  Model predicts noise → predicted_noise.shape = (4, 41)
  forward_loss = MSE between predicted and actual noise

INVERSE CONSTRAINT:
  predicted_clean = denoise using predicted_noise
  predicted_clean.shape = (4, 41)
  
  t_inverse = [15, 2, 9, 11]  ← Different random timesteps!
  
  renoised_points = add noise again using t_inverse
  renoised_points.shape = (4, 41)
  
  recovered_clean = denoise renoised_points
  recovered_clean.shape = (4, 41)
  
  inverse_loss = MSE(recovered_clean, predicted_clean)
  
  Should be small if model is consistent!

OPTIMIZATION:
  total_loss = forward_loss + 0.5 * inverse_loss
  Gradients flow → update model weights
```

## Comparison: Implicit vs Semi-Implicit

```
┌──────────────────────────────────────────────────────────────────┐
│ IMPLICIT TRAINING (Standard)                                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  target ──[add noise]──> noisy ──[model]──> pred_noise         │
│                                                  │               │
│                                                  ↓               │
│                                            compare with noise    │
│                                                  │               │
│                                                  ↓               │
│                                          forward_loss            │
│                                                  │               │
│                                                  ↓               │
│                                             backward()           │
│                                                                  │
│  Only one denoising step per training iteration                │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│ SEMI-IMPLICIT TRAINING (Our Method)                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────── Forward Path ───────────────────┐        │
│  │ target ──[add noise]──> noisy ──[model]──> pred_noise│      │
│  │                                         │             │       │
│  │                                         ↓             │       │
│  │                                   forward_loss        │       │
│  └───────────────────────────────────────────────────────┘       │
│                           ↓ (no grad)                            │
│  ┌─────────────────── Inverse Path ───────────────────┐         │
│  │                 predicted_clean                      │        │
│  │                       │                              │        │
│  │                       ↓                              │        │
│  │            [add noise again]                         │        │
│  │                       │                              │        │
│  │                       ↓                              │        │
│  │                 renoised_points                      │        │
│  │                       │                              │        │
│  │                       ↓                              │        │
│  │                   [model]                            │        │
│  │                       │                              │        │
│  │                       ↓                              │        │
│  │                recovered_clean                       │        │
│  │                       │                              │        │
│  │                       ↓                              │        │
│  │         compare with predicted_clean (detached)      │        │
│  │                       │                              │        │
│  │                       ↓                              │        │
│  │                 inverse_loss                         │        │
│  └──────────────────────────────────────────────────────┘        │
│                           │                                       │
│                           ↓                                       │
│              total_loss = forward + 0.5*inverse                  │
│                           │                                       │
│                           ↓                                       │
│                      backward()                                  │
│                                                                  │
│  Two denoising steps per iteration - enforces consistency       │
└──────────────────────────────────────────────────────────────────┘
```

## Why the .detach() is Critical

```
WITHOUT .detach():
┌────────────────────────────────────────────────────────────────┐
│  Gradients would flow:                                        │
│                                                                │
│  model → pred_noise → predicted_clean → renoised → model     │
│           ↑_______________(creates cycle!)______________|     │
│                                                                │
│  Problem: Gradient explosion, unstable training              │
└────────────────────────────────────────────────────────────────┘

WITH .detach():
┌────────────────────────────────────────────────────────────────┐
│  Gradients flow only through inverse denoising:               │
│                                                                │
│  predicted_clean (frozen) → renoised → model → recovered      │
│                                          ↑                     │
│                                    (gradients here)            │
│                                                                │
│  Benefit: Stable training, clear gradient signal              │
└────────────────────────────────────────────────────────────────┘
```

## Summary

**Semi-implicit training adds a self-consistency constraint:**
- Model learns: "If I denoise something, add noise again, and denoise, I should get the same answer"
- This is stronger than just learning to predict noise
- Helps with generalization and robustness
- Requires careful gradient management (detach) to avoid cycles

**The key insight:**
A good denoising model should be **self-inverse** in the sense that its predictions are consistent across multiple noise levels and operations.
