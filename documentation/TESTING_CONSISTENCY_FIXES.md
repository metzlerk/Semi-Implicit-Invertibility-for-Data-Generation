# Testing Consistency Fixes - October 27, 2025

## Problem Identified
The experiment testing section had inconsistent evaluation procedures across different training methods, which would have made fair comparison impossible.

## Issues Fixed

### 1. **Inconsistent Sample Generation (CRITICAL)**

**Before:**
- **Implicit training**: Generated samples separately for forward/inverse using appropriate models
- **Other methods**: Generated all samples at once using a single model call
- This meant implicit was tested differently than semi-implicit, semi-explicit, and explicit

**After:**
- **ALL methods** now follow the same testing procedure:
  1. Separate test samples by z-coordinate (forward vs inverse)
  2. Generate forward samples (z=0) using the appropriate model
  3. Generate inverse samples (z=1) using the appropriate model
  4. For implicit: uses `model_obj` for forward, `inverse_model_obj` for inverse
  5. For others: uses `model_obj` for both forward and inverse

### 2. **Inconsistent Label Passing for Convex Sampling**

**Before:**
```python
# Implicit method
batch_labels=test_labels_tensor[forward_mask] if sampling_type == 'convex' else None

# Other methods
batch_labels=test_labels_tensor  # Always passed when convex
```

**After:**
```python
# ALL methods use consistent logic:
if sampling_type == 'convex':
    # Pass labels
    batch_labels=test_labels_tensor[mask]
else:
    # Don't pass labels
    # (no batch_labels parameter)
```

### 3. **Missing Backward Pass in Implicit Training (BUG)**

**Before:**
```python
forward_loss = F.mse_loss(...)
epoch_loss += forward_loss.item()  # WRONG: Adding .item() loses gradients

# ...

optimizer.step()  # No backward() call!
```

**After:**
```python
total_batch_loss = 0

forward_loss = F.mse_loss(...)
total_batch_loss += forward_loss  # Keep as tensor
epoch_loss_dict['forward'] += forward_loss.item()  # Track separately

# ...

if total_batch_loss > 0:
    total_batch_loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters
    epoch_loss += total_batch_loss.item()  # Track for logging
```

### 4. **Redundant Code in Testing**

**Before:**
```python
# Forward and inverse masks computed once
z_values = test_input_tensor[:, -1]
forward_mask = z_values == 0
inverse_mask = z_values == 1

# ... sample generation ...

# Then computed AGAIN for metrics
z_values = test_input_tensor[:, -1]
forward_mask = z_values == 0
inverse_mask = z_values == 1
```

**After:**
- Masks computed once at the beginning
- Reused for both sample generation and metrics calculation

## Impact

### Fair Comparison Now Guaranteed
All 16 experiments now use **identical testing procedures**:
1. Same test dataset (200 samples from MNIST1D)
2. Same separation of forward/inverse samples
3. Same label handling for convex sampling
4. Same metrics calculation (overall, forward-only, inverse-only)

### Metrics Computed
For each experiment, we now consistently measure:
- **Overall metrics**: MSE, MAE, KL divergence, FID score (combined forward + inverse)
- **Forward-only metrics**: MSE, FID for denoising task (z=0)
- **Inverse-only metrics**: MSE, FID for prime inverse task (z=1)

### Training Now Works Correctly
- Implicit training actually computes gradients and updates both models
- All training methods properly accumulate losses
- Loss dictionaries track all components consistently

## Files Modified
- `scripts/diffusion_mnist1d_experiments.py`
  - Lines 1375-1440: Fixed implicit training loop
  - Lines 1487-1560: Unified testing procedure
  - Lines 1560-1580: Removed redundant mask computation

## Validation Checklist
✅ All 4 training methods use same testing procedure  
✅ Labels passed consistently for convex sampling  
✅ Forward and inverse samples generated separately for all methods  
✅ Implicit training computes gradients correctly  
✅ No redundant computations  
✅ All metrics calculated identically across experiments  

## Ready for Experiments
The code is now ready for your 16-experiment sweep. All results will be directly comparable because the testing methodology is identical across all configurations.
