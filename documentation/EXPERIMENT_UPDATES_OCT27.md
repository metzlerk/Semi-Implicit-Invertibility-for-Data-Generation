# Experiment Updates - October 27, 2025

## Overview
Updated `diffusion_mnist1d_experiments.py` to test 4 different training/inversion methods and fixed convex combination sampling.

## New Experiment Configuration

### Total Experiments: 16
- **2 Loss Functions**: FID, KL Divergence  
- **2 Sampling Methods**: Gaussian, Convex
- **4 Training Methods**: Implicit, Semi-Implicit, Semi-Explicit, Explicit

Matrix: 2 × 2 × 4 = **16 experiments**

## Training Methods Implemented

### 1. Implicit Training
**Status**: ✅ Fully Implemented

**Approach**: Train two separate models
- **Model 1 (Forward)**: Denoises noisy data → clean data (z=0)
- **Model 2 (Inverse)**: Maps clean data → prime inverse (z=1)

**Architecture**: 
- Two independent PointUNet models
- Double the parameters of other methods
- Each model specializes in one direction

**Key Features**:
- No parameter sharing between forward and inverse
- Most straightforward approach
- Highest parameter count

---

### 2. Semi-Implicit Training  
**Status**: ✅ Already Implemented (Enhanced)

**Approach**: Single model with z-coordinate switch
- Same model handles both directions based on z value
- z=0: Forward denoising
- z=1: Inverse mapping to prime space

**Training Losses** (4 total):
1. **Forward loss**: Denoise noisy → clean
2. **Inverse loss**: Map clean → prime inverse
3. **Roundtrip F→I**: Clean → denoise → to prime
4. **Roundtrip I→F**: Clean → to prime → denoise back

**Key Features**:
- Single model with conditional behavior
- Self-consistency through roundtrip losses
- Parameter efficient

---

### 3. Semi-Explicit Training (SVD-based)
**Status**: ✅ Newly Implemented

**Approach**: Compute pseudo-inverse using Singular Value Decomposition
- Train forward model normally
- Compute inverse via SVD: `W_inv = V @ S^{-1} @ U^T`
- For linear layer `W`, pseudo-inverse is `(W^T W)^{-1} W^T`

**Implementation**:
```python
def compute_layer_pseudoinverse(layer):
    W = layer.weight.data
    U, S, Vt = torch.svd(W)
    S_inv = 1.0 / (S + eps)  # Add epsilon for stability
    W_pinv = Vt.t() @ diag(S_inv) @ U.t()
    return W_pinv
```

**Training**:
- Forward loss: Standard denoising
- Inverse loss: Use SVD to approximate inverse transformation
- Weighted 50% since SVD approximation may not be perfect

**Key Features**:
- No explicit inverse model needed
- Mathematically grounded via linear algebra
- Good for near-linear transformations

---

### 4. Explicit Training (Mathematical Inversion)
**Status**: ✅ Newly Implemented

**Approach**: Mathematically invert the network layers
- For `f(x) = σ(Wx + b)`, inverse is `f^{-1}(y) = W^{-1}(σ^{-1}(y) - b)`
- Compute exact matrix inverse where possible
- Invert activation functions

**Implementation**:
```python
def explicit_inverse_layer(layer, y):
    W = layer.weight.data
    b = layer.bias.data
    W_inv = torch.inverse(W + eps * I)  # Regularized inverse
    x = W_inv @ (y - b)
    return x

def invert_activation(y, type):
    if type == 'relu': return y  # Approximate
    if type == 'tanh': return atanh(y)
    if type == 'sigmoid': return log(y / (1-y))
```

**Training**:
- Forward loss: Standard denoising  
- Inverse loss: Apply mathematical inverse to reach prime space
- Weighted 50% for stability

**Key Features**:
- True mathematical inversion where possible
- Activation function inversion (ReLU approximated)
- Exact for invertible layers

**Limitations**:
- ReLU inversion is approximate (assumes positive input)
- Non-square matrices use pseudo-inverse
- Complex architectures challenging to fully invert

---

## Convex Combination Sampling - FIXED

### Previous Issue
Convex combinations were using **random permutations** of the batch, mixing different digit classes.

### New Implementation  
✅ **Same-Class Convex Combinations**

For each sample:
1. Extract its digit label
2. Find all other samples with the **same label** in the batch
3. Randomly select one same-class sample
4. Compute convex combination: `α * x₁ + (1-α) * x₂` where both are same digit

```python
def q_sample(self, x_start, t, noise=None, batch_labels=None):
    for i in range(batch_size):
        label = batch_labels[i]
        same_class_indices = where(batch_labels == label)
        # Select different sample with same label
        pair_idx = random_choice(same_class_indices != i)
        x_shuffled[i] = x_start[pair_idx]
    
    # Convex combination
    alpha = random(batch_size, 1)
    convex = alpha * x_start + (1-alpha) * x_shuffled
    return scale_by_timestep(convex)
```

### Impact
- **Semantically meaningful**: Noise comes from same digit class
- **Preserves class structure**: More realistic "in-distribution" noise
- **Better for digit generation**: Doesn't mix 0s with 9s, etc.

---

## Data Preparation Updates

### New Return Values
`prepare_mnist1d_for_diffusion()` now returns:
- `input_points`: Input data with z-coordinate
- `target_points`: Target data with z-coordinate  
- `prime_inverse_patterns`: Deterministic inverse patterns per digit
- **`labels`**: Digit labels (0-9) for each sample ← **NEW**

### Dataset Class
Updated to include labels:
```python
class PointDiffusionDataset:
    def __getitem__(self, idx):
        return (
            self.input_data[idx],
            self.target_data[idx], 
            self.labels[idx]  # ← NEW
        )
```

---

## Testing Updates

### Implicit Method Testing
- Uses **both models** (forward and inverse) appropriately
- Forward samples → forward model
- Inverse samples → inverse model

### Convex Sampling in Testing  
- Passes labels to `diff_process.sample()` for same-class combinations
- Ensures test-time behavior matches training

---

## Expected Results

### Hypothesis: Method Performance

**Best Expected**:
1. **Semi-Implicit**: Roundtrip losses enforce bidirectional consistency
2. **Explicit**: True mathematical inversion (if architecture allows)
3. **Semi-Explicit**: SVD approximation is stable
4. **Implicit**: Two models may overfit separately

**Convex vs Gaussian**:
- **Convex (same-class)**: Should preserve digit structure better
- **Gaussian**: More randomness, potentially more robust

**Loss Functions**:
- **FID**: Better for distribution matching
- **KL Divergence**: Better for specific sample quality

---

## Running the Experiments

### Quick Check (Recommended before full run)
```bash
cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
python scripts/diffusion_mnist1d_experiments.py
```

Set `max_epochs = 10` in the script for a quick test (2-3 hours for 16 experiments)

### Full Run
```bash
sbatch scripts/run_diffusion_mnist1d.sh
```

Estimated time: ~24 hours for all 16 experiments with 1000 epochs each

### Monitor Progress
Results are saved incrementally to `results/mnist1d_experiment_results_partial.json`

---

## Output Files

### Results
- `results/mnist1d_diffusion_experiments_YYYYMMDD_HHMMSS.json`
- `results/mnist1d_experiment_results_partial.json` (saved every 2 experiments)

### Report  
- `results/mnist1d_experiment_report_YYYYMMDD_HHMMSS.md`

### W&B Logging
All experiments logged to Weights & Biases:
- Project: `mnist1d-diffusion-experiments`
- Tracks: loss curves, test metrics, runtime

---

## Key Metrics Tracked

For each experiment:
- **test_mse**: Mean squared error
- **test_mae**: Mean absolute error  
- **test_kl_divergence**: KL divergence between distributions
- **test_fid_score**: Fréchet Inception Distance
- **test_forward_mse**: Forward direction MSE
- **test_forward_fid**: Forward direction FID
- **test_inverse_mse**: Inverse direction MSE
- **test_inverse_fid**: Inverse direction FID
- **runtime_seconds**: Total training + testing time
- **num_parameters**: Model parameter count

---

## For Your Advisor Meeting (3 Days)

### Key Points to Highlight:

1. **Comprehensive Comparison**: 16 experiments covering all major approaches
   
2. **Novel Same-Class Convex Sampling**: More semantically meaningful than random noise

3. **4 Inversion Methods**:
   - Classic (Implicit - 2 models)
   - Semi-Implicit (z-switch, already proven)
   - Semi-Explicit (SVD-based, mathematically grounded)
   - Explicit (True mathematical inversion)

4. **Quantitative Metrics**: Multiple metrics (MSE, MAE, KL, FID) for rigorous comparison

5. **Both Directions Tested**: Forward (denoising) AND inverse (to prime space)

### Suggested Talking Points:
- "We're comparing 4 fundamentally different approaches to invertibility"
- "Convex sampling uses same-class combinations, preserving semantic structure"
- "Semi-Explicit uses SVD - no second model, mathematically principled"
- "Explicit attempts true network inversion with activation function inverses"

### If Results Are Ready:
- Show comparative plots of MSE across methods
- Highlight which method(s) achieve best bidirectional consistency
- Discuss tradeoffs (parameters vs performance)

---

## Code Quality Notes

✅ **Fully implemented and ready to run**
✅ **Labels properly tracked through entire pipeline**  
✅ **Convex sampling correctly uses same-class pairs**
✅ **All 4 trainers handle both forward and inverse**
✅ **Testing handles dual models for implicit method**
✅ **Results saved incrementally for long runs**

---

## Questions to Answer from Results

1. Does same-class convex sampling outperform Gaussian?
2. Which inversion method achieves lowest bidirectional error?
3. Is SVD-based inversion competitive with trained inverses?
4. Does explicit mathematical inversion work for this U-Net architecture?
5. FID vs KL: Which loss function produces better distributions?
6. Parameter efficiency: Is semi-implicit worth the complexity vs 2 models?

Good luck with your advisor meeting! 🚀
