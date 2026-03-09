# Synthetic IMS Generation - Essential Workflow

This repository contains code for generating synthetic Ion Mobility Spectrometry (IMS) data using a diffusion model trained in latent space. The approach uses separation-guided diffusion to preserve the non-Gaussian manifold structure of chemical latent spaces.

## Quick Start

If you just want to test the generation and visualization (training already complete):

```bash
# 1. Setup required files (one-time only)
cp /scratch/kjmetzler/diffusion_essentials/name_smiles_embedding_file.csv Data/
cp /scratch/kjmetzler/diffusion_essentials/test_data.feather Data/
cp /scratch/kjmetzler/diffusion_essentials/train_data.feather Data/
cp /scratch/kjmetzler/diffusion_essentials/diffusion_latent_normalized_best.pt models/
cp /scratch/kjmetzler/diffusion_essentials/autoencoder_separated.pth models/

# 2. Generate synthetic samples (~1-2 min on GPU)
sbatch scripts/run_gen_decode_full.sh

# 3. Create PCA visualizations (~1 min)
sbatch scripts/run_final_pca.sh

# 4. Check results
ls -lh results/full_generated_*.npy
ls -lh images/pca_real_vs_diffusion*.png
```

## Overview

The pipeline consists of three main stages:
1. **Training**: Train a diffusion model in the latent space of IMS spectra
2. **Generation**: Generate synthetic IMS samples using the trained diffusion model
3. **Evaluation**: Evaluate the quality of generated samples using PCA and classification metrics

## Prerequisites

### Python Packages
- torch
- numpy
- pandas
- matplotlib
- scikit-learn
- wandb (for training - optional for generation)
- seaborn

### Required Data Files
All data files should be in the `Data/` directory. These are available in `/scratch/kjmetzler/diffusion_essentials/`:
- `test_data.feather`: Test IMS spectra (354 MB, 74,173 samples)
- `train_data.feather`: Training IMS spectra (1.1 GB, 222,519 samples)
- `name_smiles_embedding_file.csv`: Pre-computed ChemNet SMILES embeddings for 8 chemicals (53 KB)

### Required Model Files
Models should be in the `models/` directory. These are available in `/scratch/kjmetzler/diffusion_essentials/`:
- `diffusion_latent_normalized_best.pt`: Pre-trained diffusion model (11 MB)
- `autoencoder_separated.pth`: Pre-trained decoder for IMS spectra (90 MB)

## Workflow

### Step 1: Train the Diffusion Model

Train a class-conditioned diffusion model on normalized latent representations:

```bash
sbatch scripts/run_train_latent_diffusion.sh
```

Or directly with Python:
```bash
python3 scripts/train_normalized_diffusion.py --beta_end 0.02
```

**What it does:**
- Loads original latent representations of IMS spectra
- Normalizes latents to zero mean, unit variance
- Trains class-conditioned diffusion model with separation loss
- Uses linear beta schedule: [0.001, 0.02]
- Saves best model to `models/diffusion_latent_normalized_best.pt`
- Logs metrics to Weights & Biases

**Key features:**
- **Separation-guided loss**: Combines noise prediction (80%) with inter-class separation loss (20%, margin=5.0)
- **Prevents mode collapse**: Maintains chemical distinctness during generation
- **Preserves manifold structure**: Designed for non-Gaussian latent spaces

**Hyperparameters:**
- Timesteps: 50
- Learning rate: 5e-5 with cosine annealing
- Batch size: 256
- Early stopping: 100 epochs patience

**Expected runtime:** ~6-12 hours on GPU
**Output:** `models/diffusion_latent_normalized_best.pt` (includes data_mean and data_std for denormalization)

### Step 2: Generate Synthetic IMS Samples

Generate synthetic spectra using the trained diffusion model:

```bash
sbatch scripts/run_gen_decode_full.sh
```

Or directly:
```bash
python3 scripts/generate_and_decode_full.py
```

**What it does:**
- Loads the trained diffusion model (`diffusion_latent_normalized_best.pt`)
- Generates 500 synthetic latent samples per chemical (8 chemicals = 4000 total)
- Uses DDIM sampling (100 steps) for fast, high-quality generation
- Decodes latent samples to IMS spectra using pre-trained decoder
- Saves outputs to `results/` directory

**Expected runtime:** ~1-2 minutes on GPU

**Outputs:**
- `results/full_generated_latents.npy`: Synthetic latent codes (4000 × 512)
- `results/full_generated_spectra.npy`: Decoded IMS spectra (4000 × 1676)
- `results/full_generated_labels.npy`: Chemical labels (4000,)

**Note:** The model uses cosine beta schedule (timesteps=1000) with denormalization for `diffusion_latent_normalized_best.pt`.

### Controlling Generation Spread

You can control the spread of generated ChemNet points by modifying the sampling distribution variance. By default, sampling starts from N(0,1), but you can increase variance for more diverse samples:

**In `generate_and_decode_full.py`, modify the sampling line:**
```python
# Default: Standard normal
x_t = torch.randn(n_samples, 512, device=device)

# More spread: Sample from N(0, 1.5^2)
x_t = torch.randn(n_samples, 512, device=device) * 1.5

# Even more spread: Sample from N(0, 2^2)
x_t = torch.randn(n_samples, 512, device=device) * 2.0
```

**Effect on generation:**
- **\sigma = 1.0** (default): Standard spread matching training distribution
- **\sigma = 1.5**: ~50% more spread, explores latent space boundaries
- **\sigma = 2.0**: 2× spread, maximum diversity but may leave training distribution

**Recommendation:** Start with \sigma=1.0, increase to 1.5 if you need more chemical diversity in generated samples.

### Step 3: Visualize Results with PCA

Create PCA plots comparing real vs generated samples:

```bash
sbatch scripts/run_final_pca.sh
```

Or directly:
```bash
python3 scripts/gen_pca_final.py
```

**What it does:**
- Loads real and generated latent representations
- Computes PCA projections
- Creates comparison plots showing:
  - Real vs Diffusion generated samples
  - Real vs Gaussian generated samples
  - Chemical separation in latent space

**Expected runtime:** ~1-2 minutes

**Outputs:**
- `images/pca_real_vs_diffusion_vs_gaussian_by_chemical.png`
- `images/pca_real_vs_diffusion_by_chemical.png`

### Additional Visualizations

**Compare diffusion vs Gaussian sampling:**
```bash
sbatch scripts/run_compare.sh
```

**Diffusion trajectory visualization:**
Visualize how samples transform through the forward diffusion process:
```bash
python3 scripts/visualize_diffusion_trajectory.py --chemical DEB --beta_end 0.02
```

**Full latent space PCA:**
```bash
python3 scripts/visualize_full_latent_pca.py
```

**Per-chemical PCA:**
```bash
python3 scripts/create_per_chemical_pca.py
```

## Testing the Workflow

### Pipeline Verification

To verify the workflow is working:

1. **Check dependencies**: Ensure all data files exist (especially `name_smiles_embedding_file.csv`)
2. **Test generation**: Run `sbatch scripts/run_gen_decode_full.sh` to generate samples
3. **Test visualization**: Run `sbatch scripts/run_final_pca.sh` to create PCA plots

## Directory Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python package requirements
├── Data/                          # Input data files (copy from /scratch)
│   ├── test_data.feather
│   ├── train_data.feather
│   └── name_smiles_embedding_file.csv
├── models/                        # Trained models (copy from /scratch)
│   ├── diffusion_latent_normalized_best.pt
│   └── autoencoder_separated.pth
├── results/                       # Generated outputs (.gitignored)
│   ├── full_generated_latents.npy
│   ├── full_generated_spectra.npy
│   ├── full_generated_labels.npy
│   ├── autoencoder_train_latent.npy           # Pre-computed latents
│   ├── autoencoder_test_latent.npy
│   └── autoencoder_*_latent_separated.npy     # Separation-enhanced latents
├── images/                        # Visualization outputs (.gitignored)
│   ├── pca_real_vs_diffusion_vs_gaussian_by_chemical.png
│   ├── pca_real_vs_diffusion_by_chemical.png
│   ├── beta_ablation_comparison.png
│   └── deb_latent_pca_beta_comparison.png
├── scripts/                       # Python and SLURM scripts
│   ├── train_normalized_diffusion.py          # Main training script
│   ├── generate_and_decode_full.py            # Generation script
│   ├── gen_pca_final.py                       # Main PCA visualization
│   ├── compare_diffusion_gaussian.py          # Diffusion vs Gaussian comparison
│   ├── visualize_diffusion_trajectory.py      # Trajectory visualization
│   └── run_*.sh                               # SLURM batch scripts
├── logs/                          # SLURM output logs (.gitignored)
└── LaTeX/                         # Paper drafts and bibliography
```

## File Sizes

**Essential files (must copy from /scratch):**
- Data files: ~1.5 GB total
  - `test_data.feather`: 354 MB
  - `train_data.feather`: 1.1 GB (only needed for training)
  - `name_smiles_embedding_file.csv`: 53 KB
- Model files: ~101 MB total
  - `diffusion_latent_normalized_best.pt`: 11 MB
  - `autoencoder_separated.pth`: 90 MB

**Generated outputs (created by scripts, not in git):**
- `results/*.npy`: ~650 MB for 4000 generated samples
- `images/*.png`: ~5-10 MB total

## Troubleshooting

### Issue: "Data files not found"
**Solution:** Copy required files from `/scratch/kjmetzler/diffusion_essentials/`:
```bash
# Copy all required files at once
cp /scratch/kjmetzler/diffusion_essentials/name_smiles_embedding_file.csv Data/
cp /scratch/kjmetzler/diffusion_essentials/test_data.feather Data/
cp /scratch/kjmetzler/diffusion_essentials/train_data.feather Data/  # Only needed for training
cp /scratch/kjmetzler/diffusion_essentials/diffusion_latent_normalized_best.pt models/
cp /scratch/kjmetzler/diffusion_essentials/autoencoder_separated.pth models/
```

### Issue: "Model file not found"
**Solution:** 
- For pre-trained model: Copy from `/scratch/kjmetzler/diffusion_essentials/`
- For custom training: Train your own model using Step 1

### Issue: "Out of memory on GPU"
**Solution:** Reduce batch size in training scripts (default: 256)
```python
BATCH_SIZE = 128  # Or lower
```

### Issue: "Wandb login required"
**Solution:** For training, either:
- Set your wandb API key in the script
- Or disable wandb: `export WANDB_MODE=disabled`
- For generation only: Wandb is not required

### Issue: "Generated samples have wrong scale"
**Solution:** The pre-trained model includes normalization statistics (data_mean, data_std) and should produce correctly scaled outputs. If samples appear too concentrated or dispersed, try adjusting the sampling variance (see "Controlling Generation Spread" section).

### Issue: "Generated samples lack diversity"
**Solution:** Increase the sampling distribution variance:
```python
# In generate_and_decode_full.py
x_t = torch.randn(n_samples, 512, device=device) * 1.5  # or 2.0
```

## Key Parameters

### Diffusion Training (`train_normalized_diffusion.py`)
- `LATENT_DIM = 512`: Dimension of latent space (ChemNet embedding size)
- `NUM_CLASSES = 8`: Number of chemical classes
- `TIMESTEPS = 50`: Number of diffusion training steps
- `BATCH_SIZE = 256`: Training batch size
- `LEARNING_RATE = 5e-5`: Learning rate with cosine annealing
- `MAX_EPOCHS = 1000`: Maximum training epochs
- `PATIENCE = 100`: Early stopping patience
- `BETA_START = 0.001`: Beta schedule start
- `BETA_END = 0.02`: Beta schedule end
- `NOISE_WEIGHT = 0.8`: Weight for noise prediction loss
- `SEPARATION_WEIGHT = 0.2`: Weight for inter-class separation loss
- `SEPARATION_MARGIN = 5.0`: Minimum distance between class centroids

### Generation (`generate_and_decode_full.py`)
- `samples_per_class = 500`: Number of samples to generate per chemical
- `ddim_steps = 100`: Number of sampling steps (DDIM acceleration)
- `timesteps = 1000`: Total diffusion timesteps (cosine schedule)
- Sampling distribution: N(0,1) by default; multiply by 1.5 or 2.0 for increased diversity

## Available Pre-trained Models

Located in `/scratch/kjmetzler/diffusion_essentials/`:

### Diffusion Model
- **`diffusion_latent_normalized_best.pt`** (11 MB)
  - Training: Normalized latents with separation-guided loss
  - Beta schedule: Linear [0.001, 0.02], 50 timesteps
  - Date: February 4, 2026
  - Epoch: 857, Loss: 0.047
  - Includes: `data_mean` and `data_std` for denormalization
  - Features: 80% noise prediction + 20% separation loss

### Decoder Model
- **`autoencoder_separated.pth`** (90 MB)
  - 9-layer generator: 512-dim latent → 1676-dim IMS spectrum
  - Date: January 28, 2026
  - Architecture: Flexible N-layer with LeakyReLU activations

### Data Files
- **`name_smiles_embedding_file.csv`** (53 KB): Pre-computed ChemNet embeddings
- **`test_data.feather`** (354 MB): 74,173 test IMS spectra
- **`train_data.feather`** (1.1 GB): 222,519 training IMS spectra

## Chemical Classes

The model generates samples for 8 chemical compounds:
1. **DEB** - Diethylene glycol dibutyl ether
2. **DEM** - Diethylene glycol diethyl ether  
3. **DMMP** - Dimethyl methylphosphonate
4. **DPM** - Oxybispropanol (Dipropylene glycol)
5. **DtBP** - Di-tert-butyl peroxide
6. **JP8** - Jet fuel (complex mixture)
7. **MES** - 2-(N-morpholino)ethanesulfonic acid
8. **TEPO** - Triethyl phosphate

## Methodology

### Separation-Guided Diffusion

This work introduces a novel approach to generating synthetic data for non-Gaussian chemical latent spaces:

1. **Problem**: Standard diffusion models assume Gaussian priors, which can cause structural collapse when generating from complex, non-Gaussian manifolds (e.g., chemical latent spaces).

2. **Solution**: Separation-guided diffusion combines:
   - **Noise prediction loss** (80%): Standard denoising objective
   - **Separation loss** (20%): Contrastive loss with margin=5.0 that encourages different chemical classes to maintain minimum inter-centroid distances

3. **Benefits**:
   - Preserves non-Gaussian structure of chemical manifolds
   - Maintains chemical distinctness during generation
   - Prevents mode collapse across chemical classes

### Architecture

- **Encoder**: Fixed ChemNet embeddings (512-D) pre-computed from SMILES strings
- **Diffusion Model**: Class-conditioned transformer-like architecture
  - Input: Noisy latent (512-D) + timestep + SMILES embedding (512-D) + class one-hot (8-D)
  - Hidden: 6 layers × 512 dimensions
  - Output: Predicted noise (512-D)
- **Decoder**: 9-layer generator with LeakyReLU: 512-D latent → 1676-D IMS spectrum

### Training Details

- **Diffusion schedule**: Linear beta schedule [0.001, 0.02]
- **Training steps**: Up to 1000 epochs with early stopping (patience=100)
- **Optimization**: AdamW with cosine annealing (lr=5e-5, weight_decay=0.01)
- **Batch size**: 256 samples
- **Data normalization**: Zero mean, unit variance
- **Dataset**: 222,519 training samples across 8 chemicals
- **Loss function**: 80% noise prediction MSE + 20% separation loss (margin=5.0)

## Citations

This code builds on work by Cate Dunham for IMS spectrum generation. The decoder architecture is adapted from her ChemicalDataGeneration repository.

If you use this code, please cite:
- Metzler, K.J. et al. "Separation-Guided Diffusion for Non-Gaussian Manifolds" (in preparation)

## Contact

For questions about this code:
- Author: Kevin Metzler
- Email: kjmetzler@wpi.edu
- Institution: Worcester Polytechnic Institute

---

## Pipeline Verification Status

**Last Updated**: March 9, 2026

### Verified Working
- **Setup**: File copying from `/scratch/kjmetzler/diffusion_essentials/` tested
- **Training**: 
  - Training script (`train_normalized_diffusion.py`) - architecture verified
  - Full training run (6-12 hours) - not re-tested after code cleanup
- **Generation**: Successfully generates 4000 samples in ~1-2 minutes on GPU
  - Outputs: `results/full_generated_*.npy` files created correctly
  - Variable sampling: Tested with \sigma \in {1.0, 1.5, 2.0}
- **Visualization**: PCA plots generated successfully in ~1 minute
  - Outputs: `images/pca_real_vs_diffusion*.png` files created correctly
- **Quality**: Generated samples maintain chemical distinctness and realistic IMS spectra structure

### Setup Checklist for New Users

1. Clone this repository
2. Copy required files from `/scratch/kjmetzler/diffusion_essentials/`:
   ```bash
   cp /scratch/kjmetzler/diffusion_essentials/name_smiles_embedding_file.csv Data/
   cp /scratch/kjmetzler/diffusion_essentials/test_data.feather Data/
   cp /scratch/kjmetzler/diffusion_essentials/diffusion_latent_normalized_best.pt models/
   cp /scratch/kjmetzler/diffusion_essentials/autoencoder_separated.pth models/
   # Only if training:
   cp /scratch/kjmetzler/diffusion_essentials/train_data.feather Data/
   ```
3. Install Python packages: `pip install -r requirements.txt`
4. Test generation: `sbatch scripts/run_gen_decode_full.sh`
5. Test visualization: `sbatch scripts/run_final_pca.sh`

### Recommended Workflow for Reproducibility

**For using pre-trained model:**
1. Setup files from `/scratch` (steps 1-2 above)
2. Run generation (Step 2 in workflow)
3. Create visualizations (Step 3 in workflow)

**For training from scratch:**
1. Setup all files including `train_data.feather`
2. Configure Weights & Biases (or disable it)
3. Run training with desired beta schedule
4. Generate and visualize results

### Known Issues
- **Wandb authentication**: Required for training unless disabled with `export WANDB_MODE=disabled`
- **GPU memory**: Batch size may need reduction on GPUs with <16GB VRAM
- **File paths**: Scripts assume repository root as working directory
