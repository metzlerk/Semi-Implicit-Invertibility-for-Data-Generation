# Synthetic IMS Generation - Essential Workflow

This branch contains the essential code for generating synthetic Ion Mobility Spectrometry (IMS) data using a diffusion model trained in latent space.

## Quick Start

If you just want to test the generation and visualization (training already complete):

```bash
# 1. Setup missing data file (one-time only)
cp "/home/kjmetzler/ChemicalDataGeneration/models/Variable Carls/name_smiles_embedding_file.csv" Data/

# 2. Generate synthetic samples (30-60 min on GPU)
sbatch scripts/run_gen_decode_full.sh

# 3. Create PCA visualizations (1-2 min)
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

### External Dependencies
- **Cate's ChemicalDataGeneration Repository**: Required for decoder models
  - Location: `/home/kjmetzler/ChemicalDataGeneration/models`
  - Contains pre-trained decoder for IMS spectra

### Python Packages
- torch
- numpy
- pandas
- matplotlib
- scikit-learn
- wandb (for logging)
- seaborn

### Required Data
All data files should be in the `Data/` directory:
- `test_data.feather`: Test IMS spectra
- `train_data.feather`: Training IMS spectra (optional for generation, needed for training)
- `name_smiles_embedding_file.csv`: SMILE embeddings for each chemical
  - **Note**: This file must be copied from ChemicalDataGeneration repo:
    ```bash
    cp "/home/kjmetzler/ChemicalDataGeneration/models/Variable Carls/name_smiles_embedding_file.csv" Data/
    ```

### Required Models
Models should be in the `models/` directory:
- `diffusion_latent_normalized_best.pt`: Trained diffusion model (from Step 1)
- `autoencoder_separated.pth`: Decoder model (from Cate's repo)
- Encoder: `/scratch/cmdunham/trained_models/spectrum/current_best_models/nine_layer_ims_to_chemnet_encoder.pth`
- Decoder: `/scratch/cmdunham/trained_models/spectrum/current_best_models/nine_layer_ChemNet_to_ims_generator_from_nine_layer__encoder.pth`

## Workflow

### Step 1: Train the Diffusion Model

Train a diffusion model in the latent space of IMS spectra:

```bash
sbatch scripts/run_train_latent_diffusion.sh
```

Or directly with Python:
```bash
python3 scripts/train_latent_diffusion.py
```

**What it does:**
- Loads pre-computed latent representations of IMS spectra
- Trains a class-conditioned diffusion model
- Saves the best model to `models/diffusion_latent_normalized_best.pt`
- Logs metrics to Weights & Biases

**Expected runtime:** ~6-12 hours on GPU
**Output:** `models/diffusion_latent_normalized_best.pt`

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
- Loads the trained diffusion model
- Generates 500 synthetic latent samples per chemical (8 chemicals = 4000 total)
- Decodes samples to IMS spectra using the pre-trained decoder
- Saves outputs to `results/` directory

**Expected runtime:** ~30-60 minutes on GPU

**Outputs:**
- `results/full_generated_latents.npy` (4000 samples × 512 latent dimensions)
- `results/full_generated_spectra.npy` (4000 samples × spectrum length)
- `results/full_generated_labels.npy` (4000 labels)

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

**Full latent space PCA:**
```bash
python3 scripts/visualize_full_latent_pca.py
```

**Per-chemical PCA:**
```bash
python3 scripts/create_per_chemical_pca.py
```

**Compare diffusion vs Gaussian sampling:**
```bash
sbatch scripts/run_compare.sh
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
├── Data/                          # Input data files
│   ├── test_data.feather
│   ├── train_data.feather
│   └── name_smiles_embedding_file.csv
├── models/                        # Trained diffusion models
│   ├── diffusion_latent_normalized_best.pt
│   └── diffusion_latent_separated_best.pt
├── results/                       # Generated outputs
│   ├── full_generated_latents.npy
│   ├── full_generated_spectra.npy
│   └── full_generated_labels.npy
├── images/                        # Visualization outputs
│   ├── pca_real_vs_diffusion_vs_gaussian_by_chemical.png
│   └── pca_real_vs_diffusion_by_chemical.png
├── scripts/                       # Python and SLURM scripts
│   ├── train_latent_diffusion.py          # Training script
│   ├── generate_and_decode_full.py        # Generation script
│   ├── gen_pca_final.py                   # Main PCA visualization
│   ├── compare_diffusion_gaussian.py      # Comparison visualization
│   ├── visualize_full_latent_pca.py       # Full latent PCA
│   ├── create_per_chemical_pca.py         # Per-chemical PCA
│   └── run_*.sh                           # SLURM batch scripts
└── logs/                          # SLURM output logs

```

## Troubleshooting

### Issue: "Cannot find ChemicalDataGeneration models"
**Solution:** Make sure the ChemicalDataGeneration repo is cloned and the path in the scripts is correct:
```python
sys.path.insert(0, '/home/kjmetzler/ChemicalDataGeneration/models')
```

### Issue: "Model file not found"
**Solution:** 
- For diffusion model: Train it first using Step 1
- For encoder/decoder: Check paths in `/scratch/cmdunham/trained_models/`

### Issue: "Out of memory on GPU"
**Solution:** Reduce `BATCH_SIZE` in `train_latent_diffusion.py` (default: 256)

### Issue: "Wandb login required"
**Solution:** Either:
- Set your wandb API key in the script
- Or disable wandb: `export WANDB_MODE=disabled`

## Key Parameters

### Diffusion Training (`train_latent_diffusion.py`)
- `LATENT_DIM = 512`: Dimension of latent space
- `NUM_CLASSES = 8`: Number of chemical classes
- `TIMESTEPS = 50`: Number of diffusion steps
- `BATCH_SIZE = 256`: Training batch size
- `LEARNING_RATE = 5e-5`: Learning rate
- `MAX_EPOCHS = 1000`: Maximum training epochs

### Generation (`generate_and_decode_full.py`)
- `samples_per_class = 500`: Number of samples to generate per chemical

## Citations

This code uses pre-trained encoder/decoder models from Cate Dunham's ChemicalDataGeneration repository.

## Contact

For questions about this code, contact: kjmetzler@wpi.edu

---

## Pipeline Verification Status

**Last Tested**: February 25, 2026

### ✓ Verified Working (Post-Cleanup)
- **Data Loading**: All required data files present and accessible
- **External Dependencies**: ChemicalDataGeneration repo and decoder models accessible
- **Step 1 (Training)**: Script syntax verified (not run due to 6-12 hour runtime)
- **Step 2 (Generation)**: ✅ Successfully tested - generates 4000 samples in ~8 seconds
  - Outputs: `results/full_generated_latents.npy`, `results/full_generated_spectra.npy`, `results/full_generated_labels.npy`
- **Step 3 (PCA Visualization)**: ✅ Successfully tested - creates plots in ~1 minute
  - Output: `images/pca_real_vs_diffusion_vs_gaussian_by_chemical.png`
  - Output: `images/pca_real_vs_diffusion_by_chemical.png`

### 📋 Required Setup
1. Copy SMILES embedding file from ChemicalDataGeneration repo (see Quick Start above)
2. Ensure trained diffusion model exists: `models/diffusion_latent_normalized_best.pt` or `models/diffusion_latent_separated_best.pt`

### 🚀 Recommended Workflow
1. Skip Step 1 (training) if model already exists
2. Run Step 2 (generation) to create fresh synthetic samples
3. Run Step 3 (PCA visualization) to verify quality
4. Use additional visualization scripts for deeper analysis
