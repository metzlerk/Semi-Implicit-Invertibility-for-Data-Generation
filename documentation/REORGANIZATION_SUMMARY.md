# Directory Reorganization Summary

This document describes the reorganization performed on October 27, 2025 and the changes made to fix file dependencies.

## New Directory Structure

```
Semi-Implicit-Invertibility-for-Data-Generation/
├── Data/                          # Original data directory (unchanged)
├── documentation/                 # All markdown documentation
├── images/                        # All PNG images and plots
├── notebooks/                     # All Jupyter notebooks
├── results/                       # All experiment results (CSV, JSON)
├── scripts/                       # All Python scripts and shell scripts
├── slurm_output/                  # All SLURM job output files
├── mnist1d_data.pkl              # Data file (kept in root for easy access)
└── .git/, .gitignore             # Git files (unchanged)
```

## Files Moved

### Notebooks (9 files) → `notebooks/`
- 4f-kjm-CHemical-Classification3-1.ipynb
- 4f-kjm-Chemical-Classification.ipynb
- 4f-kjm-Chemical-Classification2O.ipynb
- 4f-kjm-Chemical-Classification3O.ipynb
- 4g-kjm-Diffusion.ipynb
- 4g-kjm-test copy.ipynb
- 4g-kjm-test.ipynb
- 4g-kjm-test_NICE.ipynb
- Graphs.ipynb
- images.ipynb

### Scripts (11 files) → `scripts/`
- 4f-kjm-Chemical-Classification2O.py
- 4f-kjm-Chemical-Classification4O.py
- 4f-kjm-evaluatesynthetic-data.py
- 4f-kjm-masked-linear-trainer.py
- 4f-kjm-noise-generator.py
- 4f-kjm-synthetic-data-generator.py
- Data_prep.py
- diffusion_mnist1d_experiments.py
- run_diffusion_mnist1d.sh
- test_generator.py
- test_setup.py

### Images (7 files) → `images/`
- experiment_comparison_efficiency.png
- experiment_comparison_quality.png
- experiment_heatmap_mse.png
- experiment_training_curves.png
- output.png
- random_forest_accuracy_Cate.png
- random_forest_accuracy_Kevin.png

### Results (18 files) → `results/`
- diffusion_experiments_20251008_131848.json
- diffusion_experiments_20251008_132509.json
- experiment_results_partial.json
- hyperparameter_results_*.csv (14 files)

### Documentation (7 files) → `documentation/`
- COMPLETE_PACKAGE_SUMMARY.md
- CORRECTED_IMPLEMENTATION.md
- EXPERIMENT_GUIDE.md
- experiment_report_20251008_132513.md
- README_EXPERIMENTS.md
- SEMI_IMPLICIT_EXPLANATION.md
- SEMI_IMPLICIT_VISUAL.md
- VERIFICATION_WALKTHROUGH.md

### SLURM Output (9 files) → `slurm_output/`
- slurm-*.out files

## Code Changes to Fix Dependencies

### 1. `scripts/test_setup.py`
**Changes:**
- Added path resolution to find the script directory and project root
- Updated `diffusion_mnist1d_experiments.py` path to use script directory
- Updated `mnist1d_data.pkl` path to use root directory

### 2. `scripts/diffusion_mnist1d_experiments.py`
**Changes:**
- Added imports: `os`, `sys`
- Added path constants:
  ```python
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  ROOT_DIR = os.path.dirname(SCRIPT_DIR)
  DATA_DIR = ROOT_DIR
  RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
  IMAGES_DIR = os.path.join(ROOT_DIR, 'images')
  ```
- Updated `load_mnist1d()` to use `DATA_DIR` for `mnist1d_data.pkl`
- Updated intermediate results save to use `RESULTS_DIR`
- Updated final results save to use `RESULTS_DIR`
- Updated report save to use `RESULTS_DIR`

### 3. `scripts/run_diffusion_mnist1d.sh`
**Changes:**
- Updated working directory from `/home/kjmetzler/` to `/home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation`
- Updated script path from `4f-files/diffusion_mnist1d_experiments.py` to `scripts/diffusion_mnist1d_experiments.py`

### 4. `notebooks/4g-kjm-Diffusion.ipynb`
**Changes:**
- Updated all `plt.savefig()` calls to save to `../images/` directory
- Updated `experiment_results_partial.json` path to `../results/`

### 5. `notebooks/4g-kjm-test*.ipynb` (all test notebooks)
**Changes:**
- Updated all hyperparameter results CSV paths to save to `../results/`

## Files with External Dependencies (No Changes Needed)

The following files reference external paths outside this project and were **not modified**:
- Chemical classification notebooks: Reference `/home/kjmetzler/iterativenn/` and other external paths
- Scripts referencing `/home/kjmetzler/train_data_subset.feather` and similar external data files

## How to Run Scripts and Notebooks

### Running Scripts
Scripts can now be run from either the project root or the scripts directory:

```bash
# From project root
python scripts/diffusion_mnist1d_experiments.py

# From scripts directory
cd scripts
python diffusion_mnist1d_experiments.py
```

### Running Notebooks
Notebooks should be opened from the `notebooks/` directory. They will automatically save outputs to the correct locations:
- Images → `../images/`
- Results → `../results/`

### SLURM Jobs
The updated `run_diffusion_mnist1d.sh` can be submitted as before:
```bash
sbatch scripts/run_diffusion_mnist1d.sh
```

## Benefits of This Organization

1. **Cleaner root directory**: Core files are immediately visible
2. **Easier navigation**: Related files are grouped together
3. **Better version control**: Easier to ignore/track specific file types
4. **Scalability**: Easy to add more files to appropriate directories
5. **Professional structure**: Follows common project organization patterns

## Notes

- The `mnist1d_data.pkl` file was intentionally kept in the root directory for easy access by both scripts and notebooks
- All paths are now relative, making the project more portable
- The `Data/` directory structure was preserved as-is
