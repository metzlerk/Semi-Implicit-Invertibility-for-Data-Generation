#!/usr/bin/env python3
"""
Validation script to check setup before running improved diffusion training
"""

import os
import sys
import torch
import numpy as np
import pandas as pd

def check_environment():
    """Check Python environment and packages"""
    print("="*60)
    print("ENVIRONMENT CHECKS")
    print("="*60)
    
    checks = {
        "Python": f"{sys.version.split()[0]}",
        "PyTorch": torch.__version__,
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
        "CUDA Available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        checks["CUDA Device"] = torch.cuda.get_device_name(0)
        checks["CUDA Memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    
    for name, value in checks.items():
        status = "✓" if value not in [None, False] else "✗"
        print(f"  {status} {name}: {value}")
    
    return torch.cuda.is_available()

def check_data_files():
    """Check if required data files exist"""
    print("\n" + "="*60)
    print("DATA FILES CHECK")
    print("="*60)
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'Data')
    
    files_to_check = [
        ('train_data.feather', 'Training data'),
        ('test_data.feather', 'Test data'),
        ('name_smiles_embedding_file.csv', 'SMILE embeddings'),
    ]
    
    all_exist = True
    for filename, description in files_to_check:
        filepath = os.path.join(data_dir, filename)
        exists = os.path.exists(filepath)
        status = "✓" if exists else "✗"
        
        if exists:
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"  {status} {description:30s} {filename:40s} ({size_mb:.1f} MB)")
        else:
            print(f"  {status} {description:30s} {filename:40s} (MISSING!)")
            all_exist = False
    
    return all_exist

def check_output_directories():
    """Check if output directories exist"""
    print("\n" + "="*60)
    print("OUTPUT DIRECTORIES CHECK")
    print("="*60)
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    dirs_to_check = [
        ('results', 'Results output'),
        ('images', 'Image output'),
        ('models', 'Model checkpoint'),
    ]
    
    for dirname, description in dirs_to_check:
        dirpath = os.path.join(root_dir, dirname)
        exists = os.path.exists(dirpath)
        status = "✓" if exists else "⚠"
        
        if exists:
            print(f"  {status} {description:30s} {dirname:20s} (exists)")
        else:
            print(f"  {status} {description:30s} {dirname:20s} (will be created)")
            try:
                os.makedirs(dirpath, exist_ok=True)
                print(f"     → Created successfully")
            except Exception as e:
                print(f"     → Failed to create: {e}")
                return False
    
    return True

def check_scripts():
    """Check if training and comparison scripts exist"""
    print("\n" + "="*60)
    print("SCRIPT FILES CHECK")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    scripts_to_check = [
        ('diffusion_ims_improved_separation.py', 'Training script'),
        ('generate_pca_comparison.py', 'PCA comparison script'),
        ('run_improved_diffusion_pipeline.sh', 'Pipeline runner'),
    ]
    
    all_exist = True
    for filename, description in scripts_to_check:
        filepath = os.path.join(script_dir, filename)
        exists = os.path.exists(filepath)
        status = "✓" if exists else "✗"
        
        if exists:
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  {status} {description:30s} {filename:45s} ({size_kb:.1f} KB)")
        else:
            print(f"  {status} {description:30s} {filename:45s} (MISSING!)")
            all_exist = False
    
    return all_exist

def quick_data_validation():
    """Quick validation of data dimensions and content"""
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root_dir, 'Data')
    
    try:
        # Load train data
        train_df = pd.read_feather(os.path.join(data_dir, 'train_data.feather'))
        test_df = pd.read_feather(os.path.join(data_dir, 'test_data.feather'))
        
        p_cols = [c for c in train_df.columns if c.startswith('p_')]
        n_cols = [c for c in train_df.columns if c.startswith('n_')]
        onehot_cols = ['DEB', 'DEM', 'DMMP', 'DPM', 'DtBP', 'JP8', 'MES', 'TEPO']
        
        print(f"  ✓ Train samples: {len(train_df)}")
        print(f"  ✓ Test samples: {len(test_df)}")
        print(f"  ✓ Positive features (p_): {len(p_cols)}")
        print(f"  ✓ Negative features (n_): {len(n_cols)}")
        print(f"  ✓ Total IMS features: {len(p_cols) + len(n_cols)}")
        print(f"  ✓ Chemical classes: {len(onehot_cols)} {onehot_cols}")
        
        # Load SMILE embeddings
        smile_df = pd.read_csv(os.path.join(data_dir, 'name_smiles_embedding_file.csv'))
        non_na_count = smile_df['embedding'].notna().sum()
        print(f"  ✓ SMILE embeddings: {non_na_count} entries")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return False

def print_summary(checks_passed):
    """Print summary and recommendations"""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Check critical items (exclude CUDA)
    critical_checks = {k: v for k, v in checks_passed.items() if k != 'environment'}
    critical_passed = all(critical_checks.values())
    
    if critical_passed:
        cuda_warning = ""
        if not checks_passed['environment']:
            cuda_warning = "\n⚠️  WARNING: CUDA not available. Training will be SLOW on CPU (expect 10-20+ hours).\n"
        
        print(f"""{cuda_warning}
✓ All critical checks passed! Ready to train.

Next steps:
  1. Review QUICK_START_IMPROVED_DIFFUSION.md for training overview
  2. Run: cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation
  3. Execute: bash scripts/run_improved_diffusion_pipeline.sh
  
Expected runtime: 3-4 hours (GPU), 10-20+ hours (CPU)

To monitor training:
  - Check W&B project: ims-diffusion-separation
  - Or check loss plots: images/diffusion_separation_training_loss.png
""")
    else:
        print("""
✗ Some critical checks failed. Please review above and fix issues before training.

Common issues:
  - Missing data files: Check Data/ directory
  - Missing scripts: Run this validation script from scripts/ directory
  - Output directories can't be created: Check write permissions
  
Contact: Check README or IMPROVED_DIFFUSION_GUIDE.md for details
""")
    
    return critical_passed

if __name__ == '__main__':
    print("\n" + "🔍 IMPROVED DIFFUSION TRAINING - SETUP VALIDATION\n")
    
    checks_passed = {}
    
    # Run all checks
    checks_passed['environment'] = check_environment()
    checks_passed['data'] = check_data_files()
    checks_passed['directories'] = check_output_directories()
    checks_passed['scripts'] = check_scripts()
    checks_passed['data_valid'] = quick_data_validation()
    
    # Print summary
    ready = print_summary(checks_passed)
    
    sys.exit(0 if ready else 1)
