#!/bin/bash
# Quick start script for IMS Spectra Diffusion Training

echo "=========================================="
echo "IMS Spectra Diffusion - Quick Start"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/diffusion_ims_spectra_experiments.py" ]; then
    echo "ERROR: Please run this from the project root directory"
    echo "cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation"
    exit 1
fi

# Create necessary directories
mkdir -p logs results images models

echo "Choose an option:"
echo ""
echo "1) Quick test (5k samples, ~5 min on CPU)"
echo "2) Submit full training to SLURM cluster (222k samples)"
echo "3) Check job status"
echo "4) View latest log"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Starting quick test..."
        echo "This will use 5,000 samples and train for 100 epochs"
        echo ""
        python scripts/diffusion_ims_spectra_experiments.py --test
        ;;
    2)
        echo ""
        echo "Submitting full training job to SLURM..."
        sbatch scripts/run_ims_diffusion.sh
        echo ""
        echo "Job submitted! Check status with:"
        echo "  squeue -u \$USER"
        echo ""
        echo "Watch output with:"
        echo "  tail -f logs/ims_diffusion_*.out"
        ;;
    3)
        echo ""
        echo "Current jobs:"
        squeue -u $USER
        ;;
    4)
        echo ""
        latest_log=$(ls -t logs/ims_diffusion_*.out 2>/dev/null | head -1)
        if [ -n "$latest_log" ]; then
            echo "Showing last 50 lines of: $latest_log"
            echo "----------------------------------------"
            tail -50 "$latest_log"
        else
            echo "No log files found in logs/"
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
