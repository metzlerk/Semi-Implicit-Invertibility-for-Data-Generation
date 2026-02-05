#!/bin/bash
# Estimate job completion time and show detailed progress

JOB_ID=${1:-1769879}
LOG_DIR="logs"
OUT_FILE="$LOG_DIR/ims_diffusion_autoencoder_per_class_${JOB_ID}.out"

echo "=========================================="
echo "Detailed Job Progress Analysis"
echo "=========================================="
echo ""

# Extract start time
if [ -f "$OUT_FILE" ]; then
    start_time=$(grep "Start time:" "$OUT_FILE" | head -1 | sed 's/Start time: //')
    echo "Started: $start_time"
    
    # Current time
    current_time=$(date)
    echo "Current: $current_time"
    echo ""
    
    # Count completed classes and extract timing
    echo "Per-Class Results:"
    echo "---"
    grep -E "TRAINING DIFFUSION FOR CLASS|Generation MSE:" "$OUT_FILE" | while read line; do
        if echo "$line" | grep -q "TRAINING DIFFUSION"; then
            class=$(echo "$line" | sed 's/.*CLASS: //' | sed 's/ .*//')
            echo "Class: $class"
        else
            mse=$(echo "$line" | sed 's/.*Generation MSE: //')
            echo "  MSE: $mse"
        fi
    done
    
    echo ""
    echo "Estimated Timeline:"
    echo "---"
    
    # Estimated remaining time based on ~12 mins per class
    completed=$(grep -c "Generation MSE:" "$OUT_FILE")
    remaining=$((8 - completed))
    
    echo "  Completed: $completed/8 classes (~$((completed * 12)) minutes elapsed)"
    echo "  Remaining: $remaining classes (~$((remaining * 12)) minutes estimated)"
    echo ""
    echo "  Estimated completion phases:"
    echo "  - Diffusion training: ~96 minutes total"
    echo "  - Gaussian baseline generation: ~2 minutes"
    echo "  - Visualizations (PCA, spectra): ~5 minutes"
    echo "  - Classifier evaluation (45 configs, 5 runs): ~45 minutes"
    echo "  - Total estimated: ~150 minutes (~2.5 hours from start)"
    echo ""
    
    # Next step
    if grep -q "GENERATING GAUSSIAN BASELINE" "$OUT_FILE"; then
        echo "Current Phase: Gaussian baseline generation"
    elif grep -q "GENERATING VISUALIZATIONS" "$OUT_FILE"; then
        echo "Current Phase: Creating visualizations"
    elif grep -q "CLASSIFIER EVALUATION" "$OUT_FILE"; then
        echo "Current Phase: Classifier evaluation (longest phase)"
    else
        echo "Current Phase: Diffusion model training (Phase 1/$((8 - completed)) complete)"
    fi
fi
