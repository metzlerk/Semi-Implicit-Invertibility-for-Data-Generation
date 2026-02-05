#!/bin/bash
# Monitor the current diffusion job progress

JOB_ID=${1:-1769879}
LOG_DIR="logs"
OUT_FILE="$LOG_DIR/ims_diffusion_autoencoder_per_class_${JOB_ID}.out"
ERR_FILE="$LOG_DIR/ims_diffusion_autoencoder_per_class_${JOB_ID}.err"

echo "=========================================="
echo "Job Monitoring for Job ID: $JOB_ID"
echo "=========================================="
echo ""

# Check job status
echo "SLURM Job Status:"
squeue -j $JOB_ID 2>/dev/null || echo "Job completed or not found"
echo ""

# Check progress indicators
echo "Progress:"
if [ -f "$OUT_FILE" ]; then
    # Count completed classes
    completed_classes=$(grep -c "Generation MSE:" "$OUT_FILE")
    echo "  Completed diffusion models: $completed_classes/8"
    
    # Check if visualizations started
    if grep -q "GENERATING GAUSSIAN BASELINE" "$OUT_FILE"; then
        echo "  ✓ Gaussian baseline generation started"
    fi
    
    if grep -q "GENERATING VISUALIZATIONS" "$OUT_FILE"; then
        echo "  ✓ Visualization generation started"
    fi
    
    if grep -q "CLASSIFIER EVALUATION" "$OUT_FILE"; then
        echo "  ✓ Classifier evaluation started"
    fi
    
    # Show last loss from each completed class
    echo ""
    echo "Latest training loss by class:"
    grep "Epoch 1000" "$OUT_FILE" | tail -5 | while read line; do
        echo "    $line"
    done
fi

echo ""
echo "Recent error/warning messages:"
if [ -f "$ERR_FILE" ]; then
    # Show only non-pydantic warnings
    grep -v "pydantic" "$ERR_FILE" | grep -E "Error|error|Traceback|Exception" | tail -5 || echo "  No errors found"
fi

echo ""
echo "Last 20 output lines:"
echo "---"
if [ -f "$OUT_FILE" ]; then
    tail -20 "$OUT_FILE"
fi
