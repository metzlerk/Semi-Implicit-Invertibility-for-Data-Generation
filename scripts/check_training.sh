#!/bin/bash
# Monitor training progress for beta ablation study

cd /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation

echo "================================================================"
echo "Beta Ablation Study Training Monitor"
echo "================================================================"
echo ""

# Find the latest training job
JOB_ID=$(squeue -u kjmetzler -n train_diff -h -o "%i" | head -1)

if [ -z "$JOB_ID" ]; then
    echo "No training job currently running."
    echo ""
    
    # Check if model exists
    if [ -f "models/diffusion_latent_separated_beta0.20_best.pt" ]; then
        echo "✓ Training complete! Model found:"
        ls -lh models/diffusion_latent_separated_beta0.20_best.pt
        echo ""
        echo "You can now run the comparison:"
        echo "  sbatch scripts/run_beta_ablation.sh"
    else
        echo "Model not yet available. Check logs for completion:"
        LATEST_LOG=$(ls -t logs/train_diff_*.out 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo "  Latest log: $LATEST_LOG"
            echo ""
            echo "Last 10 lines:"
            tail -10 "$LATEST_LOG"
        fi
    fi
else
    echo "Training job $JOB_ID is running"
    echo ""
    
    # Show job status
    echo "Job status:"
    squeue -u kjmetzler -j $JOB_ID
    echo ""
    
    # Show latest log output
    LOG_FILE="logs/train_diff_${JOB_ID}.out"
    if [ -f "$LOG_FILE" ]; then
        echo "Latest training output (last 20 lines):"
        echo "----------------------------------------"
        tail -20 "$LOG_FILE"
    fi
fi

echo ""
echo "================================================================"
echo "Model Status:"
echo "================================================================"
echo "Existing models:"
ls -lh models/diffusion_latent_separated_beta*.pt 2>/dev/null || echo "  No beta models found yet"
echo ""
echo "================================================================"
