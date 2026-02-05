#!/bin/bash
# Master script to run the complete improved diffusion pipeline
# Usage: bash run_improved_diffusion_pipeline.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "Improved Diffusion Training Pipeline"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if Python environment is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found"
    exit 1
fi

echo "Step 1: Training improved diffusion model..."
echo "=================================================="
python "$SCRIPT_DIR/diffusion_ims_improved_separation.py"

if [ $? -ne 0 ]; then
    echo "ERROR: Diffusion training failed"
    exit 1
fi

echo ""
echo "Step 2: Generating PCA comparison plots..."
echo "=================================================="
python "$SCRIPT_DIR/generate_pca_comparison.py"

if [ $? -ne 0 ]; then
    echo "ERROR: PCA comparison generation failed"
    exit 1
fi

echo ""
echo "=================================================="
echo "Pipeline Complete!"
echo "=================================================="
echo ""
echo "Output files created:"
echo "  - images/diffusion_separation_training_loss.png"
echo "  - images/pca_separation_comparison_improved.png"
echo "  - results/diffusion_separation_training_results.json"
echo "  - results/separation_metrics.json"
echo ""
echo "Models saved:"
echo "  - models/diffusion_separation_best.pt"
echo "  - models/diffusion_separation_final.pt"
echo ""
