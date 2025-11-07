#!/bin/bash

# ============================================================================
# Complete Baseline Experiment Runner
# Days 24-28: Run all baseline experiments
# ============================================================================

set -e

echo "======================================================================"
echo "AER System - Complete Baseline Experiments"
echo "======================================================================"

# Configuration
MODEL=${MODEL:-"mock"}
DATA_DIR=${DATA_DIR:-"data/dev"}
OUTPUT_DIR="outputs/results/baseline_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Data: $DATA_DIR"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run experiments
STRATEGIES=("zero_shot" "cot" "few_shot")

for STRATEGY in "${STRATEGIES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Running: $STRATEGY"
    echo "======================================================================"
    
    python experiments/baseline/run_baseline.py \
        --model "$MODEL" \
        --strategy "$STRATEGY" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✓ $STRATEGY completed successfully"
    else
        echo "✗ $STRATEGY failed"
        exit 1
    fi
done

echo ""
echo "======================================================================"
echo "✅ All baseline experiments completed!"
echo "======================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  ls -lh $OUTPUT_DIR"
echo "  cat $OUTPUT_DIR/baseline_*.json"