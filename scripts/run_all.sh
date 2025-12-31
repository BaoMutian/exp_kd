#!/bin/bash
# Run all three KD methods for comparison
# Usage: bash scripts/run_all.sh

set -e

echo "=========================================="
echo "Running Knowledge Distillation Experiments"
echo "=========================================="

# Run SeqKD
echo ""
echo "=========================================="
echo "Step 1/3: Running SeqKD..."
echo "=========================================="
bash scripts/run_seqkd.sh

# Run SKD
echo ""
echo "=========================================="
echo "Step 2/3: Running SKD..."
echo "=========================================="
bash scripts/run_skd.sh

# Run GKD
echo ""
echo "=========================================="
echo "Step 3/3: Running GKD..."
echo "=========================================="
bash scripts/run_gkd.sh

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved to:"
echo "  - SeqKD: outputs/seqkd/"
echo "  - SKD:   outputs/skd/"
echo "  - GKD:   outputs/gkd/"

