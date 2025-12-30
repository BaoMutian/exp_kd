#!/bin/bash
# SKD Training Launch Script
# Usage: bash scripts/run_skd.sh [config_file]

set -e

# Default configuration
CONFIG_FILE="${1:-configs/skd.yaml}"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate icml26

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "Running single GPU training..."
    python train_skd.py --config "$CONFIG_FILE"
else
    # Multi-GPU training with accelerate
    echo "Running distributed training on $NUM_GPUS GPUs..."
    accelerate launch \
        --config_file configs/accelerate_config.yaml \
        --num_processes $NUM_GPUS \
        train_skd.py --config "$CONFIG_FILE"
fi

echo "Training completed!"

