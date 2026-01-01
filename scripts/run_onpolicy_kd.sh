#!/bin/bash
# On-Policy KD Training Launch Script
# Usage: bash scripts/run_onpolicy_kd.sh [config_file]

set -e

# Default configuration
CONFIG_FILE="${1:-configs/onpolicy_kd.yaml}"

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs: $NUM_GPUS"

if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "Running single GPU training..."
    python train_onpolicy_kd.py --config "$CONFIG_FILE"
else
    # Multi-GPU training with accelerate
    echo "Running distributed training on $NUM_GPUS GPUs..."
    accelerate launch \
        --config_file configs/accelerate_config.yaml \
        --num_processes $NUM_GPUS \
        train_onpolicy_kd.py --config "$CONFIG_FILE"
fi

echo "Training completed!"

