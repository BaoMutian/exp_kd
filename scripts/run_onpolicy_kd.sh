#!/bin/bash
# On-Policy KD Training Launch Script
# Usage: bash scripts/run_onpolicy_kd.sh [config_file]
#
# IMPORTANT: On-Policy KD uses model.generate() during training, which is
# incompatible with DeepSpeed ZeRO-3. This script uses ZeRO-2 configuration.

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
    # NOTE: Uses ZeRO-2 config because generate() requires full model parameters
    echo "Running distributed training on $NUM_GPUS GPUs with ZeRO-2..."
    accelerate launch \
        --config_file configs/accelerate_onpolicy_kd.yaml \
        --num_processes $NUM_GPUS \
        train_onpolicy_kd.py --config "$CONFIG_FILE"
fi

echo "Training completed!"

