#!/usr/bin/env python3
"""
GKD (Generalized Knowledge Distillation) Training Script

GKD implements on-policy distillation where the student generates outputs
and receives token-level feedback from the teacher.

Key features:
- On-policy learning: Student generates outputs, teacher provides feedback
- Flexible divergence: Can use forward KL, reverse KL, or JSD
- Mixed training: Combines on-policy and supervised (off-policy) data

The lambda parameter controls the on-policy ratio:
- lambda=0.0: Pure supervised (like SKD)
- lambda=1.0: Pure on-policy (student generates all data)
- lambda=0.5: Mix of both

Usage:
    python train_gkd.py --config configs/gkd.yaml
    
    # With overrides
    python train_gkd.py --config configs/gkd.yaml \
        --lmbda 0.8 \
        --beta 0.3
        
    # Distributed training (8 GPUs)
    accelerate launch --config_file configs/accelerate_config.yaml train_gkd.py \
        --config configs/gkd.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import create_gkd_dataset
from src.utils import load_config, get_torch_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GKD Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gkd.yaml",
        help="Path to configuration file",
    )
    # Allow overriding config values from command line
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    
    # GKD specific parameters
    parser.add_argument("--lmbda", type=float, default=None,
                       help="On-policy ratio (0=supervised, 1=on-policy)")
    parser.add_argument("--beta", type=float, default=None,
                       help="JSD interpolation (0=forward KL, 1=reverse KL)")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--seq_kd", action="store_true", default=None,
                       help="Use sequence-level KD")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Apply command line overrides
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        config["training"]["num_train_epochs"] = args.num_train_epochs
    if args.per_device_train_batch_size is not None:
        config["training"]["per_device_train_batch_size"] = args.per_device_train_batch_size
    if args.gradient_accumulation_steps is not None:
        config["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir
    if args.lmbda is not None:
        config["gkd"]["lmbda"] = args.lmbda
    if args.beta is not None:
        config["gkd"]["beta"] = args.beta
    if args.temperature is not None:
        config["gkd"]["temperature"] = args.temperature
    if args.max_new_tokens is not None:
        config["gkd"]["max_new_tokens"] = args.max_new_tokens
    if args.seq_kd is not None:
        config["gkd"]["seq_kd"] = args.seq_kd
    
    # Extract config sections
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    peft_config = config.get("peft", {})
    gkd_config = config.get("gkd", {})
    generation_config = config.get("generation", {})
    
    # Create output directory
    output_dir = Path(training_config.get("output_dir", "outputs/gkd"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GKD specific parameters
    lmbda = gkd_config.get("lmbda", 0.5)
    beta = gkd_config.get("beta", 0.5)
    temperature = gkd_config.get("temperature", 0.9)
    max_new_tokens = gkd_config.get("max_new_tokens", 256)
    seq_kd = gkd_config.get("seq_kd", False)
    disable_dropout = gkd_config.get("disable_dropout", True)
    
    logger.info("=" * 60)
    logger.info("GKD Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Student model: {model_config.get('student_path')}")
    logger.info(f"Teacher model: {model_config.get('teacher_path')}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Lambda (on-policy ratio): {lmbda}")
    logger.info(f"Beta (JSD interpolation): {beta}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Max new tokens: {max_new_tokens}")
    logger.info(f"Sequence KD: {seq_kd}")
    logger.info(f"Learning rate: {training_config.get('learning_rate')}")
    logger.info(f"Epochs: {training_config.get('num_train_epochs')}")
    logger.info("=" * 60)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get("student_path"),
        trust_remote_code=model_config.get("trust_remote_code", True),
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load models
    torch_dtype = get_torch_dtype(model_config.get("torch_dtype", "bfloat16"))
    
    logger.info("Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        model_config.get("student_path"),
        torch_dtype=torch_dtype,
        trust_remote_code=model_config.get("trust_remote_code", True),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
    )
    
    logger.info("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_config.get("teacher_path"),
        torch_dtype=torch_dtype,
        trust_remote_code=model_config.get("trust_remote_code", True),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
    )
    
    # Apply PEFT/LoRA to student if enabled
    if peft_config.get("enabled", False):
        logger.info("Applying LoRA configuration to student model...")
        lora_config = LoraConfig(
            r=peft_config.get("lora_r", 64),
            lora_alpha=peft_config.get("lora_alpha", 128),
            lora_dropout=peft_config.get("lora_dropout", 0.05),
            target_modules=peft_config.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()
    
    # Load dataset
    logger.info("Loading and preparing dataset...")
    train_dataset = create_gkd_dataset(
        teacher_data_path=data_config.get("teacher_data_path"),
        student_data_path=data_config.get("student_data_path"),
    )
    logger.info(f"Dataset size: {len(train_dataset)} samples")
    
    # Create GKD training config
    gkd_training_config = GKDConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 16),
        learning_rate=training_config.get("learning_rate", 1e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        max_steps=training_config.get("max_steps", -1),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        bf16=training_config.get("bf16", True),
        fp16=training_config.get("fp16", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        logging_steps=training_config.get("logging_steps", 10),
        logging_first_step=training_config.get("logging_first_step", True),
        eval_strategy=training_config.get("eval_strategy", "no"),
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 100),
        save_total_limit=training_config.get("save_total_limit", 3),
        seed=training_config.get("seed", 42),
        data_seed=training_config.get("data_seed", 42),
        report_to="wandb" if config.get("wandb", {}).get("enabled", False) else "none",
        # GKD specific parameters
        lmbda=lmbda,
        beta=beta,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        seq_kd=seq_kd,
        disable_dropout=disable_dropout,
        # SFT parameters for data handling
        max_length=data_config.get("max_seq_length", 4096),
    )
    
    # Custom formatting function for Qwen3 non-thinking mode
    def formatting_func(example):
        """Format messages using Qwen3 chat template with non-thinking mode."""
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=model_config.get("enable_thinking", False),
        )
        return text
    
    # Initialize trainer
    logger.info("Initializing GKDTrainer...")
    trainer = GKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=gkd_training_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(str(output_dir / "final_model"))
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("Training completed!")
    logger.info(f"Model saved to: {output_dir / 'final_model'}")


if __name__ == "__main__":
    main()

