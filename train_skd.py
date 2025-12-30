#!/usr/bin/env python3
"""
SKD (Supervised Knowledge Distillation) Training Script

SKD trains the student to match teacher's token-level probability distribution
using KL divergence on target sequences.

Key idea:
- Teacher input: Q+E (with experience) -> produces soft labels
- Student input: Q (without experience) -> learns to match teacher's distribution
- Target: Same response sequence A

Loss = alpha * KL(p_teacher || p_student) + (1 - alpha) * CE(p_student, labels)

Usage:
    python train_skd.py --config configs/skd.yaml
    
    # With overrides
    python train_skd.py --config configs/skd.yaml \
        --temperature 3.0 \
        --alpha 0.8
        
    # Distributed training (8 GPUs)
    accelerate launch --config_file configs/accelerate_config.yaml train_skd.py \
        --config configs/skd.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import create_skd_dataset
from src.trainers import SKDTrainer
from src.trainers.skd_trainer import SKDDataCollator
from src.utils import load_config, get_torch_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SKD Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/skd.yaml",
        help="Path to configuration file",
    )
    # Allow overriding config values from command line
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--kl_direction", type=str, choices=["forward", "reverse"], default=None)
    
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
    if args.temperature is not None:
        config["skd"]["temperature"] = args.temperature
    if args.alpha is not None:
        config["skd"]["alpha"] = args.alpha
    if args.kl_direction is not None:
        config["skd"]["kl_direction"] = args.kl_direction
    
    # Extract config sections
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    peft_config = config.get("peft", {})
    skd_config = config.get("skd", {})
    
    # Create output directory
    output_dir = Path(training_config.get("output_dir", "outputs/skd"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # SKD specific parameters
    temperature = skd_config.get("temperature", 2.0)
    alpha = skd_config.get("alpha", 0.7)
    kl_direction = skd_config.get("kl_direction", "forward")
    
    logger.info("=" * 60)
    logger.info("SKD Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Student model: {model_config.get('student_path')}")
    logger.info(f"Teacher model: {model_config.get('teacher_path')}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Alpha (KL weight): {alpha}")
    logger.info(f"KL direction: {kl_direction}")
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
    train_dataset = create_skd_dataset(
        teacher_data_path=data_config.get("teacher_data_path"),
        student_data_path=data_config.get("student_data_path"),
    )
    logger.info(f"Dataset size: {len(train_dataset)} samples")
    
    # Create data collator
    data_collator = SKDDataCollator(
        tokenizer=tokenizer,
        max_seq_length=data_config.get("max_seq_length", 4096),
        enable_thinking=model_config.get("enable_thinking", False),
    )
    
    # Training arguments
    training_args = TrainingArguments(
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
        remove_unused_columns=False,  # Important for custom data collator
    )
    
    # Initialize trainer
    logger.info("Initializing SKDTrainer...")
    trainer = SKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        temperature=temperature,
        alpha=alpha,
        kl_direction=kl_direction,
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

