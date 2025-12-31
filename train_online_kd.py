#!/usr/bin/env python3
"""
Online KD (On-Policy Knowledge Distillation) Training Script

True on-policy distillation for experience internalization where:
- Student model: receives Q (without experience) and generates responses A
- Teacher model: receives Q+E (with experience) and generates high-quality A'
- Student learns to match teacher's output distribution

This enables the student to internalize the experience E into its weights,
so that it can generate high-quality responses without needing E at inference time.

Usage:
    python train_online_kd.py --config configs/online_kd.yaml
    
    # Distributed training (8 GPUs)
    accelerate launch --config_file configs/accelerate_config.yaml train_online_kd.py \
        --config configs/online_kd.yaml
"""

from src.trainers.online_kd_trainer import OnlineKDTrainer, OnlineKDDataCollator
# Reuse SKD dataset (has both teacher and student messages)
from src.data import create_skd_dataset
from src.utils import load_config, get_torch_dtype
import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

# Add project root to path for local imports
sys.path.insert(0, str(Path(__file__).parent))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Online KD Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/online_kd.yaml",
        help="Path to configuration file",
    )
    # Allow overriding config values from command line
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size",
                        type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of responses to sample from student per input")
    parser.add_argument("--kl_type", type=str,
                        choices=["forward", "reverse", "jsd"], default=None)
    parser.add_argument("--beta", type=float, default=None)

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
        config["online_kd"]["temperature"] = args.temperature
    if args.max_new_tokens is not None:
        config["online_kd"]["max_new_tokens"] = args.max_new_tokens
    if args.num_samples is not None:
        config["online_kd"]["num_samples"] = args.num_samples
    if args.kl_type is not None:
        config["online_kd"]["kl_type"] = args.kl_type
    if args.beta is not None:
        config["online_kd"]["beta"] = args.beta

    # Extract config sections
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    peft_config = config.get("peft", {})
    online_kd_config = config.get("online_kd", {})
    generation_config = config.get("generation", {})

    # Create output directory
    output_dir = Path(training_config.get("output_dir", "outputs/online_kd"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Online KD specific parameters
    num_samples = online_kd_config.get("num_samples", 1)
    temperature = online_kd_config.get("temperature", 1.0)
    max_new_tokens = online_kd_config.get("max_new_tokens", 256)
    kl_type = online_kd_config.get("kl_type", "jsd")
    beta = online_kd_config.get("beta", 0.5)

    logger.info("=" * 60)
    logger.info("Online KD Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Student model: {model_config.get('student_path')}")
    logger.info(f"Teacher model: {model_config.get('teacher_path')}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Num samples per input: {num_samples}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Max new tokens: {max_new_tokens}")
    logger.info(f"KL type: {kl_type}")
    logger.info(f"Beta (JSD): {beta}")
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
        dtype=torch_dtype,
        trust_remote_code=model_config.get("trust_remote_code", True),
        attn_implementation=model_config.get(
            "attn_implementation", "flash_attention_2"),
    )

    logger.info("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_config.get("teacher_path"),
        dtype=torch_dtype,
        trust_remote_code=model_config.get("trust_remote_code", True),
        attn_implementation=model_config.get(
            "attn_implementation", "flash_attention_2"),
    )

    # Apply PEFT/LoRA to student if enabled
    if peft_config.get("enabled", False):
        logger.info("Applying LoRA configuration to student model...")
        lora_config = LoraConfig(
            r=peft_config.get("lora_r", 64),
            lora_alpha=peft_config.get("lora_alpha", 128),
            lora_dropout=peft_config.get("lora_dropout", 0.05),
            target_modules=peft_config.get(
                "target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        student_model = get_peft_model(student_model, lora_config)
        student_model.print_trainable_parameters()

    # Load dataset (reuse SKD dataset which has both teacher and student messages)
    logger.info("Loading and preparing dataset...")
    train_dataset = create_skd_dataset(
        teacher_data_path=data_config.get("teacher_data_path"),
        student_data_path=data_config.get("student_data_path"),
    )
    logger.info(f"Dataset size: {len(train_dataset)} samples")

    # Create data collator
    data_collator = OnlineKDDataCollator(
        tokenizer=tokenizer,
        max_seq_length=data_config.get("max_seq_length", 4096),
        max_new_tokens=max_new_tokens,
        enable_thinking=model_config.get("enable_thinking", False),
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training_config.get(
            "per_device_train_batch_size", 1),
        per_device_eval_batch_size=training_config.get(
            "per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training_config.get(
            "gradient_accumulation_steps", 16),
        learning_rate=training_config.get("learning_rate", 5e-6),
        weight_decay=training_config.get("weight_decay", 0.01),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        max_steps=training_config.get("max_steps", -1),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        bf16=training_config.get("bf16", True),
        fp16=training_config.get("fp16", False),
        gradient_checkpointing=training_config.get(
            "gradient_checkpointing", True),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        logging_steps=training_config.get("logging_steps", 10),
        logging_first_step=training_config.get("logging_first_step", True),
        eval_strategy=training_config.get("eval_strategy", "no"),
        save_strategy=training_config.get("save_strategy", "steps"),
        save_steps=training_config.get("save_steps", 100),
        save_total_limit=training_config.get("save_total_limit", 3),
        seed=training_config.get("seed", 42),
        data_seed=training_config.get("data_seed", 42),
        report_to="wandb" if config.get("wandb", {}).get(
            "enabled", False) else "none",
        remove_unused_columns=False,
    )

    # Initialize trainer
    logger.info("Initializing OnlineKDTrainer...")
    trainer = OnlineKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        generation_config=generation_config,
        kl_type=kl_type,
        beta=beta,
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
