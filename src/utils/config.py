"""Configuration loading and merging utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    Values in override take precedence over base.
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    config_path: Union[str, Path],
    config_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Load a configuration file with inheritance support.
    
    Supports `defaults` key for inheriting from other configs:
    ```yaml
    defaults:
      - base  # Will load base.yaml from the same directory
    
    # Override settings here
    training:
      learning_rate: 1e-5
    ```
    
    Args:
        config_path: Path to the configuration file
        config_dir: Directory containing config files (for resolving defaults)
        
    Returns:
        Merged configuration dictionary
    """
    config_path = Path(config_path)
    
    if config_dir is None:
        config_dir = config_path.parent
    else:
        config_dir = Path(config_dir)
    
    config = load_yaml(config_path)
    
    # Handle defaults (inheritance)
    defaults = config.pop("defaults", [])
    
    if defaults:
        base_config = {}
        for default in defaults:
            if isinstance(default, str):
                # Simple string reference
                default_path = config_dir / f"{default}.yaml"
                if default_path.exists():
                    parent_config = load_config(default_path, config_dir)
                    base_config = deep_merge(base_config, parent_config)
            elif isinstance(default, dict):
                # Dict with optional settings
                for name, settings in default.items():
                    default_path = config_dir / f"{name}.yaml"
                    if default_path.exists():
                        parent_config = load_config(default_path, config_dir)
                        base_config = deep_merge(base_config, parent_config)
        
        # Merge current config on top of base
        config = deep_merge(base_config, config)
    
    return config


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Optional[Dict[str, Any]] = None,
    cli_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Merge configurations with priority: cli_args > override_config > base_config.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        cli_args: Command-line arguments as dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    if override_config:
        result = deep_merge(result, override_config)
    
    if cli_args:
        # Filter out None values from CLI args
        cli_args = {k: v for k, v in cli_args.items() if v is not None}
        result = deep_merge(result, cli_args)
    
    return result


def get_torch_dtype(dtype_str: str):
    """Convert string dtype to torch dtype."""
    import torch
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    
    return dtype_map.get(dtype_str, "auto")


def config_to_training_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training arguments from config for TrainingArguments.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Dictionary suitable for TrainingArguments
    """
    training = config.get("training", {})
    
    # Map config keys to TrainingArguments keys
    args = {
        "output_dir": training.get("output_dir", "outputs/default"),
        "per_device_train_batch_size": training.get("per_device_train_batch_size", 2),
        "per_device_eval_batch_size": training.get("per_device_eval_batch_size", 2),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", 8),
        "learning_rate": training.get("learning_rate", 2e-5),
        "weight_decay": training.get("weight_decay", 0.01),
        "num_train_epochs": training.get("num_train_epochs", 3),
        "max_steps": training.get("max_steps", -1),
        "warmup_ratio": training.get("warmup_ratio", 0.1),
        "lr_scheduler_type": training.get("lr_scheduler_type", "cosine"),
        "bf16": training.get("bf16", True),
        "fp16": training.get("fp16", False),
        "gradient_checkpointing": training.get("gradient_checkpointing", True),
        "logging_steps": training.get("logging_steps", 10),
        "logging_first_step": training.get("logging_first_step", True),
        "eval_strategy": training.get("eval_strategy", "steps"),
        "eval_steps": training.get("eval_steps", 100),
        "save_strategy": training.get("save_strategy", "steps"),
        "save_steps": training.get("save_steps", 100),
        "save_total_limit": training.get("save_total_limit", 3),
        "load_best_model_at_end": training.get("load_best_model_at_end", True),
        "seed": training.get("seed", 42),
        "data_seed": training.get("data_seed", 42),
    }
    
    # Handle wandb
    wandb_config = config.get("wandb", {})
    if wandb_config.get("enabled", False):
        args["report_to"] = "wandb"
        if wandb_config.get("run_name"):
            args["run_name"] = wandb_config["run_name"]
    else:
        args["report_to"] = "none"
    
    return args

