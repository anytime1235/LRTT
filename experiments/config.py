"""
Configuration module for baseline experiments.
Contains common settings for digital and analog baseline experiments.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

# ============================================================================
# Path Configuration
# ============================================================================

# Base paths
BASE_DIR = Path("/data/LRTT_transformer")
DATA_DIR = Path("/data")
RESULTS_DIR = DATA_DIR / "AIMC_LoRA_results"
PLOTS_DIR = DATA_DIR / "AIMC_LoRA_plots"

# Script paths
QA_SCRIPT = BASE_DIR / "lora_training" / "run_qa.py"
GLUE_SCRIPT = BASE_DIR / "lora_training_glue" / "run_glue.py"

# Output directories
DIGITAL_QA_DIR = RESULTS_DIR / "digital" / "qa"
DIGITAL_GLUE_DIR = RESULTS_DIR / "digital" / "glue"
ANALOG_QA_DIR = RESULTS_DIR / "analog" / "qa"
ANALOG_GLUE_DIR = RESULTS_DIR / "analog" / "glue"
SIXT1C_QA_DIR = RESULTS_DIR / "sixt1c" / "qa"
SIXT1C_GLUE_DIR = RESULTS_DIR / "sixt1c" / "glue"
TIKITAKA_QA_DIR = RESULTS_DIR / "tikitaka" / "qa"
TIKITAKA_GLUE_DIR = RESULTS_DIR / "tikitaka" / "glue"

# ============================================================================
# Model Configuration
# ============================================================================

MODEL_NAME = "google/mobilebert-uncased"

# ============================================================================
# GLUE Tasks
# ============================================================================

GLUE_TASKS: List[str] = [
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]

# GLUE task primary metrics
GLUE_METRICS: Dict[str, str] = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "wnli": "accuracy",
}

# ============================================================================
# Training Parameters
# ============================================================================

# Digital baseline training parameters
DIGITAL_TRAINING_PARAMS: Dict[str, Any] = {
    "do_train": True,
    "do_eval": True,
    "num_train_epochs": 15,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "warmup_steps": 0,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "report_to": "wandb",
}

# Analog baseline training parameters
ANALOG_TRAINING_PARAMS: Dict[str, Any] = {
    "do_train": True,
    "do_eval": True,
    "num_train_epochs": 3,  # Reduced from 15 to 3 for faster iteration
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "warmup_steps": 0,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "save_strategy": "no",  # Disabled for analog models (safetensors incompatibility)
    "eval_strategy": "epoch",
    "load_best_model_at_end": False,  # Disabled since save_strategy is no
    "metric_for_best_model": "eval_loss",
    "report_to": "wandb",
}

# Analog-specific parameters (PCM)
ANALOG_PARAMS: Dict[str, Any] = {
    "pcm_model": "PCM_Gmax25",
    "output_noise_level": 0.04,
    "analog_optimizer": "AnalogAdam",
    "analog_lr": 0.0002,
    "num_evaluation_drift_values": 7,
    "num_evaluation_repetition": 10,
}

# Sixt1c (6T1C) training parameters (shorter training for faster iteration)
SIXT1C_TRAINING_PARAMS: Dict[str, Any] = {
    "do_train": True,
    "do_eval": True,
    "num_train_epochs": 3,  # Shorter than analog (3 vs 15)
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,  # Use ratio instead of fixed steps for small datasets
    "warmup_steps": 0,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "save_strategy": "no",
    "eval_strategy": "epoch",
    "load_best_model_at_end": False,
    "metric_for_best_model": "eval_loss",
    "report_to": "wandb",
}

# Sixt1c (6T1C) device parameters
SIXT1C_PARAMS: Dict[str, Any] = {
    "pcm_model": "PCM_Gmax25",     # Required for base_layer PCM conversion
    "dt_batch_sec": 1.0,           # Time step between batches in seconds
    "include_retention": True,      # Whether to include retention decay
    "write_noise_std": 0.0,        # Write noise (set to 0 as requested)
    "output_noise_level": 0.0,     # Output noise level
    "analog_optimizer": "AnalogAdam",
    "analog_lr": 0.0002,
    "num_evaluation_drift_values": 7,
    "num_evaluation_repetition": 10,
}

# TikiTaka v2 (2-tile: Fast 6T1C + Slow SoftBounds) parameters
# Trial 17 Best Hyperparameters
TIKITAKA_PARAMS: Dict[str, Any] = {
    # Trial 17 Best Hyperparameters
    "transfer_every": 160,         # Transfer frequency in mini-batches
    "transfer_lr": 7.36,           # Learning rate for transfer
    "fast_lr": 0.862,              # Fast tile learning rate multiplier
    "auto_granularity": 305.91,    # Granularity for auto scaling
    "in_chop_prob": 0.020,         # Probability for chopped input

    # Fixed settings (TikiTaka v2)
    "gamma": 0.0,                  # Only Slow/C tile visible (0=Slow, 1=Fast)
    "auto_scale": True,            # Enable auto scaling
    "units_in_mbatch": False,      # Mat-vec units (not mini-batch)

    # Target modules for analog conversion - Q,K,V only (no LoRA)
    "target_modules": ["query", "key", "value"],

    # Optimizer
    "analog_optimizer": "AnalogAdam",
    "analog_lr": 1.31e-04,         # Trial 17 best learning rate

    # Drift evaluation parameters (for fair comparison with PCM/sixt1c)
    "enable_drift": True,          # Enable drift evaluation on Slow tile
    "drift_nu": 0.05,              # Mean drift exponent (PCM-like)
    "drift_nu_dtod": 0.02,         # Device-to-device variation in nu
    "num_evaluation_drift_values": 7,  # Number of drift time points
    "num_evaluation_repetition": 10,   # Repetitions per drift value
}

# ============================================================================
# Drift Configuration
# ============================================================================

# Drift time values in seconds
# 0s, 1 hour, 1 day, 1 week, 1 month, 1 year, 10 years
DRIFT_VALUES_SECONDS: List[int] = [
    0,           # 0 seconds (t0)
    3600,        # 1 hour
    86400,       # 1 day
    604800,      # 1 week
    2592000,     # 1 month (30 days)
    31536000,    # 1 year
    315360000,   # 10 years
]

DRIFT_LABELS: Dict[int, str] = {
    0: "t0",
    3600: "1h",
    86400: "1d",
    604800: "1w",
    2592000: "1mo",
    31536000: "1y",
    315360000: "10y",
}

# ============================================================================
# Quick Test Configuration
# ============================================================================

QUICK_TEST_PARAMS: Dict[str, Any] = {
    "max_train_samples": 100,
    "max_eval_samples": 100,
    "num_train_epochs": 1,
    "num_evaluation_drift_values": 3,
    "num_evaluation_repetition": 2,
}

# ============================================================================
# W&B Configuration
# ============================================================================

WANDB_PROJECT = "AIMC_LoRA"
WANDB_ENTITY = None  # Use default entity

# ============================================================================
# Helper Functions
# ============================================================================

def ensure_directories() -> None:
    """Create all necessary directories if they don't exist."""
    directories = [
        RESULTS_DIR,
        PLOTS_DIR,
        DIGITAL_QA_DIR,
        DIGITAL_GLUE_DIR,
        ANALOG_QA_DIR,
        ANALOG_GLUE_DIR,
        SIXT1C_QA_DIR,
        SIXT1C_GLUE_DIR,
        TIKITAKA_QA_DIR,
        TIKITAKA_GLUE_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_output_dir(experiment_type: str, task_type: str, task_name: str = None) -> Path:
    """
    Get the output directory for a specific experiment.

    Args:
        experiment_type: 'digital', 'analog', 'sixt1c', or 'tikitaka'
        task_type: 'qa' or 'glue'
        task_name: For GLUE tasks, the specific task name

    Returns:
        Path to the output directory
    """
    if experiment_type == "digital":
        if task_type == "qa":
            return DIGITAL_QA_DIR
        else:
            return DIGITAL_GLUE_DIR / task_name if task_name else DIGITAL_GLUE_DIR
    elif experiment_type == "sixt1c":
        if task_type == "qa":
            return SIXT1C_QA_DIR
        else:
            return SIXT1C_GLUE_DIR / task_name if task_name else SIXT1C_GLUE_DIR
    elif experiment_type == "tikitaka":
        if task_type == "qa":
            return TIKITAKA_QA_DIR
        else:
            return TIKITAKA_GLUE_DIR / task_name if task_name else TIKITAKA_GLUE_DIR
    else:
        if task_type == "qa":
            return ANALOG_QA_DIR
        else:
            return ANALOG_GLUE_DIR / task_name if task_name else ANALOG_GLUE_DIR


def get_glue_tasks_subset(tasks: List[str] = None) -> List[str]:
    """Get a subset of GLUE tasks or all tasks if none specified."""
    if tasks is None:
        return GLUE_TASKS
    return [t for t in tasks if t in GLUE_TASKS]
