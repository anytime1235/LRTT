#!/usr/bin/env python
# coding=utf-8
"""Common configuration for single layer comparison experiments.

Compares LRTT vs Digital LoRA on a single MobileBERT layer.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# =============================================================================
# Model & Task Settings
# =============================================================================

MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
SEED = 42

# Target layer for single-layer experiments
TARGET_LAYER_NAME = "mobilebert.encoder.layer.0.attention.self.query"

# =============================================================================
# Common LoRA Settings
# =============================================================================

RANK = 8
LORA_ALPHA = 1.0  # For LRTT (Digital LoRA can use different alpha)

# =============================================================================
# LRTT Settings (decay mode from sweep_mobilebert_optuna.py)
# =============================================================================

LRTT_CONFIG = {
    "rank": RANK,
    "transfer_every": 100,
    "transfer_lr": 1.0,
    "lora_alpha": LORA_ALPHA,
    "reinit_mode": "decay",
    "decay_factor": 1.0,
    "reinit_gain": 0.1,
    "forward_inject": False,  # User specified: LRTT uses forward_inject=False
    "update_mode": "lora",
    "transfer_mode": "off",
}

# =============================================================================
# SoftBounds C tile config (shared by both LRTT and Digital LoRA)
# =============================================================================

SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001,
    'w_max': 1.0,
    'w_min': -1.0,
    'dw_min_dtod': 0.0,
    'dw_min_std': 0.0,
    'up_down': 0.0,
    'up_down_dtod': 0.0,
    'w_max_dtod': 0.0,
    'w_min_dtod': 0.0,
    'write_noise_std': 0.0,
    'mult_noise': True,
}

# =============================================================================
# LinearStepDevice for A/B tiles (LRTT only)
# =============================================================================

def get_ab_lifetime(dt_batch_sec: float = 1.0) -> float:
    """Calculate lifetime from physical τ for 6T1C."""
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    return 1.0 / delta if delta > 0 else 0.0


LINEARSTEP_AB_CONFIG = {
    'dw_min': 0.001981,
    'up_down': 0.0,
    'w_max': 1.0,
    'w_min': -1.0,
    'gamma_up': -0.1678,
    'gamma_down': 0.1410,
    'mult_noise': True,
    'dw_min_dtod': 0.1,
    'up_down_dtod': 0.01,
    'w_max_dtod': 0.05,
    'w_min_dtod': 0.05,
    'gamma_up_dtod': 0.05,
    'gamma_down_dtod': 0.05,
    'dw_min_std': 0.3,
    'write_noise_std': 0.0,
    'mean_bound_reference': True,
    'lifetime': get_ab_lifetime(1.0),  # 1 sec per batch
    'lifetime_dtod': 0.1,
    'reset': 0.0,
    'reset_dtod': 0.0,
}

# =============================================================================
# Digital LoRA Settings
# =============================================================================

DIGITAL_LORA_CONFIG = {
    "rank": RANK,
    "lora_alpha": 32,  # Standard LoRA alpha
    "lora_dropout": 0.0,
}

# =============================================================================
# PEFT LoRA + LRTT Best Match Settings (from synthetic experiments)
# =============================================================================
# Best match conditions from previous synthetic experiments:
# - transfer_every = 10
# - transfer_lr = 0.0125
# - lora_alpha = 4, rank = 4 → scaling = 1.0
# Results: Loss correlation 0.98, Output cosine sim 0.98, Weight distance ~2%

BEST_MATCH_RANK = 8
BEST_MATCH_LORA_ALPHA = 32  # scaling = lora_alpha / rank = 32/8 = 4.0 (same as lora_on_analog_hardware)

PEFT_LORA_CONFIG = {
    "rank": BEST_MATCH_RANK,
    "lora_alpha": BEST_MATCH_LORA_ALPHA,
    "lora_dropout": 0.0,
    "target_modules": ["query"],  # Only target query layer
}

LRTT_CONFIG_BEST_MATCH = {
    "rank": BEST_MATCH_RANK,
    "transfer_every": 10,
    "transfer_lr": 0.002,  # Optimized from 0.0125 to reduce weight distance <5%
    "lora_alpha": 1.0,  # LRTT uses lora_alpha=1.0 (scaling handled differently)
    "reinit_mode": "decay",
    "decay_factor": 1.0,
    "reinit_gain": 0.1,
    "forward_inject": False,  # LRTT uses forward_inject=False
    "update_mode": "lora",
    "transfer_mode": "off",
}

# =============================================================================
# Training Settings
# =============================================================================

LEARNING_RATE = 2e-4  # same as lora_on_analog_hardware
LOGGING_STEPS = 50
OUTPUT_DIR = "/home/jovyan/work/single_layer_comparison/results"

# =============================================================================
# Metrics Tracking Settings
# =============================================================================

TRACK_EVERY_STEPS = 10  # Snapshot weights every N steps

# =============================================================================
# Three-Method Comparison Settings
# =============================================================================

EVAL_EVERY_STEPS = 50     # Evaluate validation accuracy every N steps
EVAL_SUBSET_SIZE = 500    # Subset size for fast evaluation
NUM_COMPARISON_STEPS = 500  # Default number of training steps for comparison

# =============================================================================
# Progressive Layer Comparison Settings
# =============================================================================

PROGRESSIVE_CONFIGS = {
    "C1_classifier_only": {
        "target_modules": [],
        "description": "Classifier only (no LoRA)",
    },
    "C2_query": {
        "target_modules": ["query"],
        "description": "Q layer + Classifier",
    },
    "C3_query_value": {
        "target_modules": ["query", "value"],
        "description": "Q,V layers + Classifier",
    },
    "C4_query_key_value": {
        "target_modules": ["query", "key", "value"],
        "description": "Q,K,V layers + Classifier",
    },
}
