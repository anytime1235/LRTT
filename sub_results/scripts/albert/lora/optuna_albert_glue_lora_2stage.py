# -*- coding: utf-8 -*-
"""2-Stage Optuna HPO for ALBERT + GLUE with LoRA-LRTT.

Stage 1: Pretrain classifier + LayerNorm (run pretrain_classifier.py first)
Stage 2: Sweep LoRA-LRTT HPs with frozen classifier + LayerNorm (this script)

Requires: /data/classifier_ckpt/{task}/ckpt.pt from pretrain_classifier.py

This script automatically:
  - Loads pretrained classifier + LayerNorm from Stage 1 checkpoint
  - Freezes all digital params (classifier, LayerNorm)
  - Trains only LoRA A/B analog tiles
  - Uses STAGE0_CONFIGS for default Stage 2 epoch count

Usage:
    python optuna_albert_glue_lora_2stage.py --task sst2 --n-trials 50
    python optuna_albert_glue_lora_2stage.py --task sst2 --n-trials 50 --optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --warmup-ratio 0.1 --lora-target all --warm-alpha --convert-nontarget --no-learn-out-scaling

Note: ALBERT uses weight sharing across all encoder layers, so the number of
unique Linear layers converted to analog is much smaller than MobileBERT.
(e.g., attn = 4 unique layers shared across 12 transformer blocks)
"""

import os
import sys
import re
import string
import math
import json
import argparse
import gc
import collections

import torch
from torch import nn, no_grad, manual_seed
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, FloatingPointDevice

# LRTT config imports (direct imports to avoid __init__.py dependency issues)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

from collections import Counter


# =============================================================================
# GLUE Task Configurations
# =============================================================================

GLUE_TASKS = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

TASK_TO_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
    "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1,
}

TASK_TO_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "stsb": "spearmanr",
}


# =============================================================================
# Global Constants
# =============================================================================

DEFAULT_STUDY_NAME = "albert_glue_lora2s_main"

# 2-Stage config
PRETRAIN_CKPT_DIR = "/data/classifier_ckpt"
PRETRAIN_CKPT = None  # set via --pretrain-ckpt or auto-detected

# Per-task Stage-0/1 step budget from TS 2x schedule
# stage1_epochs = ceil((total_2x - stage0) / steps_per_epoch)
STAGE0_CONFIGS = {
    "cola":  {"stage0_steps": 2134,  "total_steps_2x": 10672, "stage1_epochs": 16},
    "stsb":  {"stage0_steps": 1439,  "total_steps_2x": 7196,  "stage1_epochs": 16},
    "sst2":  {"stage0_steps": 4187,  "total_steps_2x": 41870, "stage1_epochs": 18},
    "mnli":  {"stage0_steps": 2000,  "total_steps_2x": 20000, "stage1_epochs": 6},
    "qnli":  {"stage0_steps": 6622,  "total_steps_2x": 66224, "stage1_epochs": 19},
    "qqp":   {"stage0_steps": 2800,  "total_steps_2x": 28000, "stage1_epochs": 9},
    "rte":   {"stage0_steps": 320,   "total_steps_2x": 1600,  "stage1_epochs": 17},
    "mrpc":  {"stage0_steps": 320,   "total_steps_2x": 1600,  "stage1_epochs": 12},
}

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# GLUE task (set via --task)
TASK_NAME = "sst2"

# Paths (task-specific subdirectories created later after TASK_NAME is set)
RESULTS = None  # Set after argparse

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "albert/albert-base-v2"
MAX_SEQ_LENGTH = 128  # GLUE default (overridden per-task)

# Per-task configs (from tiki baseline)
TASK_CONFIGS = {
    "cola":  {"batch_size": 16,  "epochs": 20, "max_seq_length": 128},
    "stsb":  {"batch_size": 16,  "epochs": 20, "max_seq_length": 128},
    "sst2":  {"batch_size": 32,  "epochs": 20, "max_seq_length": 128},
    "mnli":  {"batch_size": 128, "epochs": 4,  "max_seq_length": 128},
    "qnli":  {"batch_size": 32,  "epochs": 21, "max_seq_length": 128},
    "qqp":   {"batch_size": 128, "epochs": 10, "max_seq_length": 128},
    "rte":   {"batch_size": 32,  "epochs": 21, "max_seq_length": 256},
    "mrpc":  {"batch_size": 32,  "epochs": 14, "max_seq_length": 128},
}

# Training defaults (overridden by TASK_CONFIGS per task)
N_EPOCHS = 20
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 3
OS_CALIB_EPOCHS = 0  # Out-scaling calibration epochs (0 = disabled, train alongside A/B)
LEARN_COMBINED_OS = True  # Train combined_out_scaling throughout (separate LR)
COMBINED_OS_LR = 1e-3     # Fixed LR for combined_out_scaling params
TRAIN_LN = False          # Unfreeze LayerNorm in Stage 2
TRAIN_CLS = False         # Unfreeze classifier in Stage 2

# Scheduler
WARMUP_RATIO = 0.05  # 5% of total steps
WARM_ALPHA = False

# Dynamic TE
DYNAMIC_TE = False
DYNAMIC_TE_POWER = 1.0
TE_WARMUP_STEPS = 0
TE_WARMUP_SCHEDULE = []

# Fixed LRTT parameters
REINIT_GAIN = 1.0
DECAY_FACTOR = 1.0
TRANSFER_METHOD = "onehot"  # "onehot", "direct", or "set"
AB_DEVICE = "6t1c"  # "6t1c" or "fp"
IO_NOISE = True  # If False, disable out_noise (resolution kept)
AB_PERFECT_IO = False  # If True, A/B tiles use perfect IO (no DAC/ADC)
COMBINED_OUT_SCALING = True  # Enable combined learnable out_scaling for LRTT output
LEARN_OUT_SCALING = True  # Enable learnable out_scaling for non-target frozen layers

# LoRA target options: which layers have trainable A/B tiles
# NOTE: ALBERT uses weight sharing - all 12 transformer blocks share the same
# parameters. The counts below are UNIQUE Linear layers, not per-block.
# ALBERT layer naming:
#   attention: query, key, value, dense (output projection)
#   FFN: ffn (intermediate), ffn_output (output)
#   embedding projection: albert.encoder.embedding_hidden_mapping_in
LORA_TARGET = "attn"  # default, can be set via --lora-target
HEAD_LAYER = "train"  # default, can be set via --head-layer (train | freeze)

# Non-target layer analog conversion: convert non-LRTT encoder layers to analog (frozen)
CONVERT_NONTARGET = True  # default on
LORA_TARGET_MODULES = {
    "none": [],            # No LRTT layers; fully digital
    "qonly": ["query"],    # Query only
    "konly": ["key"],      # Key only
    "vonly": ["value"],    # Value only
    "qkv": ["query", "key", "value"],  # Q/K/V (not output dense)
    "attn": ["attention"], # Attention (query/key/value/dense) -> LRTT
    "ffn":  ["ffn"],       # FFN (ffn/ffn_output) -> LRTT
    "all":  None,          # All encoder layers -> LRTT
}

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

# Global config (set by argparse)
OPT_CONFIG = {
    'optimizer': 'AnalogSGD',
    'tune_wd': False,        # weight_decay = 0 (fixed)
    'tune_momentum': False,  # momentum = 0 (fixed)
    'tune_nesterov': False,  # nesterov = False (fixed)
    'reinit_mode': 'decay',  # 'decay' fixed (standardized), or None = tune
    'no_transfer': False,    # If True, disable transfer (transfer_every = inf)
}


def get_study_name_suffix():
    """Generate study name suffix based on optimizer config."""
    opt = OPT_CONFIG['optimizer'].lower().replace('analog', '')
    suffix = opt

    # Add reinit_mode if fixed
    if OPT_CONFIG['reinit_mode'] is not None:
        suffix += f"_{OPT_CONFIG['reinit_mode']}"

    if not OPT_CONFIG['tune_wd']:
        suffix += "_nowd"
    if not OPT_CONFIG['tune_momentum']:
        suffix += "_nomom"
    if not OPT_CONFIG['tune_nesterov']:
        suffix += "_nonest"

    # Add transfer method if not default
    if TRANSFER_METHOD != "onehot":
        suffix += f"_{TRANSFER_METHOD}"

    if AB_DEVICE != "6t1c":
        suffix += f"_{AB_DEVICE.replace('-', '')}"

    if not IO_NOISE:
        suffix += "_noio"

    if AB_PERFECT_IO:
        suffix += "_abpio"

    if COMBINED_OUT_SCALING:
        suffix += "_combos"

    if TRAIN_CLS:
        suffix += "_traincls"
    elif TRAIN_LN:
        suffix += "_trainln"

    if WARM_ALPHA:
        suffix += "_warmalpha"

    if OPT_CONFIG['no_transfer']:
        suffix += "_notrans"

    # learn_out_scaling
    if not LEARN_OUT_SCALING:
        suffix += "_nolos"

    # Non-target analog conversion
    if CONVERT_NONTARGET:
        suffix += "_convnt"

    # Add lora target (always include for clarity)
    suffix += f"_{LORA_TARGET}"

    # Add head_layer if frozen (not default)
    if HEAD_LAYER == "freeze":
        suffix += "_headfreeze"

    return suffix

os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# LRTT Device Functions
# =============================================================================

def _create_ab_device(tau_sec=0.0):
    """Create A/B tile device based on AB_DEVICE setting.

    Options:
        6t1c - Full 6T1C with all noise/variation (realistic)
        fp   - FloatingPointDevice (perfect, no quantization/bounds)

    Args:
        tau_sec: Retention time constant. If 0, lifetime=0 (no decay).
    """
    if AB_DEVICE == "fp":
        return FloatingPointDevice()

    # Compute retention lifetime from tau_sec
    if tau_sec > 0:
        dt_batch_sec = 1.0
        delta = 1 - math.exp(-dt_batch_sec / tau_sec)
        lifetime = 1.0 / delta if delta > 0 else 0.0
    else:
        lifetime = 0.0

    # Default: 6t1c (full noise)
    return LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=0.3,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lifetime,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )


def _create_c_device():
    """Create noise-free SoftBoundsDevice for C tile."""
    return SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=False,
    )


def create_lrtt_config(rank, transfer_every, transfer_lr, lora_alpha, reinit_mode, tau_sec=0.0):
    """Create LRTT RPU configuration for analog layers."""
    ab_device = _create_ab_device(tau_sec=tau_sec)
    c_device = _create_c_device()

    te = transfer_every
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=lora_alpha,
        reinit_gain=REINIT_GAIN,
        reinit_mode=reinit_mode,
        decay_factor=DECAY_FACTOR,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = transfer_lr
    device_config.units_in_mbatch = True
    device_config.transfer_method = TRANSFER_METHOD
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = True  # LoRA mode: y = C*x + alpha*A*(B*x)
    # A/B tile perfect IO
    if AB_PERFECT_IO:
        from aihwkit.simulator.parameters.io import IOParameters
        device_config.a_forward_io = IOParameters(is_perfect=True)
        device_config.b_forward_io = IOParameters(is_perfect=True)
        device_config.a_backward_io = IOParameters(is_perfect=True)
        device_config.b_backward_io = IOParameters(is_perfect=True)
    # ALBERT has standard LayerNorm - combined_out_scaling not needed by default
    device_config.combined_out_scaling = COMBINED_OUT_SCALING

    # Dynamic TE
    device_config.dynamic_te = DYNAMIC_TE
    device_config.dynamic_te_power = DYNAMIC_TE_POWER
    device_config.dynamic_te_max = te * 20
    device_config.te_warmup_schedule = TE_WARMUP_SCHEDULE + [te]
    device_config.te_warmup_steps = TE_WARMUP_STEPS

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    # Set IO noise to 0.0
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0

    # Mapping configuration
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = LEARN_OUT_SCALING
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


def _create_nontarget_rpu_config():
    """SingleRPUConfig + SoftBoundsDevice for non-target frozen analog layers.

    Tile weights are frozen via noop update hook (see create_model).
    out_scaling is also frozen (learn_out_scaling=False).
    """
    from aihwkit.simulator.configs import SingleRPUConfig
    device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=False,
    )
    rpu_config = SingleRPUConfig(device=device)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = False  # Non-target: always frozen
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


# =============================================================================
# Model Functions
# =============================================================================

def list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def get_lrtt_target_module_names(lora_target):
    """Get module name patterns for LRTT conversion based on lora_target.

    Returns list of substrings that identify which encoder layers should be LRTT.
    Returns [] for none mode (fully digital, no LRTT layers).

    ALBERT layer naming:
        attention: query, key, value, dense (output projection)
        FFN: ffn (intermediate), ffn_output (output)
        embedding projection: albert.encoder.embedding_hidden_mapping_in
    """
    if lora_target == "none":
        return []
    elif lora_target == "qonly":
        return ["query"]
    elif lora_target == "konly":
        return ["key"]
    elif lora_target == "vonly":
        return ["value"]
    elif lora_target == "qkv":
        return ["query", "key", "value"]
    elif lora_target == "attn":
        return ["attention"]
    elif lora_target == "ffn":
        return ["ffn"]
    elif lora_target == "all":
        return None
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model(params):
    """Create ALBERT classification model with selective LRTT analog layers (2-stage).

    2-Stage Architecture:
        Stage 1: Pretrained classifier + LayerNorm loaded from checkpoint
        Stage 2: Only LoRA A/B analog tiles are trainable

        - LRTT Target layers (based on --lora-target) -> LRTT Analog (A/B trainable)
        - Non-target Encoder layers -> SingleRPU (frozen) if CONVERT_NONTARGET
        - classifier -> Digital FROZEN (pretrained from Stage 1)
        - LayerNorm -> Digital FROZEN (pretrained from Stage 1)
        - embedding_hidden_mapping_in -> Digital FROZEN
        - Embeddings -> Digital FROZEN

    LRTT layers have:
        - A/B tiles: TRAINABLE (LoRA low-rank update)
        - C-tile: FROZEN (pretrained weights)
        - combined_out_scaling: TRAINABLE (if enabled)
        - out_scaling: FROZEN
        - bias: FROZEN
    """
    from aihwkit.nn import AnalogLinear

    num_labels = TASK_TO_NUM_LABELS[TASK_NAME]
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    # Reinitialize classifier with FIXED seed for reproducibility
    if hasattr(model, 'classifier'):
        torch.manual_seed(SEED)
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
        print(f"  [FIX] Reinitialized classifier with FIXED seed={SEED}")

    # Load pretrained classifier + LayerNorm from Stage 1 checkpoint
    ckpt_path = PRETRAIN_CKPT or os.path.join(PRETRAIN_CKPT_DIR, f"{TASK_NAME}_adam_full", "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {ckpt_path}\n"
            f"Run: python /data/pretrain_classifier.py --tasks {TASK_NAME}"
        )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    pretrained = ckpt['state_dict']
    current_state = model.state_dict()
    loaded_keys = []
    for k, v in pretrained.items():
        if k in current_state:
            current_state[k] = v
            loaded_keys.append(k)
        else:
            print(f"  [WARN] Checkpoint key not in model: {k}")
    model.load_state_dict(current_state)
    print(f"  [Stage 1] Loaded {len(loaded_keys)} params from {ckpt_path}")
    print(f"  [Stage 1] {ckpt.get('metric_name','?')}: {ckpt.get('metric_value', 0):.4f} "
          f"(epoch {ckpt.get('best_epoch', '?')}, lr={ckpt.get('lr', '?')})")

    # Get LRTT target patterns
    lrtt_patterns = get_lrtt_target_module_names(LORA_TARGET)

    # Always exclude from any analog conversion
    always_digital = ["classifier", "albert.encoder.embedding_hidden_mapping_in"]

    def is_lrtt_target(layer_name):
        """Check if layer should be converted to LRTT Analog."""
        if any(d in layer_name for d in always_digital):
            return False
        if "encoder" not in layer_name:
            return False
        if lrtt_patterns is None:
            return True
        return any(p in layer_name for p in lrtt_patterns)

    # Build exclude list: all layers that should NOT be converted to LRTT
    all_linear_names = list_linear_layers(model)
    exclude_modules = []
    for name in all_linear_names:
        if not is_lrtt_target(name):
            exclude_modules.append(name)

    # Exclude always-digital layers
    exclude_modules.append("classifier")
    exclude_modules.append("albert.encoder.embedding_hidden_mapping_in")
    exclude_modules = list(set(exclude_modules))  # Remove duplicates

    # Step 1: Convert only LRTT target layers to LRTT Analog (skip if none mode)
    if LORA_TARGET == "none":
        analog_count = 0
    else:
        te = int(params["transfer_every"])
        lrtt_config = create_lrtt_config(
            rank=int(params["rank"]),
            transfer_every=te,
            transfer_lr=params["transfer_lr"],
            lora_alpha=params["lora_alpha"],
            reinit_mode=params["reinit_mode"],
            tau_sec=params["tau_sec"],
        )

        model = convert_to_analog(model, lrtt_config, exclude_modules=exclude_modules)
        analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))

    # --- Pass 2: Convert non-target encoder layers to frozen analog (Single RPU) ---
    if CONVERT_NONTARGET:
        from aihwkit.simulator.configs import SingleRPUConfig

        # Build NT encoder layer list (non-target, non-digital encoder layers)
        nt_encoder_layers = [
            n for n in all_linear_names
            if not is_lrtt_target(n) and "encoder" in n
            and not any(d in n for d in always_digital)
        ]
        nontarget_config = _create_nontarget_rpu_config()
        print(f"  [DEBUG] LEARN_OUT_SCALING={LEARN_OUT_SCALING}, nontarget learn_out_scaling={nontarget_config.mapping.learn_out_scaling}")
        exclude_pass2 = [n for n in all_linear_names if n not in nt_encoder_layers]
        if LORA_TARGET == "none":
            # No prior LRTT conversion — normal convert (no inplace needed)
            model = convert_to_analog(model, nontarget_config, exclude_modules=exclude_pass2)
        else:
            # inplace=True: skip deepcopy so LRTT sub-tiles (tile_a/b/c) and their
            # update hooks are preserved. Only NT nn.Linear layers get converted.
            model = convert_to_analog(model, nontarget_config, exclude_modules=exclude_pass2,
                                      inplace=True, ensure_analog_root=False)
        analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))

        # Freeze NT tile weights via noop update hook.
        # Use layer name (is_lrtt_target) to distinguish LRTT vs NT, because
        # analog_tiles() returns sub-tiles whose rpu_config is SingleRPUConfig
        # for both LRTT (tile_a/b/c) and NT — cannot use isinstance to distinguish.
        def _frozen_noop_update(x_input, d_input, *args, **kwargs):
            return None
        for name, m in model.named_modules():
            if isinstance(m, AnalogLinear) and not is_lrtt_target(name):
                for tile in m.analog_tiles():
                    tile.update = _frozen_noop_update

    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LRTT Analog layers: {analog_count}, Total params: {total_params:,}, Trainable (before grad set): {trainable_before:,}")

    # Step 2: Set requires_grad - Stage 2: ONLY LoRA A/B tiles trainable
    # Classifier + LayerNorm are FROZEN (pretrained from Stage 1)
    from aihwkit.optim.context import AnalogContext
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            # AnalogContext (tile_a/b/c analog_ctx): required for analog tile update
            param.requires_grad = True
        elif "combined_out_scaling" in name and (LEARN_COMBINED_OS or OS_CALIB_EPOCHS > 0):
            # combined_out_scaling: trainable (separate LR or calibration)
            param.requires_grad = True
        elif TRAIN_LN and ("LayerNorm" in name or "layer_norm" in name):
            # LayerNorm: trainable when TRAIN_LN enabled
            param.requires_grad = True
        elif TRAIN_CLS and "classifier" in name:
            # Classifier: trainable when TRAIN_CLS enabled
            param.requires_grad = True
        else:
            # ALL digital params frozen (classifier, embeddings, etc.)
            param.requires_grad = False

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable (after grad set): {trainable_after:,}")
    print(f"  LoRA target: {LORA_TARGET} -> {lrtt_patterns if lrtt_patterns else 'all encoder layers'}")

    pretrained_metric = ckpt.get('metric_value', 0.0)
    return model.to(DEVICE), pretrained_metric


# =============================================================================
# Data Functions
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize GLUE dataset."""
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)
    sentence1_key, sentence2_key = TASK_TO_KEYS[TASK_NAME]

    def preprocess(examples):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                max_length=MAX_SEQ_LENGTH, truncation=True,
            )
        return tokenizer(
            examples[sentence1_key], examples[sentence2_key],
            max_length=MAX_SEQ_LENGTH, truncation=True,
        )

    # Remove original text columns (keep only tokenizer output + label)
    remove_cols = [c for c in raw_datasets["train"].column_names if c != "label"]
    tokenized = raw_datasets.map(preprocess, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")

    # Training set
    train_dataset = tokenized["train"]
    if TRAIN_SUBSET_SIZE > 0:
        train_dataset = train_dataset.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(train_dataset)))
        )

    data_collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=data_collator,
        generator=torch.Generator().manual_seed(SEED),
    )

    # Eval set
    eval_key = "validation_matched" if TASK_NAME == "mnli" else "validation"
    eval_dataset = tokenized[eval_key]
    if EVAL_SUBSET_SIZE > 0:
        eval_dataset = eval_dataset.select(
            range(min(EVAL_SUBSET_SIZE, len(eval_dataset)))
        )

    eval_loader = DataLoader(
        eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=data_collator,
    )

    print(f"  GLUE {TASK_NAME}: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
    return train_loader, eval_loader


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(model, eval_loader):
    """Evaluate GLUE model. Returns (metric_value, avg_loss)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with no_grad():
        for batch in eval_loader:
            # Filter to only model input keys (exclude metadata like 'idx')
            model_inputs = {k: v.to(DEVICE) for k, v in batch.items()
                           if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
            outputs = model(**model_inputs)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * len(batch["labels"])

            if TASK_TO_NUM_LABELS[TASK_NAME] == 1:  # Regression (stsb)
                preds = logits.squeeze().cpu().numpy()
            else:  # Classification
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else [preds])
            all_labels.extend(batch["labels"].cpu().tolist())

    model.train()

    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

    # Compute task-specific metric
    is_regression = (TASK_NAME == "stsb")
    if is_regression:
        from scipy.stats import spearmanr
        metric_value = spearmanr(all_preds, all_labels)[0]
    elif TASK_NAME in ["mrpc", "qqp"]:
        from sklearn.metrics import f1_score
        metric_value = f1_score(all_labels, all_preds)
    elif TASK_NAME == "cola":
        from sklearn.metrics import matthews_corrcoef
        metric_value = matthews_corrcoef(all_labels, all_preds)
    else:
        # accuracy for sst2, qnli, rte, mnli
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        metric_value = correct / n_samples if n_samples > 0 else 0.0

    return metric_value, avg_loss


# =============================================================================
# Scheduler
# =============================================================================

def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_rate=0.0):
    """Linear schedule with warmup that decays to min_lr_rate (fraction of peak LR)."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Optuna Objective
# =============================================================================

def objective(trial, train_loader, eval_loader, tokenizer, study=None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metric_name = TASK_TO_METRIC[TASK_NAME]

    # Hyperparameters
    learning_rate = 0.01  # Fixed base LR
    if TRAIN_CLS:
        # Classifier+LN mode: fix lora HPs, sweep digital_lr only
        lora_alpha_val = 1.0
        target_ab_lr = 0.1
        digital_lr = trial.suggest_float('digital_lr', 1e-5, 1e-2, log=True)
        ln_lr = digital_lr  # shared LR for LN + classifier + combined_os
    elif TRAIN_LN:
        # LayerNorm mode: fix lora HPs, sweep ln_lr only
        lora_alpha_val = 1.0
        target_ab_lr = 0.01
        ln_lr = trial.suggest_float('ln_lr', 1e-5, 1e-2, log=True)
    else:
        lora_alpha_val = trial.suggest_float('lora_alpha', 0.005, 1.0, log=True)
        target_ab_lr = trial.suggest_float('target_ab_lr', 0.005, 0.2, log=True)
        ln_lr = None

    # LRTT parameters: skip sweep if --no-transfer (A/B frozen, no transfer happens)
    if OPT_CONFIG['no_transfer']:
        transfer_lr = 0.1
        transfer_every = 999999999
        rank = 4
        lora_alpha = 1.0
        lrtt_lr_multiplier = 0.01
        tau_sec = 0.0
    else:
        # LoRA mode: compute lr_multiplier from target_ab_lr
        transfer_lr = 0.1  # Fixed (not used in LoRA mode)
        transfer_every = 10000000  # Fixed (transfer disabled)
        rank = 16  # Fixed
        lora_alpha = lora_alpha_val
        lrtt_lr_multiplier = target_ab_lr / (learning_rate * lora_alpha)
        tau_sec = 0.0  # Fixed (no decay)

    min_lr_rate = 0.0

    # weight_decay: FIXED to 0
    weight_decay = 0.0

    # momentum: 0.9 fixed by default, 0.0 with --no-momentum
    if OPT_CONFIG['tune_momentum']:
        momentum = 0.9
    else:
        momentum = 0.0

    # nesterov: True fixed by default, False with --no-nesterov
    if OPT_CONFIG['tune_nesterov'] and momentum > 0:
        nesterov = True
    else:
        nesterov = False

    # reinit_mode: FIXED to 'decay' (standardized)
    reinit_mode = 'decay'

    # optimizer: always use config value
    optimizer_name = OPT_CONFIG['optimizer']

    params = {
        "rank": rank,
        "transfer_every": transfer_every,
        "transfer_lr": transfer_lr,
        "lora_alpha": lora_alpha,
        "reinit_mode": reinit_mode,
        "tau_sec": tau_sec,
    }

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting ({TASK_NAME}, metric={metric_name})")
    print(f"{'='*70}")
    print(f"  rank={rank}, lora_alpha={lora_alpha:.4f}, lr={learning_rate:.2e}")
    print(f"  target_ab_lr={target_ab_lr:.4f}, lr_multiplier={lrtt_lr_multiplier:.6f}")
    print(f"  effective_ab_lr = lr*alpha*mult = {learning_rate*lora_alpha*lrtt_lr_multiplier:.4f}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, reinit_mode={reinit_mode}")
    print(f"  optimizer={optimizer_name}")
    print(f"{'='*70}")

    model = None
    try:
        set_seed(SEED)

        model, pretrained_metric = create_model(params)

        if LORA_TARGET == "none" and not CONVERT_NONTARGET:
            # None mode without analog: use standard PyTorch optimizers
            if optimizer_name == "AnalogSGD":
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=learning_rate,
                    weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                )
        elif LORA_TARGET == "none" and CONVERT_NONTARGET:
            # None mode with frozen analog: use Analog optimizers (no LRTT LR fix needed)
            if optimizer_name == "AnalogSGD":
                optimizer = AnalogSGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
            else:
                optimizer = AnalogAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer.regroup_param_groups()
        else:
            # LRTT modes: use model.parameters() like tiki, then fix lr for LRTT tiles
            classifier_lr = learning_rate
            lrtt_lr = learning_rate * lrtt_lr_multiplier

            print(f"\nOptimizer configuration:")
            print(f"  Forward scaling (lora_alpha): {lora_alpha}")
            print(f"  Classifier LR: {classifier_lr}")
            print(f"  LRTT LR: {lrtt_lr} (= {learning_rate} * {lrtt_lr_multiplier})")
            print(f"  LRTT_LR_MULTIPLIER: {lrtt_lr_multiplier}")

            if optimizer_name == "AnalogSGD":
                optimizer = AnalogSGD(model.parameters(), lr=classifier_lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
            else:
                optimizer = AnalogAdam(model.parameters(), lr=classifier_lr, weight_decay=weight_decay)

            optimizer.regroup_param_groups()

            # Fix regroup lr: regroup resets all analog groups to defaults["lr"].
            # Restore lrtt_lr for LRTT tiles only (NT tiles keep classifier_lr, noop anyway).
            lrtt_tile_ids = set()
            for m in model.modules():
                if hasattr(m, 'tile_a'):
                    lrtt_tile_ids.add(id(m.tile_a))
                    lrtt_tile_ids.add(id(m.tile_b))
                    lrtt_tile_ids.add(id(m.tile_c))
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if hasattr(p, 'analog_tile') and id(p.analog_tile) in lrtt_tile_ids:
                        group["lr"] = lrtt_lr
                        p.analog_tile.set_learning_rate(lrtt_lr)

            # Separate digital trainable params (combined_out_scaling + LayerNorm) into own group
            from aihwkit.optim.context import AnalogContext
            digital_train_ids = set()
            for pname, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if isinstance(p, AnalogContext):
                    continue
                if "combined_out_scaling" in pname or \
                   (TRAIN_LN and ("LayerNorm" in pname or "layer_norm" in pname)) or \
                   (TRAIN_CLS and "classifier" in pname):
                    digital_train_ids.add(id(p))
            # Remove from existing groups, collect them
            digital_train_list = []
            for group in optimizer.param_groups:
                remaining = []
                for p in group["params"]:
                    if id(p) in digital_train_ids:
                        digital_train_list.append(p)
                    else:
                        remaining.append(p)
                group["params"] = remaining
            # Add as new group with shared LR
            if digital_train_list:
                digital_lr = ln_lr if ((TRAIN_LN or TRAIN_CLS) and ln_lr is not None) else COMBINED_OS_LR
                optimizer.add_param_group({"params": digital_train_list, "lr": digital_lr})
                n_os = sum(1 for n, p in model.named_parameters() if "combined_out_scaling" in n and p.requires_grad)
                n_cls = sum(1 for n, p in model.named_parameters() if "classifier" in n and p.requires_grad)
                n_ln = len(digital_train_list) - n_os - n_cls
                print(f"  Digital trainable group: {len(digital_train_list)} params (OS={n_os}, LN={n_ln}, CLS={n_cls}), LR={digital_lr}")

            # Separate LayerNorm into its own param group with swept LR (only when TRAIN_LN without TRAIN_CLS)
            if TRAIN_LN and not TRAIN_CLS and ln_lr is not None:
                ln_pids = set()
                for pname, p in model.named_parameters():
                    if ("LayerNorm" in pname or "layer_norm" in pname) and p.requires_grad:
                        ln_pids.add(id(p))
                ln_param_list = []
                for group in optimizer.param_groups:
                    remaining = []
                    for p in group["params"]:
                        if id(p) in ln_pids:
                            ln_param_list.append(p)
                        else:
                            remaining.append(p)
                    group["params"] = remaining
                if ln_param_list:
                    optimizer.add_param_group({"params": ln_param_list, "lr": ln_lr})
                print(f"  LayerNorm: {len(ln_param_list)} params, LR={ln_lr} (separate group)")

            # Count param groups for diagnostics
            from aihwkit.optim.context import AnalogContext
            n_analog = sum(1 for g in optimizer.param_groups for p in g["params"] if isinstance(p, AnalogContext))
            n_digital = sum(len(g["params"]) for g in optimizer.param_groups) - n_analog
            print(f"  Analog contexts in optimizer: {n_analog} (LRTT + NT)")
            print(f"  Digital params in optimizer: {n_digital}")

        num_training_steps = len(train_loader) * N_EPOCHS
        warmup_steps = int(WARMUP_RATIO * num_training_steps)
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
        )

        best_metric = -float('inf')
        epochs_without_improvement = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
            for batch_idx, batch in enumerate(pbar):
                # Filter to only model input keys (exclude metadata like 'idx')
                model_inputs = {k: v.to(DEVICE) for k, v in batch.items()
                               if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}

                optimizer.zero_grad()
                outputs = model(**model_inputs)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Alpha warm-up: linear 0 -> lora_alpha over warmup steps
                if WARM_ALPHA and LORA_TARGET != "none":
                    global_step = (epoch - 1) * len(train_loader) + batch_idx
                    if global_step < warmup_steps:
                        current_alpha = lora_alpha * (global_step / warmup_steps)
                    else:
                        current_alpha = lora_alpha
                    for m in model.modules():
                        if hasattr(m, 'controller') and hasattr(m, 'lora_alpha'):
                            m.lora_alpha = current_alpha
                            m.controller.lora_alpha = current_alpha

                loss_val = loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    print(f"\n  [NaN/Inf detected at batch {num_batches}] Aborting trial.")
                    return 0.0
                total_loss += loss_val
                num_batches += 1
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            eval_metric, eval_loss = evaluate_model(model, eval_loader)

            improved = ""
            if eval_metric > best_metric:
                best_metric = eval_metric
                epochs_without_improvement = 0
                improved = " *"
            else:
                epochs_without_improvement += 1

            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"[Trial {trial.number}] Epoch {epoch:3d} | "
                  f"{metric_name}: {eval_metric:.4f} | Best: {best_metric:.4f} | "
                  f"Loss: {train_loss:.4f} | LR: {current_lr:.2e} | "
                  f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}{improved}")

            trial.report(best_metric, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

            # Combined out-scaling calibration: freeze after OS_CALIB_EPOCHS
            if OS_CALIB_EPOCHS > 0 and epoch == OS_CALIB_EPOCHS:
                frozen_os = 0
                for pname, p in model.named_parameters():
                    if "combined_out_scaling" in pname and p.requires_grad:
                        p.requires_grad = False
                        frozen_os += 1
                tqdm.write(f"[Trial {trial.number}] Combined out-scaling frozen after epoch {epoch} "
                           f"({frozen_os} params)")

            # Pretrained baseline check: if epoch 3 metric < Stage 1 pretrained → stop
            if epoch == 3 and best_metric < pretrained_metric:
                tqdm.write(f"[Trial {trial.number}] Stopped at epoch 3: "
                           f"{best_metric:.4f} < pretrained {pretrained_metric:.4f}")
                raise optuna.exceptions.TrialPruned()

            if epoch > 3 and epochs_without_improvement >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break

            # Cross-trial pruning: compare against best trial's metric at same epoch (from epoch 3)
            # If current trial is worse at this epoch than the best trial was → prune
            if study is not None and epoch >= 3:
                completed = [t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE]
                if completed:
                    # Get best trial's metric at this epoch (from intermediate values)
                    best_trial = study.best_trial
                    best_at_epoch = best_trial.intermediate_values.get(epoch)
                    if best_at_epoch is not None and best_metric < best_at_epoch:
                        tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}: "
                                   f"{best_metric:.4f} < best_trial_ep{epoch} {best_at_epoch:.4f}")
                        raise optuna.exceptions.TrialPruned()

            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        print(f"\n[Trial {trial.number}] Finished - Best {metric_name}: {best_metric:.4f}")
        print(f"{'='*70}\n")
        return best_metric

    except Exception as e:
        error_msg = str(e)[:500]
        trial.set_user_attr("error", error_msg)
        print(f"[Trial {trial.number}] Error: {error_msg}")
        raise

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[Trial {trial.number}] GPU cache cleared")


# =============================================================================
# Visualization
# =============================================================================

def visualize_study(study, save_dir):
    """Visualize optimization history, parameter importance, and LR vs metric."""
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        print("No completed trials to visualize.")
        return

    metric_name = TASK_TO_METRIC[TASK_NAME]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    trial_numbers = [t.number for t in complete_trials]
    scores = [t.value for t in complete_trials]

    axes[0].scatter(trial_numbers, scores, alpha=0.6)
    axes[0].plot(trial_numbers,
                 [max(scores[:i+1]) for i in range(len(scores))],
                 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel(metric_name)
    axes[0].set_title('Optimization History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    try:
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys())[::-1], list(importances.values())[::-1])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Parameter Importance')
    except Exception:
        axes[1].text(0.5, 0.5, 'Not enough trials', ha='center', va='center',
                     transform=axes[1].transAxes)

    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[2].scatter(lrs, scores, alpha=0.6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel(metric_name)
    axes[2].set_title(f'Learning Rate vs {metric_name}')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"visualization_{TASK_NAME}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualization saved.")


def print_study_summary(study):
    """Print study summary."""
    metric_name = TASK_TO_METRIC[TASK_NAME]
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Study: {study.study_name}, Task: {TASK_NAME}, Trials: {len(study.trials)} ({len(complete_trials)} complete)")
    if complete_trials:
        scores = [t.value for t in complete_trials]
        print(f"Best {metric_name}: {max(scores):.4f}, Mean: {sum(scores)/len(scores):.4f}")
        print(f"Best params: {study.best_params}")


# =============================================================================
# Main
# =============================================================================

def main():
    global TASK_NAME, BATCH_SIZE, N_EPOCHS, MAX_SEQ_LENGTH, WARMUP_RATIO, WARM_ALPHA, TRANSFER_METHOD, AB_DEVICE, IO_NOISE, AB_PERFECT_IO, COMBINED_OUT_SCALING, LORA_TARGET, HEAD_LAYER, RESULTS, CONVERT_NONTARGET, PRETRAIN_CKPT, OS_CALIB_EPOCHS, LEARN_COMBINED_OS, COMBINED_OS_LR, TRAIN_LN, TRAIN_CLS

    parser = argparse.ArgumentParser(description="2-Stage Optuna sweep for ALBERT GLUE LoRA-LRTT")
    parser.add_argument('--task', type=str, default=TASK_NAME,
                        choices=GLUE_TASKS,
                        help=f'GLUE task (default: {TASK_NAME})')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: auto-generated based on config)')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--optimizer', type=str, default='AnalogSGD',
                        choices=['AnalogSGD', 'AnalogAdam'],
                        help='Optimizer type (default: AnalogSGD)')
    parser.add_argument('--no-wd', action='store_true', default=True,
                        help='Disable weight decay tuning (fix to 0)')
    parser.add_argument('--no-momentum', action='store_true', default=True,
                        help='Disable momentum tuning (fix to 0, SGD only)')
    parser.add_argument('--no-nesterov', action='store_true', default=True,
                        help='Disable nesterov tuning (fix to False, SGD only)')
    parser.add_argument('--reinit-mode', type=str, default='decay',
                        choices=['standard', 'decay', 'hybrid'],
                        help='Fix reinit mode (default: decay)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: per-task from TASK_CONFIGS)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (default: per-task from TASK_CONFIGS)')
    parser.add_argument('--warmup-ratio', type=float, default=WARMUP_RATIO,
                        help=f'LR warmup ratio (default: {WARMUP_RATIO})')
    parser.add_argument('--transfer-method', type=str, default=TRANSFER_METHOD,
                        choices=['onehot', 'direct', 'set'],
                        help=f'Transfer method (default: {TRANSFER_METHOD})')
    parser.add_argument('--ab-device', type=str, default=AB_DEVICE,
                        choices=['6t1c', 'fp'],
                        help=f'A/B tile device type (default: {AB_DEVICE})')
    parser.add_argument('--no-io-noise', action='store_true',
                        help='Disable IO out_noise (resolution kept)')
    parser.add_argument('--ab-perfect-io', action='store_true',
                        help='Use perfect IO (no DAC/ADC) for A/B tiles')
    parser.add_argument('--combined-out-scaling', action='store_true', default=True,
                        help='Use combined learnable out_scaling for full LRTT output')
    parser.add_argument('--no-combined-out-scaling', dest='combined_out_scaling', action='store_false',
                        help='Disable combined learnable out_scaling')
    parser.add_argument('--no-transfer', action='store_true',
                        help='Disable transfer (set transfer_every to infinity)')
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=['none', 'qonly', 'konly', 'vonly', 'qkv', 'attn', 'ffn', 'all'],
                        help='LoRA target: none, qonly, konly, vonly, qkv, attn, ffn, all (default: attn)')
    # Note: --head-layer removed for 2-stage (classifier always frozen from Stage 1)
    parser.add_argument('--warm-alpha', action='store_true',
                        help='Enable warm-up for lora_alpha (linear: 0 -> target over warmup steps)')
    parser.add_argument('--convert-nontarget', action='store_true', default=True,
                        help='Convert non-target layers to analog (SingleRPU+SoftBounds, frozen)')
    parser.add_argument('--no-convert-nontarget', dest='convert_nontarget', action='store_false',
                        help='Disable non-target layer analog conversion')
    parser.add_argument('--no-learn-out-scaling', dest='learn_out_scaling', action='store_false', default=True,
                        help='Disable learnable out_scaling for non-target frozen layers')
    parser.add_argument('--pretrain-ckpt', type=str, default=None,
                        help='Path to Stage 1 checkpoint (default: /data/classifier_ckpt/{task}/ckpt.pt)')
    parser.add_argument('--os-calib-epochs', type=int, default=0,
                        help='Out-scaling calibration epochs: train out_scaling for N epochs then freeze (0=disabled)')
    parser.add_argument('--train-ln', action='store_true',
                        help='Unfreeze LayerNorm in Stage 2 (train with optimizer LR)')
    parser.add_argument('--train-cls', action='store_true',
                        help='Unfreeze classifier+LN in Stage 2 (train with swept digital_lr)')
    args = parser.parse_args()

    # Update global config
    global LEARN_OUT_SCALING
    WARM_ALPHA = args.warm_alpha
    TASK_NAME = args.task
    RESULTS = f"/data/results/Analoglora_2stage/{TASK_NAME}"
    os.makedirs(RESULTS, exist_ok=True)

    # Apply per-task defaults from TASK_CONFIGS, CLI overrides take priority
    task_cfg = TASK_CONFIGS.get(TASK_NAME, {})
    BATCH_SIZE = args.batch_size if args.batch_size != 32 else task_cfg.get("batch_size", 32)

    # Stage 2 epochs from STAGE0_CONFIGS (or override with --epochs)
    s0_cfg = STAGE0_CONFIGS.get(TASK_NAME, {})
    if args.epochs != 20:
        N_EPOCHS = args.epochs
    else:
        N_EPOCHS = s0_cfg.get("stage1_epochs", task_cfg.get("epochs", 20))

    MAX_SEQ_LENGTH = task_cfg.get("max_seq_length", 128)
    WARMUP_RATIO = args.warmup_ratio
    PRETRAIN_CKPT = args.pretrain_ckpt
    OS_CALIB_EPOCHS = args.os_calib_epochs
    TRAIN_LN = args.train_ln or args.train_cls  # train_cls implies train_ln
    TRAIN_CLS = args.train_cls

    stage0_steps = s0_cfg.get('stage0_steps', 0)
    total_2x = s0_cfg.get('total_steps_2x', 0)
    print(f"[2-Stage] stage0={stage0_steps} steps, "
          f"stage1={total_2x - stage0_steps} steps -> {N_EPOCHS} epochs")
    TRANSFER_METHOD = args.transfer_method
    AB_DEVICE = args.ab_device
    IO_NOISE = not args.no_io_noise
    AB_PERFECT_IO = args.ab_perfect_io
    COMBINED_OUT_SCALING = args.combined_out_scaling
    CONVERT_NONTARGET = args.convert_nontarget
    LEARN_OUT_SCALING = args.learn_out_scaling
    LORA_TARGET = args.lora_target
    HEAD_LAYER = "freeze"  # 2-stage: classifier always frozen (pretrained from Stage 1)
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['reinit_mode'] = args.reinit_mode
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov
    OPT_CONFIG['no_transfer'] = args.no_transfer

    # Auto-generate study name based on config (includes task and batch size)
    study_name = args.study_name or f"albert_{TASK_NAME}_lora2s_bs{BATCH_SIZE}_{get_study_name_suffix()}"

    storage = f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    # Load data once (shared across all trials)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_data(tokenizer)
    print(f"Task: {TASK_NAME}, Metric: {TASK_TO_METRIC[TASK_NAME]}")
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # TPESampler: Bayesian optimization
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )

    # Enqueue seed trial
    if TRAIN_CLS:
        study.enqueue_trial({'digital_lr': 1e-3})
    elif TRAIN_LN:
        study.enqueue_trial({'ln_lr': 1e-3})
    else:
        study.enqueue_trial({
            'lora_alpha': 1.0,
            'target_ab_lr': 0.01,
        })

    print(f"\nStudy: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

    # Run trials with OOM recovery via process restart
    target_total = len(study.trials) + args.n_trials

    try:
        study.optimize(
            lambda trial: objective(trial, train_loader, eval_loader, tokenizer, study=study),
            n_trials=args.n_trials,
            catch=(Exception,),
            show_progress_bar=False,
            callbacks=[_oom_restart_callback],
        )
    except _OOMRestart:
        remaining = target_total - len(study.trials)
        if remaining > 0:
            print(f"\n[OOM Recovery] Restarting process for {remaining} remaining trials...")
            new_argv = list(sys.argv)
            for i, arg in enumerate(new_argv):
                if arg == '--n-trials' and i + 1 < len(new_argv):
                    new_argv[i + 1] = str(remaining)
                    break
            os.execv(sys.executable, [sys.executable] + new_argv)

    print_study_summary(study)
    visualize_study(study, RESULTS)

    # Save best params
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if complete_trials:
        best_params_file = os.path.join(RESULTS, f"best_params_{study_name}.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                "task": TASK_NAME,
                "metric": TASK_TO_METRIC[TASK_NAME],
                "best_value": study.best_value,
                "best_params": study.best_params,
            }, f, indent=2)
        print(f"Best params saved to: {best_params_file}")

    # Save all trials
    all_trials = []
    for t in study.trials:
        all_trials.append({
            "trial": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })
    all_trials.sort(key=lambda x: x["value"] if x["value"] is not None else -float('inf'), reverse=True)

    all_trials_file = os.path.join(RESULTS, f"all_trials_{TASK_NAME}.json")
    with open(all_trials_file, 'w') as f:
        json.dump(all_trials, f, indent=2)
    print(f"All trials saved to: {all_trials_file}")


class _OOMRestart(Exception):
    """Raised to trigger process restart after OOM."""
    pass


def _oom_restart_callback(study, trial):
    """Optuna callback: if trial failed with OOM/CUBLAS, raise to restart process."""
    if trial.state == TrialState.FAIL:
        err = trial.user_attrs.get("error", "")
        if "out of memory" in err.lower() or "cublas" in err.lower():
            print(f"\n[OOM Recovery] Trial {trial.number} failed with CUDA error, will restart process.")
            raise _OOMRestart()


if __name__ == "__main__":
    main()
