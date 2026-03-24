# -*- coding: utf-8 -*-
"""BERT + GLUE with LRTT (Low-Rank TikiTaka Training).

Single-run training script for BERT on GLUE tasks using LRTT analog layers.
Converts Q/K/V attention layers to analog; all other layers remain digital.

Supported GLUE tasks: cola, sst2, mrpc, qqp, mnli, qnli, rte, stsb, wnli

Usage:
    python fine_bert_glue_lrtt.py --task sst2
    python fine_bert_glue_lrtt.py --task mrpc
    python fine_bert_glue_lrtt.py --task stsb

Inline flags (edit directly in script):
    N_EPOCHS = 3                     # Number of training epochs
    BATCH_SIZE = 64                 # Training batch size
    LEARNING_RATE = 1.0             # Peak learning rate
    WEIGHT_DECAY = 0.0              # Weight decay
    WARMUP_STEPS = 189              # LR scheduler warmup steps (~6% of 3 epochs)
    MIN_LR_RATE = 0.0               # Min LR as fraction of peak (0 = decay to zero)
    OPTIMIZER = "AnalogSGD"         # "AnalogSGD" | "AnalogAdam"
    LRTT_RANK = 8                   # LoRA rank for LRTT
    TRANSFER_EVERY = 1000           # Transfer interval (steps)
    TRANSFER_LR = 0.00115           # Transfer learning rate
    TRANSFER_METHOD = "onehot"      # Transfer method: "onehot" | "direct" | "set"
    FAST_LR = 1.0                   # Fast LR for A/B updates
    AUTO_SCALE_MODE = "none"        # Auto-scale: "none" | "shared" | "separate"
    REINIT_MODE = "hybrid"          # Reinit mode: "standard" | "decay" | "hybrid"
    REINIT_GAIN = 1.0               # Reinitialization gain
    TAU_SEC = 0.0                   # 6T1C retention (0 = no decay)
    DYNAMIC_TE = False              # Enable dynamic transfer every
    DYNAMIC_TE_POWER = 1.0          # Power for dynamic TE scaling
    TE_WARMUP_STEPS = 0            # Steps before reaching target TE
    TE_WARMUP_SCHEDULE = []         # Warmup TE schedule list
    TARGET_MODULES = [...]          # Modules to convert to analog
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import math
import gc
import argparse

import json

import torch
from torch import nn, no_grad, manual_seed, save
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import aihwkit.optim.lrtt_grad_accum_patch  # noqa: F401  — per-micro-batch tile.update + LRTT A/B snapshot

from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, FloatingPointDevice, IdealDevice, ConstantStepDevice
from aihwkit.simulator.configs import SingleRPUConfig

# LRTT config imports (direct imports to avoid __init__.py dependency issues)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.parameters.mapping import MappingParameter


# =============================================================================
# GLUE Task Configurations
# =============================================================================

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

TASK_TO_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
    "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1, "wnli": 2,
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
    "wnli": "accuracy",
}


# =============================================================================
# Task Selection (argparse)
# =============================================================================

_parser = argparse.ArgumentParser(description="BERT GLUE LRTT fine-tuning")
_parser.add_argument('--task', type=str, default='sst2',
                     choices=list(TASK_TO_KEYS.keys()),
                     help='GLUE task name (default: sst2)')
_parser.add_argument('--grad-accum-steps', type=int, default=1,
                     help='Gradient accumulation steps (default: 1)')
_args, _ = _parser.parse_known_args()
TASK_NAME = _args.task


# =============================================================================
# Global Constants
# =============================================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = os.path.join(os.getcwd(), "results", "BERT_GLUE_LRTT_FINE")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, f"fine_bert_glue_lrtt_{TASK_NAME}_model_weight.pth")

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 128
NUM_LABELS = TASK_TO_NUM_LABELS[TASK_NAME]

# Training
N_EPOCHS = 15
SCHEDULE_EPOCHS = 0  # LR schedule horizon (0 = use N_EPOCHS; set > N_EPOCHS to match longer runs)
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 0.9348960119904873
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 3
VAL_LOSS_EARLY_STOP_PATIENCE = 2  # Stop if val loss doesn't improve for this many epochs
VAL_LOSS_THRESHOLD = 1.5  # Once val loss drops below this, rely on metric-based early stop only

# Scheduler
WARMUP_STEPS = 189  # ~6% of total steps (3 epochs)
MIN_LR_RATE = 0.0  # Fraction of peak LR (0 = decay to zero)

# Optimizer
OPTIMIZER = "AnalogSGD"  # "AnalogSGD" or "AnalogAdam"

# LRTT parameters
LRTT_RANK = 32
TRANSFER_EVERY = 1092
TRANSFER_LR = 4.706173285862282e-05
FAST_LR = 0.11465313432104135
AUTO_SCALE_MODE = "none"  # Auto-scale mode: "none", "shared", or "separate"
CORRECT_GRADIENT_MAGNITUDES = False  # Correct transfer magnitude by dividing by effective A/B LR
REINIT_MODE = "hybrid"
REINIT_GAIN = 1.0
TRANSFER_METHOD = "set"  # "onehot", "direct", or "set"
C_DW_MIN = 0.001         # C tile dw_min (relevant for onehot/direct transfer)
C_DESIRED_BL = 31        # C tile desired_bl (relevant for onehot/direct transfer)
AB_DW_MIN = 0.001981        # A/B tile dw_min
AB_DESIRED_BL = 31          # A/B tile desired_bl

# Device selection
AB_DEVICE = "6t1c"          # "6t1c", "linearstep", "linearstepideal", "constantstep", "constantstepideal", "fp", "ideal"
C_DEVICE = "softboundsideal"  # "softboundsideal", "linearstepideal", "constantstep", "constantstepideal", "ideal"

# IO / noise options
IO_NOISE = True             # If False, disable out_noise (resolution kept)
FORWARD_INJECT = False      # If True, enable forward noise injection
FI_CONTINUOUS_ALPHA = False # If True, use continuous alpha for forward injection
IS_PERFECT = False          # If True, forward/backward use ideal FP matmul (no ADC/DAC/noise)
NO_QUANT = False            # If True, disable DAC/ADC quantization (inp_res/out_res → -1)
OUT_NOISE = 0.0             # Forward out_noise value
AB_WEIGHT_SCALING_OMEGA = 0.0  # A/B tile weight scaling omega

# Pulse type
AB_PULSE_TYPE = "default"   # "default", "none", "none_with_device", "stochastic_compressed", "mean_count", "deterministic_implicit"

# Transfer options
NO_TRANSFER = False         # If True, disable transfer (set transfer_every to infinity)
NO_SCALE_TRANSFER_LR = False  # If True, disable transfer_lr scaling by rank
TRANSFER_RANK_SCHEDULE = "all"  # "all" or "round_robin"
TRANSFER_RANKS_PER_STEP = 1

# 6T1C Retention parameters
TAU_SEC = 0.0  # 0 = no decay, >0 = retention time constant

# A/B projection IO
NO_ADC_AB_PROJ = True  # If True, remove ADC between A/B projections
LEARN_OUT_SCALING = True  # If True, C tile out_scaling is trainable

# Dynamic TE (transfer every) parameters
DYNAMIC_TE = False
DYNAMIC_TE_POWER = 1.0
TE_WARMUP_STEPS = 0
TE_WARMUP_SCHEDULE = []

# LoRA target options: which layers have trainable A/B tiles
# - none: no LRTT layers (fully digital baseline)
# - qkv: only query, key, value
# - ffn: projection (attention.output) + FFN (intermediate, output, bottleneck)
# - all: all encoder linear layers
LORA_TARGET = "qkv"  # default
HEAD_LAYER = "train"  # "train" or "freeze" for classifier layer
ENCODER_ANALOG = False  # If True, non-LRTT encoder layers become frozen analog instead of digital
HEAD_ANALOG = False  # If True, classifier → frozen analog instead of digital
BACKWARD_OUT_BOUND = 12.0  # Backward pass output bound (default 12.0)
LORA_TARGET_MODULES = {
    "none": [],  # Empty = no layers converted to LRTT (fully digital)
    "qonly": ["query"],  # Query only (12 layers)
    "konly": ["key"],  # Key only (12 layers)
    "vonly": ["value"],  # Value only (12 layers)
    "qkv": ["query", "key", "value"],  # Q/K/V (36 layers)
    "qkvo": ["query", "key", "value", "attention.output"],  # Q/K/V + attention output (48 layers)
    "ffn": (["intermediate", "output.dense"], ["attention"]),  # FFN only (24 layers)
    "dense": ["dense"],  # All layers with "dense" (excludes qkv) (36 layers)
    "allnobn": None,  # Same as all (no bottleneck in BERT) (72 layers)
    "all": None,  # None means all encoder layers (no filtering) (72 layers)
}

# Diagnostic
ENABLE_DIAGNOSTIC = True   # False = no diagnostic overhead, fast training
DIAG_EPOCHS = 0            # 0 = all epochs, N = first N epochs only

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0
GRAD_ACCUM_STEPS = _args.grad_accum_steps

# WandB
WANDB_PROJECT = f"bert-{TASK_NAME}-lrtt-fine"
os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# LRTT Device Functions
# =============================================================================

def _create_ab_device(tau_sec=None, dw_min=None):
    """Create A/B tile device based on AB_DEVICE setting.

    Options:
        6t1c              - Full 6T1C with all noise/variation (realistic)
        linearstep        - LinearStepDevice with default params (no nonlinearity, default noise)
        linearstepideal   - LinearStepDevice with all noise/dtod=0, w_max=1, w_min=-1
        constantstep      - ConstantStepDevice with default params (constant step, default noise)
        constantstepideal - ConstantStepDevice with all noise/dtod=0, w_max=1, w_min=-1
        fp                - FloatingPointDevice (perfect, no quantization/bounds)
        ideal             - IdealDevice
    """
    if tau_sec is None:
        tau_sec = TAU_SEC
    if dw_min is None:
        dw_min = AB_DW_MIN

    if AB_DEVICE == "fp":
        return FloatingPointDevice()
    if AB_DEVICE == "ideal":
        return IdealDevice()
    if AB_DEVICE == "linearstep":
        return LinearStepDevice(dw_min=dw_min)
    if AB_DEVICE == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=1.0, w_min=-1.0,
            dw_min_dtod=0.0, dw_min_std=0.0,
            up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            gamma_up_dtod=0.0, gamma_down_dtod=0.0,
            write_noise_std=0.0, reset_std=0.0,
            up_down=0.0, mult_noise=False,
        )
    if AB_DEVICE == "constantstep":
        return ConstantStepDevice(dw_min=dw_min)
    if AB_DEVICE == "constantstepideal":
        return ConstantStepDevice(
            dw_min=dw_min,
            w_max=1.0, w_min=-1.0,
            dw_min_dtod=0.0, dw_min_std=0.0,
            up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            reset_std=0.0, up_down=0.0,
        )

    # Default: 6t1c (full noise)
    if tau_sec > 0:
        dt_batch_sec = 1.0
        delta = 1 - math.exp(-dt_batch_sec / tau_sec)
        lifetime = 1.0 / delta if delta > 0 else 0.0
    else:
        lifetime = 0.0

    return LinearStepDevice(
        dw_min=dw_min,
        up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lifetime, lifetime_dtod=0.0,
        reset=0.0, reset_dtod=0.0,
    )


def _create_c_device(dw_min=None):
    """Create device for C tile based on C_DEVICE setting."""
    if dw_min is None:
        dw_min = C_DW_MIN

    if C_DEVICE == "ideal":
        return IdealDevice()
    if C_DEVICE == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=1.0, w_min=-1.0,
            dw_min_dtod=0.0, dw_min_std=0.0,
            up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            gamma_up_dtod=0.0, gamma_down_dtod=0.0,
            write_noise_std=0.0, reset_std=0.0,
            up_down=0.0, mult_noise=False,
        )
    if C_DEVICE == "constantstep":
        return ConstantStepDevice(dw_min=dw_min)
    if C_DEVICE == "constantstepideal":
        return ConstantStepDevice(
            dw_min=dw_min,
            w_max=1.0, w_min=-1.0,
            dw_min_dtod=0.0, dw_min_std=0.0,
            up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            reset_std=0.0, up_down=0.0,
        )
    # Default: softboundsideal
    return SoftBoundsDevice(
        dw_min=dw_min,
        w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0,
        up_down=0.0, up_down_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, reset_std=0.0,
        mult_noise=False,
    )


def create_frozen_analog_config(lrtt_config=None, out_noise=0.0):
    """Create analog config for non-LRTT encoder layers (frozen analog).

    If lrtt_config is provided, derived from its C tile settings.
    Otherwise, creates a standalone config with default C tile settings.
    """
    from copy import deepcopy
    if lrtt_config is not None:
        rpu_config = SingleRPUConfig(
            device=deepcopy(lrtt_config.device.unit_cell_devices[2]),
        )
        rpu_config.mapping = deepcopy(lrtt_config.device.mapping_c)
        rpu_config.forward = deepcopy(lrtt_config.forward)
        rpu_config.backward = deepcopy(lrtt_config.backward)
    else:
        rpu_config = SingleRPUConfig(device=_create_c_device())
        rpu_config.mapping = MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=LEARN_OUT_SCALING,
            out_scaling_columnwise=True,
        )
        rpu_config.forward.out_noise = out_noise
        rpu_config.backward.out_noise = out_noise
        rpu_config.forward.is_perfect = IS_PERFECT
        rpu_config.backward.is_perfect = IS_PERFECT
        if NO_QUANT:
            rpu_config.forward.inp_res = -1
            rpu_config.forward.out_res = -1
            rpu_config.backward.inp_res = -1
            rpu_config.backward.out_res = -1
        if BACKWARD_OUT_BOUND != 12.0:
            rpu_config.backward.out_bound = BACKWARD_OUT_BOUND
    return rpu_config


def create_lrtt_config():
    """Create LRTT RPU configuration for analog layers."""
    ab_device = _create_ab_device()
    c_device = _create_c_device()

    te = TRANSFER_EVERY if not NO_TRANSFER else 10 ** 9
    device_config = PythonLRTTDevice(
        rank=LRTT_RANK,
        transfer_every=te,
        lora_alpha=1.0,
        fast_lr=FAST_LR,
        reinit_gain=REINIT_GAIN,
        reinit_mode=REINIT_MODE,
        unit_cell_devices=[ab_device, ab_device, c_device],
        train_c_bias=False,        # C tile bias frozen
        mapping_ab=MappingParameter(
            weight_scaling_omega=AB_WEIGHT_SCALING_OMEGA,
            learn_out_scaling=False,
            max_input_size=0 if IS_PERFECT else 512,
            max_output_size=0 if IS_PERFECT else 512,
        ),
        mapping_c=MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=LEARN_OUT_SCALING,
            out_scaling_columnwise=True,
            max_input_size=0 if IS_PERFECT else 512,
            max_output_size=0 if IS_PERFECT else 512,
        ),
    )
    device_config.transfer_lr = TRANSFER_LR
    device_config.units_in_mbatch = True
    device_config.transfer_method = TRANSFER_METHOD
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = FORWARD_INJECT
    device_config.fi_continuous_alpha = FI_CONTINUOUS_ALPHA
    device_config.no_adc_ab_projection = NO_ADC_AB_PROJ
    device_config.c_desired_bl = C_DESIRED_BL
    device_config.auto_scale_mode = AUTO_SCALE_MODE
    device_config.correct_gradient_magnitudes = CORRECT_GRADIENT_MAGNITUDES
    device_config.scale_transfer_lr = not NO_SCALE_TRANSFER_LR
    device_config.ab_pulse_type = AB_PULSE_TYPE
    device_config.transfer_rank_schedule = TRANSFER_RANK_SCHEDULE
    device_config.transfer_ranks_per_step = TRANSFER_RANKS_PER_STEP

    # Dynamic TE: increase TE as LR decays
    device_config.dynamic_te = DYNAMIC_TE
    device_config.dynamic_te_power = DYNAMIC_TE_POWER
    device_config.dynamic_te_max = te * 20
    device_config.te_warmup_schedule = TE_WARMUP_SCHEDULE + [te]
    device_config.te_warmup_steps = TE_WARMUP_STEPS

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.update.desired_bl = AB_DESIRED_BL

    out_noise = OUT_NOISE if IO_NOISE else 0.0
    rpu_config.forward.out_noise = out_noise
    rpu_config.backward.out_noise = out_noise
    rpu_config.forward.is_perfect = IS_PERFECT
    rpu_config.backward.is_perfect = IS_PERFECT
    if NO_QUANT:
        rpu_config.forward.inp_res = -1
        rpu_config.forward.out_res = -1
        rpu_config.backward.inp_res = -1
        rpu_config.backward.out_res = -1

    if BACKWARD_OUT_BOUND != 12.0:
        rpu_config.backward.out_bound = BACKWARD_OUT_BOUND

    return rpu_config


# =============================================================================
# Model Functions
# =============================================================================

def list_linear_layers(model):
    """List all linear layer names in the model."""
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_analog_layers(model):
    """Count analog layers in the model."""
    from aihwkit.nn import AnalogLinear
    return sum(1 for m in model.modules() if isinstance(m, AnalogLinear))


def get_lrtt_target_module_names(lora_target):
    """Get module name patterns for LRTT conversion based on lora_target.

    Returns list of substrings that identify which encoder layers should be LRTT.
    Returns [] for none mode (fully digital, no LRTT layers).
    """
    if lora_target == "none":
        return []  # Empty = no layers converted to LRTT (fully digital baseline)
    elif lora_target == "qonly":
        return ["query"]  # Query only (12 layers)
    elif lora_target == "konly":
        return ["key"]  # Key only (12 layers)
    elif lora_target == "vonly":
        return ["value"]  # Value only (12 layers)
    elif lora_target == "qkv":
        return ["query", "key", "value"]  # Q/K/V (36 layers)
    elif lora_target == "qkvo":
        return ["query", "key", "value", "attention.output"]  # Q/K/V + attention output (48 layers)
    elif lora_target == "ffn":
        return (["intermediate", "output.dense"], ["attention"])  # FFN only (24 layers)
    elif lora_target == "dense":
        return ["dense"]  # All layers with "dense" in name (excludes qkv) (36 layers)
    elif lora_target == "allnobn":
        return None  # Same as all (no bottleneck in BERT) (72 layers)
    elif lora_target == "all":
        # All encoder linear layers (exclude embeddings, classifier, pooler)
        return None  # None means all encoder layers (72 layers)
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model():
    """Create BERT classification model with selective LRTT analog layers.

    Architecture (follows paper's approach for efficiency):
        - LRTT Target layers (based on LORA_TARGET) → LRTT Analog
        - Non-target Encoder layers → Digital FROZEN
        - classifier → Digital TRAINABLE (weight + bias)
        - pooler → Digital FROZEN
        - Embeddings → Digital FROZEN

    LoRA Target Options (LORA_TARGET):
        - qkv: Q/K/V layers → LRTT Analog (36 layers)
        - ffn: FFN layers → LRTT Analog (24 layers)
        - all: all encoder layers → LRTT Analog (72 layers)

    LRTT layers have:
        - A/B tiles: TRAINABLE
        - C-tile: FROZEN (pretrained weights)
        - out_scaling: TRAINABLE
        - bias: FROZEN
    """
    from aihwkit.nn import AnalogLinear

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # Get LRTT target patterns
    lrtt_patterns = get_lrtt_target_module_names(LORA_TARGET)

    def is_lrtt_target(layer_name):
        """Check if layer should be converted to LRTT Analog."""
        # classifier is always digital
        if "classifier" in layer_name:
            return False
        # pooler: always digital frozen
        if "pooler" in layer_name:
            return False
        # Must be in encoder for other layers
        if "encoder" not in layer_name:
            return False
        # If lrtt_patterns is None (all mode), all encoder layers are targets
        if lrtt_patterns is None:
            return True
        if isinstance(lrtt_patterns, tuple):
            include, exclude = lrtt_patterns
            included = True if include is None else any(p in layer_name for p in include)
            return included and not any(p in layer_name for p in exclude)
        return any(p in layer_name for p in lrtt_patterns)

    # Build exclude list: all layers that should NOT be converted to LRTT
    all_linear_names = list_linear_layers(model)
    exclude_modules = []
    for name in all_linear_names:
        if not is_lrtt_target(name):
            # Use full path for exclude_modules (convert_to_analog requires exact match)
            exclude_modules.append(name)

    # Exclude classifier and pooler (always digital)
    exclude_modules.append("classifier")
    exclude_modules.append("bert.pooler.dense")
    exclude_modules = list(set(exclude_modules))  # Remove duplicates

    # Step 1: Convert only LRTT target layers to LRTT Analog (skip if none mode)
    if LORA_TARGET == "none":
        # None mode: fully digital, no analog conversion
        num_analog = 0
    else:
        lrtt_config = create_lrtt_config()
        model = convert_to_analog(model, lrtt_config, exclude_modules=exclude_modules)

        # Count analog layers
        num_analog = count_analog_layers(model)

    # Step 1.5: Convert remaining encoder layers to frozen analog (if enabled)
    frozen_analog_count = 0
    any_frozen_analog = (ENCODER_ANALOG and LORA_TARGET != "all") or HEAD_ANALOG
    if any_frozen_analog:
        # Collect existing tile IDs (LRTT sub-tiles) before frozen conversion
        existing_tile_ids = set()
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    existing_tile_ids.add(id(tile))

        frozen_config = create_frozen_analog_config(
            lrtt_config if LORA_TARGET != "none" else None,
        )
        frozen_exclude = []
        if not HEAD_ANALOG:
            frozen_exclude.append("classifier")
        # Always exclude pooler from frozen analog conversion
        frozen_exclude.append("bert.pooler.dense")
        if not ENCODER_ANALOG or LORA_TARGET == "all":
            for name in all_linear_names:
                if "encoder" in name:
                    frozen_exclude.append(name)
        model = convert_to_analog(model, frozen_config, exclude_modules=frozen_exclude)
        frozen_analog_count = count_analog_layers(model) - num_analog

        # Hook frozen analog tile updates to no-op (prevent optimizer from modifying weights).
        # AnalogSGD/Adam calls tile.update() on ALL analog tiles unconditionally;
        # LRTT tiles are already hooked in lrtt_tile.py, but frozen analog tiles need this.
        def _frozen_noop_update(x_input, d_input, *args, **kwargs):
            return None

        # Bypass AnalogFunction.apply() for frozen tiles to avoid activation/gradient
        # saving overhead. AnalogFunction saves input 3x (autograd ctx, analog_input list,
        # analog_grad_output list) for weight updates that never happen on frozen tiles.
        # This custom Function matches LRTT C tile behavior (direct tile call).
        import types
        class _FrozenAnalogFwd(torch.autograd.Function):
            @staticmethod
            @no_grad()
            def forward(ctx, analog_tile, x_input, is_test):
                ctx.analog_tile = analog_tile
                ctx.saved_analog_tensors = [x_input]
                out = analog_tile.joint_forward(x_input, is_test, ctx)
                ctx.save_for_backward(*ctx.saved_analog_tensors)
                ctx.saved_analog_tensors = []
                return out
            @staticmethod
            @no_grad()
            def backward(ctx, grad_output):
                ctx.saved_analog_tensors = list(ctx.saved_tensors)
                grad_input = ctx.analog_tile.backward(grad_output, ctx)
                ctx.saved_analog_tensors = []
                return None, grad_input, None

        def _frozen_analog_forward(self, x_input, tensor_view=None):
            out = _FrozenAnalogFwd.apply(self, x_input, not self.training)
            if tensor_view is None:
                tensor_view = self.get_tensor_view(out.dim())
            out = self.apply_out_scaling(out, tensor_view)
            if self.digital_bias:
                return out + self.bias.view(*tensor_view)
            return out

        for mod_name, m in model.named_modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    if id(tile) not in existing_tile_ids:
                        # Head analog tiles remain trainable (weight + bias)
                        if HEAD_ANALOG and "classifier" in mod_name:
                            continue
                        tile.update = _frozen_noop_update
                        tile.forward = types.MethodType(_frozen_analog_forward, tile)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Step 2: Set requires_grad
    # - LRTT layers: A/B + out_scaling TRAINABLE, C + bias FROZEN
    # - classifier: TRAINABLE if HEAD_LAYER=="train", else FROZEN
    # - pooler: always digital frozen
    # - Everything else: FROZEN
    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            param.requires_grad = not NO_TRANSFER
        elif "tile_c" in name:
            pass  # Respect lrtt_tile.py settings (train_c_bias, mapping_c)
        elif "out_scaling_alpha" in name:
            pass  # Frozen analog out_scaling: TRAINABLE (same as C tile)
        elif "classifier" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        elif "pooler" in name:
            param.requires_grad = False
        elif "LayerNorm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_params = count_parameters(model)

    print(f"\nCreated BERT model (LRTT):")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Task: {TASK_NAME} (num_labels={NUM_LABELS})")
    print(f"  Total params: {total_params:,}, Trainable: {num_params:,}")
    print(f"  LRTT Analog layers: {num_analog}")
    print(f"  LRTT config: rank={LRTT_RANK}, transfer_every={TRANSFER_EVERY}, "
          f"transfer_lr={TRANSFER_LR}, fast_lr={FAST_LR}, auto_scale={AUTO_SCALE_MODE}")
    print(f"  Reinit: mode={REINIT_MODE}, gain={REINIT_GAIN}")
    print(f"  LoRA target: {LORA_TARGET} -> {lrtt_patterns if lrtt_patterns else 'all encoder layers'}")

    try:
        return model.to(DEVICE)
    except Exception:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


# =============================================================================
# Data Functions
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize GLUE dataset for the specified task."""
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)
    sentence1_key, sentence2_key = TASK_TO_KEYS[TASK_NAME]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding=False,
            )
        return tokenizer(
            examples[sentence1_key], examples[sentence2_key],
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,
        )

    # Tokenize all splits
    cols_to_remove = [c for c in raw_datasets["train"].column_names if c != "label"]
    tokenized = raw_datasets.map(preprocess_function, batched=True, remove_columns=cols_to_remove)
    tokenized = tokenized.rename_column("label", "labels")

    # Training set
    train_dataset = tokenized["train"]
    if TRAIN_SUBSET_SIZE > 0:
        train_dataset = train_dataset.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(train_dataset)))
        )
    else:
        train_dataset = train_dataset.shuffle(seed=SEED)

    # Eval set (MNLI uses validation_matched)
    eval_key = "validation_matched" if TASK_NAME == "mnli" else "validation"
    eval_dataset = tokenized[eval_key]
    if EVAL_SUBSET_SIZE > 0:
        eval_dataset = eval_dataset.select(
            range(min(EVAL_SUBSET_SIZE, len(eval_dataset)))
        )

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE // GRAD_ACCUM_STEPS, shuffle=True,
        collate_fn=collator, num_workers=2,
        generator=torch.Generator().manual_seed(SEED)
    )

    eval_loader = DataLoader(
        eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=collator,
        num_workers=2
    )

    return train_loader, eval_loader


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(model, eval_loader):
    """Evaluate GLUE model. Returns (metric_value, val_loss)."""
    model.eval()

    is_regression = (TASK_NAME == "stsb")
    all_preds = []
    all_labels = []
    total_val_loss = 0.0
    num_val_batches = 0

    with no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            num_val_batches += 1

            if is_regression:
                preds = outputs.logits.squeeze()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.float().cpu().numpy())
            else:
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    model.train()
    val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')

    # Compute task-specific metric
    metric_name = TASK_TO_METRIC[TASK_NAME]
    if metric_name == "accuracy":
        metric_value = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100.0
    elif metric_name == "f1":
        from sklearn.metrics import f1_score
        metric_value = f1_score(all_labels, all_preds) * 100.0
    elif metric_name == "matthews_correlation":
        from sklearn.metrics import matthews_corrcoef
        metric_value = matthews_corrcoef(all_labels, all_preds) * 100.0
    elif metric_name == "spearmanr":
        from scipy.stats import spearmanr
        metric_value = spearmanr(all_preds, all_labels)[0] * 100.0

    return metric_value, val_loss


# =============================================================================
# Diagnostic Helpers
# =============================================================================

def _make_cell_indices(shape, n=10):
    """Generate n evenly-spaced cell indices for a weight matrix of given shape."""
    rows, cols = shape
    indices = []
    for i in range(n):
        r = min(int(i * rows / n), rows - 1)
        c = min(int(i * cols / n), cols - 1)
        indices.append((r, c))
    return indices


def find_first_lrtt_tile(model):
    """Find the first LRTT tile in the model."""
    for name, mod in model.named_modules():
        if hasattr(mod, 'controller'):
            return name, mod
    raise RuntimeError("No LRTT tile found in model")


def find_last_lrtt_tile(model):
    """Find the last LRTT tile in the model."""
    last_name, last_tile = None, None
    for name, mod in model.named_modules():
        if hasattr(mod, 'controller'):
            last_name, last_tile = name, mod
    if last_tile is None:
        raise RuntimeError("No LRTT tile found in model")
    return last_name, last_tile


def sample_cells(weight_matrix, cell_indices):
    """Extract values at fixed cell positions from a weight matrix."""
    values = []
    for r, c in cell_indices:
        if r < weight_matrix.shape[0] and c < weight_matrix.shape[1]:
            values.append(weight_matrix[r, c].item())
        else:
            values.append(0.0)
    return values


def get_raw_C(tile_c):
    """Get C tile raw weights WITHOUT out_scaling."""
    W_scaled = tile_c.get_weights()[0]
    if hasattr(tile_c, 'out_scaling_alpha') and tile_c.out_scaling_alpha is not None:
        alpha = tile_c.out_scaling_alpha.detach().to(W_scaled.device)
        return W_scaled / alpha.unsqueeze(1)
    return W_scaled


def snapshot_weights(tile):
    """Snapshot A, B, C weights before optimizer step."""
    return (
        tile.tile_a.get_weights()[0].clone().detach(),
        tile.tile_b.get_weights()[0].clone().detach(),
        tile.tile_c.get_weights()[0].clone().detach(),
        get_raw_C(tile.tile_c).clone().detach(),
    )


def collect_tile_diagnostics(tile, C_prev_raw, A_before, B_before, C_before,
                             C_raw_before, step, prev_num_transfers,
                             A_ci, B_ci, C_ci):
    """Collect all diagnostic data for one tile at one step."""
    controller = tile.controller
    A = tile.tile_a.get_weights()[0]
    B = tile.tile_b.get_weights()[0]
    C_raw = get_raw_C(tile.tile_c)

    norm_A = torch.norm(A).item()
    norm_B = torch.norm(B).item()
    norm_C_raw = torch.norm(C_raw).item()
    norm_AB = torch.norm(A @ B).item()

    delta_C_raw = torch.norm(C_raw - C_prev_raw).item() if C_prev_raw is not None else 0.0
    delta_A = torch.norm(A - A_before).item() if A_before is not None else 0.0
    delta_B = torch.norm(B - B_before).item() if B_before is not None else 0.0
    delta_C_raw_step = torch.norm(C_raw - C_raw_before).item() if C_raw_before is not None else 0.0

    A_cells = sample_cells(A, A_ci)
    B_cells = sample_cells(B, B_ci)
    C_cells = sample_cells(C_raw, C_ci)

    A_grad_cells, B_grad_cells, C_grad_cells = [], [], []
    if A_before is not None:
        A_grad_cells = sample_cells(A - A_before, A_ci)
    if B_before is not None:
        B_grad_cells = sample_cells(B - B_before, B_ci)
    if C_raw_before is not None:
        C_grad_cells = sample_cells(C_raw - C_raw_before, C_ci)

    transfer_counter = controller.transfer_counter
    num_transfers = controller.num_transfers
    is_transfer = num_transfers > prev_num_transfers

    record = {
        "step": step,
        "norm_A": norm_A, "norm_B": norm_B,
        "norm_C_raw": norm_C_raw, "norm_AB": norm_AB,
        "A_cells": A_cells, "B_cells": B_cells, "C_cells": C_cells,
        "A_grad_cells": A_grad_cells, "B_grad_cells": B_grad_cells,
        "C_grad_cells": C_grad_cells,
        "delta_A": delta_A, "delta_B": delta_B, "delta_C_raw": delta_C_raw_step,
        "transfer_counter": transfer_counter,
        "num_transfers": num_transfers, "is_transfer": is_transfer,
    }
    return record, C_raw.clone().detach(), num_transfers


def _cos_sim(a, b):
    """Cosine similarity between two flat tensors."""
    na, nb = torch.norm(a).item(), torch.norm(b).item()
    if na > 1e-10 and nb > 1e-10:
        return torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)).item()
    return 0.0


def make_diagnostic_plots(log_data, output_path, tile_label="",
                          A_ci=None, B_ci=None, C_ci=None):
    """Create 5x2 (10 panel) diagnostic plot for one tile."""
    steps = [r["step"] for r in log_data]
    norm_A = [r["norm_A"] for r in log_data]
    norm_B = [r["norm_B"] for r in log_data]
    norm_C_raw = [r["norm_C_raw"] for r in log_data]
    norm_AB = [r["norm_AB"] for r in log_data]
    losses = [r.get("loss", 0.0) for r in log_data]

    transfer_steps = [r["step"] for r in log_data if r["is_transfer"]]

    n_cells = len(log_data[0]["A_cells"])
    A_w = [[r["A_cells"][i] for r in log_data] for i in range(n_cells)]
    B_w = [[r["B_cells"][i] for r in log_data] for i in range(n_cells)]
    C_w = [[r["C_cells"][i] for r in log_data] for i in range(len(log_data[0]["C_cells"]))]
    A_g = [[r["A_grad_cells"][i] if r["A_grad_cells"] else 0.0 for r in log_data] for i in range(n_cells)]
    B_g = [[r["B_grad_cells"][i] if r["B_grad_cells"] else 0.0 for r in log_data] for i in range(n_cells)]
    C_g = [[r["C_grad_cells"][i] if r["C_grad_cells"] else 0.0 for r in log_data] for i in range(len(log_data[0]["C_cells"]))]

    # Use provided indices for labels, else generate generic
    a_ci = A_ci or [(i, 0) for i in range(n_cells)]
    b_ci = B_ci or [(0, i) for i in range(n_cells)]
    c_ci = C_ci or [(i, i) for i in range(len(log_data[0]["C_cells"]))]

    fig, axes = plt.subplots(5, 2, figsize=(18, 28))
    title_str = f"LRTT Diagnostic — {tile_label}" if tile_label else "LRTT Diagnostic"
    fig.suptitle(title_str, fontsize=14)

    def tl(ax):
        for ts in transfer_steps:
            ax.axvline(x=ts, color="red", alpha=0.3, linewidth=0.8)

    # (0,0) A/B/AB norms
    ax = axes[0, 0]
    ax.plot(steps, norm_A, label="||A||", alpha=0.8)
    ax.plot(steps, norm_B, label="||B||", alpha=0.8)
    ax.plot(steps, norm_AB, label="||A@B||", alpha=0.6, linestyle="--")
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Norm")
    ax.set_title("A, B, AB Norms (red = transfer)"); ax.legend(); ax.grid(True, alpha=0.3)

    # (0,1) C norm + delta
    ax = axes[0, 1]
    ax.plot(steps, norm_C_raw, label="||C_raw||", color="green", alpha=0.8)
    delta_C = [r["delta_C_raw"] for r in log_data]
    ax2 = ax.twinx()
    ax2.plot(steps, delta_C, label="delta_C_raw", color="orange", alpha=0.8)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("||C_raw||", color="green")
    ax2.set_ylabel("delta_C_raw", color="orange")
    ax.set_title("C Norm (raw) + delta_C_raw")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, loc="upper left"); ax.grid(True, alpha=0.3)

    # Rows 1-3: A/B/C cells
    for row, (ws, gs, ci, nm) in enumerate(
            [(A_w, A_g, a_ci, "A"), (B_w, B_g, b_ci, "B"), (C_w, C_g, c_ci, "C")], start=1):
        ax = axes[row, 0]
        for i, s in enumerate(ws):
            r, c = ci[i]; ax.plot(steps, s, label=f"{nm}[{r},{c}]", alpha=0.7, linewidth=0.8)
        tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Weight")
        ax.set_title(f"{nm} cells: weights"); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        for i, s in enumerate(gs):
            r, c = ci[i]; ax.plot(steps, s, label=f"d{nm}[{r},{c}]", alpha=0.7, linewidth=0.8)
        tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Delta")
        ax.set_title(f"{nm} cells: delta"); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # (4,0) G_accum / tlr*AB norms (lines) + dC norm at transfers (markers) + loss
    nG = [max(r.get("norm_G_accum", 1e-10), 1e-10) for r in log_data]
    nT = [max(r.get("norm_tlrAB", 1e-10), 1e-10) for r in log_data]
    # delta_C only at transfer steps
    t_steps_dC = [r["step"] for r in log_data if r["is_transfer"]]
    t_norms_dC = [max(r.get("norm_dC_step", 1e-10), 1e-10) for r in log_data if r["is_transfer"]]
    ax = axes[4, 0]
    ax.semilogy(steps, nG, label="||G_accum||", color="red", alpha=0.8, linewidth=0.8)
    ax.semilogy(steps, nT, label="||tlr*A@B||", color="green", alpha=0.8, linewidth=0.8)
    if t_steps_dC:
        ax.semilogy(t_steps_dC, t_norms_dC, 'o', label="||delta_C|| @T", color="blue",
                     markersize=5, alpha=0.9, zorder=5)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Norm (log)")
    axl = ax.twinx(); axl.plot(steps, losses, label="loss", color="gray", alpha=0.35, linewidth=0.6)
    axl.set_ylabel("Loss", color="gray")
    lm, llm = ax.get_legend_handles_labels(); ll, lll = axl.get_legend_handles_labels()
    ax.legend(lm+ll, llm+lll, fontsize=7, loc="upper right")
    ax.set_title("||G_accum|| vs ||tlr*A@B|| + ||delta_C|| at transfers + Loss"); ax.grid(True, alpha=0.3)

    # (4,1) cos(tlr*AB, G) line + cos(dC, *) at transfers (markers) + loss
    cTG = [r.get("cos_tlrAB_G", 0) for r in log_data]
    t_cDG = [r.get("cos_dC_G", 0) for r in log_data if r["is_transfer"]]
    t_cDT = [r.get("cos_dC_tlrAB", 0) for r in log_data if r["is_transfer"]]
    ax = axes[4, 1]
    ax.plot(steps, cTG, label="cos(tlr*AB, G)", color="green", alpha=0.7, linewidth=0.8)
    if t_steps_dC:
        ax.scatter(t_steps_dC, t_cDG, label="cos(dC, G) @T", color="blue",
                   s=25, alpha=0.9, zorder=5, marker="o")
        ax.scatter(t_steps_dC, t_cDT, label="cos(dC, tlr*AB) @T", color="purple",
                   s=25, alpha=0.9, zorder=5, marker="s")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.3)
    tl(ax); ax.set_ylabel("Cosine Similarity"); ax.set_ylim(-1.1, 1.1)
    axl2 = ax.twinx(); axl2.plot(steps, losses, label="loss", color="gray", alpha=0.35, linewidth=0.6)
    axl2.set_ylabel("Loss", color="gray")
    lc, llc = ax.get_legend_handles_labels(); ll2, lll2 = axl2.get_legend_handles_labels()
    ax.legend(lc+ll2, llc+lll2, fontsize=6, loc="lower left")
    ax.set_xlabel("Step"); ax.set_title("Cosines: dC vs G, tlr*AB vs G, dC vs tlr*AB + Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def make_xd_diagnostic_plots(log_data, output_path, tile_label=""):
    """Create x/d distribution diagnostic plots: percentile bands + histograms."""
    if not log_data:
        return
    steps = [r['step'] for r in log_data]

    xd_keys = [
        ('xa', 'tile_a input (XB = x·B^T)'),
        ('da', 'tile_a grad (raw gradient)'),
        ('xb', 'tile_b input (raw x)'),
        ('db', 'tile_b grad (DA = A^T·d)'),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(18, 18))
    fig.suptitle(f'x/d Distribution — {tile_label}', fontsize=13, y=0.99)

    for row, (prefix, desc) in enumerate(xd_keys):
        # --- Left: percentile band plot over time ---
        ax = axes[row, 0]
        p5 = [r.get(f'{prefix}_p5', 0) for r in log_data]
        p25 = [r.get(f'{prefix}_p25', 0) for r in log_data]
        p50 = [r.get(f'{prefix}_p50', 0) for r in log_data]
        p75 = [r.get(f'{prefix}_p75', 0) for r in log_data]
        p95 = [r.get(f'{prefix}_p95', 0) for r in log_data]
        mean_vals = [r.get(f'{prefix}_abs_mean', 0) for r in log_data]
        max_vals = [r.get(f'{prefix}_abs_max', 0) for r in log_data]

        ax.fill_between(steps, p5, p95, alpha=0.15, color='blue', label='p5-p95')
        ax.fill_between(steps, p25, p75, alpha=0.3, color='blue', label='p25-p75')
        ax.plot(steps, p50, 'b-', linewidth=0.8, label='median')
        ax.plot(steps, mean_vals, 'g--', linewidth=0.6, alpha=0.7, label='mean')
        ax.plot(steps, max_vals, 'r-', linewidth=0.4, alpha=0.5, label='max')
        ax.set_title(f'|{prefix}| percentiles — {desc}')
        ax.set_ylabel(f'|{prefix}|')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        if row == 3:
            ax.set_xlabel('Step')

        # --- Right: histograms at sampled time points ---
        ax = axes[row, 1]
        hist_steps = [r for r in log_data if 'xd_hist' in r and prefix in r['xd_hist']]
        if hist_steps:
            n_hist = len(hist_steps)
            sample_idx = [0, n_hist // 3, 2 * n_hist // 3, n_hist - 1]
            sample_idx = sorted(set(min(i, n_hist - 1) for i in sample_idx))
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sample_idx)))
            for ci, idx in enumerate(sample_idx):
                h = hist_steps[idx]['xd_hist'][prefix]
                counts = h['counts']
                bin_edges = np.linspace(h['min'], h['max'], len(counts) + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                total = sum(counts)
                if total > 0:
                    normed = [c / total for c in counts]
                    ax.plot(bin_centers, normed, color=colors[ci], linewidth=1.0,
                            label=f'step {hist_steps[idx]["step"]}', alpha=0.8)
            ax.set_title(f'|{prefix}| distribution')
            ax.set_ylabel('density')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No histogram data', transform=ax.transAxes, ha='center')
        if row == 3:
            ax.set_xlabel(f'|{prefix}|')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


# =============================================================================
# Optimizer & Scheduler
# =============================================================================

def create_optimizer(model):
    """Create optimizer. Uses Analog optimizers when model has analog tiles (LRTT or frozen analog)."""
    if LORA_TARGET == "none" and not ENCODER_ANALOG and not HEAD_ANALOG:
        # None mode (no analog tiles): use standard PyTorch optimizers
        if OPTIMIZER == "AnalogSGD":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=LEARNING_RATE,
                weight_decay=0.0, momentum=0.0, nesterov=False
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
    else:
        # Analog optimizers: required for LRTT tiles and frozen analog tiles
        # (AnalogSGD/Adam calls analog_ctx.reset() to prevent memory leak)
        if OPTIMIZER == "AnalogSGD":
            optimizer = AnalogSGD(
                model.parameters(), lr=LEARNING_RATE,
                weight_decay=0.0, momentum=0.0, nesterov=False
            )
        else:
            optimizer = AnalogAdam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
        optimizer.regroup_param_groups()
        optimizer._grad_accum_steps = GRAD_ACCUM_STEPS

    return optimizer


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
# Main
# =============================================================================

def main():
    """Train BERT with LRTT on GLUE."""
    metric_name = TASK_TO_METRIC[TASK_NAME]

    manual_seed(SEED)
    set_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"bert_lrtt_{TASK_NAME}_r{LRTT_RANK}_te{TRANSFER_EVERY}_bs{BATCH_SIZE}",
        config={
            "model": MODEL_NAME, "dataset": f"GLUE/{TASK_NAME}",
            "task": TASK_NAME, "metric": metric_name,
            "lrtt_rank": LRTT_RANK, "transfer_every": TRANSFER_EVERY,
            "transfer_lr": TRANSFER_LR, "fast_lr": FAST_LR, "auto_scale_mode": AUTO_SCALE_MODE,
            "reinit_mode": REINIT_MODE, "reinit_gain": REINIT_GAIN,
            "tau_sec": TAU_SEC,
            "dynamic_te": DYNAMIC_TE, "te_warmup_steps": TE_WARMUP_STEPS,
            "epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "optimizer": OPTIMIZER, "warmup_steps": WARMUP_STEPS,
            "min_lr_rate": MIN_LR_RATE, "seed": SEED,
            "lora_target": LORA_TARGET,
        }
    )

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # Create model, optimizer, scheduler
    model = create_model()
    optimizer = create_optimizer(model)

    schedule_ep = SCHEDULE_EPOCHS if SCHEDULE_EPOCHS > 0 else N_EPOCHS
    num_training_steps = len(train_loader) * schedule_ep // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps,
        min_lr_rate=MIN_LR_RATE,
    )

    # =========================================================================
    # Diagnostic setup (skipped if ENABLE_DIAGNOSTIC=False)
    # =========================================================================
    first_gc, last_gc = {}, {}
    first_log, last_log = [], []
    first_C_prev_raw, last_C_prev_raw = None, None
    first_prev_nt, last_prev_nt = 0, 0
    first_name = last_name = ""
    first_tile = last_tile = None
    A_CI = B_CI = C_CI = []
    A_shape = B_shape = C_shape = ()

    if ENABLE_DIAGNOSTIC:
        first_name, first_tile = find_first_lrtt_tile(model)
        last_name, last_tile = find_last_lrtt_tile(model)

        # Enable controller-level diagnostics for transfer delta tracking
        first_tile.controller.enable_diagnostics = True
        last_tile.controller.enable_diagnostics = True

        A_shape = tuple(first_tile.tile_a.get_weights()[0].shape)
        B_shape = tuple(first_tile.tile_b.get_weights()[0].shape)
        C_shape = tuple(first_tile.tile_c.get_weights()[0].shape)
        A_CI = _make_cell_indices(A_shape)
        B_CI = _make_cell_indices(B_shape)
        C_CI = _make_cell_indices(C_shape)

        print(f"\nDiag tile (first): {first_name}  A{A_shape} B{B_shape} C{C_shape}")
        print(f"Diag tile (last):  {last_name}")
        print(f"Diag epochs: {'all' if DIAG_EPOCHS == 0 else f'first {DIAG_EPOCHS}'}")

        def _install_hook(diag_tile, device, gc_dict):
            d_size, x_size = diag_tile.tile_c.get_weights()[0].shape
            gc_dict['G_accum'] = torch.zeros(d_size, x_size, device=device)
            gc_dict['active'] = True
            ctrl = diag_tile.controller

            def _abs_stats(t):
                """Return (mean, max) of |t| as floats."""
                a = t.abs()
                return a.mean().item(), a.max().item()

            def _capture_common(x_b, d_a, x_a, d_b):
                """Accumulate G and record AB + x/d stats after both tiles update.

                Args:
                    x_b: tile_b's x input (raw batch input)
                    d_a: tile_a's d input (raw gradient, alpha already removed in FI)
                    x_a: tile_a's x input (XB = B·x in FI; same as x_b projected in NFI)
                    d_b: tile_b's d input (DA = A^T·d)
                """
                with torch.no_grad():
                    x_2d = x_b.reshape(-1, x_b.shape[-1])
                    d_2d = d_a.reshape(-1, d_a.shape[-1])
                    gc_dict['G_accum'] = gc_dict['G_accum'] + d_2d.t() @ x_2d
                    A = diag_tile.tile_a.get_weights()[0].to(device)
                    B = diag_tile.tile_b.get_weights()[0].to(device)
                    AB = A @ B
                    gc_dict['AB_matrix'] = AB.clone()
                    gc_dict['norm_AB_pre'] = torch.norm(AB).item()
                    gc_dict['norm_G_accum'] = torch.norm(gc_dict['G_accum']).item()
                    AB_flat = AB.flatten(); G_flat = gc_dict['G_accum'].flatten()
                    gc_dict['cos_AB_G'] = (torch.nn.functional.cosine_similarity(
                        AB_flat.unsqueeze(0), G_flat.unsqueeze(0)).item()
                        if gc_dict['norm_AB_pre'] > 1e-10 and gc_dict['norm_G_accum'] > 1e-10 else 0.0)
                    # tile_a sees (XB, d_raw); tile_b sees (x_raw, DA)
                    gc_dict['xa_abs_mean'], gc_dict['xa_abs_max'] = _abs_stats(x_a.to(device))
                    gc_dict['da_abs_mean'], gc_dict['da_abs_max'] = _abs_stats(d_a.to(device))
                    gc_dict['xb_abs_mean'], gc_dict['xb_abs_max'] = _abs_stats(x_b.to(device))
                    gc_dict['db_abs_mean'], gc_dict['db_abs_max'] = _abs_stats(d_b.to(device))
                    # Percentiles (p5, p25, p50, p75, p95)
                    _pcts = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=device)
                    for _prefix, _t in [('xa', x_a), ('da', d_a), ('xb', x_b), ('db', d_b)]:
                        _flat = _t.to(device).abs().flatten()
                        _q = torch.quantile(_flat.float(), _pcts).tolist()
                        gc_dict[f'{_prefix}_p5'] = _q[0]
                        gc_dict[f'{_prefix}_p25'] = _q[1]
                        gc_dict[f'{_prefix}_p50'] = _q[2]
                        gc_dict[f'{_prefix}_p75'] = _q[3]
                        gc_dict[f'{_prefix}_p95'] = _q[4]
                    # Histogram (every 100 steps)
                    gc_dict['_capture_count'] = gc_dict.get('_capture_count', 0) + 1
                    if gc_dict['_capture_count'] % 100 == 1:
                        _hists = {}
                        for _prefix, _t in [('xa', x_a), ('da', d_a), ('xb', x_b), ('db', d_b)]:
                            _flat = _t.to(device).abs().flatten().float()
                            _max_val = _flat.max().item()
                            if _max_val > 0:
                                _counts = torch.histc(_flat, bins=50, min=0, max=_max_val).tolist()
                                _hists[_prefix] = {'counts': _counts, 'min': 0.0, 'max': _max_val}
                            else:
                                _hists[_prefix] = {'counts': [float(_flat.numel())] + [0.0]*49, 'min': 0.0, 'max': 1.0}
                        gc_dict['_last_hist'] = _hists
                        gc_dict['_hist_ready'] = True
                    else:
                        gc_dict['_hist_ready'] = False

            if ctrl.forward_inject_enabled:
                # FI mode: ab_weight_update is never called.
                # Hook tile_b._orig_update which fires last (after tile_a._orig_update).
                original_b_update = diag_tile.tile_b._orig_update

                def hooked_b(x_input, d_input, *args, **kwargs):
                    result = original_b_update(x_input, d_input, *args, **kwargs)
                    if gc_dict.get('active'):
                        _capture_common(
                            x_b=x_input,       # raw x
                            d_a=ctrl._fi_a_d,  # raw d (alpha already removed)
                            x_a=ctrl._fi_a_x,  # XB = B·x
                            d_b=d_input,       # DA = A^T·d
                        )
                    return result

                diag_tile.tile_b._orig_update = hooked_b
            else:
                # NFI mode: tile_c.update triggers ab_weight_update(x, d, lr).
                original_fn = ctrl.ab_weight_update

                def hooked(x, d, lr, **kwargs):
                    if gc_dict.get('active'):
                        with torch.no_grad():
                            A_w = diag_tile.tile_a.get_weights()[0].to(device)
                            B_w = diag_tile.tile_b.get_weights()[0].to(device)
                            x_dev = x.to(device); d_dev = d.to(device)
                            x_2d = x_dev.reshape(-1, x_dev.shape[-1])
                            d_2d = d_dev.reshape(-1, d_dev.shape[-1])
                            XB = x_2d @ B_w.t()   # [batch, rank]
                            DA = d_2d @ A_w        # [batch, rank]
                            gc_dict['_nfi_XB'] = XB
                            gc_dict['_nfi_DA'] = DA
                            gc_dict['_nfi_x'] = x_2d
                            gc_dict['_nfi_d'] = d_2d
                    result = original_fn(x, d, lr, **kwargs)
                    if gc_dict.get('active'):
                        with torch.no_grad():
                            XB = gc_dict.pop('_nfi_XB')
                            DA = gc_dict.pop('_nfi_DA')
                            x_2d = gc_dict.pop('_nfi_x')
                            d_2d = gc_dict.pop('_nfi_d')
                            _capture_common(x_b=x_2d, d_a=d_2d, x_a=XB, d_b=DA)
                    return result

                ctrl.ab_weight_update = hooked

        _install_hook(first_tile, DEVICE, first_gc)
        _install_hook(last_tile, DEVICE, last_gc)
        print("Gradient tracking hooks installed")

    # Initial evaluation
    init_acc, init_val_loss = evaluate_model(model, eval_loader)
    wandb.log({"epoch": 0, f"eval/{metric_name}": init_acc})
    print(f"Initial eval: {metric_name}={init_acc:.2f}%")

    # Training loop
    best_acc = init_acc
    best_epoch = 0
    epochs_without_improvement = 0
    best_val_loss = float('inf')
    val_loss_no_improvement = 0
    val_loss_crossed_threshold = False  # True once val loss drops below threshold
    global_step = 0

    print(f"\nStarting training: {N_EPOCHS} epochs (max), early stopping patience={EARLY_STOP_PATIENCE}")

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        optimizer.zero_grad()
        for micro_step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Diagnostic: snapshot before optimizer step
                diag_active = ENABLE_DIAGNOSTIC and (DIAG_EPOCHS == 0 or epoch <= DIAG_EPOCHS)
                if diag_active:
                    first_snap = snapshot_weights(first_tile)
                    last_snap = snapshot_weights(last_tile)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if diag_active:
                    # --- Collect diagnostics ---
                    for tile, snap, gcd, log_list, prev_state in [
                        (first_tile, first_snap, first_gc, first_log, "first"),
                        (last_tile, last_snap, last_gc, last_log, "last"),
                    ]:
                        A_bef, B_bef, C_bef, Craw_bef = snap
                        if prev_state == "first":
                            rec, first_C_prev_raw, first_prev_nt = collect_tile_diagnostics(
                                tile, first_C_prev_raw, A_bef, B_bef, C_bef, Craw_bef,
                                global_step, first_prev_nt, A_CI, B_CI, C_CI)
                        else:
                            rec, last_C_prev_raw, last_prev_nt = collect_tile_diagnostics(
                                tile, last_C_prev_raw, A_bef, B_bef, C_bef, Craw_bef,
                                global_step, last_prev_nt, A_CI, B_CI, C_CI)
                        rec["loss"] = loss.item() * GRAD_ACCUM_STEPS
                        rec["norm_G_accum"] = gcd.get('norm_G_accum', 0.0)
                        rec["norm_AB_pre"] = gcd.get('norm_AB_pre', 0.0)
                        rec["cos_AB_G"] = gcd.get('cos_AB_G', 0.0)
                        # tile_a sees (XB, d_raw); tile_b sees (x_raw, DA)
                        rec["xa_abs_mean"] = gcd.get('xa_abs_mean', 0.0)
                        rec["xa_abs_max"] = gcd.get('xa_abs_max', 0.0)
                        rec["da_abs_mean"] = gcd.get('da_abs_mean', 0.0)
                        rec["da_abs_max"] = gcd.get('da_abs_max', 0.0)
                        rec["xb_abs_mean"] = gcd.get('xb_abs_mean', 0.0)
                        rec["xb_abs_max"] = gcd.get('xb_abs_max', 0.0)
                        rec["db_abs_mean"] = gcd.get('db_abs_mean', 0.0)
                        rec["db_abs_max"] = gcd.get('db_abs_max', 0.0)
                        # Percentiles
                        for _pf in ['xa', 'da', 'xb', 'db']:
                            for _pp in ['p5', 'p25', 'p50', 'p75', 'p95']:
                                rec[f'{_pf}_{_pp}'] = gcd.get(f'{_pf}_{_pp}', 0.0)
                        # Histogram (only when captured)
                        if gcd.get('_hist_ready'):
                            rec['xd_hist'] = gcd['_last_hist']

                        with torch.no_grad():
                            C_raw_after = get_raw_C(tile.tile_c).to(DEVICE)
                            delta_C_mat = C_raw_after - Craw_bef.to(DEVICE)
                            AB_mat = gcd.get('AB_matrix')
                            tlr_AB = TRANSFER_LR * AB_mat if AB_mat is not None else torch.zeros_like(delta_C_mat)
                            # Use controller's exact deltas for cosine comparison at transfer steps
                            if rec["is_transfer"]:
                                ctrl_delta = tile.controller.last_transfer_delta
                                actual_delta = tile.controller.last_actual_delta
                                if ctrl_delta is not None:
                                    tlr_AB = ctrl_delta.to(DEVICE)
                                if actual_delta is not None:
                                    delta_C_mat = actual_delta.to(DEVICE)
                            dC_f = delta_C_mat.flatten()
                            G_f = gcd.get('G_accum', torch.zeros_like(delta_C_mat)).flatten()
                            tlr_f = tlr_AB.flatten()
                            rec["cos_dC_G"] = _cos_sim(dC_f, G_f)
                            rec["cos_tlrAB_G"] = _cos_sim(tlr_f, G_f)
                            rec["cos_dC_tlrAB"] = _cos_sim(dC_f, tlr_f)
                            rec["norm_dC_step"] = torch.norm(delta_C_mat).item()
                            rec["norm_tlrAB"] = torch.norm(tlr_AB).item()

                        if rec["is_transfer"]:
                            gcd['G_accum'] = torch.zeros_like(gcd['G_accum'])

                        log_list.append(rec)

                    tag = ""
                    if first_log[-1]["is_transfer"]: tag += " [T1]"
                    if last_log[-1]["is_transfer"]: tag += " [T2]"
                    pbar.set_postfix_str(
                        f"loss={loss.item() * GRAD_ACCUM_STEPS:.4f} ||A||={first_log[-1]['norm_A']:.3f} "
                        f"T1={first_log[-1]['num_transfers']} T2={last_log[-1]['num_transfers']}{tag}")
                else:
                    pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            num_batches += 1

        # Deactivate hooks after DIAG_EPOCHS
        if ENABLE_DIAGNOSTIC and DIAG_EPOCHS > 0 and epoch == DIAG_EPOCHS:
            first_gc['active'] = False
            last_gc['active'] = False
            print(f"Diagnostic collection stopped after epoch {epoch}")

        train_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Evaluate
        eval_acc, val_loss = evaluate_model(model, eval_loader)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch, "train/loss": train_loss,
            f"eval/{metric_name}": eval_acc,
            "learning_rate": current_lr,
        })

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            save(model.state_dict(), WEIGHT_PATH)
        else:
            epochs_without_improvement += 1

        val_loss_improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_loss_no_improvement = 0
            val_loss_improved = " ↓"
        else:
            val_loss_no_improvement += 1

        # Reset metric patience when val loss first crosses threshold
        if not val_loss_crossed_threshold and best_val_loss <= VAL_LOSS_THRESHOLD:
            val_loss_crossed_threshold = True
            epochs_without_improvement = 0

        tqdm.write(
            f"Epoch {epoch}: Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}{val_loss_improved} | "
            f"{metric_name} {eval_acc:.2f}% | "
            f"Best {best_acc:.2f}% | LR {current_lr:.2e} | "
            f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}"
        )

        if not val_loss_crossed_threshold and val_loss_no_improvement >= VAL_LOSS_EARLY_STOP_PATIENCE:
            print(f"Val loss early stop at epoch {epoch} "
                  f"(val_loss={val_loss:.4f} > {VAL_LOSS_THRESHOLD}, no improvement for {val_loss_no_improvement} epochs)")
            break

        if val_loss_crossed_threshold and epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest {metric_name}: {best_acc:.2f}% at epoch {best_epoch}")

    # =========================================================================
    # Save diagnostic outputs
    # =========================================================================
    if ENABLE_DIAGNOSTIC and first_log:
        stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"
        first_transfers = [r["step"] for r in first_log if r["is_transfer"]]
        last_transfers = [r["step"] for r in last_log if r["is_transfer"]]
        diag_steps = len(first_log)
        print(f"\nDiag: {diag_steps}/{global_step} steps, T1={len(first_transfers)}, T2={len(last_transfers)}")

        json_path = os.path.join(RESULTS, f"{TASK_NAME}_diagnostic_log_{stamp}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "config": {
                    "learning_rate": LEARNING_RATE, "transfer_lr": TRANSFER_LR,
                    "transfer_every": TRANSFER_EVERY, "lrtt_rank": LRTT_RANK,
                    "fast_lr": FAST_LR, "auto_scale_mode": AUTO_SCALE_MODE, "reinit_mode": REINIT_MODE,
                    "transfer_method": TRANSFER_METHOD, "optimizer": OPTIMIZER,
                    "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS,
                    "diag_epochs": DIAG_EPOCHS,
                },
                "task": TASK_NAME, "metric": metric_name,
                "best_metric": best_acc, "best_epoch": best_epoch,
                "total_steps": global_step, "diag_steps": diag_steps,
                "first_tile": {
                    "name": first_name,
                    "A_shape": list(A_shape), "B_shape": list(B_shape), "C_shape": list(C_shape),
                    "A_cell_indices": A_CI, "B_cell_indices": B_CI, "C_cell_indices": C_CI,
                    "total_transfers": len(first_transfers), "transfer_steps": first_transfers,
                    "steps": first_log,
                },
                "last_tile": {
                    "name": last_name,
                    "A_shape": list(A_shape), "B_shape": list(B_shape), "C_shape": list(C_shape),
                    "A_cell_indices": A_CI, "B_cell_indices": B_CI, "C_cell_indices": C_CI,
                    "total_transfers": len(last_transfers), "transfer_steps": last_transfers,
                    "steps": last_log,
                },
            }, f, indent=2)
        print(f"Saved: {json_path}")

        make_diagnostic_plots(first_log,
            os.path.join(RESULTS, f"{TASK_NAME}_diag_first_{stamp}.png"),
            tile_label=f"First tile ({first_name})", A_ci=A_CI, B_ci=B_CI, C_ci=C_CI)
        make_diagnostic_plots(last_log,
            os.path.join(RESULTS, f"{TASK_NAME}_diag_last_{stamp}.png"),
            tile_label=f"Last tile ({last_name})", A_ci=A_CI, B_ci=B_CI, C_ci=C_CI)

        # x/d distribution plots
        make_xd_diagnostic_plots(first_log,
            os.path.join(RESULTS, f"{TASK_NAME}_diag_xd_first_{stamp}.png"),
            tile_label=f"First tile ({first_name})")
        make_xd_diagnostic_plots(last_log,
            os.path.join(RESULTS, f"{TASK_NAME}_diag_xd_last_{stamp}.png"),
            tile_label=f"Last tile ({last_name})")

        steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
        diag_ep = DIAG_EPOCHS if DIAG_EPOCHS > 0 else N_EPOCHS
        for ep in range(1, diag_ep + 1):
            s0, s1 = (ep-1)*steps_per_epoch, ep*steps_per_epoch
            ef, el = first_log[s0:s1], last_log[s0:s1]
            if not ef: break
            make_diagnostic_plots(ef,
                os.path.join(RESULTS, f"{TASK_NAME}_diag_first_{stamp}_ep{ep}.png"),
                tile_label=f"First tile ({first_name}) — Epoch {ep}",
                A_ci=A_CI, B_ci=B_CI, C_ci=C_CI)
            make_diagnostic_plots(el,
                os.path.join(RESULTS, f"{TASK_NAME}_diag_last_{stamp}_ep{ep}.png"),
                tile_label=f"Last tile ({last_name}) — Epoch {ep}",
                A_ci=A_CI, B_ci=B_CI, C_ci=C_CI)

    # Memory cleanup
    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU cache cleared")

    wandb.finish()


if __name__ == "__main__":
    main()
