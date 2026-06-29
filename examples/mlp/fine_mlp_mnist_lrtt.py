# -*- coding: utf-8 -*-
"""MLP + MNIST with LRTT (Low-Rank TikiTaka Training).

Single-run training script for a 2-layer MLP on MNIST using LRTT analog layers.
Converts linear1 (input projection) to analog; classifier (output head) remains digital.

Based on fine_bert_squad_lrtt.py with BERT/SQuAD-specific bits replaced by MLP/MNIST.

Inline flags (edit directly in script):
    N_EPOCHS = 15                    # Number of training epochs
    BATCH_SIZE = 64                 # Training batch size
    LEARNING_RATE = 0.00362         # Peak learning rate
    WEIGHT_DECAY = 0.0              # Weight decay
    STEP_LR_SIZE = 10              # Decay LR every N epochs
    STEP_LR_GAMMA = 0.5            # Multiplier per decay
    OPTIMIZER = "AnalogSGD"         # "AnalogSGD" | "AnalogAdam"
    LRTT_RANK = 8                   # LoRA rank for LRTT
    TRANSFER_EVERY = 1000           # Transfer interval (steps)
    TRANSFER_LR = 0.00115           # Transfer learning rate
    TRANSFER_METHOD = "onehot"      # Transfer method: "onehot" | "direct" | "set"
    FAST_LR = 1.0                   # Fast LR for A/B updates
    AUTO_SCALE_MODE = "none"        # Auto-scale: "none" | "shared" | "separate"
    REINIT_MODE = "hybrid"          # Reinit mode: "standard" | "decay" | "hybrid" | "orthogonal_zero" | "orthogonal_decay" | "gauss_b_zero" | "gauss_b_decay" | "gauss_a_zero" | "gauss_a_decay" | "selector_b_zero" | "selector_b_decay" | "selector_a_zero" | "selector_a_decay" | "sparse_a_zero" | "sparse_b_zero" | "binary_a_zero" | "binary_b_zero"
    REINIT_GAIN = 1.0               # Reinitialization gain
    TAU_SEC = 0.0                   # 6T1C retention (0 = no decay)
    DYNAMIC_TE = False              # Enable dynamic transfer every
    DYNAMIC_TE_POWER = 1.0          # Power for dynamic TE scaling
    TE_WARMUP_STEPS = 0            # Steps before reaching target TE
    TE_WARMUP_SCHEDULE = []         # Warmup TE schedule list
    TARGET_MODULES = [...]          # Modules to convert to analog
"""

import os
import sys
import math
import gc

import json

import torch
from torch import nn, no_grad, manual_seed, save
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import wandb
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

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
# Global Constants
# =============================================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = os.path.join(os.getcwd(), "results", "MLP_MNIST_LRTT_FINE")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "fine_mlp_mnist_lrtt_model_weight.pth")

# Reproducibility
SEED = 42

# Model
HIDDEN_DIM = 128

# Training
N_EPOCHS = 10
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 0.1
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 2
TRAIN_LOSS_EARLY_STOP_PATIENCE = 1  # Stop if train loss doesn't improve for this many epochs
TRAIN_LOSS_THRESHOLD = 0.5  # Once train loss drops below this, rely on metric-based early stop only

# Scheduler (StepLR per epoch — matches MLP v2 MNIST)
STEP_LR_SIZE = 10  # Decay LR every N epochs
STEP_LR_GAMMA = 0.5  # Multiplier per decay

# Optimizer
OPTIMIZER = "AnalogSGD"  # "AnalogSGD" or "AnalogAdam"

# LRTT parameters
LRTT_RANK = 8
TRANSFER_EVERY = 32
TRANSFER_LR = 0.5
FAST_LR = 1.0
AUTO_SCALE_MODE = "none"  # Auto-scale mode: "none", "shared", or "separate"
CORRECT_GRADIENT_MAGNITUDES = False  # Correct transfer magnitude by dividing by effective A/B LR
REINIT_MODE = "decay"
REINIT_GAIN = 1.0
A_DENSITY = 1.0  # for sparse_a_zero: fraction of nonzero entries in A (±1 Rademacher)
B_DENSITY = 1.0  # for sparse_b_zero: fraction of nonzero entries in B
TRANSFER_METHOD = "onehot"  # "onehot", "direct", or "set"
C_DW_MIN = 0.001953         # C tile dw_min (10bit)
C_DESIRED_BL = 31           # C tile desired_bl (relevant for onehot/direct transfer)
AB_DW_MIN = 0.001981  # A/B tile dw_min (BERT original default; v2 MNIST equivalent)
AB_DESIRED_BL = 31    # A/B tile desired_bl
AB_MULTILEVEL = None  # If int, w_max = 2^multilevel * AB_DW_MIN / 2; None = device default w_max=1.0

# Device selection
AB_DEVICE = "6t1c"  # "6t1c", "linearstep", "linearstepideal", "constantstep", "constantstepideal", "constantstep6t1cgamma", "fp", "ideal"
A_DEVICE = None  # Optional override for A tile device. None → use AB_DEVICE for both A and B (backward compatible).
B_DEVICE = None  # Optional override for B tile device. None → use AB_DEVICE for both A and B (backward compatible).
C_DEVICE = "constantstepideal"   # "softboundsideal", "linearstepideal", "constantstep", "constantstepideal", "constantstep6t1cgamma", "ideal"

# IO / noise options
IO_NOISE = True             # If False, disable out_noise (resolution kept)
FORWARD_INJECT = False       # If True, enable forward noise injection
FI_CONTINUOUS_ALPHA = False  # If True, use continuous alpha for forward injection
IS_PERFECT = True           # If True, forward/backward use ideal FP matmul (no ADC/DAC/noise)
NO_QUANT = False            # If True, disable DAC/ADC quantization (inp_res/out_res → -1)
DAC_BITS = 8             # DAC (inp_res) bits. None=keep aihwkit default (~7-bit); N→res=1/(2**N-2)
ADC_BITS = 8             # ADC (out_res) bits. None=keep aihwkit default (~9-bit); N→res=1/(2**N-2)
OUT_NOISE = 0.0             # Forward out_noise value

# Per-module IO bit override (MANUALLY EDIT; None = use the module-level DAC_BITS/ADC_BITS).
# Set ADC(out_res)/DAC(inp_res) bits per module TYPE — e.g. give the LRTT input projection a
# different precision than the (frozen-analog) output head. Applied at build time by converting
# each (dac,adc) group in its own pass so the bits land on the tile's forward+backward
# inp_res(DAC)/out_res(ADC). Build-time only — post-conversion override does NOT propagate.
# All-None (default) preserves the original single-pass conversion (byte-identical behavior).
# MLP keys: 'linear1' = input projection (LRTT-eligible), 'classifier' = output head.
LAYER_IO_BITS = {
    'linear1':    {'dac': None, 'adc': None},   # input projection (LRTT)
    'classifier': {'dac': None, 'adc': None},   # output head (digital / frozen-analog)
}
AB_WEIGHT_SCALING_OMEGA = 0.0  # A/B tile weight scaling omega
C_WEIGHT_SCALING_OMEGA = 0.6   # C tile weight scaling omega (from-scratch; 1.0 for fine-tuning)

# Pulse type
AB_PULSE_TYPE = "default"  # "default", "none", "none_with_device", "stochastic_compressed", "mean_count", "deterministic_implicit"

# Transfer options
NO_TRANSFER = False         # If True, disable transfer (set transfer_every to infinity)
NO_SCALE_TRANSFER_LR = True  # If True, disable transfer_lr scaling by rank
TRANSFER_RANK_SCHEDULE = "all"  # "all" or "round_robin"
TRANSFER_RANKS_PER_STEP = 1

# 6T1C Retention parameters
TAU_SEC = 46505.0  # 0 = no decay, >0 = retention time constant

# Dynamic TE (transfer every) parameters
DYNAMIC_TE = False
DYNAMIC_TE_POWER = 1.0
TE_WARMUP_STEPS = 0
TE_WARMUP_SCHEDULE = []

# LoRA target options: which layers have trainable A/B tiles
# - none: no LRTT layers (fully digital baseline)
# - linear1: only the input projection (classifier output head always digital)
AB_IO_PERFECT = False  # If True, A/B tiles fully ideal (no out_noise/ADC/DAC)
LEARN_OUT_SCALING = False  # If True, C tile out_scaling is trainable
LORA_TARGET = "linear1"  # default
HEAD_LAYER = "train"  # "train" or "freeze" for output head (classifier)
ENCODER_ANALOG = False  # If True, non-LRTT linear layers become frozen analog instead of digital
HEAD_ANALOG = False  # If True, output head (classifier) → frozen analog instead of digital trainable
BACKWARD_OUT_BOUND = 12.0  # Backward pass output bound (default 12.0)
LORA_TARGET_MODULES = {
    "none": [],              # No LRTT (digital baseline)
    "linear1": ["linear1"],  # Only LRTT-able layer (classifier = output head, always digital)
}

# Diagnostic
ENABLE_DIAGNOSTIC = True   # False = no diagnostic overhead, fast training
DIAG_EPOCHS = 5            # 0 = all epochs, N = first N epochs only
MULTI_TILE_DIAG = True     # Multi-tile tracking across LRTT-eligible layers (no-op when there is only one such layer, e.g. MLP linear1)
ERANK_RATE_LIMIT_STEPS = 0 # If >0, compute erank only at transfer events AND with ≥N step gap (saves SVD time). 0 = every step

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

GRAD_ACCUM_STEPS = 1

# WandB
WANDB_PROJECT = "mlp-mnist-lrtt-fine"
os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# LRTT Device Functions
# =============================================================================

def _create_ab_device(tau_sec=None, dw_min=None, multilevel=None, device_name=None):
    """Create A/B tile device based on AB_DEVICE setting.

    Options:
        6t1c              - Full 6T1C with all noise/variation (realistic)
        linearstep        - LinearStepDevice with default params (no nonlinearity, default noise)
        linearstepideal   - LinearStepDevice with all noise/dtod=0, w_max=1, w_min=-1
        constantstep      - ConstantStepDevice with default params (constant step, default noise)
        constantstepideal - ConstantStepDevice with all noise/dtod=0, w_max=1, w_min=-1
        constantstep6t1cgamma - LinearStepDevice with all noise/dtod=0 but gamma_up/gamma_down from 6t1c
        fp                - FloatingPointDevice (perfect, no quantization/bounds)
        ideal             - IdealDevice

    If multilevel (or AB_MULTILEVEL) is set to an int > 0, the ideal device branches
    (linearstepideal, constantstepideal) use w_max = 2^multilevel * dw_min / 2 and
    w_min = -w_max instead of +/-1.0. The 6t1c default branch is unaffected.
    """
    if tau_sec is None:
        tau_sec = TAU_SEC
    if dw_min is None:
        dw_min = AB_DW_MIN

    # Compute retention lifetime from tau_sec
    if tau_sec > 0:
        dt_batch_sec = 1.0
        delta = 1 - math.exp(-dt_batch_sec / tau_sec)
        lifetime = 1.0 / delta if delta > 0 else 0.0
    else:
        lifetime = 0.0

    if multilevel is None:
        multilevel = AB_MULTILEVEL
    if multilevel is not None and multilevel > 0:
        w_max = (2 ** multilevel) * dw_min / 2.0
    else:
        w_max = 1.0
    w_min = -w_max

    name = device_name if device_name is not None else AB_DEVICE
    if name == "fp":
        return FloatingPointDevice()
    if name == "ideal":
        return IdealDevice()
    if name == "linearstep":
        return LinearStepDevice(dw_min=dw_min, lifetime=lifetime)
    if name == "linearstepideal":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max, w_min=w_min,
            dw_min_dtod=0.0, dw_min_std=0.0,
            up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            gamma_up_dtod=0.0, gamma_down_dtod=0.0,
            write_noise_std=0.0, reset_std=0.0,
            up_down=0.0, mult_noise=False,
            lifetime=lifetime,
        )
    if name == "constantstep":
        return ConstantStepDevice(dw_min=dw_min, lifetime=lifetime)
    if name == "constantstepideal":
        return ConstantStepDevice(
            dw_min=dw_min,
            w_max=w_max, w_min=w_min,
            dw_min_dtod=0.0, dw_min_std=0.0,
            up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
            reset_std=0.0, up_down=0.0,
            lifetime=lifetime,
        )
    if name == "constantstep6t1cgamma":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=w_max,
            w_min=w_min,
            gamma_up=-0.1678,
            gamma_down=0.1410,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            gamma_up_dtod=0.0,
            gamma_down_dtod=0.0,
            write_noise_std=0.0,
            reset_std=0.0,
            up_down=0.0,
            mult_noise=False,
            lifetime=lifetime,
        )

    return LinearStepDevice(
        dw_min=dw_min,
        up_down=0.0, w_max=w_max, w_min=w_min,
        gamma_up=-0.1678, gamma_down=0.1410,
        mult_noise=False,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lifetime, lifetime_dtod=0.1,
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
    if C_DEVICE == "constantstep6t1cgamma":
        return LinearStepDevice(
            dw_min=dw_min,
            w_max=1.0,
            w_min=-1.0,
            gamma_up=-0.1678,
            gamma_down=0.1410,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            gamma_up_dtod=0.0,
            gamma_down_dtod=0.0,
            write_noise_std=0.0,
            reset_std=0.0,
            up_down=0.0,
            mult_noise=False,
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


def _bits_to_res(bits):
    """Convert a converter bit-count to aihwkit resolution (1/steps).

    None / <2  → no override (keep current inp_res/out_res: aihwkit default or NO_QUANT).
    N>=2       → 1/(2**N - 2)  (N-bit signed converter).
    """
    if bits is None or bits < 2:
        return None
    return 1.0 / (2 ** bits - 2)


def _apply_quant_bits(rpu_config, dac_bits, adc_bits):
    """Override forward/backward DAC (inp_res) and ADC (out_res) from bit-counts.

    Applied in BOTH create_lrtt_config and the create_frozen_analog_config standalone
    branch so every analog path (LRTT layers AND none-mode frozen layers) gets identical
    quantization. The derived frozen branch inherits via deepcopy of forward/backward.
    Has no effect under is_perfect (aihwkit ignores all IOParameters then).
    """
    dac_res = _bits_to_res(dac_bits)
    if dac_res is not None:
        rpu_config.forward.inp_res = dac_res
        rpu_config.backward.inp_res = dac_res
    adc_res = _bits_to_res(adc_bits)
    if adc_res is not None:
        rpu_config.forward.out_res = adc_res
        rpu_config.backward.out_res = adc_res
    return rpu_config


def _module_io_key(name):
    """Map an MLP Linear module name to its LAYER_IO_BITS key (None if unmatched)."""
    if 'linear1' in name:
        return 'linear1'
    if 'classifier' in name:
        return 'classifier'
    return None


def _resolve_io_bits(name, default_dac, default_adc):
    """Per-module (dac, adc) bits from LAYER_IO_BITS, falling back to defaults when None."""
    b = LAYER_IO_BITS.get(_module_io_key(name) or '', {})
    dac, adc = b.get('dac'), b.get('adc')
    return (dac if dac is not None else default_dac,
            adc if adc is not None else default_adc)


def _any_layer_io_override():
    """True if any LAYER_IO_BITS entry sets a non-default (non-None) dac/adc bit-count."""
    return any((b.get('dac') is not None or b.get('adc') is not None)
               for b in LAYER_IO_BITS.values())


def create_frozen_analog_config(lrtt_config=None, out_noise=0.0, dac_bits=None, adc_bits=None):
    """Create analog config for non-LRTT linear layers (frozen analog).

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
        # Apply (possibly per-module) bits, overriding the inherited quantization.
        # No-op when dac_bits/adc_bits are None (default path) — preserves prior behavior.
        _apply_quant_bits(rpu_config, dac_bits, adc_bits)
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
        _apply_quant_bits(rpu_config, DAC_BITS, ADC_BITS)
        if BACKWARD_OUT_BOUND != 12.0:
            rpu_config.backward.out_bound = BACKWARD_OUT_BOUND
    return rpu_config


def create_lrtt_config():
    """Create LRTT RPU configuration for analog layers."""
    # A and B tile devices: independently overridable via A_DEVICE / B_DEVICE.
    # When both are None, both A and B use AB_DEVICE (legacy behavior preserved).
    a_device = _create_ab_device(device_name=A_DEVICE)
    b_device = _create_ab_device(device_name=B_DEVICE)
    c_device = _create_c_device()

    # Scale B init if multilevel is set (B init scales with new w_max/w_min).
    if AB_MULTILEVEL is not None and AB_MULTILEVEL > 0:
        ab_w_max = (2 ** AB_MULTILEVEL) * AB_DW_MIN / 2.0
        b_reinit_gain = REINIT_GAIN * ab_w_max
    else:
        b_reinit_gain = REINIT_GAIN

    te = TRANSFER_EVERY if not NO_TRANSFER else 10 ** 9
    device_config = PythonLRTTDevice(
        rank=LRTT_RANK,
        transfer_every=te,
        lora_alpha=1.0,
        fast_lr=(FAST_LR if not NO_TRANSFER else 1e-30),
        reinit_gain=b_reinit_gain,
        reinit_mode=REINIT_MODE,
        unit_cell_devices=[a_device, b_device, c_device],
        train_c_bias=False,
        mapping_ab=MappingParameter(
            weight_scaling_omega=AB_WEIGHT_SCALING_OMEGA,
            learn_out_scaling=False,
            max_input_size=0 if IS_PERFECT else 512,
            max_output_size=0 if IS_PERFECT else 512,
        ),
        mapping_c=MappingParameter(
            weight_scaling_omega=C_WEIGHT_SCALING_OMEGA,
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
    device_config.a_density = A_DENSITY
    device_config.b_density = B_DENSITY
    device_config.forward_inject = FORWARD_INJECT
    device_config.fi_continuous_alpha = FI_CONTINUOUS_ALPHA
    device_config.ab_io_perfect = AB_IO_PERFECT
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
    rpu_config.mapping.max_input_size = 0 if IS_PERFECT else 512
    rpu_config.mapping.max_output_size = 0 if IS_PERFECT else 512
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
    _apply_quant_bits(rpu_config, DAC_BITS, ADC_BITS)

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

    Returns list of substrings identifying which linear layers should be LRTT.
    Returns [] for none mode (fully digital baseline).
    """
    if lora_target == "none":
        return []  # No LRTT (digital baseline)
    elif lora_target == "linear1":
        return ["linear1"]  # Only LRTT-able layer in 2-layer MLP (classifier excluded)
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


class MLP(nn.Module):
    """2-layer MLP for MNIST: 784 → HIDDEN_DIM → 10.

    Layer naming convention:
        - linear1: input projection (LRTT-eligible)
        - classifier: output head (always digital, controlled by HEAD_LAYER/HEAD_ANALOG)
    """
    def __init__(self, hidden=HIDDEN_DIM):
        super().__init__()
        self.linear1 = nn.Linear(784, hidden)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten 28×28 → 784
        x = self.relu(self.linear1(x))
        x = self.classifier(x)
        return self.logsoftmax(x)


def create_model():
    """Create MLP model for MNIST with selective LRTT analog layers.

    Architecture:
        - LRTT Target layer (based on LORA_TARGET) → LRTT Analog
        - classifier (output head) → Digital TRAINABLE / FROZEN / Frozen Analog
          (controlled by HEAD_LAYER, HEAD_ANALOG)

    LoRA Target Options (LORA_TARGET):
        - none: fully digital baseline
        - linear1: linear1 → LRTT Analog (default)

    LRTT layers have:
        - A/B tiles: TRAINABLE
        - C-tile: FROZEN (initial weights)
        - out_scaling: TRAINABLE
        - bias: FROZEN
    """
    from aihwkit.nn import AnalogLinear

    model = MLP(hidden=HIDDEN_DIM)

    # Get LRTT target patterns
    lrtt_patterns = get_lrtt_target_module_names(LORA_TARGET)

    def is_lrtt_target(layer_name):
        """Check if layer should be converted to LRTT Analog."""
        # classifier (output head) is always digital
        if "classifier" in layer_name:
            return False
        if not lrtt_patterns:
            return False  # none mode
        return any(p in layer_name for p in lrtt_patterns)

    # Build exclude list: all layers that should NOT be converted to LRTT
    all_linear_names = list_linear_layers(model)
    exclude_modules = []
    for name in all_linear_names:
        if not is_lrtt_target(name):
            # Use full path for exclude_modules (convert_to_analog requires exact match)
            exclude_modules.append(name)

    # Exclude classifier (output head, always digital — controlled by HEAD_LAYER/HEAD_ANALOG)
    exclude_modules.append("classifier")
    exclude_modules = list(set(exclude_modules))  # Remove duplicates

    # Step 1: Convert only LRTT target layers to LRTT Analog (skip if none mode)
    if LORA_TARGET == "none":
        # None mode: fully digital, no analog conversion
        num_analog = 0
    else:
        lrtt_config = create_lrtt_config()

        # Convert to analog with exclusions (only LRTT targets get converted).
        # Per-module IO bits: with LAYER_IO_BITS overrides, convert each bit-group in its
        # own pass (build-time bits — post-conversion override does NOT propagate to tiles).
        if not _any_layer_io_override():
            model = convert_to_analog(model, lrtt_config, exclude_modules=exclude_modules)
        else:
            import copy as _copy
            _groups = {}
            for _n in all_linear_names:
                if is_lrtt_target(_n):
                    _key = _resolve_io_bits(_n, DAC_BITS, ADC_BITS)
                    _groups.setdefault(_key, []).append(_n)
            for (_gd, _ga), _names in _groups.items():
                _cfg = _copy.deepcopy(lrtt_config)
                _apply_quant_bits(_cfg, _gd, _ga)
                _excl = [_n for _n in all_linear_names if _n not in _names]
                model = convert_to_analog(model, _cfg, exclude_modules=_excl)

        # Count analog layers
        num_analog = count_analog_layers(model)

    # Step 1.5: Convert non-LRTT linear and/or classifier to frozen analog (if enabled)
    frozen_analog_count = 0
    any_frozen_analog = (ENCODER_ANALOG and LORA_TARGET != "linear1") or HEAD_ANALOG
    if any_frozen_analog:
        # Collect existing tile IDs (LRTT sub-tiles) before frozen conversion
        existing_tile_ids = set()
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    existing_tile_ids.add(id(tile))

        _lc = lrtt_config if LORA_TARGET != "none" else None
        frozen_exclude = []
        if not HEAD_ANALOG:
            frozen_exclude.append("classifier")
        # Exclude already-LRTT layers (don't re-convert) and non-LRTT non-classifier
        # linear layers (if ENCODER_ANALOG is off — leave them digital).
        for name in all_linear_names:
            if is_lrtt_target(name):
                frozen_exclude.append(name)            # already LRTT
            elif not ENCODER_ANALOG and "classifier" not in name:
                frozen_exclude.append(name)            # non-LRTT linear, keep digital
        if not _any_layer_io_override():
            frozen_config = create_frozen_analog_config(
                _lc, dac_bits=DAC_BITS, adc_bits=ADC_BITS,
            )
            model = convert_to_analog(model, frozen_config, exclude_modules=frozen_exclude)
        else:
            # Per-module IO bits for frozen-analog layers: group by resolved bits.
            _ftargets = [_n for _n in all_linear_names
                         if _n not in frozen_exclude and not is_lrtt_target(_n)]
            _groups = {}
            for _n in _ftargets:
                _key = _resolve_io_bits(_n, DAC_BITS, ADC_BITS)
                _groups.setdefault(_key, []).append(_n)
            for (_gd, _ga), _names in _groups.items():
                _cfg = create_frozen_analog_config(_lc, dac_bits=_gd, adc_bits=_ga)
                _excl = [_n for _n in all_linear_names if _n not in _names]
                model = convert_to_analog(model, _cfg, exclude_modules=_excl)
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
    # - LRTT layers: A/B TRAINABLE, C hook-protected (transfer-only), bias TRAINABLE (from-scratch)
    # - classifier: TRAINABLE if HEAD_LAYER=="train", else FROZEN
    # - Other linear layers (e.g., linear1 in "none" mode, future linear2/3): TRAINABLE (from-scratch)
    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            param.requires_grad = not NO_TRANSFER
        elif "tile_c" in name:
            pass  # tile_c.weight hook-protected (transfer-only); bias via train_c_bias in lrtt_tile.py
        elif "out_scaling_alpha" in name:
            pass  # out_scaling: trainable per mapping_c
        elif "classifier" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        else:
            param.requires_grad = True  # digital layers (incl. future linear2/3): trainable from-scratch

    num_params = count_parameters(model)

    print(f"\nCreated MLP model (LRTT):")
    print(f"  Architecture: 784 → {HIDDEN_DIM} → 10")
    print(f"  Total params: {total_params:,}, Trainable: {num_params:,}")
    print(f"  LRTT Analog layers: {num_analog}")
    print(f"  LRTT config: rank={LRTT_RANK}, transfer_every={TRANSFER_EVERY}, "
          f"transfer_lr={TRANSFER_LR}, fast_lr={FAST_LR}, auto_scale={AUTO_SCALE_MODE}")
    print(f"  Reinit: mode={REINIT_MODE}, gain={REINIT_GAIN}")
    print(f"  LoRA target: {LORA_TARGET} -> {lrtt_patterns if lrtt_patterns else 'none (digital baseline)'}")

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

def load_data():
    """Load MNIST dataset and return (train_loader, val_loader)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST(root="/tmp/mnist", train=True, download=True, transform=transform)
    val_set = datasets.MNIST(root="/tmp/mnist", train=False, download=True, transform=transform)

    if TRAIN_SUBSET_SIZE > 0:
        indices = torch.randperm(
            len(train_set), generator=torch.Generator().manual_seed(SEED)
        )[:TRAIN_SUBSET_SIZE].tolist()
        train_set = Subset(train_set, indices)
    if EVAL_SUBSET_SIZE > 0:
        val_set = Subset(val_set, range(min(EVAL_SUBSET_SIZE, len(val_set))))

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE // GRAD_ACCUM_STEPS, shuffle=True,
        num_workers=2, generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_set, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=2,
    )

    return train_loader, val_loader


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(model, val_loader):
    """Evaluate MLP model on MNIST. Returns (accuracy_pct, val_loss)."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.NLLLoss(reduction='sum')
    with no_grad():
        for data, target in val_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    model.train()
    acc = 100.0 * correct / max(1, total)
    avg_loss = total_loss / max(1, total)
    return acc, avg_loss


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
    for name, mod in model.named_modules():
        if hasattr(mod, 'controller'):
            return name, mod
    raise RuntimeError("No LRTT tile found")


def find_last_lrtt_tile(model):
    last_name, last_tile = None, None
    for name, mod in model.named_modules():
        if hasattr(mod, 'controller'):
            last_name, last_tile = name, mod
    if last_tile is None:
        raise RuntimeError("No LRTT tile found")
    return last_name, last_tile


def find_target_lrtt_tiles(model, layer_indices=(0, 6, 11),
                            sublayers=("query", "key", "value", "attention.output")):
    """Find LRTT tiles for specific layers and sublayers.

    Returns:
        dict mapping short_name → (full_name, module)
        e.g. "L0_query" → ("<full_module_path>.analog_module", tile)
        Pattern matching here is BERT-style (layer.<i>.attention.self.<sublayer>); for MLP
        this returns an empty dict since MLP has no such tiles.
    """
    tiles = {}
    for name, mod in model.named_modules():
        if not hasattr(mod, 'controller'):
            continue
        for li in layer_indices:
            for sl in sublayers:
                # Match patterns like "layer.0.attention.self.query" or "layer.0.attention.output.dense"
                layer_str = f"layer.{li}."
                if layer_str in name and sl in name:
                    short = f"L{li}_{sl.split('.')[-1]}"
                    tiles[short] = (name, mod)
    return tiles


def sample_cells(weight_matrix, cell_indices):
    values = []
    for r, c in cell_indices:
        if r < weight_matrix.shape[0] and c < weight_matrix.shape[1]:
            values.append(weight_matrix[r, c].item())
        else:
            values.append(0.0)
    return values


def get_raw_C(tile_c):
    W_scaled = tile_c.get_weights()[0]
    alpha = tile_c.get_scales()
    if alpha is not None:
        return W_scaled / alpha.to(W_scaled.device).unsqueeze(1)
    return W_scaled


def snapshot_weights(tile):
    return (
        tile.tile_a.get_weights()[0].clone().detach(),
        tile.tile_b.get_weights()[0].clone().detach(),
        tile.tile_c.get_weights()[0].clone().detach(),
        get_raw_C(tile.tile_c).clone().detach(),
    )


def collect_tile_diagnostics(tile, C_prev_raw, A_before, B_before, C_before,
                             C_raw_before, step, prev_num_transfers,
                             A_ci, B_ci, C_ci, A_pre_transfer=None,
                             C_initial_eff=None, compute_erank=True):
    controller = tile.controller
    A = A_pre_transfer if A_pre_transfer is not None else tile.tile_a.get_weights()[0]
    B = tile.tile_b.get_weights()[0]
    C_eff = tile.tile_c.get_weights()[0]
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

    num_transfers = controller.num_transfers
    is_transfer = num_transfers > prev_num_transfers

    record = {
        "step": step,
        "norm_A": norm_A, "norm_B": norm_B,
        "norm_C_raw": norm_C_raw, "norm_AB": norm_AB,
        "A_eff_min": A.min().item(), "A_eff_max": A.max().item(),
        "B_eff_min": B.min().item(), "B_eff_max": B.max().item(),
        "C_eff_min": C_eff.min().item(), "C_eff_max": C_eff.max().item(),
        "C_raw_min": C_raw.min().item(), "C_raw_max": C_raw.max().item(),
        "A_cells": A_cells, "B_cells": B_cells, "C_cells": C_cells,
        "A_grad_cells": A_grad_cells, "B_grad_cells": B_grad_cells,
        "C_grad_cells": C_grad_cells,
        "delta_A": delta_A, "delta_B": delta_B, "delta_C_raw": delta_C_raw_step,
        "transfer_counter": controller.transfer_counter,
        "num_transfers": num_transfers, "is_transfer": is_transfer,
    }
    if compute_erank:
        record["erank_C"] = _effective_rank(C_eff)
        record["erank_C_delta"] = _effective_rank(C_eff - C_initial_eff) if C_initial_eff is not None else 0.0
        record["erank_A"] = _effective_rank(A)
        record["erank_B"] = _effective_rank(B)
        record["erank_AB"] = _effective_rank(A @ B)
    else:
        record["erank_C"] = None
        record["erank_C_delta"] = None
        record["erank_A"] = None
        record["erank_B"] = None
        record["erank_AB"] = None
    return record, C_raw.clone().detach(), num_transfers


def _compute_multi_mean(multi_logs):
    """Compute per-step mean of numeric metrics across all tracked tiles."""
    # Get all tile keys that have data
    keys_with_data = [k for k, v in multi_logs.items() if v]
    if not keys_with_data:
        return []
    n_steps = len(multi_logs[keys_with_data[0]])
    # Numeric fields to average
    fields = ["norm_A", "norm_B", "norm_C_raw", "norm_AB",
              "A_eff_min", "A_eff_max", "B_eff_min", "B_eff_max",
              "C_eff_min", "C_eff_max", "C_raw_min", "C_raw_max",
              "erank_C", "erank_C_delta", "erank_A", "erank_B", "erank_AB"]
    mean_log = []
    for i in range(n_steps):
        rec = {"step": multi_logs[keys_with_data[0]][i]["step"]}
        for f in fields:
            vals = [multi_logs[k][i].get(f) for k in keys_with_data
                    if multi_logs[k][i].get(f) is not None]
            rec[f] = (sum(vals) / len(vals)) if vals else None
        mean_log.append(rec)
    return mean_log


def _effective_rank(M):
    """Compute effective rank = exp(entropy of normalized singular values)."""
    s = torch.linalg.svdvals(M.float().cuda())
    s = s[s > 1e-10]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()
    return entropy.exp().item()


def _cos_sim(a, b):
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

    a_ci = A_ci or [(i, 0) for i in range(n_cells)]
    b_ci = B_ci or [(0, i) for i in range(n_cells)]
    c_ci = C_ci or [(i, i) for i in range(len(log_data[0]["C_cells"]))]

    fig, axes = plt.subplots(6, 2, figsize=(18, 34))
    fig.suptitle(f"LRTT Diagnostic — {tile_label}" if tile_label else "LRTT Diagnostic",
                 fontsize=14, y=1.01)

    def tl(ax):
        for ts in transfer_steps:
            ax.axvline(x=ts, color="red", alpha=0.3, linewidth=0.8)

    A_eff_mins = [r.get("A_eff_min", r.get("A_min", 0)) for r in log_data]
    A_eff_maxs = [r.get("A_eff_max", r.get("A_max", 0)) for r in log_data]
    B_eff_mins = [r.get("B_eff_min", r.get("B_min", 0)) for r in log_data]
    B_eff_maxs = [r.get("B_eff_max", r.get("B_max", 0)) for r in log_data]
    C_eff_mins = [r.get("C_eff_min", 0) for r in log_data]
    C_eff_maxs = [r.get("C_eff_max", 0) for r in log_data]
    C_raw_mins = [r.get("C_raw_min", r.get("C_min", 0)) for r in log_data]
    C_raw_maxs = [r.get("C_raw_max", r.get("C_max", 0)) for r in log_data]

    # (0,0) A, B norms + eff min/max
    ax = axes[0, 0]
    ax.plot(steps, norm_A, label="||A||", alpha=0.8)
    ax.plot(steps, norm_B, label="||B||", alpha=0.8)
    ax.plot(steps, norm_AB, label="||A@B||", alpha=0.6, linestyle="--")
    ax_mm = ax.twinx()
    ax_mm.plot(steps, A_eff_maxs, label="A eff max", color="red", alpha=0.5, linewidth=0.7, linestyle=":")
    ax_mm.plot(steps, A_eff_mins, label="A eff min", color="red", alpha=0.5, linewidth=0.7, linestyle="--")
    ax_mm.plot(steps, B_eff_maxs, label="B eff max", color="blue", alpha=0.5, linewidth=0.7, linestyle=":")
    ax_mm.plot(steps, B_eff_mins, label="B eff min", color="blue", alpha=0.5, linewidth=0.7, linestyle="--")
    ax_mm.set_ylabel("eff min/max", fontsize=8)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Norm")
    ax.set_title("A, B, AB Norms + eff min/max (red = transfer)")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax_mm.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # (0,1) C norm + delta_C
    ax = axes[0, 1]
    ax.plot(steps, norm_C_raw, label="||C_raw||", color="green", alpha=0.8)
    delta_C = [r["delta_C_raw"] for r in log_data]
    ax2 = ax.twinx()
    ax2.plot(steps, delta_C, label="delta_C_raw", color="orange", alpha=0.8)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("||C_raw||", color="green")
    ax2.set_ylabel("delta_C_raw", color="orange")
    ax.set_title("C Norm (raw) + delta_C_raw")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, loc="upper left", fontsize=6); ax.grid(True, alpha=0.3)

    # (1,0) C raw + eff min/max combined
    ax = axes[1, 0]
    ax.plot(steps, C_raw_maxs, label="C raw max", color="red", alpha=0.8, linewidth=1.0)
    ax.plot(steps, C_raw_mins, label="C raw min", color="red", alpha=0.8, linewidth=1.0, linestyle="--")
    ax.plot(steps, C_eff_maxs, label="C eff max", color="purple", alpha=0.8, linewidth=1.0)
    ax.plot(steps, C_eff_mins, label="C eff min", color="purple", alpha=0.8, linewidth=1.0, linestyle="--")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.4)
    ax.axhline(y=-1.0, color="gray", linestyle=":", alpha=0.4)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Weight value")
    ax.set_title("C weight min/max (raw=red, eff=purple)")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # (1,1) Effective rank (filter None entries from rate-limiting)
    er_steps = [r["step"] for r in log_data if r.get("erank_C") is not None]
    erank_C = [r["erank_C"] for r in log_data if r.get("erank_C") is not None]
    erd_steps = [r["step"] for r in log_data if r.get("erank_C_delta") is not None]
    erank_C_delta = [r["erank_C_delta"] for r in log_data if r.get("erank_C_delta") is not None]
    ax = axes[1, 1]
    if er_steps:
        ax.plot(er_steps, erank_C, label="erank(C)", color="green", alpha=0.8, linewidth=1.0, marker='.', markersize=3)
    if erd_steps:
        ax.plot(erd_steps, erank_C_delta, label="erank(C - C_init)", color="blue", alpha=0.8, linewidth=1.0, marker='.', markersize=3)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Effective rank")
    ax.set_title("Effective rank of C and C delta")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # (2,0)-(4,1) A, B, C cell weights and deltas
    for row, (ws, gs, ci, nm) in enumerate(
            [(A_w, A_g, a_ci, "A"), (B_w, B_g, b_ci, "B"), (C_w, C_g, c_ci, "C")], start=2):
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

    # (5,0) G_accum norm (line) + tlr*AB and dC norms at transfers (scatter) + loss
    nG = [max(r.get("norm_G_accum", 1e-10), 1e-10) for r in log_data]
    t_steps_dC = [r["step"] for r in log_data if r["is_transfer"]]
    t_norms_dC = [max(r.get("norm_dC_step", 1e-10), 1e-10) for r in log_data if r["is_transfer"]]
    t_norms_tlr = [max(r.get("norm_tlrAB", 1e-10), 1e-10) for r in log_data if r["is_transfer"]]
    ax = axes[5, 0]
    ax.semilogy(steps, nG, label="||G_accum||", color="red", alpha=0.8, linewidth=0.8)
    if t_steps_dC:
        ax.semilogy(t_steps_dC, t_norms_tlr, '^', label="||tlr*A@B|| @T", color="green",
                     markersize=5, alpha=0.9, zorder=5)
        ax.semilogy(t_steps_dC, t_norms_dC, 'o', label="||delta_C|| @T", color="blue",
                     markersize=5, alpha=0.9, zorder=5)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Norm (log)")
    axl = ax.twinx(); axl.plot(steps, losses, label="loss", color="gray", alpha=0.35, linewidth=0.6)
    axl.set_ylabel("Loss", color="gray")
    lm, llm = ax.get_legend_handles_labels(); ll, lll = axl.get_legend_handles_labels()
    ax.legend(lm+ll, llm+lll, fontsize=7, loc="upper right")
    ax.set_title("||G_accum|| vs ||tlr*A@B|| + ||delta_C|| at transfers + Loss"); ax.grid(True, alpha=0.3)

    # (5,1) cosines at transfer steps only (G_accum is reset per transfer)
    t_cTG = [r.get("cos_tlrAB_G", 0) for r in log_data if r["is_transfer"]]
    t_cDG = [r.get("cos_dC_G", 0) for r in log_data if r["is_transfer"]]
    t_cDT = [r.get("cos_dC_tlrAB", 0) for r in log_data if r["is_transfer"]]
    ax = axes[5, 1]
    if t_steps_dC:
        ax.scatter(t_steps_dC, t_cTG, label="cos(tlr*AB, G) @T", color="green",
                   s=25, alpha=0.9, zorder=5, marker="^")
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

    plt.tight_layout(rect=[0, 0, 1, 0.98])
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
        ('xc', 'tile_c transfer input (B weights)'),
        ('dc', 'tile_c transfer grad (A weights)'),
    ]

    fig, axes = plt.subplots(6, 2, figsize=(18, 27))
    fig.suptitle(f'x/d Distribution — {tile_label}', fontsize=13, y=0.99)

    for row, (prefix, desc) in enumerate(xd_keys):
        is_transfer_key = prefix in ('xc', 'dc')

        # For xc/dc, filter to transfer steps only
        if is_transfer_key:
            plot_data = [r for r in log_data if r.get('is_transfer') and f'{prefix}_abs_max' in r]
            plot_steps = [r['step'] for r in plot_data]
        else:
            plot_data = log_data
            plot_steps = steps

        # --- Left: percentile band plot over time ---
        ax = axes[row, 0]
        if plot_data:
            p5 = [r.get(f'{prefix}_p5', 0) for r in plot_data]
            p25 = [r.get(f'{prefix}_p25', 0) for r in plot_data]
            p50 = [r.get(f'{prefix}_p50', 0) for r in plot_data]
            p75 = [r.get(f'{prefix}_p75', 0) for r in plot_data]
            p95 = [r.get(f'{prefix}_p95', 0) for r in plot_data]
            mean_vals = [r.get(f'{prefix}_abs_mean', 0) for r in plot_data]
            max_vals = [r.get(f'{prefix}_abs_max', 0) for r in plot_data]

            ax.fill_between(plot_steps, p5, p95, alpha=0.15, color='blue', label='p5-p95')
            ax.fill_between(plot_steps, p25, p75, alpha=0.3, color='blue', label='p25-p75')
            ax.plot(plot_steps, p50, 'b-', linewidth=0.8, label='median')
            ax.plot(plot_steps, mean_vals, 'g--', linewidth=0.6, alpha=0.7, label='mean')
            ax.plot(plot_steps, max_vals, 'r-', linewidth=0.4, alpha=0.5, label='max')
        ax.set_title(f'|{prefix}| percentiles — {desc}')
        ax.set_ylabel(f'|{prefix}|')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        if row == len(xd_keys) - 1:
            ax.set_xlabel('Step')

        # --- Right: histograms at sampled time points ---
        ax = axes[row, 1]
        # xc/dc histograms are in 'xc_dc_hist', xa/da/xb/db in 'xd_hist'
        hist_key = 'xc_dc_hist' if is_transfer_key else 'xd_hist'
        hist_steps = [r for r in plot_data if hist_key in r and prefix in r[hist_key]]
        if hist_steps:
            n_hist = len(hist_steps)
            sample_idx = [0, n_hist // 3, 2 * n_hist // 3, n_hist - 1]
            sample_idx = sorted(set(min(i, n_hist - 1) for i in sample_idx))
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sample_idx)))
            for ci, idx in enumerate(sample_idx):
                h = hist_steps[idx][hist_key][prefix]
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
        if row == len(xd_keys) - 1:
            ax.set_xlabel(f'|{prefix}|')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def make_multi_tile_plots(multi_logs, mean_log, output_path):
    """Plot key metrics across all tracked tiles + mean."""
    if not mean_log:
        return

    metrics = [
        ("norm_A", "||A|| (pre-transfer)"),
        ("norm_AB", "||A@B||"),
        ("erank_C", "Effective Rank of C"),
        ("erank_C_delta", "Effective Rank of C - C_init"),
        ("A_eff_max", "A weight max"),
        ("C_raw_max", "C raw conductance max"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Multi-Tile Diagnostic Comparison", fontsize=14, y=1.01)

    # Color map for tiles
    tile_keys = sorted(multi_logs.keys())
    colors = plt.cm.tab20(range(len(tile_keys)))

    for idx, (field, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Plot each tile (filter None for rate-limited erank fields)
        for i, k in enumerate(tile_keys):
            steps_data = multi_logs[k]
            if not steps_data:
                continue
            pairs = [(s["step"], s.get(field)) for s in steps_data if s.get(field) is not None]
            if not pairs:
                continue
            steps_v, vals_v = zip(*pairs)
            ax.plot(steps_v, vals_v, color=colors[i], alpha=0.4, linewidth=0.7, label=k)

        # Plot mean (filter None)
        mpairs = [(s["step"], s.get(field)) for s in mean_log if s.get(field) is not None]
        if mpairs:
            steps_v, vals_v = zip(*mpairs)
            ax.plot(steps_v, vals_v, color="black", linewidth=2.0, label="mean")

        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.set_title(title)

    # Single legend outside
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
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


# =============================================================================
# Main
# =============================================================================

def main():
    """Train MLP with LRTT on MNIST."""
    manual_seed(SEED)
    np.random.seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"mlp_lrtt_r{LRTT_RANK}_te{TRANSFER_EVERY}_bs{BATCH_SIZE}",
        config={
            "model": "MLP", "dataset": "MNIST",
            "hidden_dim": HIDDEN_DIM,
            "lrtt_rank": LRTT_RANK, "transfer_every": TRANSFER_EVERY,
            "transfer_lr": TRANSFER_LR, "fast_lr": FAST_LR, "auto_scale_mode": AUTO_SCALE_MODE,
            "reinit_mode": REINIT_MODE, "reinit_gain": REINIT_GAIN,
            "tau_sec": TAU_SEC,
            "dynamic_te": DYNAMIC_TE, "te_warmup_steps": TE_WARMUP_STEPS,
            "epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "optimizer": OPTIMIZER, "step_lr_size": STEP_LR_SIZE,
            "step_lr_gamma": STEP_LR_GAMMA, "seed": SEED,
            "lora_target": LORA_TARGET,
        }
    )

    # Load data
    train_loader, val_loader = load_data()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model, optimizer, scheduler
    model = create_model()
    optimizer = create_optimizer(model)

    scheduler = StepLR(optimizer, step_size=STEP_LR_SIZE, gamma=STEP_LR_GAMMA)

    # =========================================================================
    # Diagnostic setup (skipped if ENABLE_DIAGNOSTIC=False)
    # =========================================================================
    first_gc, last_gc = {}, {}
    first_log, last_log = [], []
    first_C_prev_raw, last_C_prev_raw = None, None
    first_C_initial_eff, last_C_initial_eff = None, None
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

        # Rate-limit erank computation for first/last tile (gated by ERANK_RATE_LIMIT_STEPS)
        first_last_erank_step = -10**9
        last_last_erank_step = -10**9

        A_shape = tuple(first_tile.tile_a.get_weights()[0].shape)
        B_shape = tuple(first_tile.tile_b.get_weights()[0].shape)
        C_shape = tuple(first_tile.tile_c.get_weights()[0].shape)
        A_CI = _make_cell_indices(A_shape)
        B_CI = _make_cell_indices(B_shape)
        C_CI = _make_cell_indices(C_shape)

        # Capture initial C weights for effective rank of delta
        first_C_initial_eff = first_tile.tile_c.get_weights()[0].clone().detach()
        last_C_initial_eff = last_tile.tile_c.get_weights()[0].clone().detach()

        print(f"\nDiag tile (first): {first_name}  A{A_shape} B{B_shape} C{C_shape}")
        print(f"Diag tile (last):  {last_name}")
        print(f"Diag epochs: {'all' if DIAG_EPOCHS == 0 else f'first {DIAG_EPOCHS}'}")

        # Multi-tile tracking: BERT-style layer×sublayer scan (empty for MLP, no harm)
        if MULTI_TILE_DIAG:
            multi_tiles = find_target_lrtt_tiles(model)
        else:
            multi_tiles = {}
            print(f"  Multi-tile diagnostic: DISABLED")
        multi_logs = {k: [] for k in multi_tiles}
        multi_C_initial = {}
        multi_last_erank_step = {k: -10**9 for k in multi_tiles}
        for k, (tname, tmod) in multi_tiles.items():
            multi_C_initial[k] = tmod.tile_c.get_weights()[0].clone().detach()
            tmod.controller.enable_diagnostics = True
            print(f"  Multi-diag: {k} → {tname}")

        def _collect_multi_tile_metrics(step):
            """Collect lightweight metrics for all tracked tiles."""
            if not multi_tiles:
                return
            for k, (tname, tmod) in multi_tiles.items():
                A = tmod.tile_a.get_weights()[0]
                B = tmod.tile_b.get_weights()[0]
                C_eff = tmod.tile_c.get_weights()[0]
                C_raw = get_raw_C(tmod.tile_c)
                ctrl = tmod.controller
                rec = {
                    "step": step,
                    "norm_A": torch.norm(A).item(),
                    "norm_B": torch.norm(B).item(),
                    "norm_C_raw": torch.norm(C_raw).item(),
                    "norm_AB": torch.norm(A @ B).item(),
                    "mean_A": A.mean().item(), "mean_B": B.mean().item(),
                    "mean_C_raw": C_raw.mean().item(), "mean_C_eff": C_eff.mean().item(),
                    "A_eff_min": A.min().item(), "A_eff_max": A.max().item(),
                    "B_eff_min": B.min().item(), "B_eff_max": B.max().item(),
                    "C_eff_min": C_eff.min().item(), "C_eff_max": C_eff.max().item(),
                    "C_raw_min": C_raw.min().item(), "C_raw_max": C_raw.max().item(),
                    "num_transfers": ctrl.num_transfers,
                    "is_transfer": False,  # updated below
                }
                # Detect transfer
                if multi_logs[k]:
                    rec["is_transfer"] = ctrl.num_transfers > multi_logs[k][-1]["num_transfers"]
                multi_logs[k].append(rec)

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
                    # Guard: with grad_accum>1 + FI, tile_a/b process groups
                    # independently so x_b and d_a may come from different
                    # micro-batches with different seq lengths (dynamic padding)
                    # Truncate to the shorter batch dim so G_accum is always updated
                    min_batch = min(d_2d.shape[0], x_2d.shape[0])
                    gc_dict['G_accum'] = gc_dict['G_accum'] + d_2d[:min_batch].t() @ x_2d[:min_batch]
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

            # ── Transfer diagnostic: capture x/d going into C tile ──
            # - direct/onehot: hook tile_c._orig_update / tile_c.update
            # - set mode: tile_c.tile.update is C-ext (read-only), so we
            #   read A/B weights before transfer (= the x/d that set passes)
            gc_dict['_transfer_xc_all'] = []
            gc_dict['_transfer_dc_all'] = []
            gc_dict['_transfer_lr_c'] = 0.0
            gc_dict['_in_transfer'] = False

            _orig_transfer = ctrl.ab_weight_transfer

            # Hook outer tile_c update (used by direct/onehot)
            _orig_outer_update = getattr(diag_tile.tile_c, '_orig_update',
                                         diag_tile.tile_c.update)

            def _capture_c_update_outer(x_input, d_input, *args, **kwargs):
                if gc_dict.get('_in_transfer'):
                    gc_dict['_transfer_xc_all'].append(x_input.detach().clone())
                    gc_dict['_transfer_dc_all'].append(d_input.detach().clone())
                    gc_dict['_transfer_lr_c'] = diag_tile.tile_c.get_learning_rate()
                return _orig_outer_update(x_input, d_input, *args, **kwargs)

            def _compute_transfer_stats(gc_dict):
                """Compute xc/dc stats from accumulated tensors."""
                with torch.no_grad():
                    _pcts = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95],
                                         device=device)
                    xc_list = gc_dict['_transfer_xc_all']
                    dc_list = gc_dict['_transfer_dc_all']
                    if xc_list:
                        xc_cat = torch.cat(xc_list, dim=0).to(device)
                        dc_cat = torch.cat(dc_list, dim=0).to(device)
                        gc_dict['xc_abs_mean'], gc_dict['xc_abs_max'] = _abs_stats(xc_cat)
                        gc_dict['dc_abs_mean'], gc_dict['dc_abs_max'] = _abs_stats(dc_cat)
                        for _prefix, _t in [('xc', xc_cat), ('dc', dc_cat)]:
                            _flat = _t.abs().flatten()
                            _q = torch.quantile(_flat.float(), _pcts).tolist()
                            gc_dict[f'{_prefix}_p5'] = _q[0]
                            gc_dict[f'{_prefix}_p25'] = _q[1]
                            gc_dict[f'{_prefix}_p50'] = _q[2]
                            gc_dict[f'{_prefix}_p75'] = _q[3]
                            gc_dict[f'{_prefix}_p95'] = _q[4]
                        _hists_c = {}
                        for _prefix, _t in [('xc', xc_cat), ('dc', dc_cat)]:
                            _flat = _t.abs().flatten().float()
                            _max_val = _flat.max().item()
                            if _max_val > 0:
                                _counts = torch.histc(_flat, bins=50, min=0, max=_max_val).tolist()
                                _hists_c[_prefix] = {'counts': _counts, 'min': 0.0, 'max': _max_val}
                            else:
                                _hists_c[_prefix] = {'counts': [float(_flat.numel())] + [0.0]*49, 'min': 0.0, 'max': 1.0}
                        gc_dict['_transfer_hist'] = _hists_c
                        gc_dict['transfer_lr_c'] = gc_dict['_transfer_lr_c']
                        gc_dict['transfer_n_calls'] = len(xc_list)
                        del xc_cat, dc_cat
                    gc_dict['_transfer_xc_all'] = []
                    gc_dict['_transfer_dc_all'] = []

            def hooked_transfer(method=None):
                if not gc_dict.get('active'):
                    _orig_transfer(method=method)
                    return

                # Capture A weights before transfer (will be reinit'd after)
                with torch.no_grad():
                    gc_dict['_A_pre_transfer'] = diag_tile.tile_a.get_weights()[0].clone().detach()

                _method = method if method is not None else ctrl.transfer_method
                gc_dict['_transfer_xc_all'] = []
                gc_dict['_transfer_dc_all'] = []
                gc_dict['_in_transfer'] = True

                # For set mode: C-ext tile.update is read-only, so read A/B
                # weights before transfer to reconstruct the x/d values
                if _method == "set":
                    with torch.no_grad():
                        A_raw = diag_tile.tile_a.tile.get_weights().to(device)
                        B_raw = diag_tile.tile_b.tile.get_weights().to(device)
                        alpha_a = diag_tile.tile_a.get_scales()
                        alpha_b = diag_tile.tile_b.get_scales()
                        alpha_c = diag_tile.tile_c.get_scales()
                        A_val = A_raw * alpha_a.view(-1, 1) if alpha_a is not None else A_raw
                        B_val = B_raw * alpha_b.view(-1, 1) if alpha_b is not None else B_raw
                        A_adj = A_val / alpha_c.view(-1, 1) if alpha_c is not None else A_val
                        # set mode passes: x=B_val [rank, x_size], d=(-A_adj).t() [rank, d_size]
                        gc_dict['_transfer_xc_all'].append(B_val.detach())
                        gc_dict['_transfer_dc_all'].append((-A_adj).t().detach())
                        eff_lr = ctrl._compute_effective_transfer_lr()
                        gc_dict['_transfer_lr_c'] = abs(eff_lr)

                _orig_transfer(method=method)
                gc_dict['_in_transfer'] = False
                _compute_transfer_stats(gc_dict)

            # Install hooks
            if hasattr(diag_tile.tile_c, '_orig_update'):
                diag_tile.tile_c._orig_update = _capture_c_update_outer
            else:
                diag_tile.tile_c.update = _capture_c_update_outer
            ctrl.ab_weight_transfer = hooked_transfer

            if ctrl.forward_inject_enabled:
                # FI mode: ab_weight_update is never called.
                # Hook tile_b._orig_update which fires last (after tile_a._orig_update).
                # At call time:
                #   x_input  = ctrl._fi_b_x = raw x         (tile_b's x)
                #   d_input  = ctrl._fi_b_d = DA = A^T·d    (tile_b's d)
                #   ctrl._fi_a_x = XB = B·x                 (tile_a's x, still cached)
                #   ctrl._fi_a_d = raw d (alpha removed)     (tile_a's d, still cached)
                # del happens after both _orig_update calls, so ctrl._fi_a_* are alive here.
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
                # x = raw batch input, d = raw gradient.
                # Tile_a physically sees (XB=B·x, d); tile_b physically sees (x, DA=A^T·d).
                # Compute XB and DA here so stats match FI mode.
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
    init_acc, init_loss = evaluate_model(model, val_loader)
    wandb.log({"epoch": 0, "eval/acc": init_acc, "eval/loss": init_loss})
    print(f"Initial eval: acc={init_acc:.2f}%, loss={init_loss:.4f}")

    # Training loop
    best_acc = init_acc
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_history = []  # per-epoch accuracy, loss, etc.
    best_train_loss = float('inf')
    train_loss_no_improvement = 0
    global_step = 0
    criterion = nn.NLLLoss()

    print(f"\nStarting training: {N_EPOCHS} epochs (max), early stopping patience={EARLY_STOP_PATIENCE}")

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        optimizer.zero_grad()
        for micro_step, batch in enumerate(pbar):
            data, target = batch
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            output = model(data)
            loss = criterion(output, target) / GRAD_ACCUM_STEPS
            loss.backward()

            if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                diag_active = ENABLE_DIAGNOSTIC and (DIAG_EPOCHS == 0 or epoch <= DIAG_EPOCHS)
                if diag_active:
                    first_snap = snapshot_weights(first_tile)
                    last_snap = snapshot_weights(last_tile)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if diag_active:
                    for tile, snap, gcd, log_list, prev_state in [
                        (first_tile, first_snap, first_gc, first_log, "first"),
                        (last_tile, last_snap, last_gc, last_log, "last"),
                    ]:
                        A_bef, B_bef, C_bef, Craw_bef = snap
                        A_pre = gcd.pop('_A_pre_transfer', None)
                        # Rate-limited erank: compute only on transfer events with min step gap
                        if prev_state == "first":
                            _is_xfer = tile.controller.num_transfers > first_prev_nt
                            _do_erank = (ERANK_RATE_LIMIT_STEPS <= 0) or (
                                _is_xfer and (global_step - first_last_erank_step) >= ERANK_RATE_LIMIT_STEPS
                            )
                            if _do_erank:
                                first_last_erank_step = global_step
                            rec, first_C_prev_raw, first_prev_nt = collect_tile_diagnostics(
                                tile, first_C_prev_raw, A_bef, B_bef, C_bef, Craw_bef,
                                global_step, first_prev_nt, A_CI, B_CI, C_CI,
                                A_pre_transfer=A_pre, C_initial_eff=first_C_initial_eff,
                                compute_erank=_do_erank)
                        else:
                            _is_xfer = tile.controller.num_transfers > last_prev_nt
                            _do_erank = (ERANK_RATE_LIMIT_STEPS <= 0) or (
                                _is_xfer and (global_step - last_last_erank_step) >= ERANK_RATE_LIMIT_STEPS
                            )
                            if _do_erank:
                                last_last_erank_step = global_step
                            rec, last_C_prev_raw, last_prev_nt = collect_tile_diagnostics(
                                tile, last_C_prev_raw, A_bef, B_bef, C_bef, Craw_bef,
                                global_step, last_prev_nt, A_CI, B_CI, C_CI,
                                A_pre_transfer=A_pre, C_initial_eff=last_C_initial_eff,
                                compute_erank=_do_erank)
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
                        # Transfer C tile x/d diagnostics (recorded at transfer steps)
                        if rec["is_transfer"]:
                            rec['xc_abs_mean'] = gcd.get('xc_abs_mean', 0.0)
                            rec['xc_abs_max'] = gcd.get('xc_abs_max', 0.0)
                            rec['dc_abs_mean'] = gcd.get('dc_abs_mean', 0.0)
                            rec['dc_abs_max'] = gcd.get('dc_abs_max', 0.0)
                            rec['transfer_lr_c'] = gcd.get('transfer_lr_c', 0.0)
                            rec['transfer_n_calls'] = gcd.get('transfer_n_calls', 0)
                            for _pf in ['xc', 'dc']:
                                for _pp in ['p5', 'p25', 'p50', 'p75', 'p95']:
                                    rec[f'{_pf}_{_pp}'] = gcd.get(f'{_pf}_{_pp}', 0.0)
                            if gcd.get('_transfer_hist'):
                                rec['xc_dc_hist'] = gcd['_transfer_hist']

                        with torch.no_grad():
                            # Cosines and norms only at transfer steps (G_accum resets per transfer)
                            if rec["is_transfer"]:
                                C_eff_after = tile.tile_c.get_weights()[0].to(DEVICE)
                                C_eff_bef = snap[2].to(DEVICE)  # C_before from snapshot (effective)
                                delta_C_mat = C_eff_after - C_eff_bef
                                ctrl_delta = tile.controller.last_transfer_delta
                                tlr_AB = ctrl_delta.to(DEVICE) if ctrl_delta is not None else torch.zeros_like(delta_C_mat)
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

                    # Override snap A with pre-transfer values for next step's A_before
                    A_pre_first = first_gc.get('_A_pre_transfer')
                    if A_pre_first is not None:
                        first_snap = (A_pre_first,) + first_snap[1:]
                    A_pre_last = last_gc.get('_A_pre_transfer')
                    if A_pre_last is not None:
                        last_snap = (A_pre_last,) + last_snap[1:]

                    # Multi-tile metrics
                    with torch.no_grad():
                        _collect_multi_tile_metrics(global_step)

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

        # Step LR scheduler (per-epoch, StepLR)
        scheduler.step()

        # Evaluate
        eval_acc, eval_loss = evaluate_model(model, val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch, "train/loss": train_loss,
            "eval/acc": eval_acc, "eval/loss": eval_loss,
            "learning_rate": current_lr,
        })

        epoch_history.append({"epoch": epoch, "acc": eval_acc, "eval_loss": eval_loss, "train_loss": train_loss, "lr": current_lr})

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            save(model.state_dict(), WEIGHT_PATH)
        else:
            epochs_without_improvement += 1

        train_loss_improved = ""
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            train_loss_no_improvement = 0
            train_loss_improved = " ↓"
        else:
            train_loss_no_improvement += 1

        tqdm.write(
            f"Epoch {epoch}: Train loss: {train_loss:.4f}{train_loss_improved} | "
            f"Acc {eval_acc:.2f}% | Val loss {eval_loss:.4f} | "
            f"Best acc {best_acc:.2f}% | LR {current_lr:.2e} | "
            f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}"
        )

        if best_train_loss > TRAIN_LOSS_THRESHOLD and train_loss_no_improvement >= TRAIN_LOSS_EARLY_STOP_PATIENCE:
            tqdm.write(f"Train loss early stop at epoch {epoch} "
                       f"(train_loss={train_loss:.4f} > {TRAIN_LOSS_THRESHOLD}, no improvement for {train_loss_no_improvement} epochs)")
            break

        if best_train_loss <= TRAIN_LOSS_THRESHOLD and epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest acc: {best_acc:.2f}% at epoch {best_epoch}")

    # =========================================================================
    # Save diagnostic outputs
    # =========================================================================
    if ENABLE_DIAGNOSTIC and first_log:
        stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"
        first_transfers = [r["step"] for r in first_log if r["is_transfer"]]
        last_transfers = [r["step"] for r in last_log if r["is_transfer"]]
        diag_steps = len(first_log)
        print(f"\nDiag: {diag_steps}/{global_step} steps, T1={len(first_transfers)}, T2={len(last_transfers)}")

        json_path = os.path.join(RESULTS, f"mlp_diagnostic_log_{stamp}.json")
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
                "best_acc": best_acc, "best_epoch": best_epoch,
                "epoch_history": epoch_history,
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
                # Multi-tile data
                "multi_tiles": {k: {"name": multi_tiles[k][0], "steps": multi_logs[k]}
                                for k in multi_tiles if multi_logs[k]},
                # Mean across all multi-tiles per step
                "multi_mean": _compute_multi_mean(multi_logs) if any(multi_logs.values()) else [],
            }, f, indent=2)
        print(f"Saved: {json_path}")

        make_diagnostic_plots(first_log,
            os.path.join(RESULTS, f"mlp_diag_first_{stamp}.png"),
            tile_label=f"First tile ({first_name})", A_ci=A_CI, B_ci=B_CI, C_ci=C_CI)
        make_diagnostic_plots(last_log,
            os.path.join(RESULTS, f"mlp_diag_last_{stamp}.png"),
            tile_label=f"Last tile ({last_name})", A_ci=A_CI, B_ci=B_CI, C_ci=C_CI)

        # x/d distribution plots
        make_xd_diagnostic_plots(first_log,
            os.path.join(RESULTS, f"mlp_diag_xd_first_{stamp}.png"),
            tile_label=f"First tile ({first_name})")
        make_xd_diagnostic_plots(last_log,
            os.path.join(RESULTS, f"mlp_diag_xd_last_{stamp}.png"),
            tile_label=f"Last tile ({last_name})")

        # Multi-tile comparison plot
        if any(multi_logs.values()):
            make_multi_tile_plots(
                multi_logs, _compute_multi_mean(multi_logs),
                os.path.join(RESULTS, f"mlp_diag_multi_{stamp}.png"))

        steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
        diag_ep = DIAG_EPOCHS if DIAG_EPOCHS > 0 else N_EPOCHS
        for ep in range(1, diag_ep + 1):
            s0, s1 = (ep-1)*steps_per_epoch, ep*steps_per_epoch
            ef, el = first_log[s0:s1], last_log[s0:s1]
            if not ef: break
            make_diagnostic_plots(ef,
                os.path.join(RESULTS, f"mlp_diag_first_{stamp}_ep{ep}.png"),
                tile_label=f"First tile ({first_name}) — Epoch {ep}",
                A_ci=A_CI, B_ci=B_CI, C_ci=C_CI)
            make_diagnostic_plots(el,
                os.path.join(RESULTS, f"mlp_diag_last_{stamp}_ep{ep}.png"),
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
