# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for BERT-base + SQuAD v1.1 with TikiTaka / IdealDevice.

Usage:
    python optuna_bert_squad_tiki.py --n-trials 1
    python optuna_bert_squad_tiki.py --visualize
    python optuna_bert_squad_tiki.py --n-trials 50 --optimizer AnalogSGD --no-wd --no-momentum --no-nesterov

All flags:
    python optuna_bert_squad_tiki.py \
        --study-name <str>          # Study name (default: auto-generated)
        --n-trials <int>            # Number of Optuna trials (default: 50)
        --visualize                 # Visualize study results and exit
        --optimizer <str>           # AnalogSGD | AnalogAdam (default: AnalogAdam)
        --no-wd                     # Disable weight decay tuning (fix to 0)
        --no-momentum               # Disable momentum tuning (fix to 0, SGD only)
        --no-nesterov               # Disable nesterov tuning (fix to False, SGD only)
        --batch-size <int>          # Batch size (default: 48)
        --epochs <int>              # Number of epochs (default: 2)
        --warmup-ratio <float>      # LR warmup ratio (default: 0.05)
        --lora-target <str>         # Target: none|qonly|konly|vonly|qkv|ffn|all (default: qkv)
        --head-layer <str>          # qa_outputs: train | freeze (default: train)
        --target-ideal              # Convert target layers to IdealDevice (FP32 trainable)
        --lr <float>                # Override learning rate
        --classifier-lr <float>     # Separate LR for digital params (qa_outputs, LayerNorm)

Inline flags (edit directly in script):
    TRAIN_SUBSET_SIZE = 0           # Training data subset (0 = full)
    EVAL_SUBSET_SIZE = 0            # Evaluation data subset (0 = full)
"""

import os
import sys
import re
import string
import json
import argparse
import gc
import collections

# --- LRTT-v2 source shim -------------------------------------------------
# ENVIRONMENT_SETUP.md Step 6: LRTT Python modules are loaded via sys.path,
# shadowing the installed aihwkit-gpu wheel (which supplies the compiled
# rpu_base backend). LRTT_SRC must point at a checkout that contains the
# LRTT-v2 selector/shuffle controller (MLP branch).
_LRTT_SRC = os.environ.get("LRTT_SRC", "/root/LRTT/src")
if _LRTT_SRC not in sys.path:
    sys.path.insert(0, _LRTT_SRC)
os.environ.setdefault("LRTT_SILENT", "1")
# ------------------------------------------------------------------------

import torch
from torch import nn, no_grad, manual_seed
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

import optuna
from optuna.trial import TrialState
from optuna.samplers import GridSampler
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
import evaluate

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, IdealDevice, ConstantStepDevice
from aihwkit.simulator.configs import SingleRPUConfig, UnitCellRPUConfig, IOParameters, UpdateParameters
from aihwkit.simulator.configs.compounds import TransferCompound, ChoppedTransferCompound
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType
# LRTT-v2 selector/shuffle entry points (resolved from _LRTT_SRC shim above)
from aihwkit.simulator.configs import FloatingPointRPUConfig
from aihwkit.simulator.configs.devices import FloatingPointDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.optim.context import AnalogContext
from aihwkit.optim.analog_optimizer import AnalogOptimizerMixin

from collections import Counter


# =============================================================================
# Global Constants
# =============================================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = os.environ.get("LRTT_RESULTS", "/root/LRTT/experiments/bert_squad_lrttv2/results")
os.makedirs(RESULTS, exist_ok=True)

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "bert-base-uncased"

# SQuAD v1.1 settings
BATCH_SIZE = 48
EVAL_BATCH_SIZE = 256
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
N_EPOCHS = 2
EARLY_STOP_PATIENCE = 2

# Scheduler
WARMUP_RATIO = 0.05

# Fixed LR (SQuAD default)
SQUAD_LR = 2e-3

# TPE search ranges
TPE_FLR_RANGE = (1.0, 1.0)          # fixed at 1.0
TPE_TLR_RANGE = (1.0, 1000.0)       # log-uniform
SAMPLER_TYPE = "tpe"

# Target options
LORA_TARGET = "qkv"
HEAD_LAYER = "train"
TARGET_LAYERS = None  # None = all layers, list of 0-indexed ints = specific layers
TARGET_IDEAL = False   # When True, target layers use IdealDevice (FP32 update, trainable)
TARGET_ANALOG = False  # When True, target layers use SingleRPU (SoftBounds, trainable)

LORA_TARGET_MODULES = {
    "none": [],
    "qonly": ["query"],
    "konly": ["key"],
    "vonly": ["value"],
    "qkv": ["query", "key", "value"],
    "ffn": ["dense"],
    "all": None,
}

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

# Per-tile analog gradient clip+floor
CLIP_ANALOG_GRAD = False
ANALOG_TILE_MAX_NORM = 1.0
ANALOG_TILE_MIN_NORM = 0.1

# Global config (set by argparse)
OPT_CONFIG = {
    'optimizer': 'AnalogSGD',
    'tune_wd': False,
    'tune_momentum': False,
    'tune_nesterov': False,
    'shared_lr': True,
    'lr_range': [1e-4, 1.0],
}


def get_study_name_suffix():
    """Generate study name suffix based on optimizer config."""
    opt = OPT_CONFIG['optimizer'].lower().replace('analog', '')
    suffix = opt

    if not OPT_CONFIG['tune_wd']:
        suffix += "_nowd"
    if not OPT_CONFIG['tune_momentum']:
        suffix += "_nomom"
    if not OPT_CONFIG['tune_nesterov']:
        suffix += "_nonest"

    suffix += f"_{LORA_TARGET}"

    if TARGET_IDEAL:
        suffix += "_ideal"
    elif TARGET_ANALOG:
        suffix += "_singlerpu"

    if TARGET_LAYERS is not None:
        layer_str = "_".join(str(i + 1) for i in sorted(TARGET_LAYERS))
        suffix += f"_L{layer_str}"

    if HEAD_LAYER == "freeze":
        suffix += "_headfreeze"

    if OPT_CONFIG.get('learn_out_scaling', False):
        suffix += "_los"

    if OPT_CONFIG.get('nontarget_ideal', False):
        suffix += "_ntideal"
    elif OPT_CONFIG.get('nontarget_digital', False):
        suffix += "_ntdig"

    if OPT_CONFIG.get('backward_perfect', False):
        suffix += "_bwdperf"

    if not OPT_CONFIG.get('analog_only_warmup', True):
        suffix += "_allwarmup"

    return suffix

os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# TikiTaka v1 Device Functions
# =============================================================================

def _create_a_device():
    """Create A tile: 6T1C LinearStepDevice (fast, noisy)."""
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
        lifetime=0.0,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )


def _create_b_device():
    """Create B tile: noise-free SoftBoundsDevice (slow, accurate)."""
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


def create_tikitaka_config(transfer_every, transfer_lr, fast_lr, auto_scale=False, desired_bl=31, use_v2=False):
    """Create TikiTaka RPU configuration for analog layers."""
    a_device = _create_a_device()
    b_device = _create_b_device()

    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[a_device, b_device],
            transfer_every=transfer_every,
            units_in_mbatch=OPT_CONFIG.get('units_in_mbatch', True),
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=OPT_CONFIG.get('scale_transfer_lr', use_v2),
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=desired_bl,
                update_bl_management=False if use_v2 else True,
                update_management=False if use_v2 else True,
            ),
            no_buffer=not use_v2,
            in_chop_prob=0.1 if use_v2 else 0.0,
            out_chop_prob=0.0,
            auto_scale=auto_scale,
            auto_momentum=0.99,
        )
    )

    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0

    if OPT_CONFIG.get('backward_perfect', False):
        rpu_config.backward.is_perfect = True
    if OPT_CONFIG.get('forward_perfect', False):
        rpu_config.forward.is_perfect = True

    io_bits = OPT_CONFIG.get('io_bits', None)
    if io_bits is not None:
        io_res = 1.0 / (2 ** io_bits - 2)
        rpu_config.forward.inp_res = io_res
        rpu_config.forward.out_res = io_res
        rpu_config.backward.inp_res = io_res
        rpu_config.backward.out_res = io_res

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = OPT_CONFIG.get('learn_out_scaling', False)
    rpu_config.mapping.out_scaling_columnwise = OPT_CONFIG.get('learn_out_scaling', False)

    return rpu_config


def _bits_to_dw_min(weight_bits, w_min=-1.0, w_max=1.0):
    """ConstantStepDevice step for `weight_bits`-bit weights over [w_min, w_max].

    n_states = 2**bits; dw_min = (w_max - w_min) / n_states.
    10-bit over [-1, 1] -> 2 / 1024 = 0.001953125.
    """
    return (w_max - w_min) / float(2 ** int(weight_bits))


def _make_lrtt_tile_device(device_type, weight_bits):
    """Build one analog array device for an LRTT tile (Core or Auxiliary)."""
    if device_type == "constant_step":
        return ConstantStepDevice(
            dw_min=_bits_to_dw_min(weight_bits),
            w_max=1.0,
            w_min=-1.0,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
        )
    if device_type == "floatingpoint":
        return FloatingPointDevice()
    raise ValueError(f"Unsupported LRTT-v2 device_type: {device_type}")


def create_lrtt_v2_config(rank, transfer_every, transfer_lr,
                          selector_policy="shuffled_cycle", cap_rho=1.0,
                          device_type="constant_step", weight_bits=10):
    """Create an LRTT-v2 (row-coordinate selector + blockwise transfer) RPU config.

    Selector/shuffle controller logic mirrors the known-good
    experiments/smoke_lrtt_v2_mnist.py reference. The 3 LRTT unit-cell tiles
    are the Auxiliary low-rank arrays (A, B) and the Core visible array (C);
    here all three use the same analog array spec. With
    device_type='constant_step', weight_bits=10 -> ConstantStepDevice with
    dw_min = 2/1024 (10-bit Core array AND 10-bit Auxiliary array).
    """
    core = _make_lrtt_tile_device(device_type, weight_bits)
    aux_a = _make_lrtt_tile_device(device_type, weight_bits)
    aux_b = _make_lrtt_tile_device(device_type, weight_bits)
    dev = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        update_mode="selector_reconstruction",   # LRTT-v2 selector path
        transfer_method="blockwise",
        forward_inject=False,
        b_init_mode="zero",
        selector_axis="row",
        selector_policy=selector_policy,         # 'shuffled_cycle' -> reshuffle each cycle
        selector_seed=0,
        selector_reset_b_on_advance=True,
        cap_stabilizer_enabled=True,
        cap_rho=cap_rho,
        cap_compensate_transfer=True,
        unit_cell_devices=[aux_a, aux_b, core],
    )
    return PythonLRTTRPUConfig(device=dev)


def create_single_rpu_config(dw_min=None, device_type="softbounds"):
    """Create Single RPU configuration for analog layers.

    Args:
        dw_min: Override dw_min (default: use device default).
        device_type: "softbounds" (default) or "constant_step" (ConstantStepDevice).
    """
    if device_type == "constant_step":
        _dw = dw_min if dw_min is not None else 0.001
        device = ConstantStepDevice(
            dw_min=_dw,
            w_max=1.0,
            w_min=-1.0,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
        )
    else:
        device = _create_b_device()
        if dw_min is not None:
            device.dw_min = dw_min

    rpu_config = SingleRPUConfig(device=device)

    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    if OPT_CONFIG.get('backward_perfect', False):
        rpu_config.backward.is_perfect = True
    if OPT_CONFIG.get('forward_perfect', False):
        rpu_config.forward.is_perfect = True

    # Update params (desired_bl matters for pulsed update resolution)
    rpu_config.update.desired_bl = OPT_CONFIG.get('desired_bl', 31)

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = OPT_CONFIG.get('learn_out_scaling', False)
    rpu_config.mapping.out_scaling_columnwise = OPT_CONFIG.get('learn_out_scaling', False)

    return rpu_config


def create_ideal_config():
    """Create IdealDevice RPU configuration for target analog layers (trainable).

    Uses IdealDevice: floating point (FP32) update behavior.
    No pulsed update noise -- ideal for upper-bound comparison.
    """
    rpu_config = SingleRPUConfig(device=IdealDevice())

    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    if OPT_CONFIG.get('backward_perfect', False):
        rpu_config.backward.is_perfect = True
    if OPT_CONFIG.get('forward_perfect', False):
        rpu_config.forward.is_perfect = True

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = OPT_CONFIG.get('learn_out_scaling', False)
    rpu_config.mapping.out_scaling_columnwise = OPT_CONFIG.get('learn_out_scaling', False)

    return rpu_config


# =============================================================================
# Model Functions
# =============================================================================

def list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def _classify_encoder_layer(layer_name):
    """Classify BERT-base encoder Linear layer.

    BERT-base encoder layer structure (per block, x12):
        attention:  attention.self.query/key/value, attention.output.dense (W_O)
        ffn:        intermediate.dense, output.dense
    """
    if 'attention' in layer_name:
        return 'attention'
    return 'ffn'


def get_target_module_names(lora_target):
    """Get target category info for display purposes."""
    if lora_target == "none":
        return []
    elif lora_target in ("qonly", "konly", "vonly"):
        return {"qonly": ["query"], "konly": ["key"], "vonly": ["value"]}[lora_target]
    elif lora_target == "qkv":
        return ["attention (q,k,v,W_O)"]
    elif lora_target == "ffn":
        return ["ffn (intermediate, output)"]
    elif lora_target == "all":
        return ["attention + ffn"]
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model(params):
    """Create BERT-base QA model with selective TikiTaka / IdealDevice analog layers.

    qa_outputs is reinitialized with FIXED seed=42 for reproducibility.
    """
    from aihwkit.nn import AnalogLinear

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Reinitialize qa_outputs with FIXED seed for reproducibility
    if hasattr(model, 'qa_outputs'):
        torch.manual_seed(SEED)
        nn.init.normal_(model.qa_outputs.weight, mean=0.0, std=0.02)
        if model.qa_outputs.bias is not None:
            nn.init.zeros_(model.qa_outputs.bias)
        print(f"  [FIX] Reinitialized qa_outputs with FIXED seed={SEED}")

    # Always digital (never analog): qa_outputs + pooler
    always_digital = ["qa_outputs", "pooler"]

    def is_tikitaka_target(layer_name):
        """Check if encoder layer should be TikiTaka/Ideal (trainable analog)."""
        if any(d in layer_name for d in always_digital):
            return False
        if "encoder" not in layer_name:
            return False
        # Filter by target layer indices if specified
        if TARGET_LAYERS is not None:
            m = re.search(r'layer\.(\d+)', layer_name)
            if m is None or int(m.group(1)) not in TARGET_LAYERS:
                return False
        cat = _classify_encoder_layer(layer_name)
        if LORA_TARGET == "none":
            return False
        elif LORA_TARGET == "qkv":
            return cat == 'attention'
        elif LORA_TARGET == "ffn":
            return cat == 'ffn'
        elif LORA_TARGET == "all":
            return cat in ('attention', 'ffn')
        elif LORA_TARGET in ("qonly", "konly", "vonly"):
            patterns = {"qonly": ["query"], "konly": ["key"], "vonly": ["value"]}[LORA_TARGET]
            return any(p in layer_name for p in patterns)
        return False

    all_linear_names = list_linear_layers(model)

    # Classify layers
    tikitaka_layers = [n for n in all_linear_names if is_tikitaka_target(n)]
    non_target_encoder_layers = [
        n for n in all_linear_names
        if n not in tikitaka_layers and "encoder" in n
        and not any(d in n for d in always_digital)
    ]

    # --- Pass 1: Convert target layers to TikiTaka, IdealDevice, or SingleRPU ---
    tikitaka_count = 0
    ideal_count = 0
    target_analog_count = 0
    if tikitaka_layers and LORA_TARGET != "none" and OPT_CONFIG.get('mode') == 'lrtt_v2':
        # LRTT-v2: row-coordinate selector + shuffled-cycle blockwise transfer
        _v2_dev = OPT_CONFIG.get('lrtt_device_type', 'constant_step')
        _v2_bits = OPT_CONFIG.get('lrtt_weight_bits', 10)
        lrtt_config = create_lrtt_v2_config(
            rank=int(params["rank"]),
            transfer_every=int(params["transfer_every"]),
            transfer_lr=params["transfer_lr"],
            selector_policy=params.get("selector_policy", "shuffled_cycle"),
            cap_rho=params.get("cap_rho", 1.0),
            device_type=_v2_dev,
            weight_bits=_v2_bits,
        )
        lrtt_exclude = [n for n in all_linear_names if n not in tikitaka_layers]
        model = convert_to_analog(model, lrtt_config, exclude_modules=lrtt_exclude)
        tikitaka_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
        _dwm = _bits_to_dw_min(_v2_bits) if _v2_dev == 'constant_step' else None
        print(f"  [LRTT-v2] {tikitaka_count} target layers -> selector_reconstruction "
              f"(policy={params.get('selector_policy', 'shuffled_cycle')}, "
              f"rank={int(params['rank'])}, Core+Aux={_v2_dev}"
              f"{f'/{_v2_bits}bit(dw_min={_dwm:.6g})' if _dwm else ''})")
    elif tikitaka_layers and LORA_TARGET != "none" and TARGET_IDEAL:
        # IdealDevice: FP32 trainable analog (no noop hook)
        ideal_config = create_ideal_config()
        ideal_exclude = [n for n in all_linear_names if n not in tikitaka_layers]
        model = convert_to_analog(model, ideal_config, exclude_modules=ideal_exclude)
        ideal_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    elif tikitaka_layers and LORA_TARGET != "none" and TARGET_ANALOG:
        # SingleRPU: pulsed update trainable analog (no TikiTaka transfer)
        single_config = create_single_rpu_config(
            dw_min=OPT_CONFIG.get('dw_min', None),
            device_type=OPT_CONFIG.get('device_type', 'softbounds'),
        )
        single_exclude = [n for n in all_linear_names if n not in tikitaka_layers]
        model = convert_to_analog(model, single_config, exclude_modules=single_exclude)
        target_analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    elif tikitaka_layers and LORA_TARGET != "none":
        tiki_config = create_tikitaka_config(
            transfer_every=int(params["transfer_every"]),
            transfer_lr=params["transfer_lr"],
            fast_lr=params["fast_lr"],
            auto_scale=OPT_CONFIG.get('auto_scale', False),
            desired_bl=int(params["desired_bl"]),
            use_v2=OPT_CONFIG.get('use_v2', False),
        )
        tiki_exclude = [n for n in all_linear_names if n not in tikitaka_layers]
        model = convert_to_analog(model, tiki_config, exclude_modules=tiki_exclude)
        tikitaka_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))

    # --- Pass 2: Convert non-target encoder layers ---
    #   --nontarget-ideal:   IdealDevice frozen (analog forward, FP32, no update)
    #   --nontarget-analog:  SingleRPU frozen (analog forward, SoftBounds, no update)
    #   --nontarget-digital: keep as digital nn.Linear frozen (default)
    #
    # IMPORTANT: Pass 2 must use inplace=True to preserve Pass 1 tile object IDs.
    # Without inplace, convert_to_analog deepcopies the model, changing all object IDs,
    # which causes the freeze loop (id-based) to accidentally freeze target tiles too.
    single_rpu_count = 0
    nt_ideal_count = 0
    # Use name-based tracking (robust against deepcopy id changes)
    target_layer_names = set(tikitaka_layers)

    def _frozen_noop_update(x_input, d_input, *args, **kwargs):
        return None

    if non_target_encoder_layers and OPT_CONFIG.get('nontarget_ideal', False):
        # Non-target -> IdealDevice frozen
        nt_ideal_config = create_ideal_config()
        nt_exclude = [n for n in all_linear_names if n not in non_target_encoder_layers]
        model = convert_to_analog(model, nt_ideal_config, exclude_modules=nt_exclude, inplace=True)
        nt_ideal_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear)) - tikitaka_count - ideal_count - target_analog_count

        # Freeze: apply noop to non-target tiles only (by name)
        for name, m in model.named_modules():
            if isinstance(m, AnalogLinear) and name not in target_layer_names:
                for tile in m.analog_tiles():
                    tile.update = _frozen_noop_update
        print(f"  [IDEAL FROZEN] {nt_ideal_count} non-target layers -> IdealDevice frozen")

    elif non_target_encoder_layers and not OPT_CONFIG.get('nontarget_digital', False):
        # Non-target -> SingleRPU frozen
        single_config = create_single_rpu_config()
        single_exclude = [n for n in all_linear_names if n not in non_target_encoder_layers]
        model = convert_to_analog(model, single_config, exclude_modules=single_exclude, inplace=True)
        single_rpu_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear)) - tikitaka_count - ideal_count - target_analog_count

        for name, m in model.named_modules():
            if isinstance(m, AnalogLinear) and name not in target_layer_names:
                for tile in m.analog_tiles():
                    tile.update = _frozen_noop_update

    elif non_target_encoder_layers and OPT_CONFIG.get('nontarget_digital', False):
        print(f"  [DIGITAL] Keeping {len(non_target_encoder_layers)} non-target layers as digital (frozen)")

    total_params = sum(p.numel() for p in model.parameters())
    total_analog = tikitaka_count + ideal_count + target_analog_count + single_rpu_count + nt_ideal_count
    print(f"  TikiTaka: {tikitaka_count}, Ideal(train): {ideal_count}, SingleRPU(train): {target_analog_count}, "
          f"Ideal(frozen): {nt_ideal_count}, NT SingleRPU: {single_rpu_count}, "
          f"Total analog: {total_analog}, Total params: {total_params:,}")

    # Set requires_grad
    from aihwkit.optim.context import AnalogContext
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            param.requires_grad = True  # required for analog tile update
        elif "qa_outputs" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        elif "LayerNorm" in name or "layer_norm" in name:
            if OPT_CONFIG.get('train_layernorm', False):
                param.requires_grad = True  # explicit --train-layernorm
            elif TARGET_LAYERS is not None:
                m = re.search(r'layer\.(\d+)', name)
                param.requires_grad = (m is not None and int(m.group(1)) in TARGET_LAYERS)
            elif LORA_TARGET != "none":
                param.requires_grad = True  # all encoder layers are target -> train LayerNorm
            else:
                param.requires_grad = False  # no target -> freeze LayerNorm
        elif "out_scaling" in name:
            param.requires_grad = OPT_CONFIG.get('learn_out_scaling', False)
        else:
            param.requires_grad = False

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable_after:,}")
    print(f"  Target: {LORA_TARGET} -> {get_target_module_names(LORA_TARGET)}")

    # --- Verification: ensure target tiles are NOT frozen ---
    _frozen_targets = 0
    _trainable_targets = 0
    _frozen_nontargets = 0
    for name, m in model.named_modules():
        if isinstance(m, AnalogLinear):
            tiles = list(m.analog_tiles())
            is_noop = any(getattr(t, 'update', None) == _frozen_noop_update for t in tiles)
            if name in target_layer_names:
                if is_noop:
                    _frozen_targets += 1
                else:
                    _trainable_targets += 1
            else:
                if is_noop:
                    _frozen_nontargets += 1
    _analog_weight_params = sum(
        m.get_weights()[0].numel() for m in model.modules() if isinstance(m, AnalogLinear)
    )
    print(f"  [VERIFY] Target tiles: {_trainable_targets} trainable, {_frozen_targets} FROZEN"
          f" | Non-target frozen: {_frozen_nontargets}"
          f" | Analog weight params: {_analog_weight_params:,}")
    if _frozen_targets > 0:
        print(f"  [WARNING] {_frozen_targets} target tiles are incorrectly frozen!")

    return model.to(DEVICE)


# =============================================================================
# Data Functions
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize SQuAD v1.1 dataset."""
    raw_datasets = load_dataset("squad")

    if EVAL_SUBSET_SIZE > 0:
        eval_examples = raw_datasets["validation"].select(
            range(min(EVAL_SUBSET_SIZE, len(raw_datasets["validation"])))
        )
    else:
        eval_examples = raw_datasets["validation"]

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]

        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]

            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_eval(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs["offset_mapping"]

        for i in range(len(inputs["input_ids"])):
            sequence_ids = inputs.sequence_ids(i)
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(offset_mapping[i])
            ]

        inputs["example_id"] = [
            examples["id"][sample_map[i]] for i in range(len(inputs["input_ids"]))
        ]

        return inputs

    tokenized_train = raw_datasets["train"].map(
        preprocess_train, batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    if TRAIN_SUBSET_SIZE > 0:
        train_subset = tokenized_train.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(tokenized_train)))
        )
    else:
        train_subset = tokenized_train.shuffle(seed=SEED)

    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=default_data_collator,
        generator=torch.Generator().manual_seed(SEED)
    )

    return train_loader, tokenized_eval, eval_examples


# =============================================================================
# Evaluation Functions
# =============================================================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def postprocess_squad_predictions(
    examples, features, all_start_logits, all_end_logits,
    n_best_size=20, max_answer_length=30,
):
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    prelim_predictions.append({
                        "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                        "score": start_logits[start_index] + end_logits[end_index],
                    })

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        if len(predictions) == 0:
            all_predictions[example["id"]] = ""
        else:
            best_pred = predictions[0]
            start_char, end_char = best_pred["offsets"]
            all_predictions[example["id"]] = context[start_char:end_char]

    return all_predictions


def evaluate_model(model, eval_features, eval_examples, tokenizer):
    """Evaluate SQuAD model. Returns (F1, EM)."""
    model.eval()

    all_start_logits = []
    all_end_logits = []

    def squad_eval_collate_fn(features):
        offset_mappings = [f.pop("offset_mapping") for f in features]
        example_ids = [f.pop("example_id") for f in features]
        batch = default_data_collator(features)
        batch["offset_mapping"] = offset_mappings
        batch["example_id"] = example_ids
        for i, f in enumerate(features):
            f["offset_mapping"] = offset_mappings[i]
            f["example_id"] = example_ids[i]
        return batch

    eval_loader = DataLoader(
        eval_features, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=squad_eval_collate_fn
    )

    with no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    model.train()

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    predictions = postprocess_squad_predictions(
        eval_examples, eval_features,
        all_start_logits, all_end_logits,
        n_best_size=20, max_answer_length=30
    )

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)

    return results["f1"], results["exact_match"]


# =============================================================================
# Scheduler
# =============================================================================


def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps,
                                    min_lr_rate=0.0, warmup_analog_only=True):
    """Linear schedule with warmup that decays to min_lr_rate (fraction of peak LR).
    If warmup_analog_only=True, warmup is applied only to AnalogContext param groups;
    digital params (classifier, LayerNorm) start at peak LR immediately.
    """
    from aihwkit.optim.context import AnalogContext

    def _make_lambda(apply_warmup):
        def lr_lambda(current_step):
            if apply_warmup and current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = max(0.0, float(current_step - num_warmup_steps)) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
        return lr_lambda

    if warmup_analog_only:
        lambdas = []
        for group in optimizer.param_groups:
            is_analog = any(isinstance(p, AnalogContext) for p in group["params"])
            lambdas.append(_make_lambda(apply_warmup=is_analog))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _make_lambda(apply_warmup=True))


# =============================================================================
# Optuna Objective
# =============================================================================

def objective(trial, train_loader, eval_features, eval_examples, tokenizer):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Learning rate: grid (HP-region sweep), sweep range, or fixed
    _lr_grid = OPT_CONFIG.get('lr_grid', None)
    _lr_range = OPT_CONFIG.get('lr_range', None)
    if _lr_grid is not None:
        learning_rate = trial.suggest_categorical('learning_rate', _lr_grid)
    elif _lr_range is not None:
        learning_rate = trial.suggest_float('learning_rate', _lr_range[0], _lr_range[1], log=True)
    else:
        learning_rate = OPT_CONFIG.get('lr_override', None) or SQUAD_LR

    # Classifier LR: sweep range or fixed
    if OPT_CONFIG.get('shared_lr', False):
        # Both analog and digital params share the same LR parameter
        classifier_lr = learning_rate
    else:
        _cls_lr_range = OPT_CONFIG.get('classifier_lr_range', None)
        if _cls_lr_range is not None:
            classifier_lr = trial.suggest_float('classifier_lr', _cls_lr_range[0], _cls_lr_range[1], log=True)
        elif OPT_CONFIG.get('classifier_lr') is not None:
            classifier_lr = OPT_CONFIG['classifier_lr']
        else:
            classifier_lr = None

    # fast_lr, transfer_lr: sweep or fixed
    # Skip TikiTaka params when using IdealDevice (TARGET_IDEAL=True) -- not applicable
    _has_tikitaka = LORA_TARGET != "none" and not TARGET_IDEAL and not TARGET_ANALOG
    if not _has_tikitaka:
        fast_lr = 1.0
        transfer_lr = 1.0
    else:
        if TPE_FLR_RANGE[0] == TPE_FLR_RANGE[1]:
            fast_lr = TPE_FLR_RANGE[0]
        else:
            fast_lr = trial.suggest_float('fast_lr', TPE_FLR_RANGE[0], TPE_FLR_RANGE[1], log=True)
        _tlr_grid = OPT_CONFIG.get('tlr_grid', None)
        if _tlr_grid is not None:
            transfer_lr = trial.suggest_categorical('transfer_lr', _tlr_grid)
        elif TPE_TLR_RANGE[0] == TPE_TLR_RANGE[1]:
            transfer_lr = TPE_TLR_RANGE[0]
        else:
            # scale_transfer_lr (tlr_upper = 1/lr) is a TikiTaka-v2
            # (ChoppedTransferCompound) heuristic. For LRTT-v2 the
            # PythonLRTTDevice.transfer_lr is a direct device parameter, so it
            # must be TPE-searched over the literal --tpe-tlr-range (not
            # coupled to learning_rate, which would make the range differ per
            # server/trial).
            _scale_tlr = OPT_CONFIG.get('scale_transfer_lr', OPT_CONFIG.get('use_v2', False))
            if OPT_CONFIG.get('mode') == 'lrtt_v2':
                _scale_tlr = False
            if _scale_tlr:
                _tlr_upper = 1.0 / learning_rate
            else:
                _tlr_upper = TPE_TLR_RANGE[1]
            transfer_lr = trial.suggest_float('transfer_lr', TPE_TLR_RANGE[0], _tlr_upper, log=True)

    # desired_bl: sweep or fixed (skip if no TikiTaka tiles)
    if not _has_tikitaka:
        desired_bl = 1
    else:
        _bl_range = OPT_CONFIG.get('bl_sweep', None)
        _bl_grid = OPT_CONFIG.get('bl_grid', None)
        if _bl_grid is not None:
            desired_bl = trial.suggest_categorical('desired_bl', _bl_grid)
        elif _bl_range is not None:
            desired_bl = trial.suggest_int('desired_bl', _bl_range[0], _bl_range[1])
        else:
            desired_bl = OPT_CONFIG.get('desired_bl', 1)

    # transfer_every: fixed at 1 (uim=True -> every mini-batch)
    transfer_every = OPT_CONFIG.get('transfer_every_override', 1)

    min_lr_rate = OPT_CONFIG.get('min_lr_rate', 0.5)  # fraction of peak LR (paper default 0.5)

    if OPT_CONFIG['tune_wd']:
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
    else:
        weight_decay = 0.0

    if OPT_CONFIG['tune_momentum']:
        momentum = 0.9
    else:
        momentum = 0.0

    if OPT_CONFIG['tune_nesterov'] and momentum > 0:
        nesterov = True
    else:
        nesterov = False

    optimizer_name = OPT_CONFIG['optimizer']

    params = {
        "transfer_every": transfer_every,
        "transfer_lr": transfer_lr,
        "fast_lr": fast_lr,
        "desired_bl": desired_bl,
        # LRTT-v2 selector/shuffle params (fixed via CLI for the validation run)
        "rank": OPT_CONFIG.get("rank", 8),
        "selector_policy": OPT_CONFIG.get("selector_policy", "shuffled_cycle"),
        "cap_rho": OPT_CONFIG.get("cap_rho", 1.0),
    }
    if OPT_CONFIG.get('mode') == 'lrtt_v2':
        print(f"  [LRTT-v2] rank={params['rank']}, selector_policy={params['selector_policy']}, "
              f"cap_rho={params['cap_rho']}, transfer_every={transfer_every}, "
              f"transfer_lr={transfer_lr:.4e}")

    _cls_lr_str = f", classifier_lr={classifier_lr:.2e}" if classifier_lr is not None else ""
    _warmup_analog_only = OPT_CONFIG.get('analog_only_warmup', True)
    _warmup_str = "analog_tile_only" if _warmup_analog_only else "all_groups"
    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting (squad, metric=F1)")
    print(f"{'='*70}")
    print(f"  transfer_every={transfer_every}, transfer_lr={transfer_lr:.4e}, fast_lr={fast_lr:.4e}, desired_bl={desired_bl}")
    print(f"  lr={learning_rate:.2e}{_cls_lr_str}, wd={weight_decay:.2e}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, optimizer={optimizer_name}")
    print(f"  min_lr_rate={min_lr_rate:.4f}")
    print(f"  warmup_ratio={WARMUP_RATIO:.4f}, warmup_target={_warmup_str}  [classifier/LayerNorm: {'no warmup' if _warmup_analog_only else 'warmed up'}]")
    print(f"{'='*70}")

    model = None
    try:
        set_seed(SEED)

        if torch.cuda.is_available():
            _alloc = torch.cuda.memory_allocated() / 1024**3
            _reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  [GPU] Before model: alloc={_alloc:.2f}GB, reserved={_reserved:.2f}GB")

        model = create_model(params)

        # Separate LR for analog (LRTT/TikiTaka auxiliary tiles) vs digital
        # (classifier qa_outputs + LayerNorm) whenever a classifier_lr is
        # provided. Works for ALL modes incl. --mode lrtt_v2 (previously this
        # split was gated to IdealDevice/SingleRPU targets only). With
        # classifier_lr=None the behavior is unchanged: a single learning_rate.
        _cls_lr = classifier_lr
        if _cls_lr is not None:
            from aihwkit.optim.context import AnalogContext as _AC2
            analog_params = [p for p in model.parameters()
                             if isinstance(p, _AC2) and p.requires_grad]
            digital_params = [p for p in model.parameters()
                              if not isinstance(p, _AC2) and p.requires_grad]
            # aihwkit specifics: regroup_param_groups() rebuilds one group per
            # analog tile WITHOUT an explicit lr, so every analog tile's LR is
            # taken from optimizer.defaults["lr"] (the top-level lr=). The
            # digital group keeps its own lr (used by torch SGD/Adam for the
            # non-analog params). Hence:
            #   top-level lr  = learning_rate  -> analog (LRTT auxiliary) LR
            #   digital group = classifier_lr  -> classifier/LayerNorm LR
            param_groups = [
                {"params": analog_params},                  # -> defaults lr (learning_rate)
                {"params": digital_params, "lr": _cls_lr},   # -> classifier_lr
            ]
            if optimizer_name == "AnalogSGD":
                optimizer = AnalogSGD(
                    param_groups, lr=learning_rate,
                    weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                )
            else:
                optimizer = AnalogAdam(
                    param_groups, lr=learning_rate, weight_decay=weight_decay,
                )
            print(f"  [LR-SPLIT] analog(LRTT aux) lr={learning_rate:.3e} | "
                  f"digital(classifier/LayerNorm) lr={_cls_lr:.3e} | "
                  f"analog params={len(analog_params)}, digital params={len(digital_params)}")
        else:
            if optimizer_name == "AnalogSGD":
                optimizer = AnalogSGD(
                    model.parameters(), lr=learning_rate,
                    weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                )
            else:
                optimizer = AnalogAdam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                )
            print(f"  [LR-SINGLE] shared lr={learning_rate:.3e} "
                  f"(no classifier_lr -> analog & digital share LR)")
        optimizer.regroup_param_groups()

        if torch.cuda.is_available():
            _alloc = torch.cuda.memory_allocated() / 1024**3
            _reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  [GPU] After model+optimizer: alloc={_alloc:.2f}GB, reserved={_reserved:.2f}GB")

        _grad_accum = max(1, int(OPT_CONFIG.get('grad_accum_steps', 1)))
        _opt_steps_per_epoch = len(train_loader) // _grad_accum
        num_training_steps = _opt_steps_per_epoch * N_EPOCHS
        warmup_steps = int(num_training_steps * WARMUP_RATIO)
        _warmup_analog_only = OPT_CONFIG.get('analog_only_warmup', True)
        print(f"  [GRAD-ACCUM] batch={BATCH_SIZE} x accum={_grad_accum} "
              f"= effective {BATCH_SIZE * _grad_accum} | "
              f"opt steps/epoch={_opt_steps_per_epoch}")
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
            warmup_analog_only=_warmup_analog_only,
        )

        best_f1 = 0.0
        epochs_without_improvement = 0
        global_step = 0

        _diag_update = OPT_CONFIG.get('diag_update', False)
        _diag_steps = OPT_CONFIG.get('diag_steps', 200)
        _diag_tracker = None
        if _diag_update:
            from update_diag import UpdateDiagTracker
            _diag_tracker = UpdateDiagTracker(
                model, dw_min_A=0.001981, dw_min_B=0.001,
                desired_bl=desired_bl, transfer_every=transfer_every, w_max=1.0,
            )

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
            for _micro_idx, batch in enumerate(pbar):
                _window_start = (_micro_idx % _grad_accum == 0)
                _is_boundary = ((_micro_idx + 1) % _grad_accum == 0)

                if _window_start:
                    global_step += 1  # one global_step per optimizer step
                    optimizer.zero_grad()
                _diag_active = _diag_update and _diag_tracker and global_step <= _diag_steps

                if _diag_active and _window_start:
                    _diag_tracker.snapshot_before()

                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                start_positions = batch['start_positions'].to(DEVICE)
                end_positions = batch['end_positions'].to(DEVICE)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    start_positions=start_positions, end_positions=end_positions,
                )
                # Digital params: standard grad accumulation (loss/=accum,
                # grads sum over the window, one step at the boundary).
                loss = outputs.loss / _grad_accum
                loss.backward()

                # --- Analog grad-accum: paper_experiment.py memory fix ---
                # With grad_accum>1, do NOT let AnalogContext accumulate
                # (x, d) tensors across micro-batches (that piles 3x[N,feat]
                # per tile x 48 tiles -> OOM, and leaks across trials).
                # Instead apply each tile's analog update immediately per
                # micro-batch and p.reset() to free analog_input/grad. The
                # analog optimizer part of the boundary step is then skipped.
                if _grad_accum > 1:
                    with torch.no_grad():
                        for p in model.parameters():
                            if not isinstance(p, AnalogContext):
                                continue
                            if not p.requires_grad:
                                p.reset()
                                continue
                            if p.use_torch_update or not p.has_gradient():
                                continue
                            # Optional per-tile analog grad clip+floor
                            if CLIP_ANALOG_GRAD and LORA_TARGET != "none" and p.analog_grad_output:
                                _mx, _mn = ANALOG_TILE_MAX_NORM, ANALOG_TILE_MIN_NORM
                                for i, go in enumerate(p.analog_grad_output):
                                    tn = go.detach().norm()
                                    sc = torch.where(
                                        tn > _mx, _mx / (tn + 1e-6),
                                        torch.where((tn < _mn) & (tn > 1e-10),
                                                    _mn / (tn + 1e-6), tn.new_ones(())))
                                    p.analog_grad_output[i] = go * sc
                            analog_tile = p.analog_tile
                            runtime = analog_tile.get_runtime()
                            if p.use_indexed:
                                for x_i, d_i in zip(p.analog_input, p.analog_grad_output):
                                    analog_tile.update_indexed(
                                        x_i.to(analog_tile.device) if runtime.offload_input else x_i,
                                        d_i.to(analog_tile.device) if runtime.offload_gradient else d_i,
                                    )
                            else:
                                x_input = torch.cat(p.analog_input,
                                                    axis=-1 if analog_tile.in_trans else 0)
                                d_input = torch.cat(p.analog_grad_output,
                                                    axis=-1 if analog_tile.out_trans else 0)
                                analog_tile.update(
                                    x_input.to(analog_tile.device) if runtime.offload_input else x_input,
                                    d_input.to(analog_tile.device) if runtime.offload_gradient else d_input,
                                )
                            p.reset()  # free analog_ctx -> memory bounded to 1 micro-batch

                if _is_boundary:
                    # Digital-only grad clipping
                    _digital_params = [p for p in model.parameters()
                                       if not isinstance(p, AnalogContext) and p.grad is not None]
                    if _digital_params:
                        torch.nn.utils.clip_grad_norm_(_digital_params, max_norm=1.0)

                    if _diag_active:
                        _current_lr = optimizer.param_groups[0]['lr']
                        _diag_tracker.record_signals(
                            global_step, model, lr=_current_lr,
                            clip_analog_grad=CLIP_ANALOG_GRAD,
                            max_norm=ANALOG_TILE_MAX_NORM, min_norm=ANALOG_TILE_MIN_NORM,
                        )

                    scheduler.step()
                    # Sync analog tile lr with scheduler
                    for _pg in optimizer.param_groups:
                        for _p in _pg['params']:
                            if isinstance(_p, AnalogContext):
                                _p.analog_tile.set_learning_rate(_pg['lr'])

                    if _grad_accum > 1:
                        # Analog updates already applied per micro-batch above:
                        # run only the digital optimizer + analog post-step.
                        super(AnalogOptimizerMixin, optimizer).step()
                        for _pg in optimizer.param_groups:
                            for _p in _pg['params']:
                                if isinstance(_p, AnalogContext) and _p.requires_grad:
                                    if hasattr(_p.analog_tile, 'post_update_step'):
                                        _p.analog_tile.post_update_step()
                    else:
                        optimizer.step()  # grad_accum==1: analog + digital

                    if _diag_active:
                        _diag_tracker.record_after(global_step)

                loss_val = loss.item() * _grad_accum  # report unscaled loss
                total_loss += loss_val
                num_batches += 1
                pbar.set_postfix(loss=f"{loss_val:.4f}")

                # Loss divergence detection
                if not np.isfinite(loss_val) or loss_val > 1e8:
                    tqdm.write(f"[Trial {trial.number}] Loss diverged at step {global_step} "
                              f"(loss={loss_val:.2e}), stopping early.")
                    trial.set_user_attr("diverged", True)
                    return -1.0

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            eval_f1, eval_em = evaluate_model(model, eval_features, eval_examples, tokenizer)

            improved = ""
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                epochs_without_improvement = 0
                improved = " *"
            else:
                epochs_without_improvement += 1

            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"[Trial {trial.number}] Epoch {epoch:3d} | "
                  f"F1: {eval_f1:6.2f}% | EM: {eval_em:6.2f}% | Best F1: {best_f1:6.2f}% | "
                  f"Loss: {train_loss:.4f} | LR: {current_lr:.2e} | "
                  f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}{improved}")

            trial.report(eval_f1, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

            # Hard prune gate: at --prune-at-epoch, if F1 <= threshold, prune
            # this trial so TPE moves on to the next search point.
            _pae = OPT_CONFIG.get('prune_at_epoch', 0)
            _pf1 = OPT_CONFIG.get('prune_f1_threshold', 0.0)
            if _pae and epoch == _pae and eval_f1 <= _pf1:
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch} "
                           f"(F1 {eval_f1:.2f}% <= {_pf1:.2f}% threshold) "
                           f"-> next TPE trial")
                trial.set_user_attr("pruned_reason",
                                    f"epoch{epoch}_f1<={_pf1}")
                raise optuna.exceptions.TrialPruned()

            # Early stopping (manual break, independent of the Optuna pruner)
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch} "
                          f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

            # Optuna pruner
            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        if _diag_update and _diag_tracker:
            _diag_dir = os.path.join(RESULTS, f"diag_trial_{trial.number}")
            os.makedirs(_diag_dir, exist_ok=True)
            _diag_tracker.save_csvs(_diag_dir)
            _diag_tracker.print_summary()

        print(f"\n[Trial {trial.number}] Finished - Best F1: {best_f1:.2f}%")
        print(f"{'='*70}\n")
        return best_f1

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
    """Visualize optimization history."""
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        print("No completed trials to visualize.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    trial_numbers = [t.number for t in complete_trials]
    f1_scores = [t.value for t in complete_trials]

    axes[0].scatter(trial_numbers, f1_scores, alpha=0.6)
    axes[0].plot(trial_numbers,
                 [max(f1_scores[:i+1]) for i in range(len(f1_scores))],
                 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('F1 (%)')
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

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "visualization_bert_squad.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualization saved.")


def print_study_summary(study):
    """Print study summary."""
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Study: {study.study_name}, Trials: {len(study.trials)} ({len(complete_trials)} complete)")
    if complete_trials:
        f1_scores = [t.value for t in complete_trials]
        print(f"Best F1: {max(f1_scores):.2f}%, Mean F1: {sum(f1_scores)/len(f1_scores):.2f}%")
        print(f"Best params: {study.best_params}")


# =============================================================================
# Main
# =============================================================================

def main():
    global BATCH_SIZE, EVAL_BATCH_SIZE, N_EPOCHS, WARMUP_RATIO, LORA_TARGET, HEAD_LAYER, MAX_SEQ_LENGTH, SQUAD_LR, TARGET_IDEAL, TARGET_ANALOG

    parser = argparse.ArgumentParser(description="Optuna sweep for BERT-base SQuAD TikiTaka")
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: auto-generated based on config)')
    parser.add_argument('--n-trials', type=int, default=10)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--optimizer', type=str, default='AnalogSGD',
                        choices=['AnalogSGD', 'AnalogAdam'],
                        help='Optimizer type (default: AnalogSGD)')
    parser.add_argument('--no-wd', action='store_true', default=True,
                        help='Disable weight decay tuning (fix to 0)')
    parser.add_argument('--tune-wd', dest='no_wd', action='store_false',
                        help='Enable weight decay tuning')
    parser.add_argument('--no-momentum', action='store_true', default=True,
                        help='Disable momentum tuning (fix to 0, SGD only)')
    parser.add_argument('--tune-momentum', dest='no_momentum', action='store_false',
                        help='Enable momentum tuning')
    parser.add_argument('--no-nesterov', action='store_true', default=True,
                        help='Disable nesterov tuning (fix to False, SGD only)')
    parser.add_argument('--tune-nesterov', dest='no_nesterov', action='store_false',
                        help='Enable nesterov tuning')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--eval-batch-size', type=int, default=EVAL_BATCH_SIZE,
                        help=f'Eval batch size (default: {EVAL_BATCH_SIZE}). '
                             f'Large values spike analog-tile CUDA memory at '
                             f'the eval->next-epoch boundary; lower to avoid OOM.')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS,
                        help=f'Number of epochs (default: {N_EPOCHS})')
    parser.add_argument('--warmup-ratio', type=float, default=WARMUP_RATIO,
                        help=f'LR warmup ratio (default: {WARMUP_RATIO})')
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=['none', 'qonly', 'konly', 'vonly', 'qkv', 'ffn', 'all'],
                        help='Target: none, qonly, konly, vonly, qkv, ffn, all (default: qkv)')
    parser.add_argument('--head-layer', type=str, default=HEAD_LAYER,
                        choices=['train', 'freeze'],
                        help='qa_outputs layer: train or freeze (default: train)')
    parser.add_argument('--target-ideal', action='store_true', default=False,
                        help='Convert target layers to IdealDevice (FP32 trainable analog) instead of TikiTaka')
    parser.add_argument('--no-target-ideal', action='store_false', dest='target_ideal',
                        help='Disable IdealDevice, use TikiTaka instead')
    parser.add_argument('--target-analog', action='store_true', default=False,
                        help='Convert target layers to SingleRPU (SoftBounds, trainable) instead of IdealDevice/TikiTaka')
    parser.add_argument('--device-type', type=str, default='softbounds',
                        choices=['softbounds', 'constant_step'],
                        help='Device type for --target-analog (default: softbounds)')
    parser.add_argument('--dw-min', type=float, default=None,
                        help='Override dw_min for target analog device (e.g. 0.125 for 4-bit)')
    parser.add_argument('--target-layers', type=int, nargs='+', default=None,
                        help='Target encoder layer indices, 1-indexed (e.g. --target-layers 1 12). Default: all layers')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (e.g. --lr 1e-4)')
    parser.add_argument('--lr-range', type=float, nargs=2, default=None,
                        help='Sweep learning rate range [min max] (e.g. --lr-range 1e-5 1e-2)')
    parser.add_argument('--classifier-lr', type=float, default=None,
                        help='Separate LR for digital params (qa_outputs, LayerNorm). Analog tiles use --lr.')
    parser.add_argument('--classifier-lr-range', type=float, nargs=2, default=None,
                        help='Sweep classifier LR range [min max] (e.g. --classifier-lr-range 1e-5 1e-2)')
    parser.add_argument('--learn-out-scaling', action='store_true', default=False,
                        help='Enable learn_out_scaling and out_scaling_columnwise (default: False)')
    parser.add_argument('--no-learn-out-scaling', dest='learn_out_scaling', action='store_false',
                        help='Disable learn_out_scaling')
    parser.add_argument('--io-bits', type=int, default=None,
                        help='ADC/DAC resolution bits for forward/backward IO (e.g. 10). Default: no quantization (-1)')
    parser.add_argument('--clip-analog-grad', action='store_true', default=False,
                        help='Enable per-tile analog gradient clip+floor (default: False)')
    parser.add_argument('--no-clip-analog-grad', dest='clip_analog_grad', action='store_false',
                        help='Disable analog gradient clipping')
    parser.add_argument('--nontarget-digital', action='store_true', default=True,
                        help='Keep non-target encoder layers as digital (default: True)')
    parser.add_argument('--nontarget-analog', dest='nontarget_digital', action='store_false',
                        help='Convert non-target encoder layers to frozen SingleRPU analog')
    parser.add_argument('--nontarget-ideal', action='store_true', default=False,
                        help='Convert non-target encoder layers to frozen IdealDevice analog')
    parser.add_argument('--train-layernorm', action='store_true', default=True,
                        help='Force LayerNorm trainable (useful with --lora-target none --nontarget-ideal)')
    parser.add_argument('--analog-only-warmup', action='store_true', default=True,
                        help='Apply LR warmup only to analog tile params; classifier/LayerNorm start at full LR (default: True)')
    parser.add_argument('--no-analog-only-warmup', action='store_false', dest='analog_only_warmup',
                        help='Apply warmup to all param groups equally')
    parser.add_argument('--backward-perfect', action='store_true', default=False,
                        help='Use perfect backward pass (no DAC/ADC quantization on gradients)')
    parser.add_argument('--forward-perfect', action='store_true', default=False,
                        help='Use perfect forward pass (no DAC/ADC quantization on activations)')
    parser.add_argument('--auto-scale', action='store_true', default=True,
                        help='Enable auto_scale (default: True)')
    parser.add_argument('--no-auto-scale', action='store_false', dest='auto_scale',
                        help='Disable auto_scale')
    parser.add_argument('--transfer-every', type=int, default=None,
                        help='Override transfer_every (default: 1)')
    parser.add_argument('--uim', action='store_true', dest='units_in_mbatch', default=True,
                        help='units_in_mbatch=True (default)')
    parser.add_argument('--no-uim', action='store_false', dest='units_in_mbatch',
                        help='units_in_mbatch=False')
    parser.add_argument('--desired-bl', type=int, default=31,
                        help='Transfer update desired_bl (default: 31)')
    parser.add_argument('--bl-sweep', type=int, nargs=2, default=None,
                        help='Sweep desired_bl range [min max]')
    parser.add_argument('--bl-grid', type=int, nargs='+', default=None,
                        help='Grid of desired_bl values')
    parser.add_argument('--use-v2', action='store_true', default=True,
                        help='Use TikiTaka v2 (default: True)')
    parser.add_argument('--no-v2', action='store_false', dest='use_v2',
                        help='Disable TikiTaka v2, use v1 instead')
    parser.add_argument('--no-scale-transfer-lr', action='store_true', default=False,
                        help='Force scale_transfer_lr=False')
    parser.add_argument('--scale-transfer-lr', action='store_false', dest='no_scale_transfer_lr',
                        help='Enable scale_transfer_lr')
    parser.add_argument('--lr-upper-mult', type=float, default=10.0,
                        help='LR upper bound multiplier (default: 10.0)')
    parser.add_argument('--sampler', type=str, default='tpe', choices=['grid', 'tpe'],
                        help='Sampler type (default: tpe)')
    parser.add_argument('--tpe-flr-range', type=float, nargs=2, default=[1.0, 1.0],
                        help='TPE fast_lr range (default: 1.0 1.0, fixed)')
    parser.add_argument('--tpe-tlr-range', type=float, nargs=2, default=[1.0, 1000.0],
                        help='TPE transfer_lr range (default: 1.0 1000.0)')
    parser.add_argument('--shared-lr', action='store_true', default=False,
                        help='Use a single shared LR for both analog and digital params (classifier_lr = learning_rate)')
    parser.add_argument('--diag-update', action='store_true', default=False,
                        help='Enable weight-update diagnostics')
    parser.add_argument('--diag-steps', type=int, default=200,
                        help='Record diagnostics for first N steps (default: 200)')
    parser.add_argument('--train-subset', type=int, default=0,
                        help='Limit training data size (0 = full dataset)')
    parser.add_argument('--eval-subset', type=int, default=0,
                        help='Limit eval data size (0 = full dataset)')
    parser.add_argument('--mode', type=str, default='tiki',
                        choices=['tiki', 'lrtt_v2'],
                        help='Target-layer config: tiki (TikiTaka, default) or '
                             'lrtt_v2 (LRTT-v2 row-selector + shuffled-cycle transfer)')
    parser.add_argument('--rank', type=int, default=8,
                        help='LRTT-v2 LoRA rank (= selector block size). Default: 8')
    parser.add_argument('--selector-policy', type=str, default='shuffled_cycle',
                        choices=['shuffled_cycle', 'cyclic', 'random'],
                        help='LRTT-v2 selector schedule (default: shuffled_cycle)')
    parser.add_argument('--cap-rho', type=float, default=1.0,
                        help='LRTT-v2 capacitor leak factor rho (1.0 = disabled)')
    parser.add_argument('--lrtt-device-type', type=str, default='constant_step',
                        choices=['constant_step', 'floatingpoint'],
                        help='LRTT-v2 Core+Auxiliary array device (default: constant_step)')
    parser.add_argument('--lrtt-weight-bits', type=int, default=10,
                        help='LRTT-v2 array weight resolution in bits for constant_step '
                             '(10 -> dw_min=2/1024). Default: 10')
    # --- Multi-server HP-region-partitioned 2D grid sweep (LR x transfer_lr) ---
    parser.add_argument('--lr-grid', type=float, nargs='+', default=None,
                        help='Explicit learning_rate grid values (GridSampler). '
                             'Partitioned (disjoint) across servers.')
    parser.add_argument('--tlr-grid', type=float, nargs='+', default=None,
                        help='Explicit transfer_lr grid values (GridSampler). '
                             'Swept in full on every server.')
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch = '
                             'batch_size * grad_accum_steps (paper regime: '
                             '16 x 3 = 48). loss/=accum, optimizer.step() only '
                             'on accumulation boundary.')
    parser.add_argument('--min-lr-rate', type=float, default=0.5,
                        help='LR scheduler floor as a fraction of peak LR '
                             '(paper_experiment.py default 0.5; decays linearly '
                             'to min_lr_rate * peak).')
    parser.add_argument('--prune-at-epoch', type=int, default=0,
                        help='Hard prune gate: at this epoch (1-indexed), if '
                             'eval F1 <= --prune-f1-threshold, prune the trial '
                             'and let TPE move to the next one. 0 = disabled. '
                             'When >0 the Optuna MedianPruner is replaced by '
                             'NopPruner so this gate is the only auto-prune.')
    parser.add_argument('--prune-f1-threshold', type=float, default=0.0,
                        help='F1 (%%) threshold for --prune-at-epoch (prune if '
                             'eval F1 <= this value at that epoch).')
    parser.add_argument('--num-servers', type=int, default=1,
                        help='Total servers sharing this sweep (LR grid is split N ways)')
    parser.add_argument('--server-id', type=int, default=0,
                        help='This server index in [0, num-servers). Owns LR sub-grid.')
    args = parser.parse_args()

    # Update global config
    global TRAIN_SUBSET_SIZE, EVAL_SUBSET_SIZE
    TRAIN_SUBSET_SIZE = args.train_subset
    EVAL_SUBSET_SIZE = args.eval_subset
    OPT_CONFIG['mode'] = args.mode
    OPT_CONFIG['rank'] = args.rank
    OPT_CONFIG['selector_policy'] = args.selector_policy
    OPT_CONFIG['cap_rho'] = args.cap_rho
    OPT_CONFIG['lrtt_device_type'] = args.lrtt_device_type
    OPT_CONFIG['lrtt_weight_bits'] = args.lrtt_weight_bits
    OPT_CONFIG['grad_accum_steps'] = max(1, int(args.grad_accum_steps))
    OPT_CONFIG['min_lr_rate'] = args.min_lr_rate
    OPT_CONFIG['prune_at_epoch'] = args.prune_at_epoch
    OPT_CONFIG['prune_f1_threshold'] = args.prune_f1_threshold

    # --- HP-region partition: split LR grid disjointly across servers ---
    _server_lr_grid = None
    if args.lr_grid is not None:
        if not (0 <= args.server_id < args.num_servers):
            parser.error(f"--server-id must be in [0, {args.num_servers})")
        full_lr = sorted(set(args.lr_grid))
        # Round-robin assignment keeps each server's LR range spread (not clustered)
        _server_lr_grid = [v for i, v in enumerate(full_lr)
                           if i % args.num_servers == args.server_id]
        if not _server_lr_grid:
            parser.error(f"server {args.server_id}: empty LR shard "
                         f"(grid={full_lr}, num_servers={args.num_servers})")
        OPT_CONFIG['lr_grid'] = _server_lr_grid
        OPT_CONFIG['lr_range'] = None
        print(f"[SHARD] server {args.server_id}/{args.num_servers}: "
              f"LR grid {_server_lr_grid} (of full {full_lr})")
    if args.tlr_grid is not None:
        OPT_CONFIG['tlr_grid'] = sorted(set(args.tlr_grid))
    BATCH_SIZE = args.batch_size
    EVAL_BATCH_SIZE = args.eval_batch_size
    N_EPOCHS = args.epochs
    WARMUP_RATIO = args.warmup_ratio
    LORA_TARGET = args.lora_target
    HEAD_LAYER = args.head_layer
    TARGET_IDEAL = args.target_ideal
    TARGET_ANALOG = args.target_analog
    if TARGET_ANALOG:
        TARGET_IDEAL = False  # target-analog overrides target-ideal
    OPT_CONFIG['device_type'] = args.device_type
    if args.dw_min is not None:
        OPT_CONFIG['dw_min'] = args.dw_min
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov
    OPT_CONFIG['learn_out_scaling'] = args.learn_out_scaling
    OPT_CONFIG['nontarget_digital'] = args.nontarget_digital
    OPT_CONFIG['nontarget_ideal'] = args.nontarget_ideal
    OPT_CONFIG['analog_only_warmup'] = args.analog_only_warmup
    OPT_CONFIG['train_layernorm'] = args.train_layernorm
    if args.nontarget_ideal:
        OPT_CONFIG['nontarget_digital'] = False  # nontarget-ideal overrides nontarget-digital
    OPT_CONFIG['backward_perfect'] = args.backward_perfect
    OPT_CONFIG['forward_perfect'] = args.forward_perfect
    if args.io_bits is not None:
        OPT_CONFIG['io_bits'] = args.io_bits
    OPT_CONFIG['auto_scale'] = args.auto_scale
    if args.lr is not None:
        OPT_CONFIG['lr_override'] = args.lr
        OPT_CONFIG['lr_range'] = None  # --lr fixes LR, disable sweep
        SQUAD_LR = args.lr
    if args.lr_range is not None:
        OPT_CONFIG['lr_range'] = args.lr_range
    if args.classifier_lr is not None:
        OPT_CONFIG['classifier_lr'] = args.classifier_lr
    if args.classifier_lr_range is not None:
        OPT_CONFIG['classifier_lr_range'] = args.classifier_lr_range
    OPT_CONFIG['shared_lr'] = args.shared_lr
    if args.transfer_every is not None:
        OPT_CONFIG['transfer_every_override'] = args.transfer_every
    OPT_CONFIG['units_in_mbatch'] = args.units_in_mbatch
    OPT_CONFIG['desired_bl'] = args.desired_bl
    if args.bl_sweep is not None:
        OPT_CONFIG['bl_sweep'] = args.bl_sweep
    if args.bl_grid is not None:
        OPT_CONFIG['bl_grid'] = args.bl_grid
    OPT_CONFIG['use_v2'] = args.use_v2
    OPT_CONFIG['scale_transfer_lr'] = not args.no_scale_transfer_lr
    OPT_CONFIG['lr_upper_mult'] = args.lr_upper_mult
    OPT_CONFIG['diag_update'] = args.diag_update
    OPT_CONFIG['diag_steps'] = args.diag_steps

    global CLIP_ANALOG_GRAD, TARGET_LAYERS
    CLIP_ANALOG_GRAD = args.clip_analog_grad
    if args.target_layers is not None:
        TARGET_LAYERS = [i - 1 for i in args.target_layers]  # 1-indexed -> 0-indexed

    global TPE_FLR_RANGE, TPE_TLR_RANGE, SAMPLER_TYPE
    SAMPLER_TYPE = args.sampler
    if args.tpe_flr_range:
        TPE_FLR_RANGE = tuple(args.tpe_flr_range)
    if args.tpe_tlr_range:
        TPE_TLR_RANGE = tuple(args.tpe_tlr_range)

    # --- Per-server TPE sweep-range partition ---------------------------
    # Split the full LR range into num_servers EQUAL log sub-ranges; each
    # server runs an independent TPE search within its own 1/N sub-range
    # (transfer_lr is TPE-searched over the full --tpe-tlr-range on every
    # server). No grid, no shared storage. Skipped if an explicit --lr-grid
    # or a fixed --lr was given.
    if (OPT_CONFIG.get('lr_grid') is None
            and OPT_CONFIG.get('lr_range') is not None
            and args.num_servers > 1):
        if not (0 <= args.server_id < args.num_servers):
            parser.error(f"--server-id must be in [0, {args.num_servers})")
        _flo, _fhi = float(OPT_CONFIG['lr_range'][0]), float(OPT_CONFIG['lr_range'][1])
        _ratio = (_fhi / _flo) ** (1.0 / args.num_servers)
        _slo = _flo * (_ratio ** args.server_id)
        _shi = _flo * (_ratio ** (args.server_id + 1))
        OPT_CONFIG['lr_range'] = [_slo, _shi]
        print(f"[TPE-SHARD] server {args.server_id}/{args.num_servers}: "
              f"LR sub-range [{_slo:.3e}, {_shi:.3e}] of full "
              f"[{_flo:.3e}, {_fhi:.3e}] (equal log split) | "
              f"transfer_lr TPE over {TPE_TLR_RANGE} | n_trials={args.n_trials}")

    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    study_name = args.study_name or f"bert_squad_tiki_{timestamp}"
    # Per-server study + DB so the 4 servers run fully independently
    # (HP-region partition = no shared storage needed); merge JSON afterwards.
    if args.num_servers > 1:
        study_name = f"{study_name}_srv{args.server_id}of{args.num_servers}"

    storage = f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    # Load data once (shared across all trials)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_data(tokenizer)
    print(f"Task: SQuAD v1.1, Metric: F1")
    print(f"BSZ: {BATCH_SIZE}, max_seq: {MAX_SEQ_LENGTH}, epochs: {N_EPOCHS}")
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    task_lr = OPT_CONFIG.get('lr_override', SQUAD_LR)
    _cls_lr_report = OPT_CONFIG.get('classifier_lr', None)
    _cls_lr_range_report = OPT_CONFIG.get('classifier_lr_range', None)
    _lr_range_report = OPT_CONFIG.get('lr_range', None)
    _analog_only_warmup = OPT_CONFIG.get('analog_only_warmup', True)
    print(f"\n{'='*70}")
    print(f"[CONFIG REPORT]")
    print(f"{'='*70}")
    print(f"  Model         : {MODEL_NAME}")
    print(f"  Task          : SQuAD v1.1 | Metric: F1")
    print(f"  Batch size    : {BATCH_SIZE} (train), {EVAL_BATCH_SIZE} (eval)")
    print(f"  Max seq len   : {MAX_SEQ_LENGTH}, Doc stride: {DOC_STRIDE}")
    print(f"  Epochs        : {N_EPOCHS}, Early stop patience: {EARLY_STOP_PATIENCE}")
    print(f"  Optimizer     : {OPT_CONFIG['optimizer']}")
    _lr_str = f"sweep {_lr_range_report} (log)" if _lr_range_report else f"fixed {task_lr:.2e}"
    _cls_str = (f"sweep {_cls_lr_range_report} (log)" if _cls_lr_range_report
                else (f"fixed {_cls_lr_report:.2e}" if _cls_lr_report else "same as analog LR"))
    print(f"  Analog LR     : {_lr_str}")
    print(f"  Classifier/LN : {_cls_str}")
    print(f"  Weight decay  : {'tuned' if OPT_CONFIG['tune_wd'] else '0 (fixed)'}")
    print(f"  Warmup ratio  : {WARMUP_RATIO:.4f}")
    print(f"  Warmup target : {'analog tile ONLY (classifier/LayerNorm -> no warmup, full LR from step 0)' if _analog_only_warmup else 'all param groups'}")
    print(f"  min_lr_rate   : {OPT_CONFIG.get('min_lr_rate', 0.5)} (decay to that fraction of peak LR)")
    print(f"  LORA target   : {LORA_TARGET} -> {get_target_module_names(LORA_TARGET)}")
    _target_cfg_str = 'IdealDevice' if TARGET_IDEAL else f"SingleRPU({OPT_CONFIG.get('device_type', 'softbounds')}, dw_min={OPT_CONFIG.get('dw_min', 'default')})" if TARGET_ANALOG else 'TikiTaka'
    print(f"  Target config : {_target_cfg_str}")
    print(f"  Head layer    : {HEAD_LAYER}")
    print(f"  Target layers : {'all' if TARGET_LAYERS is None else [i+1 for i in TARGET_LAYERS]}")
    print(f"  Nontarget     : {'digital(frozen)' if OPT_CONFIG.get('nontarget_digital') else 'ideal(frozen)' if OPT_CONFIG.get('nontarget_ideal') else 'singleRPU(frozen)'}")
    print(f"  transfer_every: 1 (fixed, uim={OPT_CONFIG.get('units_in_mbatch', True)}, every mini-batch)")
    print(f"  TLR range     : {TPE_TLR_RANGE}, FLR range: {TPE_FLR_RANGE}")
    print(f"  desired_bl    : {OPT_CONFIG.get('desired_bl', 31)}")
    _io_bits = OPT_CONFIG.get('io_bits', None)
    _io_str = f"{_io_bits}-bit (res={1.0/(2**_io_bits-2):.6f})" if _io_bits else "infinite (no quantization)"
    print(f"  IO bits       : {_io_str}")
    print(f"  Clip grad     : analog={CLIP_ANALOG_GRAD}, digital=norm(1.0)")
    print(f"  Seed          : {SEED}")
    print(f"  Sampler       : {SAMPLER_TYPE}")
    print(f"  Train batches : {len(train_loader)}, Eval features: {len(eval_features)}")
    print(f"{'='*70}\n")

    # GridSampler when an explicit LR/transfer_lr grid is given (HP-region
    # sweep). This server enumerates exactly (its LR shard) x (full TLR grid).
    _grid_lr = OPT_CONFIG.get('lr_grid', None)
    _grid_tlr = OPT_CONFIG.get('tlr_grid', None)
    n_trials_eff = args.n_trials
    if _grid_lr is not None and _grid_tlr is not None:
        search_space = {'learning_rate': list(_grid_lr), 'transfer_lr': list(_grid_tlr)}
        sampler = GridSampler(search_space)
        n_trials_eff = len(_grid_lr) * len(_grid_tlr)
        print(f"[GRID] server {args.server_id}/{args.num_servers}: "
              f"{len(_grid_lr)} LR x {len(_grid_tlr)} TLR = {n_trials_eff} trials")
    else:
        # Per-server TPE seed so the 4 servers explore independently
        # (each within its own LR sub-range).
        sampler = optuna.samplers.TPESampler(seed=SEED + args.server_id)
        print(f"[TPE] server {args.server_id}/{args.num_servers}: "
              f"TPESampler(seed={SEED + args.server_id}), n_trials={n_trials_eff}")

    # Pruner: when the hard --prune-at-epoch gate is active, use NopPruner so
    # the explicit gate is the ONLY automatic prune (predictable). Early
    # stopping (manual break, patience=2) stays active regardless. Otherwise
    # fall back to MedianPruner.
    prune_warmup = max(1, N_EPOCHS // 3)
    if args.prune_at_epoch and args.prune_at_epoch > 0:
        _pruner = optuna.pruners.NopPruner()
        print(f"[PRUNE] hard gate: epoch {args.prune_at_epoch} F1 <= "
              f"{args.prune_f1_threshold:.2f}% -> prune (MedianPruner OFF; "
              f"early-stopping patience={EARLY_STOP_PATIENCE} ON)")
    else:
        _pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=prune_warmup,
        )
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=sampler,
        pruner=_pruner,
        load_if_exists=True,
    )
    print(f"  Early stop patience: {EARLY_STOP_PATIENCE}, "
          f"Pruner: Median, startup=5, warmup={prune_warmup}")

    print(f"\nStudy: {study_name}, Device: {DEVICE}, New trials: {n_trials_eff}")

    target_total = len(study.trials) + n_trials_eff

    try:
        study.optimize(
            lambda trial: objective(trial, train_loader, eval_features, eval_examples, tokenizer),
            n_trials=n_trials_eff,
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
                "task": "squad",
                "metric": "f1",
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
    all_trials.sort(key=lambda x: x["value"] if x["value"] is not None else -1, reverse=True)

    all_trials_file = os.path.join(RESULTS, f"all_trials_bert_squad.json")
    with open(all_trials_file, 'w') as f:
        json.dump(all_trials, f, indent=2)
    print(f"All trials saved to: {all_trials_file}")


class _OOMRestart(Exception):
    pass


def _oom_restart_callback(study, trial):
    if trial.state == TrialState.FAIL:
        err = trial.user_attrs.get("error", "")
        if "out of memory" in err.lower() or "cublas" in err.lower():
            print(f"\n[OOM Recovery] Trial {trial.number} failed with CUDA error, will restart process.")
            raise _OOMRestart()


if __name__ == "__main__":
    main()
