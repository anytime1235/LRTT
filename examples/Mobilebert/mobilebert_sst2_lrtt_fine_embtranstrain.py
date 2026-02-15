# -*- coding: utf-8 -*-
"""MobileBERT + SST-2 with LRTT (Low-Rank TikiTaka Training).

Single-run training script for MobileBERT on SST-2 using LRTT analog layers.
Converts Q/K/V attention layers to analog; all other layers remain digital.

Based on mobilebert_squad_lrtt_fine.py, adapted for SST-2 classification.

*** ORIGINAL VERSION: embedding_transformation is Digital TRAINABLE ***
*** Best Trial 50 reproduction (84.40%) with A/B/C tile diagnostics ***

Inline flags (edit directly in script):
    N_EPOCHS = 3                     # Number of training epochs
    BATCH_SIZE = 64                 # Training batch size
    LEARNING_RATE = 0.208075        # Peak learning rate (Trial 50)
    WEIGHT_DECAY = 0.0              # Weight decay
    WARMUP_STEPS = 189              # LR scheduler warmup steps (~6% of 3 epochs)
    MIN_LR_RATE = 0.0               # Min LR as fraction of peak (0 = decay to zero)
    OPTIMIZER = "AnalogSGD"         # "AnalogSGD" | "AnalogAdam"
    LRTT_RANK = 2                   # LoRA rank for LRTT (Trial 50)
    TRANSFER_EVERY = 16210          # Transfer interval in samples (Trial 50)
    TRANSFER_LR = 0.01              # Transfer learning rate (Trial 50)
    TRANSFER_METHOD = "set"         # Transfer method: "onehot" | "direct" | "set"
    LORA_ALPHA = 0.411396           # LoRA alpha scaling (Trial 50)
    REINIT_MODE = "hybrid"          # Reinit mode: "standard" | "decay" | "hybrid"
    REINIT_GAIN = 1.0               # Reinitialization gain
    DECAY_FACTOR = 1.0              # Decay factor for reinit
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
    default_data_collator,
    set_seed,
)
from datasets import load_dataset

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

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
RESULTS = os.path.join(os.getcwd(), "results", "MOBILEBERT_SST2_LRTT_FINE")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "mobilebert_sst2_lrtt_fine_model_weight.pth")

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128
NUM_LABELS = 2  # SST-2: negative (0), positive (1)

# Training
N_EPOCHS = 3
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 0.208075  # Trial 50 best
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 3

# Scheduler
WARMUP_STEPS = 189  # ~6% of total steps (3 epochs)
MIN_LR_RATE = 0.0  # Fraction of peak LR (0 = decay to zero)

# Optimizer
OPTIMIZER = "AnalogSGD"  # "AnalogSGD" or "AnalogAdam"

# LRTT parameters (Trial 50 best)
LRTT_RANK = 2
TRANSFER_EVERY = 16210  # Transfer interval in samples (Trial 50, ~253 steps)
TRANSFER_LR = 0.01
LORA_ALPHA = 0.411396
REINIT_MODE = "hybrid"
REINIT_GAIN = 1.0
DECAY_FACTOR = 1.0
TRANSFER_METHOD = "set"  # "onehot", "direct", or "set"

# 6T1C Retention parameters
TAU_SEC = 0.0  # 0 = no decay, >0 = retention time constant

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
LORA_TARGET_MODULES = {
    "none": [],  # Empty = no layers converted to LRTT (fully digital)
    "qonly": ["query"],  # Query only (24 layers)
    "konly": ["key"],  # Key only (24 layers)
    "vonly": ["value"],  # Value only (24 layers)
    "qkv": ["query", "key", "value"],  # Q/K/V (72 layers)
    "ffn": ["dense"],  # All layers with "dense" (excludes qkv) (288 layers)
    "all": None,  # None means all encoder layers (no filtering) (360 layers)
}

# Diagnostic
ENABLE_DIAGNOSTIC = True   # False = no diagnostic overhead, fast training
DIAG_EPOCHS = 0            # 0 = all epochs, N = first N epochs only

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

# WandB
WANDB_PROJECT = "mobilebert-sst2-lrtt-fine"
os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# LRTT Device Functions
# =============================================================================

def _create_6t1c_device():
    """Create 6T1C LinearStepDevice.

    Uses TAU_SEC for retention lifetime. If TAU_SEC=0, lifetime=0 (no decay).
    """
    if TAU_SEC > 0:
        dt_batch_sec = 1.0
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        lifetime = 1.0 / delta if delta > 0 else 0.0
    else:
        lifetime = 0.0

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
        mult_noise=False,  # No multiplicative noise for C tile
    )


def create_lrtt_config():
    """Create LRTT RPU configuration for analog layers."""
    ab_device = _create_6t1c_device()
    c_device = _create_c_device()

    te = TRANSFER_EVERY
    device_config = PythonLRTTDevice(
        rank=LRTT_RANK,
        transfer_every=te,
        lora_alpha=LORA_ALPHA,
        reinit_gain=REINIT_GAIN,
        reinit_mode=REINIT_MODE,
        decay_factor=DECAY_FACTOR,
        unit_cell_devices=[ab_device, ab_device, c_device],
        train_c_bias=False,        # C tile bias frozen
        mapping_ab=MappingParameter(
            weight_scaling_omega=0.0,
            learn_out_scaling=False,
        ),
        mapping_c=MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=True,
            out_scaling_columnwise=True,
        ),
    )
    device_config.transfer_lr = TRANSFER_LR
    device_config.units_in_mbatch = True
    device_config.transfer_method = TRANSFER_METHOD
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = False

    # Dynamic TE: increase TE as LR decays
    device_config.dynamic_te = DYNAMIC_TE
    device_config.dynamic_te_power = DYNAMIC_TE_POWER
    device_config.dynamic_te_max = te * 20
    device_config.te_warmup_schedule = TE_WARMUP_SCHEDULE + [te]
    device_config.te_warmup_steps = TE_WARMUP_STEPS

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    # Set IO noise to 0.0 (per spec)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0

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
        return ["query"]  # Query only (24 layers)
    elif lora_target == "konly":
        return ["key"]  # Key only (24 layers)
    elif lora_target == "vonly":
        return ["value"]  # Value only (24 layers)
    elif lora_target == "qkv":
        return ["query", "key", "value"]  # Q/K/V (72 layers)
    elif lora_target == "ffn":
        return ["dense"]  # All layers with "dense" in name (excludes qkv) (288 layers)
    elif lora_target == "all":
        # All encoder linear layers (exclude embeddings, classifier, embedding_transformation)
        return None  # None means all encoder layers (360 layers)
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model():
    """Create MobileBERT classification model with selective LRTT analog layers.

    *** ORIGINAL VERSION: embedding_transformation is always Digital TRAINABLE ***

    Architecture (follows paper's approach for efficiency):
        - LRTT Target layers (based on LORA_TARGET) -> LRTT Analog
        - Non-target Encoder layers -> Digital FROZEN
        - classifier -> Digital TRAINABLE (weight + bias)
        - embedding_transformation -> Digital TRAINABLE (weight + bias)
        - Embeddings -> Digital FROZEN

    LoRA Target Options (LORA_TARGET):
        - qkv: Q/K/V layers -> LRTT Analog (72 layers)
        - ffn: projection + FFN layers -> LRTT Analog (288 layers)
        - all: all encoder layers -> LRTT Analog (360 layers)

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
        # embedding_transformation is always digital (ORIGINAL BEHAVIOR)
        if "embedding_transformation" in layer_name:
            return False
        # Must be in encoder for other layers
        if "encoder" not in layer_name:
            return False
        # If lrtt_patterns is None (all mode), all encoder layers are targets
        if lrtt_patterns is None:
            return True
        return any(p in layer_name for p in lrtt_patterns)

    # Build exclude list: all layers that should NOT be converted to LRTT
    all_linear_names = list_linear_layers(model)
    exclude_modules = []
    for name in all_linear_names:
        if not is_lrtt_target(name):
            # Use full path for exclude_modules (convert_to_analog requires exact match)
            exclude_modules.append(name)

    # Always exclude classifier and embedding_transformation (ORIGINAL BEHAVIOR)
    exclude_modules.append("classifier")
    exclude_modules.append("mobilebert.embeddings.embedding_transformation")
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
    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Step 2: Set requires_grad
    # - LRTT layers: A/B + out_scaling TRAINABLE, C + bias FROZEN
    # - classifier: TRAINABLE if HEAD_LAYER=="train", else FROZEN
    # - embedding_transformation: TRAINABLE (ORIGINAL BEHAVIOR)
    # - Everything else: FROZEN
    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            param.requires_grad = True
        elif "tile_c" in name:
            pass  # Respect lrtt_tile.py settings (train_c_bias, mapping_c)
        elif "classifier" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        elif "embedding_transformation" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_params = count_parameters(model)

    print(f"\nCreated MobileBERT model (LRTT):")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Task: SST-2 (num_labels={NUM_LABELS})")
    print(f"  Total params: {total_params:,}, Trainable: {num_params:,}")
    print(f"  LRTT Analog layers: {num_analog}")
    print(f"  LRTT config: rank={LRTT_RANK}, transfer_every={TRANSFER_EVERY}, "
          f"transfer_lr={TRANSFER_LR}, lora_alpha={LORA_ALPHA}")
    print(f"  Reinit: mode={REINIT_MODE}, gain={REINIT_GAIN}")
    print(f"  LoRA target: {LORA_TARGET} -> {lrtt_patterns if lrtt_patterns else 'all encoder layers'}")
    print(f"  embedding_transformation: Digital TRAINABLE (original behavior)")

    return model.to(DEVICE)


# =============================================================================
# Data Functions
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize SST-2 dataset."""
    raw_datasets = load_dataset("glue", "sst2")

    def preprocess_function(examples):
        return tokenizer(
            examples["sentence"],
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding="max_length",
        )

    # Tokenize datasets
    tokenized_train = raw_datasets["train"].map(
        preprocess_function, batched=True,
        remove_columns=["sentence", "idx"]
    )

    tokenized_eval = raw_datasets["validation"].map(
        preprocess_function, batched=True,
        remove_columns=["sentence", "idx"]
    )

    # Use subset if specified
    if TRAIN_SUBSET_SIZE > 0:
        tokenized_train = tokenized_train.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(tokenized_train)))
        )
    else:
        tokenized_train = tokenized_train.shuffle(seed=SEED)

    if EVAL_SUBSET_SIZE > 0:
        tokenized_eval = tokenized_eval.select(
            range(min(EVAL_SUBSET_SIZE, len(tokenized_eval)))
        )

    # Rename "label" to "labels" for HuggingFace compatibility
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_eval = tokenized_eval.rename_column("label", "labels")

    train_loader = DataLoader(
        tokenized_train, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=default_data_collator,
        generator=torch.Generator().manual_seed(SEED)
    )

    eval_loader = DataLoader(
        tokenized_eval, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=default_data_collator
    )

    return train_loader, eval_loader


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(model, eval_loader):
    """Evaluate SST-2 model. Returns accuracy."""
    model.eval()

    all_preds = []
    all_labels = []

    with no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    model.train()

    # Compute accuracy
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100.0

    return accuracy


# =============================================================================
# Optimizer & Scheduler
# =============================================================================

def create_optimizer(model):
    """Create optimizer. Uses standard PyTorch for none mode, Analog for LRTT modes."""
    if LORA_TARGET == "none":
        # None mode: use standard PyTorch optimizers
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
        # LRTT modes: use Analog optimizers
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
# Diagnostic Constants
# =============================================================================

# Fixed cell indices for A [d_size, rank=2], B [rank=2, x_size], C [d_size, x_size]
# A: [128, 2] — pick 10 from the small matrix
A_CELL_INDICES = [
    (0, 0), (0, 1), (1, 0), (1, 1), (10, 0),
    (10, 1), (32, 0), (64, 0), (64, 1), (127, 0),
]
# B: [2, 128] — pick 10 from the small matrix
B_CELL_INDICES = [
    (0, 0), (0, 1), (0, 32), (0, 64), (0, 127),
    (1, 0), (1, 1), (1, 32), (1, 64), (1, 127),
]
# C: [128, 128] — spread across the matrix
C_CELL_INDICES = [
    (0, 0), (0, 64), (0, 127), (16, 16), (32, 32),
    (48, 48), (64, 64), (80, 80), (96, 96), (127, 127),
]


# =============================================================================
# Diagnostic Helpers
# =============================================================================

def find_first_lrtt_tile(model):
    """Find the first LRTT tile (layer 0 query) in the model."""
    for name, mod in model.named_modules():
        if hasattr(mod, 'analog_module'):
            am = mod.analog_module
            if hasattr(am, 'controller'):
                return name, am
    raise RuntimeError("No LRTT tile found in model")


def find_last_lrtt_tile(model):
    """Find the last LRTT tile (layer 23 value) in the model."""
    last_name, last_tile = None, None
    for name, mod in model.named_modules():
        if hasattr(mod, 'analog_module'):
            am = mod.analog_module
            if hasattr(am, 'controller'):
                last_name, last_tile = name, am
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
    if hasattr(tile_c, 'out_scaling_alpha'):
        alpha = tile_c.out_scaling_alpha.detach().to(W_scaled.device)
        return W_scaled / alpha.unsqueeze(1)
    return W_scaled


def snapshot_weights(tile):
    """Snapshot A, B, C weights (clone + detach) before optimizer step."""
    return (
        tile.tile_a.get_weights()[0].clone().detach(),
        tile.tile_b.get_weights()[0].clone().detach(),
        tile.tile_c.get_weights()[0].clone().detach(),
        get_raw_C(tile.tile_c).clone().detach(),
    )


def collect_tile_diagnostics(tile, C_prev_raw, A_before, B_before, C_before,
                             C_raw_before, step, prev_num_transfers):
    """Collect all diagnostic data for one tile at one step."""
    controller = tile.controller

    A = tile.tile_a.get_weights()[0]
    B = tile.tile_b.get_weights()[0]
    C = tile.tile_c.get_weights()[0]
    C_raw = get_raw_C(tile.tile_c)

    norm_A = torch.norm(A).item()
    norm_B = torch.norm(B).item()
    norm_C_raw = torch.norm(C_raw).item()
    norm_AB = torch.norm(A @ B).item()

    delta_C_raw = torch.norm(C_raw - C_prev_raw).item() if C_prev_raw is not None else 0.0
    delta_A = torch.norm(A - A_before).item() if A_before is not None else 0.0
    delta_B = torch.norm(B - B_before).item() if B_before is not None else 0.0
    delta_C_raw_step = torch.norm(C_raw - C_raw_before).item() if C_raw_before is not None else 0.0

    A_cells = sample_cells(A, A_CELL_INDICES)
    B_cells = sample_cells(B, B_CELL_INDICES)
    C_cells = sample_cells(C_raw, C_CELL_INDICES)

    # Gradient deltas (before/after step)
    A_grad_cells = []
    B_grad_cells = []
    C_grad_cells = []
    if A_before is not None:
        dA = A - A_before
        A_grad_cells = sample_cells(dA, A_CELL_INDICES)
    if B_before is not None:
        dB = B - B_before
        B_grad_cells = sample_cells(dB, B_CELL_INDICES)
    if C_raw_before is not None:
        dC = C_raw - C_raw_before
        C_grad_cells = sample_cells(dC, C_CELL_INDICES)

    transfer_counter = controller.transfer_counter
    num_transfers = controller.num_transfers
    is_transfer = num_transfers > prev_num_transfers

    record = {
        "step": step,
        "norm_A": norm_A,
        "norm_B": norm_B,
        "norm_C_raw": norm_C_raw,
        "norm_AB": norm_AB,
        "A_cells": A_cells,
        "B_cells": B_cells,
        "C_cells": C_cells,
        "A_grad_cells": A_grad_cells,
        "B_grad_cells": B_grad_cells,
        "C_grad_cells": C_grad_cells,
        "delta_A": delta_A,
        "delta_B": delta_B,
        "delta_C_raw": delta_C_raw_step,
        "transfer_counter": transfer_counter,
        "num_transfers": num_transfers,
        "is_transfer": is_transfer,
    }

    return record, C_raw.clone().detach(), num_transfers


# =============================================================================
# Diagnostic Plotting
# =============================================================================

def make_diagnostic_plots(log_data, output_path, tile_label=""):
    """Create 5x2 (10 panel) diagnostic plot for one tile."""
    steps = [r["step"] for r in log_data]
    norm_A = [r["norm_A"] for r in log_data]
    norm_B = [r["norm_B"] for r in log_data]
    norm_C_raw = [r["norm_C_raw"] for r in log_data]
    norm_AB = [r["norm_AB"] for r in log_data]
    losses = [r.get("loss", 0.0) for r in log_data]

    transfer_steps = [r["step"] for r in log_data if r["is_transfer"]]

    n_cells = len(log_data[0]["A_cells"])
    A_w_series = [[r["A_cells"][i] for r in log_data] for i in range(n_cells)]
    B_w_series = [[r["B_cells"][i] for r in log_data] for i in range(n_cells)]
    C_w_series = [[r["C_cells"][i] for r in log_data] for i in range(len(log_data[0]["C_cells"]))]
    A_g_series = [[r["A_grad_cells"][i] if r["A_grad_cells"] else 0.0 for r in log_data] for i in range(n_cells)]
    B_g_series = [[r["B_grad_cells"][i] if r["B_grad_cells"] else 0.0 for r in log_data] for i in range(n_cells)]
    C_g_series = [[r["C_grad_cells"][i] if r["C_grad_cells"] else 0.0 for r in log_data] for i in range(len(log_data[0]["C_cells"]))]

    fig, axes = plt.subplots(5, 2, figsize=(18, 28))
    title_str = f"SST-2 LRTT Diagnostic — {tile_label}" if tile_label else "SST-2 LRTT Diagnostic"
    fig.suptitle(title_str, fontsize=14)

    def add_transfer_lines(ax):
        for ts in transfer_steps:
            ax.axvline(x=ts, color="red", alpha=0.3, linewidth=0.8)

    # Panel (0,0): A/B/AB norms
    ax = axes[0, 0]
    ax.plot(steps, norm_A, label="||A||", alpha=0.8)
    ax.plot(steps, norm_B, label="||B||", alpha=0.8)
    ax.plot(steps, norm_AB, label="||A@B||", alpha=0.6, linestyle="--")
    add_transfer_lines(ax)
    ax.set_xlabel("Step"); ax.set_ylabel("Norm")
    ax.set_title("A, B, AB Norms (red = transfer)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel (0,1): C norm
    ax = axes[0, 1]
    ax.plot(steps, norm_C_raw, label="||C_raw||", color="green", alpha=0.8)
    delta_C = [r["delta_C_raw"] for r in log_data]
    ax2 = ax.twinx()
    ax2.plot(steps, delta_C, label="delta_C_raw", color="orange", alpha=0.8)
    add_transfer_lines(ax)
    ax.set_xlabel("Step"); ax.set_ylabel("||C_raw||", color="green")
    ax2.set_ylabel("delta_C_raw", color="orange")
    ax.set_title("C Norm (raw) + delta_C_raw")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel (1,0): A cells weight
    ax = axes[1, 0]
    for i, series in enumerate(A_w_series):
        r, c = A_CELL_INDICES[i]
        ax.plot(steps, series, label=f"A[{r},{c}]", alpha=0.7, linewidth=0.8)
    add_transfer_lines(ax)
    ax.set_xlabel("Step"); ax.set_ylabel("Weight value")
    ax.set_title("A cells: weight values")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # Panel (1,1): A cells grad (delta)
    ax = axes[1, 1]
    for i, series in enumerate(A_g_series):
        r, c = A_CELL_INDICES[i]
        ax.plot(steps, series, label=f"dA[{r},{c}]", alpha=0.7, linewidth=0.8)
    add_transfer_lines(ax)
    ax.set_xlabel("Step"); ax.set_ylabel("Delta (before/after)")
    ax.set_title("A cells: per-step delta")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # Panel (2,0): B cells weight
    ax = axes[2, 0]
    for i, series in enumerate(B_w_series):
        r, c = B_CELL_INDICES[i]
        ax.plot(steps, series, label=f"B[{r},{c}]", alpha=0.7, linewidth=0.8)
    add_transfer_lines(ax)
    ax.set_xlabel("Step"); ax.set_ylabel("Weight value")
    ax.set_title("B cells: weight values")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # Panel (2,1): B cells grad (delta)
    ax = axes[2, 1]
    for i, series in enumerate(B_g_series):
        r, c = B_CELL_INDICES[i]
        ax.plot(steps, series, label=f"dB[{r},{c}]", alpha=0.7, linewidth=0.8)
    add_transfer_lines(ax)
    ax.set_xlabel("Step"); ax.set_ylabel("Delta (before/after)")
    ax.set_title("B cells: per-step delta")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # Panel (3,0): C cells weight
    ax = axes[3, 0]
    for i, series in enumerate(C_w_series):
        r, c = C_CELL_INDICES[i]
        ax.plot(steps, series, label=f"C[{r},{c}]", alpha=0.7, linewidth=0.8)
    add_transfer_lines(ax)
    ax.set_xlabel("Step"); ax.set_ylabel("Weight value")
    ax.set_title("C cells (raw): weight values")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # Panel (3,1): C cells grad (delta)
    ax = axes[3, 1]
    for i, series in enumerate(C_g_series):
        r, c = C_CELL_INDICES[i]
        ax.plot(steps, series, label=f"dC[{r},{c}]", alpha=0.7, linewidth=0.8)
    add_transfer_lines(ax)
    ax.set_xlabel("Step"); ax.set_ylabel("Delta (before/after)")
    ax.set_title("C cells (raw): per-step delta")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # Panel (4,0): G_accum / tlr*AB norms (lines) + dC norm at transfers (markers) + loss
    norm_G_accum = [max(r.get("norm_G_accum", 1e-10), 1e-10) for r in log_data]
    norm_tlrAB = [max(r.get("norm_tlrAB", 1e-10), 1e-10) for r in log_data]
    # delta_C only at transfer steps
    t_steps_dC = [r["step"] for r in log_data if r["is_transfer"]]
    t_norms_dC = [max(r.get("norm_dC_step", 1e-10), 1e-10) for r in log_data if r["is_transfer"]]

    ax = axes[4, 0]
    ax.semilogy(steps, norm_G_accum, label="||G_accum||", color="red", alpha=0.8, linewidth=0.8)
    ax.semilogy(steps, norm_tlrAB, label="||tlr*A@B||", color="green", alpha=0.8, linewidth=0.8)
    if t_steps_dC:
        ax.semilogy(t_steps_dC, t_norms_dC, 'o', label="||delta_C|| @T", color="blue",
                     markersize=5, alpha=0.9, zorder=5)
    add_transfer_lines(ax)
    ax.set_xlabel("Step"); ax.set_ylabel("Norm (log)")
    # Loss on secondary y-axis
    ax_loss = ax.twinx()
    ax_loss.plot(steps, losses, label="loss", color="gray", alpha=0.35, linewidth=0.6)
    ax_loss.set_ylabel("Loss", color="gray")
    lines_main, labels_main = ax.get_legend_handles_labels()
    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    ax.legend(lines_main + lines_loss, labels_main + labels_loss, fontsize=7, loc="upper right")
    ax.set_title("||G_accum|| vs ||tlr*A@B|| + ||delta_C|| at transfers + Loss")
    ax.grid(True, alpha=0.3)

    # Panel (4,1): cos(tlr*AB, G) line + cos(dC, *) at transfers (markers) + loss
    cos_tlrAB_G = [r.get("cos_tlrAB_G", 0.0) for r in log_data]
    t_cos_dC_G = [r.get("cos_dC_G", 0.0) for r in log_data if r["is_transfer"]]
    t_cos_dC_tlrAB = [r.get("cos_dC_tlrAB", 0.0) for r in log_data if r["is_transfer"]]

    ax = axes[4, 1]
    ax.plot(steps, cos_tlrAB_G, label="cos(tlr*AB, G)", color="green", alpha=0.7, linewidth=0.8)
    if t_steps_dC:
        ax.scatter(t_steps_dC, t_cos_dC_G, label="cos(dC, G) @T", color="blue",
                   s=25, alpha=0.9, zorder=5, marker="o")
        ax.scatter(t_steps_dC, t_cos_dC_tlrAB, label="cos(dC, tlr*AB) @T", color="purple",
                   s=25, alpha=0.9, zorder=5, marker="s")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.3)
    add_transfer_lines(ax)
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(-1.1, 1.1)
    # Loss on secondary y-axis
    ax_loss2 = ax.twinx()
    ax_loss2.plot(steps, losses, label="loss", color="gray", alpha=0.35, linewidth=0.6)
    ax_loss2.set_ylabel("Loss", color="gray")
    lines_cos, labels_cos = ax.get_legend_handles_labels()
    lines_l2, labels_l2 = ax_loss2.get_legend_handles_labels()
    ax.legend(lines_cos + lines_l2, labels_cos + labels_l2, fontsize=6, loc="lower left")
    ax.set_xlabel("Step")
    ax.set_title("Cosines: dC vs G, tlr*AB vs G, dC vs tlr*AB + Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


# =============================================================================
# Main
# =============================================================================

def _cos_sim(a, b):
    """Cosine similarity between two flat tensors."""
    na, nb = torch.norm(a).item(), torch.norm(b).item()
    if na > 1e-10 and nb > 1e-10:
        return torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)).item()
    return 0.0


def main():
    """Train MobileBERT with LRTT on SST-2 with A/B/C tile diagnostics."""
    manual_seed(SEED)
    set_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"mobilebert_lrtt_r{LRTT_RANK}_te{TRANSFER_EVERY}_bs{BATCH_SIZE}",
        config={
            "model": MODEL_NAME, "dataset": "SST-2",
            "lrtt_rank": LRTT_RANK, "transfer_every": TRANSFER_EVERY,
            "transfer_lr": TRANSFER_LR, "lora_alpha": LORA_ALPHA,
            "reinit_mode": REINIT_MODE, "reinit_gain": REINIT_GAIN,
            "tau_sec": TAU_SEC,
            "dynamic_te": DYNAMIC_TE, "te_warmup_steps": TE_WARMUP_STEPS,
            "epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "optimizer": OPTIMIZER, "warmup_steps": WARMUP_STEPS,
            "min_lr_rate": MIN_LR_RATE, "seed": SEED,
            "lora_target": LORA_TARGET,
            "transfer_method": TRANSFER_METHOD,
        }
    )

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # Create model, optimizer, scheduler
    model = create_model()
    optimizer = create_optimizer(model)

    num_training_steps = len(train_loader) * N_EPOCHS
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

    if ENABLE_DIAGNOSTIC:
        first_name, first_tile = find_first_lrtt_tile(model)
        last_name, last_tile = find_last_lrtt_tile(model)
        print(f"\nDiag tile (first): {first_name}")
        print(f"  A: {first_tile.tile_a.get_weights()[0].shape}, "
              f"B: {first_tile.tile_b.get_weights()[0].shape}, "
              f"C: {first_tile.tile_c.get_weights()[0].shape}")
        print(f"Diag tile (last):  {last_name}")
        print(f"  A: {last_tile.tile_a.get_weights()[0].shape}, "
              f"B: {last_tile.tile_b.get_weights()[0].shape}, "
              f"C: {last_tile.tile_c.get_weights()[0].shape}")
        print(f"Diag epochs: {'all' if DIAG_EPOCHS == 0 else f'first {DIAG_EPOCHS}'}")

        def _install_hook(diag_tile, device, gc_dict):
            C_shape_h = diag_tile.tile_c.get_weights()[0].shape
            d_size, x_size = C_shape_h
            gc_dict['G_accum'] = torch.zeros(d_size, x_size, device=device)
            gc_dict['active'] = True

            original_fn = diag_tile.controller.ab_weight_update

            def hooked(x, d, lr, **kwargs):
                if gc_dict.get('active'):
                    with torch.no_grad():
                        x_2d = x.reshape(-1, x.shape[-1])
                        d_2d = d.reshape(-1, d.shape[-1])
                        gc_dict['G_accum'] = gc_dict['G_accum'] + d_2d.t() @ x_2d
                result = original_fn(x, d, lr, **kwargs)
                if gc_dict.get('active'):
                    A = diag_tile.tile_a.get_weights()[0].to(device)
                    B = diag_tile.tile_b.get_weights()[0].to(device)
                    AB = A @ B
                    gc_dict['AB_matrix'] = AB.clone()
                    gc_dict['norm_AB_pre'] = torch.norm(AB).item()
                    gc_dict['norm_G_accum'] = torch.norm(gc_dict['G_accum']).item()
                    AB_flat = AB.flatten()
                    G_flat = gc_dict['G_accum'].flatten()
                    cos_sim = 0.0
                    if gc_dict['norm_AB_pre'] > 1e-10 and gc_dict['norm_G_accum'] > 1e-10:
                        cos_sim = torch.nn.functional.cosine_similarity(
                            AB_flat.unsqueeze(0), G_flat.unsqueeze(0)
                        ).item()
                    gc_dict['cos_AB_G'] = cos_sim
                return result

            diag_tile.controller.ab_weight_update = hooked

        _install_hook(first_tile, DEVICE, first_gc)
        _install_hook(last_tile, DEVICE, last_gc)
        print("Gradient tracking hooks installed")

    # Initial evaluation
    init_acc = evaluate_model(model, eval_loader)
    wandb.log({"epoch": 0, "eval/accuracy": init_acc})
    print(f"Initial eval: Accuracy={init_acc:.2f}%")

    # Training loop
    best_acc = init_acc
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    print(f"\nStarting training: {N_EPOCHS} epochs (max), early stopping patience={EARLY_STOP_PATIENCE}")
    print(f"Expected transfer interval: ~{TRANSFER_EVERY} steps (units_in_mbatch=True -> ~{TRANSFER_EVERY // BATCH_SIZE} batches)")

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Diagnostic: snapshot before optimizer step
            diag_active = ENABLE_DIAGNOSTIC and (DIAG_EPOCHS == 0 or epoch <= DIAG_EPOCHS)
            if diag_active:
                first_snap = snapshot_weights(first_tile)
                last_snap = snapshot_weights(last_tile)

            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            num_batches += 1

            if diag_active:
                # --- Collect diagnostics for both tiles ---
                # First tile
                first_A_bef, first_B_bef, first_C_bef, first_Craw_bef = first_snap
                rec_first, first_C_prev_raw, first_prev_nt = collect_tile_diagnostics(
                    first_tile, first_C_prev_raw, first_A_bef, first_B_bef,
                    first_C_bef, first_Craw_bef, global_step, first_prev_nt
                )
                rec_first["loss"] = loss.item()
                rec_first["norm_G_accum"] = first_gc.get('norm_G_accum', 0.0)
                rec_first["norm_AB_pre"] = first_gc.get('norm_AB_pre', 0.0)
                rec_first["cos_AB_G"] = first_gc.get('cos_AB_G', 0.0)

                with torch.no_grad():
                    C_raw_after = get_raw_C(first_tile.tile_c).to(DEVICE)
                    delta_C_mat = C_raw_after - first_Craw_bef.to(DEVICE)
                    AB_mat = first_gc.get('AB_matrix')
                    tlr_AB = TRANSFER_LR * AB_mat if AB_mat is not None else torch.zeros_like(delta_C_mat)
                    # Use controller's exact deltas for cosine comparison at transfer steps
                    if rec_first["is_transfer"]:
                        ctrl_delta = first_tile.controller.last_transfer_delta
                        actual_delta = first_tile.controller.last_actual_delta
                        if ctrl_delta is not None:
                            tlr_AB = ctrl_delta.to(DEVICE)
                        if actual_delta is not None:
                            delta_C_mat = actual_delta.to(DEVICE)
                    dC_flat = delta_C_mat.flatten()
                    G_flat = first_gc.get('G_accum', torch.zeros_like(delta_C_mat)).flatten()
                    tlr_flat = tlr_AB.flatten()
                    rec_first["cos_dC_G"] = _cos_sim(dC_flat, G_flat)
                    rec_first["cos_tlrAB_G"] = _cos_sim(tlr_flat, G_flat)
                    rec_first["cos_dC_tlrAB"] = _cos_sim(dC_flat, tlr_flat)
                    rec_first["norm_dC_step"] = torch.norm(delta_C_mat).item()
                    rec_first["norm_tlrAB"] = torch.norm(tlr_AB).item()

                if rec_first["is_transfer"]:
                    rec_first["norm_G_at_transfer"] = torch.norm(first_gc.get('G_accum', torch.zeros(1))).item()
                    first_gc['G_accum'] = torch.zeros_like(first_gc['G_accum'])
                first_log.append(rec_first)

                # Last tile
                last_A_bef, last_B_bef, last_C_bef, last_Craw_bef = last_snap
                rec_last, last_C_prev_raw, last_prev_nt = collect_tile_diagnostics(
                    last_tile, last_C_prev_raw, last_A_bef, last_B_bef,
                    last_C_bef, last_Craw_bef, global_step, last_prev_nt
                )
                rec_last["loss"] = loss.item()
                rec_last["norm_G_accum"] = last_gc.get('norm_G_accum', 0.0)
                rec_last["norm_AB_pre"] = last_gc.get('norm_AB_pre', 0.0)
                rec_last["cos_AB_G"] = last_gc.get('cos_AB_G', 0.0)

                with torch.no_grad():
                    C_raw_after_l = get_raw_C(last_tile.tile_c).to(DEVICE)
                    delta_C_mat_l = C_raw_after_l - last_Craw_bef.to(DEVICE)
                    AB_mat_l = last_gc.get('AB_matrix')
                    tlr_AB_l = TRANSFER_LR * AB_mat_l if AB_mat_l is not None else torch.zeros_like(delta_C_mat_l)
                    # Use controller's exact deltas for cosine comparison at transfer steps
                    if rec_last["is_transfer"]:
                        ctrl_delta_l = last_tile.controller.last_transfer_delta
                        actual_delta_l = last_tile.controller.last_actual_delta
                        if ctrl_delta_l is not None:
                            tlr_AB_l = ctrl_delta_l.to(DEVICE)
                        if actual_delta_l is not None:
                            delta_C_mat_l = actual_delta_l.to(DEVICE)
                    dC_flat_l = delta_C_mat_l.flatten()
                    G_flat_l = last_gc.get('G_accum', torch.zeros_like(delta_C_mat_l)).flatten()
                    tlr_flat_l = tlr_AB_l.flatten()
                    rec_last["cos_dC_G"] = _cos_sim(dC_flat_l, G_flat_l)
                    rec_last["cos_tlrAB_G"] = _cos_sim(tlr_flat_l, G_flat_l)
                    rec_last["cos_dC_tlrAB"] = _cos_sim(dC_flat_l, tlr_flat_l)
                    rec_last["norm_dC_step"] = torch.norm(delta_C_mat_l).item()
                    rec_last["norm_tlrAB"] = torch.norm(tlr_AB_l).item()

                if rec_last["is_transfer"]:
                    rec_last["norm_G_at_transfer"] = torch.norm(last_gc.get('G_accum', torch.zeros(1))).item()
                    last_gc['G_accum'] = torch.zeros_like(last_gc['G_accum'])
                last_log.append(rec_last)

                tag = ""
                if rec_first["is_transfer"]: tag += " [T-first]"
                if rec_last["is_transfer"]: tag += " [T-last]"
                pbar.set_postfix_str(
                    f"loss={loss.item():.4f} ||A1||={rec_first['norm_A']:.3f} "
                    f"T1={rec_first['num_transfers']} T2={rec_last['num_transfers']}{tag}")

                if rec_first["is_transfer"]:
                    tqdm.write(f"  [TRANSFER first] step={global_step}, T={rec_first['num_transfers']}, "
                               f"||dC||={rec_first['norm_dC_step']:.6f}")
                if rec_last["is_transfer"]:
                    tqdm.write(f"  [TRANSFER last]  step={global_step}, T={rec_last['num_transfers']}, "
                               f"||dC||={rec_last['norm_dC_step']:.6f}")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Deactivate hooks after DIAG_EPOCHS
        if ENABLE_DIAGNOSTIC and DIAG_EPOCHS > 0 and epoch == DIAG_EPOCHS:
            first_gc['active'] = False
            last_gc['active'] = False
            print(f"Diagnostic collection stopped after epoch {epoch}")

        train_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Evaluate
        eval_acc = evaluate_model(model, eval_loader)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch, "train/loss": train_loss,
            "eval/accuracy": eval_acc,
            "learning_rate": current_lr,
        })

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            save(model.state_dict(), WEIGHT_PATH)
        else:
            epochs_without_improvement += 1

        tqdm.write(
            f"Epoch {epoch}: Train Loss {train_loss:.4f} | "
            f"Acc {eval_acc:.2f}% | "
            f"Best Acc {best_acc:.2f}% | LR {current_lr:.2e} | "
            f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}"
        )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest Accuracy: {best_acc:.2f}% at epoch {best_epoch}")

    # =========================================================================
    # Save diagnostic outputs
    # =========================================================================
    first_transfers = [r["step"] for r in first_log if r["is_transfer"]]
    last_transfers = [r["step"] for r in last_log if r["is_transfer"]]

    if ENABLE_DIAGNOSTIC and first_log:
        first_transfers = [r["step"] for r in first_log if r["is_transfer"]]
        last_transfers = [r["step"] for r in last_log if r["is_transfer"]]
        diag_steps = len(first_log)

        print(f"\nDiagnostic Summary:")
        print(f"  Diag steps: {diag_steps}/{global_step}")
        print(f"  First tile transfers: {len(first_transfers)}")
        print(f"  Last tile transfers:  {len(last_transfers)}")

        stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"

        json_path = os.path.join(RESULTS, f"sst2_diagnostic_log_{stamp}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "config": {
                    "learning_rate": LEARNING_RATE, "transfer_lr": TRANSFER_LR,
                    "transfer_every": TRANSFER_EVERY, "lrtt_rank": LRTT_RANK,
                    "lora_alpha": LORA_ALPHA, "reinit_mode": REINIT_MODE,
                    "transfer_method": TRANSFER_METHOD, "optimizer": OPTIMIZER,
                    "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS,
                    "warmup_steps": WARMUP_STEPS, "min_lr_rate": MIN_LR_RATE,
                    "diag_epochs": DIAG_EPOCHS,
                },
                "best_accuracy": best_acc,
                "best_epoch": best_epoch,
                "total_steps": global_step, "diag_steps": diag_steps,
                "first_tile": {
                    "name": first_name,
                    "A_shape": list(first_tile.tile_a.get_weights()[0].shape),
                    "B_shape": list(first_tile.tile_b.get_weights()[0].shape),
                    "C_shape": list(first_tile.tile_c.get_weights()[0].shape),
                    "A_cell_indices": A_CELL_INDICES,
                    "B_cell_indices": B_CELL_INDICES,
                    "C_cell_indices": C_CELL_INDICES,
                    "total_transfers": len(first_transfers),
                    "transfer_steps": first_transfers,
                    "steps": first_log,
                },
                "last_tile": {
                    "name": last_name,
                    "A_shape": list(last_tile.tile_a.get_weights()[0].shape),
                    "B_shape": list(last_tile.tile_b.get_weights()[0].shape),
                    "C_shape": list(last_tile.tile_c.get_weights()[0].shape),
                    "A_cell_indices": A_CELL_INDICES,
                    "B_cell_indices": B_CELL_INDICES,
                    "C_cell_indices": C_CELL_INDICES,
                    "total_transfers": len(last_transfers),
                    "transfer_steps": last_transfers,
                    "steps": last_log,
                },
            }, f, indent=2)
        print(f"Saved: {json_path}")

        make_diagnostic_plots(first_log,
            os.path.join(RESULTS, f"sst2_diag_first_{stamp}.png"),
            tile_label=f"First tile ({first_name})")
        make_diagnostic_plots(last_log,
            os.path.join(RESULTS, f"sst2_diag_last_{stamp}.png"),
            tile_label=f"Last tile ({last_name})")

        steps_per_epoch = len(train_loader)
        diag_ep = DIAG_EPOCHS if DIAG_EPOCHS > 0 else N_EPOCHS
        for ep in range(1, diag_ep + 1):
            s_start = (ep - 1) * steps_per_epoch
            s_end = ep * steps_per_epoch
            epoch_first = first_log[s_start:s_end]
            epoch_last = last_log[s_start:s_end]
            if not epoch_first:
                break
            make_diagnostic_plots(epoch_first,
                os.path.join(RESULTS, f"sst2_diag_first_{stamp}_ep{ep}.png"),
                tile_label=f"First tile ({first_name}) — Epoch {ep}")
            make_diagnostic_plots(epoch_last,
                os.path.join(RESULTS, f"sst2_diag_last_{stamp}_ep{ep}.png"),
                tile_label=f"Last tile ({last_name}) — Epoch {ep}")

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
