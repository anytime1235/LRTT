# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for ALBERT + GLUE with TikiTaka v1.

Usage:
    python optuna_albert_glue_tiki.py --task sst2 --n-trials 50
    python optuna_albert_glue_tiki.py --task sst2 --visualize
    python optuna_albert_glue_tiki.py --task cola --n-trials 50 --optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --batch-size 64 --epochs 3 --lora-target attn

All flags:
    python optuna_albert_glue_tiki.py \
        --task <str>                # GLUE task: cola|sst2|mrpc|qqp|mnli|qnli|rte|stsb (default: sst2)
        --study-name <str>          # Study name (default: auto-generated)
        --n-trials <int>            # Number of Optuna trials (default: 50)
        --visualize                 # Visualize study results and exit
        --optimizer <str>           # AnalogSGD | AnalogAdam (default: AnalogAdam)
        --no-wd                     # Disable weight decay tuning (fix to 0)
        --no-momentum               # Disable momentum tuning (fix to 0, SGD only)
        --no-nesterov               # Disable nesterov tuning (fix to False, SGD only)
        --batch-size <int>          # Batch size (default: 64)
        --epochs <int>              # Number of epochs (default: 3)
        --warmup-ratio <float>      # LR warmup ratio (default: 0.1)
        --lora-target <str>         # Target: none | attn | ffn | all (default: attn)
        --head-layer <str>          # classifier: train | freeze (default: train)
        --convert-nontarget         # Convert non-target layers to analog
        --no-convert-nontarget      # Keep non-target layers as digital (nn.Linear, frozen) (default: off)

Inline flags (edit directly in script):
    TRAIN_SUBSET_SIZE = 0           # Training data subset (0 = full)
    EVAL_SUBSET_SIZE = 0            # Evaluation data subset (0 = full)

Note: ALBERT uses weight sharing across all encoder layers, so the number of
unique Linear layers converted to analog is much smaller than MobileBERT.
(e.g., attn = 4 unique layers shared across 12 transformer blocks)
Encoder Linear layer conversion:
  - Target layers -> TikiTaka v1 (TransferCompound: A + B tiles)
  - Non-target layers -> Single RPU (SoftBoundsDevice = B tile only) [default]
                      -> Digital (nn.Linear, frozen) [with --no-convert-nontarget]
"""

import os
import sys
import json
import argparse
import gc

import torch
from torch import nn, no_grad, manual_seed
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

import optuna
from optuna.trial import TrialState
from optuna.samplers import GridSampler, TPESampler
import matplotlib.pyplot as plt

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs import SingleRPUConfig, UnitCellRPUConfig, IOParameters, UpdateParameters
from aihwkit.simulator.configs.compounds import TransferCompound, ChoppedTransferCompound
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType


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

# Per-task training configs from Albert_setup.txt (2x epoch condition)
TASK_CONFIGS = {
    "cola":  {"batch_size": 16,  "max_seq_length": 128, "epochs": 10, "lr": 1.49e-3},
    "stsb":  {"batch_size": 16,  "max_seq_length": 128, "epochs": 20, "lr": 1.45e-3},
    "sst2":  {"batch_size": 32,  "max_seq_length": 128, "epochs": 10, "lr": 5.61e-4},
    "mnli":  {"batch_size": 128, "max_seq_length": 128, "epochs": 4,  "lr": 2.91e-3},
    "qnli":  {"batch_size": 32,  "max_seq_length": 128, "epochs": 11, "lr": 1e-3},
    "qqp":   {"batch_size": 128, "max_seq_length": 128, "epochs": 5,  "lr": 1e-3},
    "rte":   {"batch_size": 32,  "max_seq_length": 256, "epochs": 11, "lr": 6.78e-4},
    "mrpc":  {"batch_size": 32,  "max_seq_length": 128, "epochs": 14, "lr": 1.08e-3},
}


# =============================================================================
# Global Constants
# =============================================================================

DEFAULT_STUDY_NAME = "albert_glue_tiki_main"

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# GLUE task (set via --task)
TASK_NAME = "sst2"

# Paths
RESULTS = "/data/results/tikitakav1"
os.makedirs(RESULTS, exist_ok=True)

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "albert/albert-base-v2"
MAX_SEQ_LENGTH = 128  # Default; overridden per-task from TASK_CONFIGS

# Training defaults (overridden per-task from TASK_CONFIGS)
N_EPOCHS = 20
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64

# Scheduler
WARMUP_RATIO = 0.05  # GLUE: 5% of total steps

# Target options: which layers to convert to analog
# NOTE: ALBERT uses weight sharing — all 12 transformer blocks share the same
# parameters. The counts below are UNIQUE Linear layers, not per-block.
#   - Target layers -> TikiTaka v1 (TransferCompound: A + B tiles)
#   - Non-target encoder layers -> Single RPU (SoftBoundsDevice, same as B tile)
#   - classifier, embedding_hidden_mapping_in -> Digital (not converted)
NUM_HIDDEN_LAYERS = None  # None = use default (12); set to 1 for fast sweep
LORA_TARGET = "attn"  # default, can be set via --lora-target
HEAD_LAYER = "train"  # default, can be set via --head-layer (train | freeze)
CONVERT_NONTARGET = False  # default off (non-target encoder layers stay digital, frozen)
FREEZE_TARGET = False  # If True, target layers become SingleRPU frozen (not TikiTaka trainable)
BACKWARD_PERFECT = False  # If True, backward pass uses ideal FP32 (no DAC/ADC quantization)
LORA_TARGET_MODULES = {
    "none": [],            # No TikiTaka layers; all encoder layers -> Single RPU
    "attn": ["attention"], # Attention (query/key/value/dense) -> TikiTaka; FFN -> Single RPU
    "ffn":  ["ffn"],       # FFN (ffn/ffn_output) -> TikiTaka; Attention -> Single RPU
    "all":  None,          # All encoder layers -> TikiTaka (no Single RPU)
}

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

# Global config (set by argparse)
OPT_CONFIG = {
    'optimizer': 'AnalogAdam',
    'tune_wd': False,        # weight_decay = 0 (fixed)
    'tune_momentum': False,  # momentum = 0 (fixed)
    'tune_nesterov': False,  # nesterov = False (fixed)
    'fix_wd': None,          # None = use tune_wd logic; float = fixed wd
}

# Fixed LR (per-task, from TASK_CONFIGS)
FIXED_LR = 1e-3  # default; overridden per-task in main()

# Grid values (overridable via --fix-flr, --fix-tlr)
GRID_FLR = [1.0, 5.0]
GRID_TLR = None  # None = fixed at 1.0; list = grid values for transfer_lr

# TPE search ranges (used when --sampler tpe)
TPE_FLR_RANGE = (0.01, 1.0)      # log-uniform
TPE_TLR_RANGE = (0.01, 1.0)      # log-uniform
SAMPLER_TYPE = "tpe"              # "grid" or "tpe"


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

    # Non-target analog conversion
    if CONVERT_NONTARGET:
        suffix += "_convnt"

    # Freeze target (SingleRPU frozen instead of TikiTaka)
    if FREEZE_TARGET:
        suffix += "_frtgt"

    if BACKWARD_PERFECT:
        suffix += "_bwdperf"

    # Add lora target (always include for clarity)
    suffix += f"_{LORA_TARGET}"

    # Add head_layer if frozen (not default)
    if HEAD_LAYER == "freeze":
        suffix += "_headfreeze"

    # v1 (scale_transfer_lr=False) marker
    if not OPT_CONFIG.get('use_v2', True):
        suffix += "_v1"

    # transfer_lr sweep marker
    if OPT_CONFIG.get('tlr_sweep', None) is not None:
        suffix += "_tlr"

    # desired_bl grid marker
    if OPT_CONFIG.get('bl_grid', None) is not None:
        bls = OPT_CONFIG['bl_grid']
        suffix += "_bl" + "_".join(str(b) for b in bls)

    # num_hidden_layers marker
    if NUM_HIDDEN_LAYERS is not None:
        suffix += "_nl{}".format(NUM_HIDDEN_LAYERS)

    return suffix

os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# TikiTaka v1 Device Functions
# =============================================================================

def _create_a_device():
    """Create A tile: 6T1C LinearStepDevice (fast, noisy).

    Identical to LRTT's A/B tile config with lifetime=0 (no retention decay).
    """
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
    """Create B tile: noise-free SoftBoundsDevice (slow, accurate).

    Identical to LRTT's C tile config.
    """
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
    """Create TikiTaka v1 RPU configuration for analog layers.

    Uses ChoppedTransferCompound with no_buffer=True to be identical to
    TikiTaka v1 (TransferCompound with gamma=0), while enabling access to
    auto_scale for dynamic fast_lr normalisation.
    """
    a_device = _create_a_device()
    b_device = _create_b_device()

    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[a_device, b_device],
            transfer_every=transfer_every,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=OPT_CONFIG.get('scale_transfer_lr', use_v2),  # v1: False, v2: True (override with --no-scale-transfer-lr)
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=desired_bl,
                update_bl_management=False if use_v2 else True,
                update_management=False if use_v2 else True,
            ),
            # --- v1 vs v2 ---
            no_buffer=not use_v2,
            in_chop_prob=0.1 if use_v2 else 0.0,
            out_chop_prob=0.0,
            # --- auto_scale (new) ---
            auto_scale=auto_scale,
            auto_momentum=0.99,
        )
    )

    # Forward/Backward IO: set out_noise to 0.0 (aihwkit default is 0.06)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0

    # Backward perfect: skip all backward quantization (DAC/ADC/noise_management)
    if BACKWARD_PERFECT:
        rpu_config.backward.is_perfect = True

    # Mapping
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


def create_single_rpu_config():
    """Create Single RPU configuration for non-target frozen analog layers.

    Uses the same SoftBoundsDevice as TikiTaka's B tile (slow, accurate).
    Tile weights are frozen via noop update hook (see create_model).
    """
    b_device = _create_b_device()

    rpu_config = SingleRPUConfig(device=b_device)

    # IO settings: identical to TikiTaka config
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    if BACKWARD_PERFECT:
        rpu_config.backward.is_perfect = True

    # Mapping: frozen analog
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


# =============================================================================
# Model Functions
# =============================================================================

def list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def get_target_module_names(lora_target):
    """Get module name patterns for TikiTaka analog conversion based on lora_target.

    Returns list of substrings that identify which encoder layers get TikiTaka config.
    Non-target encoder layers get Single RPU (SoftBoundsDevice) config instead.

    ALBERT layer naming:
        attention: query, key, value, dense (output projection)
        FFN: ffn (intermediate), ffn_output (output)
        embedding projection: albert.encoder.embedding_hidden_mapping_in
    """
    if lora_target == "none":
        return []
    elif lora_target == "attn":
        return ["attention"]
    elif lora_target == "ffn":
        return ["ffn"]
    elif lora_target == "all":
        return None
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model(params):
    """Create ALBERT classification model with selective analog layers.

    Architecture:
        - Target encoder layers (--lora-target) -> TikiTaka v1 (TransferCompound)
        - Non-target encoder layers -> Single RPU (frozen) if CONVERT_NONTARGET
                                    -> Digital (nn.Linear, frozen) if not CONVERT_NONTARGET
        - classifier -> Digital TRAINABLE (based on HEAD_LAYER)
        - embedding_hidden_mapping_in -> Digital FROZEN
        - Embeddings -> Digital FROZEN

    ALBERT uses weight sharing: all transformer blocks share the same
    parameters, so the actual number of unique analog layers is small.
    """
    from aihwkit.nn import AnalogLinear

    num_labels = TASK_TO_NUM_LABELS[TASK_NAME]
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    if NUM_HIDDEN_LAYERS is not None:
        model_config.num_hidden_layers = NUM_HIDDEN_LAYERS
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    # Reinitialize classifier with FIXED seed for reproducibility
    if hasattr(model, 'classifier'):
        torch.manual_seed(SEED)
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
        print(f"  [FIX] Reinitialized classifier with FIXED seed={SEED}")

    # Get target patterns for TikiTaka conversion
    target_patterns = get_target_module_names(LORA_TARGET)

    # Always exclude from any analog conversion
    always_digital = ["classifier", "albert.encoder.embedding_hidden_mapping_in"]

    def is_tikitaka_target(layer_name):
        """Check if layer should be converted to TikiTaka Analog."""
        if any(d in layer_name for d in always_digital):
            return False
        if "encoder" not in layer_name:
            return False
        if target_patterns is None:
            return True
        return any(p in layer_name for p in target_patterns)

    all_linear_names = list_linear_layers(model)

    # Classify layers
    tikitaka_layers = [n for n in all_linear_names if is_tikitaka_target(n)]
    non_target_encoder_layers = [
        n for n in all_linear_names
        if n not in tikitaka_layers and "encoder" in n
        and not any(d in n for d in always_digital)
    ]

    # --- Pass 1: Convert target layers to TikiTaka or frozen SingleRPU ---
    tikitaka_count = 0
    frozen_target_count = 0
    if tikitaka_layers and not FREEZE_TARGET:
        # Normal: target layers become TikiTaka (trainable)
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
    elif tikitaka_layers and FREEZE_TARGET:
        # Freeze target: target layers become SingleRPU (frozen analog)
        single_config = create_single_rpu_config()
        target_exclude = [n for n in all_linear_names if n not in tikitaka_layers]
        model = convert_to_analog(model, single_config, exclude_modules=target_exclude)
        frozen_target_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))

    # --- Pass 2: Convert non-target encoder layers to frozen analog (Single RPU) ---
    single_rpu_count = 0
    if CONVERT_NONTARGET and non_target_encoder_layers:
        single_config = create_single_rpu_config()
        single_exclude = [n for n in all_linear_names if n not in non_target_encoder_layers]
        model = convert_to_analog(model, single_config, exclude_modules=single_exclude)
        single_rpu_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear)) - tikitaka_count - frozen_target_count

    # Freeze all SingleRPU tile weights via noop update hook
    frozen_analog_count = frozen_target_count + single_rpu_count
    if frozen_analog_count > 0:
        def _frozen_noop_update(x_input, d_input, *args, **kwargs):
            return None
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    if isinstance(tile.rpu_config, SingleRPUConfig):
                        tile.update = _frozen_noop_update

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  TikiTaka layers: {tikitaka_count}, Frozen target layers: {frozen_target_count}, "
          f"Frozen nontarget layers: {single_rpu_count}, "
          f"Total analog: {tikitaka_count + frozen_target_count + single_rpu_count}, Total params: {total_params:,}")

    # Set requires_grad: AnalogContext, classifier, LayerNorm are trainable
    # out_scaling follows learn_out_scaling in RPU config
    from aihwkit.optim.context import AnalogContext
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            param.requires_grad = True  # required for analog tile update
        elif "classifier" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
        elif "out_scaling" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable_after:,}")
    print(f"  Target: {LORA_TARGET} -> TikiTaka: {tikitaka_layers}, Single RPU: {non_target_encoder_layers}")

    return model.to(DEVICE)


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

    return train_loader, eval_loader


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(model, eval_loader):
    """Evaluate GLUE model. Returns (metric_value, avg_loss)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    is_regression = TASK_NAME == "stsb"
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    with no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            if is_regression:
                labels = labels.float()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze() if is_regression else outputs.logits
            loss = criterion(logits, labels)

            if is_regression:
                all_preds.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item() * labels.size(0)

    model.train()
    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

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
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        metric_value = correct / n_samples if n_samples > 0 else 0.0

    return metric_value, avg_loss


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Adaptive early stopping with warmup protection and overfitting detection.

    Criteria for stopping (any one triggers):
      1. Patience exhausted: no improvement >= min_delta for `patience` epochs
      2. Overfitting: eval_loss increases for `overfit_window` consecutive epochs
         while train_loss decreases

    Warmup protection: no stopping in first `warmup_epochs` epochs.

    Patience and warmup scale with total epochs:
      patience = max(3, N_EPOCHS // 5)
      warmup_epochs = ceil(N_EPOCHS * warmup_ratio)  — matches LR scheduler warmup
    """

    def __init__(self, n_epochs, warmup_ratio=0.05, min_delta=0.001, patience=None, warmup_epochs=None):
        import math
        self.patience = patience if patience is not None else max(3, n_epochs // 5)
        self.warmup_epochs = warmup_epochs if warmup_epochs is not None else max(1, math.ceil(n_epochs * warmup_ratio))
        self.overfit_window = 3
        self.min_delta = min_delta

        self.best_metric = -float('inf')
        self.epochs_no_improve = 0
        self.eval_loss_history = []
        self.train_loss_history = []
        self.stop_reason = None

    def step(self, epoch, eval_metric, eval_loss, train_loss):
        """Returns True if training should stop."""
        self.eval_loss_history.append(eval_loss)
        self.train_loss_history.append(train_loss)

        # Check improvement
        improved = False
        if eval_metric > self.best_metric + self.min_delta:
            self.best_metric = eval_metric
            self.epochs_no_improve = 0
            improved = True
        else:
            self.epochs_no_improve += 1

        # Warmup protection
        if epoch <= self.warmup_epochs:
            return False

        # 1) Patience check
        if self.epochs_no_improve >= self.patience:
            self.stop_reason = (f"patience exhausted ({self.epochs_no_improve} epochs "
                                f"without >{self.min_delta} improvement)")
            return True

        # 2) Overfitting check: eval_loss increasing + train_loss decreasing
        n = self.overfit_window
        if len(self.eval_loss_history) >= n + 1:
            recent_eval = self.eval_loss_history[-n:]
            recent_train = self.train_loss_history[-n:]
            prev_eval = self.eval_loss_history[-(n + 1)]
            prev_train = self.train_loss_history[-(n + 1)]

            eval_increasing = all(
                recent_eval[i] > recent_eval[i - 1] if i > 0 else recent_eval[0] > prev_eval
                for i in range(n)
            )
            train_decreasing = all(
                recent_train[i] < recent_train[i - 1] if i > 0 else recent_train[0] < prev_train
                for i in range(n)
            )
            if eval_increasing and train_decreasing:
                self.stop_reason = (f"overfitting detected (eval_loss increasing + "
                                    f"train_loss decreasing for {n} consecutive epochs)")
                return True

        return False

    def status_str(self):
        return f"No imp: {self.epochs_no_improve}/{self.patience}"


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

def objective(trial, train_loader, eval_loader):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metric_name = TASK_TO_METRIC[TASK_NAME]

    # If no TikiTaka layers (none or freeze-target), use fixed LR and skip analog HP
    if LORA_TARGET == "none" or FREEZE_TARGET:
        learning_rate = FIXED_LR
        desired_bl = OPT_CONFIG.get('desired_bl', 1)
        transfer_every = 1
        fast_lr = 1.0
        transfer_lr = 1.0
        min_lr_rate = 0.5
    else:
        # Learning rate: fixed or sweep around per-task fixed LR [1/10, mult*x]
        if OPT_CONFIG.get('fix_lr', False):
            learning_rate = FIXED_LR
        else:
            _lr_upper_mult = OPT_CONFIG.get('lr_upper_mult', 10)
            learning_rate = trial.suggest_float('learning_rate', FIXED_LR / 10, FIXED_LR * _lr_upper_mult, log=True)
        # desired_bl: grid, sweep range, or fixed
        _bl_grid = OPT_CONFIG.get('bl_grid', None)
        _bl_range = OPT_CONFIG.get('bl_sweep', None)
        if _bl_grid is not None:
            desired_bl = trial.suggest_categorical('desired_bl', _bl_grid)
        elif _bl_range is not None:
            desired_bl = trial.suggest_int('desired_bl', _bl_range[0], _bl_range[1])
        else:
            desired_bl = OPT_CONFIG.get('desired_bl', 1)

        # transfer_every: fixed at 1 (uim=True → every mini-batch)
        transfer_every = OPT_CONFIG.get('transfer_every_override', 1)

        # fast_lr: fixed at 1.0
        fast_lr = 1.0

        # transfer_lr: log sweep or fixed at 1.0
        _tlr_sweep = OPT_CONFIG.get('tlr_sweep', None)
        if _tlr_sweep is not None:
            transfer_lr = trial.suggest_float('transfer_lr', _tlr_sweep[0], _tlr_sweep[1], log=True)
        else:
            transfer_lr = 1.0
        min_lr_rate = 0.5

    # weight_decay: fixed value, tune, or 0
    if OPT_CONFIG['fix_wd'] is not None:
        weight_decay = OPT_CONFIG['fix_wd']
    elif OPT_CONFIG['tune_wd']:
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
    else:
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

    # optimizer: always use config value
    optimizer_name = OPT_CONFIG['optimizer']

    params = {
        "transfer_every": transfer_every,
        "transfer_lr": transfer_lr,
        "fast_lr": fast_lr,
        "desired_bl": desired_bl,
    }

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting ({TASK_NAME}, metric={metric_name})")
    print(f"{'='*70}")
    print(f"  transfer_every={transfer_every}, transfer_lr={transfer_lr:.4e}, fast_lr={fast_lr:.4e}")
    print(f"  lr={learning_rate:.2e}, wd={weight_decay:.2e}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, optimizer={optimizer_name}")
    print(f"  min_lr_rate={min_lr_rate:.4f}, warmup_ratio={WARMUP_RATIO}")
    print(f"{'='*70}")

    is_regression = TASK_NAME == "stsb"
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    model = None
    try:
        from aihwkit.nn import AnalogLinear
        set_seed(SEED)

        model = create_model(params)

        # Choose optimizer based on whether analog layers exist
        has_analog = any(isinstance(m, AnalogLinear) for m in model.modules())
        if has_analog:
            if optimizer_name == "AnalogSGD":
                optimizer = AnalogSGD(
                    model.parameters(), lr=learning_rate,
                    weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                )
            else:
                optimizer = AnalogAdam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                )
            optimizer.regroup_param_groups()
        else:
            # No analog layers (e.g. --lora-target none --no-convert-nontarget)
            if optimizer_name == "AnalogSGD":
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=learning_rate,
                    weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                )

        num_training_steps = len(train_loader) * N_EPOCHS
        warmup_steps = int(num_training_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
        )

        early_stop = EarlyStopping(N_EPOCHS, warmup_ratio=WARMUP_RATIO, min_delta=0.001,
                                       patience=3, warmup_epochs=3)
        print(f"  EarlyStopping: patience={early_stop.patience}, "
              f"warmup_epochs={early_stop.warmup_epochs}, "
              f"overfit_window={early_stop.overfit_window}, "
              f"min_delta={early_stop.min_delta}")

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
            for batch in pbar:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                if is_regression:
                    labels = labels.float()

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze() if is_regression else outputs.logits
                loss = criterion(logits, labels)

                loss.backward()
                # Digital-only grad clipping (AnalogContext .grad does not affect tile update)
                from aihwkit.optim.context import AnalogContext as _AC
                _digital_params = [p for p in model.parameters()
                                   if not isinstance(p, _AC) and p.grad is not None]
                if _digital_params:
                    torch.nn.utils.clip_grad_norm_(_digital_params, max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            eval_metric, eval_loss = evaluate_model(model, eval_loader)

            improved = " *" if eval_metric > early_stop.best_metric + early_stop.min_delta else ""
            should_stop = early_stop.step(epoch, eval_metric, eval_loss, train_loss)

            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"[Trial {trial.number}] Epoch {epoch:3d} | "
                  f"{metric_name}: {eval_metric:.4f} | Best: {early_stop.best_metric:.4f} | "
                  f"Loss: {train_loss:.4f} | LR: {current_lr:.2e} | "
                  f"{early_stop.status_str()}{improved}")

            trial.report(eval_metric, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

            if should_stop:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}: "
                           f"{early_stop.stop_reason}")
                break

            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        print(f"\n[Trial {trial.number}] Finished - Best {metric_name}: {early_stop.best_metric:.4f}")
        print(f"{'='*70}\n")
        return early_stop.best_metric

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
    global BATCH_SIZE, N_EPOCHS, MAX_SEQ_LENGTH, WARMUP_RATIO, LORA_TARGET, HEAD_LAYER, TASK_NAME, CONVERT_NONTARGET, FREEZE_TARGET, BACKWARD_PERFECT, FIXED_LR

    parser = argparse.ArgumentParser(description="Optuna sweep for ALBERT GLUE TikiTaka v1")
    parser.add_argument('--task', type=str, default=TASK_NAME,
                        choices=GLUE_TASKS,
                        help=f'GLUE task (default: {TASK_NAME})')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: auto-generated based on config)')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--optimizer', type=str, default='AnalogAdam',
                        choices=['AnalogSGD', 'AnalogAdam'],
                        help='Optimizer type (default: AnalogAdam)')
    parser.add_argument('--no-wd', action='store_true',
                        help='Disable weight decay tuning (fix to 0)')
    parser.add_argument('--no-momentum', action='store_true', default=True,
                        help='Disable momentum tuning (fix to 0, SGD only)')
    parser.add_argument('--no-nesterov', action='store_true', default=True,
                        help='Disable nesterov tuning (fix to False, SGD only)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: per-task from TASK_CONFIGS)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: per-task from TASK_CONFIGS)')
    parser.add_argument('--warmup-ratio', type=float, default=WARMUP_RATIO,
                        help=f'LR warmup ratio (default: {WARMUP_RATIO})')
    parser.add_argument('--lr', type=float, default=None,
                        help='Fixed learning rate (default: per-task from TASK_CONFIGS)')
    parser.add_argument('--fix-wd', type=float, default=None,
                        help='Fix weight_decay value. e.g. --fix-wd 0.01')
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=['none', 'attn', 'ffn', 'all'],
                        help='Target: none, attn, ffn, all (default: attn)')
    parser.add_argument('--head-layer', type=str, default=HEAD_LAYER,
                        choices=['train', 'freeze'],
                        help='classifier layer: train or freeze (default: train)')
    parser.add_argument('--convert-nontarget', action='store_true', default=False,
                        help='Convert non-target layers to analog (SingleRPU+SoftBounds, frozen)')
    parser.add_argument('--no-convert-nontarget', dest='convert_nontarget', action='store_false',
                        help='Keep non-target layers as digital (nn.Linear, frozen) [default]')
    parser.add_argument('--freeze-target', action='store_true', default=False,
                        help='Convert target layers to SingleRPU frozen (not TikiTaka trainable)')
    parser.add_argument('--no-scale-transfer-lr', action='store_true', default=True,
                        help='Force scale_transfer_lr=False (default: True)')
    parser.add_argument('--scale-transfer-lr', action='store_false', dest='no_scale_transfer_lr',
                        help='Enable scale_transfer_lr (v2 default behavior)')
    parser.add_argument('--lr-upper-mult', type=float, default=10.0,
                        help='LR upper bound multiplier (default: 10). lr sweep range = [fixed_lr/10, fixed_lr*mult]')
    parser.add_argument('--fix-lr', action='store_true', default=False,
                        help='Use per-task fixed LR without sweep')
    parser.add_argument('--backward-perfect', action='store_true', default=False,
                        help='Use perfect backward pass (no DAC/ADC quantization on gradients)')
    parser.add_argument('--fix-flr', type=float, nargs='+', default=None,
                        help='Fix fast_lr grid to these values (e.g. --fix-flr 1.0)')
    parser.add_argument('--fix-tlr', type=float, nargs='+', default=None,
                        help='Grid transfer_lr values (e.g. --fix-tlr 0.1 1.0). If not set, transfer_lr=1.0 fixed.')
    parser.add_argument('--sampler', type=str, default='grid', choices=['grid', 'tpe'],
                        help='Sampler type: grid (GridSampler) or tpe (TPESampler, default: grid)')
    parser.add_argument('--tpe-flr-range', type=float, nargs=2, default=None,
                        help='TPE fast_lr range (e.g. --tpe-flr-range 0.5 10.0)')
    parser.add_argument('--tpe-tlr-range', type=float, nargs=2, default=None,
                        help='TPE transfer_lr range (e.g. --tpe-tlr-range 0.01 1.0). If not set, transfer_lr=1.0 fixed.')
    parser.add_argument('--enqueue', type=float, nargs='+', default=None,
                        metavar='VAL',
                        help='Enqueue first trial: TE FLR [TLR] (e.g. --enqueue 1 0.035 0.5)')
    parser.add_argument('--auto-scale', action='store_true', default=False,
                        help='Enable auto_scale: dynamically normalise fast_lr by gradient magnitude')
    parser.add_argument('--transfer-every', type=int, default=None,
                        help='Override transfer_every (default: 1). Large value = effectively no transfer')
    parser.add_argument('--desired-bl', type=int, default=1,
                        help='Transfer update desired_bl (default: 1)')
    parser.add_argument('--bl-sweep', type=int, nargs=2, default=None,
                        help='Sweep desired_bl range [min max] (e.g. --bl-sweep 1 31)')
    parser.add_argument('--bl-grid', type=int, nargs='+', default=None,
                        help='Grid values for desired_bl (e.g. --bl-grid 31 60)')
    parser.add_argument('--tlr-sweep', type=float, nargs=2, default=None,
                        help='Log sweep transfer_lr range [min max] (e.g. --tlr-sweep 0.01 1.0)')
    parser.add_argument('--use-v2', action='store_true', default=True,
                        help='Use TikiTaka v2 (ChoppedTransfer with buffer+chopper, bl=1)')
    parser.add_argument('--no-v2', action='store_true', default=False,
                        help='Use TikiTaka v1 (scale_transfer_lr=False)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of hidden layers (default: 12). Set to 1 for fast sweep.')
    args = parser.parse_args()

    # Update global config
    TASK_NAME = args.task

    # Apply per-task config from Albert_setup.txt (2x epoch condition)
    task_cfg = TASK_CONFIGS[TASK_NAME]
    MAX_SEQ_LENGTH = task_cfg["max_seq_length"]
    BATCH_SIZE = args.batch_size if args.batch_size is not None else task_cfg["batch_size"]
    N_EPOCHS = args.epochs if args.epochs is not None else task_cfg["epochs"]
    WARMUP_RATIO = args.warmup_ratio
    LORA_TARGET = args.lora_target
    HEAD_LAYER = args.head_layer
    CONVERT_NONTARGET = args.convert_nontarget
    FREEZE_TARGET = args.freeze_target
    BACKWARD_PERFECT = args.backward_perfect
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov
    OPT_CONFIG['fix_wd'] = args.fix_wd
    OPT_CONFIG['auto_scale'] = args.auto_scale
    if args.transfer_every is not None:
        OPT_CONFIG['transfer_every_override'] = args.transfer_every
    OPT_CONFIG['desired_bl'] = args.desired_bl
    if args.bl_sweep is not None:
        OPT_CONFIG['bl_sweep'] = args.bl_sweep
    if args.bl_grid is not None:
        OPT_CONFIG['bl_grid'] = args.bl_grid
    if args.tlr_sweep is not None:
        OPT_CONFIG['tlr_sweep'] = args.tlr_sweep
    OPT_CONFIG['use_v2'] = args.use_v2 and not args.no_v2
    if args.no_scale_transfer_lr:
        OPT_CONFIG['scale_transfer_lr'] = False
    OPT_CONFIG['lr_upper_mult'] = args.lr_upper_mult
    OPT_CONFIG['fix_lr'] = args.fix_lr

    global NUM_HIDDEN_LAYERS
    NUM_HIDDEN_LAYERS = args.num_layers

    # Fixed LR: --lr overrides per-task default
    FIXED_LR = args.lr if args.lr is not None else task_cfg["lr"]

    print(f"Per-task config: {TASK_NAME} -> BSZ={BATCH_SIZE}, epochs={N_EPOCHS}, "
          f"max_seq={MAX_SEQ_LENGTH}, lr={FIXED_LR} (fixed), padding=dynamic")

    # Auto-generate study name based on config (includes task and batch size)
    study_name = args.study_name or f"albert_glue_tiki_{TASK_NAME}_bs{BATCH_SIZE}_{get_study_name_suffix()}"

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

    global GRID_FLR, GRID_TLR, SAMPLER_TYPE, TPE_FLR_RANGE, TPE_TLR_RANGE
    SAMPLER_TYPE = args.sampler

    if args.tpe_flr_range:
        TPE_FLR_RANGE = tuple(args.tpe_flr_range)
    if args.tpe_tlr_range:
        TPE_TLR_RANGE = tuple(args.tpe_tlr_range)

    # Grid transfer_lr
    if args.fix_tlr:
        GRID_TLR = args.fix_tlr

    print(f"  lr={FIXED_LR} (fixed, per-task)")

    print(f"  transfer_every=1 (fixed, uim=True, every mini-batch)")

    if SAMPLER_TYPE == "tpe":
        sampler = optuna.samplers.TPESampler(seed=SEED)
        print(f"Sampler: TPE | flr={TPE_FLR_RANGE} | tlr={TPE_TLR_RANGE}")
    else:
        GRID_FLR = args.fix_flr if args.fix_flr else [1.0, 5.0]
        grid_search_space = {
            "fast_lr": GRID_FLR,
        }
        if GRID_TLR is not None:
            grid_search_space["transfer_lr"] = GRID_TLR
        sampler = GridSampler(grid_search_space)
        print(f"Sampler: Grid | flr={GRID_FLR} | tlr={GRID_TLR}")

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.PercentilePruner(
            percentile=50.0,      # keep trials in top 50%
            n_startup_trials=5,   # start pruning after 5 completed trials
            n_warmup_steps=3,     # don't prune until after epoch 3 (match warmup)
        ),
        load_if_exists=True,
    )

    # Enqueue seed trial (best params as starting point)
    if args.enqueue and len(study.trials) == 0:
        eq_params = {
            'transfer_every': int(args.enqueue[0]),
            'fast_lr': args.enqueue[1],
        }
        if len(args.enqueue) >= 3 and TPE_TLR_RANGE is not None:
            eq_params['transfer_lr'] = args.enqueue[2]
        study.enqueue_trial(eq_params)
        print(f"Enqueued seed trial: {eq_params}")

    print(f"\nStudy: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

    # Run trials with OOM recovery via process restart
    target_total = len(study.trials) + args.n_trials

    try:
        study.optimize(
            lambda trial: objective(trial, train_loader, eval_loader),
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
