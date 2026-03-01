# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for MobileBERT + GLUE with TikiTaka v1.

Usage:
    python optuna_mobilebert_glue_tiki.py --task sst2 --n-trials 50
    python optuna_mobilebert_glue_tiki.py --task sst2 --visualize
    python optuna_mobilebert_glue_tiki.py --task cola --n-trials 50 --optimizer AnalogSGD --no-wd --no-momentum --no-nesterov

All flags:
    python optuna_mobilebert_glue_tiki.py \
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
        --lora-target <str>         # Target: none|qonly|konly|vonly|qkv|ffn|all (default: qkv)
        --head-layer <str>          # classifier: train | freeze (default: train)

Inline flags (edit directly in script):
    TRAIN_SUBSET_SIZE = 0           # Training data subset (0 = full)
    EVAL_SUBSET_SIZE = 0            # Evaluation data subset (0 = full)
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
from optuna.samplers import GridSampler
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
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, IdealDevice
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

# Per-task settings from Albert_setup.txt (exp column)
TASK_TO_BSZ = {
    "cola": 16, "stsb": 16, "sst2": 32, "mnli": 128,
    "qnli": 32, "qqp": 128, "rte": 32, "mrpc": 32,
}
TASK_TO_MAXSEQ = {
    "cola": 128, "stsb": 128, "sst2": 128, "mnli": 128,
    "qnli": 128, "qqp": 128, "rte": 256, "mrpc": 128,
}
TASK_TO_EPOCHS = {
    "cola": 10, "stsb": 10, "sst2": 10, "mnli": 4,
    "qnli": 11, "qqp": 5, "rte": 11, "mrpc": 7,
}
# Early stopping patience per task (~1/3 of total epochs, min 2)
TASK_TO_ES_PATIENCE = {
    "rte": 3, "mrpc": 2, "stsb": 3, "cola": 3,
    "sst2": 3, "qnli": 3, "qqp": 2, "mnli": 2,
}
# Per-task fixed learning rate
TASK_TO_LR = {
    "rte": 2e-3, "mrpc": 6e-3, "stsb": 2e-3, "cola": 4e-3,
    "sst2": 1e-3, "qnli": 1e-3, "qqp": 5e-3, "mnli": 3e-3,
}


# =============================================================================
# ConfigAwareBoTorchSampler with Periodic Exploration
# =============================================================================

## ConfigAwareBoTorchSampler removed — using GridSampler for grid search


# =============================================================================
# Global Constants
# =============================================================================

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
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128  # default, overridden per task

# Training defaults (overridden per task from TASK_TO_* dicts)
N_EPOCHS = 5
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 3

# Scheduler
WARMUP_RATIO = 0.05  # fixed warmup ratio

# Target options
LORA_TARGET = "qkv"
HEAD_LAYER = "train"  # "train" or "freeze" for classifier layer
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
# Each tile's d is independently clipped/floored to [min_norm, max_norm]
# - max_norm: prevents inter-layer gradient explosion (spike suppression)
# - min_norm: ensures d > dw_min threshold for meaningful pulse generation
CLIP_ANALOG_GRAD = False
ANALOG_TILE_MAX_NORM = 1.0
ANALOG_TILE_MIN_NORM = 0.1

# Global config (set by argparse)
OPT_CONFIG = {
    'optimizer': 'AnalogSGD',
    'tune_wd': False,
    'tune_momentum': False,
    'tune_nesterov': False,
}

# Grid values (overridable via --fix-lr, --fix-te)
GRID_LR = [0.001]
GRID_TE = [1000]

# TPE search ranges (default sampler)
TPE_LR_RANGE = (0.001, 0.1)         # log-uniform
TPE_FLR_RANGE = (1.0, 1.0)          # fixed at 1.0 (default)
TPE_TLR_RANGE = (1.0, 1000.0)       # log-uniform, scale_transfer_lr=True (eff = tlr * opt_lr)
SAMPLER_TYPE = "tpe"                 # "grid" or "tpe"

# Approximate GLUE train set sizes (for te_max calculation)
TASK_TO_TRAIN_SIZE = {
    "rte": 2490, "mrpc": 3668, "stsb": 5749, "cola": 8551,
    "sst2": 67349, "qnli": 104743, "qqp": 363846, "mnli": 392702,
}

# Number of columns per attention tile (MobileBERT Q/K/V/W_O = Linear(128,128))
ATTN_TILE_COLUMNS = 128


def compute_te_bounds(task_name):
    """Compute per-task transfer_every bounds (uim=False, mat-vec units).

    te_max: all columns transferred >= 1 time per epoch (upper bound, max_seq_length).
    te_min: te_max // 100 (sweep lower bound).
    """
    bsz = TASK_TO_BSZ[task_name]
    max_seq = TASK_TO_MAXSEQ[task_name]
    train_size = TASK_TO_TRAIN_SIZE[task_name]
    steps_per_epoch = -(-train_size // bsz)  # ceil division
    mat_vecs_per_epoch = steps_per_epoch * bsz * max_seq
    te_max = mat_vecs_per_epoch // ATTN_TILE_COLUMNS
    te_min = max(1, te_max // 100)
    return te_min, te_max


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

    if HEAD_LAYER == "freeze":
        suffix += "_headfreeze"

    if OPT_CONFIG.get('learn_out_scaling', False):
        suffix += "_los"

    if OPT_CONFIG.get('nontarget_digital', False):
        suffix += "_ntdig"

    if OPT_CONFIG.get('backward_perfect', False):
        suffix += "_bwdperf"

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
            # --- match TransferCompound defaults ---
            units_in_mbatch=OPT_CONFIG.get('units_in_mbatch', True),  # Chopped default=False, Transfer default=True
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=OPT_CONFIG.get('scale_transfer_lr', use_v2),  # v1: False (absolute), v2: True (scale with optimizer lr)
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
            no_buffer=not use_v2,       # v1: True (no buffer), v2: False (buffer enabled)
            in_chop_prob=0.1 if use_v2 else 0.0,   # v2: chopper enabled
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
    if OPT_CONFIG.get('backward_perfect', False):
        rpu_config.backward.is_perfect = True

    # Mapping
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = OPT_CONFIG.get('learn_out_scaling', False)
    rpu_config.mapping.out_scaling_columnwise = OPT_CONFIG.get('learn_out_scaling', False)

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
    if OPT_CONFIG.get('backward_perfect', False):
        rpu_config.backward.is_perfect = True

    # Mapping: frozen analog
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = OPT_CONFIG.get('learn_out_scaling', False)
    rpu_config.mapping.out_scaling_columnwise = OPT_CONFIG.get('learn_out_scaling', False)

    return rpu_config


def create_ideal_rpu_config():
    """Create Ideal RPU configuration for target analog layers (trainable).

    Uses IdealDevice — floating-point weight updates (no pulsed noise),
    forward/backward can still have analog non-idealities if configured.
    """
    rpu_config = SingleRPUConfig(device=IdealDevice())

    # IO settings: identical to TikiTaka config
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    if OPT_CONFIG.get('backward_perfect', False):
        rpu_config.backward.is_perfect = True

    # Mapping
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
    """Classify MobileBERT encoder Linear layer.

    MobileBERT encoder layer structure (per block, x24):
        attention:  attention.self.query/key/value, attention.output.dense (W_O)
        ffn:        intermediate.dense, output.dense, ffn.{0,1,2}.intermediate.dense,
                    ffn.{0,1,2}.output.dense
        bottleneck: bottleneck.input.dense, output.bottleneck.dense,
                    bottleneck.attention.dense
    """
    if 'bottleneck' in layer_name:
        return 'bottleneck'
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
        return ["ffn (intermediate, output, ffn.*)"]
    elif lora_target == "all":
        return ["attention + ffn"]
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model(params):
    """Create MobileBERT classification model with selective TikiTaka v1 analog layers.

    Architecture (matching mobilebert_layer.txt):
        QKV mode:  96 LRTT (attention), 264 NT Analog (ffn+bottleneck), 2 Digital
        FFN mode: 192 LRTT (ffn),        168 NT Analog (attention+bottleneck), 2 Digital
        ALL mode: 288 LRTT (attn+ffn),    72 NT Analog (bottleneck), 2 Digital
        Digital = classifier + embedding_transformation (always)

    Classifier is reinitialized with FIXED seed=42 for reproducibility.
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

    # Always digital (never analog): classifier + embedding_transformation
    always_digital = ["classifier", "embedding_transformation"]

    def is_tikitaka_target(layer_name):
        """Check if encoder layer should be TikiTaka (trainable analog)."""
        if any(d in layer_name for d in always_digital):
            return False
        if "encoder" not in layer_name:
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

    # --- Pass 1: Convert target layers to TikiTaka or Ideal ---
    tikitaka_count = 0
    if tikitaka_layers and LORA_TARGET != "none":
        if OPT_CONFIG.get('target_ideal', False):
            target_config = create_ideal_rpu_config()
            print(f"  [IDEAL] Using IdealDevice for {len(tikitaka_layers)} target layers (trainable)")
        else:
            target_config = create_tikitaka_config(
                transfer_every=int(params["transfer_every"]),
                transfer_lr=params["transfer_lr"],
                fast_lr=params["fast_lr"],
                auto_scale=OPT_CONFIG.get('auto_scale', False),
                desired_bl=int(params["desired_bl"]),
                use_v2=OPT_CONFIG.get('use_v2', False),
            )
        tiki_exclude = [n for n in all_linear_names if n not in tikitaka_layers]
        model = convert_to_analog(model, target_config, exclude_modules=tiki_exclude)
        tikitaka_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))

    # --- Pass 2: Convert non-target encoder layers to frozen analog (Single RPU) ---
    # OR keep them digital if --nontarget-digital is set
    single_rpu_count = 0
    if non_target_encoder_layers and not OPT_CONFIG.get('nontarget_digital', False):
        single_config = create_single_rpu_config()
        single_exclude = [n for n in all_linear_names if n not in non_target_encoder_layers]
        model = convert_to_analog(model, single_config, exclude_modules=single_exclude)
        single_rpu_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear)) - tikitaka_count

        # Freeze Single RPU tile weights via noop update hook
        def _frozen_noop_update(x_input, d_input, *args, **kwargs):
            return None
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    if isinstance(tile.rpu_config, SingleRPUConfig):
                        tile.update = _frozen_noop_update
    elif non_target_encoder_layers and OPT_CONFIG.get('nontarget_digital', False):
        print(f"  [DIGITAL] Keeping {len(non_target_encoder_layers)} non-target layers as digital (frozen)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  TikiTaka: {tikitaka_count}, NT Analog: {single_rpu_count}, "
          f"Total analog: {tikitaka_count + single_rpu_count}, Total params: {total_params:,}")

    # Set requires_grad
    from aihwkit.optim.context import AnalogContext
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            param.requires_grad = True  # required for analog tile update
        elif "classifier" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = False  # MobileBERT uses NoNorm (no normalization) - must freeze
        elif "out_scaling" in name:
            param.requires_grad = OPT_CONFIG.get('learn_out_scaling', False)
        else:
            param.requires_grad = False

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable_after:,}")
    print(f"  Target: {LORA_TARGET} -> {get_target_module_names(LORA_TARGET)}")

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

    tokenized = raw_datasets.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    # Remove string columns that collator can't handle
    keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    for split in tokenized:
        remove_cols = [c for c in tokenized[split].column_names if c not in keep_cols]
        tokenized[split] = tokenized[split].remove_columns(remove_cols)

    collator = DataCollatorWithPadding(tokenizer)

    # Training set
    train_dataset = tokenized["train"]
    if TRAIN_SUBSET_SIZE > 0:
        train_dataset = train_dataset.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(train_dataset)))
        )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator,
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
        collate_fn=collator,
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
                labels = labels.float() / 5.0  # normalize STS-B labels to [0, 1]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze() if is_regression else outputs.logits
            loss = criterion(logits, labels)

            if is_regression:
                # scale back to original [0, 5] for metric computation
                all_preds.extend((logits * 5.0).cpu().numpy())
                all_labels.extend((labels * 5.0).cpu().numpy())
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

    te_min, te_max = compute_te_bounds(TASK_NAME)

    # Learning rate: fixed per-task base LR (or override)
    _base_lr = TASK_TO_LR.get(TASK_NAME, 0.001)
    _lr_override = OPT_CONFIG.get('lr_override', None)
    _lr_range = OPT_CONFIG.get('lr_range', None)

    _target_ideal = OPT_CONFIG.get('target_ideal', False)

    # When lr_range is set: sweep learning_rate in [min, max]
    if _lr_range is not None:
        learning_rate = trial.suggest_float('learning_rate', _lr_range[0], _lr_range[1], log=True)
    # When target_ideal + classifier_lr set + NO lr_override: sweep single lr for both
    elif _target_ideal and OPT_CONFIG.get('classifier_lr', None) is not None and _lr_override is None:
        learning_rate = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
        OPT_CONFIG['_trial_classifier_lr'] = learning_rate  # same lr for analog and classifier
    elif _lr_override is not None:
        learning_rate = _lr_override
    else:
        learning_rate = _base_lr

    # fast_lr, transfer_lr: sweep or fixed (skip for ideal target)
    if _target_ideal:
        fast_lr = 1.0
        transfer_lr = 1.0
    elif TPE_FLR_RANGE[0] == TPE_FLR_RANGE[1]:
        fast_lr = TPE_FLR_RANGE[0]
    else:
        fast_lr = trial.suggest_float('fast_lr', TPE_FLR_RANGE[0], TPE_FLR_RANGE[1], log=True)
    if not _target_ideal:
        if TPE_TLR_RANGE[0] == TPE_TLR_RANGE[1]:
            transfer_lr = TPE_TLR_RANGE[0]
        else:
            _scale_tlr = OPT_CONFIG.get('scale_transfer_lr', OPT_CONFIG.get('use_v2', False))
            if _scale_tlr:
                # effective_tlr = transfer_lr * adam_lr ≤ 1.0 → max = 1/adam_lr
                _tlr_upper = 1.0 / learning_rate
            else:
                _tlr_upper = TPE_TLR_RANGE[1]
            transfer_lr = trial.suggest_float('transfer_lr', TPE_TLR_RANGE[0], _tlr_upper, log=True)

    # desired_bl: grid, sweep, or fixed (skip for ideal target)
    if _target_ideal:
        desired_bl = OPT_CONFIG.get('desired_bl', 1)
    else:
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

    min_lr_rate = 0.5  # decay to 50% of peak lr

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
    }

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting ({TASK_NAME}, metric={metric_name})")
    print(f"{'='*70}")
    print(f"  transfer_every={transfer_every}, transfer_lr={transfer_lr:.4e}, fast_lr={fast_lr:.4e}, desired_bl={desired_bl}")
    print(f"  lr={learning_rate:.2e}, wd={weight_decay:.2e}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, optimizer={optimizer_name}")
    print(f"  min_lr_rate={min_lr_rate:.4f}")
    print(f"{'='*70}")

    is_regression = TASK_NAME == "stsb"
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    model = None
    try:
        set_seed(SEED)

        model = create_model(params)

        if LORA_TARGET == "none":
            if optimizer_name == "AnalogSGD":
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=learning_rate,
                    weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                )
        else:
            _clf_lr = OPT_CONFIG.get('_trial_classifier_lr', OPT_CONFIG.get('classifier_lr', None))
            if _clf_lr is not None:
                # Separate param groups: classifier (digital) vs analog
                from aihwkit.optim.context import AnalogContext
                clf_params = []
                other_digital_params = []
                analog_params = []
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if isinstance(param, AnalogContext):
                        analog_params.append(param)
                    elif "classifier" in name:
                        clf_params.append(param)
                    else:
                        other_digital_params.append(param)
                param_groups = [
                    {"params": analog_params, "lr": learning_rate},
                    {"params": clf_params, "lr": _clf_lr},
                    {"params": other_digital_params, "lr": _clf_lr},
                ]
                print(f"  [LR SPLIT] analog={learning_rate:.2e}, classifier={_clf_lr:.2e}")
                if optimizer_name == "AnalogSGD":
                    optimizer = AnalogSGD(
                        param_groups, lr=learning_rate,
                        weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
                    )
                else:
                    optimizer = AnalogAdam(
                        param_groups, lr=learning_rate, weight_decay=weight_decay,
                    )
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
            optimizer.regroup_param_groups()

        num_training_steps = len(train_loader) * N_EPOCHS
        warmup_steps = int(num_training_steps * WARMUP_RATIO)
        _clf_lr_split = OPT_CONFIG.get('_trial_classifier_lr', OPT_CONFIG.get('classifier_lr', None))
        if _clf_lr_split is not None:
            # Per-group lambda: freeze→mini-warmup→decay for analog, constant for classifier
            def _analog_lambda(current_step):
                # Phase 1: freeze (classifier stabilizes first)
                if current_step <= warmup_steps:
                    return 0.0
                # Phase 2: mini-warmup for analog tile
                post = current_step - warmup_steps
                if post < warmup_steps:
                    return float(post) / float(max(1, warmup_steps))
                # Phase 3: linear decay
                progress = float(post - warmup_steps) / float(
                    max(1, num_training_steps - 2 * warmup_steps))
                return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
            _constant_lambda = lambda current_step: 1.0
            # param_groups order: [analog, classifier, other_digital]
            lambdas = []
            for pg in optimizer.param_groups:
                has_analog = any(isinstance(p, AnalogContext) for p in pg['params'])
                lambdas.append(_analog_lambda if has_analog else _constant_lambda)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
        else:
            scheduler = get_linear_schedule_with_min_lr(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                min_lr_rate=min_lr_rate,
            )

        best_metric = -float('inf')
        epochs_without_improvement = 0
        global_step = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
            for batch in pbar:
                global_step += 1


                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                if is_regression:
                    labels = labels.float() / 5.0  # normalize STS-B labels to [0, 1]

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze() if is_regression else outputs.logits
                loss = criterion(logits, labels)

                loss.backward()
                # Digital-only grad clipping (AnalogContext .grad does not affect tile update)
                from aihwkit.optim.context import AnalogContext as _AC
                _digital_params = [p for p in model.parameters()
                                   if not isinstance(p, _AC) and p.grad is not None]
                total_norm = torch.nn.utils.clip_grad_norm_(_digital_params, max_norm=1.0) if _digital_params else 0.0
                # Per-tile analog grad clip+floor (GPU-only, no .item() sync)
                if CLIP_ANALOG_GRAD and LORA_TARGET != "none":
                    from aihwkit.optim.context import AnalogContext
                    _max = ANALOG_TILE_MAX_NORM
                    _min = ANALOG_TILE_MIN_NORM
                    for p in model.parameters():
                        if isinstance(p, AnalogContext) and p.analog_grad_output:
                            for i, go in enumerate(p.analog_grad_output):
                                tile_norm = go.detach().norm()
                                scale = torch.where(
                                    tile_norm > _max, _max / (tile_norm + 1e-6),
                                    torch.where(
                                        (tile_norm < _min) & (tile_norm > 1e-10),
                                        _min / (tile_norm + 1e-6),
                                        tile_norm.new_ones(()),
                                    ),
                                )
                                p.analog_grad_output[i] = go * scale
                scheduler.step()
                # Sync analog tile lr with scheduler (lambda handles freeze/warmup/decay)
                for _pg in optimizer.param_groups:
                    for _p in _pg['params']:
                        if isinstance(_p, AnalogContext):
                            _p.analog_tile.set_learning_rate(_pg['lr'])
                optimizer.step()

                loss_val = loss.item()
                total_loss += loss_val
                num_batches += 1
                pbar.set_postfix(loss=f"{loss_val:.4f}")

                # Loss divergence detection: NaN only (regression can have large initial loss)
                if not np.isfinite(loss_val):
                    tqdm.write(f"[Trial {trial.number}] Loss diverged at step {global_step} "
                              f"(loss={loss_val:.2e}), stopping early.")
                    trial.set_user_attr("diverged", True)
                    return -float('inf') if not is_regression else -1.0

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

            trial.report(eval_metric, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

            # Early stopping: per-task patience
            es_patience = TASK_TO_ES_PATIENCE.get(TASK_NAME, 3)
            if epochs_without_improvement >= es_patience:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch} "
                          f"(no improvement for {es_patience} epochs)")
                break

            # Optuna pruner: median pruner can prune from epoch 2
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
    global BATCH_SIZE, N_EPOCHS, WARMUP_RATIO, LORA_TARGET, HEAD_LAYER, TASK_NAME, MAX_SEQ_LENGTH

    parser = argparse.ArgumentParser(description="Optuna sweep for MobileBERT GLUE TikiTaka v1")
    parser.add_argument('--task', type=str, default=TASK_NAME,
                        choices=GLUE_TASKS,
                        help=f'GLUE task (default: {TASK_NAME})')
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
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS,
                        help=f'Number of epochs (default: {N_EPOCHS})')
    parser.add_argument('--warmup-ratio', type=float, default=WARMUP_RATIO,
                        help=f'LR warmup ratio (default: {WARMUP_RATIO})')
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=['none', 'qonly', 'konly', 'vonly', 'qkv', 'ffn', 'all'],
                        help='Target: none, qonly, konly, vonly, qkv, ffn, all (default: qkv)')
    parser.add_argument('--head-layer', type=str, default=HEAD_LAYER,
                        choices=['train', 'freeze'],
                        help='classifier layer: train or freeze (default: train)')
    parser.add_argument('--target-ideal', action='store_true', default=False,
                        help='Use IdealDevice (FP updates) for target layers instead of TikiTaka')
    parser.add_argument('--lr-range', type=float, nargs=2, default=None,
                        help='Sweep learning rate range [min max] (e.g. --lr-range 1e-4 1.0)')
    parser.add_argument('--lr-override', type=float, default=None,
                        help='Override learning rate to this fixed value')
    parser.add_argument('--classifier-lr', type=float, default=None,
                        help='Separate LR for classifier (digital). If not set, uses same as analog lr')
    parser.add_argument('--fix-lr', type=float, nargs='+', default=None,
                        help='Fix learning_rate grid to these values (e.g. --fix-lr 0.1)')
    parser.add_argument('--fix-te', type=int, nargs='+', default=None,
                        help='Fix transfer_every grid to these values (e.g. --fix-te 1)')
    parser.add_argument('--fix-flr', type=float, nargs='+', default=None,
                        help='Fix fast_lr grid to these values (e.g. --fix-flr 1.0)')
    parser.add_argument('--learn-out-scaling', action='store_true', default=True,
                        help='Enable learn_out_scaling and out_scaling_columnwise (default: True)')
    parser.add_argument('--no-learn-out-scaling', dest='learn_out_scaling', action='store_false',
                        help='Disable learn_out_scaling')
    parser.add_argument('--clip-analog-grad', action='store_true', default=False,
                        help='Enable per-tile analog gradient clip+floor (default: False)')
    parser.add_argument('--no-clip-analog-grad', dest='clip_analog_grad', action='store_false',
                        help='Disable analog gradient clipping')
    parser.add_argument('--nontarget-digital', action='store_true', default=True,
                        help='Keep non-target encoder layers as digital (default: True)')
    parser.add_argument('--nontarget-analog', dest='nontarget_digital', action='store_false',
                        help='Convert non-target encoder layers to frozen SingleRPU analog')
    parser.add_argument('--backward-perfect', action='store_true', default=False,
                        help='Use perfect backward pass (no DAC/ADC quantization on gradients)')
    parser.add_argument('--auto-scale', action='store_true', default=True,
                        help='Enable auto_scale: dynamically normalise fast_lr by gradient magnitude')
    parser.add_argument('--no-auto-scale', action='store_false', dest='auto_scale',
                        help='Disable auto_scale')
    parser.add_argument('--transfer-every', type=int, default=None,
                        help='Override transfer_every (default: 1). Large value = effectively no transfer')
    parser.add_argument('--uim', action='store_true', dest='units_in_mbatch', default=True,
                        help='units_in_mbatch=True (default)')
    parser.add_argument('--no-uim', action='store_false', dest='units_in_mbatch',
                        help='units_in_mbatch=False (count in mat-vec units)')
    parser.add_argument('--desired-bl', type=int, default=31,
                        help='Transfer update desired_bl (default: 31)')
    parser.add_argument('--bl-sweep', type=int, nargs=2, default=None,
                        help='Sweep desired_bl range [min max] (e.g. --bl-sweep 1 31)')
    parser.add_argument('--bl-grid', type=int, nargs='+', default=None,
                        help='Grid of desired_bl values (e.g. --bl-grid 31 60)')
    parser.add_argument('--use-v2', action='store_true', default=True,
                        help='Use TikiTaka v2 (ChoppedTransfer with buffer+chopper, bl=1)')
    parser.add_argument('--no-v2', action='store_false', dest='use_v2',
                        help='Disable TikiTaka v2, use v1 instead')
    parser.add_argument('--no-scale-transfer-lr', action='store_true', default=False,
                        help='Force scale_transfer_lr=False (override --use-v2 default)')
    parser.add_argument('--scale-transfer-lr', action='store_false', dest='no_scale_transfer_lr',
                        help='Enable scale_transfer_lr (v2 default behavior, default)')
    parser.add_argument('--lr-upper-mult', type=float, default=10.0,
                        help='LR upper bound multiplier (default: 10.0, e.g. 10 for 10x base_lr)')
    parser.add_argument('--sampler', type=str, default='tpe', choices=['grid', 'tpe'],
                        help='Sampler type: tpe (TPESampler) or grid (GridSampler) (default: tpe)')
    parser.add_argument('--tpe-lr-range', type=float, nargs=2, default=None,
                        help='TPE lr range (e.g. --tpe-lr-range 0.001 0.1)')
    parser.add_argument('--tpe-flr-range', type=float, nargs=2, default=[1.0, 1.0],
                        help='TPE fast_lr range (default: 1.0 1.0, fixed)')
    parser.add_argument('--tpe-tlr-range', type=float, nargs=2, default=[1.0, 1000.0],
                        help='TPE transfer_lr range (default: 0.01 1.0)')
    args = parser.parse_args()

    # Update global config
    TASK_NAME = args.task

    # Apply per-task settings from Albert_setup.txt
    BATCH_SIZE = TASK_TO_BSZ.get(TASK_NAME, args.batch_size)
    N_EPOCHS = TASK_TO_EPOCHS.get(TASK_NAME, args.epochs)
    MAX_SEQ_LENGTH = TASK_TO_MAXSEQ.get(TASK_NAME, 128)

    WARMUP_RATIO = args.warmup_ratio
    LORA_TARGET = args.lora_target
    HEAD_LAYER = args.head_layer
    OPT_CONFIG['target_ideal'] = args.target_ideal
    if args.lr_range is not None:
        OPT_CONFIG['lr_range'] = tuple(args.lr_range)
    if args.lr_override is not None:
        OPT_CONFIG['lr_override'] = args.lr_override
    if args.classifier_lr is not None:
        OPT_CONFIG['classifier_lr'] = args.classifier_lr
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov
    OPT_CONFIG['learn_out_scaling'] = args.learn_out_scaling
    OPT_CONFIG['nontarget_digital'] = args.nontarget_digital
    OPT_CONFIG['backward_perfect'] = args.backward_perfect
    OPT_CONFIG['auto_scale'] = args.auto_scale
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

    global CLIP_ANALOG_GRAD
    CLIP_ANALOG_GRAD = args.clip_analog_grad

    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    _ideal_suffix = "_ideal" if args.target_ideal else ""
    study_name = args.study_name or f"mobilebert_tiki_{TASK_NAME}{_ideal_suffix}_{timestamp}"

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
    print(f"BSZ: {BATCH_SIZE}, max_seq: {MAX_SEQ_LENGTH}, epochs: {N_EPOCHS}")
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    global GRID_LR, GRID_TE, SAMPLER_TYPE, TPE_LR_RANGE, TPE_FLR_RANGE, TPE_TLR_RANGE
    SAMPLER_TYPE = args.sampler

    if args.tpe_lr_range:
        TPE_LR_RANGE = tuple(args.tpe_lr_range)
    if args.tpe_flr_range:
        TPE_FLR_RANGE = tuple(args.tpe_flr_range)
    if args.tpe_tlr_range:
        TPE_TLR_RANGE = tuple(args.tpe_tlr_range)

    task_lr = TASK_TO_LR.get(TASK_NAME, 0.001)
    print(f"  lr={task_lr} (fixed, per-task)")
    print(f"  transfer_every=1 (fixed, uim=True, every mini-batch)")
    print(f"  fast_lr=1.0 (fixed), transfer_lr=1.0 (fixed)")

    sampler = optuna.samplers.TPESampler(seed=SEED)

    # Pruner: MedianPruner — prune below median, warmup = epochs // 3
    prune_warmup = max(2, N_EPOCHS // 3)
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=prune_warmup,
        ),
        load_if_exists=True,
    )
    print(f"  Early stop patience: {TASK_TO_ES_PATIENCE.get(TASK_NAME, 3)}, "
          f"Pruner: Median, startup=5, warmup={prune_warmup}")

    print(f"\nStudy: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

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
    pass


def _oom_restart_callback(study, trial):
    if trial.state == TrialState.FAIL:
        err = trial.user_attrs.get("error", "")
        if "out of memory" in err.lower() or "cublas" in err.lower():
            print(f"\n[OOM Recovery] Trial {trial.number} failed with CUDA error, will restart process.")
            raise _OOMRestart()


if __name__ == "__main__":
    main()
