# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for MobileBERT + SST-2 with LRTT.

Usage:
    python optuna_mobilebert_sst2_lrtt.py --n-trials 50
    python optuna_mobilebert_sst2_lrtt.py --visualize
    python optuna_mobilebert_sst2_lrtt.py --n-trials 50 --optimizer AnalogSGD --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 64 --epochs 15 --warmup-steps 189 --transfer-method set --no-io-noise --lora-target qkv

All flags:
    python optuna_mobilebert_sst2_lrtt.py \
        --study-name <str>          # Study name (default: auto-generated)
        --n-trials <int>            # Number of Optuna trials (default: 50)
        --visualize                 # Visualize study results and exit
        --optimizer <str>           # AnalogSGD | AnalogAdam (default: AnalogSGD)
        --no-wd                     # Disable weight decay tuning (fix to 0)
        --no-momentum               # Disable momentum tuning (fix to 0, SGD only)
        --no-nesterov               # Disable nesterov tuning (fix to False, SGD only)
        --reinit-mode <str>         # Fix reinit mode: standard | decay | hybrid (default: tune all)
        --batch-size <int>          # Batch size (default: 64)
        --epochs <int>              # Number of epochs (default: 15)
        --warmup-steps <int>        # LR warmup steps (default: 189)
        --transfer-method <str>     # Transfer method: onehot | direct | set (default: onehot)
        --ab-device <str>           # A/B tile device: 6t1c | fp (default: 6t1c)
        --no-io-noise               # Disable IO out_noise (resolution kept)
        --lora-target <str>         # LoRA target: none | qonly | konly | vonly | qkv | ffn | all (default: qkv)
        --head-layer <str>          # classifier: train | freeze (default: train)
        --no-transfer               # Disable LRTT transfer (A/B frozen, skip LRTT param sweep)

Inline flags (edit directly in script):
    DYNAMIC_TE = False              # Enable dynamic transfer every
    DYNAMIC_TE_POWER = 1.0          # Power for dynamic TE scaling
    TE_WARMUP_STEPS = 0            # Steps before reaching target TE
    TE_WARMUP_SCHEDULE = []         # Warmup TE schedule list
    REINIT_GAIN = 1.0               # Reinitialization gain
    DECAY_FACTOR = 1.0              # Decay factor for reinit
    TARGET_MODULES = [...]          # Modules to convert to analog
    TRAIN_SUBSET_SIZE = 0           # Training data subset (0 = full)
    EVAL_SUBSET_SIZE = 0            # Evaluation data subset (0 = full)

Enqueue
python3 << 'EOF'                                                                                                                                                                                                                                                      
import optuna                                                
study = optuna.load_study(                                                                                                                                                                                                                                            
study_name='mobilebert_sst2_lrtt_bs64_sgd_hybrid_nowd_nomom_nonest_set_noio_qkv',                                                                                                                                                                                
storage='sqlite:///results/optuna_mobilebert_sst2_lrtt/optuna_mobilebert_sst2_lrtt_bs64_sgd_hybrid_nowd_nomom_nonest_set_noio_qkv.db')
study.enqueue_trial({
'learning_rate': 0.2080749864869466,
'transfer_lr': 0.010000000000000004,
'transfer_every': 16210,
'rank_exp': 1,
'lora_alpha': 0.41139594231202437,
'tau_sec': 0.0,
'min_lr_rate': 0.0})
print('Enqueued!')
EOF
  
"""

import os
import sys
import math
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
from optuna_integration import BoTorchSampler
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
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice, FloatingPointDevice

# LRTT config imports (direct imports to avoid __init__.py dependency issues)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.parameters.mapping import MappingParameter


# =============================================================================
# ConfigAwareBoTorchSampler with Periodic Exploration
# =============================================================================

class ConfigAwareBoTorchSampler(BoTorchSampler):
    """BoTorchSampler that respects OPT_CONFIG and avoids duplicate running trials."""

    def _is_almost_identical_to_running(self, params, study, threshold=0.01):
        """Check if params are almost identical to any running trial (within 1%)."""
        running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        for rt in running_trials:
            identical = True
            for key, val in params.items():
                if key in rt.params and isinstance(val, (int, float)):
                    rt_val = rt.params[key]
                    if rt_val != 0:
                        if abs(val - rt_val) / abs(rt_val) > threshold:
                            identical = False
                            break
                    elif val != 0:
                        identical = False
                        break
            if identical:
                return True
        return False

    def _add_jitter(self, params, search_space, jitter_ratio=0.1):
        """Add small jitter to params while keeping within search space bounds."""
        import random
        jittered = dict(params)
        for key, val in params.items():
            if key in search_space and isinstance(val, float):
                dist = search_space[key]
                if hasattr(dist, 'low') and hasattr(dist, 'high'):
                    # Add ±10% jitter
                    jitter = val * random.uniform(-jitter_ratio, jitter_ratio)
                    new_val = val + jitter
                    # Clip to bounds
                    new_val = max(dist.low, min(dist.high, new_val))
                    jittered[key] = new_val
        return jittered

    def sample_relative(self, study, trial, search_space):
        params = super().sample_relative(study, trial, search_space)

        # If almost identical to running trial, add jitter to GP params
        if self._is_almost_identical_to_running(params, study):
            params = self._add_jitter(params, search_space)

        # Force reinit_mode if fixed in config
        if OPT_CONFIG['reinit_mode'] is not None and 'reinit_mode' in params:
            params['reinit_mode'] = OPT_CONFIG['reinit_mode']
        # Force optimizer if fixed in config
        if 'optimizer' in params:
            params['optimizer'] = OPT_CONFIG['optimizer']
        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        if param_name == 'reinit_mode' and OPT_CONFIG['reinit_mode'] is not None:
            return OPT_CONFIG['reinit_mode']
        if param_name == 'optimizer':
            return OPT_CONFIG['optimizer']
        return super().sample_independent(study, trial, param_name, param_distribution)


# =============================================================================
# Global Constants
# =============================================================================

DEFAULT_STUDY_NAME = "mobilebert_sst2_lrtt_main"

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = os.path.join(os.getcwd(), "results", "optuna_mobilebert_sst2_lrtt")
os.makedirs(RESULTS, exist_ok=True)

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128
NUM_LABELS = 2  # SST-2: negative (0), positive (1)

# Training defaults
N_EPOCHS = 15
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
EARLY_STOP_PATIENCE = 3

# Scheduler
WARMUP_STEPS = 500  # warmup steps

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

# LoRA target options: which layers have trainable A/B tiles
# - none: no LRTT layers (fully digital baseline)
# - qkv: only query, key, value
# - ffn: projection (attention.output) + FFN (intermediate, output, bottleneck)
# - all: all encoder linear layers
LORA_TARGET = "qkv"  # default, can be set via --lora-target
HEAD_LAYER = "train"  # default, can be set via --head-layer (train | freeze) - classifier for SST-2
LORA_TARGET_MODULES = {
    "none": [],  # Empty = no layers converted to LRTT (fully digital)
    "qonly": ["query"],  # Query only (24 layers)
    "konly": ["key"],  # Key only (24 layers)
    "vonly": ["value"],  # Value only (24 layers)
    "qkv": ["query", "key", "value"],  # Q/K/V (72 layers)
    "ffn": ["dense"],  # All layers with "dense" (excludes qkv) (288 layers)
    "all": None,  # None means all encoder layers (no filtering) (360 layers)
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
    'reinit_mode': None,    # None = tune, or 'standard'/'decay'/'hybrid' = fixed
    'no_transfer': False,   # If True, disable transfer (transfer_every = inf)
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

    if OPT_CONFIG['no_transfer']:
        suffix += "_notrans"

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
        mult_noise=False,  # No multiplicative noise for C tile
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
    device_config.transfer_lr = transfer_lr
    device_config.units_in_mbatch = True
    device_config.transfer_method = TRANSFER_METHOD
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = False

    # Dynamic TE
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
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


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


def create_model(params):
    """Create MobileBERT classification model with selective LRTT analog layers.

    Architecture (follows paper's approach for efficiency):
        - LRTT Target layers (based on --lora-target) → LRTT Analog
        - Non-target Encoder layers → Digital FROZEN
        - classifier → Digital TRAINABLE (weight + bias)
        - embedding_transformation → Digital FROZEN
        - Embeddings → Digital FROZEN

    LoRA Target Options (--lora-target):
        - qkv: Q/K/V layers → LRTT Analog (72 layers)
        - ffn: projection + FFN layers → LRTT Analog (288 layers)
        - all: all encoder layers → LRTT Analog (360 layers)

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
        # embedding_transformation: LRTT for "all" mode only
        if "embedding_transformation" in layer_name:
            return (LORA_TARGET == "all")
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

    # Exclude classifier (always digital)
    # embedding_transformation: LRTT for "all" mode, digital frozen otherwise
    exclude_modules.append("classifier")
    if LORA_TARGET != "all":
        exclude_modules.append("mobilebert.embeddings.embedding_transformation")
    exclude_modules = list(set(exclude_modules))  # Remove duplicates

    # Step 1: Convert only LRTT target layers to LRTT Analog (skip if none mode)
    if LORA_TARGET == "none":
        # None mode: fully digital, no analog conversion
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

        # Convert to analog with exclusions (only LRTT targets get converted)
        model = convert_to_analog(model, lrtt_config, exclude_modules=exclude_modules)

        # Count analog layers
        analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LRTT Analog layers: {analog_count}, Total params: {total_params:,}, Trainable (before grad set): {trainable_before:,}")

    # Step 2: Set requires_grad
    # - LRTT layers: A/B + out_scaling TRAINABLE, C + bias FROZEN
    # - classifier: TRAINABLE if HEAD_LAYER=="train", else FROZEN
    # - embedding_transformation: LRTT for "all" mode (A/B trainable, C frozen), digital frozen otherwise
    # - Everything else: FROZEN
    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            # LRTT A/B tiles: TRAINABLE, but FROZEN if no_transfer (no point updating without transfer)
            param.requires_grad = not OPT_CONFIG['no_transfer']
        elif "tile_c" in name:
            pass  # Respect lrtt_tile.py settings (train_c_bias, mapping_c)
        elif "classifier" in name:
            # classifier: TRAINABLE or FROZEN based on setting
            param.requires_grad = (HEAD_LAYER == "train")
        elif "embedding_transformation" in name:
            param.requires_grad = False
        else:
            param.requires_grad = False

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable (after grad set): {trainable_after:,}")
    print(f"  LoRA target: {LORA_TARGET} -> {lrtt_patterns if lrtt_patterns else 'all encoder layers'}")

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

def objective(trial, train_loader, eval_loader, tokenizer):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 6e-3, 2e1, log=True)

    # LRTT parameters: skip sweep if --no-transfer (A/B frozen, no transfer happens)
    if OPT_CONFIG['no_transfer']:
        transfer_lr = 0.1        # fixed (not used anyway)
        transfer_every = 999999999
        rank_exp = 2             # fixed (A=0 init, no effect)
        rank = 4
        lora_alpha = 1.0         # fixed (no effect)
        tau_sec = 0.0            # fixed
    else:
        transfer_lr = trial.suggest_float('transfer_lr', 1e-5, 2e-2, log=True)
        transfer_every = trial.suggest_int('transfer_every', 63, 32000, log=True)
        rank_exp = trial.suggest_int('rank_exp', 0, 5)
        rank = 2 ** rank_exp
        lora_alpha = trial.suggest_float('lora_alpha', 1e-5, 4e1, log=True)
        tau_sec = trial.suggest_float('tau_sec', 0, 0, log=False)  # 0 = no decay

    min_lr_rate = trial.suggest_float('min_lr_rate', 0.0, 0.0)

    # weight_decay: tune or fix to 0
    if OPT_CONFIG['tune_wd']:
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

    # reinit_mode: use fixed value if set, otherwise tune
    if OPT_CONFIG['reinit_mode'] is not None:
        reinit_mode = OPT_CONFIG['reinit_mode']
    else:
        reinit_mode = trial.suggest_categorical('reinit_mode', ['standard', 'decay', 'hybrid'])

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
    print(f"Trial {trial.number} Starting")
    print(f"{'='*70}")
    print(f"  rank={rank}, transfer_every={transfer_every}, transfer_lr={transfer_lr:.4e}")
    print(f"  lora_alpha={lora_alpha:.4f}, lr={learning_rate:.2e}, wd={weight_decay:.2e}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, reinit_mode={reinit_mode}")
    print(f"  tau_sec={tau_sec:.1f}, optimizer={optimizer_name}, min_lr_rate={min_lr_rate:.4f}")
    print(f"{'='*70}")

    model = None
    try:
        set_seed(SEED)

        model = create_model(params)

        if LORA_TARGET == "none":
            # None mode: use standard PyTorch optimizers
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
            # LRTT modes: use Analog optimizers
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
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
        )

        best_acc = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
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
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            eval_acc = evaluate_model(model, eval_loader)

            improved = ""
            if eval_acc > best_acc:
                best_acc = eval_acc
                epochs_without_improvement = 0
                improved = " ★"
            else:
                epochs_without_improvement += 1

            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"[Trial {trial.number}] Epoch {epoch:3d} | "
                      f"Acc: {eval_acc:6.2f}% | Best Acc: {best_acc:6.2f}% | "
                      f"Loss: {train_loss:.4f} | LR: {current_lr:.2e} | "
                      f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}{improved}")

            trial.report(best_acc, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break

            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        tqdm.write(f"\n[Trial {trial.number}] Finished - Best Acc: {best_acc:.2f}%")
        tqdm.write(f"{'='*70}\n")
        return best_acc

    except Exception as e:
        error_msg = str(e)[:500]
        trial.set_user_attr("error", error_msg)
        tqdm.write(f"[Trial {trial.number}] Error: {error_msg}")
        raise

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        tqdm.write(f"[Trial {trial.number}] GPU cache cleared")


# =============================================================================
# Visualization
# =============================================================================

def visualize_study(study, save_dir):
    """Visualize optimization history, parameter importance, and LR vs Acc."""
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        print("No completed trials to visualize.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    trial_numbers = [t.number for t in complete_trials]
    acc_scores = [t.value for t in complete_trials]

    # Optimization history
    axes[0].scatter(trial_numbers, acc_scores, alpha=0.6)
    axes[0].plot(trial_numbers,
                 [max(acc_scores[:i+1]) for i in range(len(acc_scores))],
                 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Optimization History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Parameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys())[::-1], list(importances.values())[::-1])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Parameter Importance')
    except Exception:
        axes[1].text(0.5, 0.5, 'Not enough trials', ha='center', va='center',
                     transform=axes[1].transAxes)

    # LR vs Acc
    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[2].scatter(lrs, acc_scores, alpha=0.6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Learning Rate vs Accuracy')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "visualization.png"), dpi=150, bbox_inches='tight')
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
        acc_scores = [t.value for t in complete_trials]
        print(f"Best Acc: {max(acc_scores):.2f}%, Mean Acc: {sum(acc_scores)/len(acc_scores):.2f}%")
        print(f"Best params: {study.best_params}")


# =============================================================================
# Main
# =============================================================================

def main():
    global BATCH_SIZE, N_EPOCHS, WARMUP_STEPS, TRANSFER_METHOD, AB_DEVICE, IO_NOISE, LORA_TARGET, HEAD_LAYER

    parser = argparse.ArgumentParser(description="Optuna sweep for MobileBERT SST-2 LRTT")
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: auto-generated based on config)')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--optimizer', type=str, default='AnalogAdam',
                        choices=['AnalogSGD', 'AnalogAdam'],
                        help='Optimizer type (default: AnalogAdam)')
    parser.add_argument('--no-wd', action='store_true',
                        help='Disable weight decay tuning (fix to 0)')
    parser.add_argument('--no-momentum', action='store_true',
                        help='Disable momentum tuning (fix to 0, SGD only)')
    parser.add_argument('--no-nesterov', action='store_true',
                        help='Disable nesterov tuning (fix to False, SGD only)')
    parser.add_argument('--reinit-mode', type=str, default=None,
                        choices=['standard', 'decay', 'hybrid'],
                        help='Fix reinit mode (default: tune all three)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS,
                        help=f'Number of epochs (default: {N_EPOCHS})')
    parser.add_argument('--warmup-steps', type=int, default=WARMUP_STEPS,
                        help=f'LR warmup steps (default: {WARMUP_STEPS})')
    parser.add_argument('--transfer-method', type=str, default=TRANSFER_METHOD,
                        choices=['onehot', 'direct', 'set'],
                        help=f'Transfer method (default: {TRANSFER_METHOD})')
    parser.add_argument('--ab-device', type=str, default=AB_DEVICE,
                        choices=['6t1c', 'fp'],
                        help=f'A/B tile device type (default: {AB_DEVICE})')
    parser.add_argument('--no-io-noise', action='store_true',
                        help='Disable IO out_noise (resolution kept)')
    parser.add_argument('--no-transfer', action='store_true',
                        help='Disable transfer (set transfer_every to infinity)')
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=['none', 'qonly', 'konly', 'vonly', 'qkv', 'ffn', 'all'],
                        help='LoRA target: none, qonly, konly, vonly, qkv, ffn, all (default: qkv)')
    parser.add_argument('--head-layer', type=str, default=HEAD_LAYER,
                        choices=['train', 'freeze'],
                        help='classifier layer: train or freeze (default: train)')
    args = parser.parse_args()

    # Update global config
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    WARMUP_STEPS = args.warmup_steps
    TRANSFER_METHOD = args.transfer_method
    AB_DEVICE = args.ab_device
    IO_NOISE = not args.no_io_noise
    LORA_TARGET = args.lora_target
    HEAD_LAYER = args.head_layer
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['reinit_mode'] = args.reinit_mode
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov
    OPT_CONFIG['no_transfer'] = args.no_transfer

    # Auto-generate study name based on config (includes batch size)
    study_name = args.study_name or f"mobilebert_sst2_lrtt_bs{BATCH_SIZE}_{get_study_name_suffix()}"

    storage = f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    # Load data once (shared across all trials)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=ConfigAwareBoTorchSampler(n_startup_trials=10),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

    # Run trials with OOM recovery via process restart
    target_total = len(study.trials) + args.n_trials
    completed_before = len(study.trials)

    try:
        study.optimize(
            lambda trial: objective(trial, train_loader, eval_loader, tokenizer),
            n_trials=args.n_trials,
            catch=(Exception,),
            show_progress_bar=False,
            callbacks=[_oom_restart_callback],
        )
    except _OOMRestart:
        remaining = target_total - len(study.trials)
        if remaining > 0:
            print(f"\n[OOM Recovery] Restarting process for {remaining} remaining trials...")
            # Replace --n-trials in argv with remaining count
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
                "best_acc": study.best_value,
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

    all_trials_file = os.path.join(RESULTS, "all_trials.json")
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
