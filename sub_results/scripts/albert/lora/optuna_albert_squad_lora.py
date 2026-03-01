# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for ALBERT + SQuAD with LoRA-LRTT.

LoRA mode: forward_inject=True, transfer disabled (transfer_every=10^7)
Searches: learning_rate, lora_alpha, target_ab_lr

Usage:
    python optuna_albert_squad_lora.py --n-trials 50
    python optuna_albert_squad_lora.py --visualize
    python optuna_albert_squad_lora.py --n-trials 50 --optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --batch-size 64 --epochs 15 --warmup-steps 500 --lora-target attn

All flags:
    python optuna_albert_squad_lora.py \
        --study-name <str>          # Study name (default: auto-generated)
        --n-trials <int>            # Number of Optuna trials (default: 50)
        --visualize                 # Visualize study results and exit
        --optimizer <str>           # AnalogSGD | AnalogAdam (default: AnalogSGD)
        --no-wd                     # Disable weight decay tuning (fix to 0)
        --no-momentum               # Disable momentum tuning (fix to 0, SGD only)
        --no-nesterov               # Disable nesterov tuning (fix to False, SGD only)
        --reinit-mode <str>         # Fix reinit mode: standard | decay | hybrid (default: decay)
        --batch-size <int>          # Batch size (default: 64)
        --epochs <int>              # Number of epochs (default: 15)
        --warmup-steps <int>        # LR warmup steps (default: 500)
        --warmup-ratio <float>      # LR warmup ratio (overrides --warmup-steps if > 0)
        --transfer-method <str>     # Transfer method: onehot | direct | set (default: onehot)
        --ab-device <str>           # A/B tile device: 6t1c | fp (default: 6t1c)
        --no-io-noise               # Disable IO out_noise (resolution kept)
        --lora-target <str>         # LoRA target: none | qonly | konly | vonly | qkv | attn | ffn | all (default: attn)
        --head-layer <str>          # qa_outputs: train | freeze (default: train)
        --no-transfer               # Disable LRTT transfer (A/B frozen, skip LRTT param sweep)
        --warm-alpha                # Enable warm-up for lora_alpha
        --convert-nontarget         # Convert non-target layers to analog (default: on)
        --no-convert-nontarget      # Disable non-target layer analog conversion

Inline flags (edit directly in script):
    TRAIN_SUBSET_SIZE = 0           # Training data subset (0 = full)
    EVAL_SUBSET_SIZE = 0            # Evaluation data subset (0 = full)

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
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
import evaluate

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
# Global Constants
# =============================================================================

DEFAULT_STUDY_NAME = "albert_squad_lora_main"

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = "/data/results/Analoglora_v2/squad"
os.makedirs(RESULTS, exist_ok=True)

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "albert/albert-base-v2"
MAX_SEQ_LENGTH = 384

# Training defaults
N_EPOCHS = 2
BATCH_SIZE = 48
EVAL_BATCH_SIZE = 256
EARLY_STOP_PATIENCE = 999  # disabled (2 epoch only)

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
    'optimizer': 'AnalogAdam',
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

    if OPT_CONFIG['no_transfer']:
        suffix += "_notrans"

    # Non-target analog conversion
    if CONVERT_NONTARGET:
        suffix += "_convnt"

    # Add lora target (always include for clarity)
    suffix += f"_{LORA_TARGET}"

    # Add head_layer if frozen (not default)
    if HEAD_LAYER == "freeze":
        suffix += "_headfreeze"

    if WARM_ALPHA:
        suffix += "_warmalpha"

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
    rpu_config.mapping.learn_out_scaling = True
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
    rpu_config.mapping.learn_out_scaling = True
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
    """Create ALBERT QA model with selective LRTT analog layers.

    Architecture:
        - LRTT Target layers (based on --lora-target) -> LRTT Analog
        - Non-target Encoder layers -> SingleRPU (frozen) if CONVERT_NONTARGET
        - qa_outputs -> Digital TRAINABLE (based on HEAD_LAYER)
        - embedding_hidden_mapping_in -> Digital FROZEN
        - Embeddings -> Digital FROZEN

    ALBERT uses weight sharing: all transformer blocks share the same
    parameters, so the actual number of unique analog layers is small.

    LRTT layers have:
        - A/B tiles: TRAINABLE
        - C-tile: FROZEN (pretrained weights)
        - combined_out_scaling: TRAINABLE (if enabled)
        - out_scaling: FROZEN
        - bias: FROZEN
    """
    from aihwkit.nn import AnalogLinear

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Get LRTT target patterns
    lrtt_patterns = get_lrtt_target_module_names(LORA_TARGET)

    # Always exclude from any analog conversion
    always_digital = ["qa_outputs", "albert.encoder.embedding_hidden_mapping_in"]

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
    exclude_modules.append("qa_outputs")
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

    # Step 2: Set requires_grad (matching tiki pattern: all frozen except LRTT A/B, qa_outputs, LayerNorm)
    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            # LRTT A/B tiles: TRAINABLE (LoRA low-rank update)
            param.requires_grad = True
        elif "tile_c" in name and "analog_ctx" in name:
            # tile_c.analog_ctx: TRAINABLE (forward_inject mechanism)
            param.requires_grad = True
        elif "out_scaling" in name:
            # out_scaling: TRAINABLE (weight scaling compensation)
            param.requires_grad = True
        elif "qa_outputs" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
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
    """Load and tokenize SQuAD dataset."""
    raw_datasets = load_dataset("squad")

    # Use full dataset if EVAL_SUBSET_SIZE == 0, otherwise subset
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
            stride=128, return_overflowing_tokens=True,
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
            stride=128, return_overflowing_tokens=True,
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
    # Use full dataset if TRAIN_SUBSET_SIZE == 0, otherwise subset
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

def objective(trial, train_loader, eval_features, eval_examples, tokenizer):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Hyperparameters - TPE search over lr, lora_alpha, target_ab_lr
    # learning_rate: digital params, sweep range = baseline best 1.58e-3 × [1/10, 3]
    learning_rate = trial.suggest_float('learning_rate', 1.58e-4, 4.74e-3, log=True)
    lora_alpha_val = trial.suggest_float('lora_alpha', 1e-4, 1e-1, log=True)
    target_ab_lr = trial.suggest_float('target_ab_lr', 1e-3, 1e-1, log=True)

    # LRTT parameters: skip sweep if --no-transfer (A/B frozen, no transfer happens)
    if OPT_CONFIG['no_transfer']:
        transfer_lr = 0.1
        transfer_every = 999999999
        rank = 16
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

    min_lr_rate = 0.5  # min LR = 50% of peak LR

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
    print(f"Trial {trial.number} Starting")
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

        model = create_model(params)

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

        # Sync analog tile lr with scheduler-adjusted initial lr (lr_lambda(0) applied at init)
        for group in optimizer.param_groups:
            for p in group["params"]:
                if hasattr(p, 'analog_tile'):
                    p.analog_tile.set_learning_rate(group["lr"])

        best_f1 = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                start_positions = batch['start_positions'].to(DEVICE)
                end_positions = batch['end_positions'].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    start_positions=start_positions, end_positions=end_positions,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Alpha warm-up: linear 0 -> lora_alpha over warmup steps
                if WARM_ALPHA and LORA_TARGET != "none":
                    global_step = (epoch - 1) * len(train_loader) + batch_idx
                    if global_step < WARMUP_STEPS:
                        current_alpha = lora_alpha * (global_step / WARMUP_STEPS)
                    else:
                        current_alpha = lora_alpha
                    for m in model.modules():
                        if hasattr(m, 'controller') and hasattr(m, 'lora_alpha'):
                            m.lora_alpha = current_alpha
                            m.controller.lora_alpha = current_alpha

                loss_val = loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    print(f"\n  [NaN/Inf detected at batch {num_batches}] Aborting trial.")
                    return 0.0  # Return worst score
                total_loss += loss_val
                num_batches += 1
                pbar.set_postfix(loss=f"{loss_val:.4f}")

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

            trial.report(best_f1, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

            # Abort hopeless trials: F1 < 20% after epoch 1
            if epoch == 1 and eval_f1 < 20.0:
                tqdm.write(f"[Trial {trial.number}] F1={eval_f1:.2f}% < 20% at epoch 1 → abort")
                break

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break

            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

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
    """Visualize optimization history, parameter importance, and LR vs F1."""
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        print("No completed trials to visualize.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    trial_numbers = [t.number for t in complete_trials]
    f1_scores = [t.value for t in complete_trials]

    # Optimization history
    axes[0].scatter(trial_numbers, f1_scores, alpha=0.6)
    axes[0].plot(trial_numbers,
                 [max(f1_scores[:i+1]) for i in range(len(f1_scores))],
                 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('F1 (%)')
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

    # LR vs F1
    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[2].scatter(lrs, f1_scores, alpha=0.6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel('F1 (%)')
    axes[2].set_title('Learning Rate vs F1')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "visualization_squad.png"), dpi=150, bbox_inches='tight')
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
    global BATCH_SIZE, N_EPOCHS, WARMUP_STEPS, WARM_ALPHA, TRANSFER_METHOD, AB_DEVICE, IO_NOISE, LORA_TARGET, HEAD_LAYER, AB_PERFECT_IO, COMBINED_OUT_SCALING, CONVERT_NONTARGET

    parser = argparse.ArgumentParser(description="Optuna sweep for ALBERT SQuAD LoRA-LRTT")
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
    parser.add_argument('--batch-size', type=int, default=48,
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS,
                        help=f'Number of epochs (default: {N_EPOCHS})')
    parser.add_argument('--warmup-steps', type=int, default=WARMUP_STEPS,
                        help=f'LR warmup steps (default: {WARMUP_STEPS})')
    parser.add_argument('--warmup-ratio', type=float, default=WARMUP_RATIO,
                        help=f'LR warmup ratio (fraction of total steps, overrides --warmup-steps if > 0, default: {WARMUP_RATIO})')
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
    parser.add_argument('--no-transfer', action='store_true',
                        help='Disable transfer (set transfer_every to infinity)')
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=['none', 'qonly', 'konly', 'vonly', 'qkv', 'attn', 'ffn', 'all'],
                        help='LoRA target: none, qonly, konly, vonly, qkv, attn, ffn, all (default: attn)')
    parser.add_argument('--head-layer', type=str, default=HEAD_LAYER,
                        choices=['train', 'freeze'],
                        help='qa_outputs layer: train or freeze (default: train)')
    parser.add_argument('--warm-alpha', action='store_true',
                        help='Enable warm-up for lora_alpha (linear: 0 -> target over warmup steps)')
    parser.add_argument('--convert-nontarget', action='store_true', default=True,
                        help='Convert non-target layers to analog (SingleRPU+SoftBounds, frozen)')
    parser.add_argument('--no-convert-nontarget', dest='convert_nontarget', action='store_false',
                        help='Disable non-target layer analog conversion')
    args = parser.parse_args()

    # Update global config
    WARM_ALPHA = args.warm_alpha
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    WARMUP_STEPS = args.warmup_steps
    TRANSFER_METHOD = args.transfer_method
    AB_DEVICE = args.ab_device
    IO_NOISE = not args.no_io_noise
    AB_PERFECT_IO = args.ab_perfect_io
    COMBINED_OUT_SCALING = args.combined_out_scaling
    CONVERT_NONTARGET = args.convert_nontarget
    LORA_TARGET = args.lora_target
    HEAD_LAYER = args.head_layer
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['reinit_mode'] = args.reinit_mode
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov
    OPT_CONFIG['no_transfer'] = args.no_transfer

    # Auto-generate study name based on config (includes batch size)
    study_name = args.study_name or f"albert_squad_lrtt_bs{BATCH_SIZE}_{get_study_name_suffix()}"

    storage = f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    # Load data once (shared across all trials)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    # Compute warmup from ratio if specified
    if args.warmup_ratio > 0:
        num_total_steps = len(train_loader) * N_EPOCHS
        WARMUP_STEPS = int(num_total_steps * args.warmup_ratio)
        print(f"Warmup ratio {args.warmup_ratio} -> {WARMUP_STEPS} steps (total: {num_total_steps})")

    # TPESampler: Bayesian optimization
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )

    # Enqueue seed trial (MobileBERT SQuAD LoRA best params)
    # Skip if trials already exist (e.g. injected from previous study)
    if len(study.trials) == 0:
        study.enqueue_trial({
            'learning_rate': 0.2858051065806936,
            'lora_alpha': 0.06097839109531514,
            'target_ab_lr': 0.009093929525644107,
        })

    print(f"\nStudy: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

    # Run trials
    target_total = len(study.trials) + args.n_trials
    completed_before = len(study.trials)

    study.optimize(
        lambda trial: objective(trial, train_loader, eval_features, eval_examples, tokenizer),
        n_trials=args.n_trials,
        catch=(Exception,),
        show_progress_bar=False,
    )

    print_study_summary(study)
    visualize_study(study, RESULTS)

    # Save best params
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if complete_trials:
        best_params_file = os.path.join(RESULTS, f"best_params_{study_name}.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                "best_f1": study.best_value,
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

    all_trials_file = os.path.join(RESULTS, f"all_trials_{study_name}.json")
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
