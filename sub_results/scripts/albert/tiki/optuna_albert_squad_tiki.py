# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for ALBERT + SQuAD with TikiTaka v1.

Usage:
    python optuna_albert_squad_tiki.py --n-trials 50
    python optuna_albert_squad_tiki.py --visualize
    python optuna_albert_squad_tiki.py --n-trials 50 --optimizer AnalogSGD --no-wd --no-momentum --no-nesterov --batch-size 64 --epochs 15 --lora-target attn

All flags:
    python optuna_albert_squad_tiki.py \
        --study-name <str>          # Study name (default: auto-generated)
        --n-trials <int>            # Number of Optuna trials (default: 50)
        --visualize                 # Visualize study results and exit
        --optimizer <str>           # AnalogSGD | AnalogAdam (default: AnalogAdam)
        --no-wd                     # Disable weight decay tuning (fix to 0)
        --no-momentum               # Disable momentum tuning (fix to 0, SGD only)
        --no-nesterov               # Disable nesterov tuning (fix to False, SGD only)
        --batch-size <int>          # Batch size (default: 48)
        --epochs <int>              # Number of epochs (default: 5)
        --warmup-ratio <float>      # LR warmup ratio (default: 0.05)
        --lora-target <str>         # Target: none | attn | ffn | all (default: attn)
        --head-layer <str>          # qa_outputs: train | freeze (default: train)
        --convert-nontarget         # Convert non-target layers to analog
        --no-convert-nontarget      # Keep non-target layers as digital (nn.Linear, frozen) (default: off)
        --fix-wd <float>            # Fix weight_decay to a specific value
        --fix-flr <float...>        # Grid fast_lr values (default: 1.0 5.0)
        --fix-tlr <float...>        # Grid transfer_lr values
        --sampler grid|tpe          # Sampler type (default: grid)
        --tpe-flr-range <lo> <hi>   # TPE fast_lr range
        --tpe-tlr-range <lo> <hi>   # TPE transfer_lr range
        --enqueue <TE> <FLR> [TLR]  # Enqueue seed trial

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
import re
import string
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
from optuna.samplers import GridSampler, TPESampler
import matplotlib.pyplot as plt

from transformers import (
    AutoConfig,
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
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs import SingleRPUConfig, UnitCellRPUConfig, IOParameters, UpdateParameters
from aihwkit.simulator.configs.compounds import TransferCompound, ChoppedTransferCompound
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType

from collections import Counter


# =============================================================================
# Global Constants
# =============================================================================

DEFAULT_STUDY_NAME = "albert_squad_tiki_main"

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = "/data/results/tikitakav1"
os.makedirs(RESULTS, exist_ok=True)

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "albert/albert-base-v2"
MAX_SEQ_LENGTH = 384
NUM_HIDDEN_LAYERS = None  # None = use default (12); set to 1 for fast sweep

# Training defaults
N_EPOCHS = 5
BATCH_SIZE = 48
EVAL_BATCH_SIZE = 256

# Scheduler
WARMUP_RATIO = 0.05  # 5% warmup (default)

# Target options: which layers to convert to analog
# NOTE: ALBERT uses weight sharing — all 12 transformer blocks share the same
# parameters. The counts below are UNIQUE Linear layers, not per-block.
#   - Target layers -> TikiTaka v1 (TransferCompound: A + B tiles)
#   - Non-target encoder layers -> Single RPU (SoftBoundsDevice, same as B tile)
#   - qa_outputs, embedding_hidden_mapping_in -> Digital (not converted)
LORA_TARGET = "attn"  # default, can be set via --lora-target
HEAD_LAYER = "train"  # default, can be set via --head-layer (train | freeze)
CONVERT_NONTARGET = False  # default off (non-target encoder layers stay digital, frozen)
FREEZE_TARGET = False  # If True, target layers use SingleRPU (frozen) instead of TikiTaka
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

# Fixed LR (SQuAD default)
FIXED_LR = 1.58e-3

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

    if BACKWARD_PERFECT:
        suffix += "_bwdperf"

    if FREEZE_TARGET:
        suffix += "_fztgt"

    # Add lora target (always include for clarity)
    suffix += f"_{LORA_TARGET}"

    # Add head_layer if frozen (not default)
    if HEAD_LAYER == "freeze":
        suffix += "_headfreeze"

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
            scale_transfer_lr=False,     # absolute transfer_lr (not scaled by optimizer lr)
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
    """Create ALBERT QA model with selective analog layers.

    Architecture:
        - Target encoder layers (--lora-target) -> TikiTaka v1 (TransferCompound)
        - Non-target encoder layers -> Single RPU (frozen) if CONVERT_NONTARGET
                                    -> Digital (nn.Linear, frozen) if not CONVERT_NONTARGET
        - qa_outputs -> Digital TRAINABLE (based on HEAD_LAYER)
        - embedding_hidden_mapping_in -> Digital FROZEN
        - Embeddings -> Digital FROZEN

    ALBERT uses weight sharing: all transformer blocks share the same
    parameters, so the actual number of unique analog layers is small.
    """
    from aihwkit.nn import AnalogLinear

    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    if NUM_HIDDEN_LAYERS is not None:
        model_config.num_hidden_layers = NUM_HIDDEN_LAYERS
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, config=model_config)

    # Reinitialize qa_outputs with FIXED seed for reproducibility
    if hasattr(model, 'qa_outputs'):
        torch.manual_seed(SEED)
        nn.init.normal_(model.qa_outputs.weight, mean=0.0, std=0.02)
        if model.qa_outputs.bias is not None:
            nn.init.zeros_(model.qa_outputs.bias)
        print(f"  [FIX] Reinitialized qa_outputs with FIXED seed={SEED}")

    # Get target patterns for TikiTaka conversion
    target_patterns = get_target_module_names(LORA_TARGET)

    # Always exclude from any analog conversion
    always_digital = ["qa_outputs", "albert.encoder.embedding_hidden_mapping_in"]

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

    # --- Pass 1: Convert target layers to TikiTaka or SingleRPU (frozen) ---
    tikitaka_count = 0
    if tikitaka_layers:
        if FREEZE_TARGET:
            # Target layers → SingleRPU (frozen analog, same as nontarget)
            target_config = create_single_rpu_config()
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
    single_rpu_count = 0
    if CONVERT_NONTARGET and non_target_encoder_layers:
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

    # Freeze target layers too if --freeze-target
    if FREEZE_TARGET and tikitaka_layers:
        def _frozen_noop_update_target(x_input, d_input, *args, **kwargs):
            return None
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    tile.update = _frozen_noop_update_target

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  TikiTaka layers: {tikitaka_count}, Frozen analog layers: {single_rpu_count}, "
          f"Total analog: {tikitaka_count + single_rpu_count}, Total params: {total_params:,}")

    # Set requires_grad: AnalogContext, qa_outputs, LayerNorm are trainable
    # out_scaling follows learn_out_scaling in RPU config
    from aihwkit.optim.context import AnalogContext
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            param.requires_grad = True  # required for analog tile update
        elif "qa_outputs" in name:
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

def objective(trial, train_loader, eval_features, eval_examples, tokenizer):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Learning rate: fixed or sweep
    if OPT_CONFIG.get('fix_lr', False):
        learning_rate = FIXED_LR
    else:
        learning_rate = trial.suggest_float('learning_rate', FIXED_LR / 10, FIXED_LR * 10, log=True)

    # No TikiTaka → skip transfer param sweep
    _no_tiki = (LORA_TARGET == "none" or FREEZE_TARGET)

    # desired_bl: grid categorical or fixed
    if _no_tiki:
        desired_bl = OPT_CONFIG.get('desired_bl', 1)
    else:
        _bl_range = OPT_CONFIG.get('bl_sweep', None)
        if _bl_range is not None:
            desired_bl = trial.suggest_categorical('desired_bl', _bl_range)
        else:
            desired_bl = OPT_CONFIG.get('desired_bl', 1)

    # transfer_every: fixed at 1 (uim=True → every mini-batch)
    transfer_every = OPT_CONFIG.get('transfer_every_override', 1)

    # fast_lr: fixed at 1.0
    fast_lr = 1.0

    # transfer_lr: log sweep [0.01, 1.0] (skip if no TikiTaka)
    if _no_tiki:
        transfer_lr = 1.0
    else:
        transfer_lr = trial.suggest_float('transfer_lr', 0.01, 1.0, log=True)
    min_lr_rate = 0.0

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
    print(f"Trial {trial.number} Starting")
    print(f"{'='*70}")
    print(f"  transfer_every={transfer_every}, transfer_lr={transfer_lr:.4e}, fast_lr={fast_lr:.4e}, desired_bl={desired_bl}")
    print(f"  lr={learning_rate:.2e}, wd={weight_decay:.2e}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, optimizer={optimizer_name}")
    print(f"  min_lr_rate={min_lr_rate:.4f}, warmup_ratio={WARMUP_RATIO}")
    print(f"{'='*70}")

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
                                       patience=2, warmup_epochs=1)
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
                start_positions = batch['start_positions'].to(DEVICE)
                end_positions = batch['end_positions'].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    start_positions=start_positions, end_positions=end_positions,
                )
                loss = outputs.loss
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

            eval_f1, eval_em = evaluate_model(model, eval_features, eval_examples, tokenizer)

            improved = " *" if eval_f1 > early_stop.best_metric + early_stop.min_delta else ""
            should_stop = early_stop.step(epoch, eval_f1, 0.0, train_loss)

            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"[Trial {trial.number}] Epoch {epoch:3d} | "
                  f"F1: {eval_f1:6.2f}% | EM: {eval_em:6.2f}% | Best F1: {early_stop.best_metric:6.2f}% | "
                  f"Loss: {train_loss:.4f} | LR: {current_lr:.2e} | "
                  f"{early_stop.status_str()}{improved}")

            trial.report(eval_f1, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

            if should_stop:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}: "
                           f"{early_stop.stop_reason}")
                break

            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        print(f"\n[Trial {trial.number}] Finished - Best F1: {early_stop.best_metric:.2f}%")
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
    global BATCH_SIZE, N_EPOCHS, WARMUP_RATIO, LORA_TARGET, HEAD_LAYER, CONVERT_NONTARGET, FREEZE_TARGET, BACKWARD_PERFECT, FIXED_LR

    parser = argparse.ArgumentParser(description="Optuna sweep for ALBERT SQuAD TikiTaka v1")
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
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS,
                        help=f'Number of epochs (default: {N_EPOCHS})')
    parser.add_argument('--warmup-ratio', type=float, default=WARMUP_RATIO,
                        help=f'LR warmup ratio (default: {WARMUP_RATIO})')
    parser.add_argument('--lr', type=float, default=FIXED_LR,
                        help=f'Fixed learning rate (default: {FIXED_LR})')
    parser.add_argument('--fix-wd', type=float, default=None,
                        help='Fix weight_decay value. e.g. --fix-wd 0.01')
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=['none', 'attn', 'ffn', 'all'],
                        help='Target: none, attn, ffn, all (default: attn)')
    parser.add_argument('--head-layer', type=str, default=HEAD_LAYER,
                        choices=['train', 'freeze'],
                        help='qa_outputs layer: train or freeze (default: train)')
    parser.add_argument('--convert-nontarget', action='store_true', default=False,
                        help='Convert non-target layers to analog (SingleRPU+SoftBounds, frozen)')
    parser.add_argument('--no-convert-nontarget', dest='convert_nontarget', action='store_false',
                        help='Keep non-target layers as digital (nn.Linear, frozen) [default]')
    parser.add_argument('--backward-perfect', action='store_true', default=False,
                        help='Use perfect backward pass (no DAC/ADC quantization on gradients)')
    parser.add_argument('--freeze-target', action='store_true', default=False,
                        help='Freeze target layers (use SingleRPU instead of TikiTaka, analog forward only)')
    parser.add_argument('--fix-lr', action='store_true', default=False,
                        help='Use fixed LR (no sweep). Uses --lr value directly.')
    parser.add_argument('--fix-flr', type=float, nargs='+', default=None,
                        help='Fix fast_lr grid to these values (e.g. --fix-flr 1.0)')
    parser.add_argument('--fix-tlr', type=float, nargs='+', default=None,
                        help='Grid transfer_lr values (e.g. --fix-tlr 0.1 1.0). If not set, transfer_lr=1.0 fixed.')
    parser.add_argument('--sampler', type=str, default='tpe', choices=['grid', 'tpe'],
                        help='Sampler type: grid (GridSampler) or tpe (TPESampler, default: tpe)')
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
    parser.add_argument('--bl-sweep', type=int, nargs='+', default=[31, 60],
                        help='Grid values for desired_bl categorical sweep (default: [31, 60])')
    parser.add_argument('--use-v2', action='store_true', default=True,
                        help='Use TikiTaka v2 (ChoppedTransfer with buffer+chopper, bl=1)')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of hidden layers (default: 12). Set to 1 for fast sweep.')
    args = parser.parse_args()

    # Update global config
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    WARMUP_RATIO = args.warmup_ratio
    LORA_TARGET = args.lora_target
    HEAD_LAYER = args.head_layer
    CONVERT_NONTARGET = args.convert_nontarget
    BACKWARD_PERFECT = args.backward_perfect
    FREEZE_TARGET = args.freeze_target
    OPT_CONFIG['fix_lr'] = args.fix_lr
    FIXED_LR = args.lr
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
    OPT_CONFIG['use_v2'] = args.use_v2

    global NUM_HIDDEN_LAYERS
    NUM_HIDDEN_LAYERS = args.num_layers

    # Auto-generate study name based on config (includes batch size)
    study_name = args.study_name or f"albert_squad_tiki_bs{BATCH_SIZE}_{get_study_name_suffix()}"

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

    global GRID_FLR, GRID_TLR, SAMPLER_TYPE, TPE_FLR_RANGE, TPE_TLR_RANGE
    SAMPLER_TYPE = args.sampler

    if args.tpe_flr_range:
        TPE_FLR_RANGE = tuple(args.tpe_flr_range)
    if args.tpe_tlr_range:
        TPE_TLR_RANGE = tuple(args.tpe_tlr_range)

    # Grid transfer_lr
    if args.fix_tlr:
        GRID_TLR = args.fix_tlr

    print(f"  lr={FIXED_LR} (fixed)")
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
            n_warmup_steps=1,     # don't prune until after epoch 1
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
            lambda trial: objective(trial, train_loader, eval_features, eval_examples, tokenizer),
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
    all_trials.sort(key=lambda x: x["value"] if x["value"] is not None else -float('inf'), reverse=True)

    all_trials_file = os.path.join(RESULTS, "all_trials_squad.json")
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
