# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for MobileBERT + SQuAD with TikiTaka v1.

Usage:
    python optuna_mobilebert_squad_tiki.py --n-trials 1
    python optuna_mobilebert_squad_tiki.py --visualize

All flags:
    python optuna_mobilebert_squad_tiki.py \
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
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs import SingleRPUConfig, UnitCellRPUConfig, IOParameters, UpdateParameters
from aihwkit.simulator.configs.compounds import TransferCompound, ChoppedTransferCompound
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType

from collections import Counter


# =============================================================================
# Global Constants
# =============================================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = "/data/results/tikitakav1"
os.makedirs(RESULTS, exist_ok=True)

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "google/mobilebert-uncased"

# SQuAD v1.1 settings from Albert_setup.txt (orig column)
BATCH_SIZE = 48
EVAL_BATCH_SIZE = 256
MAX_SEQ_LENGTH = 384
N_EPOCHS = 2          # orig column: ep=2
EARLY_STOP_PATIENCE = 2

# Scheduler
WARMUP_RATIO = 0.05   # matched with GLUE script

# Fixed LR (SQuAD default)
SQUAD_LR = 1e-3
TPE_FLR_RANGE = (0.01, 1.0)
TPE_TLR_RANGE = (0.01, 1.0)

# Target options
LORA_TARGET = "qkv"
HEAD_LAYER = "train"
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
    'optimizer': 'AnalogAdam',
    'tune_wd': False,
    'tune_momentum': False,
    'tune_nesterov': False,
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
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=use_v2,    # v1: False (absolute), v2: True (scale with optimizer lr)
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
    """Create Single RPU configuration for non-target frozen analog layers."""
    b_device = _create_b_device()

    rpu_config = SingleRPUConfig(device=b_device)

    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    if OPT_CONFIG.get('backward_perfect', False):
        rpu_config.backward.is_perfect = True

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
    """Classify MobileBERT encoder Linear layer."""
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
    """Create MobileBERT QA model with selective TikiTaka v1 analog layers."""
    from aihwkit.nn import AnalogLinear

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Reinitialize qa_outputs with FIXED seed for reproducibility
    if hasattr(model, 'qa_outputs'):
        torch.manual_seed(SEED)
        nn.init.normal_(model.qa_outputs.weight, mean=0.0, std=0.02)
        if model.qa_outputs.bias is not None:
            nn.init.zeros_(model.qa_outputs.bias)
        print(f"  [FIX] Reinitialized qa_outputs with FIXED seed={SEED}")

    # Always digital (never analog): qa_outputs + embedding_transformation
    always_digital = ["qa_outputs", "embedding_transformation"]

    def is_tikitaka_target(layer_name):
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

    tikitaka_layers = [n for n in all_linear_names if is_tikitaka_target(n)]
    non_target_encoder_layers = [
        n for n in all_linear_names
        if n not in tikitaka_layers and "encoder" in n
        and not any(d in n for d in always_digital)
    ]

    # --- Pass 1: Convert target layers to TikiTaka ---
    tikitaka_count = 0
    if tikitaka_layers and LORA_TARGET != "none":
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

    # --- Pass 2: Non-target encoder layers ---
    single_rpu_count = 0
    if non_target_encoder_layers and not OPT_CONFIG.get('nontarget_digital', False):
        single_config = create_single_rpu_config()
        single_exclude = [n for n in all_linear_names if n not in non_target_encoder_layers]
        model = convert_to_analog(model, single_config, exclude_modules=single_exclude)
        single_rpu_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear)) - tikitaka_count

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
            param.requires_grad = True
        elif "qa_outputs" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = False  # MobileBERT uses NoNorm - must freeze
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
    """Load and tokenize SQuAD dataset."""
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

    # Learning rate: sweep around fixed LR
    _lr_upper_mult = OPT_CONFIG.get('lr_upper_mult', 3)
    learning_rate = trial.suggest_float('learning_rate', SQUAD_LR / 10, SQUAD_LR * _lr_upper_mult, log=True)

    if TPE_FLR_RANGE[0] == TPE_FLR_RANGE[1]:
        fast_lr = TPE_FLR_RANGE[0]
    else:
        fast_lr = trial.suggest_float('fast_lr', TPE_FLR_RANGE[0], TPE_FLR_RANGE[1], log=True)

    if TPE_TLR_RANGE[0] == TPE_TLR_RANGE[1]:
        transfer_lr = TPE_TLR_RANGE[0]
    else:
        transfer_lr = trial.suggest_float('transfer_lr', TPE_TLR_RANGE[0], TPE_TLR_RANGE[1], log=True)

    # desired_bl: sweep or fixed
    _bl_range = OPT_CONFIG.get('bl_sweep', None)
    if _bl_range is not None:
        desired_bl = trial.suggest_int('desired_bl', _bl_range[0], _bl_range[1])
    else:
        desired_bl = OPT_CONFIG.get('desired_bl', 1)

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
    print(f"Trial {trial.number} Starting (squad, metric=F1)")
    print(f"{'='*70}")
    print(f"  transfer_every={transfer_every}, transfer_lr={transfer_lr:.4e}, fast_lr={fast_lr:.4e}")
    print(f"  lr={learning_rate:.2e}, wd={weight_decay:.2e}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, optimizer={optimizer_name}")
    print(f"  min_lr_rate={min_lr_rate:.4f}")
    print(f"{'='*70}")

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
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
        )

        best_f1 = 0.0
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
                start_positions = batch['start_positions'].to(DEVICE)
                end_positions = batch['end_positions'].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    start_positions=start_positions, end_positions=end_positions,
                )
                loss = outputs.loss
                loss.backward()

                # Digital-only grad clipping
                from aihwkit.optim.context import AnalogContext as _AC
                _digital_params = [p for p in model.parameters()
                                   if not isinstance(p, _AC) and p.grad is not None]
                if _digital_params:
                    torch.nn.utils.clip_grad_norm_(_digital_params, max_norm=1.0)

                # Per-tile analog grad clip+floor
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

                optimizer.step()
                scheduler.step()

                loss_val = loss.item()
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

            # Early stopping
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch} "
                          f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

            # Optuna pruner
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
    global BATCH_SIZE, N_EPOCHS, WARMUP_RATIO, LORA_TARGET, HEAD_LAYER, MAX_SEQ_LENGTH, SQUAD_LR

    parser = argparse.ArgumentParser(description="Optuna sweep for MobileBERT SQuAD TikiTaka v1")
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: auto-generated based on config)')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--optimizer', type=str, default='AnalogAdam',
                        choices=['AnalogSGD', 'AnalogAdam'],
                        help='Optimizer type (default: AnalogAdam)')
    parser.add_argument('--no-wd', action='store_true', default=True,
                        help='Disable weight decay tuning (fix to 0)')
    parser.add_argument('--tune-wd', dest='no_wd', action='store_false',
                        help='Enable weight decay tuning')
    parser.add_argument('--no-momentum', action='store_true',
                        help='Disable momentum tuning (fix to 0, SGD only)')
    parser.add_argument('--no-nesterov', action='store_true',
                        help='Disable nesterov tuning (fix to False, SGD only)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
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
    parser.add_argument('--learn-out-scaling', action='store_true', default=True,
                        help='Enable learn_out_scaling (default: True)')
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
    parser.add_argument('--lr', type=float, default=SQUAD_LR,
                        help=f'Fixed learning rate (default: {SQUAD_LR})')
    parser.add_argument('--auto-scale', action='store_true', default=False,
                        help='Enable auto_scale: dynamically normalise fast_lr by gradient magnitude')
    parser.add_argument('--transfer-every', type=int, default=None,
                        help='Override transfer_every (default: 1). Large value = effectively no transfer')
    parser.add_argument('--desired-bl', type=int, default=1,
                        help='Transfer update desired_bl (default: 1)')
    parser.add_argument('--bl-sweep', type=int, nargs=2, default=None,
                        help='Sweep desired_bl range [min max] (e.g. --bl-sweep 1 31)')
    parser.add_argument('--use-v2', action='store_true', default=False,
                        help='Use TikiTaka v2 (ChoppedTransfer with buffer+chopper, bl=1)')
    parser.add_argument('--lr-upper-mult', type=float, default=3.0,
                        help='LR upper bound multiplier (default: 3.0, e.g. 10 for 10x base_lr)')
    parser.add_argument('--tpe-flr-range', type=float, nargs=2, default=None,
                        help='TPE fast_lr range (e.g. --tpe-flr-range 1.0 1.0 to fix)')
    parser.add_argument('--tpe-tlr-range', type=float, nargs=2, default=None,
                        help='TPE transfer_lr range (e.g. --tpe-tlr-range 1.0 1.0 to fix)')
    args = parser.parse_args()

    # Update global config
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    WARMUP_RATIO = args.warmup_ratio
    LORA_TARGET = args.lora_target
    HEAD_LAYER = args.head_layer
    SQUAD_LR = args.lr
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
    OPT_CONFIG['desired_bl'] = args.desired_bl
    if args.bl_sweep is not None:
        OPT_CONFIG['bl_sweep'] = args.bl_sweep
    OPT_CONFIG['use_v2'] = args.use_v2
    OPT_CONFIG['lr_upper_mult'] = args.lr_upper_mult

    global TPE_FLR_RANGE, TPE_TLR_RANGE
    TPE_FLR_RANGE = tuple(args.tpe_flr_range) if args.tpe_flr_range else (0.01, 1.0)
    TPE_TLR_RANGE = tuple(args.tpe_tlr_range) if args.tpe_tlr_range else (0.01, 1.0)

    global CLIP_ANALOG_GRAD
    CLIP_ANALOG_GRAD = args.clip_analog_grad

    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d_%H%M")
    study_name = args.study_name or f"mobilebert_squad_tiki_{timestamp}"

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
    print(f"  lr={SQUAD_LR} (fixed)")
    print(f"  transfer_every=1 (fixed, uim=True, every mini-batch)")
    print(f"  fast_lr=1.0 (fixed), transfer_lr=1.0 (fixed)")

    sampler = optuna.samplers.TPESampler(seed=SEED)

    # Pruner: MedianPruner
    prune_warmup = max(1, N_EPOCHS // 3)
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=prune_warmup,
        ),
        load_if_exists=True,
    )
    print(f"  Early stop patience: {EARLY_STOP_PATIENCE}, "
          f"Pruner: Median, startup=5, warmup={prune_warmup}")

    print(f"\nStudy: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

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

    all_trials_file = os.path.join(RESULTS, "all_trials_squad.json")
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
