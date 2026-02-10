# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for MobileBERT + SQuAD with LRTT.

Usage:
    python optuna_mobilebert_squad_lrtt.py --n-trials 50
    python optuna_mobilebert_squad_lrtt.py --visualize
    python optuna_mobilebert_squad_lrtt.py --n-trials 50 --optimizer AnalogAdam --reinit-mode hybrid --no-wd --no-momentum --no-nesterov --batch-size 256 --epochs 15 --warmup-steps 0 --transfer-method set

All flags:
    python optuna_mobilebert_squad_lrtt.py \
        --study-name <str>          # Study name (default: auto-generated)
        --n-trials <int>            # Number of Optuna trials (default: 50)
        --visualize                 # Visualize study results and exit
        --optimizer <str>           # AnalogSGD | AnalogAdam (default: AnalogAdam)
        --no-wd                     # Disable weight decay tuning (fix to 0)
        --no-momentum               # Disable momentum tuning (fix to 0, SGD only)
        --no-nesterov               # Disable nesterov tuning (fix to False, SGD only)
        --reinit-mode <str>         # Fix reinit mode: standard | decay | hybrid (default: tune all)
        --batch-size <int>          # Batch size (default: 32)
        --epochs <int>              # Number of epochs (default: 3)
        --warmup-steps <int>        # LR warmup steps (default: 0)
        --transfer-method <str>     # Transfer method: onehot | direct | set (default: onehot)
        --ab-device <str>           # A/B tile device: 6t1c | fp (default: 6t1c)
        --no-io-noise               # Disable IO out_noise (resolution kept)

Inline flags (edit directly in script):
    DYNAMIC_TE = False              # Enable dynamic transfer every
    DYNAMIC_TE_POWER = 1.0          # Power for dynamic TE scaling
    TE_WARMUP_STEPS = 0            # Steps before reaching target TE
    TE_WARMUP_SCHEDULE = []         # Warmup TE schedule list
    REINIT_GAIN = 0.1               # Reinitialization gain
    DECAY_FACTOR = 1.0              # Decay factor for reinit
    TARGET_MODULES = [...]          # Modules to convert to analog
    TRAIN_SUBSET_SIZE = 10000       # Training data subset size
    EVAL_SUBSET_SIZE = 2000         # Evaluation data subset size
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
from optuna_integration import BoTorchSampler
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
# ConfigAwareBoTorchSampler
# =============================================================================

class ConfigAwareBoTorchSampler(BoTorchSampler):
    """BoTorchSampler that respects OPT_CONFIG for fixed parameters."""
    def sample_relative(self, study, trial, search_space):
        params = super().sample_relative(study, trial, search_space)
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

DEFAULT_STUDY_NAME = "mobilebert_squad_lrtt_main"

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = os.path.join(os.getcwd(), "results", "optuna_mobilebert_squad_lrtt")
os.makedirs(RESULTS, exist_ok=True)

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 384

# Training defaults
N_EPOCHS = 15
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 128
EARLY_STOP_PATIENCE = 3

# Scheduler
WARMUP_STEPS = 0

# Dynamic TE
DYNAMIC_TE = False
DYNAMIC_TE_POWER = 1.0
TE_WARMUP_STEPS = 0
TE_WARMUP_SCHEDULE = []

# Fixed LRTT parameters
REINIT_GAIN = 0.1
DECAY_FACTOR = 1.0
TRANSFER_METHOD = "onehot"  # "onehot", "direct", or "set"
AB_DEVICE = "6t1c"  # "6t1c" or "fp"
IO_NOISE = True  # If False, disable out_noise (resolution kept)

# Target modules
TARGET_MODULES = ["query", "key", "value"]

# Data subset sizes
TRAIN_SUBSET_SIZE = 10000
EVAL_SUBSET_SIZE = 2000

# Global config (set by argparse)
OPT_CONFIG = {
    'optimizer': 'AnalogAdam',
    'tune_wd': True,
    'tune_momentum': True,
    'tune_nesterov': True,
    'reinit_mode': None,    # None = tune, or 'standard'/'decay'/'hybrid' = fixed
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

    return suffix

os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# LRTT Device Functions
# =============================================================================

def _create_ab_device(tau_sec=46505.0):
    """Create A/B tile device based on AB_DEVICE setting.

    Options:
        6t1c - Full 6T1C with all noise/variation (realistic)
        fp   - FloatingPointDevice (perfect, no quantization/bounds)
    """
    if AB_DEVICE == "fp":
        return FloatingPointDevice()

    # Compute retention lifetime from tau_sec
    dt_batch_sec = 1.0
    delta = 1 - math.exp(-dt_batch_sec / tau_sec)
    lifetime = 1.0 / delta if delta > 0 else 0.0

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
        write_noise_std=0.0, #0.0182
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
        mult_noise=True,
    )


def create_lrtt_config(rank, transfer_every, transfer_lr, lora_alpha,
                        reinit_mode, tau_sec):
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
    device_config.forward_inject = False

    # Dynamic TE
    device_config.dynamic_te = DYNAMIC_TE
    device_config.dynamic_te_power = DYNAMIC_TE_POWER
    device_config.dynamic_te_max = te * 20
    device_config.te_warmup_schedule = TE_WARMUP_SCHEDULE + [te]
    device_config.te_warmup_steps = TE_WARMUP_STEPS

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    # Disable IO noise if requested (only out_noise; resolution kept)
    if not IO_NOISE:
        rpu_config.forward.out_noise = 0.0
        rpu_config.backward.out_noise = 0.0

    # Weight scaling for C tile
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


def create_model(params):
    """Create MobileBERT QA model with LRTT analog Q/K/V layers."""
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("qa_outputs")

    te = int(params["transfer_every"])
    rpu_config = create_lrtt_config(
        rank=int(params["rank"]),
        transfer_every=te,
        transfer_lr=params["transfer_lr"],
        lora_alpha=params["lora_alpha"],
        reinit_mode=params["reinit_mode"],
        tau_sec=params["tau_sec"],
    )

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    # Count analog layers
    from aihwkit.nn import AnalogLinear
    analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Analog layers: {analog_count}, Total params: {total_params:,}, Trainable (before grad set): {trainable_params:,}")

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        param.requires_grad = is_target or "qa_outputs" in name

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable (after grad set): {trainable_after:,}")

    return model.to(DEVICE)


# =============================================================================
# Data Functions
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize SQuAD dataset."""
    raw_datasets = load_dataset("squad")

    eval_examples = raw_datasets["validation"].select(
        range(min(EVAL_SUBSET_SIZE, len(raw_datasets["validation"])))
    )

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
    train_subset = tokenized_train.shuffle(seed=SEED).select(
        range(min(TRAIN_SUBSET_SIZE, len(tokenized_train)))
    )

    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=default_data_collator
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

    # Hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e0, log=True)  #trial.suggest_float('learning_rate', 1e-5, 1e0, log=True)
    transfer_lr = trial.suggest_float('transfer_lr', 1e-2, 1e2, log=True)      #trial.suggest_float('transfer_lr', 1e-3, 1e0, log=True)
    transfer_every = trial.suggest_int('transfer_every', 10, 6400, log=True)   #trial.suggest_int('transfer_every', 1, 300000, log=True)
    rank_exp = trial.suggest_int('rank_exp', 0, 6)
    rank = 2 ** rank_exp
    lora_alpha = trial.suggest_float('lora_alpha', 1e-3, 1e3, log=True)
    tau_sec = trial.suggest_float('tau_sec', 1e0, 1e9, log=True)
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
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}")
    print(f"  tau_sec={tau_sec:.1f}, reinit_mode={reinit_mode}, optimizer={optimizer_name}")
    print(f"  min_lr_rate={min_lr_rate:.4f}")
    print(f"{'='*70}")

    model = None
    try:
        set_seed(SEED)

        model = create_model(params)

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

        best_f1 = 0.0
        epochs_without_improvement = 0

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            eval_f1, eval_em = evaluate_model(model, eval_features, eval_examples, tokenizer)

            improved = ""
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                epochs_without_improvement = 0
                improved = " ★"
            else:
                epochs_without_improvement += 1

            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Trial {trial.number}] Epoch {epoch:3d} | "
                  f"F1: {eval_f1:6.2f}% | EM: {eval_em:6.2f}% | Best F1: {best_f1:6.2f}% | "
                  f"Loss: {train_loss:.4f} | LR: {current_lr:.2e} | "
                  f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}{improved}")

            trial.report(best_f1, epoch)

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break

            if trial.should_prune():
                print(f"[Trial {trial.number}] Pruned at epoch {epoch}")
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
        f1_scores = [t.value for t in complete_trials]
        print(f"Best F1: {max(f1_scores):.2f}%, Mean F1: {sum(f1_scores)/len(f1_scores):.2f}%")
        print(f"Best params: {study.best_params}")


# =============================================================================
# Main
# =============================================================================

def main():
    global BATCH_SIZE, N_EPOCHS, WARMUP_STEPS, TRANSFER_METHOD, AB_DEVICE, IO_NOISE

    parser = argparse.ArgumentParser(description="Optuna sweep for MobileBERT SQuAD LRTT")
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
    args = parser.parse_args()

    # Update global config
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    WARMUP_STEPS = args.warmup_steps
    TRANSFER_METHOD = args.transfer_method
    AB_DEVICE = args.ab_device
    IO_NOISE = not args.no_io_noise
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['reinit_mode'] = args.reinit_mode
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov

    # Auto-generate study name based on config (includes batch size)
    study_name = args.study_name or f"mobilebert_squad_lrtt_bs{BATCH_SIZE}_{get_study_name_suffix()}"

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

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=ConfigAwareBoTorchSampler(), pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

    # Run trials with OOM recovery via process restart
    target_total = len(study.trials) + args.n_trials
    completed_before = len(study.trials)

    try:
        study.optimize(
            lambda trial: objective(trial, train_loader, eval_features, eval_examples, tokenizer),
            n_trials=args.n_trials,
            catch=(Exception,),
            show_progress_bar=True,
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
