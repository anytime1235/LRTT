# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for ALBERT + GLUE/SQuAD with AIMC (PEFT LoRA + Analog Inference).

AIMC mode: TorchInferenceRPUConfig with PEFT LoRA (software LoRA on digital,
pretrained weights on analog hardware with PCM noise/quantization).

Searches: learning_rate, lora_r, lora_alpha, lora_dropout
Optional: output_noise_level, weight_noise_std tuning

Usage:
    # GLUE tasks
    python optuna_albert_glue_squad_aimc.py --task sst2 --n-trials 50
    python optuna_albert_glue_squad_aimc.py --task cola --n-trials 30

    # SQuAD
    python optuna_albert_glue_squad_aimc.py --task squad --n-trials 50

    # Visualize
    python optuna_albert_glue_squad_aimc.py --task sst2 --visualize

    # With drift evaluation (optional)
    python optuna_albert_glue_squad_aimc.py --task sst2 --n-trials 50 --enable-drift --drift-repeats 3

All flags:
    python optuna_albert_glue_squad_aimc.py \\
        --task <str>                # Task: cola|sst2|mrpc|qqp|mnli|qnli|rte|stsb|squad (default: sst2)
        --study-name <str>          # Study name (default: auto-generated)
        --n-trials <int>            # Number of Optuna trials (default: 50)
        --visualize                 # Visualize study results and exit
        --optimizer <str>           # AnalogSGD | AnalogAdam (default: AnalogAdam)
        --batch-size <int>          # Batch size (default: per-task)
        --epochs <int>              # Number of epochs (default: per-task)
        --warmup-ratio <float>      # LR warmup ratio (default: 0.05)
        --lora-target <str>         # LoRA target: qkv | attn | ffn | all (default: attn)
        --head-layer <str>          # Head layer: train | freeze (default: train)
        --output-noise <float>      # AIMC output noise level (default: 0.04)
        --weight-noise-std <float>  # AIMC weight modifier std_dev (default: 0.067)
        --pcm-model <str>           # PCM model: PCM_Gmax25 | none (default: PCM_Gmax25)
        --inp-res <int>             # Input resolution bits (default: 8, -1=infinite)
        --out-res <int>             # Output resolution bits (default: 8, -1=infinite)
        --enable-drift              # Enable post-training drift evaluation
        --drift-repeats <int>       # Drift evaluation repetitions (default: 3)
        --no-convert-nontarget      # Keep non-LoRA-target layers digital

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
import copy
import string
import math
import json
import argparse
import gc
import collections

import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
import evaluate as hf_evaluate

from peft import LoraConfig, get_peft_model

# aihwkit imports
from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import TorchInferenceRPUConfig
from aihwkit.simulator.parameters.enums import (
    WeightModifierType,
    WeightRemapType,
    WeightClipType,
)
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.optim import AnalogSGD, AnalogAdam

from collections import Counter


# =============================================================================
# Task Configurations (GLUE + SQuAD)
# =============================================================================

ALL_TASKS = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb", "squad"]
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
    "squad": 0,  # not used for QA
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
    "squad": "f1",
}

# Per-task configs
TASK_CONFIGS = {
    "cola":  {"batch_size": 16,  "epochs": 10, "max_seq_length": 128, "base_lr": 1.49e-3},
    "stsb":  {"batch_size": 16,  "epochs": 10, "max_seq_length": 128, "base_lr": 1.45e-3},
    "sst2":  {"batch_size": 32,  "epochs": 10, "max_seq_length": 128, "base_lr": 5.61e-4},
    "mnli":  {"batch_size": 128, "epochs": 4,  "max_seq_length": 128, "base_lr": 1e-3},
    "qnli":  {"batch_size": 32,  "epochs": 11, "max_seq_length": 128, "base_lr": 5.61e-4},
    "qqp":   {"batch_size": 128, "epochs": 5,  "max_seq_length": 128, "base_lr": 1e-3},
    "rte":   {"batch_size": 32,  "epochs": 11, "max_seq_length": 256, "base_lr": 6.78e-4},
    "mrpc":  {"batch_size": 32,  "epochs": 7,  "max_seq_length": 128, "base_lr": 1.08e-3},
    "squad": {"batch_size": 48,  "epochs": 2,  "max_seq_length": 384, "base_lr": 1.58e-3},
}

TASK_TO_ES_PATIENCE = {
    "rte": 3, "mrpc": 2, "stsb": 3, "cola": 3,
    "sst2": 3, "qnli": 3, "qqp": 2, "mnli": 2,
    "squad": 999,  # disabled for 2-epoch squad
}


# =============================================================================
# Global Constants
# =============================================================================

DEFAULT_STUDY_NAME = "albert_aimc_main"

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

TASK_NAME = "sst2"
RESULTS = None  # Set after argparse

SEED = 42
MODEL_NAME = "albert/albert-base-v2"
MAX_SEQ_LENGTH = 128

# Training defaults (overridden by TASK_CONFIGS per task)
N_EPOCHS = 10
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 5

# Scheduler
WARMUP_RATIO = 0.05

# AIMC hardware config
OUTPUT_NOISE_LEVEL = 0.04
WEIGHT_NOISE_STD = 0.067
PCM_MODEL = "PCM_Gmax25"
INP_RES_BITS = 8
OUT_RES_BITS = 8

# LoRA target options
LORA_TARGET = "attn"
HEAD_LAYER = "train"
CONVERT_NONTARGET = True

LORA_TARGET_MODULES_MAP = {
    "qkv": ["query", "key", "value"],
    "attn": ["query", "key", "value", "dense"],
    "ffn": ["ffn", "ffn_output"],
    "all": ["query", "key", "value", "dense", "ffn", "ffn_output"],
}

# Drift evaluation
ENABLE_DRIFT = False
DRIFT_REPEATS = 3
ALL_DRIFT_VALUES = [0, 3600, 86400, 604800, 2592000, 31536000, 315360000]
# 0s, 1hr, 1day, 1wk, 1mo, 1yr, 10yr

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

# Global config (set by argparse)
OPT_CONFIG = {
    'optimizer': 'AnalogAdam',
}

os.environ["WANDB_MODE"] = "offline"


def get_study_name_suffix():
    """Generate study name suffix based on config."""
    opt = OPT_CONFIG['optimizer'].lower().replace('analog', '')
    suffix = f"aimc_{opt}"
    suffix += f"_{LORA_TARGET}"
    if HEAD_LAYER == "freeze":
        suffix += "_headfreeze"
    if not CONVERT_NONTARGET:
        suffix += "_noconv"
    if PCM_MODEL == "none":
        suffix += "_nopcm"
    if ENABLE_DRIFT:
        suffix += "_drift"
    return suffix


# =============================================================================
# AIMC RPU Configuration
# =============================================================================

def gen_rpu_config(output_noise_level=None, weight_noise_std=None):
    """Create TorchInferenceRPUConfig for AIMC hardware simulation.

    Args:
        output_noise_level: Forward pass output noise (default: OUTPUT_NOISE_LEVEL)
        weight_noise_std: Weight modifier noise std_dev (default: WEIGHT_NOISE_STD)
    """
    if output_noise_level is None:
        output_noise_level = OUTPUT_NOISE_LEVEL
    if weight_noise_std is None:
        weight_noise_std = WEIGHT_NOISE_STD

    rpu_config = TorchInferenceRPUConfig()

    # Mapping
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    # Weight modifier (training-time noise injection)
    rpu_config.modifier.std_dev = weight_noise_std
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL

    # Weight remap
    rpu_config.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC

    # Forward pass (inference noise + quantization)
    rpu_config.forward = IOParameters()
    rpu_config.forward.out_noise = output_noise_level
    rpu_config.forward.is_perfect = False

    # DAC/ADC resolution
    if INP_RES_BITS > 0:
        rpu_config.forward.inp_res = 1 / (2**INP_RES_BITS - 2)
    else:
        rpu_config.forward.inp_res = -1  # infinite resolution
    if OUT_RES_BITS > 0:
        rpu_config.forward.out_res = 1 / (2**OUT_RES_BITS - 2)
    else:
        rpu_config.forward.out_res = -1

    # Weight clipping
    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 3

    # PCM noise model (for drift evaluation)
    if PCM_MODEL != "none":
        from aihwkit.inference import PCMLikeNoiseModel
        if PCM_MODEL == "PCM_Gmax25":
            rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
        else:
            raise ValueError(f"Unknown PCM model: {PCM_MODEL}")

    return rpu_config


# =============================================================================
# Layer Utility Functions
# =============================================================================

def list_analog_linear_layers(module, parent_name=''):
    """Recursively list all AnalogLinear layer names."""
    layers = []
    for name, child in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, AnalogLinear):
            layers.append(full_name)
        else:
            layers.extend(list_analog_linear_layers(child, full_name))
    return layers


def replace_layer(model, digital_model, layer_name):
    """Replace a layer in model with the corresponding layer from digital_model."""
    parts = layer_name.split('.')
    parent = model
    digital_parent = digital_model
    for p in parts[:-1]:
        parent = getattr(parent, p)
        digital_parent = getattr(digital_parent, p)
    setattr(parent, parts[-1], getattr(digital_parent, parts[-1]))


# =============================================================================
# Model Functions
# =============================================================================

def create_model(lora_r, lora_alpha, lora_dropout, output_noise_level=None, weight_noise_std=None):
    """Create ALBERT model with PEFT LoRA + AIMC analog conversion.

    Architecture:
        1. Load pretrained ALBERT
        2. Apply PEFT LoRA wrapping on target modules
        3. Convert entire model to analog (TorchInferenceRPUConfig)
        4. Replace LoRA adapter layers (lora_A, lora_B) back to digital
        5. Freeze all AnalogLinear parameters (pretrained weights on analog HW)
        6. LoRA adapter params + LayerNorm + head remain trainable
    """
    is_squad = (TASK_NAME == "squad")

    if is_squad:
        model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    else:
        num_labels = TASK_TO_NUM_LABELS[TASK_NAME]
        model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    # Reinitialize head with fixed seed for reproducibility
    head_name = "qa_outputs" if is_squad else "classifier"
    head = getattr(model, head_name, None)
    if head is not None and hasattr(head, 'weight'):
        torch.manual_seed(SEED)
        nn.init.normal_(head.weight, mean=0.0, std=0.02)
        if head.bias is not None:
            nn.init.zeros_(head.bias)

    # PEFT LoRA config
    target_modules = LORA_TARGET_MODULES_MAP.get(LORA_TARGET, ["query", "key", "value", "dense"])
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    print(f"  PEFT LoRA applied: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    model.print_trainable_parameters()

    # Save digital model reference for layer replacement
    digital_model = copy.deepcopy(model)

    # Convert entire model to analog
    rpu_config = gen_rpu_config(output_noise_level, weight_noise_std)
    model = convert_to_analog(model, rpu_config)

    # Find all AnalogLinear layers
    analog_layer_names = list_analog_linear_layers(model)

    # Replace LoRA adapter layers back to digital
    # Keep "base_layer" as AnalogLinear (pretrained weights on analog HW)
    # Replace everything else (lora_A, lora_B adapters) back to digital
    layers_to_replace = [
        name for name in analog_layer_names
        if "base_layer" not in name
    ]
    for layer_name in layers_to_replace:
        replace_layer(model, digital_model, layer_name)
    del digital_model

    n_analog = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  AnalogLinear layers (after replacement): {n_analog}")

    # Freeze all AnalogLinear parameters (pretrained weights on analog HW)
    for module in model.modules():
        if isinstance(module, AnalogLinear):
            for param in module.parameters():
                param.requires_grad = False

    # Ensure LoRA params, LayerNorm, and head are trainable
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
        elif head_name in name:
            param.requires_grad = (HEAD_LAYER == "train")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}, Trainable: {trainable:,} ({100*trainable/total_params:.2f}%)")

    return model.to(DEVICE)


# =============================================================================
# Data Functions - GLUE
# =============================================================================

def load_glue_data(tokenizer):
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

    remove_cols = [c for c in raw_datasets["train"].column_names if c != "label"]
    tokenized = raw_datasets.map(preprocess, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")

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

    print(f"  GLUE {TASK_NAME}: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
    return train_loader, eval_loader, None, None  # last two for SQuAD compat


# =============================================================================
# Data Functions - SQuAD
# =============================================================================

def load_squad_data(tokenizer):
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
        remove_columns=raw_datasets["train"].column_names,
    )
    if TRAIN_SUBSET_SIZE > 0:
        train_subset = tokenized_train.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(tokenized_train)))
        )
    else:
        train_subset = tokenized_train.shuffle(seed=SEED)

    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=default_data_collator,
        generator=torch.Generator().manual_seed(SEED),
    )

    print(f"  SQuAD: Train={len(train_subset)}, Eval features={len(tokenized_eval)}, Eval examples={len(eval_examples)}")
    return train_loader, None, tokenized_eval, eval_examples  # eval_loader=None for SQuAD


def load_data(tokenizer):
    """Load data based on task type."""
    if TASK_NAME == "squad":
        return load_squad_data(tokenizer)
    else:
        return load_glue_data(tokenizer)


# =============================================================================
# Evaluation Functions - GLUE
# =============================================================================

def evaluate_glue(model, eval_loader):
    """Evaluate GLUE model. Returns (metric_value, avg_loss)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with no_grad():
        for batch in eval_loader:
            model_inputs = {k: v.to(DEVICE) for k, v in batch.items()
                           if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
            outputs = model(**model_inputs)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * len(batch["labels"])

            if TASK_TO_NUM_LABELS[TASK_NAME] == 1:  # Regression (stsb)
                preds = logits.squeeze().cpu().numpy()
            else:
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else [preds])
            all_labels.extend(batch["labels"].cpu().tolist())

    model.train()

    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

    # Task-specific metric
    is_regression = (TASK_NAME == "stsb")
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
# Evaluation Functions - SQuAD
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


def postprocess_squad_predictions(examples, features, all_start_logits, all_end_logits,
                                   n_best_size=20, max_answer_length=30):
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
                    if (start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None):
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


def evaluate_squad(model, eval_features, eval_examples, tokenizer):
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
        collate_fn=squad_eval_collate_fn,
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
        n_best_size=20, max_answer_length=30,
    )

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    squad_metric = hf_evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)
    return results["f1"], results["exact_match"]


# =============================================================================
# Drift Evaluation (Optional)
# =============================================================================

def evaluate_with_drift(model, eval_loader, eval_features, eval_examples, tokenizer,
                        drift_values=None, n_repeats=3):
    """Run drift evaluation at multiple time points. Returns dict of results.

    Requires PCM noise model to be configured in RPU config.
    """
    if drift_values is None:
        drift_values = ALL_DRIFT_VALUES

    # Save checkpoint for drift evaluation
    checkpoint = copy.deepcopy(model.state_dict())
    is_squad = (TASK_NAME == "squad")
    metric_name = TASK_TO_METRIC[TASK_NAME]

    results = {}
    for drift_sec in drift_values:
        drift_metrics = []
        for rep in range(n_repeats):
            # Restore weights before each drift
            model.load_state_dict(checkpoint)
            if drift_sec > 0:
                model.drift_analog_weights(drift_sec)

            if is_squad:
                f1, em = evaluate_squad(model, eval_features, eval_examples, tokenizer)
                drift_metrics.append({"f1": f1, "em": em})
            else:
                metric_val, _ = evaluate_glue(model, eval_loader)
                drift_metrics.append({metric_name: metric_val})

        # Compute mean/std
        if is_squad:
            f1_vals = [m["f1"] for m in drift_metrics]
            em_vals = [m["em"] for m in drift_metrics]
            results[drift_sec] = {
                "f1_mean": np.mean(f1_vals), "f1_std": np.std(f1_vals),
                "em_mean": np.mean(em_vals), "em_std": np.std(em_vals),
            }
        else:
            vals = [m[metric_name] for m in drift_metrics]
            results[drift_sec] = {
                f"{metric_name}_mean": np.mean(vals),
                f"{metric_name}_std": np.std(vals),
            }

        drift_label = f"{drift_sec}s"
        if drift_sec >= 86400:
            drift_label = f"{drift_sec/86400:.0f}d"
        print(f"  Drift {drift_label}: {results[drift_sec]}")

    # Restore original weights
    model.load_state_dict(checkpoint)
    return results


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

def objective(trial, train_loader, eval_loader, eval_features, eval_examples, tokenizer):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    is_squad = (TASK_NAME == "squad")
    metric_name = TASK_TO_METRIC[TASK_NAME]

    # --- Hyperparameter search ---
    _base_lr = TASK_CONFIGS.get(TASK_NAME, {}).get("base_lr", 1e-3)
    learning_rate = trial.suggest_float('learning_rate', _base_lr / 10, _base_lr * 3, log=True)
    lora_r = trial.suggest_categorical('lora_r', [4, 8, 16, 32])
    lora_alpha_val = trial.suggest_int('lora_alpha', lora_r, lora_r * 4, step=lora_r)
    lora_dropout = trial.suggest_float('lora_dropout', 0.0, 0.3, step=0.05)

    optimizer_name = OPT_CONFIG['optimizer']
    min_lr_rate = 0.5

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting ({TASK_NAME}, metric={metric_name})")
    print(f"{'='*70}")
    print(f"  lr={learning_rate:.2e}, lora_r={lora_r}, lora_alpha={lora_alpha_val}")
    print(f"  lora_dropout={lora_dropout:.2f}, optimizer={optimizer_name}")
    print(f"  AIMC: out_noise={OUTPUT_NOISE_LEVEL}, weight_noise={WEIGHT_NOISE_STD}, pcm={PCM_MODEL}")
    print(f"{'='*70}")

    model = None
    try:
        set_seed(SEED)

        model = create_model(
            lora_r=lora_r,
            lora_alpha=lora_alpha_val,
            lora_dropout=lora_dropout,
        )

        # Optimizer
        if optimizer_name == "AnalogSGD":
            optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
        else:
            optimizer = AnalogAdam(model.parameters(), lr=learning_rate)

        num_training_steps = len(train_loader) * N_EPOCHS
        warmup_steps = int(WARMUP_RATIO * num_training_steps)
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
        )

        best_metric = -float('inf') if not is_squad else 0.0
        epochs_without_improvement = 0
        es_patience = TASK_TO_ES_PATIENCE.get(TASK_NAME, 3)

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
            for batch_idx, batch in enumerate(pbar):
                if is_squad:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    start_positions = batch['start_positions'].to(DEVICE)
                    end_positions = batch['end_positions'].to(DEVICE)

                    optimizer.zero_grad()
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        start_positions=start_positions, end_positions=end_positions,
                    )
                else:
                    model_inputs = {k: v.to(DEVICE) for k, v in batch.items()
                                   if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
                    optimizer.zero_grad()
                    outputs = model(**model_inputs)

                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                loss_val = loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    print(f"\n  [NaN/Inf detected at batch {num_batches}] Aborting trial.")
                    return 0.0
                total_loss += loss_val
                num_batches += 1
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            # Evaluation
            if is_squad:
                eval_f1, eval_em = evaluate_squad(model, eval_features, eval_examples, tokenizer)
                eval_metric = eval_f1
                metric_str = f"F1: {eval_f1:6.2f}% | EM: {eval_em:6.2f}%"
            else:
                eval_metric, eval_loss = evaluate_glue(model, eval_loader)
                metric_str = f"{metric_name}: {eval_metric:.4f}"

            improved = ""
            if eval_metric > best_metric:
                best_metric = eval_metric
                epochs_without_improvement = 0
                improved = " *"
            else:
                epochs_without_improvement += 1

            current_lr = scheduler.get_last_lr()[0]
            tqdm.write(f"[Trial {trial.number}] Epoch {epoch:3d} | "
                  f"{metric_str} | Best: {best_metric:.4f} | "
                  f"Loss: {train_loss:.4f} | LR: {current_lr:.2e} | "
                  f"No imp: {epochs_without_improvement}/{es_patience}{improved}")

            trial.report(best_metric, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

            # Abort hopeless SQuAD trials
            if is_squad and epoch == 1 and eval_f1 < 20.0:
                tqdm.write(f"[Trial {trial.number}] F1={eval_f1:.2f}% < 20% at epoch 1 -> abort")
                break

            if epochs_without_improvement >= es_patience:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break

            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        # Optional drift evaluation on best trial
        if ENABLE_DRIFT and best_metric > 0:
            print(f"\n  [Drift Evaluation] Running drift analysis...")
            drift_results = evaluate_with_drift(
                model, eval_loader, eval_features, eval_examples, tokenizer,
                drift_values=ALL_DRIFT_VALUES, n_repeats=DRIFT_REPEATS,
            )
            trial.set_user_attr("drift_results", {str(k): v for k, v in drift_results.items()})

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

    # Optimization history
    axes[0].scatter(trial_numbers, scores, alpha=0.6)
    axes[0].plot(trial_numbers,
                 [max(scores[:i+1]) for i in range(len(scores))],
                 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel(metric_name)
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

    # LR vs metric
    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[2].scatter(lrs, scores, alpha=0.6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel(metric_name)
    axes[2].set_title(f'Learning Rate vs {metric_name}')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"visualization_{TASK_NAME}_aimc.png"), dpi=150, bbox_inches='tight')
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
    global TASK_NAME, BATCH_SIZE, N_EPOCHS, MAX_SEQ_LENGTH, WARMUP_RATIO
    global LORA_TARGET, HEAD_LAYER, CONVERT_NONTARGET, RESULTS
    global OUTPUT_NOISE_LEVEL, WEIGHT_NOISE_STD, PCM_MODEL, INP_RES_BITS, OUT_RES_BITS
    global ENABLE_DRIFT, DRIFT_REPEATS, EVAL_BATCH_SIZE

    parser = argparse.ArgumentParser(description="Optuna sweep for ALBERT GLUE/SQuAD with AIMC")
    parser.add_argument('--task', type=str, default=TASK_NAME,
                        choices=ALL_TASKS,
                        help=f'Task (default: {TASK_NAME})')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: auto-generated)')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--visualize', action='store_true')

    # Training config
    parser.add_argument('--optimizer', type=str, default='AnalogAdam',
                        choices=['AnalogSGD', 'AnalogAdam'],
                        help='Optimizer (default: AnalogAdam)')
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Batch size (default: per-task from TASK_CONFIGS)')
    parser.add_argument('--eval-batch-size', type=int, default=EVAL_BATCH_SIZE,
                        help=f'Eval batch size (default: {EVAL_BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=0,
                        help='Number of epochs (default: per-task from TASK_CONFIGS)')
    parser.add_argument('--warmup-ratio', type=float, default=WARMUP_RATIO,
                        help=f'LR warmup ratio (default: {WARMUP_RATIO})')

    # LoRA config
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=list(LORA_TARGET_MODULES_MAP.keys()),
                        help=f'LoRA target modules (default: {LORA_TARGET})')
    parser.add_argument('--head-layer', type=str, default=HEAD_LAYER,
                        choices=['train', 'freeze'],
                        help=f'Head layer: train or freeze (default: {HEAD_LAYER})')

    # AIMC hardware config
    parser.add_argument('--output-noise', type=float, default=OUTPUT_NOISE_LEVEL,
                        help=f'AIMC output noise level (default: {OUTPUT_NOISE_LEVEL})')
    parser.add_argument('--weight-noise-std', type=float, default=WEIGHT_NOISE_STD,
                        help=f'AIMC weight noise std_dev (default: {WEIGHT_NOISE_STD})')
    parser.add_argument('--pcm-model', type=str, default=PCM_MODEL,
                        choices=['PCM_Gmax25', 'none'],
                        help=f'PCM noise model (default: {PCM_MODEL})')
    parser.add_argument('--inp-res', type=int, default=INP_RES_BITS,
                        help=f'Input DAC resolution bits (default: {INP_RES_BITS}, -1=infinite)')
    parser.add_argument('--out-res', type=int, default=OUT_RES_BITS,
                        help=f'Output ADC resolution bits (default: {OUT_RES_BITS}, -1=infinite)')

    # Drift evaluation (optional)
    parser.add_argument('--enable-drift', action='store_true',
                        help='Enable post-training drift evaluation per trial')
    parser.add_argument('--drift-repeats', type=int, default=DRIFT_REPEATS,
                        help=f'Drift evaluation repetitions (default: {DRIFT_REPEATS})')

    # Non-target conversion
    parser.add_argument('--convert-nontarget', action='store_true', default=True,
                        help='Convert all layers to analog (default: on)')
    parser.add_argument('--no-convert-nontarget', dest='convert_nontarget', action='store_false',
                        help='Keep non-LoRA-target layers digital')

    args = parser.parse_args()

    # Update global config
    TASK_NAME = args.task
    RESULTS = f"/data/results/aimc_optuna/{TASK_NAME}"
    os.makedirs(RESULTS, exist_ok=True)

    # Apply per-task defaults
    task_cfg = TASK_CONFIGS.get(TASK_NAME, {})
    BATCH_SIZE = args.batch_size if args.batch_size > 0 else task_cfg.get("batch_size", 32)
    N_EPOCHS = args.epochs if args.epochs > 0 else task_cfg.get("epochs", 10)
    MAX_SEQ_LENGTH = task_cfg.get("max_seq_length", 128)
    EVAL_BATCH_SIZE = args.eval_batch_size
    WARMUP_RATIO = args.warmup_ratio

    LORA_TARGET = args.lora_target
    HEAD_LAYER = args.head_layer
    CONVERT_NONTARGET = args.convert_nontarget
    OUTPUT_NOISE_LEVEL = args.output_noise
    WEIGHT_NOISE_STD = args.weight_noise_std
    PCM_MODEL = args.pcm_model
    INP_RES_BITS = args.inp_res
    OUT_RES_BITS = args.out_res
    ENABLE_DRIFT = args.enable_drift
    DRIFT_REPEATS = args.drift_repeats
    OPT_CONFIG['optimizer'] = args.optimizer

    # Auto-generate study name
    study_name = args.study_name or f"albert_{TASK_NAME}_bs{BATCH_SIZE}_{get_study_name_suffix()}"
    storage = f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    print(f"\n{'='*70}")
    print(f"AIMC Optuna Sweep: ALBERT + {TASK_NAME}")
    print(f"{'='*70}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  LoRA target: {LORA_TARGET} -> {LORA_TARGET_MODULES_MAP[LORA_TARGET]}")
    print(f"  AIMC: noise={OUTPUT_NOISE_LEVEL}, weight_noise={WEIGHT_NOISE_STD}, pcm={PCM_MODEL}")
    print(f"  Resolution: inp={INP_RES_BITS}bit, out={OUT_RES_BITS}bit")
    print(f"  Drift: {'enabled' if ENABLE_DRIFT else 'disabled'}")
    print(f"  Batch: {BATCH_SIZE}, Epochs: {N_EPOCHS}, Warmup: {WARMUP_RATIO}")
    print(f"{'='*70}\n")

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    # Load data once (shared across all trials)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader, eval_features, eval_examples = load_data(tokenizer)
    print(f"Task: {TASK_NAME}, Metric: {TASK_TO_METRIC[TASK_NAME]}")
    print(f"Train batches: {len(train_loader)}")

    # Create Optuna study
    if TASK_NAME == "squad":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=TPESampler(seed=SEED, n_startup_trials=10),
        pruner=pruner,
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

    # Run trials with OOM recovery
    target_total = len(study.trials) + args.n_trials

    try:
        study.optimize(
            lambda trial: objective(trial, train_loader, eval_loader, eval_features, eval_examples, tokenizer),
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
                "aimc_config": {
                    "output_noise": OUTPUT_NOISE_LEVEL,
                    "weight_noise_std": WEIGHT_NOISE_STD,
                    "pcm_model": PCM_MODEL,
                    "inp_res_bits": INP_RES_BITS,
                    "out_res_bits": OUT_RES_BITS,
                },
            }, f, indent=2)
        print(f"Best params saved to: {best_params_file}")

    # Save all trials
    all_trials = []
    for t in study.trials:
        trial_data = {
            "trial": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        }
        if "drift_results" in t.user_attrs:
            trial_data["drift_results"] = t.user_attrs["drift_results"]
        all_trials.append(trial_data)
    all_trials.sort(key=lambda x: x["value"] if x["value"] is not None else -float('inf'), reverse=True)

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
