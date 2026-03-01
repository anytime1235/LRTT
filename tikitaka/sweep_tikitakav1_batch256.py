#!/usr/bin/env python
# coding=utf-8
"""TikiTaka v1 Bayesian Optimization for ALL tasks (GLUE + SQuAD) — Batch 256.

TikiTaka v1 uses TransferCompound (no chopping, no auto-scale).
Tuned: lr, transfer_lr, fast_lr, transfer_every
"""

import os
import sys
import json
import re
import string
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter

import torch
import torch.nn as nn

import optuna
from optuna_integration import BoTorchSampler

from transformers import get_linear_schedule_with_warmup
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import collections
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef

# aihwkit imports
sys.path.insert(0, '/data/LRTT_transformer/src')
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    UnitCellRPUConfig,
    IOParameters,
    UpdateParameters,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.configs.compounds import TransferCompound
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice


# =============================================================================
# (Scheduler is created inline in run_trial, matching sixt1c scripts)
# =============================================================================


# =============================================================================
# SQuAD F1 Evaluation Helpers
# =============================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
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


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# =============================================================================
# Task Configurations
# =============================================================================

# Tasks ordered by dataset size (smallest first)
GLUE_TASKS = ["rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp", "mnli"]
#              2.5K   3.7K   7K     8.5K   67K    105K   364K   393K
QA_TASKS = ["squad"]  # 87K (full dataset)
ALL_TASKS = GLUE_TASKS + QA_TASKS

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
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
    "squad": "f1",
}

# =============================================================================
# Search Space
# =============================================================================

# =============================================================================
# Search Space (same as sweep_squad_15ep_bs256.py)
# =============================================================================
# Search params
FIXED_LR = None                  # Fixed analog learning rate (set None to tune)
LR_MIN, LR_MAX = 1e-4, 1e-2      # Analog lr search range (log scale)
FIXED_CLS_LR = 0.01             # Fixed classifier/head lr (set None to tune)
CLS_LR_GRID = [1.0, 0.1, 0.01]  # Classifier lr grid (categorical, used when FIXED_CLS_LR=None)
TRANSFER_LR_MIN, TRANSFER_LR_MAX = 0.2, 4.0
FAST_LR_MIN, FAST_LR_MAX = 0.1, 5.0
TRANSFER_EVERY_MIN, TRANSFER_EVERY_MAX = 1, 1000

# Warm start — based on v2 best params (adapted for v1)
WARM_START_PARAMS = {
    "learning_rate": 1e-2,
    "transfer_lr": 1.0,
    "fast_lr": 1.0,
    "transfer_every": 100,
}

# =============================================================================
# Fixed Parameters
# =============================================================================

N_TRIALS = 30
SQUAD_EPOCHS = 15             # SQuAD: 15 epochs (same as sixt1c_lora_squad)
GLUE_EPOCHS = 3               # GLUE: 3 epochs (same as sixt1c_lora_glue)
TARGET_MODULES = ["query", "key", "value"]
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128              # GLUE (same as sixt1c_lora_glue)
SQUAD_MAX_SEQ_LENGTH = 320        # SQuAD (same as sixt1c_lora_squad)
SQUAD_DOC_STRIDE = 128            # SQuAD (same as sixt1c_lora_squad)
BATCH_SIZE = 64                   # Train batch (same as sixt1c_lora)
SQUAD_EVAL_BATCH_SIZE = 256       # SQuAD eval (same as sixt1c_lora_squad)
GLUE_EVAL_BATCH_SIZE = 64         # GLUE eval (same as sixt1c_lora_glue)
WARMUP_RATIO = 0.05               # 5% warmup for all tasks
EARLY_STOP_PATIENCE = 3           # Early stopping patience (epochs without improvement)
SEED = 42

LOGGING_STEPS = 100
OUTPUT_DIR = "/data/LRTT_transformer/tikitaka/results"


# =============================================================================
# TikiTaka v1 Config
# =============================================================================

def create_tikitaka_v1_config(
    transfer_every: int,
    transfer_lr: float,
    fast_lr: float,
) -> UnitCellRPUConfig:
    """Create TikiTaka v1 config (TransferCompound, no chopping/auto-scale)."""

    sixt1c_device = LinearStepDevice(
        dw_min=0.001981,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=0.3,
        write_noise_std=0.0,
        mult_noise=True,
        mean_bound_reference=True,
        lifetime=0.0,
    )

    # B tile: SoftBoundsDevice (noise-free) — matches LRTT_setup.txt C tile
    softbounds_device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_std=0.0,
        dw_min_dtod=0.0,
        write_noise_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        mult_noise=False,
    )

    rpu_config = UnitCellRPUConfig(
        device=TransferCompound(
            unit_cell_devices=[sixt1c_device, softbounds_device],
            transfer_every=transfer_every,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=True,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            # transfer_update: use aihwkit default (desired_bl=31, management=True)
            transfer_update=UpdateParameters(),
        )
    )

    # Forward/Backward IO — matches LRTT_setup.txt
    rpu_config.forward = IOParameters(
        is_perfect=False,
        inp_noise=0.0,
        out_noise=0.0,
        out_noise_std=0.0,
        w_noise=0.0,
        noise_management=NoiseManagementType.ABS_MAX,
        bound_management=BoundManagementType.ITERATIVE,
        inp_res=1.0 / (2**7 - 2),   # 1/126 ≈ 0.00794
        out_res=1.0 / (2**9 - 2),   # 1/510 ≈ 0.00196
    )
    rpu_config.backward = IOParameters(
        is_perfect=False,
        inp_noise=0.0,
        out_noise=0.0,
        out_noise_std=0.0,
        w_noise=0.0,
        noise_management=NoiseManagementType.ABS_MAX,
        bound_management=BoundManagementType.ITERATIVE,
        inp_res=1.0 / (2**7 - 2),   # 1/126
        out_res=1.0 / (2**9 - 2),   # 1/510
    )

    # Update — matches LRTT_setup.txt
    rpu_config.update = UpdateParameters(
        desired_bl=31,
        update_bl_management=True,
        update_management=True,
    )

    # Mapping configuration (weight scaling + learnable output scaling)
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


def list_linear_layers(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


# =============================================================================
# GLUE Model & Data
# =============================================================================

def create_glue_model(task_name: str, params: Dict, device: torch.device) -> nn.Module:
    """Create GLUE model with TikiTaka v2."""
    num_labels = TASK_TO_NUM_LABELS[task_name]
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    # Reinitialize classifier with FIXED seed only if classifier is FROZEN (not in TARGET_MODULES)
    classifier_trainable = any("classifier" in t for t in TARGET_MODULES)
    if hasattr(model, 'classifier') and not classifier_trainable:
        torch.manual_seed(SEED)  # CRITICAL: Fix seed=42
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
        print(f"[FIX] Reinitialized classifier with FIXED seed={SEED} (frozen)")
    elif classifier_trainable:
        print(f"[INFO] Classifier is TRAINABLE (in TARGET_MODULES) — using pretrained init")

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("classifier")  # classifier always digital (no analog conversion)

    rpu_config = create_tikitaka_v1_config(
        transfer_every=params["transfer_every"],
        transfer_lr=params["transfer_lr"],
        fast_lr=params["fast_lr"],
    )

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        is_head = "classifier" in name
        if is_head:
            param.requires_grad = True  # classifier always trainable (weight + bias)
        elif "bias" in name:
            param.requires_grad = False
        else:
            param.requires_grad = is_target

    return model.to(device)


def load_glue_data(task_name: str, tokenizer):
    """Load and tokenize GLUE dataset."""
    raw_datasets = load_dataset("nyu-mll/glue", task_name)
    sentence1_key, sentence2_key = TASK_TO_KEYS[task_name]

    def preprocess(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], padding="max_length",
                           max_length=MAX_SEQ_LENGTH, truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key],
                        padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True)

    tokenized = raw_datasets.map(preprocess, batched=True)

    # Handle label column
    if task_name == "stsb":
        tokenized = tokenized.rename_column("label", "labels")
    else:
        tokenized = tokenized.rename_column("label", "labels")

    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=default_data_collator)

    eval_key = "validation_matched" if task_name == "mnli" else "validation"
    eval_loader = DataLoader(tokenized[eval_key], batch_size=GLUE_EVAL_BATCH_SIZE,
                            shuffle=False, collate_fn=default_data_collator)

    return train_loader, eval_loader


def evaluate_glue(model, eval_loader, task_name, device) -> Tuple[float, float]:
    """Evaluate GLUE model (matches sixt1c_lora_glue evaluation)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    is_regression = task_name == "stsb"
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

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

    if is_regression:
        metric = spearmanr(all_preds, all_labels)[0]
    elif task_name in ["mrpc", "qqp"]:
        metric = f1_score(all_labels, all_preds)
    elif task_name == "cola":
        metric = matthews_corrcoef(all_labels, all_preds)
    else:
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        metric = correct / n_samples if n_samples > 0 else 0.0

    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    return metric, avg_loss


# =============================================================================
# SQuAD Model & Data
# =============================================================================

def create_squad_model(params: Dict, device: torch.device) -> nn.Module:
    """Create SQuAD model with TikiTaka v2."""
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Reinitialize qa_outputs with FIXED seed only if frozen (not in TARGET_MODULES)
    qa_trainable = any("qa_outputs" in t for t in TARGET_MODULES)
    if hasattr(model, 'qa_outputs') and not qa_trainable:
        torch.manual_seed(SEED)  # CRITICAL: Fix seed=42
        nn.init.normal_(model.qa_outputs.weight, mean=0.0, std=0.02)
        if model.qa_outputs.bias is not None:
            nn.init.zeros_(model.qa_outputs.bias)
        print(f"[FIX] Reinitialized qa_outputs with FIXED seed={SEED} (frozen)")
    elif qa_trainable:
        print(f"[INFO] qa_outputs is TRAINABLE (in TARGET_MODULES) — using pretrained init")

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("qa_outputs")  # qa_outputs always digital (no analog conversion)

    rpu_config = create_tikitaka_v1_config(
        transfer_every=params["transfer_every"],
        transfer_lr=params["transfer_lr"],
        fast_lr=params["fast_lr"],
    )

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        is_head = "qa_outputs" in name
        if is_head:
            param.requires_grad = True  # qa_outputs always trainable (weight + bias)
        elif "bias" in name:
            param.requires_grad = False
        else:
            param.requires_grad = is_target

    return model.to(device)


def load_squad_data(tokenizer):
    """Load and tokenize SQuAD dataset (same as digital baseline)."""
    raw_datasets = load_dataset("squad")

    # Use subset for faster search
    eval_examples = raw_datasets["validation"].select(range(min(2000, len(raw_datasets["validation"]))))

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=SQUAD_MAX_SEQ_LENGTH, truncation="only_second",
            stride=SQUAD_DOC_STRIDE, return_overflowing_tokens=True,
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

            # Find context start/end
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # Check if answer is in context
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
        """Preprocess for evaluation - keep offset mapping for answer extraction."""
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=SQUAD_MAX_SEQ_LENGTH, truncation="only_second",
            stride=SQUAD_DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs["offset_mapping"]

        # Mark context tokens (sequence_id == 1) in offset_mapping
        # Non-context positions get None
        for i in range(len(inputs["input_ids"])):
            sequence_ids = inputs.sequence_ids(i)
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(offset_mapping[i])
            ]

        inputs["example_id"] = [examples["id"][sample_map[i]] for i in range(len(inputs["input_ids"]))]

        return inputs

    # Tokenize train data (full dataset, same as sixt1c_lora_squad)
    tokenized_train = raw_datasets["train"].map(
        preprocess_train, batched=True,
        remove_columns=raw_datasets["train"].column_names
    )

    # Tokenize eval data (keep offset mapping for postprocessing)
    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )

    train_loader = DataLoader(tokenized_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

    return train_loader, tokenized_eval, eval_examples


def postprocess_squad_predictions(
    examples,
    features,
    all_start_logits,
    all_end_logits,
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    """
    Post-process SQuAD predictions (same as digital baseline).
    Extracts best answer spans using n-best selection.
    """
    # Build mapping from example to features
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

            # Get n-best start and end indices
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip invalid indices
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Skip invalid spans
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    prelim_predictions.append({
                        "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                        "score": start_logits[start_index] + end_logits[end_index],
                        "start_logit": start_logits[start_index],
                        "end_logit": end_logits[end_index],
                    })

        # Sort by score and pick best
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        if len(predictions) == 0:
            all_predictions[example["id"]] = ""
        else:
            best_pred = predictions[0]
            start_char, end_char = best_pred["offsets"]
            all_predictions[example["id"]] = context[start_char:end_char]

    return all_predictions


def squad_eval_collate_fn(features):
    """Custom collate function to handle offset_mapping with None values."""
    offset_mappings = [f.pop("offset_mapping") for f in features]
    example_ids = [f.pop("example_id") for f in features]
    batch = default_data_collator(features)
    batch["offset_mapping"] = offset_mappings
    batch["example_id"] = example_ids
    for i, f in enumerate(features):
        f["offset_mapping"] = offset_mappings[i]
        f["example_id"] = example_ids[i]
    return batch


def evaluate_squad(model, eval_features, eval_examples, tokenizer, device,
                   squad_eval_loader=None, squad_metric=None) -> Tuple[float, float]:
    """Evaluate SQuAD model. Accepts pre-built eval_loader and metric for efficiency."""
    model.eval()

    all_start_logits = []
    all_end_logits = []

    if squad_eval_loader is None:
        squad_eval_loader = DataLoader(eval_features, batch_size=SQUAD_EVAL_BATCH_SIZE,
                                       shuffle=False, collate_fn=squad_eval_collate_fn)

    with torch.no_grad():
        for batch in squad_eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    model.train()

    # Concatenate all logits
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    # Postprocess predictions (same as digital baseline)
    predictions = postprocess_squad_predictions(
        eval_examples, eval_features,
        all_start_logits, all_end_logits,
        n_best_size=20, max_answer_length=30
    )

    # Format for official metric
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    if squad_metric is None:
        squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)

    return results["f1"], results["exact_match"]


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, optimizer, scheduler, train_loader, device, task_name,
                epoch_num, trial_num) -> float:
    """Train for one epoch."""
    model.train()
    total_loss, num_batches = 0.0, 0
    is_qa = task_name == "squad"
    is_regression = task_name == "stsb"
    n_total = len(train_loader)

    criterion = None if is_qa else (nn.MSELoss() if is_regression else nn.CrossEntropyLoss())

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        if is_qa:
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
        else:
            labels = batch['labels'].to(device)
            if is_regression:
                labels = labels.float()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze() if is_regression else outputs.logits
            loss = criterion(logits, labels)

        loss_val = loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss_val
        num_batches += 1

        if num_batches % LOGGING_STEPS == 0:
            avg = total_loss / num_batches
            print(f"  T{trial_num} {task_name} Ep{epoch_num} [{num_batches}/{n_total}] loss={loss_val:.4f} avg={avg:.4f}")

    return total_loss / num_batches if num_batches > 0 else 0.0


# =============================================================================
# Optuna Trial
# =============================================================================

def run_trial(trial: optuna.Trial, task_name: str, train_loader, eval_loader,
              device: torch.device, results_dir: str, num_epochs: int,
              tokenizer=None, eval_examples=None, eval_features=None,
              squad_eval_loader=None, squad_metric=None) -> float:
    """Run single trial with Optuna pruning."""

    params = {
        "learning_rate": FIXED_LR if FIXED_LR is not None else trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True),
        "cls_lr": FIXED_CLS_LR if FIXED_CLS_LR is not None else trial.suggest_categorical("cls_lr", CLS_LR_GRID),
        "transfer_lr": trial.suggest_float("transfer_lr", TRANSFER_LR_MIN, TRANSFER_LR_MAX, log=True),
        "fast_lr": trial.suggest_float("fast_lr", FAST_LR_MIN, FAST_LR_MAX, log=True),
        "transfer_every": trial.suggest_int("transfer_every", TRANSFER_EVERY_MIN, TRANSFER_EVERY_MAX),
    }

    print(f"\n--- Trial {trial.number} ---")
    print(f"  lr={params['learning_rate']:.6f} cls_lr={params['cls_lr']:.6f} "
          f"transfer_lr={params['transfer_lr']:.4f} "
          f"fast_lr={params['fast_lr']:.4f} transfer_every={params['transfer_every']}")

    try:
        set_seed(SEED)

        if task_name == "squad":
            model = create_squad_model(params, device)
            head_key = "qa_outputs"
        else:
            model = create_glue_model(task_name, params, device)
            head_key = "classifier"

        # Build param->name map before regroup
        param_to_name = {id(p): name for name, p in model.named_parameters()}

        optimizer = AnalogSGD(model.parameters(), lr=params["learning_rate"])
        optimizer.regroup_param_groups(model)

        # Split group 0: separate head (classifier/qa_outputs) params with cls_lr
        main_group = optimizer.param_groups[0]
        head_params = []
        other_params = []
        for p in main_group['params']:
            name = param_to_name.get(id(p), "")
            if head_key in name:
                head_params.append(p)
            else:
                other_params.append(p)

        if head_params:
            main_group['params'] = other_params
            optimizer.add_param_group({
                'params': head_params,
                'lr': params["cls_lr"],
            })

        num_training_steps = len(train_loader) * num_epochs
        warmup_steps = int(num_training_steps * WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        is_squad = (task_name == "squad")
        best_metric = -float('inf')
        epochs_without_improvement = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = train_epoch(
                model, optimizer, scheduler, train_loader, device,
                task_name, epoch, trial.number,
            )

            if is_squad:
                eval_metric, eval_em = evaluate_squad(
                    model, eval_features, eval_examples, tokenizer, device,
                    squad_eval_loader=squad_eval_loader, squad_metric=squad_metric)
                print(f"  [Epoch {epoch}/{num_epochs}] Loss: {train_loss:.4f}, "
                      f"F1: {eval_metric:.2f}, EM: {eval_em:.2f}")
            else:
                eval_metric, eval_loss = evaluate_glue(model, eval_loader, task_name, device)
                print(f"  [Epoch {epoch}/{num_epochs}] Loss: {train_loss:.4f}, "
                      f"Metric: {eval_metric:.4f}")

            if eval_metric > best_metric:
                best_metric = eval_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            trial.report(best_metric, epoch)

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

        print(f"  Trial {trial.number} done: best={best_metric:.4f}")

        del model
        torch.cuda.empty_cache()

        return best_metric

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise


# =============================================================================
# Main
# =============================================================================

def run_task_sweep(task_name: str, device: torch.device, results_dir: str,
                   n_trials: int = 10, num_epochs: int = 3):
    """Run Bayesian optimization for a single task."""
    print(f"\n{'='*60}")
    print(f"Starting sweep for: {task_name.upper()}")
    print(f"{'='*60}")
    print(f"  Scheduler: linear warmup + decay to 0")
    print(f"  Tuned: lr, transfer_lr, fast_lr, transfer_every")
    print(f"  B tile: SoftBoundsReferenceDevice (noise=0)")

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    eval_examples = None
    eval_features = None
    squad_eval_loader = None
    squad_metric_cached = None
    if task_name == "squad":
        train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
        eval_loader = None
        # Pre-build eval loader and metric (reused across all trials/epochs)
        squad_eval_loader = DataLoader(eval_features, batch_size=SQUAD_EVAL_BATCH_SIZE,
                                       shuffle=False, collate_fn=squad_eval_collate_fn)
        squad_metric_cached = evaluate.load("squad")
        print(f"  Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")
    else:
        train_loader, eval_loader = load_glue_data(task_name, tokenizer)
        print(f"  Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    db_path = os.path.join(results_dir, f"{task_name}_study.db")
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=f"tikitaka_v1_bs256_{task_name}",
        direction="maximize",
        sampler=BoTorchSampler(),
        pruner=optuna.pruners.NopPruner(),
        storage=storage,
        load_if_exists=True,
    )

    # Warm-start: enqueue only if study is fresh (no completed trials)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) == 0:
        study.enqueue_trial(WARM_START_PARAMS)
        print(f"  Fresh study — enqueued warm-start params")
    else:
        print(f"  Resuming study — {len(completed)} trials already completed")

    # Optimize (subtract already completed trials)
    remaining = max(0, n_trials - len(completed))
    print(f"  Running {remaining} trials (target={n_trials}, done={len(completed)})")

    def objective(trial):
        return run_trial(trial, task_name, train_loader, eval_loader, device, results_dir,
                        num_epochs=num_epochs,
                        tokenizer=tokenizer, eval_examples=eval_examples, eval_features=eval_features,
                        squad_eval_loader=squad_eval_loader, squad_metric=squad_metric_cached)

    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    else:
        print(f"  Already completed {len(completed)} >= {n_trials} trials, skipping.")

    # Save results
    task_results = {
        "tag": "Real",
        "description": "Validated experiment: ABS_MAX/ITERATIVE IO, out_noise=0.0, bias frozen, full dataset",
        "task": task_name,
        "best_trial": study.best_trial.number,
        "best_metric": study.best_value,
        "best_params": study.best_params,
        "fixed_params": {
            "noise_management": "ABS_MAX",
            "bound_management": "ITERATIVE",
            "out_noise": 0.0,
            "bias_frozen": True,
        },
        "metric_name": TASK_TO_METRIC.get(task_name, "metric"),
        "num_epochs": num_epochs,
        "batch_size": BATCH_SIZE,
        "scheduler": "linear_warmup_decay_to_0",
        "warmup_ratio": WARMUP_RATIO,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }

    results_file = os.path.join(results_dir, f"{task_name}_best_params.json")
    with open(results_file, 'w') as f:
        json.dump(task_results, f, indent=2)

    print(f"\n{task_name.upper()} Results:")
    print(f"  Best Trial: {study.best_trial.number}")
    print(f"  Best {TASK_TO_METRIC.get(task_name, 'metric')}: {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")
    print(f"  Saved to: {results_file}")

    return task_results


def main():
    parser = argparse.ArgumentParser(description="TikiTaka v1 Batch-256 GLUE+SQuAD Sweep")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                       help="Tasks to run (default: all)")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS,
                       help="Number of trials per task")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from existing results dir (e.g. Real_sweep_bias_frozen_20260210_170807)")
    parser.add_argument("--target_modules", nargs="+", default=None,
                       help="Override TARGET_MODULES (e.g. --target_modules query key value classifier)")
    parser.add_argument("--fixed_lr", type=float, default=None,
                       help="Override FIXED_LR (e.g. --fixed_lr 1.0). If not set, uses global FIXED_LR.")
    parser.add_argument("--fixed_cls_lr", type=float, default=None,
                       help="Override FIXED_CLS_LR (e.g. --fixed_cls_lr 0.01). If not set, uses global FIXED_CLS_LR.")
    args = parser.parse_args()

    # Override TARGET_MODULES if specified via CLI
    global TARGET_MODULES
    if args.target_modules is not None:
        TARGET_MODULES = args.target_modules
        print(f"[CLI] TARGET_MODULES overridden: {TARGET_MODULES}")

    # Override FIXED_LR if specified via CLI
    global FIXED_LR
    if args.fixed_lr is not None:
        FIXED_LR = args.fixed_lr
        print(f"[CLI] FIXED_LR overridden: {FIXED_LR}")

    # Override FIXED_CLS_LR if specified via CLI
    global FIXED_CLS_LR
    if args.fixed_cls_lr is not None:
        FIXED_CLS_LR = args.fixed_cls_lr
        print(f"[CLI] FIXED_CLS_LR overridden: {FIXED_CLS_LR}")

    n_trials = args.n_trials

    if args.resume:
        results_dir = os.path.join(OUTPUT_DIR, args.resume)
        if not os.path.exists(results_dir):
            print(f"Resume dir not found: {results_dir}")
            sys.exit(1)
        print(f"Resuming from: {results_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(OUTPUT_DIR, f"Real_sweep_ttv1_bias_frozen_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("TikiTaka v1 Batch-256 Sweep")
    print("="*60)
    print(f"Tasks: {args.tasks}")
    print(f"Trials per task: {n_trials}")
    print(f"Epochs: SQuAD={SQUAD_EPOCHS}, GLUE={GLUE_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Early stopping: patience={EARLY_STOP_PATIENCE} (no Optuna pruner)")
    print(f"Scheduler: linear warmup + decay to 0 (warmup: {WARMUP_RATIO:.0%} for all tasks)")
    print(f"Tuned: lr={'FIXED='+str(FIXED_LR) if FIXED_LR else f'[{LR_MIN}, {LR_MAX}]'}, "
          f"cls_lr={'FIXED='+str(FIXED_CLS_LR) if FIXED_CLS_LR else f'grid{CLS_LR_GRID}'}, "
          f"transfer_lr=[{TRANSFER_LR_MIN}, {TRANSFER_LR_MAX}], "
          f"fast_lr=[{FAST_LR_MIN}, {FAST_LR_MAX}], "
          f"transfer_every=[{TRANSFER_EVERY_MIN}, {TRANSFER_EVERY_MAX}]")
    print(f"B tile: SoftBoundsReferenceDevice (noise=0)")
    print(f"Results dir: {results_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}

    for task in args.tasks:
        if task not in ALL_TASKS:
            print(f"Skipping unknown task: {task}")
            continue

        num_epochs = SQUAD_EPOCHS if task == "squad" else GLUE_EPOCHS

        try:
            result = run_task_sweep(task, device, results_dir,
                                   n_trials=n_trials, num_epochs=num_epochs)
            all_results[task] = result
        except Exception as e:
            print(f"Failed to complete {task}: {e}")
            all_results[task] = {"error": str(e)}

    # Save combined results
    summary_file = os.path.join(results_dir, "all_tasks_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("ALL TASKS COMPLETE")
    print("="*60)
    print(f"Summary saved to: {summary_file}")

    for task, result in all_results.items():
        if "error" in result:
            print(f"  {task}: FAILED - {result['error']}")
        else:
            print(f"  {task}: {result['best_metric']:.4f} (trial {result['best_trial']})")


if __name__ == "__main__":
    main()
