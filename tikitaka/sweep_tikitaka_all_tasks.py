#!/usr/bin/env python
# coding=utf-8
"""TikiTaka v2 Bayesian Optimization for ALL tasks (GLUE + SQuAD).

Runs 10 trials x 1 epoch for each task to find optimal hyperparameters.
"""

import os
import sys
import json
import csv
import re
import string
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

import torch
import torch.nn as nn
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler
import wandb

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import collections

# aihwkit imports
sys.path.insert(0, '/data/LRTT_transformer/src')
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs import (
    UnitCellRPUConfig,
    IOParameters,
    UpdateParameters,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsReferenceDevice
from aihwkit.simulator.configs.utils import WeightClipType


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
QA_TASKS = ["squad"]  # 87K (uses 10K subset)
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

# SST-2 best params from previous search (Trial 17)
SST2_BEST_PARAMS = {
    "learning_rate": 0.000131,
    "transfer_lr": 7.36,
    "transfer_every": 160,
    "fast_lr": 0.86,
    "auto_granularity": 306.0,
    "in_chop_prob": 0.020,
}

# Default search ranges (wide)
DEFAULT_SEARCH_SPACE = {
    "lr_min": 1e-5, "lr_max": 1e-2,
    "transfer_lr_min": 0.1, "transfer_lr_max": 10.0,
    "transfer_every_min": 10, "transfer_every_max": 500,
    "fast_lr_min": 0.1, "fast_lr_max": 2.0,
    "auto_gran_min": 50, "auto_gran_max": 1000,
    "chop_prob_min": 0.0, "chop_prob_max": 0.1,
}

# Search space (user specified)
LR_MIN, LR_MAX = 5e-5, 5e-4
TRANSFER_LR_MIN, TRANSFER_LR_MAX = 0.1, 10.0
TRANSFER_EVERY_MIN, TRANSFER_EVERY_MAX = 10, 300
FAST_LR_MIN, FAST_LR_MAX = 0.1, 1.5
AUTO_GRAN_MIN, AUTO_GRAN_MAX = 100, 600
CHOP_PROB_MIN, CHOP_PROB_MAX = 0.005, 0.05

# =============================================================================
# Fixed Parameters
# =============================================================================

N_TRIALS = 10
NUM_EPOCHS = 1
TARGET_MODULES = ["query", "key", "value"]
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
WARMUP_STEPS = 500
SEED = 42

WANDB_PROJECT = "tikitaka-v2-all-tasks-sweep"
OUTPUT_DIR = "/data/AIMC_LoRA_results/tikitaka_sweep"


# =============================================================================
# TikiTaka v2 Config
# =============================================================================

def create_tikitaka_v2_config(
    transfer_every: int,
    transfer_lr: float,
    fast_lr: float,
    auto_granularity: float,
    in_chop_prob: float,
) -> UnitCellRPUConfig:
    """Create TikiTaka v2 config."""

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

    softbounds_device = SoftBoundsReferenceDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )

    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[sixt1c_device, softbounds_device],
            transfer_every=transfer_every,
            units_in_mbatch=False,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=True,
            auto_scale=True,
            auto_granularity=auto_granularity,
            buffer_granularity=1.0,
            auto_momentum=0.99,
            in_chop_prob=in_chop_prob,
            in_chop_random=True,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=1,
                update_bl_management=False,
                update_management=False,
            ),
        )
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

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("classifier")

    rpu_config = create_tikitaka_v2_config(
        transfer_every=params["transfer_every"],
        transfer_lr=params["transfer_lr"],
        fast_lr=params["fast_lr"],
        auto_granularity=params["auto_granularity"],
        in_chop_prob=params["in_chop_prob"],
    )

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        param.requires_grad = is_target or "classifier" in name

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
    eval_loader = DataLoader(tokenized[eval_key], batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=default_data_collator)

    return train_loader, eval_loader


def evaluate_glue(model, eval_loader, task_name, device) -> Tuple[float, float]:
    """Evaluate GLUE model."""
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
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
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0) if is_regression else 0

    model.train()

    if is_regression:
        from scipy.stats import spearmanr
        metric = spearmanr(all_preds, all_labels)[0]
    else:
        metric = correct / total if total > 0 else 0.0

    avg_loss = total_loss / max(total, len(all_labels)) if (total > 0 or all_labels) else 0.0
    return metric, avg_loss


# =============================================================================
# SQuAD Model & Data
# =============================================================================

def create_squad_model(params: Dict, device: torch.device) -> nn.Module:
    """Create SQuAD model with TikiTaka v2."""
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("qa_outputs")

    rpu_config = create_tikitaka_v2_config(
        transfer_every=params["transfer_every"],
        transfer_lr=params["transfer_lr"],
        fast_lr=params["fast_lr"],
        auto_granularity=params["auto_granularity"],
        in_chop_prob=params["in_chop_prob"],
    )

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        param.requires_grad = is_target or "qa_outputs" in name

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
            max_length=384, truncation="only_second",
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
            max_length=384, truncation="only_second",
            stride=128, return_overflowing_tokens=True,
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

    # Tokenize train data
    tokenized_train = raw_datasets["train"].map(
        preprocess_train, batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    train_subset = tokenized_train.shuffle(seed=SEED).select(range(min(10000, len(tokenized_train))))

    # Tokenize eval data (keep offset mapping for postprocessing)
    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

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


def evaluate_squad(model, eval_features, eval_examples, tokenizer, device) -> Tuple[float, float]:
    """
    Evaluate SQuAD model using official metric (same as digital baseline).
    Uses evaluate.load("squad") for F1/EM computation.
    """
    model.eval()

    # Collect all predictions
    all_start_logits = []
    all_end_logits = []

    # Custom collate function to handle offset_mapping with None values
    def squad_eval_collate_fn(features):
        # Separate non-tensor fields before collation
        offset_mappings = [f.pop("offset_mapping") for f in features]
        example_ids = [f.pop("example_id") for f in features]
        # Use default collator for tensor fields
        batch = default_data_collator(features)
        # Add back non-tensor fields (keep as lists, not tensors)
        batch["offset_mapping"] = offset_mappings
        batch["example_id"] = example_ids
        # Restore features for next iteration (features are mutated)
        for i, f in enumerate(features):
            f["offset_mapping"] = offset_mappings[i]
            f["example_id"] = example_ids[i]
        return batch

    eval_loader = DataLoader(eval_features, batch_size=BATCH_SIZE, shuffle=False, collate_fn=squad_eval_collate_fn)

    with torch.no_grad():
        for batch in eval_loader:
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

    # Use official SQuAD metric (same as digital baseline)
    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)

    return results["f1"], results["exact_match"]


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, optimizer, scheduler, train_loader, device, task_name, trial_num) -> float:
    """Train for one epoch with warmup scheduler."""
    model.train()
    total_loss, num_batches = 0.0, 0
    is_qa = task_name == "squad"
    is_regression = task_name == "stsb"

    criterion = None if is_qa else (nn.MSELoss() if is_regression else nn.CrossEntropyLoss())

    pbar = tqdm(train_loader, desc=f"Trial {trial_num} {task_name}", leave=False)
    for batch in pbar:
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

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / num_batches if num_batches > 0 else 0.0


# =============================================================================
# Optuna Trial
# =============================================================================

def run_trial(trial: optuna.Trial, task_name: str, train_loader, eval_loader,
              device: torch.device, results_dir: str,
              tokenizer=None, eval_examples=None, eval_features=None) -> float:
    """Run single trial for a task."""

    # Sample hyperparameters
    params = {
        "learning_rate": trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True),
        "transfer_lr": trial.suggest_float("transfer_lr", TRANSFER_LR_MIN, TRANSFER_LR_MAX),
        "transfer_every": trial.suggest_int("transfer_every", TRANSFER_EVERY_MIN, TRANSFER_EVERY_MAX, log=True),
        "fast_lr": trial.suggest_float("fast_lr", FAST_LR_MIN, FAST_LR_MAX),
        "auto_granularity": trial.suggest_float("auto_granularity", AUTO_GRAN_MIN, AUTO_GRAN_MAX, log=True),
        "in_chop_prob": trial.suggest_float("in_chop_prob", CHOP_PROB_MIN, CHOP_PROB_MAX),
    }

    # WandB logging
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{task_name}_trial_{trial.number}",
        config={"task": task_name, "trial": trial.number, **params},
        reinit=True,
    )

    try:
        set_seed(SEED)

        # Create model
        if task_name == "squad":
            model = create_squad_model(params, device)
        else:
            model = create_glue_model(task_name, params, device)

        optimizer = AnalogAdam(model.parameters(), lr=params["learning_rate"])
        optimizer.regroup_param_groups(model)  # Required for analog tiles to receive learning rate

        # Create warmup scheduler
        num_training_steps = len(train_loader) * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=num_training_steps
        )

        # Initial evaluation
        if task_name == "squad":
            init_metric, init_em = evaluate_squad(model, eval_features, eval_examples, tokenizer, device)
            init_loss = 0.0  # Not computed in F1 mode
        else:
            init_metric, init_loss = evaluate_glue(model, eval_loader, task_name, device)

        if task_name == "squad":
            wandb.log({"epoch": 0, "eval/f1": init_metric, "eval/em": init_em})
        else:
            wandb.log({"epoch": 0, "eval/metric": init_metric, "eval/loss": init_loss})

        # Train
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_epoch(model, optimizer, scheduler, train_loader, device, task_name, trial.number)

            if task_name == "squad":
                eval_metric, eval_em = evaluate_squad(model, eval_features, eval_examples, tokenizer, device)
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "eval/f1": eval_metric,
                    "eval/em": eval_em,
                })
            else:
                eval_metric, eval_loss = evaluate_glue(model, eval_loader, task_name, device)
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "eval/metric": eval_metric,
                    "eval/loss": eval_loss,
                })

        # Final metric
        final_metric = eval_metric
        wandb.log({"final/metric": final_metric, "final/improvement": final_metric - init_metric})

        del model
        torch.cuda.empty_cache()

        return final_metric

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        wandb.log({"error": str(e)})
        raise

    finally:
        wandb.finish()


# =============================================================================
# Main
# =============================================================================

def run_task_sweep(task_name: str, device: torch.device, results_dir: str, n_trials: int = 10, n_jobs: int = 1):
    """Run Bayesian optimization for a single task."""
    print(f"\n{'='*60}")
    print(f"Starting sweep for: {task_name.upper()} (n_jobs={n_jobs})")
    print(f"{'='*60}")

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    eval_examples = None  # Only used for SQuAD F1 computation
    eval_features = None  # Only used for SQuAD evaluation
    if task_name == "squad":
        train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
        eval_loader = None  # SQuAD uses eval_features directly
        print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")
    else:
        train_loader, eval_loader = load_glue_data(task_name, tokenizer)
        print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # Create study
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=f"tikitaka_{task_name}",
        direction="maximize",
        sampler=sampler,
    )

    # Enqueue SST-2 best params as first trial (good starting point)
    study.enqueue_trial(SST2_BEST_PARAMS)

    # Optimize
    def objective(trial):
        return run_trial(trial, task_name, train_loader, eval_loader, device, results_dir,
                        tokenizer=tokenizer, eval_examples=eval_examples, eval_features=eval_features)

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    # Save results
    task_results = {
        "task": task_name,
        "best_trial": study.best_trial.number,
        "best_metric": study.best_value,
        "best_params": study.best_params,
        "metric_name": TASK_TO_METRIC.get(task_name, "metric"),
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
    global NUM_EPOCHS  # Allow overriding from command line

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS,
                       help="Tasks to run (default: all)")
    parser.add_argument("--n_trials", type=int, default=10,
                       help="Number of trials per task")
    parser.add_argument("--n_jobs", type=int, default=1,
                       help="Number of parallel jobs per task")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                       help="Number of epochs per trial")
    args = parser.parse_args()

    n_trials = args.n_trials
    n_jobs = args.n_jobs
    NUM_EPOCHS = args.epochs  # Override global

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(OUTPUT_DIR, f"sweep_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("TikiTaka v2 Bayesian Optimization - All Tasks")
    print("="*60)
    print(f"Tasks: {args.tasks}")
    print(f"Trials per task: {n_trials}")
    print(f"Parallel jobs: {n_jobs}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Results dir: {results_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}

    for task in args.tasks:
        if task not in ALL_TASKS:
            print(f"Skipping unknown task: {task}")
            continue

        try:
            result = run_task_sweep(task, device, results_dir, n_trials=n_trials, n_jobs=n_jobs)
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
