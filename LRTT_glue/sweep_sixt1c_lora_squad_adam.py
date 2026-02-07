#!/home/jovyan/work/ml/.venv310/bin/python
# coding=utf-8
"""Sixt1c-LoRA Bayesian Optimization for SQuAD - RANK 8 (AnalogAdam).

Sixt1c-LoRA mode: forward_inject=True, no transfer (transfer_every=1000000).
Search space: learning_rate, lora_alpha.
Supports different target modules: Q, K, V, QKV, all (excluding qa_outputs).
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

# aihwkit imports (use installed package first)
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

# LRTT config imports (direct imports to avoid __init__.py dependency issues)
sys.path.insert(0, '/home/jovyan/work/LRTT/src')
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


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

TASK_TO_METRIC = {"squad": "f1"}

# Target module configurations
TARGET_CONFIGS = {
    "Q": ["query"],
    "K": ["key"],
    "V": ["value"],
    "QKV": ["query", "key", "value"],
    "all": None,  # Will include all linear layers except qa_outputs
}

# =============================================================================
# Sixt1c-LoRA Search Space
# =============================================================================

# Search space for sixt1c_lora parameters
LR_MIN, LR_MAX = 1e-6, 1e-1                        # learning rate
LORA_ALPHA_MIN, LORA_ALPHA_MAX = 0.1, 10.0         # lora scaling factor

# Fixed parameters
RANK = 8
REINIT_GAIN = 0.1
TRANSFER_EVERY = 1000000  # No transfer (sixt1c_lora)

# Default config (starting point)
DEFAULT_PARAMS = {
    "learning_rate": 1e-4,
    "lora_alpha": 1.0,
}

# =============================================================================
# Fixed Parameters
# =============================================================================

N_TRIALS = 30
NUM_EPOCHS = 3
TARGET_MODULES = ["value"]  # Default, will be overridden by --target
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
WARMUP_STEPS = 500
SEED = 42

WANDB_PROJECT = "sixt1c-lora-squad-sgd-sweep"
OUTPUT_DIR = "/data/results/sixt1c_lora_sweep"

os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# Sixt1c-LoRA Config
# =============================================================================

def create_sixt1c_lora_config(
    rank: int,
    lora_alpha: float = 1.0,
    reinit_gain: float = 0.1,
):
    """Create Sixt1c-LoRA config (forward_inject=True, no transfer).

    Uses LRTT 3-tile structure but with:
    - forward_inject=True: y = C·x + α·A·(B·x)
    - transfer_every=1000000: Effectively no transfer

    Args:
        rank: LRTT rank dimension (similar to LoRA rank)
        lora_alpha: LoRA scaling factor α
        reinit_gain: Kaiming initialization gain for B matrix

    Returns:
        Configured PythonLRTTRPUConfig
    """
    import math

    # Calculate lifetime for 6T1C
    TAU_SEC = 46505.0
    dt_batch_sec = 1.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    # A/B tiles: LinearStepDevice (6T1C)
    ab_device = LinearStepDevice(
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

    # C tile: SoftBoundsDevice (noise-free)
    c_device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=3.0,
        w_min=-3.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )

    # Sixt1c-LoRA Device config
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=TRANSFER_EVERY,  # No transfer
        lora_alpha=lora_alpha,
        reinit_gain=reinit_gain,
        reinit_mode="hybrid",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.001  # Not used (placeholder)
    device_config.units_in_mbatch = True
    device_config.forward_inject = True  # Key: sixt1c_lora mode
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    # Configure weight scaling
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


def list_linear_layers(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


# =============================================================================
# SQuAD Model & Data
# =============================================================================

def create_squad_model(params: Dict, device: torch.device, target_modules: List[str]) -> nn.Module:
    """Create SQuAD model with Sixt1c-LoRA."""
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Remove bias from target Linear layers (LRTT doesn't support bias)
    # Note: Only remove from nn.Linear, not LayerNorm etc.
    # Note: Do NOT remove bias from qa_outputs (not converted to analog)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue  # Only process Linear layers
        if "qa_outputs" in name:
            continue  # Keep qa_outputs bias intact
        if target_modules is None or any(t in name for t in target_modules):
            if module.bias is not None:
                module.bias = None

    all_linear = list_linear_layers(model)

    if target_modules is None:
        # "all" mode: only exclude qa_outputs
        exclude = ["qa_outputs"]
    else:
        exclude = [name for name in all_linear if not any(t in name for t in target_modules)]
        exclude.append("qa_outputs")

    rpu_config = create_sixt1c_lora_config(
        rank=RANK,
        lora_alpha=params["lora_alpha"],
        reinit_gain=REINIT_GAIN,
    )

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        if target_modules is None:
            # "all" mode: train everything except qa_outputs base weights
            param.requires_grad = "qa_outputs" in name or "analog" in name.lower()
        else:
            is_target = any(t in name for t in target_modules)
            param.requires_grad = is_target or "qa_outputs" in name

    return model.to(device)


def load_squad_data(tokenizer):
    """Load and tokenize SQuAD dataset."""
    raw_datasets = load_dataset("squad")

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
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
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
            max_length=384, truncation="only_second",
            stride=128, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )
        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])
            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    train_dataset = raw_datasets["train"].map(
        preprocess_train, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    eval_features = eval_examples.map(
        preprocess_eval, batched=True, remove_columns=eval_examples.column_names
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)

    return train_loader, eval_features, eval_examples


def postprocess_qa_predictions(examples, features, raw_predictions, tokenizer):
    """Post-process QA predictions."""
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    predictions = {}
    n_best_size = 20
    max_answer_length = 30

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            start_indexes = np.argsort(start_logits)[-1:-n_best_size-1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size-1:-1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append({
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": example["context"][start_char:end_char]
                    })

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions


def evaluate_squad(model, eval_features, eval_examples, tokenizer, device) -> Tuple[float, float]:
    """Evaluate SQuAD model and return F1 and EM scores."""
    model.eval()
    all_start_logits = []
    all_end_logits = []

    eval_dataloader = DataLoader(
        eval_features.remove_columns(["example_id", "offset_mapping"]),
        batch_size=BATCH_SIZE, collate_fn=default_data_collator
    )

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    all_start_logits = np.concatenate(all_start_logits)
    all_end_logits = np.concatenate(all_end_logits)

    predictions = postprocess_qa_predictions(
        eval_examples, eval_features, (all_start_logits, all_end_logits), tokenizer
    )

    f1_scores = []
    em_scores = []
    for example in eval_examples:
        pred = predictions.get(example["id"], "")
        answers = example["answers"]["text"]
        if len(answers) == 0:
            continue
        f1 = max(compute_f1(pred, ans) for ans in answers)
        em = max(compute_exact_match(pred, ans) for ans in answers)
        f1_scores.append(f1)
        em_scores.append(em)

    model.train()
    return np.mean(f1_scores) * 100, np.mean(em_scores) * 100


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, optimizer, scheduler, train_loader, device, trial_num) -> float:
    """Train for one epoch."""
    model.train()
    total_loss, num_batches = 0.0, 0

    pbar = tqdm(train_loader, desc=f"Trial {trial_num}", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss

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

def run_trial(trial: optuna.Trial, train_loader, eval_loader,
              device: torch.device, results_dir: str, target_modules: List[str],
              tokenizer=None, eval_examples=None, eval_features=None,
              init_results=None) -> float:
    """Run single trial.

    Args:
        init_results: Cached initial evaluation results to avoid redundant computation.
                     For SQuAD: {"f1": float, "em": float}
    """

    # Sample hyperparameters: lr, lora_alpha
    params = {
        "learning_rate": trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True),
        "lora_alpha": trial.suggest_float("lora_alpha", LORA_ALPHA_MIN, LORA_ALPHA_MAX, log=True),
    }

    target_name = "all" if target_modules is None else "_".join(target_modules)

    # WandB logging
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{target_name}_trial_{trial.number}",
        config={"target": target_name, "trial": trial.number, **params},
        reinit=True,
    )

    try:
        set_seed(SEED)

        # Create model
        model = create_squad_model(params, device, target_modules)

        # Create optimizer with AnalogAdam
        optimizer = AnalogAdam(model.parameters(), lr=params["learning_rate"])
        optimizer.regroup_param_groups()

        # Create warmup scheduler
        num_training_steps = len(train_loader) * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=num_training_steps
        )

        # Use cached initial evaluation (computed once in run_target_sweep)
        init_metric = init_results["f1"]
        init_em = init_results["em"]
        wandb.log({"epoch": 0, "eval/f1": init_metric, "eval/em": init_em})

        # Train
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_epoch(model, optimizer, scheduler, train_loader, device, trial.number)

            eval_metric, eval_em = evaluate_squad(model, eval_features, eval_examples, tokenizer, device)

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "eval/f1": eval_metric,
                "eval/em": eval_em
            })

        final_metric = eval_metric

        # Log final results
        improvement = final_metric - init_metric
        wandb.log({
            "final/metric": final_metric,
            "final/improvement": improvement
        })

        # Save trial result
        trial_result = {
            "trial": trial.number,
            "target": target_name,
            "params": params,
            "init_f1": init_metric,
            "final_f1": final_metric,
            "improvement": improvement,
            "optimizer": "AnalogAdam",
            "mode": "sixt1c_lora",
        }

        trial_file = os.path.join(results_dir, f"trial_{trial.number}.json")
        with open(trial_file, 'w') as f:
            json.dump(trial_result, f, indent=2)

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

def run_target_sweep(target_key: str, device: torch.device, results_dir: str, n_trials: int = 30):
    """Run sweep for a single target configuration."""
    target_modules = TARGET_CONFIGS[target_key]
    target_name = target_key

    print(f"\n{'='*60}")
    print(f"Starting Sixt1c-LoRA sweep for: {target_name}")
    print(f"Target modules: {target_modules if target_modules else 'all (except qa_outputs)'}")
    print(f"{'='*60}")

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    # Compute initial evaluation ONCE (optimization: avoid redundant computation per trial)
    set_seed(SEED)
    ref_model = create_squad_model(DEFAULT_PARAMS, device, target_modules)
    init_f1, init_em = evaluate_squad(ref_model, eval_features, eval_examples, tokenizer, device)
    init_results = {"f1": init_f1, "em": init_em}
    del ref_model
    torch.cuda.empty_cache()
    print(f"Initial evaluation (computed once): {init_results}")

    # Create study
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=f"sixt1c_lora_{target_name}",
        direction="maximize",
        sampler=sampler,
    )

    # Enqueue default params as first trial
    study.enqueue_trial(DEFAULT_PARAMS)

    # Optimize
    def objective(trial):
        return run_trial(trial, train_loader, None, device, results_dir, target_modules,
                        tokenizer=tokenizer, eval_examples=eval_examples, eval_features=eval_features,
                        init_results=init_results)

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    # Save all trial results
    all_trials = []
    for trial in study.trials:
        trial_data = {
            "trial": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": str(trial.state),
        }
        all_trials.append(trial_data)

    all_trials.sort(key=lambda t: t["value"] if t["value"] is not None else -1, reverse=True)

    result = {
        "target": target_name,
        "target_modules": target_modules,
        "mode": "sixt1c_lora",
        "rank": RANK,
        "epochs": NUM_EPOCHS,
        "n_trials": n_trials,
        "search_space": {
            "learning_rate": {"min": LR_MIN, "max": LR_MAX, "scale": "log"},
            "lora_alpha": {"min": LORA_ALPHA_MIN, "max": LORA_ALPHA_MAX, "scale": "log"},
        },
        "fixed_params": {
            "rank": RANK,
            "reinit_gain": REINIT_GAIN,
            "transfer_every": TRANSFER_EVERY,
            "forward_inject": True,
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "model": MODEL_NAME,
            "optimizer": "AnalogAdam",
        },
        "best_trial": study.best_trial.number,
        "best_f1": study.best_value,
        "best_params": study.best_params,
        "trials": all_trials,
    }

    result_file = os.path.join(results_dir, f"{target_name}_results.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{target_name} Results:")
    print(f"  Best Trial: {study.best_trial.number}")
    print(f"  Best F1: {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")
    print(f"  Saved to: {result_file}")

    return result


def main():
    global NUM_EPOCHS, N_TRIALS

    parser = argparse.ArgumentParser(description="Sixt1c-LoRA Bayesian Optimization for SQuAD")
    parser.add_argument("--target", type=str, default="V", choices=["Q", "K", "V", "QKV", "all"],
                       help="Target modules: Q, K, V, QKV, or all (default: V)")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS,
                       help="Number of trials (default: 30)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                       help="Number of epochs per trial (default: 3)")
    parser.add_argument("--run_all", action="store_true",
                       help="Run all target configurations sequentially")
    args = parser.parse_args()

    N_TRIALS = args.n_trials
    NUM_EPOCHS = args.epochs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(OUTPUT_DIR, f"sixt1c_lora_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("Sixt1c-LoRA Bayesian Optimization - SQuAD")
    print("="*60)
    print(f"Mode: sixt1c_lora (forward_inject=True, no transfer)")
    print(f"Trials: {N_TRIALS}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Results dir: {results_dir}")
    print()
    print("Search Space:")
    print(f"  learning_rate: [{LR_MIN}, {LR_MAX}] (log)")
    print(f"  lora_alpha: [{LORA_ALPHA_MIN}, {LORA_ALPHA_MAX}] (log)")
    print()
    print("Fixed Parameters:")
    print(f"  rank: {RANK}")
    print(f"  reinit_gain: {REINIT_GAIN}")
    print(f"  transfer_every: {TRANSFER_EVERY} (no transfer)")
    print(f"  forward_inject: True")
    print()
    print("Default starting point:")
    print(f"  lr: {DEFAULT_PARAMS['learning_rate']}")
    print(f"  lora_alpha: {DEFAULT_PARAMS['lora_alpha']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    all_results = {}

    if args.run_all:
        targets = ["Q", "K", "V", "QKV", "all"]
        print(f"\nRunning all targets: {targets}")
    else:
        targets = [args.target]
        print(f"\nRunning target: {args.target}")

    for target in targets:
        try:
            result = run_target_sweep(target, device, results_dir, n_trials=N_TRIALS)
            all_results[target] = result
        except Exception as e:
            print(f"Failed to complete {target}: {e}")
            all_results[target] = {"error": str(e)}

    # Save combined results
    summary_file = os.path.join(results_dir, "all_targets_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("ALL SWEEPS COMPLETE")
    print("="*60)
    print(f"Summary saved to: {summary_file}")

    for target, result in all_results.items():
        if "error" in result:
            print(f"  {target}: FAILED - {result['error']}")
        else:
            print(f"  {target}: F1={result['best_f1']:.4f} (trial {result['best_trial']})")


if __name__ == "__main__":
    main()
