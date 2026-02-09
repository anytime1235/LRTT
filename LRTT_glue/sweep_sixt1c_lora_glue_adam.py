#!/usr/bin/env python
# coding=utf-8
"""Sixt1c-LoRA Bayesian Optimization for GLUE tasks (AnalogAdam).

Sixt1c-LoRA mode: forward_inject=True, no transfer (transfer_every=1000000).
Search space: learning_rate, lora_alpha.
Supports GLUE tasks: cola, sst2, mrpc, qqp, mnli, qnli, rte, stsb
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler
import wandb

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np

# aihwkit imports (use installed package first)
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

# LRTT config imports (direct imports to avoid __init__.py dependency issues)
sys.path.insert(0, '/data/LRTT_transformer/src')
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


# =============================================================================
# Task Configurations
# =============================================================================

GLUE_TASKS = ["rte", "mrpc", "stsb", "cola", "sst2", "qnli", "qqp", "mnli"]

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

# Target module configurations
TARGET_CONFIGS = {
    "Q": ["query"],
    "K": ["key"],
    "V": ["value"],
    "QKV": ["query", "key", "value"],
    "all": None,  # Will include all linear layers except classifier
}

# =============================================================================
# Sixt1c-LoRA Search Space
# =============================================================================

LR_MIN, LR_MAX = 5e-4, 5e-3
LORA_ALPHA_MIN, LORA_ALPHA_MAX = 0.005, 0.03

DEFAULT_PARAMS = {
    "learning_rate": 1e-3,
    "lora_alpha": 0.01,
}

# Fixed parameters
RANK = 8
REINIT_GAIN = 0.1
TRANSFER_EVERY = 1000000  # No transfer (sixt1c_lora)

# =============================================================================
# Fixed Parameters
# =============================================================================

N_TRIALS = 10
NUM_EPOCHS = 3
TARGET_MODULES = ["value"]  # Default
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 256
WARMUP_STEPS = 0
MIN_LR_RATIO = 0.05
SEED = 42

WANDB_PROJECT = "sixt1c-lora-glue-adam-sweep"
OUTPUT_DIR = "/data/results/sixt1c_lora_glue"

os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# Sixt1c-LoRA Config
# =============================================================================

def create_sixt1c_lora_config(
    rank: int,
    lora_alpha: float = 1.0,
    reinit_gain: float = 0.1,
):
    """Create Sixt1c-LoRA config (forward_inject=True, no transfer)."""
    import math

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

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=lora_alpha,
        reinit_gain=reinit_gain,
        reinit_mode="hybrid",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.001
    device_config.units_in_mbatch = True
    device_config.forward_inject = True  # sixt1c_lora mode
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    rpu_config = PythonLRTTRPUConfig(device=device_config)

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

def create_glue_model(task_name: str, params: Dict, device: torch.device, target_modules: List[str]) -> nn.Module:
    """Create GLUE model with Sixt1c-LoRA."""
    num_labels = TASK_TO_NUM_LABELS[task_name]
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    all_linear = list_linear_layers(model)

    if target_modules is None:
        exclude = ["classifier"]
    else:
        exclude = [name for name in all_linear if not any(t in name for t in target_modules)]
        exclude.append("classifier")

    rpu_config = create_sixt1c_lora_config(
        rank=RANK,
        lora_alpha=params["lora_alpha"],
        reinit_gain=REINIT_GAIN,
    )

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        if target_modules is None:
            param.requires_grad = "classifier" in name or "analog" in name.lower()
        else:
            is_target = any(t in name for t in target_modules)
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

    model.train()

    if is_regression:
        from scipy.stats import spearmanr
        metric = spearmanr(all_preds, all_labels)[0]
        avg_loss = total_loss / len(all_labels)
    elif task_name in ["mrpc", "qqp"]:
        # F1 score for mrpc and qqp
        from sklearn.metrics import f1_score
        all_preds_np = np.array(all_preds) if all_preds else []
        all_labels_np = np.array(all_labels) if all_labels else []
        # Re-collect predictions for F1
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        model.train()
        metric = f1_score(all_labels, all_preds)
        avg_loss = total_loss / total if total > 0 else 0.0
    elif task_name == "cola":
        # Matthews correlation for cola
        from sklearn.metrics import matthews_corrcoef
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        model.train()
        metric = matthews_corrcoef(all_labels, all_preds)
        avg_loss = total_loss / total if total > 0 else 0.0
    else:
        # Accuracy for other tasks
        metric = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

    return metric, avg_loss


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, optimizer, scheduler, train_loader, device, task_name, trial_num) -> float:
    """Train for one epoch."""
    model.train()
    total_loss, num_batches = 0.0, 0
    is_regression = task_name == "stsb"
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    pbar = tqdm(train_loader, desc=f"Trial {trial_num} {task_name}", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if is_regression:
            labels = labels.float()

        optimizer.zero_grad()
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
              device: torch.device, results_dir: str, target_modules: List[str],
              init_results: Dict) -> float:
    """Run single trial."""

    params = {
        "learning_rate": trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True),
        "lora_alpha": trial.suggest_float("lora_alpha", LORA_ALPHA_MIN, LORA_ALPHA_MAX, log=True),
    }

    target_name = "all" if target_modules is None else "_".join(target_modules)

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{task_name}_{target_name}_trial_{trial.number}",
        config={"task": task_name, "target": target_name, "trial": trial.number, **params},
        reinit=True,
    )

    try:
        set_seed(SEED)

        model = create_glue_model(task_name, params, device, target_modules)

        initial_lr = params["learning_rate"]
        optimizer = AnalogAdam(model.parameters(), lr=initial_lr)
        optimizer.regroup_param_groups()

        num_training_steps = len(train_loader) * NUM_EPOCHS

        def lr_lambda(current_step: int):
            if current_step < WARMUP_STEPS:
                return float(current_step) / float(max(1, WARMUP_STEPS))
            progress = float(current_step - WARMUP_STEPS) / float(max(1, num_training_steps - WARMUP_STEPS))
            return max(MIN_LR_RATIO, 1.0 - (1.0 - MIN_LR_RATIO) * progress)

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)

        init_metric = init_results["metric"]
        init_loss = init_results["loss"]
        wandb.log({"epoch": 0, "eval/metric": init_metric, "eval/loss": init_loss})

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_epoch(model, optimizer, scheduler, train_loader, device, task_name, trial.number)
            eval_metric, eval_loss = evaluate_glue(model, eval_loader, task_name, device)

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "eval/metric": eval_metric,
                "eval/loss": eval_loss
            })

        final_metric = eval_metric
        improvement = final_metric - init_metric
        wandb.log({
            "final/metric": final_metric,
            "final/improvement": improvement
        })

        trial_result = {
            "trial": trial.number,
            "task": task_name,
            "target": target_name,
            "params": params,
            "init_metric": init_metric,
            "final_metric": final_metric,
            "improvement": improvement,
            "optimizer": "AnalogAdam",
            "mode": "sixt1c_lora",
        }

        trial_file = os.path.join(results_dir, f"{task_name}_trial_{trial.number}.json")
        with open(trial_file, 'w') as f:
            json.dump(trial_result, f, indent=2)

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

def run_task_sweep(task_name: str, target_key: str, device: torch.device, results_dir: str, n_trials: int = 10):
    """Run sweep for a single task and target configuration."""
    target_modules = TARGET_CONFIGS[target_key]
    target_name = target_key

    print(f"\n{'='*60}")
    print(f"Starting Sixt1c-LoRA sweep for: {task_name.upper()} - {target_name}")
    print(f"Target modules: {target_modules if target_modules else 'all (except classifier)'}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_glue_data(task_name, tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # Compute initial evaluation ONCE
    set_seed(SEED)
    ref_model = create_glue_model(task_name, DEFAULT_PARAMS, device, target_modules)
    init_metric, init_loss = evaluate_glue(ref_model, eval_loader, task_name, device)
    init_results = {"metric": init_metric, "loss": init_loss}
    del ref_model
    torch.cuda.empty_cache()
    print(f"Initial evaluation (computed once): {init_results}")

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=f"sixt1c_lora_{task_name}_{target_name}",
        direction="maximize",
        sampler=sampler,
    )

    study.enqueue_trial(DEFAULT_PARAMS)

    def objective(trial):
        return run_trial(trial, task_name, train_loader, eval_loader, device, results_dir,
                        target_modules, init_results)

    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    all_trials = []
    for trial in study.trials:
        trial_data = {
            "trial": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": str(trial.state),
        }
        all_trials.append(trial_data)

    all_trials.sort(key=lambda t: t["value"] if t["value"] is not None else -float('inf'), reverse=True)

    result = {
        "task": task_name,
        "target": target_name,
        "target_modules": target_modules,
        "mode": "sixt1c_lora",
        "rank": RANK,
        "epochs": NUM_EPOCHS,
        "n_trials": n_trials,
        "metric_name": TASK_TO_METRIC[task_name],
        "search_space": {
            "learning_rate": {"min": LR_MIN, "max": LR_MAX, "scale": "log"},
            "lora_alpha": {"min": LORA_ALPHA_MIN, "max": LORA_ALPHA_MAX, "scale": "log"},
        },
        "best_trial": study.best_trial.number,
        "best_metric": study.best_value,
        "best_params": study.best_params,
        "trials": all_trials,
    }

    result_file = os.path.join(results_dir, f"{task_name}_{target_name}_results.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{task_name.upper()} - {target_name} Results:")
    print(f"  Best Trial: {study.best_trial.number}")
    print(f"  Best {TASK_TO_METRIC[task_name]}: {study.best_value:.4f}")
    print(f"  Best Params: {study.best_params}")
    print(f"  Saved to: {result_file}")

    return result


def main():
    global NUM_EPOCHS, N_TRIALS

    parser = argparse.ArgumentParser(description="Sixt1c-LoRA Bayesian Optimization for GLUE tasks")
    parser.add_argument("--task", type=str, default="sst2", choices=GLUE_TASKS,
                       help="GLUE task (default: sst2)")
    parser.add_argument("--target", type=str, default="V", choices=["Q", "K", "V", "QKV", "all"],
                       help="Target modules (default: V)")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS,
                       help="Number of trials (default: 10)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                       help="Number of epochs per trial (default: 3)")
    parser.add_argument("--run_all_tasks", action="store_true",
                       help="Run all GLUE tasks sequentially")
    args = parser.parse_args()

    N_TRIALS = args.n_trials
    NUM_EPOCHS = args.epochs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(OUTPUT_DIR, f"sixt1c_lora_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("Sixt1c-LoRA Bayesian Optimization - GLUE")
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

    if args.run_all_tasks:
        tasks = GLUE_TASKS
        print(f"\nRunning all GLUE tasks: {tasks}")
    else:
        tasks = [args.task]
        print(f"\nRunning task: {args.task}")

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task.upper()}")
        print(f"{'='*60}")

        try:
            result = run_task_sweep(task, args.target, device, results_dir, n_trials=N_TRIALS)
            all_results[task] = result
        except Exception as e:
            print(f"Failed to complete {task}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task] = {"error": str(e)}

    summary_file = os.path.join(results_dir, "all_tasks_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("ALL SWEEPS COMPLETE")
    print("="*60)
    print(f"Summary saved to: {summary_file}")

    for task, result in all_results.items():
        if "error" in result:
            print(f"  {task}: FAILED - {result['error']}")
        else:
            print(f"  {task}: {TASK_TO_METRIC[task]}={result['best_metric']:.4f} (trial {result['best_trial']})")


if __name__ == "__main__":
    main()
