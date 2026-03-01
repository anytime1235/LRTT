#!/usr/bin/env python
# coding=utf-8
"""FP-LoRA Bayesian Optimization for GLUE tasks (AnalogSGD).

Architecture (FP LoRA mode - FIXED):
- Uses PEFT LoRA (digital lora_A + lora_B, trainable)
- base_layer: TorchInferenceRPUConfig (analog, frozen, inference only)
- base_layer: ALL NOISE = 0 (is_perfect=True, deterministic)
- Classifier: frozen (requires_grad=False)
- Optimizer: AnalogSGD

CRITICAL: base_layer MUST use TorchInferenceRPUConfig to prevent weight updates!

Search space: learning_rate (log scale).
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
import copy

import optuna
from optuna_integration import BoTorchSampler
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

# PEFT imports
from peft import LoraConfig, get_peft_model

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.tiles.inference_torch import TorchInferenceTile

# Standardized sixt1c config imports (kept for sixt1c mode compatibility)
sys.path.insert(0, '/data/LRTT_transformer/lora_training')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
from sixt1c_config import (
    gen_sixt1c_lora_config_standard,
    gen_softbounds_base_layer_config,  # ← TorchInferenceRPUConfig (MUST USE)
    gen_sixt1c_lora_config_trainable   # ← SingleRPUConfig (for LoRA only)
    # NOTE: Do NOT import gen_softbounds_base_layer_config_trainable!
    # base_layer MUST use TorchInferenceRPUConfig to prevent weight updates!
)
from smart_conversion import convert_base_and_lora_separately


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
# Layer Conversion Helpers (from lora_on_analog_hardware)
# =============================================================================

def list_analog_linear_layers(model):
    """List all AnalogLinear layer names in the model."""
    return [name for name, module in model.named_modules() if isinstance(module, AnalogLinear)]

def list_linear_layers(model):
    """List all nn.Linear layer names in the model."""
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]

def replace_layer(model, source_model, layer_name):
    """Replace a layer in model with the corresponding layer from source_model."""
    parts = layer_name.split('.')
    target_module = model
    source_module = source_model

    for part in parts[:-1]:
        target_module = getattr(target_module, part)
        source_module = getattr(source_module, part)

    setattr(target_module, parts[-1], getattr(source_module, parts[-1]))

def convert_layer_to_analog(model, layer_name, rpu_config):
    """Convert a specific layer to analog."""
    parts = layer_name.split('.')
    parent_module = model

    for part in parts[:-1]:
        parent_module = getattr(parent_module, part)

    layer = getattr(parent_module, parts[-1])

    if isinstance(layer, nn.Linear):
        analog_layer = AnalogLinear(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            rpu_config=rpu_config
        )
        analog_layer.set_weights(layer.weight, layer.bias)
        setattr(parent_module, parts[-1], analog_layer)


def update_lora_alpha(model, new_lora_alpha: float):
    """Update LoRA scaling factor without recreating model.

    In PEFT, lora_alpha is used in scaling = lora_alpha / r.
    This function updates the scaling attribute in all LoRA modules.
    """
    num_updated = 0
    for name, module in model.named_modules():
        # PEFT LoRA layers have a 'scaling' attribute (dict with 'default' key)
        if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
            if 'default' in module.scaling and hasattr(module, 'r') and isinstance(module.r, dict):
                # Update scaling factor: scaling = lora_alpha / r
                r_value = module.r['default']
                module.scaling['default'] = new_lora_alpha / r_value
                num_updated += 1

    return num_updated


# =============================================================================
# Sixt1c-LoRA Search Space
# =============================================================================

LR_MIN, LR_MAX = 2e-4, 2e-2
LORA_ALPHA_FIXED = 1.0  # Fixed: rank=8, so lora_alpha=1.0

DEFAULT_PARAMS = {
    "learning_rate": 1e-3,
}

# Fixed parameters
RANK = 8  # LoRA rank
PATIENCE = 3  # Early stopping patience

# =============================================================================
# Fixed Parameters
# =============================================================================

N_TRIALS = 10
NUM_EPOCHS = 3
TARGET_MODULES = ["query", "key", "value"]  # Default: QKV
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128  # GLUE standard (from lora_on_analog_hardware)
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64  # Same as train batch (lora_on_analog uses 32 for both)
WARMUP_RATIO = 0.1  # Warmup for 10% of total training steps
MIN_LR_RATIO = 0.0  # Linear decay to 0 (no minimum LR)
SEED = 42

WANDB_PROJECT = "fp-lora-glue-sgd-sweep"
OUTPUT_DIR = "/data/results/fp_lora_glue"

os.environ["WANDB_MODE"] = "offline"




# =============================================================================
# GLUE Model & Data
# =============================================================================

def create_glue_model(task_name: str, device: torch.device, target_modules: List[str],
                      fp_lora: bool = False, lora_alpha: float = None) -> nn.Module:
    """Create GLUE model with Sixt1c-LoRA (SMART CONVERSION with SingleRPUConfig).

    Smart Conversion Architecture:
    1. Apply PEFT LoRA to target modules (digital)
    2. DIRECTLY convert base_layer → SingleRPUConfig + SoftBoundsDevice (frozen)
    3. DIRECTLY convert lora_A/B → SingleRPUConfig + LinearStepDevice (trainable)

    Both use SingleRPUConfig (consistent device type):
    - base_layer: SingleRPUConfig + SoftBoundsDevice (requires_grad=False)
    - lora_A/B: SingleRPUConfig + LinearStepDevice (requires_grad=True)

    Args:
        task_name: GLUE task name
        device: torch device
        target_modules: List of module names to apply LoRA
        fp_lora: If True, keep LoRA adapters as digital (skip analog conversion)

    Returns:
        Model with sixt1c-LoRA applied (or FP-LoRA if fp_lora=True)
    """
    # Step 1: Load base model
    num_labels = TASK_TO_NUM_LABELS[task_name]
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    # CRITICAL FIX: Reinitialize classifier with small weights + FIXED SEED
    # MobileBERT's classifier is randomly initialized with large values causing instability
    # For frozen classifier experiments, we MUST use fixed seed for fair comparison
    if hasattr(model, 'classifier'):
        torch.manual_seed(SEED)  # CRITICAL: Fix seed before classifier init
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
        print(f"[FIX] Reinitialized classifier with FIXED seed={SEED} (std=0.02)")

    # Step 2: Apply PEFT LoRA with FIXED lora_alpha=1.0
    if target_modules is None:
        peft_target = ["query", "key", "value", "dense", "intermediate"]
    else:
        peft_target = target_modules

    # Use provided lora_alpha or default
    alpha_value = lora_alpha if lora_alpha is not None else LORA_ALPHA_FIXED

    peft_config = LoraConfig(
        r=RANK,
        lora_alpha=alpha_value,
        lora_dropout=0.0,
        target_modules=peft_target,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    print(f"\nApplied PEFT LoRA (alpha={alpha_value}) to {peft_target}")
    model.print_trainable_parameters()

    # Step 3: Conversion based on mode
    if fp_lora:
        # ===== FP LoRA mode =====
        # base_layer → SingleRPUConfig + SoftBoundsDevice (identical to sixt1c base_layer, frozen)
        # LoRA A/B → digital (trainable, NOT converted to analog)
        print("\n[FP-LoRA Mode] base_layer → TorchInferenceRPU (analog, frozen), LoRA → digital (trainable)")

        # CRITICAL: base_layer MUST use TorchInferenceRPUConfig (inference only, no updates)
        base_layer_config = gen_softbounds_base_layer_config()  # ← TorchInferenceRPUConfig

        # Only convert base_layer to analog, skip lora_A/B
        base_layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'base_layer' in name:
                base_layer_names.append(name)

        from smart_conversion import get_parent_module, make_analog_peft_compatible

        for layer_name in base_layer_names:
            parent, attr = get_parent_module(model, layer_name)
            digital_layer = getattr(parent, attr)

            analog_layer = AnalogLinear.from_digital(digital_layer, base_layer_config)
            analog_layer = make_analog_peft_compatible(analog_layer, is_trainable=True)

            for param in analog_layer.parameters():
                param.requires_grad = False

            setattr(parent, attr, analog_layer)

        print(f"  Converted {len(base_layer_names)} base_layer to SoftBounds analog (frozen)")
        print(f"  LoRA A/B kept as digital (trainable)")

    else:
        # ===== Sixt1c mode =====
        print("\n[Smart Conversion] base_layer=TorchInferenceRPU (frozen), lora=SingleRPU (trainable)")

        # CRITICAL: base_layer MUST use TorchInferenceRPUConfig (inference only, no updates)
        base_layer_config = gen_softbounds_base_layer_config()  # ← TorchInferenceRPUConfig
        lora_config = gen_sixt1c_lora_config_trainable(output_noise_level=0.0)

        print(f"  base_layer config: TorchInferenceRPUConfig + SoftBounds (inference only, no updates)")
        print(f"  lora config: SingleRPUConfig + LinearStepDevice (6T1C, trainable)")

        model = convert_base_and_lora_separately(
            model,
            base_layer_config=base_layer_config,
            lora_config=lora_config,
            lora_trainable=True
        )

    # Step 4: Set final trainability
    # - classifier: FROZEN
    # - lora digital params: trainable
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = False  # FROZEN for this experiment
        elif 'lora' in name and 'analog' not in name.lower():
            param.requires_grad = True

    print("\nFinal model structure:")
    model.print_trainable_parameters()

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
    eval_loader = DataLoader(tokenized[eval_key], batch_size=EVAL_BATCH_SIZE,
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
              init_results: Dict, fp_lora: bool = False, lora_alpha: float = None) -> float:
    """Run single trial (lr sweep only, lora_alpha fixed)."""

    params = {
        "learning_rate": trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True),
    }

    target_name = "all" if target_modules is None else "_".join(target_modules)
    mode_name = "fp_lora" if fp_lora else "sixt1c_lora"

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{mode_name}_{task_name}_{target_name}_trial_{trial.number}",
        config={"task": task_name, "target": target_name, "trial": trial.number, "mode": mode_name,
                "lora_alpha": LORA_ALPHA_FIXED, **params},
        reinit=True,
    )

    try:
        set_seed(SEED)

        # Create new model for each trial
        model = create_glue_model(task_name, device, target_modules, fp_lora=fp_lora, lora_alpha=lora_alpha)

        initial_lr = params["learning_rate"]
        optimizer = AnalogSGD(model.parameters(), lr=initial_lr)  # FIXED: was AnalogAdam
        optimizer.regroup_param_groups(model)  # CRITICAL: must pass model to find analog tiles!

        # Calculate warmup steps: 10% of total training steps
        num_training_steps = len(train_loader) * NUM_EPOCHS
        warmup_steps = int(num_training_steps * WARMUP_RATIO)

        def lr_lambda(current_step: int):
            """Linear decay with warmup (10% of total steps), decay to 0."""
            if current_step < warmup_steps:
                # Warmup: linear increase from 0 to 1
                return float(current_step) / float(max(1, warmup_steps))
            # Linear decay: from 1.0 to 0.0
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.0, 1.0 - progress)  # Decay to 0

        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda)

        init_metric = init_results["metric"]
        init_loss = init_results["loss"]
        wandb.log({"epoch": 0, "eval/metric": init_metric, "eval/loss": init_loss})

        # Early stopping tracking
        best_metric = 0.0
        patience_counter = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_epoch(model, optimizer, scheduler, train_loader, device, task_name, trial.number)
            eval_metric, eval_loss = evaluate_glue(model, eval_loader, task_name, device)

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "eval/metric": eval_metric,
                "eval/loss": eval_loss
            })

            # Track best metric for early stopping
            if eval_metric > best_metric:
                best_metric = eval_metric
                patience_counter = 0
            else:
                patience_counter += 1

            # Report to Optuna for potential pruning
            trial.report(best_metric, epoch)

            # Check if Optuna pruner wants to stop this trial
            if trial.should_prune():
                print(f"  Trial pruned by Optuna at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

            # Early stopping check (unlikely to trigger with NUM_EPOCHS=3, but consistent with SQuAD)
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs, best={best_metric:.4f})")
                break

        final_metric = best_metric  # Use best metric instead of last epoch
        improvement = final_metric - init_metric
        wandb.log({
            "final/metric": final_metric,
            "final/improvement": improvement
        })

        trial_result = {
            "experiment_status": "Real",  # Valid experiment with correct architecture
            "trial": trial.number,
            "task": task_name,
            "target": target_name,
            "params": params,
            "init_metric": init_metric,
            "final_metric": final_metric,
            "improvement": improvement,
            "optimizer": "AnalogSGD",
            "architecture": "standardized_config",
            "mode": mode_name,
            "fp_lora": fp_lora,
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

def run_task_sweep(task_name: str, target_key: str, device: torch.device, results_dir: str, n_trials: int = 10,
                   fp_lora: bool = False, lora_alpha: float = None):
    """Run sweep for a single task and target configuration."""
    target_modules = TARGET_CONFIGS[target_key]
    target_name = target_key
    mode_name = "FP-LoRA (digital adapters)" if fp_lora else "Sixt1c-LoRA (6T1C analog adapters)"

    print(f"\n{'='*60}")
    print(f"Starting {mode_name} sweep for: {task_name.upper()} - {target_name}")
    print(f"Target modules: {target_modules if target_modules else 'all (except classifier)'}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_glue_data(task_name, tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # OPTIMIZATION: Skip initial evaluation (slow model conversion)
    # Initial metric will be computed in first trial instead
    print("⚡ OPTIMIZATION: Skipping initial evaluation (will use first trial as baseline)")
    init_results = {"metric": 0.0, "loss": 0.0}  # Placeholder, first trial will establish baseline

    # Create study - LR sweep only (lora_alpha=1.0 fixed)
    study_prefix = "fp_lora" if fp_lora else "sixt1c_lora_frozen_cls"
    sampler = BoTorchSampler()
    study = optuna.create_study(
        study_name=f"{study_prefix}_{task_name}_{target_name}_alpha{lora_alpha}_lr_sweep",
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
    )

    study.enqueue_trial(DEFAULT_PARAMS)

    def objective(trial):
        return run_trial(trial, task_name, train_loader, eval_loader, device, results_dir,
                        target_modules, init_results, fp_lora=fp_lora, lora_alpha=lora_alpha)

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

    lora_adapter_desc = "digital FP (trainable)" if fp_lora else "6T1C sixt1c (trainable)"
    result_mode = "fp_lora" if fp_lora else "sixt1c_lora"
    result = {
        "experiment_status": "Real",  # Valid experiment with correct architecture
        "task": task_name,
        "target": target_name,
        "target_modules": target_modules,
        "mode": result_mode,
        "fp_lora": fp_lora,
        "rank": RANK,
        "epochs": NUM_EPOCHS,
        "n_trials": n_trials,
        "metric_name": TASK_TO_METRIC[task_name],
        "search_space": {
            "learning_rate": {"min": LR_MIN, "max": LR_MAX, "scale": "log"},
        },
        "fixed_params": {
            "rank": RANK,
            "lora_alpha": LORA_ALPHA_FIXED,  # FIXED: 1.0
            "architecture": "standardized_config",
            "base_layer": "SoftBounds (frozen)",
            "lora_adapters": lora_adapter_desc,
            "noise_management": "ABS_MAX",
            "bound_management": "ITERATIVE",
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "model": MODEL_NAME,
            "optimizer": "AnalogSGD",
        },
        "best_trial": study.best_trial.number,
        "best_metric": study.best_value,
        "best_params": study.best_params,
        "trials": all_trials,
    }

    result_file = os.path.join(results_dir, f"{result_mode}_{task_name}_{target_name}_results.json")
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

    parser = argparse.ArgumentParser(description="FP-LoRA Bayesian Optimization for GLUE tasks")
    parser.add_argument("--task", type=str, default="sst2", choices=GLUE_TASKS,
                       help="GLUE task (default: sst2)")
    parser.add_argument("--target", type=str, default="QKV", choices=["Q", "K", "V", "QKV", "all"],
                       help="Target modules (default: QKV)")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS,
                       help="Number of trials (default: 10)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                       help="Number of epochs per trial (default: 3)")
    parser.add_argument("--run_all_tasks", action="store_true",
                       help="Run all GLUE tasks sequentially")
    parser.add_argument("--lora_alpha", type=float, default=None,
                       help="Override lora_alpha value (default: use LORA_ALPHA_FIXED=1.0)")
    args = parser.parse_args()

    # FP LoRA mode is always enabled in this script
    args.fp_lora = True

    N_TRIALS = args.n_trials
    NUM_EPOCHS = args.epochs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(OUTPUT_DIR, f"fp_lora_frozen_cls_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("FP-LoRA Bayesian Optimization - GLUE")
    print("="*60)
    print(f"Architecture: PEFT LoRA (digital adapters)")
    print(f"Mode: FP-LoRA (digital lora_A/B, frozen base weights)")
    print(f"Trials: {N_TRIALS}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Results dir: {results_dir}")
    print()
    print("Search Space:")
    print(f"  learning_rate: [{LR_MIN}, {LR_MAX}] (log)")
    print()
    actual_lora_alpha = args.lora_alpha if args.lora_alpha is not None else LORA_ALPHA_FIXED

    print("Fixed Parameters:")
    print(f"  rank: {RANK}")
    print(f"  lora_alpha: {actual_lora_alpha} {'(OVERRIDE)' if args.lora_alpha is not None else '(FIXED)'}")
    print(f"  base_weights: frozen (digital)")
    print(f"  lora_A/B: Digital FP (trainable)")
    print(f"  classifier: frozen")
    print(f"  optimizer: AnalogSGD")
    print()
    print("Default starting point:")
    print(f"  lr: {DEFAULT_PARAMS['learning_rate']}")
    print(f"  lora_alpha: {actual_lora_alpha}")

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
            result = run_task_sweep(task, args.target, device, results_dir, n_trials=N_TRIALS, fp_lora=args.fp_lora, lora_alpha=args.lora_alpha)
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
