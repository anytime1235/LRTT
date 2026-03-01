# -*- coding: utf-8 -*-
"""Optuna baseline sweep for ALBERT + GLUE: Analog QKV (frozen) + digital head/LN training.

Architecture:
    - Attention (query, key, value, dense) layers -> SingleRPU Analog (frozen weights, trainable out_scaling)
    - All other layers -> Digital (frozen except classifier, LayerNorm)
    - Trainable: classifier, LayerNorm, out_scaling
    - Optimizer: AnalogAdam, weight_decay=0
    - Sweep: learning_rate [1e-4 ~ 1e-2]

Usage:
    python optuna_baseline_glue.py --task sst2
    python optuna_baseline_glue.py --task sst2 --visualize
"""

import os
import sys
import math
import json
import argparse
import gc

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
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import SoftBoundsDevice

os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# GLUE Task Configurations
# =============================================================================

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

# Per-task config: epochs = 2× org epoch from Albert_setup.txt
# patience ≈ epoch // 4
TASK_CONFIGS = {
    "cola":  {"batch_size": 16,  "epochs": 20, "max_seq_length": 128, "patience": 5, "n_trials": 20},
    "stsb":  {"batch_size": 16,  "epochs": 20, "max_seq_length": 128, "patience": 5, "n_trials": 20},
    "sst2":  {"batch_size": 32,  "epochs": 20, "max_seq_length": 128, "patience": 5, "n_trials": 20},
    "mnli":  {"batch_size": 128, "epochs": 8,  "max_seq_length": 128, "patience": 3, "n_trials": 15},
    "qnli":  {"batch_size": 32,  "epochs": 22, "max_seq_length": 128, "patience": 6, "n_trials": 10},
    "qqp":   {"batch_size": 128, "epochs": 10, "max_seq_length": 128, "patience": 3, "n_trials": 15},
    "rte":   {"batch_size": 32,  "epochs": 22, "max_seq_length": 256, "patience": 6, "n_trials": 20},
    "mrpc":  {"batch_size": 32,  "epochs": 14, "max_seq_length": 128, "patience": 4, "n_trials": 20},
}


# =============================================================================
# Global Constants
# =============================================================================

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

SEED = 42
MODEL_NAME = "albert/albert-base-v2"
EVAL_BATCH_SIZE = 64
WARMUP_RATIO = 0.05  # 5% of total steps

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0


# =============================================================================
# RPU Config for QKV (frozen analog)
# =============================================================================

def _create_qkv_rpu_config():
    """SingleRPUConfig + SoftBoundsDevice for QKV frozen analog layers.

    Tile weights frozen via noop update hook. out_scaling is trainable.
    """
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
# Model
# =============================================================================

def create_model():
    """Create ALBERT with analog attention QKVO (frozen) + digital rest.

    Trainable: classifier, LayerNorm, out_scaling
    """
    from aihwkit.nn import AnalogLinear

    num_labels = TASK_TO_NUM_LABELS[TASK_NAME]
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    # Reinitialize classifier with fixed seed
    if hasattr(model, 'classifier'):
        torch.manual_seed(SEED)
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
        print(f"  [FIX] Reinitialized classifier with FIXED seed={SEED}")

    # Attention (QKVO) pattern matching — same as --lora-target attn
    # "attention" matches: query, key, value, dense (output projection)
    attn_pattern = "attention"

    def is_attn_encoder(layer_name):
        if "encoder" not in layer_name:
            return False
        return attn_pattern in layer_name

    # Build exclude list: everything that is NOT attention encoder
    all_linear_names = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude_modules = [n for n in all_linear_names if not is_attn_encoder(n)]

    # Convert attention layers to analog
    attn_config = _create_qkv_rpu_config()
    model = convert_to_analog(model, attn_config, exclude_modules=exclude_modules)
    analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog attention (QKVO) layers: {analog_count}")

    # Freeze tile weights via noop update hook
    def _frozen_noop_update(x_input, d_input, *args, **kwargs):
        return None
    for name, m in model.named_modules():
        if isinstance(m, AnalogLinear):
            for tile in m.analog_tiles():
                tile.update = _frozen_noop_update

    # Set requires_grad: only classifier, LayerNorm, out_scaling
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
        elif "out_scaling" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}, Trainable: {trainable:,}")

    return model.to(DEVICE)


# =============================================================================
# Data
# =============================================================================

def load_data(tokenizer):
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)
    sentence1_key, sentence2_key = TASK_TO_KEYS[TASK_NAME]

    def preprocess(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], max_length=MAX_SEQ_LENGTH, truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key],
                         max_length=MAX_SEQ_LENGTH, truncation=True)

    remove_cols = [c for c in raw_datasets["train"].column_names if c != "label"]
    tokenized = raw_datasets.map(preprocess, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")

    train_dataset = tokenized["train"]
    if TRAIN_SUBSET_SIZE > 0:
        train_dataset = train_dataset.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(train_dataset))))

    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=data_collator,
                              generator=torch.Generator().manual_seed(SEED))

    eval_key = "validation_matched" if TASK_NAME == "mnli" else "validation"
    eval_dataset = tokenized[eval_key]
    if EVAL_SUBSET_SIZE > 0:
        eval_dataset = eval_dataset.select(range(min(EVAL_SUBSET_SIZE, len(eval_dataset))))

    eval_loader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                             collate_fn=data_collator)

    print(f"  GLUE {TASK_NAME}: Train={len(train_dataset)}, Eval={len(eval_dataset)}")
    return train_loader, eval_loader


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
# Evaluation
# =============================================================================

def evaluate_model(model, eval_loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with no_grad():
        for batch in eval_loader:
            model_inputs = {k: v.to(DEVICE) for k, v in batch.items()
                           if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
            outputs = model(**model_inputs)
            total_loss += outputs.loss.item() * len(batch["labels"])

            if TASK_TO_NUM_LABELS[TASK_NAME] == 1:
                preds = outputs.logits.squeeze().cpu().numpy()
            else:
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else [preds])
            all_labels.extend(batch["labels"].cpu().tolist())

    model.train()

    n = len(all_labels)
    avg_loss = total_loss / n if n > 0 else 0.0

    if TASK_NAME == "stsb":
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
        metric_value = correct / n if n > 0 else 0.0

    return metric_value, avg_loss


# =============================================================================
# Optuna Objective
# =============================================================================

def objective(trial, train_loader, eval_loader):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metric_name = TASK_TO_METRIC[TASK_NAME]

    # Single hyperparameter: learning_rate
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    print(f"\n{'='*60}")
    print(f"Trial {trial.number} | {TASK_NAME} | LR={learning_rate:.2e}")
    print(f"{'='*60}")

    model = None
    try:
        set_seed(SEED)
        model = create_model()

        optimizer = AnalogAdam(model.parameters(), lr=learning_rate, weight_decay=0.0)
        optimizer.regroup_param_groups()

        num_training_steps = len(train_loader) * N_EPOCHS
        warmup_steps = int(WARMUP_RATIO * num_training_steps)
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_rate=0.0,
        )

        best_metric = -float('inf')
        epochs_without_improvement = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
            for batch in pbar:
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
                    print(f"  [NaN/Inf at batch {num_batches}] Aborting.")
                    return 0.0
                total_loss += loss_val
                num_batches += 1
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            eval_metric, eval_loss = evaluate_model(model, eval_loader)

            improved = ""
            if eval_metric > best_metric:
                best_metric = eval_metric
                epochs_without_improvement = 0
                improved = " *"
            else:
                epochs_without_improvement += 1

            current_lr = scheduler.get_last_lr()[0]
            tqdm.write(f"[Trial {trial.number}] Ep {epoch:3d} | "
                       f"{metric_name}: {eval_metric:.4f} | Best: {best_metric:.4f} | "
                       f"Loss: {train_loss:.4f} | LR: {current_lr:.2e} | "
                       f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}{improved}")

            trial.report(best_metric, epoch)

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break

        print(f"\n[Trial {trial.number}] Best {metric_name}: {best_metric:.4f}")
        return best_metric

    except Exception as e:
        error_msg = str(e)[:200]
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


# =============================================================================
# Visualization
# =============================================================================

def visualize_study(study, save_dir):
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        print("No completed trials to visualize.")
        return

    metric_name = TASK_TO_METRIC[TASK_NAME]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    trial_numbers = [t.number for t in complete_trials]
    scores = [t.value for t in complete_trials]

    axes[0].scatter(trial_numbers, scores, alpha=0.6)
    axes[0].plot(trial_numbers,
                 [max(scores[:i+1]) for i in range(len(scores))],
                 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel(metric_name)
    axes[0].set_title('Optimization History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[1].scatter(lrs, scores, alpha=0.6)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Learning Rate')
    axes[1].set_ylabel(metric_name)
    axes[1].set_title(f'LR vs {metric_name}')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"baseline_viz_{TASK_NAME}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualization saved.")


def print_study_summary(study):
    metric_name = TASK_TO_METRIC[TASK_NAME]
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Study: {study.study_name}, Task: {TASK_NAME}, "
          f"Trials: {len(study.trials)} ({len(complete_trials)} complete)")
    if complete_trials:
        scores = [t.value for t in complete_trials]
        print(f"Best {metric_name}: {max(scores):.4f}, Mean: {sum(scores)/len(scores):.4f}")
        print(f"Best params: {study.best_params}")


# =============================================================================
# OOM Recovery
# =============================================================================

class _OOMRestart(Exception):
    pass

def _oom_restart_callback(study, trial):
    if trial.state == TrialState.FAIL:
        err = trial.user_attrs.get("error", "")
        if "out of memory" in err.lower() or "cublas" in err.lower():
            print(f"\n[OOM Recovery] Trial {trial.number} failed, will restart.")
            raise _OOMRestart()


# =============================================================================
# Main
# =============================================================================

def main():
    global TASK_NAME, BATCH_SIZE, N_EPOCHS, MAX_SEQ_LENGTH, EARLY_STOP_PATIENCE

    parser = argparse.ArgumentParser(description="Baseline: Analog QKV (frozen) + digital head/LN")
    parser.add_argument('--task', type=str, default='sst2', choices=GLUE_TASKS,
                        help='GLUE task (default: sst2)')
    parser.add_argument('--study-name', type=str, default=None)
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Override per-task trial count')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    TASK_NAME = args.task
    task_cfg = TASK_CONFIGS[TASK_NAME]
    BATCH_SIZE = task_cfg["batch_size"]
    N_EPOCHS = task_cfg["epochs"]
    MAX_SEQ_LENGTH = task_cfg["max_seq_length"]
    EARLY_STOP_PATIENCE = task_cfg["patience"]
    n_trials = args.n_trials if args.n_trials is not None else task_cfg["n_trials"]

    RESULTS = f"/data/results/baseline_qkv/{TASK_NAME}"
    os.makedirs(RESULTS, exist_ok=True)

    study_name = args.study_name or f"albert_baseline_qkv_{TASK_NAME}_bs{BATCH_SIZE}_ep{N_EPOCHS}"
    journal_path = os.path.join(RESULTS, f"optuna_{study_name}.log")
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(journal_path),
    )

    print(f"Task: {TASK_NAME}, Epochs: {N_EPOCHS}, BS: {BATCH_SIZE}, "
          f"Patience: {EARLY_STOP_PATIENCE}, Trials: {n_trials}")

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_data(tokenizer)

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=TPESampler(seed=SEED, n_startup_trials=min(5, n_trials // 2)),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}, Device: {DEVICE}, Trials: {n_trials}")

    target_total = len(study.trials) + n_trials

    try:
        study.optimize(
            lambda trial: objective(trial, train_loader, eval_loader),
            n_trials=n_trials,
            catch=(Exception,),
            show_progress_bar=False,
            callbacks=[_oom_restart_callback],
        )
    except _OOMRestart:
        remaining = target_total - len(study.trials)
        if remaining > 0:
            print(f"\n[OOM Recovery] Restarting for {remaining} remaining trials...")
            new_argv = list(sys.argv)
            for i, arg in enumerate(new_argv):
                if arg == '--n-trials' and i + 1 < len(new_argv):
                    new_argv[i + 1] = str(remaining)
                    break
            os.execv(sys.executable, [sys.executable] + new_argv)

    print_study_summary(study)
    visualize_study(study, RESULTS)

    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if complete_trials:
        best_file = os.path.join(RESULTS, f"best_params_{study_name}.json")
        with open(best_file, 'w') as f:
            json.dump({
                "task": TASK_NAME,
                "metric": TASK_TO_METRIC[TASK_NAME],
                "best_value": study.best_value,
                "best_params": study.best_params,
                "config": {
                    "epochs": N_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "patience": EARLY_STOP_PATIENCE,
                    "optimizer": "AnalogAdam",
                    "weight_decay": 0.0,
                    "analog_layers": "qkv (frozen)",
                    "trainable": "classifier + LayerNorm + out_scaling",
                },
            }, f, indent=2)
        print(f"Best params saved: {best_file}")

    all_trials_file = os.path.join(RESULTS, f"all_trials_{TASK_NAME}.json")
    all_trials = sorted(
        [{"trial": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
         for t in study.trials],
        key=lambda x: x["value"] if x["value"] is not None else -float('inf'),
        reverse=True,
    )
    with open(all_trials_file, 'w') as f:
        json.dump(all_trials, f, indent=2)
    print(f"All trials saved: {all_trials_file}")


if __name__ == "__main__":
    main()
