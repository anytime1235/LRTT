#!/usr/bin/env python
# coding=utf-8
"""TikiTaka v2 Refined Bayesian Search around Best Parameters.

Narrowed search space around Trial 48 optimal parameters:
- learning_rate: 3.710e-04 -> [1e-4, 1e-3]
- transfer_lr: 1.362 -> [0.8, 2.0]
- transfer_every: 100 -> [50, 200]
- fast_lr: 0.926 -> [0.7, 1.0]
- auto_granularity: 107.09 -> [50, 500]
- in_chop_prob: 0.071 -> [0.02, 0.15]

Fixed settings:
- target_modules: ["query"] (Q only)
- epochs: 1
- gamma: 0.0 (slow tile only visible)
- auto_scale: True
- units_in_mbatch: False (mat-vec units)
- model: MobileBERT
- dataset: SST-2
"""

import os
import sys
import json
import csv
from datetime import datetime
from typing import Dict, Any, List, Tuple

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


# =============================================================================
# Refined Search Space (around Trial 48 best params)
# =============================================================================

# Original best: 3.710e-04
LR_MIN, LR_MAX = 1e-4, 1e-3

# Original best: 1.362 (explore higher values)
TRANSFER_LR_MIN, TRANSFER_LR_MAX = 0.8, 10.0

# Original best: 100 (was at max boundary, explore higher)
TRANSFER_EVERY_MIN, TRANSFER_EVERY_MAX = 50, 200

# Original best: 0.926
FAST_LR_MIN, FAST_LR_MAX = 0.7, 1.0

# Original best: 107.09 (was near min boundary, explore lower)
AUTO_GRAN_MIN, AUTO_GRAN_MAX = 50, 500

# Original best: 0.071
CHOP_PROB_MIN, CHOP_PROB_MAX = 0.02, 0.15

N_TRIALS = 50

# =============================================================================
# Fixed Parameters
# =============================================================================

NUM_EPOCHS = 1
TARGET_MODULES = ["query"]
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
SEED = 42

# WandB settings
WANDB_PROJECT = "tikitaka-v2-refined-sweep"
WANDB_ENTITY = None

# Output directory
OUTPUT_DIR = "/data"


# =============================================================================
# CSV Logger
# =============================================================================

class CSVLogger:
    """Simple CSV logger for trial results."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.fieldnames = [
            'trial', 'learning_rate', 'transfer_lr', 'transfer_every',
            'fast_lr', 'auto_granularity', 'in_chop_prob',
            'initial_accuracy', 'final_accuracy', 'improvement',
            'initial_loss', 'final_loss', 'status'
        ]
        # Write header
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, Any]):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


# =============================================================================
# Utility Functions
# =============================================================================

def list_linear_layers(model: nn.Module) -> List[str]:
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(name)
    return linear_layers


# =============================================================================
# TikiTaka v2 Config Creation
# =============================================================================

def create_tikitaka_v2_config(
    transfer_every: int,
    transfer_lr: float,
    fast_lr: float,
    auto_granularity: float,
    in_chop_prob: float,
) -> UnitCellRPUConfig:
    """Create TikiTaka v2 config with 6T1C Fast + SoftBoundsReference Slow."""

    # 6T1C Fast Tile (LinearStepDevice)
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
        write_noise_std=0.0,  # Training noise off
        mult_noise=True,
        mean_bound_reference=True,
        lifetime=0.0,
    )

    # SoftBounds Slow Tile (SoftBoundsReferenceDevice)
    softbounds_device = SoftBoundsReferenceDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )

    # TikiTaka v2 Config (ChoppedTransferCompound)
    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[sixt1c_device, softbounds_device],
            # Transfer settings
            transfer_every=transfer_every,
            units_in_mbatch=False,     # v2: mat-vec units
            n_reads_per_transfer=1,
            transfer_columns=True,
            # Visibility
            gamma=0.0,                 # Slow tile only visible in forward
            # Learning rates
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=True,
            # v2 specific - Buffer & Auto-scaling
            auto_scale=True,
            auto_granularity=auto_granularity,
            buffer_granularity=1.0,
            auto_momentum=0.99,
            # Chopper
            in_chop_prob=in_chop_prob,
            in_chop_random=True,
            # Transfer IO
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

    return rpu_config


# =============================================================================
# Model Creation
# =============================================================================

def create_tikitaka_v2_model(
    transfer_every: int,
    transfer_lr: float,
    fast_lr: float,
    auto_granularity: float,
    in_chop_prob: float,
    target_modules: List[str],
    device: torch.device,
) -> nn.Module:
    """Create model with TikiTaka v2 analog conversion."""
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in target_modules)]
    exclude.append("classifier")

    rpu_config = create_tikitaka_v2_config(
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        fast_lr=fast_lr,
        auto_granularity=auto_granularity,
        in_chop_prob=in_chop_prob,
    )

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in target_modules)
        if is_target or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.to(device)
    return model


# =============================================================================
# Training & Evaluation
# =============================================================================

def evaluate_model(model: nn.Module, eval_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    model.train()
    return correct / total if total > 0 else 0.0, total_loss / total if total > 0 else 0.0


def train_epoch(
    model: nn.Module, optimizer, train_loader: DataLoader,
    device: torch.device, epoch: int, trial_number: int,
) -> Tuple[float, List[float]]:
    model.train()
    total_loss, num_batches = 0.0, 0
    batch_losses = []
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(train_loader, desc=f"Trial {trial_number} Epoch {epoch}", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        batch_losses.append(loss_val)
        num_batches += 1
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    return total_loss / num_batches if num_batches > 0 else 0.0, batch_losses


def run_trial(
    trial: optuna.Trial,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    csv_logger: CSVLogger,
) -> float:
    """Run single Optuna trial with WandB logging."""
    # Sample hyperparameters (refined search space)
    lr = trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True)
    transfer_lr = trial.suggest_float("transfer_lr", TRANSFER_LR_MIN, TRANSFER_LR_MAX)
    transfer_every = trial.suggest_int("transfer_every", TRANSFER_EVERY_MIN, TRANSFER_EVERY_MAX, log=True)
    # TikiTaka v2 specific
    fast_lr = trial.suggest_float("fast_lr", FAST_LR_MIN, FAST_LR_MAX)
    auto_granularity = trial.suggest_float("auto_granularity", AUTO_GRAN_MIN, AUTO_GRAN_MAX, log=True)
    in_chop_prob = trial.suggest_float("in_chop_prob", CHOP_PROB_MIN, CHOP_PROB_MAX)

    run = wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY,
        name=f"trial_{trial.number}",
        config={
            "trial_number": trial.number,
            "learning_rate": lr,
            "transfer_lr": transfer_lr,
            "transfer_every": transfer_every,
            "fast_lr": fast_lr,
            "auto_granularity": auto_granularity,
            "in_chop_prob": in_chop_prob,
            "num_epochs": NUM_EPOCHS,
            "target_modules": TARGET_MODULES,
            "model_name": MODEL_NAME,
            "task_name": TASK_NAME,
            "batch_size": BATCH_SIZE,
            "gamma": 0.0,
            "auto_scale": True,
            "units_in_mbatch": False,
            "device_type": "tikitaka_v2_6t1c_softbounds",
            "search_type": "refined",
        },
        reinit=True,
    )

    csv_row = {
        'trial': trial.number,
        'learning_rate': lr,
        'transfer_lr': transfer_lr,
        'transfer_every': transfer_every,
        'fast_lr': fast_lr,
        'auto_granularity': auto_granularity,
        'in_chop_prob': in_chop_prob,
        'status': 'running',
    }

    try:
        set_seed(SEED)
        model = create_tikitaka_v2_model(
            transfer_every=transfer_every,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            auto_granularity=auto_granularity,
            in_chop_prob=in_chop_prob,
            target_modules=TARGET_MODULES,
            device=device,
        )
        optimizer = AnalogAdam(model.parameters(), lr=lr)

        csv_row['initial_accuracy'] = None
        csv_row['initial_loss'] = None

        global_step = 0
        for epoch in range(1, NUM_EPOCHS + 1):
            avg_train_loss, batch_losses = train_epoch(
                model, optimizer, train_loader, device, epoch, trial.number
            )
            for batch_loss in batch_losses:
                global_step += 1
                wandb.log({"global_step": global_step, "train/batch_loss": batch_loss})

            eval_acc, eval_loss = evaluate_model(model, eval_loader, device)
            wandb.log({
                "epoch": epoch, "train/epoch_loss": avg_train_loss,
                "eval/accuracy": eval_acc, "eval/loss": eval_loss,
            })
            trial.report(eval_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        final_acc, final_loss = evaluate_model(model, eval_loader, device)
        wandb.log({"final/accuracy": final_acc, "final/loss": final_loss})
        wandb.summary["final_accuracy"] = final_acc
        wandb.summary["improvement"] = None

        csv_row['final_accuracy'] = final_acc
        csv_row['final_loss'] = final_loss
        csv_row['improvement'] = None
        csv_row['status'] = 'completed'

        del model
        torch.cuda.empty_cache()
        return final_acc

    except optuna.TrialPruned:
        csv_row['status'] = 'pruned'
        raise

    except Exception as e:
        csv_row['status'] = f'error: {str(e)}'
        wandb.log({"error": str(e)})
        raise

    finally:
        csv_logger.log(csv_row)
        wandb.finish()


# =============================================================================
# Optuna Objective
# =============================================================================

class Objective:
    def __init__(self, train_loader, eval_loader, device, csv_logger):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.csv_logger = csv_logger

    def __call__(self, trial: optuna.Trial) -> float:
        return run_trial(trial, self.train_loader, self.eval_loader, self.device, self.csv_logger)


# =============================================================================
# Main
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("TikiTaka v2 REFINED Bayesian Optimization Sweep")
    print("Search space narrowed around Trial 48 best parameters")
    print("6T1C Fast Tile + SoftBoundsReference Slow Tile")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print(f"\nRefined Search Space ({N_TRIALS} trials):")
    print(f"  learning_rate: [{LR_MIN:.0e}, {LR_MAX:.0e}] (log) - best was 3.71e-04")
    print(f"  transfer_lr: [{TRANSFER_LR_MIN}, {TRANSFER_LR_MAX}] (linear) - best was 1.362")
    print(f"  transfer_every: [{TRANSFER_EVERY_MIN}, {TRANSFER_EVERY_MAX}] (log) - best was 100")
    print(f"  fast_lr: [{FAST_LR_MIN}, {FAST_LR_MAX}] (linear) - best was 0.926")
    print(f"  auto_granularity: [{AUTO_GRAN_MIN}, {AUTO_GRAN_MAX}] (log) - best was 107.09")
    print(f"  in_chop_prob: [{CHOP_PROB_MIN}, {CHOP_PROB_MAX}] (linear) - best was 0.071")
    print(f"\nFixed: gamma=0.0, auto_scale=True, units_in_mbatch=False, epochs={NUM_EPOCHS}")

    # Load data
    print("\nLoading data...")
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)

    def preprocess(examples):
        return tokenizer(examples["sentence"], padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True)

    tokenized = raw_datasets.map(preprocess, batched=True)
    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)
    eval_loader = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=default_data_collator)

    print(f"Train: {len(tokenized['train'])}, Eval: {len(tokenized['validation'])}")

    # CSV logger
    csv_path = os.path.join(OUTPUT_DIR, f'tikitaka_v2_refined_results_{timestamp}.csv')
    csv_logger = CSVLogger(csv_path)
    print(f"\nCSV output: {csv_path}")

    # Optuna study
    print("\nStarting Refined Bayesian Optimization...")
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(study_name=f"tikitaka_v2_refined_{timestamp}", direction="maximize", sampler=sampler)
    objective = Objective(train_loader, eval_loader, device, csv_logger)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # Results
    print("\n" + "=" * 80)
    print("REFINED OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Accuracy: {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Compare with original best
    print("\n--- Comparison with Original Trial 48 ---")
    original_best = {
        "learning_rate": 3.710e-04,
        "transfer_lr": 1.362,
        "transfer_every": 100,
        "fast_lr": 0.926,
        "auto_granularity": 107.09,
        "in_chop_prob": 0.071,
    }
    print(f"{'Parameter':<20} {'Original':<15} {'Refined':<15}")
    print("-" * 50)
    for k in original_best:
        orig_val = original_best[k]
        new_val = study.best_params[k]
        if isinstance(orig_val, float):
            print(f"{k:<20} {orig_val:<15.6f} {new_val:<15.6f}")
        else:
            print(f"{k:<20} {orig_val:<15} {new_val:<15}")

    # Save JSON summary
    json_path = os.path.join(OUTPUT_DIR, f'tikitaka_v2_refined_summary_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_accuracy": study.best_value,
            "best_params": study.best_params,
            "original_best_params": original_best,
            "n_trials": N_TRIALS,
            "csv_path": csv_path,
            "search_type": "refined",
        }, f, indent=2)
    print(f"\nJSON summary: {json_path}")

    # Top 5
    print("\nTop 5 Trials:")
    top5 = sorted([t for t in study.trials if t.value], key=lambda t: t.value, reverse=True)[:5]
    for i, t in enumerate(top5, 1):
        print(f"  {i}. Trial {t.number}: acc={t.value:.4f}, lr={t.params['learning_rate']:.2e}, "
              f"tr_lr={t.params['transfer_lr']:.2f}, te={t.params['transfer_every']}, "
              f"fast_lr={t.params['fast_lr']:.2f}, auto_gran={t.params['auto_granularity']:.0f}, "
              f"chop={t.params['in_chop_prob']:.3f}")

    # WandB summary run
    wandb.init(project=WANDB_PROJECT, name=f"summary_{timestamp}", config={"best_accuracy": study.best_value, **study.best_params})
    wandb.summary.update({"best_trial": study.best_trial.number, "best_accuracy": study.best_value, **study.best_params})
    wandb.finish()

    print(f"\nDone! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
