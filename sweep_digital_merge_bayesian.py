#!/home/jovyan/work/ml/.venv310/bin/python
# coding=utf-8
"""Digital Merge Bayesian Optimization Sweep with WandB Logging.

Digital A/B tiles (FloatingPointDevice) + Analog C tile (SoftBoundsDevice).
Uses forward_inject=True for ReLoRA-style forward pass.
Uses Optuna for Bayesian optimization.

Search space:
- learning_rate: [2e-4, 2e-1] (log scale)
- transfer_lr: [1e-3, 1e-1] (log scale)
- transfer_every: [1, 1000] (log scale, integer)

Fixed settings:
- rank: 8
- target_modules: ["query"] (Q only)
- epochs: 1
- lora_alpha: 1.0
- forward_inject: True
- transfer_method: onehot
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

# aihwkit imports (must import before LRTT path)
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import FloatingPointDevice, SoftBoundsDevice

# LRTT imports
sys.path.insert(0, '/home/jovyan/work/LRTT/src')
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

# =============================================================================
# Search Space (Bayesian)
# =============================================================================

LR_MIN, LR_MAX = 2e-4, 2e-1
TRANSFER_LR_MIN, TRANSFER_LR_MAX = 1e-3, 1e-1
TRANSFER_EVERY_MIN, TRANSFER_EVERY_MAX = 1, 1000

N_TRIALS = 50

# =============================================================================
# Fixed Parameters
# =============================================================================

RANK = 8
LORA_ALPHA = 1.0
NUM_EPOCHS = 1
TARGET_MODULES = ["query"]
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
SEED = 42

# WandB settings
WANDB_PROJECT = "lrtt-verification-sweep"
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
# Model Creation
# =============================================================================

def create_digital_merge_config(
    rank: int,
    transfer_every: int,
    transfer_lr: float,
    lora_alpha: float,
) -> PythonLRTTRPUConfig:
    """Create LRTT config with Digital A,B + Analog C (digital merge style).

    Key differences from 6T1C A/B tile sweep:
    - A, B tiles: FloatingPointDevice (digital, exact computation)
    - C tile: SoftBoundsDevice (analog)
    - forward_inject: True (ReLoRA style: y = C(x) + α * A(B(x)))
    """
    # A, B tiles: FloatingPointDevice (digital, exact)
    ab_device = FloatingPointDevice()

    # C tile: SoftBoundsDevice (analog, no noise)
    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, write_noise_std=0.0, mult_noise=True,
    )

    device_config = PythonLRTTDevice(
        rank=rank, transfer_every=transfer_every, lora_alpha=lora_alpha,
        reinit_gain=0.1, reinit_mode="decay", decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = transfer_lr
    device_config.forward_inject = True  # Digital merge key setting!
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    return PythonLRTTRPUConfig(device=device_config)


def create_lrtt_model(
    rank: int, transfer_every: int, transfer_lr: float, lora_alpha: float,
    target_modules: List[str], device: torch.device,
) -> nn.Module:
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in target_modules)]
    exclude.append("classifier")

    rpu_config = create_digital_merge_config(
        rank=rank, transfer_every=transfer_every,
        transfer_lr=transfer_lr, lora_alpha=lora_alpha,
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
    lr = trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True)
    transfer_lr = trial.suggest_float("transfer_lr", TRANSFER_LR_MIN, TRANSFER_LR_MAX, log=True)
    transfer_every = trial.suggest_int("transfer_every", TRANSFER_EVERY_MIN, TRANSFER_EVERY_MAX, log=True)

    run = wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY,
        name=f"digital_merge_trial_{trial.number}",
        config={
            "trial_number": trial.number,
            "learning_rate": lr, "transfer_lr": transfer_lr, "transfer_every": transfer_every,
            "rank": RANK, "lora_alpha": LORA_ALPHA, "num_epochs": NUM_EPOCHS,
            "target_modules": TARGET_MODULES, "model_name": MODEL_NAME, "task_name": TASK_NAME,
            "batch_size": BATCH_SIZE, "forward_inject": True, "transfer_method": "onehot",
            "device_type": "digital_merge",
        },
        reinit=True,
    )

    csv_row = {
        'trial': trial.number,
        'learning_rate': lr,
        'transfer_lr': transfer_lr,
        'transfer_every': transfer_every,
        'status': 'running',
    }

    try:
        set_seed(SEED)
        model = create_lrtt_model(
            rank=RANK, transfer_every=transfer_every, transfer_lr=transfer_lr,
            lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES, device=device,
        )
        optimizer = AnalogAdam(model.parameters(), lr=lr)

        init_acc, init_loss = evaluate_model(model, eval_loader, device)
        wandb.log({"epoch": 0, "eval/accuracy": init_acc, "eval/loss": init_loss})

        csv_row['initial_accuracy'] = init_acc
        csv_row['initial_loss'] = init_loss

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
        wandb.log({"final/accuracy": final_acc, "final/loss": final_loss, "final/improvement": final_acc - init_acc})
        wandb.summary["final_accuracy"] = final_acc
        wandb.summary["improvement"] = final_acc - init_acc

        csv_row['final_accuracy'] = final_acc
        csv_row['final_loss'] = final_loss
        csv_row['improvement'] = final_acc - init_acc
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
    print("Digital Merge Bayesian Optimization Sweep with WandB")
    print("digital_merge: FloatingPoint A/B + SoftBounds C + forward_inject=True")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print(f"\nSearch Space ({N_TRIALS} trials):")
    print(f"  learning_rate: [{LR_MIN:.0e}, {LR_MAX:.0e}]")
    print(f"  transfer_lr: [{TRANSFER_LR_MIN:.0e}, {TRANSFER_LR_MAX:.0e}]")
    print(f"  transfer_every: [{TRANSFER_EVERY_MIN}, {TRANSFER_EVERY_MAX}]")
    print(f"\nFixed: rank={RANK}, lora_alpha={LORA_ALPHA}, epochs={NUM_EPOCHS}")
    print(f"       forward_inject=True, transfer_method=onehot")

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
    csv_path = os.path.join(OUTPUT_DIR, f'digital_merge_bayesian_results_{timestamp}.csv')
    csv_logger = CSVLogger(csv_path)
    print(f"\nCSV output: {csv_path}")

    # Optuna study
    print("\nStarting Bayesian Optimization...")
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(study_name=f"digital_merge_{timestamp}", direction="maximize", sampler=sampler)
    objective = Objective(train_loader, eval_loader, device, csv_logger)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # Results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Accuracy: {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save JSON summary
    json_path = os.path.join(OUTPUT_DIR, f'digital_merge_bayesian_summary_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_accuracy": study.best_value,
            "best_params": study.best_params,
            "n_trials": N_TRIALS,
            "csv_path": csv_path,
        }, f, indent=2)
    print(f"\nJSON summary: {json_path}")

    # Top 5
    print("\nTop 5 Trials:")
    top5 = sorted([t for t in study.trials if t.value], key=lambda t: t.value, reverse=True)[:5]
    for i, t in enumerate(top5, 1):
        print(f"  {i}. Trial {t.number}: acc={t.value:.4f}, lr={t.params['learning_rate']:.2e}, "
              f"tr_lr={t.params['transfer_lr']:.2e}, te={t.params['transfer_every']}")

    # WandB summary run
    wandb.init(project=WANDB_PROJECT, name=f"digital_merge_summary_{timestamp}", config={"best_accuracy": study.best_value, **study.best_params})
    wandb.summary.update({"best_trial": study.best_trial.number, "best_accuracy": study.best_value, **study.best_params})
    wandb.finish()

    print(f"\nDone! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
