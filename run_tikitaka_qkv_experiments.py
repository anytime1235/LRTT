#!/usr/bin/env python
# coding=utf-8
"""TikiTaka Q/K/V/QKV 4-way Experiment with Trial 17 Best Hyperparameters.

This script runs 4 sequential experiments with different target_modules:
1. Q only: ["query"]
2. K only: ["key"]
3. V only: ["value"]
4. QKV all: ["query", "key", "value"]

Tile Configuration (matching lrtt_python.py sixt1c_ab):
- Fast tile (A): 6T1C LinearStepDevice (write_noise_std=0)
- Slow tile (C): SoftBoundsDevice (noise=0)

Trial 17 Best Hyperparameters (76.15% accuracy):
- learning_rate: 1.31e-04
- transfer_lr: 7.36
- transfer_every: 160
- fast_lr: 0.862
- auto_granularity: 305.91
- in_chop_prob: 0.020

Fixed settings:
- epochs: 1 (test mode)
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
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam

# Import TikiTaka v2 config (6T1C Fast + SoftBounds Slow)
from tikitaka_config import create_tikitaka_v2_config, TIKITAKA_V2_DEFAULT_PARAMS


# =============================================================================
# Trial 17 Best Hyperparameters
# =============================================================================

LEARNING_RATE = 1.31e-04
TRANSFER_LR = 7.36
TRANSFER_EVERY = 160
FAST_LR = 0.862
AUTO_GRANULARITY = 305.91
IN_CHOP_PROB = 0.020

# =============================================================================
# Fixed Parameters
# =============================================================================

NUM_EPOCHS = 3  # Same as original experiment
BATCH_SIZE = 32
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
SEED = 42

# V_only test (same as previous experiment)
EXPERIMENTS = [
    {"name": "V_only", "target_modules": ["value"]},
]

# WandB settings
WANDB_PROJECT = "tikitaka-qkv-experiments"
WANDB_ENTITY = None

# Output directory
OUTPUT_DIR = "/data"


# =============================================================================
# Utility Functions
# =============================================================================

def list_linear_layers(model: nn.Module) -> List[str]:
    """List all linear layer names in the model."""
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append(name)
    return linear_layers


# =============================================================================
# Model Creation
# =============================================================================

def create_model(
    target_modules: List[str],
    device: torch.device,
) -> nn.Module:
    """Create model with TikiTaka v2 analog conversion (6T1C Fast + SoftBounds Slow)."""
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    # Get all linear layers and exclude those not in target_modules
    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in target_modules)]
    exclude.append("classifier")  # Always exclude classifier

    # Create RPU config using tikitaka_config.py
    rpu_config = create_tikitaka_v2_config(
        transfer_every=TRANSFER_EVERY,
        transfer_lr=TRANSFER_LR,
        fast_lr=FAST_LR,
        auto_granularity=AUTO_GRANULARITY,
        in_chop_prob=IN_CHOP_PROB,
    )

    # Convert to analog
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    # Freeze non-target layers
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
    """Evaluate model and return (accuracy, loss)."""
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
    model: nn.Module,
    optimizer,
    train_loader: DataLoader,
    device: torch.device,
    epoch: int,
    exp_name: str,
) -> Tuple[float, List[float]]:
    """Train for one epoch and return (avg_loss, batch_losses)."""
    model.train()
    total_loss, num_batches = 0.0, 0
    batch_losses = []
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(train_loader, desc=f"[{exp_name}] Epoch {epoch}/{NUM_EPOCHS}", leave=True)
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


# =============================================================================
# Run Single Experiment
# =============================================================================

def run_experiment(
    exp_config: Dict[str, Any],
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    timestamp: str,
) -> Dict[str, Any]:
    """Run a single experiment with given target_modules."""
    exp_name = exp_config["name"]
    target_modules = exp_config["target_modules"]

    print(f"\n{'=' * 80}")
    print(f"Experiment: {exp_name}")
    print(f"Target modules: {target_modules}")
    print(f"{'=' * 80}")

    # Initialize WandB run
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{exp_name}_{timestamp}",
        config={
            "experiment": exp_name,
            "target_modules": target_modules,
            "learning_rate": LEARNING_RATE,
            "transfer_lr": TRANSFER_LR,
            "transfer_every": TRANSFER_EVERY,
            "fast_lr": FAST_LR,
            "auto_granularity": AUTO_GRANULARITY,
            "in_chop_prob": IN_CHOP_PROB,
            "num_epochs": NUM_EPOCHS,
            "model_name": MODEL_NAME,
            "task_name": TASK_NAME,
            "batch_size": BATCH_SIZE,
            "gamma": 0.0,
            "auto_scale": True,
            "units_in_mbatch": False,
            "device_type": "tikitaka_6t1c_softbounds",
            "fast_tile": "6T1C LinearStepDevice",
            "slow_tile": "SoftBoundsDevice (noise=0)",
            "seed": SEED,
        },
        reinit=True,
    )

    # Create model
    set_seed(SEED)
    model = create_model(target_modules=target_modules, device=device)
    optimizer = AnalogAdam(model.parameters(), lr=LEARNING_RATE)

    # Count analog layers
    analog_count = sum(1 for name, m in model.named_modules() if hasattr(m, 'analog_tile'))
    print(f"Analog layers: {analog_count}")

    # Results storage
    results = {
        "experiment": exp_name,
        "target_modules": target_modules,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "transfer_lr": TRANSFER_LR,
            "transfer_every": TRANSFER_EVERY,
            "fast_lr": FAST_LR,
            "auto_granularity": AUTO_GRANULARITY,
            "in_chop_prob": IN_CHOP_PROB,
        },
        "analog_layers": analog_count,
        "epoch_results": [],
    }

    # Initial evaluation
    init_acc, init_loss = evaluate_model(model, eval_loader, device)
    print(f"Initial - Accuracy: {init_acc:.4f}, Loss: {init_loss:.4f}")
    wandb.log({"epoch": 0, "eval/accuracy": init_acc, "eval/loss": init_loss})

    results["initial_accuracy"] = init_acc
    results["initial_loss"] = init_loss

    # Training loop
    global_step = 0
    best_acc = init_acc

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        avg_train_loss, batch_losses = train_epoch(
            model, optimizer, train_loader, device, epoch, exp_name
        )

        # Log batch losses
        for batch_loss in batch_losses:
            global_step += 1
            wandb.log({"global_step": global_step, "train/batch_loss": batch_loss})

        # Evaluate
        eval_acc, eval_loss = evaluate_model(model, eval_loader, device)

        # Log epoch metrics
        wandb.log({
            "epoch": epoch,
            "train/epoch_loss": avg_train_loss,
            "eval/accuracy": eval_acc,
            "eval/loss": eval_loss,
        })

        # Store epoch results
        epoch_result = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "eval_accuracy": eval_acc,
            "eval_loss": eval_loss,
        }
        results["epoch_results"].append(epoch_result)

        # Track best
        if eval_acc > best_acc:
            best_acc = eval_acc

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Eval Acc={eval_acc:.4f}, Eval Loss={eval_loss:.4f}")

    # Final summary
    final_acc = results["epoch_results"][-1]["eval_accuracy"]
    final_loss = results["epoch_results"][-1]["eval_loss"]
    improvement = final_acc - init_acc

    results["final_accuracy"] = final_acc
    results["final_loss"] = final_loss
    results["best_accuracy"] = best_acc
    results["improvement"] = improvement

    wandb.log({
        "final/accuracy": final_acc,
        "final/loss": final_loss,
        "final/best_accuracy": best_acc,
        "final/improvement": improvement,
    })
    wandb.summary["final_accuracy"] = final_acc
    wandb.summary["best_accuracy"] = best_acc
    wandb.summary["improvement"] = improvement

    wandb.finish()

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print(f"\n{exp_name} Complete: Final Acc={final_acc:.4f}, Best Acc={best_acc:.4f}, Improvement={improvement:+.4f}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("TikiTaka Q/K/V/QKV 4-way Experiments")
    print("Trial 17 Best Hyperparameters with SoftBounds Fast + Slow Tiles")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print(f"\nTrial 17 Best Hyperparameters:")
    print(f"  learning_rate: {LEARNING_RATE:.2e}")
    print(f"  transfer_lr: {TRANSFER_LR}")
    print(f"  transfer_every: {TRANSFER_EVERY}")
    print(f"  fast_lr: {FAST_LR}")
    print(f"  auto_granularity: {AUTO_GRANULARITY}")
    print(f"  in_chop_prob: {IN_CHOP_PROB}")

    print(f"\nTile Configuration:")
    print(f"  Fast tile (A): 6T1C LinearStepDevice (write_noise_std=0)")
    print(f"  Slow tile (C): SoftBoundsDevice (noise=0)")

    print(f"\nFixed: gamma=0.0, auto_scale=True, units_in_mbatch=False")
    print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}")

    print(f"\n4 Experiments to run:")
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"  {i}. {exp['name']}: {exp['target_modules']}")

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

    # Run all experiments
    all_results = {
        "timestamp": timestamp,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "transfer_lr": TRANSFER_LR,
            "transfer_every": TRANSFER_EVERY,
            "fast_lr": FAST_LR,
            "auto_granularity": AUTO_GRANULARITY,
            "in_chop_prob": IN_CHOP_PROB,
        },
        "config": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "model_name": MODEL_NAME,
            "task_name": TASK_NAME,
            "fast_tile": "6T1C LinearStepDevice",
            "slow_tile": "SoftBoundsDevice (noise=0)",
            "seed": SEED,
        },
        "experiments": [],
    }

    for exp_config in EXPERIMENTS:
        result = run_experiment(
            exp_config=exp_config,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            timestamp=timestamp,
        )
        all_results["experiments"].append(result)

    # Save results to JSON
    json_path = os.path.join(OUTPUT_DIR, f"tikitaka_qkv_experiments_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, f"tikitaka_qkv_experiments_results_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'experiment', 'target_modules', 'analog_layers',
            'initial_accuracy', 'final_accuracy', 'best_accuracy', 'improvement',
            'epoch1_acc', 'epoch2_acc', 'epoch3_acc',
            'epoch1_loss', 'epoch2_loss', 'epoch3_loss',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp in all_results["experiments"]:
            row = {
                'experiment': exp['experiment'],
                'target_modules': str(exp['target_modules']),
                'analog_layers': exp['analog_layers'],
                'initial_accuracy': exp['initial_accuracy'],
                'final_accuracy': exp['final_accuracy'],
                'best_accuracy': exp['best_accuracy'],
                'improvement': exp['improvement'],
            }
            for i, er in enumerate(exp['epoch_results'], 1):
                row[f'epoch{i}_acc'] = er['eval_accuracy']
                row[f'epoch{i}_loss'] = er['eval_loss']
            writer.writerow(row)

    # Print final summary
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)

    print(f"\n{'Experiment':<12} {'Target':<25} {'Analog':<8} {'Init Acc':<10} {'Final Acc':<10} {'Best Acc':<10} {'Improve':<10}")
    print("-" * 95)
    for exp in all_results["experiments"]:
        print(f"{exp['experiment']:<12} {str(exp['target_modules']):<25} {exp['analog_layers']:<8} "
              f"{exp['initial_accuracy']:<10.4f} {exp['final_accuracy']:<10.4f} "
              f"{exp['best_accuracy']:<10.4f} {exp['improvement']:+.4f}")

    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    # WandB summary run
    wandb.init(
        project=WANDB_PROJECT,
        name=f"summary_{timestamp}",
        config={
            "type": "summary",
            "num_experiments": len(EXPERIMENTS),
            **all_results["hyperparameters"],
            **all_results["config"],
        },
    )

    # Log comparison table
    for exp in all_results["experiments"]:
        wandb.summary[f"{exp['experiment']}_final_acc"] = exp['final_accuracy']
        wandb.summary[f"{exp['experiment']}_best_acc"] = exp['best_accuracy']
        wandb.summary[f"{exp['experiment']}_improvement"] = exp['improvement']

    wandb.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()
