#!/usr/bin/env python
# coding=utf-8
"""TikiTaka Dense Subset Experiments: Identify NaN-causing layers

This script tests individual Dense layer groups to identify which causes NaN:
1. attention_output_dense: Attention output projection only
2. bottleneck_dense: MobileBERT bottleneck layers
3. intermediate_output_dense: Main MLP (intermediate + output, non-FFN)

Background:
- Dense_only experiment shows NaN at Epoch 2
- FFN_only experiment works fine (71.33%)
- This script isolates the problematic layer group

Uses Trial 17 Best Hyperparameters (same as layer experiments).
"""

import os
import sys
import json
import csv
import math
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
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs import (
    UnitCellRPUConfig,
    IOParameters,
    UpdateParameters,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import SoftBoundsReferenceDevice


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

NUM_EPOCHS = 3
BATCH_SIZE = 32
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
SEED = 42

# Gradient clipping (to help prevent NaN)
GRADIENT_CLIP_NORM = 1.0

# Dense Subset Experiment Configurations
# Each targets a specific subset of dense layers to identify NaN source
EXPERIMENTS = [
    {
        "name": "attention_output_dense",
        "description": "Attention output projection only",
        "target_modules": ["attention.output.dense"],
        "exclude_patterns": [],  # No additional exclusions
    },
    {
        "name": "bottleneck_dense",
        "description": "MobileBERT bottleneck layers",
        "target_modules": ["bottleneck"],
        "exclude_patterns": [],
    },
    {
        "name": "intermediate_output_dense",
        "description": "Main MLP (intermediate + output dense, excluding FFN and bottleneck)",
        "target_modules": ["intermediate.dense", "output.dense"],
        "exclude_patterns": ["ffn", "bottleneck"],  # Exclude FFN and bottleneck
    },
]

# WandB settings
WANDB_PROJECT = "tikitaka-dense-subset-experiments"
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


def filter_layers_with_exclusions(
    all_layers: List[str],
    target_modules: List[str],
    exclude_patterns: List[str],
) -> Tuple[List[str], List[str]]:
    """Filter layers based on target_modules and exclude_patterns.

    Returns: (target_layers, exclude_layers)
    """
    target_layers = []
    exclude_layers = []

    for layer in all_layers:
        # Check if layer matches any target pattern
        matches_target = any(t in layer for t in target_modules)
        # Check if layer matches any exclusion pattern
        matches_exclude = any(e in layer for e in exclude_patterns) if exclude_patterns else False

        if matches_target and not matches_exclude and "classifier" not in layer:
            target_layers.append(layer)
        else:
            exclude_layers.append(layer)

    return target_layers, exclude_layers


# =============================================================================
# TikiTaka SoftBounds Config Creation
# =============================================================================

def create_tikitaka_softbounds_config(
    transfer_every: int,
    transfer_lr: float,
    fast_lr: float,
    auto_granularity: float,
    in_chop_prob: float,
) -> UnitCellRPUConfig:
    """Create TikiTaka config with SoftBounds Fast + SoftBounds Slow (both noise=0)."""

    # Fast Tile (A): SoftBounds (noise=0)
    fast_device = SoftBoundsReferenceDevice(
        dw_min=0.001,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        mult_noise=False,
        dw_min_dtod=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
        diffusion=0.0,
        lifetime=0.0,
    )

    # Slow Tile (C): SoftBounds (noise=0)
    slow_device = SoftBoundsReferenceDevice(
        dw_min=0.001,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        mult_noise=False,
        dw_min_dtod=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
        diffusion=0.0,
        lifetime=0.0,
    )

    # TikiTaka Config (ChoppedTransferCompound)
    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[fast_device, slow_device],
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

    return rpu_config


# =============================================================================
# Model Creation
# =============================================================================

def create_model(
    target_modules: List[str],
    exclude_patterns: List[str],
    device: torch.device,
) -> Tuple[nn.Module, List[str], int]:
    """Create model with TikiTaka SoftBounds analog conversion.

    Returns: (model, target_layer_names, analog_count)
    """
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    # Get all linear layers and filter with exclusions
    all_linear = list_linear_layers(model)
    target_layers, exclude_layers = filter_layers_with_exclusions(
        all_linear, target_modules, exclude_patterns
    )

    # Always exclude classifier
    if "classifier" not in exclude_layers:
        exclude_layers.append("classifier")

    # Create RPU config
    rpu_config = create_tikitaka_softbounds_config(
        transfer_every=TRANSFER_EVERY,
        transfer_lr=TRANSFER_LR,
        fast_lr=FAST_LR,
        auto_granularity=AUTO_GRANULARITY,
        in_chop_prob=IN_CHOP_PROB,
    )

    # Convert to analog
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude_layers)

    # Freeze non-target layers
    for name, param in model.named_parameters():
        is_target = any(t in name for t in target_modules)
        is_excluded = any(e in name for e in exclude_patterns) if exclude_patterns else False
        if (is_target and not is_excluded) or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Count analog layers (correct method using isinstance)
    analog_count = sum(1 for _, m in model.named_modules() if isinstance(m, AnalogLayerBase))

    model.to(device)
    return model, target_layers, analog_count


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
) -> Tuple[float, List[float], bool]:
    """Train for one epoch and return (avg_loss, batch_losses, nan_occurred)."""
    model.train()
    total_loss, num_batches = 0.0, 0
    batch_losses = []
    criterion = nn.CrossEntropyLoss()
    nan_occurred = False

    pbar = tqdm(train_loader, desc=f"[{exp_name}] Epoch {epoch}/{NUM_EPOCHS}", leave=True)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)

        # Check for NaN
        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print(f"\n[WARNING] NaN/Inf loss detected at batch {num_batches + 1}")
            nan_occurred = True
            batch_losses.append(float('nan'))
            num_batches += 1
            continue

        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)

        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        batch_losses.append(loss_val)
        num_batches += 1
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
    return avg_loss, batch_losses, nan_occurred


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
    """Run a single experiment with given configuration."""
    exp_name = exp_config["name"]
    target_modules = exp_config["target_modules"]
    exclude_patterns = exp_config.get("exclude_patterns", [])
    description = exp_config["description"]

    print(f"\n{'=' * 80}")
    print(f"Experiment: {exp_name}")
    print(f"Description: {description}")
    print(f"Target modules: {target_modules}")
    print(f"Exclude patterns: {exclude_patterns}")
    print(f"Gradient clipping: {GRADIENT_CLIP_NORM}")
    print(f"{'=' * 80}")

    # Create model
    set_seed(SEED)
    model, target_layers, analog_count = create_model(
        target_modules=target_modules,
        exclude_patterns=exclude_patterns,
        device=device,
    )

    print(f"Target layers ({len(target_layers)}):")
    for i, layer in enumerate(target_layers[:10]):
        print(f"  {layer}")
    if len(target_layers) > 10:
        print(f"  ... and {len(target_layers) - 10} more")
    print(f"Analog layers: {analog_count}")

    # Initialize WandB run
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"{exp_name}_{timestamp}",
        config={
            "experiment": exp_name,
            "description": description,
            "target_modules": target_modules,
            "exclude_patterns": exclude_patterns,
            "num_target_layers": len(target_layers),
            "analog_layers": analog_count,
            "learning_rate": LEARNING_RATE,
            "transfer_lr": TRANSFER_LR,
            "transfer_every": TRANSFER_EVERY,
            "fast_lr": FAST_LR,
            "auto_granularity": AUTO_GRANULARITY,
            "in_chop_prob": IN_CHOP_PROB,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "num_epochs": NUM_EPOCHS,
            "model_name": MODEL_NAME,
            "task_name": TASK_NAME,
            "batch_size": BATCH_SIZE,
            "gamma": 0.0,
            "auto_scale": True,
            "units_in_mbatch": False,
            "device_type": "tikitaka_softbounds_softbounds",
            "seed": SEED,
        },
        reinit=True,
    )

    optimizer = AnalogAdam(model.parameters(), lr=LEARNING_RATE)

    # Results storage
    results = {
        "experiment": exp_name,
        "description": description,
        "target_modules": target_modules,
        "exclude_patterns": exclude_patterns,
        "target_layers": target_layers,
        "num_target_layers": len(target_layers),
        "analog_layers": analog_count,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "transfer_lr": TRANSFER_LR,
            "transfer_every": TRANSFER_EVERY,
            "fast_lr": FAST_LR,
            "auto_granularity": AUTO_GRANULARITY,
            "in_chop_prob": IN_CHOP_PROB,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
        },
        "epoch_results": [],
        "nan_occurred": False,
        "nan_epoch": None,
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
        avg_train_loss, batch_losses, nan_occurred = train_epoch(
            model, optimizer, train_loader, device, epoch, exp_name
        )

        # Log batch losses
        for batch_loss in batch_losses:
            global_step += 1
            if not math.isnan(batch_loss):
                wandb.log({"global_step": global_step, "train/batch_loss": batch_loss})

        # Check for NaN
        if nan_occurred:
            print(f"\n[ERROR] NaN detected in epoch {epoch}. Recording and continuing...")
            results["nan_occurred"] = True
            results["nan_epoch"] = epoch
            wandb.log({"epoch": epoch, "nan_occurred": 1})

        # Evaluate
        eval_acc, eval_loss = evaluate_model(model, eval_loader, device)

        # Log epoch metrics
        log_data = {
            "epoch": epoch,
            "train/epoch_loss": avg_train_loss if not math.isnan(avg_train_loss) else None,
            "eval/accuracy": eval_acc,
            "eval/loss": eval_loss,
        }
        wandb.log({k: v for k, v in log_data.items() if v is not None})

        # Store epoch results
        epoch_result = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "eval_accuracy": eval_acc,
            "eval_loss": eval_loss,
            "nan_occurred": nan_occurred,
        }
        results["epoch_results"].append(epoch_result)

        # Track best
        if eval_acc > best_acc:
            best_acc = eval_acc

        loss_str = f"{avg_train_loss:.4f}" if not math.isnan(avg_train_loss) else "NaN"
        print(f"Epoch {epoch}: Train Loss={loss_str}, Eval Acc={eval_acc:.4f}, Eval Loss={eval_loss:.4f}")

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
        "final/nan_occurred": int(results["nan_occurred"]),
    })
    wandb.summary["final_accuracy"] = final_acc
    wandb.summary["best_accuracy"] = best_acc
    wandb.summary["improvement"] = improvement
    wandb.summary["nan_occurred"] = results["nan_occurred"]
    if results["nan_epoch"]:
        wandb.summary["nan_epoch"] = results["nan_epoch"]

    wandb.finish()

    # Cleanup
    del model
    torch.cuda.empty_cache()

    nan_status = f" (NaN at epoch {results['nan_epoch']})" if results["nan_occurred"] else ""
    print(f"\n{exp_name} Complete: Final Acc={final_acc:.4f}, Best Acc={best_acc:.4f}, Improvement={improvement:+.4f}{nan_status}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("TikiTaka Dense Subset Experiments: Identify NaN-causing Layers")
    print("Testing individual dense layer groups with gradient clipping")
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

    print(f"\nStability Settings:")
    print(f"  gradient_clip_norm: {GRADIENT_CLIP_NORM}")

    print(f"\nTile Configuration:")
    print(f"  Fast tile (A): SoftBounds (noise=0)")
    print(f"  Slow tile (C): SoftBounds (noise=0)")

    print(f"\nFixed: gamma=0.0, auto_scale=True, units_in_mbatch=False")
    print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}")

    print(f"\n{len(EXPERIMENTS)} Experiments to run:")
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"  {i}. {exp['name']}: {exp['description']}")
        if exp.get('exclude_patterns'):
            print(f"      (excluding: {exp['exclude_patterns']})")

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
        "purpose": "Identify NaN-causing dense layer groups",
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "transfer_lr": TRANSFER_LR,
            "transfer_every": TRANSFER_EVERY,
            "fast_lr": FAST_LR,
            "auto_granularity": AUTO_GRANULARITY,
            "in_chop_prob": IN_CHOP_PROB,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
        },
        "config": {
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "model_name": MODEL_NAME,
            "task_name": TASK_NAME,
            "fast_tile": "SoftBounds (noise=0)",
            "slow_tile": "SoftBounds (noise=0)",
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
    json_path = os.path.join(OUTPUT_DIR, f"tikitaka_dense_subset_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        # Don't save full target_layers list to keep JSON readable
        save_results = all_results.copy()
        for exp in save_results["experiments"]:
            exp["target_layers"] = f"{len(exp['target_layers'])} layers"
        json.dump(save_results, f, indent=2)

    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, f"tikitaka_dense_subset_results_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'experiment', 'description', 'num_target_layers', 'analog_layers',
            'initial_accuracy', 'final_accuracy', 'best_accuracy', 'improvement',
            'nan_occurred', 'nan_epoch',
            'epoch1_acc', 'epoch2_acc', 'epoch3_acc',
            'epoch1_loss', 'epoch2_loss', 'epoch3_loss',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp in all_results["experiments"]:
            row = {
                'experiment': exp['experiment'],
                'description': exp['description'],
                'num_target_layers': exp['num_target_layers'],
                'analog_layers': exp['analog_layers'],
                'initial_accuracy': exp['initial_accuracy'],
                'final_accuracy': exp['final_accuracy'],
                'best_accuracy': exp['best_accuracy'],
                'improvement': exp['improvement'],
                'nan_occurred': exp['nan_occurred'],
                'nan_epoch': exp['nan_epoch'],
            }
            for i, er in enumerate(exp['epoch_results'], 1):
                row[f'epoch{i}_acc'] = er['eval_accuracy']
                row[f'epoch{i}_loss'] = er['eval_loss']
            writer.writerow(row)

    # Print final summary
    print("\n" + "=" * 80)
    print("ALL DENSE SUBSET EXPERIMENTS COMPLETE")
    print("=" * 80)

    print(f"\n{'Experiment':<28} {'Layers':<8} {'Analog':<8} {'Final Acc':<10} {'NaN?':<8} {'NaN Epoch':<10}")
    print("-" * 82)
    for exp in all_results["experiments"]:
        nan_str = "Yes" if exp['nan_occurred'] else "No"
        nan_epoch = str(exp['nan_epoch']) if exp['nan_epoch'] else "-"
        print(f"{exp['experiment']:<28} {exp['num_target_layers']:<8} {exp['analog_layers']:<8} "
              f"{exp['final_accuracy']:<10.4f} {nan_str:<8} {nan_epoch:<10}")

    # Analysis summary
    print("\n" + "-" * 82)
    print("Analysis Summary:")
    nan_experiments = [exp for exp in all_results["experiments"] if exp['nan_occurred']]
    stable_experiments = [exp for exp in all_results["experiments"] if not exp['nan_occurred']]

    if nan_experiments:
        print(f"  NaN occurred in: {[exp['experiment'] for exp in nan_experiments]}")
    else:
        print("  No NaN occurred in any experiment (gradient clipping may have helped)")

    if stable_experiments:
        best_stable = max(stable_experiments, key=lambda x: x['final_accuracy'])
        print(f"  Best stable experiment: {best_stable['experiment']} ({best_stable['final_accuracy']:.2%})")

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
        wandb.summary[f"{exp['experiment']}_nan_occurred"] = exp['nan_occurred']
        wandb.summary[f"{exp['experiment']}_num_layers"] = exp['num_target_layers']

    wandb.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()
