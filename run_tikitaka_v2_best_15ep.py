#!/usr/bin/env python
# coding=utf-8
"""TikiTaka v2 Best Hyperparameters - 15 Epoch Training.

Uses the optimal hyperparameters from Trial 48 of the Bayesian sweep.
6T1C Fast Tile + SoftBoundsReference Slow Tile with chopped transfer.

Best Hyperparameters (Trial 48):
- learning_rate: 3.710e-04
- transfer_lr: 1.362
- transfer_every: 100
- fast_lr: 0.926
- auto_granularity: 107.09
- in_chop_prob: 0.071

Fixed settings:
- target_modules: ["query"] (Q only)
- epochs: 15
- gamma: 0.0 (slow tile only visible)
- auto_scale: True
- units_in_mbatch: False (mat-vec units)
- model: MobileBERT
- dataset: SST-2
"""

import os
import sys
import json
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
# Best Hyperparameters from Trial 48
# =============================================================================

LEARNING_RATE = 3.710e-04
TRANSFER_LR = 1.362
TRANSFER_EVERY = 100
FAST_LR = 0.926
AUTO_GRANULARITY = 107.09
IN_CHOP_PROB = 0.071

# =============================================================================
# Fixed Parameters
# =============================================================================

NUM_EPOCHS = 15
TARGET_MODULES = ["query"]
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
SEED = 42

# WandB settings
WANDB_PROJECT = "tikitaka-v2-best-15ep"
WANDB_ENTITY = None

# Output directory
OUTPUT_DIR = "/data"


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
    device: torch.device, epoch: int,
) -> Tuple[float, List[float]]:
    model.train()
    total_loss, num_batches = 0.0, 0
    batch_losses = []
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=True)
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
# Main Training Loop
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("TikiTaka v2 Best Hyperparameters - 15 Epoch Training")
    print("6T1C Fast Tile + SoftBoundsReference Slow Tile")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print(f"\nBest Hyperparameters (Trial 48):")
    print(f"  learning_rate: {LEARNING_RATE:.3e}")
    print(f"  transfer_lr: {TRANSFER_LR}")
    print(f"  transfer_every: {TRANSFER_EVERY}")
    print(f"  fast_lr: {FAST_LR}")
    print(f"  auto_granularity: {AUTO_GRANULARITY}")
    print(f"  in_chop_prob: {IN_CHOP_PROB}")
    print(f"\nFixed: gamma=0.0, auto_scale=True, units_in_mbatch=False")
    print(f"Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}")
    print(f"Target modules: {TARGET_MODULES}")

    # Initialize WandB
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=f"best_15ep_{timestamp}",
        config={
            "learning_rate": LEARNING_RATE,
            "transfer_lr": TRANSFER_LR,
            "transfer_every": TRANSFER_EVERY,
            "fast_lr": FAST_LR,
            "auto_granularity": AUTO_GRANULARITY,
            "in_chop_prob": IN_CHOP_PROB,
            "num_epochs": NUM_EPOCHS,
            "target_modules": TARGET_MODULES,
            "model_name": MODEL_NAME,
            "task_name": TASK_NAME,
            "batch_size": BATCH_SIZE,
            "gamma": 0.0,
            "auto_scale": True,
            "units_in_mbatch": False,
            "device_type": "tikitaka_v2_6t1c_softbounds",
            "seed": SEED,
        },
    )

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

    # Create model
    print("\nCreating model...")
    model = create_tikitaka_v2_model(
        transfer_every=TRANSFER_EVERY,
        transfer_lr=TRANSFER_LR,
        fast_lr=FAST_LR,
        auto_granularity=AUTO_GRANULARITY,
        in_chop_prob=IN_CHOP_PROB,
        target_modules=TARGET_MODULES,
        device=device,
    )
    optimizer = AnalogAdam(model.parameters(), lr=LEARNING_RATE)

    # Training results storage
    results = {
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
            "target_modules": TARGET_MODULES,
            "model_name": MODEL_NAME,
            "task_name": TASK_NAME,
            "seed": SEED,
        },
        "epoch_results": [],
        "timestamp": timestamp,
    }

    # Initial evaluation
    print("\nInitial evaluation...")
    init_acc, init_loss = evaluate_model(model, eval_loader, device)
    print(f"Initial - Accuracy: {init_acc:.4f}, Loss: {init_loss:.4f}")
    wandb.log({"epoch": 0, "eval/accuracy": init_acc, "eval/loss": init_loss})

    results["initial_accuracy"] = init_acc
    results["initial_loss"] = init_loss

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    global_step = 0
    best_acc = init_acc

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")

        # Train
        avg_train_loss, batch_losses = train_epoch(
            model, optimizer, train_loader, device, epoch
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

    # Save results to JSON
    json_path = os.path.join(OUTPUT_DIR, "tikitaka_v2_best_15ep_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nInitial Accuracy: {init_acc:.4f}")
    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Improvement: {improvement:+.4f}")
    print(f"\nResults saved to: {json_path}")

    # Print epoch-by-epoch summary
    print("\n--- Epoch-by-Epoch Summary ---")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Eval Acc':<12} {'Eval Loss':<12}")
    print("-" * 44)
    for er in results["epoch_results"]:
        print(f"{er['epoch']:<8} {er['train_loss']:<12.4f} {er['eval_accuracy']:<12.4f} {er['eval_loss']:<12.4f}")

    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
