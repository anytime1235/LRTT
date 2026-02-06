#!/usr/bin/env python
# coding=utf-8
"""BERT-tiny GLUE LRTT Accuracy Sweep.

Verify LRTT fine-tuning improves accuracy over baseline.
- Compare pre-trained baseline vs LRTT fine-tuned
- Sweep: transfer_every (te), learning_rate (lr), transfer_lr (tlr)
- 10 experiments with rank=4

Configuration (from sweep_softbounds_lifetime.py):
- A/B tiles: LinearStepDevice (6T1C)
- C tile: SoftBoundsDevice (no noise)
- Mode: decay, forward_inject=False
"""

import math
import os
import sys
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add LRTT src to path for local development (optional)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LRTT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

import datasets
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

# =============================================================================
# Configuration
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "prajjwal1/bert-tiny"
TASK_NAME = "sst2"
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128
NUM_EPOCHS = 3
SEED = 42
RANK = 4

# SoftBounds config (no noise)
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}

# 10 Experiment configurations: (te, lr, tlr)
EXPERIMENTS = [
    # Varying transfer_every
    {"te": 10,   "lr": 0.01,  "tlr": 0.01},
    {"te": 50,   "lr": 0.01,  "tlr": 0.01},
    {"te": 100,  "lr": 0.01,  "tlr": 0.01},
    {"te": 500,  "lr": 0.01,  "tlr": 0.01},
    # Varying learning rate
    {"te": 100,  "lr": 0.001, "tlr": 0.01},
    {"te": 100,  "lr": 0.05,  "tlr": 0.01},
    {"te": 100,  "lr": 0.1,   "tlr": 0.01},
    # Varying transfer_lr
    {"te": 100,  "lr": 0.01,  "tlr": 0.001},
    {"te": 100,  "lr": 0.01,  "tlr": 0.1},
    {"te": 100,  "lr": 0.01,  "tlr": 1.0},
]


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    """Convert lifetime to dt_batch_sec."""
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    return -TAU_SEC * math.log(1 - delta)


def create_lrtt_config(rank: int, te: int, tlr: float) -> PythonLRTTRPUConfig:
    """Create LRTT config with SoftBounds C + 6T1C A/B."""
    lifetime = 46505.0
    dt_batch_sec = lifetime_to_dt_batch_sec(lifetime)
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    ab_lifetime = 1.0 / delta if delta > 0 else 0.0

    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime, lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    return PythonLRTTRPUConfig(device=device_config)


def load_data(tokenizer):
    """Load and preprocess SST-2 dataset."""
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
        )

    tokenized = raw_datasets.map(preprocess, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(
        tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        tokenized["validation"], batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )
    return train_loader, val_loader


def evaluate_model(model, val_loader):
    """Evaluate accuracy on validation set."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def run_baseline(tokenizer, train_loader, val_loader):
    """Evaluate pre-trained model without fine-tuning."""
    print("\n" + "=" * 60)
    print("BASELINE: Pre-trained BERT-tiny (no fine-tuning)")
    print("=" * 60)

    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    model.to(DEVICE)

    acc = evaluate_model(model, val_loader)
    print(f"Pre-trained accuracy: {acc:.2f}%")

    del model
    torch.cuda.empty_cache()
    return acc


def run_experiment(exp_config, tokenizer, train_loader, val_loader, exp_id):
    """Run single LRTT experiment."""
    te, lr, tlr = exp_config["te"], exp_config["lr"], exp_config["tlr"]

    print(f"\n{'=' * 60}")
    print(f"Experiment {exp_id}: te={te}, lr={lr}, tlr={tlr}")
    print("=" * 60)

    set_seed(SEED)

    # Load fresh model
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # Pre-training accuracy
    model.to(DEVICE)
    pre_acc = evaluate_model(model, val_loader)
    print(f"Before LRTT fine-tuning: {pre_acc:.2f}%")

    # Convert to LRTT
    rpu_config = create_lrtt_config(RANK, te, tlr)
    model = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
    model.to(DEVICE)

    # Count LRTT layers
    lrtt_count = sum(1 for _, m in model.named_modules()
                     if hasattr(m, 'analog_module') and hasattr(m.analog_module, 'controller'))
    print(f"LRTT layers: {lrtt_count}")

    # Create optimizer
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training
    best_acc = pre_acc
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        acc = evaluate_model(model, val_loader)
        best_acc = max(best_acc, acc)
        print(f"  Epoch {epoch}: loss={loss:.4f}, acc={acc:.2f}%")

    print(f"Best accuracy: {best_acc:.2f}% (improvement: {best_acc - pre_acc:+.2f}%)")

    result = {
        "exp_id": exp_id,
        "te": te,
        "lr": lr,
        "tlr": tlr,
        "pre_acc": pre_acc,
        "best_acc": best_acc,
        "improvement": best_acc - pre_acc,
    }

    del model
    torch.cuda.empty_cache()
    return result


def main():
    print("=" * 60)
    print("BERT-tiny LRTT Accuracy Sweep")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Rank: {RANK}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Device: {DEVICE}")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print("=" * 60)

    set_seed(SEED)

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Run baseline
    baseline_acc = run_baseline(tokenizer, train_loader, val_loader)

    # Run experiments
    results = []
    for i, exp_config in enumerate(EXPERIMENTS, 1):
        result = run_experiment(exp_config, tokenizer, train_loader, val_loader, i)
        result["baseline_acc"] = baseline_acc
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline (pre-trained): {baseline_acc:.2f}%\n")
    print(f"{'Exp':<4} {'TE':<6} {'LR':<8} {'TLR':<8} {'Pre':<8} {'Best':<8} {'Δ':<8}")
    print("-" * 60)

    for r in results:
        print(f"{r['exp_id']:<4} {r['te']:<6} {r['lr']:<8.4f} {r['tlr']:<8.4f} "
              f"{r['pre_acc']:<8.2f} {r['best_acc']:<8.2f} {r['improvement']:+8.2f}")

    # Find best
    best = max(results, key=lambda x: x["best_acc"])
    print("-" * 60)
    print(f"Best: Exp {best['exp_id']} (te={best['te']}, lr={best['lr']}, tlr={best['tlr']})")
    print(f"      Accuracy: {best['best_acc']:.2f}% (Δ={best['improvement']:+.2f}%)")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"/tmp/bert_tiny_sweep_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({"baseline": baseline_acc, "experiments": results}, f, indent=2)
    print(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    main()
