#!/home/jovyan/work/ml/.venv310/bin/python
# coding=utf-8
"""Digital Merge with AdamW Optimizer (HuggingFace Trainer style).

Single run with optimal hyperparameters from Trial #42.
Uses AdamW (torch fused) instead of AnalogAdam for fair comparison with lora_on_analog_hardware.

Key settings:
- Optimizer: AdamW (torch fused)
- Learning Rate: 0.000659
- Transfer LR: 0.00531
- Transfer Every: 63
- Rank: 8
- LoRA Alpha: 1.0
- Target Modules: ["query"]
- Epochs: 1
- Model: MobileBERT
- Dataset: SST-2
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

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
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.devices import FloatingPointDevice, SoftBoundsDevice

# LRTT imports
sys.path.insert(0, '/home/jovyan/work/LRTT/src')
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

# =============================================================================
# Optimal Hyperparameters (Trial #42)
# =============================================================================

LEARNING_RATE = 0.000659
TRANSFER_LR = 0.00531
TRANSFER_EVERY = 63

# AdamW parameters (matching HuggingFace Trainer defaults)
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-08

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

# Output
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

def create_digital_merge_config(
    rank: int,
    transfer_every: int,
    transfer_lr: float,
    lora_alpha: float,
) -> PythonLRTTRPUConfig:
    """Create LRTT config with Digital A,B + Analog C (digital merge style).

    Key settings:
    - A, B tiles: FloatingPointDevice (digital, exact computation)
    - C tile: SoftBoundsDevice (analog)
    - forward_inject: True (ReLoRA style: y = C(x) + alpha * A(B(x)))
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
    """Create LRTT model with digital merge configuration."""
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
    """Evaluate model and return accuracy and loss."""
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
    """Train for one epoch and return average loss and batch losses."""
    model.train()
    total_loss, num_batches = 0.0, 0
    batch_losses = []
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
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
# Main
# =============================================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("Digital Merge with AdamW Optimizer (HuggingFace Trainer style)")
    print("=" * 80)
    print(f"\nTimestamp: {timestamp}")
    print("\nOptimal Hyperparameters (Trial #42):")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Transfer LR: {TRANSFER_LR}")
    print(f"  Transfer Every: {TRANSFER_EVERY}")
    print(f"  Rank: {RANK}")
    print(f"  LoRA Alpha: {LORA_ALPHA}")
    print(f"  Target Modules: {TARGET_MODULES}")
    print("\nOptimizer Settings:")
    print(f"  Optimizer: AdamW")
    print(f"  adam_beta1: {ADAM_BETA1}")
    print(f"  adam_beta2: {ADAM_BETA2}")
    print(f"  adam_epsilon: {ADAM_EPSILON}")
    print("\nModel Settings:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Dataset: {TASK_NAME}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  forward_inject: True")
    print(f"  transfer_method: onehot")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load data
    print("\n" + "-" * 40)
    print("Loading data...")
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)

    def preprocess(examples):
        return tokenizer(examples["sentence"], padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True)

    tokenized = raw_datasets.map(preprocess, batched=True)
    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)
    eval_loader = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=default_data_collator)

    print(f"Train samples: {len(tokenized['train'])}")
    print(f"Eval samples: {len(tokenized['validation'])}")

    # Create model
    print("\n" + "-" * 40)
    print("Creating model...")
    model = create_lrtt_model(
        rank=RANK, transfer_every=TRANSFER_EVERY, transfer_lr=TRANSFER_LR,
        lora_alpha=LORA_ALPHA, target_modules=TARGET_MODULES, device=device,
    )

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Create optimizer - AdamW (matching HuggingFace Trainer)
    # Note: fused=False for compatibility with LRTT analog layers
    print("\n" + "-" * 40)
    print("Creating AdamW optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON,
        fused=False,
    )

    # Initial evaluation
    print("\n" + "-" * 40)
    print("Initial evaluation...")
    init_acc, init_loss = evaluate_model(model, eval_loader, device)
    print(f"Initial Accuracy: {init_acc:.4f}")
    print(f"Initial Loss: {init_loss:.4f}")

    # Training
    print("\n" + "-" * 40)
    print("Training...")
    all_batch_losses = []
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_train_loss, batch_losses = train_epoch(model, optimizer, train_loader, device, epoch)
        all_batch_losses.extend(batch_losses)

        eval_acc, eval_loss = evaluate_model(model, eval_loader, device)
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Eval Accuracy: {eval_acc:.4f}")
        print(f"  Eval Loss: {eval_loss:.4f}")

    # Final evaluation
    print("\n" + "-" * 40)
    print("Final evaluation...")
    final_acc, final_loss = evaluate_model(model, eval_loader, device)
    improvement = final_acc - init_acc

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nInitial Accuracy: {init_acc:.4f}")
    print(f"Final Accuracy: {final_acc:.4f}")
    print(f"Improvement: {improvement:+.4f}")
    print(f"\nInitial Loss: {init_loss:.4f}")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"\nReference (AnalogAdam): 0.5161")
    print(f"Difference from AnalogAdam: {final_acc - 0.5161:+.4f}")

    # Save results
    result = {
        "timestamp": timestamp,
        "experiment": "digital_merge_adamw",
        "optimizer": "AdamW",
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "adam_beta1": ADAM_BETA1,
            "adam_beta2": ADAM_BETA2,
            "adam_epsilon": ADAM_EPSILON,
            "transfer_lr": TRANSFER_LR,
            "transfer_every": TRANSFER_EVERY,
            "rank": RANK,
            "lora_alpha": LORA_ALPHA,
            "target_modules": TARGET_MODULES,
        },
        "model": {
            "name": MODEL_NAME,
            "task": TASK_NAME,
            "batch_size": BATCH_SIZE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "num_epochs": NUM_EPOCHS,
        },
        "config": {
            "forward_inject": True,
            "transfer_method": "onehot",
            "device_type": "digital_merge",
        },
        "results": {
            "initial_accuracy": init_acc,
            "final_accuracy": final_acc,
            "improvement": improvement,
            "initial_loss": init_loss,
            "final_loss": final_loss,
        },
        "comparison": {
            "analog_adam_accuracy": 0.5161,
            "difference_from_analog_adam": final_acc - 0.5161,
        },
    }

    output_path = os.path.join(OUTPUT_DIR, "digital_merge_adamw_result.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
