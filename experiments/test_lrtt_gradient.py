"""
Quick test script for LRTT gradient behavior with specific hyperparameters.

Test configuration:
- classifier_lr = 1.0
- analog_lr (lora_tile_lr) = 0.01
- lora_alpha = 0.01
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lora_training_glue"))

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

from aihwkit.optim import AnalogSGD
from aihwkit.optim.context import AnalogContext

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

# Test hyperparameters (matching original mobilebert_squad_lrtt_scratch.py)
UNIFIED_LR = 0.00362  # Same LR for ALL parameters (original approach)
LORA_ALPHA = 0.01
RANK = 8
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"

print("=" * 80)
print("LRTT GRADIENT TEST")
print("=" * 80)
print(f"\nHyperparameters:")
print(f"  unified_lr (ALL parameters): {UNIFIED_LR}")
print(f"  lora_alpha: {LORA_ALPHA}")
print(f"  rank: {RANK}")
print(f"  model: {MODEL_NAME}")
print(f"  task: {TASK_NAME}")
print()

# 1. Load tokenizer and model
print("[1/6] Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)
print(f"✓ Model loaded: {MODEL_NAME}")

# 2. Convert to LRTT-LoRA
print("\n[2/6] Converting to LRTT-LoRA...")
lrtt_config = create_lrtt_lora_config(
    rank=RANK,
    lora_alpha=LORA_ALPHA,
    output_noise_level=0.0,
    use_floating_point=False,
)
model = convert_model_to_lrtt_lora(
    model,
    lrtt_config,
    target_modules=["query", "key", "value"],
)
model = model.cuda()
print("✓ Model converted and moved to GPU")

# 3. Prepare data (small subset for quick test)
print("\n[3/6] Loading dataset...")
dataset = load_dataset("glue", TASK_NAME)
train_dataset = dataset["train"].select(range(32))  # Only 32 samples

def preprocess_function(examples):
    result = tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    result["labels"] = examples["label"]
    return result

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
)
train_dataset.set_format(type="torch")
print(f"✓ Dataset loaded: {len(train_dataset)} samples")

# 4. Setup optimizer (matching original: single LR for all parameters)
print("\n[4/6] Setting up optimizer...")
print(f"  Using UNIFIED_LR={UNIFIED_LR} for ALL parameters (like original)")

# Original approach: pass model.parameters() with single LR
optimizer = AnalogSGD(
    model.parameters(),
    lr=UNIFIED_LR,
    weight_decay=0.0,
    momentum=0.0,
    nesterov=False
)
optimizer.regroup_param_groups()
print(f"✓ Optimizer created with lr={UNIFIED_LR} for all parameters")

# 5. Training setup
print("\n[5/6] Setting up training...")
training_args = TrainingArguments(
    output_dir="/tmp/lrtt_gradient_test",
    per_device_train_batch_size=8,
    max_steps=10,
    logging_steps=1,
    save_steps=1000,
    save_total_limit=1,
    report_to="none",
    max_grad_norm=1.0,  # Gradient clipping
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=default_data_collator,
    optimizers=(optimizer, None),
)
print("✓ Trainer configured")

# 6. Run training
print("\n[6/6] Running training (10 steps)...")
print("=" * 80)

try:
    trainer.train()
    print("\n" + "=" * 80)
    print("✓ Training completed successfully!")
    print("=" * 80)
except Exception as e:
    print("\n" + "=" * 80)
    print(f"✗ Training failed with error:")
    print(f"  {type(e).__name__}: {e}")
    print("=" * 80)
    import traceback
    traceback.print_exc()

# Print final model stats
print("\n[Final Model Stats]")
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad_norm = param.grad.norm().item()
        param_norm = param.norm().item()
        print(f"  {name:60s} param_norm={param_norm:.4f}, grad_norm={grad_norm:.4f}")
