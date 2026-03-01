#!/usr/bin/env python
"""
Test decoupled learning rates: classifier lr=1.0, QKV lr=0.01, lora_alpha=10
Single trial on SST-2 to verify parameter group functionality.
"""

import sys
import os
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lora_training_glue"))

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from datasets import load_dataset
from aihwkit.optim import AnalogSGD
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
import numpy as np
from sklearn.metrics import accuracy_score

# Configuration
MODEL_NAME = "google/mobilebert-uncased"
TASK = "sst2"
RANK = 8
LORA_ALPHA = 10.0  # Forward/backward scaling
CLASSIFIER_LR = 1.0  # Classifier learning rate
QKV_LR = 0.01  # QKV (tile_a/tile_b) learning rate
BATCH_SIZE = 64
NUM_EPOCHS = 3
TARGET_MODULES = ["query", "key", "value", "classifier"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("DECOUPLED LR TEST")
print("=" * 80)
print(f"Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Task: {TASK}")
print(f"  LoRA rank: {RANK}")
print(f"  LoRA alpha (forward/backward): {LORA_ALPHA}")
print(f"  Classifier LR: {CLASSIFIER_LR}")
print(f"  QKV (tile) LR: {QKV_LR}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Target modules: {TARGET_MODULES}")
print("=" * 80)

# Load dataset
print("\nLoading SST-2 dataset...")
dataset = load_dataset("glue", TASK)
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# Load tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["sentence", "idx"])
eval_dataset = eval_dataset.remove_columns(["sentence", "idx"])
train_dataset = train_dataset.rename_column("label", "labels")
eval_dataset = eval_dataset.rename_column("label", "labels")
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

# Convert to LRTT-LoRA
print("\nConverting to LRTT-LoRA...")
lrtt_config = create_lrtt_lora_config(
    rank=RANK,
    lora_alpha=LORA_ALPHA,
    output_noise_level=0.0,
    use_floating_point=False,
)
model = convert_model_to_lrtt_lora(model, lrtt_config, TARGET_MODULES)
model.to(device)

# Create optimizer with decoupled learning rates
print("\nCreating optimizer with parameter groups...")
lora_tile_params = []
classifier_params = []
other_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if 'tile_a' in name or 'tile_b' in name:
            lora_tile_params.append(param)
        elif 'classifier' in name:
            classifier_params.append(param)
        else:
            other_params.append(param)

print(f"  LoRA tile params (QKV): {len(lora_tile_params)} params, lr={QKV_LR}")
print(f"  Classifier params: {len(classifier_params)} params, lr={CLASSIFIER_LR}")
print(f"  Other params: {len(other_params)} params")

# Create parameter groups
param_groups = [
    {'params': lora_tile_params, 'lr': QKV_LR},
    {'params': classifier_params + other_params, 'lr': CLASSIFIER_LR}
]

optimizer = AnalogSGD(param_groups, lr=CLASSIFIER_LR, momentum=0)
optimizer.regroup_param_groups(model)

# Compute metrics
def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc}

# Training arguments
training_args = TrainingArguments(
    output_dir="/tmp/test_decoupled_lr",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    warmup_ratio=0.05,
    report_to="none",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None),
)

# Train
print("\nStarting training...")
print("=" * 80)
trainer.train()

# Final evaluation
print("\n" + "=" * 80)
print("FINAL EVALUATION")
print("=" * 80)
final_metrics = trainer.evaluate()
print(f"Final accuracy: {final_metrics['eval_accuracy']:.4f}")
print(f"Final loss: {final_metrics['eval_loss']:.4f}")
print("=" * 80)
