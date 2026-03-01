"""
Digital LoRA baseline test (no analog conversion)

Tests if gradient explosion is specific to analog layers or a general issue.
Uses standard PyTorch LoRA (PEFT library).
"""

import os
import sys
import torch
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import numpy as np

# Configuration
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
RANK = 8
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
SEED = 42

# Test hyperparameters (matching LRTT test)
LORA_ALPHA = 1.0
LEARNING_RATE = 0.0001  # Unified LR for digital LoRA

print("=" * 80)
print("DIGITAL LORA BASELINE TEST (No Analog)")
print("=" * 80)
print(f"\nHyperparameters:")
print(f"  lora_alpha: {LORA_ALPHA}")
print(f"  learning_rate: {LEARNING_RATE}")
print(f"  rank: {RANK}")
print(f"  Implementation: PEFT (standard digital LoRA)")
print()

set_seed(SEED)

# Load data
print("Loading dataset...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = load_dataset("glue", TASK_NAME)

def preprocess_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
    )

train_dataset = dataset["train"].map(preprocess_function, batched=True)
eval_dataset = dataset["validation"].map(preprocess_function, batched=True)

print(f"✓ Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Load model
print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# Apply digital LoRA using PEFT
print("\nApplying digital LoRA (PEFT)...")
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["query", "key", "value"],
    lora_dropout=0.0,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Metric
def compute_metrics(eval_pred):
    metric = evaluate.load("glue", TASK_NAME)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Training
print("\nStarting training...")
training_args = TrainingArguments(
    output_dir="/tmp/digital_lora_baseline",
    evaluation_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.05,
    logging_steps=50,
    report_to="none",
    seed=SEED,
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

try:
    result = trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    # Evaluate
    eval_result = trainer.evaluate()
    print(f"\nFinal Results:")
    print(f"  eval_accuracy: {eval_result['eval_accuracy']:.4f}")
    print(f"  eval_loss: {eval_result['eval_loss']:.4f}")
    print("=" * 80)

except Exception as e:
    print("\n" + "=" * 80)
    print("TRAINING FAILED")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
