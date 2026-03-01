"""
Quick test with lora_alpha=0.000001 (extremely small)

This tests if very small alpha prevents gradient explosion.
"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lora_training_glue"))

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)
import evaluate

from aihwkit.optim import AnalogSGD
from aihwkit.optim.context import AnalogContext

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

# Configuration
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
RANK = 8
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
SEED = 42

# Test hyperparameters
LORA_ALPHA = 0.000001  # Extremely small
ANALOG_LR = 0.0001
CLASSIFIER_LR = 0.01

print("=" * 80)
print("LRTT TEST: lora_alpha=0.000001")
print("=" * 80)
print(f"\nHyperparameters:")
print(f"  lora_alpha: {LORA_ALPHA}")
print(f"  analog_lr: {ANALOG_LR}")
print(f"  classifier_lr: {CLASSIFIER_LR}")
print(f"  rank: {RANK}")
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

# Convert to LRTT-LoRA
print("\nConverting to LRTT-LoRA...")
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

# Setup optimizer
print("\nSetting up optimizer...")
param_groups = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    if isinstance(param, AnalogContext):
        param_groups.append({"params": [param], "lr": ANALOG_LR})
    elif "out_scaling_alpha" in name:
        param_groups.append({"params": [param], "lr": ANALOG_LR})
    else:
        param_groups.append({"params": [param], "lr": CLASSIFIER_LR})

optimizer = AnalogSGD(param_groups, lr=ANALOG_LR)
print(f"✓ Optimizer created")

# Metric
def compute_metrics(eval_pred):
    metric = evaluate.load("glue", TASK_NAME)
    logits, labels = eval_pred
    import numpy as np
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Training
print("\nStarting training...")
training_args = TrainingArguments(
    output_dir="/tmp/lrtt_alpha_tiny_test",
    evaluation_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
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
    optimizers=(optimizer, None),
)

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
