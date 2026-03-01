#!/usr/bin/env python
# coding=utf-8
"""
Test single trial to verify training works correctly.
"""

import sys
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_dataset
from aihwkit.optim import AnalogSGD

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

# Settings
MODEL_NAME = "google/mobilebert-uncased"
TASK = "sst2"
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128
NUM_EPOCHS = 1
SEED = 42

# Hyperparameters (safe values)
LORA_ALPHA = 1.0
LEARNING_RATE = 5e-4  # Safe middle value
RANK = 8

print("=" * 80)
print("SINGLE TRIAL TEST")
print("=" * 80)
print(f"Task: {TASK}")
print(f"Mode: sixt1c_lora")
print(f"Learning rate: {LEARNING_RATE}")
print(f"LoRA alpha: {LORA_ALPHA}")
print(f"Rank: {RANK}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print("=" * 80)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Load dataset
print("\n[1/6] Loading dataset...")
dataset = load_dataset("glue", TASK)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, max_length=MAX_SEQ_LENGTH)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
print(f"✓ Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

# Load model
print("\n[2/6] Loading model...")
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
print("✓ Model loaded")

# Create LRTT-LoRA config
print("\n[3/6] Creating LRTT-LoRA config...")
lrtt_config = create_lrtt_lora_config(
    rank=RANK,
    lora_alpha=LORA_ALPHA,
    output_noise_level=0.0,
    use_floating_point=False,  # 6T1C mode
)
print("✓ Config created")

# Convert model
print("\n[4/6] Converting to LRTT-LoRA...")
model = convert_model_to_lrtt_lora(model, lrtt_config, ["query", "key", "value"])
model.to(device)
print("✓ Model converted and moved to device")

# Create optimizer
print("\n[5/6] Setting up training...")
optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
optimizer.regroup_param_groups(model)

# Calculate warmup steps
total_steps = len(train_dataset) // BATCH_SIZE * NUM_EPOCHS
warmup_steps = int(0.1 * total_steps)

# Training arguments
training_args = TrainingArguments(
    output_dir="/tmp/test_single_trial",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="no",
    warmup_steps=warmup_steps,
    seed=SEED,
    report_to="none",
    disable_tqdm=False,  # Show progress bar
)

# Metrics
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, None),
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

print("✓ Trainer created")

# Train
print("\n[6/6] Starting training...")
print("=" * 80)
try:
    train_result = trainer.train()
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Final loss: {train_result.training_loss:.4f}")

    # Evaluate
    print("\nEvaluating...")
    eval_result = trainer.evaluate()
    print(f"✓ Evaluation accuracy: {eval_result['eval_accuracy']:.4f}")
    print("=" * 80)

except Exception as e:
    print("\n" + "=" * 80)
    print("✗ TRAINING FAILED!")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Single trial test PASSED!")
