#!/usr/bin/env python
"""
Debug script v5 - Check with actual GLUE data and proper training setup
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

print("=" * 60)
print("DEBUG v5 - Proper Setup with GLUE RTE")
print("=" * 60)

# Load RTE dataset
print("\n[1] Loading RTE dataset...")
dataset = load_dataset("glue", "rte", split="train[:10]")
print(f"  Loaded {len(dataset)} examples")
print(f"  First example: {dataset[0]}")

# Load tokenizer and model properly
print("\n[2] Loading model with proper config...")
config = AutoConfig.from_pretrained(
    "google/mobilebert-uncased",
    num_labels=2,
    finetuning_task="rte"
)
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased",
    config=config
)

# Check classifier initialization
print(f"\n[3] Classifier layer:")
print(f"  Weight shape: {model.classifier.weight.shape}")
print(f"  Weight range: [{model.classifier.weight.min().item():.4f}, {model.classifier.weight.max().item():.4f}]")
print(f"  Bias: {model.classifier.bias}")

# Test with proper tokenization
print("\n[4] Testing with RTE example...")
example = dataset[0]
inputs = tokenizer(
    example["sentence1"],
    example["sentence2"],
    padding="max_length",
    max_length=128,
    truncation=True,
    return_tensors="pt"
)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print(f"  Input sentence1: {example['sentence1'][:50]}...")
print(f"  Input sentence2: {example['sentence2'][:50]}...")
print(f"  Label: {example['label']}")
print(f"  Logits: {logits}")
print(f"  Probabilities: {torch.softmax(logits, dim=-1)}")

# Test with multiple examples
print("\n[5] Testing with multiple examples...")
for i in range(min(5, len(dataset))):
    ex = dataset[i]
    inp = tokenizer(ex["sentence1"], ex["sentence2"],
                   padding="max_length", max_length=128,
                   truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    print(f"  Ex {i}: logits={out.logits.numpy().flatten()}, label={ex['label']}")

# Test loss computation
print("\n[6] Testing loss computation...")
labels = torch.tensor([example["label"]])
with torch.no_grad():
    outputs = model(**inputs, labels=labels)
print(f"  Loss: {outputs.loss.item():.4f}")

# Now add LoRA and test again
print("\n[7] Adding LoRA...")
peft_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.1,
    target_modules=["query", "key", "value"]
)
model = get_peft_model(model, peft_config)
model.eval()

with torch.no_grad():
    outputs = model(**inputs, labels=labels)
print(f"  Logits with LoRA: {outputs.logits}")
print(f"  Loss with LoRA: {outputs.loss.item():.4f}")

# Test training step
print("\n[8] Testing one training step...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

for step in range(3):
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"  Step {step}: Loss is NaN/Inf!")
        break

    loss.backward()

    # Check grad norm
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    print(f"  Step {step}: loss={loss.item():.4f}, grad_norm={total_norm:.4f}")

    if np.isnan(total_norm) or np.isinf(total_norm):
        print(f"  Gradient is NaN/Inf!")
        break

    optimizer.step()

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
