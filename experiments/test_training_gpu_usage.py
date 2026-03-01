#!/usr/bin/env python
"""Test if training actually uses GPU"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
import time
from transformers import AutoModelForSequenceClassification
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.optim import AnalogSGD

MODEL_NAME = "google/mobilebert-uncased"
device = torch.device("cuda")

print("=" * 80)
print("TESTING TRAINING GPU USAGE")
print("=" * 80)

# Load and convert model
print("\n[1] Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
lrtt_config = create_lrtt_lora_config(rank=8, lora_alpha=1.0, use_floating_point=False)
model = convert_model_to_lrtt_lora(model, lrtt_config, ["query", "key", "value"])

print("\n[2] Moving model to CUDA...")
model.to(device)

print("\n[3] Creating optimizer...")
optimizer = AnalogSGD(model.parameters(), lr=1e-3)
optimizer.regroup_param_groups(model)

print("\n[4] Running 10 training steps...")
model.train()

# Create dummy data
batch_size = 32
seq_length = 128

times = []
for step in range(10):
    # Create dummy batch on GPU
    input_ids = torch.randint(0, 30522, (batch_size, seq_length), device=device)
    attention_mask = torch.ones(batch_size, seq_length, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)

    start = time.time()

    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    elapsed = time.time() - start
    times.append(elapsed)

    print(f"  Step {step}: {elapsed:.3f}s, loss={loss.item():.4f}")

print(f"\n[Results]")
print(f"  Average time: {sum(times)/len(times):.3f}s")
print(f"  Min time: {min(times):.3f}s")
print(f"  Max time: {max(times):.3f}s")
print(f"  Throughput: {1/(sum(times)/len(times)):.2f} it/s")

print("\n✓ If training is fast (~0.5s/it or less), GPU is being used!")
print("✗ If training is slow (~10s/it), tiles are still on CPU!")

print("=" * 80)
