#!/usr/bin/env python3
"""Minimal test to isolate update issue."""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

from sixt1c_config import (
    gen_sixt1c_lora_config_trainable,
    gen_softbounds_base_layer_config
)
from smart_conversion import convert_base_and_lora_separately

print("Creating model...")
model_config = AutoConfig.from_pretrained('google/mobilebert-uncased')
model_config.num_labels = 2
model_config.num_hidden_layers = 1

model = AutoModelForSequenceClassification.from_pretrained(
    'google/mobilebert-uncased',
    config=model_config,
    ignore_mismatched_sizes=True
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=1.0,
    target_modules=['query'],
    bias='none',
    lora_dropout=0.0,
)
model = get_peft_model(model, peft_config)

# Convert
base_config = gen_softbounds_base_layer_config()
lora_config = gen_sixt1c_lora_config_trainable()

print("\nConverting to analog...")
model = convert_base_and_lora_separately(
    model,
    base_layer_config=base_config,
    lora_config=lora_config,
    lora_trainable=True
)

# Freeze base_layer
print("\nFreezing parameters...")
for name, param in model.named_parameters():
    if 'base_layer' in name or 'classifier' in name:
        param.requires_grad = False
    elif 'lora' in name:
        param.requires_grad = True

# Find modules
lora_a = None
lora_b = None
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        if 'lora_A' in name and lora_a is None:
            lora_a = module
            lora_a_name = name
        elif 'lora_B' in name and lora_b is None:
            lora_b = module
            lora_b_name = name

print(f"\nFound:")
print(f"  lora_A: {lora_a_name}")
print(f"  lora_B: {lora_b_name}")

# Check initial weights
print("\nInitial weights:")
w_a = lora_a.get_weights()
w_a = w_a[0] if isinstance(w_a, tuple) else w_a
print(f"  lora_A: shape={w_a.shape}, norm={w_a.norm().item():.4f}, range=[{w_a.min().item():.4f}, {w_a.max().item():.4f}]")

w_b = lora_b.get_weights()
w_b = w_b[0] if isinstance(w_b, tuple) else w_b
print(f"  lora_B: shape={w_b.shape}, norm={w_b.norm().item():.4f}, range=[{w_b.min().item():.4f}, {w_b.max().item():.4f}]")

# Check if weights are on analog tiles
print("\nChecking tile status...")
print(f"  lora_A analog_module type: {type(lora_a.analog_module)}")
print(f"  lora_A has rpu_config: {hasattr(lora_a.analog_module, 'rpu_config')}")
if hasattr(lora_a.analog_module, 'rpu_config'):
    print(f"  lora_A config type: {type(lora_a.analog_module.rpu_config).__name__}")

# Setup optimizer
print("\nSetting up optimizer...")
optimizer = AnalogSGD(model.parameters(), lr=0.001)
optimizer.regroup_param_groups(model)

# Prepare input
tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
inputs = tokenizer(
    ["Test sentence one.", "Test sentence two."],
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors='pt'
)
labels = torch.tensor([1, 0])

# Training step
print("\n" + "="*80)
print("TRAINING STEP")
print("="*80)

# Capture before
w_a_before = lora_a.get_weights()
w_a_before = (w_a_before[0] if isinstance(w_a_before, tuple) else w_a_before).clone()

w_b_before = lora_b.get_weights()
w_b_before = (w_b_before[0] if isinstance(w_b_before, tuple) else w_b_before).clone()

# Forward + backward
model.train()
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")

loss.backward()

# Check if gradients exist
print("\nChecking gradients...")
grad_found = False
for name, param in model.named_parameters():
    if 'lora_A' in name and 'query' in name and param.requires_grad:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"  {name}: grad_norm={grad_norm:.2e}")
                grad_found = True

if not grad_found:
    print("  ⚠️ No gradients found in lora_A parameters!")

# Optimizer step
print("\nCalling optimizer.step()...")
optimizer.step()

# Capture after
w_a_after = lora_a.get_weights()
w_a_after = (w_a_after[0] if isinstance(w_a_after, tuple) else w_a_after).clone()

w_b_after = lora_b.get_weights()
w_b_after = (w_b_after[0] if isinstance(w_b_after, tuple) else w_b_after).clone()

# Check changes
print("\nWeight changes:")
delta_a = (w_a_after - w_a_before).abs()
delta_b = (w_b_after - w_b_before).abs()

print(f"  lora_A:")
print(f"    max: {delta_a.max().item():.6e}")
print(f"    mean: {delta_a.mean().item():.6e}")
print(f"    changed (>1e-8): {(delta_a > 1e-8).sum().item()} / {delta_a.numel()}")
print(f"    changed (>1e-6): {(delta_a > 1e-6).sum().item()} / {delta_a.numel()}")

print(f"  lora_B:")
print(f"    max: {delta_b.max().item():.6e}")
print(f"    mean: {delta_b.mean().item():.6e}")
print(f"    changed (>1e-8): {(delta_b > 1e-8).sum().item()} / {delta_b.numel()}")
print(f"    changed (>1e-6): {(delta_b > 1e-6).sum().item()} / {delta_b.numel()}")

# Verdict
print("\n" + "="*80)
if delta_a.max().item() > 1e-6 and delta_b.max().item() > 1e-6:
    print("✓ SUCCESS: Both lora_A and lora_B updated!")
elif delta_a.max().item() > 1e-6 or delta_b.max().item() > 1e-6:
    print("⚠️  PARTIAL: Some updates but not both")
else:
    print("✗ FAIL: No significant updates (<1e-6)")
print("="*80)
