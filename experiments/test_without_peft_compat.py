#!/usr/bin/env python3
"""Test analog LoRA without PEFT compatibility layer to see if updates work."""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

from sixt1c_config import (
    gen_sixt1c_lora_config_trainable,
    gen_softbounds_base_layer_config
)

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

# Manual conversion WITHOUT PEFT compatibility
print("\nManual conversion (without PEFT compat)...")

def get_parent_module(model, layer_name):
    parts = layer_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]

base_config = gen_softbounds_base_layer_config()
lora_config = gen_sixt1c_lora_config_trainable()

# Convert base_layer
for name, module in list(model.named_modules()):
    if isinstance(module, nn.Linear) and 'base_layer' in name:
        parent, attr = get_parent_module(model, name)
        digital_layer = getattr(parent, attr)

        analog_layer = AnalogLinear.from_digital(digital_layer, base_config)

        # NO PEFT compatibility - just set requires_grad=False
        for param in analog_layer.parameters():
            param.requires_grad = False

        setattr(parent, attr, analog_layer)
        print(f"  Converted base_layer: {name}")

# Convert lora_A/B
for name, module in list(model.named_modules()):
    if isinstance(module, nn.Linear) and ('lora_A' in name or 'lora_B' in name):
        parent, attr = get_parent_module(model, name)
        digital_layer = getattr(parent, attr)

        analog_layer = AnalogLinear.from_digital(digital_layer, lora_config)

        # NO PEFT compatibility - just set requires_grad=True
        for param in analog_layer.parameters():
            param.requires_grad = True

        setattr(parent, attr, analog_layer)
        print(f"  Converted lora: {name}")

print("✓ Conversion complete (without PEFT compat)")

# Find modules
lora_a = None
lora_b = None
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear):
        if 'lora_A' in name and lora_a is None:
            lora_a = module
            lora_a_name = name
        elif 'lora_B' in name and lora_b is None:
            lora_b = module
            lora_b_name = name

print(f"\nTarget modules:")
print(f"  lora_A: {lora_a_name}")
print(f"  lora_B: {lora_b_name}")

# Check if .weight attribute exists
print(f"\nChecking attributes:")
print(f"  lora_A has .weight: {hasattr(lora_a, 'weight')}")
if hasattr(lora_a, 'weight'):
    print(f"    .weight type: {type(lora_a.weight)}")

# Setup optimizer
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

# Check gradients
print("\nChecking gradients...")
for name, param in model.named_parameters():
    if 'lora' in name and 'query' in name and param.requires_grad:
        if param.grad is not None and param.grad.norm() > 0:
            print(f"  {name}: grad_norm={param.grad.norm().item():.2e}")

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
print(f"    changed (>1e-6): {(delta_a > 1e-6).sum().item()} / {delta_a.numel()}")

print(f"  lora_B:")
print(f"    max: {delta_b.max().item():.6e}")
print(f"    changed (>1e-6): {(delta_b > 1e-6).sum().item()} / {delta_b.numel()}")

# Verdict
print("\n" + "="*80)
if delta_a.max().item() > 1e-6 and delta_b.max().item() > 1e-6:
    print("✓ SUCCESS: Both lora_A and lora_B updated!")
    print("\n→ The issue WAS the PEFT compatibility layer!")
elif delta_a.max().item() > 1e-6 or delta_b.max().item() > 1e-6:
    print("⚠️  PARTIAL: Some updates but not both")
else:
    print("✗ FAIL: No significant updates (<1e-6)")
    print("\n→ The issue is NOT the PEFT compatibility layer")
print("="*80)
