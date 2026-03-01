#!/usr/bin/env python3
"""Measure RAW tile weight changes, not scaled weights."""

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

# Create model
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

model = convert_base_and_lora_separately(
    model,
    base_layer_config=base_config,
    lora_config=lora_config,
    lora_trainable=True
)

# Freeze
for name, param in model.named_parameters():
    if 'base_layer' in name or 'classifier' in name:
        param.requires_grad = False
    elif 'lora' in name:
        param.requires_grad = True

# Find modules
lora_a = lora_b = None
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        if 'lora_A' in name and lora_a is None:
            lora_a = module
            lora_a_name = name
        elif 'lora_B' in name and lora_b is None:
            lora_b = module
            lora_b_name = name

print("=" * 80)
print("MEASURING RAW TILE WEIGHT CHANGES")
print("=" * 80)

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

# Capture RAW tile weights before
w_a_tile_before = lora_a.analog_module.tile.get_weights().clone()
w_b_tile_before = lora_b.analog_module.tile.get_weights().clone()

# Also capture scaled weights for comparison
w_a_scaled_before = lora_a.get_weights()
w_a_scaled_before = (w_a_scaled_before[0] if isinstance(w_a_scaled_before, tuple) else w_a_scaled_before).clone()

w_b_scaled_before = lora_b.get_weights()
w_b_scaled_before = (w_b_scaled_before[0] if isinstance(w_b_scaled_before, tuple) else w_b_scaled_before).clone()

print(f"\nBefore training:")
print(f"  lora_A RAW tile weights: norm={w_a_tile_before.norm().item():.4f}, range=[{w_a_tile_before.min().item():.4f}, {w_a_tile_before.max().item():.4f}]")
print(f"  lora_A scaled weights:   norm={w_a_scaled_before.norm().item():.4f}, range=[{w_a_scaled_before.min().item():.4f}, {w_a_scaled_before.max().item():.4f}]")

# Training step
model.train()
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
print(f"\nLoss: {loss.item():.4f}")

loss.backward()
optimizer.step()

# Capture RAW tile weights after
w_a_tile_after = lora_a.analog_module.tile.get_weights().clone()
w_b_tile_after = lora_b.analog_module.tile.get_weights().clone()

# Also capture scaled weights after
w_a_scaled_after = lora_a.get_weights()
w_a_scaled_after = (w_a_scaled_after[0] if isinstance(w_a_scaled_after, tuple) else w_a_scaled_after).clone()

w_b_scaled_after = lora_b.get_weights()
w_b_scaled_after = (w_b_scaled_after[0] if isinstance(w_b_scaled_after, tuple) else w_b_scaled_after).clone()

# Compute deltas
delta_a_tile = (w_a_tile_after - w_a_tile_before).abs()
delta_b_tile = (w_b_tile_after - w_b_tile_before).abs()

delta_a_scaled = (w_a_scaled_after - w_a_scaled_before).abs()
delta_b_scaled = (w_b_scaled_after - w_b_scaled_before).abs()

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nlora_A RAW TILE weights:")
print(f"  max change: {delta_a_tile.max().item():.6f}")
print(f"  mean change: {delta_a_tile.mean().item():.6f}")
print(f"  changed (>dw_min=0.002): {(delta_a_tile > 0.002).sum().item()} / {delta_a_tile.numel()}")
print(f"  changed (>1e-6): {(delta_a_tile > 1e-6).sum().item()} / {delta_a_tile.numel()}")

print(f"\nlora_A SCALED weights (what we measured before):")
print(f"  max change: {delta_a_scaled.max().item():.6e}")
print(f"  mean change: {delta_a_scaled.mean().item():.6e}")
print(f"  Ratio (scaled/tile): {delta_a_scaled.max().item() / delta_a_tile.max().item() if delta_a_tile.max() > 0 else 0:.6f}")

print(f"\nlora_B RAW TILE weights:")
print(f"  max change: {delta_b_tile.max().item():.6f}")
print(f"  mean change: {delta_b_tile.mean().item():.6f}")
print(f"  changed (>dw_min=0.002): {(delta_b_tile > 0.002).sum().item()} / {delta_b_tile.numel()}")
print(f"  changed (>1e-6): {(delta_b_tile > 1e-6).sum().item()} / {delta_b_tile.numel()}")

print(f"\nlora_B SCALED weights (what we measured before):")
print(f"  max change: {delta_b_scaled.max().item():.6e}")
print(f"  mean change: {delta_b_scaled.mean().item():.6e}")
print(f"  Ratio (scaled/tile): {delta_b_scaled.max().item() / delta_b_tile.max().item() if delta_b_tile.max() > 0 else 0:.6f}")

print("\n" + "=" * 80)
if delta_a_tile.max().item() > 0.002 or delta_b_tile.max().item() > 0.002:
    print("✓ SUCCESS: RAW tile weights ARE updating (>dw_min)!")
    print("\nConclusion:")
    print("  - Analog tiles ARE working correctly")
    print("  - The issue was we were measuring SCALED weights")
    print("  - Scaled weights have smaller changes due to weight_scaling_omega")
elif delta_a_tile.max().item() > 1e-6 or delta_b_tile.max().item() > 1e-6:
    print("⚠️  PARTIAL: Tile weights updating but below dw_min")
    print("\nThis shouldn't happen - updates should be >= dw_min")
else:
    print("✗ FAIL: Even RAW tile weights not updating")
print("=" * 80)
