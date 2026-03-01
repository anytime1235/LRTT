#!/usr/bin/env python3
"""Manually call tile.update() to test if it works."""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear

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

# Find lora_A
lora_a = None
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'lora_A' in name and 'query' in name:
        lora_a = module
        lora_a_name = name
        break

print("=" * 80)
print("MANUALLY CALLING TILE.UPDATE()")
print("=" * 80)

tile = lora_a.analog_module.tile

# Set learning rate
tile.set_learning_rate(0.1)  # Use larger LR for visibility
lr = tile.get_learning_rate()
print(f"\nTile learning rate: {lr}")

# Get weights before
w_before = tile.get_weights().clone()
print(f"\nWeights before:")
print(f"  norm: {w_before.norm().item():.4f}")
print(f"  range: [{w_before.min().item():.4f}, {w_before.max().item():.4f}]")

# Create synthetic inputs and gradients
batch_size = 2
seq_len = 128
d_in = 128  # input size
d_out = 8   # output size (rank)

x_input = torch.randn(batch_size * seq_len, d_in)  # Shape: [256, 128]
d_output = torch.randn(batch_size * seq_len, d_out)  # Shape: [256, 8]

print(f"\nInput shapes:")
print(f"  x_input: {x_input.shape}, norm: {x_input.norm().item():.4f}")
print(f"  d_output: {d_output.shape}, norm: {d_output.norm().item():.4f}")

# Call tile.update() with bias=False
print(f"\n[Calling tile.update(x_input, d_output, bias=False)]")
try:
    tile.update(x_input, d_output, bias=False)
    print("  ✓ update() succeeded")
except Exception as e:
    print(f"  ✗ update() failed: {e}")

# Get weights after
w_after = tile.get_weights().clone()
print(f"\nWeights after:")
print(f"  norm: {w_after.norm().item():.4f}")
print(f"  range: [{w_after.min().item():.4f}, {w_after.max().item():.4f}]")

# Check changes
delta = (w_after - w_before).abs()
print(f"\nWeight changes:")
print(f"  max: {delta.max().item():.6f}")
print(f"  mean: {delta.mean().item():.6f}")
print(f"  changed (>dw_min=0.002): {(delta > 0.002).sum().item()} / {delta.numel()}")
print(f"  changed (>1e-6): {(delta > 1e-6).sum().item()} / {delta.numel()}")

print("\n" + "=" * 80)
if delta.max().item() > 0.002:
    print("✓ SUCCESS: Tile CAN be updated manually!")
    print("\n→ The tile hardware is working")
    print("→ The problem must be in how AnalogSGD calls update()")
elif delta.max().item() > 1e-6:
    print("⚠️  PARTIAL: Tile updated but changes < dw_min")
    print(f"\n→ Updates happened but are smaller than expected")
else:
    print("✗ FAIL: Even manual update doesn't work")
    print("\n→ There may be a fundamental issue with the tile configuration")
print("=" * 80)
