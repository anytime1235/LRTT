#!/usr/bin/env python
# coding=utf-8
"""
Diagnose LRTT-LoRA scaling issue

Check:
1. Is lora_alpha being applied correctly in forward pass?
2. Is quantization affecting the outputs?
3. What are the actual forward operations?
"""

import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_linear_to_lrtt

print("=" * 80)
print("LRTT-LoRA Scaling Diagnosis")
print("=" * 80)
print()

# Simple setup
batch_size = 2
d_in = 8
d_out = 4
rank = 2
lora_alpha = 1.0

torch.manual_seed(42)

# Create base linear
base_linear = nn.Linear(d_in, d_out, bias=False)

# Set simple weights for easier debugging
with torch.no_grad():
    base_linear.weight.fill_(0.1)  # Simple value

# Create LRTT config
lrtt_config = create_lrtt_lora_config(
    rank=rank,
    lora_alpha=lora_alpha,
    use_floating_point=False  # 6T1C mode
)

# Convert to LRTT
lrtt_lora = convert_linear_to_lrtt(base_linear, lrtt_config)

# Get tiles
analog_module = lrtt_lora.analog_module

# Set simple weights
w_a, _ = analog_module.tile_a.get_weights()
w_b, _ = analog_module.tile_b.get_weights()
w_c, _ = analog_module.tile_c.get_weights()

print("Setting simple weights for debugging:")
with torch.no_grad():
    w_a.fill_(0.2)  # tile_a (like B)
    w_b.fill_(0.3)  # tile_b (like A)
    w_c.fill_(0.1)  # tile_c (like W)

    analog_module.tile_a.set_weights(w_a)
    analog_module.tile_b.set_weights(w_b)
    analog_module.tile_c.set_weights(w_c)

print(f"  tile_a (B): all 0.2, shape {w_a.shape}")
print(f"  tile_b (A): all 0.3, shape {w_b.shape}")
print(f"  tile_c (W): all 0.1, shape {w_c.shape}")
print()

# Create simple input
x = torch.ones(batch_size, d_in)

print(f"Input: all 1.0, shape {x.shape}")
print()

# Forward pass
y = lrtt_lora(x)

print(f"Output: shape {y.shape}")
print(f"Output values:\n{y}")
print()

# Manual calculation
print("=" * 80)
print("Manual Calculation (expected values)")
print("=" * 80)
print()

# Standard LoRA formula: y = W·x + (alpha/rank) · B·(A·x)
scaling = lora_alpha / rank
print(f"Scaling factor: alpha/rank = {lora_alpha}/{rank} = {scaling}")
print()

# C·x
c_x = torch.ones(batch_size, d_out) * (0.1 * d_in)  # 0.1 * 8 = 0.8
print(f"C·x (W·x): all {0.1 * d_in}")

# B·x (tile_b·x)
b_x = torch.ones(batch_size, rank) * (0.3 * d_in)  # 0.3 * 8 = 2.4
print(f"tile_b·x: all {0.3 * d_in}")

# A·(B·x) (tile_a·(tile_b·x))
a_b_x = torch.ones(batch_size, d_out) * (0.2 * rank * 0.3 * d_in)  # 0.2 * 2 * 2.4 = 0.96
print(f"tile_a·(tile_b·x): all {0.2 * rank * 0.3 * d_in}")

# Final with scaling
y_expected = c_x + scaling * a_b_x
print(f"\nExpected y = C·x + (alpha/rank)·A·(B·x)")
print(f"           = {0.1 * d_in} + {scaling} * {0.2 * rank * 0.3 * d_in}")
print(f"           = {0.1 * d_in} + {scaling * 0.2 * rank * 0.3 * d_in}")
print(f"           = {0.1 * d_in + scaling * 0.2 * rank * 0.3 * d_in}")
print()

print("=" * 80)
print("Comparison")
print("=" * 80)
print()

print(f"Expected output: all {0.1 * d_in + scaling * 0.2 * rank * 0.3 * d_in:.6f}")
print(f"Actual output mean: {y.mean().item():.6f}")
print(f"Difference: {abs(y.mean().item() - (0.1 * d_in + scaling * 0.2 * rank * 0.3 * d_in)):.6f}")
print()

# Check if weights are quantized
w_a_after, _ = analog_module.tile_a.get_weights()
w_b_after, _ = analog_module.tile_b.get_weights()

print("Weight values after setting (check quantization):")
print(f"  tile_a: {w_a_after[0, 0].item():.8f} (set to 0.2)")
print(f"  tile_b: {w_b_after[0, 0].item():.8f} (set to 0.3)")
print()

if abs(w_a_after[0, 0].item() - 0.2) > 1e-6:
    print("⚠️  tile_a weights are QUANTIZED!")
    print(f"   Quantization error: {abs(w_a_after[0, 0].item() - 0.2):.8f}")

if abs(w_b_after[0, 0].item() - 0.3) > 1e-6:
    print("⚠️  tile_b weights are QUANTIZED!")
    print(f"   Quantization error: {abs(w_b_after[0, 0].item() - 0.3):.8f}")

print()
print("=" * 80)
print("Device Configuration Check")
print("=" * 80)
print()

device = lrtt_config.device
print(f"lora_alpha in device: {device.lora_alpha}")
print(f"rank in device: {device.rank}")
print(f"Scaling in device: {device.lora_alpha / device.rank}")
print(f"forward_inject: {device.forward_inject}")
print(f"update_mode: {device.update_mode}")
print()

# Check tile configs
print("Tile A device:")
print(f"  Type: {type(device.tile_a_device).__name__}")
if hasattr(device.tile_a_device, 'dw_min'):
    print(f"  dw_min: {device.tile_a_device.dw_min}")
    print(f"  w_min: {device.tile_a_device.w_min}")
    print(f"  w_max: {device.tile_a_device.w_max}")

print("\nTile B device:")
print(f"  Type: {type(device.tile_b_device).__name__}")
if hasattr(device.tile_b_device, 'dw_min'):
    print(f"  dw_min: {device.tile_b_device.dw_min}")
    print(f"  w_min: {device.tile_b_device.w_min}")
    print(f"  w_max: {device.tile_b_device.w_max}")

print()
print("=" * 80)
print("Conclusion")
print("=" * 80)
print()

expected_val = 0.1 * d_in + scaling * 0.2 * rank * 0.3 * d_in
actual_val = y.mean().item()
diff_percent = abs(actual_val - expected_val) / expected_val * 100

print(f"Expected: {expected_val:.6f}")
print(f"Actual:   {actual_val:.6f}")
print(f"Difference: {abs(actual_val - expected_val):.6f} ({diff_percent:.2f}%)")
print()

if diff_percent < 5:
    print("✅ LRTT forward pass is CORRECT (within 5% tolerance)")
    print("   6T1C quantization causes small difference")
elif diff_percent < 20:
    print("⚠️  LRTT forward pass has MODERATE mismatch")
    print("   May be due to quantization or scaling issue")
else:
    print("❌ LRTT forward pass has LARGE mismatch")
    print("   Likely incorrect scaling or composition")
