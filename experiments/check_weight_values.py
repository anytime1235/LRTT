#!/usr/bin/env python
# coding=utf-8
"""
Check if weights are truly identical between standard and LRTT
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_linear_to_lrtt

torch.manual_seed(42)

d_in, d_out, rank = 4, 3, 2

# Standard LoRA
class StandardLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

std_lora = StandardLoRA(d_in, d_out, rank)

# LRTT
base_linear = nn.Linear(d_in, d_out, bias=False)
with torch.no_grad():
    base_linear.weight.copy_(std_lora.weight)

lrtt_config = create_lrtt_lora_config(rank=rank, lora_alpha=1.0, use_floating_point=True)
lrtt_lora = convert_linear_to_lrtt(base_linear, lrtt_config)

analog_module = lrtt_lora.analog_module

print("=" * 80)
print("Weight Value Comparison")
print("=" * 80)
print()

# Check C tile (should match W)
w_c, _ = analog_module.tile_c.get_weights()

print("Base weights (W vs C):")
print(f"  Standard W shape: {std_lora.weight.shape}")
print(f"  LRTT C shape: {w_c.shape}")
print()

print("  Standard W[0, :]: {std_lora.weight[0, :].tolist()}")
print(f"  LRTT C[0, :]: {w_c[0, :].tolist()}")
print()

print(f"  Element-wise difference:")
diff = (std_lora.weight - w_c).abs()
print(f"    Max: {diff.max().item():.10f}")
print(f"    Mean: {diff.mean().item():.10f}")
print()

# Manual forward to double-check
x = torch.ones(1, d_in)

y_std = torch.nn.functional.linear(x, std_lora.weight)
y_lrtt_c = analog_module.tile_c.forward(x)

print("Forward pass with same input (all 1.0):")
print(f"  Standard W·x: {y_std[0, :].tolist()}")
print(f"  LRTT C·x: {y_lrtt_c[0, :].tolist()}")
print()

diff_out = (y_std - y_lrtt_c).abs().max().item()
print(f"  Output difference: {diff_out:.10f}")
print()

if diff_out > 1e-5:
    print("⚠️  C·x differs from W·x even with identical weights!")
    print("   This suggests the analog tile forward() is not computing x @ W.T correctly")
    print()

    # Check if it's a transpose issue
    y_lrtt_c_manual = x @ w_c.t()
    print(f"  Manual x @ C.T: {y_lrtt_c_manual[0, :].tolist()}")
    print(f"  Matches LRTT C·x: {torch.allclose(y_lrtt_c_manual, y_lrtt_c, atol=1e-6)}")
    print(f"  Matches Standard W·x: {torch.allclose(y_lrtt_c_manual, y_std, atol=1e-6)}")
else:
    print("✅ C·x matches W·x - analog tile forward() is correct")
