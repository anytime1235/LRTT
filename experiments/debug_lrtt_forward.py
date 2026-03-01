#!/usr/bin/env python
# coding=utf-8
"""
Debug LRTT forward pass to understand the difference
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_linear_to_lrtt

print("=" * 80)
print("Debug LRTT Forward Pass")
print("=" * 80)
print()

# Simple setup for easier debugging
batch_size = 2
d_in = 4
d_out = 3
rank = 2

alpha_standard = 2.0
alpha_lrtt = alpha_standard / rank  # 1.0

torch.manual_seed(42)

# Standard LoRA
class StandardLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        out_w = torch.nn.functional.linear(x, self.weight)
        out_lora = torch.nn.functional.linear(x, self.lora_A)
        out_lora = torch.nn.functional.linear(out_lora, self.lora_B)

        # Print intermediates
        print("Standard LoRA forward:")
        print(f"  x: {x[0, :2].tolist()}")
        print(f"  W·x: {out_w[0, :].tolist()}")
        print(f"  A·x: {torch.nn.functional.linear(x, self.lora_A)[0, :].tolist()}")
        print(f"  B·(A·x): {out_lora[0, :].tolist()}")
        print(f"  scaling: {self.scaling}")
        print(f"  scaled B·(A·x): {(out_lora * self.scaling)[0, :].tolist()}")

        return out_w + out_lora * self.scaling

std_lora = StandardLoRA(d_in, d_out, rank, alpha_standard)

# LRTT-LoRA
base_linear = nn.Linear(d_in, d_out, bias=False)
with torch.no_grad():
    base_linear.weight.copy_(std_lora.weight)

lrtt_config = create_lrtt_lora_config(
    rank=rank,
    lora_alpha=alpha_lrtt,
    use_floating_point=True
)

lrtt_lora = convert_linear_to_lrtt(base_linear, lrtt_config)

# Copy weights
analog_module = lrtt_lora.analog_module
w_a, _ = analog_module.tile_a.get_weights()
w_b, _ = analog_module.tile_b.get_weights()
w_c, _ = analog_module.tile_c.get_weights()

print("Weight shapes:")
print(f"  Standard: W={std_lora.weight.shape}, A={std_lora.lora_A.shape}, B={std_lora.lora_B.shape}")
print(f"  LRTT: C={w_c.shape}, tile_b={w_b.shape}, tile_a={w_a.shape}")
print()

with torch.no_grad():
    w_b.copy_(std_lora.lora_A.data)
    w_a.copy_(std_lora.lora_B.data)
    analog_module.tile_a.set_weights(w_a)
    analog_module.tile_b.set_weights(w_b)

# Verify weights copied correctly
w_a_check, _ = analog_module.tile_a.get_weights()
w_b_check, _ = analog_module.tile_b.get_weights()
w_c_check, _ = analog_module.tile_c.get_weights()

print("Weight verification:")
print(f"  std W == LRTT C: {torch.allclose(std_lora.weight, w_c_check)}")
print(f"  std A == LRTT tile_b: {torch.allclose(std_lora.lora_A, w_b_check)}")
print(f"  std B == LRTT tile_a: {torch.allclose(std_lora.lora_B, w_a_check)}")
print()

# Forward pass
x = torch.randn(batch_size, d_in)

print("=" * 80)
y_std = std_lora(x)
print()

print("=" * 80)
print("LRTT-LoRA forward:")

# Manual LRTT forward to see what's happening
y_c = analog_module.tile_c.forward(x)
y_b = analog_module.tile_b.forward(x)  # tile_b·x = A·x
y_a = analog_module.tile_a.forward(y_b)  # tile_a·(tile_b·x) = B·(A·x)

print(f"  x: {x[0, :2].tolist()}")
print(f"  C·x: {y_c[0, :].tolist()}")
print(f"  tile_b·x (A·x): {y_b[0, :].tolist()}")
print(f"  tile_a·(tile_b·x) (B·(A·x)): {y_a[0, :].tolist()}")
print(f"  alpha: {lrtt_config.device.lora_alpha}")

# Now call actual forward
y_lrtt = lrtt_lora(x)
print(f"  LRTT output: {y_lrtt[0, :].tolist()}")
print()

print("=" * 80)
print("Comparison:")
print(f"  Standard LoRA: {y_std[0, :].tolist()}")
print(f"  LRTT-LoRA:     {y_lrtt[0, :].tolist()}")
print(f"  Difference:    {(y_std - y_lrtt)[0, :].tolist()}")
print(f"  Max diff:      {(y_std - y_lrtt).abs().max().item():.8f}")
print()

# Check if the LoRA contribution matches
std_lora_contrib = torch.nn.functional.linear(x, std_lora.lora_A)
std_lora_contrib = torch.nn.functional.linear(std_lora_contrib, std_lora.lora_B)
std_lora_contrib = std_lora_contrib * std_lora.scaling

lrtt_lora_contrib = lrtt_config.device.lora_alpha * y_a

print("LoRA contribution comparison:")
print(f"  Standard: {std_lora_contrib[0, :].tolist()}")
print(f"  LRTT:     {lrtt_lora_contrib[0, :].tolist()}")
print(f"  Match:    {torch.allclose(std_lora_contrib, lrtt_lora_contrib, atol=1e-5)}")
