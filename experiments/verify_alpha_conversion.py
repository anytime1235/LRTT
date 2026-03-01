#!/usr/bin/env python
# coding=utf-8
"""
Verify alpha conversion: LRTT with (alpha/rank) should match standard LoRA with alpha
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_linear_to_lrtt

print("=" * 80)
print("Alpha Conversion Verification")
print("=" * 80)
print()

# Setup
batch_size = 2
d_in = 8
d_out = 4
rank = 2

# Standard LoRA parameters
alpha_standard = 2.0  # This is what we'd use in standard LoRA
alpha_lrtt = alpha_standard / rank  # Convert for LRTT: 2.0 / 2 = 1.0

print(f"Configuration:")
print(f"  Rank: {rank}")
print(f"  Standard LoRA alpha: {alpha_standard}")
print(f"  LRTT alpha (converted): {alpha_lrtt}")
print(f"  Expected scaling: {alpha_standard / rank}")
print()

torch.manual_seed(42)

# =============================================================================
# Standard LoRA
# =============================================================================

class StandardLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank  # Standard LoRA scaling

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        out_w = torch.nn.functional.linear(x, self.weight)
        out_lora = torch.nn.functional.linear(x, self.lora_A)
        out_lora = torch.nn.functional.linear(out_lora, self.lora_B)
        return out_w + out_lora * self.scaling

std_lora = StandardLoRA(d_in, d_out, rank, alpha_standard)

print(f"Standard LoRA:")
print(f"  Scaling: {std_lora.scaling:.4f}")
print()

# =============================================================================
# LRTT-LoRA with converted alpha
# =============================================================================

base_linear = nn.Linear(d_in, d_out, bias=False)
with torch.no_grad():
    base_linear.weight.copy_(std_lora.weight)

lrtt_config = create_lrtt_lora_config(
    rank=rank,
    lora_alpha=alpha_lrtt,  # Use converted alpha
    use_floating_point=True  # FP for exact comparison
)

lrtt_lora = convert_linear_to_lrtt(base_linear, lrtt_config)

print(f"LRTT-LoRA:")
print(f"  Alpha: {lrtt_config.device.lora_alpha:.4f}")
print(f"  Scaling (direct): {lrtt_config.device.lora_alpha:.4f}")
print()

# Copy weights
analog_module = lrtt_lora.analog_module
w_a, _ = analog_module.tile_a.get_weights()
w_b, _ = analog_module.tile_b.get_weights()

with torch.no_grad():
    w_b.copy_(std_lora.lora_A.data)
    w_a.copy_(std_lora.lora_B.data)
    analog_module.tile_a.set_weights(w_a)
    analog_module.tile_b.set_weights(w_b)

# =============================================================================
# Forward pass comparison
# =============================================================================

x = torch.randn(batch_size, d_in)

y_std = std_lora(x)
y_lrtt = lrtt_lora(x)

print("Forward Pass Comparison:")
print(f"  Standard LoRA output norm: {y_std.norm().item():.6f}")
print(f"  LRTT-LoRA output norm: {y_lrtt.norm().item():.6f}")
print()

diff = (y_std - y_lrtt).abs().max().item()
print(f"  Max absolute difference: {diff:.8f}")
print()

# =============================================================================
# Result
# =============================================================================

print("=" * 80)
if diff < 1e-5:
    print("✅ SUCCESS: LRTT with (alpha/rank) MATCHES standard LoRA with alpha!")
    print(f"   Difference: {diff:.8f} < 1e-5")
else:
    print("❌ FAILURE: Outputs still differ")
    print(f"   Difference: {diff:.8f}")
print("=" * 80)
print()

print("Verification:")
print(f"  Standard LoRA scaling: {alpha_standard} / {rank} = {alpha_standard/rank:.4f}")
print(f"  LRTT alpha (direct):   {alpha_lrtt:.4f}")
print(f"  Match: {abs(alpha_standard/rank - alpha_lrtt) < 1e-6}")
