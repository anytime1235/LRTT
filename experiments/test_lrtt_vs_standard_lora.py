#!/usr/bin/env python
# coding=utf-8
"""
Test: LRTT-LoRA (sixt1c_lora mode) vs Standard LoRA update rule comparison

This script verifies that LRTT's update rule produces the same results as
standard LoRA by comparing:
1. Forward pass outputs
2. Gradient computation
3. Weight updates after one training step

We use the actual sixt1c_lora configuration from sweep_lrtt_lora_optuna.py
"""

import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_linear_to_lrtt
from aihwkit.optim import AnalogSGD

print("=" * 80)
print("LRTT-LoRA vs Standard LoRA: Update Rule Comparison Test")
print("=" * 80)
print()

# =============================================================================
# Test Configuration
# =============================================================================
print("[1/7] Configuration")
print("-" * 80)

batch_size = 4
d_in = 64
d_out = 32
rank = 8
lr = 0.001
lora_alpha = 1.0

print(f"Dimensions: input={d_in}, output={d_out}, rank={rank}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {lr}")
print(f"LoRA alpha: {lora_alpha}")
print()

torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# Standard LoRA Implementation
# =============================================================================
print("[2/7] Creating Standard LoRA layer")
print("-" * 80)

class StandardLoRA(nn.Module):
    """Standard LoRA: y = W*x + (B @ A) * x * (alpha/rank)"""

    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Pretrained weights (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA matrices
        # A: [rank, in_features] - random init
        # B: [out_features, rank] - zero init
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # W * x
        out_w = torch.nn.functional.linear(x, self.weight)

        # B @ A @ x
        # x: [batch, in_features]
        # A: [rank, in_features]
        # B: [out_features, rank]
        out_lora = torch.nn.functional.linear(x, self.lora_A)  # [batch, rank]
        out_lora = torch.nn.functional.linear(out_lora, self.lora_B)  # [batch, out_features]

        return out_w + out_lora * self.scaling

# Create standard LoRA layer
std_lora = StandardLoRA(d_in, d_out, rank, lora_alpha)

print(f"Standard LoRA created:")
print(f"  W: {std_lora.weight.shape} (frozen)")
print(f"  A: {std_lora.lora_A.shape} (trainable, random)")
print(f"  B: {std_lora.lora_B.shape} (trainable, zero)")
print(f"  Scaling: {std_lora.scaling:.4f}")
print()

# =============================================================================
# LRTT-LoRA Implementation (sixt1c_lora mode)
# =============================================================================
print("[3/7] Creating LRTT-LoRA layer (sixt1c_lora mode)")
print("-" * 80)

# Create base linear layer
base_linear = nn.Linear(d_in, d_out, bias=False)

# Copy weights from standard LoRA to ensure same starting point
with torch.no_grad():
    base_linear.weight.copy_(std_lora.weight)

# Create LRTT config (sixt1c_lora mode)
lrtt_config = create_lrtt_lora_config(
    rank=rank,
    lora_alpha=lora_alpha,
    use_floating_point=False  # sixt1c_lora mode (6T1C)
)

print(f"LRTT Config:")
print(f"  Mode: 6T1C (sixt1c_lora)")
print(f"  Rank: {rank}")
print(f"  Alpha: {lora_alpha}")
print(f"  Update mode: {lrtt_config.device.update_mode}")
print(f"  Forward inject: {lrtt_config.device.forward_inject}")
print()

# Convert to LRTT-LoRA
lrtt_lora = convert_linear_to_lrtt(base_linear, lrtt_config)

print("LRTT-LoRA layer created")
print()

# Copy A and B weights from standard LoRA to LRTT tiles
print("[4/7] Copying weights from Standard LoRA to LRTT-LoRA")
print("-" * 80)

analog_module = lrtt_lora.analog_module

# Get current weights
w_a, _ = analog_module.tile_a.get_weights()
w_b, _ = analog_module.tile_b.get_weights()
w_c, _ = analog_module.tile_c.get_weights()

print(f"Before copy:")
print(f"  tile_a: {w_a.shape}, norm={w_a.norm().item():.6f}")
print(f"  tile_b: {w_b.shape}, norm={w_b.norm().item():.6f}")
print(f"  tile_c: {w_c.shape}, norm={w_c.norm().item():.6f}")
print()

# Standard LoRA: A [rank, in], B [out, rank]
# LRTT: tile_a [out, rank], tile_b [rank, in]
# Mapping: tile_a ≈ B, tile_b ≈ A

with torch.no_grad():
    # tile_b ← A (both [rank, in_features])
    w_b.copy_(std_lora.lora_A.data)
    analog_module.tile_b.set_weights(w_b)

    # tile_a ← B (both [out_features, rank])
    w_a.copy_(std_lora.lora_B.data)
    analog_module.tile_a.set_weights(w_a)

# Verify copy
w_a_after, _ = analog_module.tile_a.get_weights()
w_b_after, _ = analog_module.tile_b.get_weights()

print(f"After copy:")
print(f"  tile_a: norm={w_a_after.norm().item():.6f} (should match std_lora.B)")
print(f"  tile_b: norm={w_b_after.norm().item():.6f} (should match std_lora.A)")
print(f"  std_lora.B norm: {std_lora.lora_B.norm().item():.6f}")
print(f"  std_lora.A norm: {std_lora.lora_A.norm().item():.6f}")
print()

# =============================================================================
# Forward Pass Comparison
# =============================================================================
print("[5/7] Forward pass comparison")
print("-" * 80)

# Create input
x = torch.randn(batch_size, d_in)

# Standard LoRA forward
y_std = std_lora(x)

# LRTT-LoRA forward
y_lrtt = lrtt_lora(x)

print(f"Input shape: {x.shape}")
print(f"Standard LoRA output: {y_std.shape}, norm={y_std.norm().item():.6f}")
print(f"LRTT-LoRA output: {y_lrtt.shape}, norm={y_lrtt.norm().item():.6f}")
print()

# Compare outputs
output_diff = (y_std - y_lrtt).abs().max().item()
print(f"Output difference (max abs): {output_diff:.8f}")

if output_diff < 1e-3:
    print("✅ Forward pass MATCH (within tolerance)")
else:
    print(f"⚠️  Forward pass MISMATCH (diff={output_diff:.6f})")
print()

# =============================================================================
# Backward Pass and Weight Update
# =============================================================================
print("[6/7] Backward pass and weight updates")
print("-" * 80)

# Create target and compute loss
target = torch.randn(batch_size, d_out)

# Standard LoRA
loss_std = ((y_std - target) ** 2).sum()
loss_std.backward()

print("Standard LoRA gradients:")
print(f"  ∂L/∂A: norm={std_lora.lora_A.grad.norm().item():.6f}")
print(f"  ∂L/∂B: norm={std_lora.lora_B.grad.norm().item():.6f}")
print()

# Save gradients
grad_A_std = std_lora.lora_A.grad.clone()
grad_B_std = std_lora.lora_B.grad.clone()

# LRTT-LoRA
loss_lrtt = ((y_lrtt - target) ** 2).sum()
loss_lrtt.backward()

print(f"Loss comparison:")
print(f"  Standard LoRA loss: {loss_std.item():.6f}")
print(f"  LRTT-LoRA loss: {loss_lrtt.item():.6f}")
print()

# Update Standard LoRA weights
optimizer_std = torch.optim.SGD([std_lora.lora_A, std_lora.lora_B], lr=lr)
optimizer_std.step()

# Update LRTT-LoRA weights
optimizer_lrtt = AnalogSGD(lrtt_lora.parameters(), lr=lr)
optimizer_lrtt.step()

print("✅ Weight updates completed")
print()

# =============================================================================
# Compare Weight Changes
# =============================================================================
print("[7/7] Weight change comparison")
print("=" * 80)
print()

# Get updated LRTT weights
w_a_new, _ = analog_module.tile_a.get_weights()
w_b_new, _ = analog_module.tile_b.get_weights()

# Compute weight changes
# Standard LoRA
delta_A_std = -lr * grad_A_std  # SGD update rule
delta_B_std = -lr * grad_B_std

# LRTT-LoRA
delta_a_lrtt = w_a_new - w_a_after
delta_b_lrtt = w_b_new - w_b_after

print("Standard LoRA weight changes:")
print(f"  ΔA: norm={delta_A_std.norm().item():.6f}, max={delta_A_std.abs().max().item():.8f}")
print(f"  ΔB: norm={delta_B_std.norm().item():.6f}, max={delta_B_std.abs().max().item():.8f}")
print()

print("LRTT-LoRA weight changes:")
print(f"  Δtile_a: norm={delta_a_lrtt.norm().item():.6f}, max={delta_a_lrtt.abs().max().item():.8f}")
print(f"  Δtile_b: norm={delta_b_lrtt.norm().item():.6f}, max={delta_b_lrtt.abs().max().item():.8f}")
print()

# Compare changes (tile_a ≈ B, tile_b ≈ A)
diff_a = (delta_B_std - delta_a_lrtt).abs().max().item()
diff_b = (delta_A_std - delta_b_lrtt).abs().max().item()

print("Weight change comparison:")
print(f"  |ΔB_std - Δtile_a_lrtt| max: {diff_a:.8f}")
print(f"  |ΔA_std - Δtile_b_lrtt| max: {diff_b:.8f}")
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

# Tolerances
forward_tol = 1e-3  # Forward pass (may have quantization error)
weight_tol = 5e-3   # Weight updates (6T1C quantization)

checks = []

# Check 1: Forward pass
forward_ok = output_diff < forward_tol
checks.append(("Forward pass match", forward_ok, output_diff, forward_tol))

# Check 2: A/tile_b updates
a_ok = diff_b < weight_tol
checks.append(("A ↔ tile_b update match", a_ok, diff_b, weight_tol))

# Check 3: B/tile_a updates
b_ok = diff_a < weight_tol
checks.append(("B ↔ tile_a update match", b_ok, diff_a, weight_tol))

for name, passed, value, tolerance in checks:
    status = "✅" if passed else "❌"
    print(f"{status} {name}")
    print(f"   Value: {value:.8f}, Tolerance: {tolerance:.8f}")
    print()

all_passed = all(c[1] for c in checks)

print("=" * 80)
if all_passed:
    print("🎉 LRTT-LoRA update rule MATCHES standard LoRA!")
    print("   (within expected quantization tolerances)")
else:
    print("⚠️  LRTT-LoRA update rule DIFFERS from standard LoRA")
    print("   This may indicate an implementation issue.")
print("=" * 80)
print()

# Additional diagnostics
print("Detailed diagnostics:")
print(f"  Standard LoRA A updated: {delta_A_std.abs().max().item() > 1e-8}")
print(f"  Standard LoRA B updated: {delta_B_std.abs().max().item() > 1e-8}")
print(f"  LRTT tile_a updated: {delta_a_lrtt.abs().max().item() > 1e-8}")
print(f"  LRTT tile_b updated: {delta_b_lrtt.abs().max().item() > 1e-8}")
print()

if not all_passed:
    print("Possible issues:")
    if not forward_ok:
        print("  - Forward pass mismatch suggests incorrect composition")
    if not a_ok:
        print("  - A/tile_b update mismatch suggests incorrect gradient routing")
    if not b_ok:
        print("  - B/tile_a update mismatch suggests incorrect gradient routing")
