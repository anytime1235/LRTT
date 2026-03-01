#!/usr/bin/env python
# coding=utf-8
"""
Verify LRTT LoRA update rule is mathematically correct.

Question: With tile_a=0, tile_b=random (matching standard LoRA notation),
should tile_b receive gradients?

Mathematical derivation:
  Standard LoRA: y = Wx + B(Ax)
    ∂L/∂A = B^T · ∂L/∂y · x
    ∂L/∂B = ∂L/∂y · (Ax)^T

  LRTT LoRA: y = Cx + tile_a(tile_b·x)
    ∂L/∂tile_a = ∂L/∂y · (tile_b·x)^T
    ∂L/∂tile_b = tile_a^T · ∂L/∂y · x

  Mapping: tile_a ≈ B, tile_b ≈ A
    ∂L/∂tile_b = tile_a^T · ∂L/∂y · x

  If tile_a = 0:
    ∂L/∂tile_b = 0^T · ∂L/∂y · x = 0
    → tile_b does NOT receive gradients!
"""

import torch
import numpy as np

print("=" * 80)
print("LRTT LORA UPDATE RULE VERIFICATION")
print("=" * 80)
print()

# Setup
batch_size = 2
d_in = 4
d_out = 3
rank = 2

print("Dimensions:")
print(f"  Input:  [{batch_size}, {d_in}]")
print(f"  Output: [{batch_size}, {d_out}]")
print(f"  Rank:   {rank}")
print()

# Initialize
x = torch.randn(batch_size, d_in)
tile_a = torch.zeros(d_out, rank)  # A=0 (matching standard B=0)
tile_b = torch.randn(rank, d_in)   # B=random (matching standard A=random)
tile_c = torch.randn(d_out, d_in)  # Pretrained

print("Initialization (matching standard LoRA):")
print(f"  tile_a (≈ standard B): ZERO")
print(f"  tile_b (≈ standard A): RANDOM")
print()

# Forward
y_c = x @ tile_c.t()                    # [batch, d_out]
y_b = x @ tile_b.t()                    # [batch, rank]
y_a = y_b @ tile_a.t()                  # [batch, d_out]
y = y_c + y_a                           # [batch, d_out]

print("Forward pass:")
print(f"  y = Cx + tile_a(tile_b·x)")
print(f"  y_c norm: {y_c.norm().item():.4f}")
print(f"  y_a norm: {y_a.norm().item():.4f} (should be ~0 since tile_a=0)")
print(f"  y norm:   {y.norm().item():.4f}")
print()

# Simulate gradient
grad_y = torch.randn(batch_size, d_out)  # ∂L/∂y

print("Backward pass - Mathematical derivation:")
print("-" * 80)

# Gradient for tile_a (correct, should work)
print("\n1. Gradient for tile_a:")
print("   ∂L/∂tile_a = ∂L/∂y · (tile_b·x)^T")
print()

grad_tile_a_math = grad_y.t() @ y_b      # [d_out, batch] @ [batch, rank] = [d_out, rank]
print(f"   Mathematical: grad norm = {grad_tile_a_math.norm().item():.6f}")
print(f"   This depends on tile_b (random), so non-zero ✓")
print()

# Gradient for tile_b (problem!)
print("2. Gradient for tile_b:")
print("   ∂L/∂tile_b = tile_a^T · ∂L/∂y · x")
print()

# Method 1: Direct computation
grad_tile_b_direct = tile_a.t() @ grad_y.t() @ x  # [rank, d_out] @ [d_out, batch] @ [batch, d_in]
print(f"   Direct computation: grad norm = {grad_tile_b_direct.norm().item():.6f}")
print(f"   Since tile_a = 0:")
print(f"     = 0^T @ grad_y^T @ x = 0  ❌")
print()

# What LRTT controller actually computes
print("3. LRTT Controller implementation:")
print("   Code: DA = tile_a.backward(grad_y)")
print("         tile_b.update(x, DA)")
print()

# Simulate tile_a.backward()
# This computes: tile_a^T @ grad_y^T → [rank, d_out] @ [d_out, batch] = [rank, batch]
DA = tile_a.t() @ grad_y.t()  # [rank, batch]
DA = DA.t()                    # [batch, rank] for update interface

print(f"   DA = tile_a^T @ grad_y^T:")
print(f"     DA norm = {DA.norm().item():.6f}")
print(f"     Since tile_a = 0, DA = 0  ❌")
print()

# The tile.update(x, DA) computes: ΔW = -lr * DA^T @ x
delta_b_lrtt = DA.t() @ x  # [rank, batch] @ [batch, d_in] = [rank, d_in]
print(f"   tile_b update: ΔB = DA^T @ x")
print(f"     ΔB norm = {delta_b_lrtt.norm().item():.6f}")
print(f"     Since DA = 0, ΔB = 0  ❌")
print()

print("=" * 80)
print("VERIFICATION RESULT:")
print("=" * 80)
print()

print("✓ tile_a gradient: CORRECT (depends on tile_b·x, non-zero)")
print("❌ tile_b gradient: BLOCKED (depends on tile_a, which is zero)")
print()

print("Mathematical proof:")
print("  LRTT LoRA: y = Cx + tile_a(tile_b·x)")
print("  ∂L/∂tile_b = ∂(tile_a·tile_b·x)/∂tile_b · ∂L/∂y")
print("             = tile_a^T · ∂L/∂y  (by chain rule)")
print("             = 0^T · ∂L/∂y  (when tile_a = 0)")
print("             = 0")
print()

print("Comparison with standard LoRA:")
print("-" * 80)
print()
print("Standard LoRA: y = Wx + B(Ax)")
print("  A: random, B: zero")
print("  ∂L/∂A = B^T · ∂L/∂y · x = 0^T · ∂L/∂y · x = 0  (A blocked!)")
print("  ∂L/∂B = ∂L/∂y · (Ax)^T  (B gets gradient since A is random)")
print()

print("LRTT LoRA: y = Cx + tile_a(tile_b·x)")
print("  tile_b: random, tile_a: zero")
print("  ∂L/∂tile_b = tile_a^T · ∂L/∂y · x = 0^T · ∂L/∂y · x = 0  (tile_b blocked!)")
print("  ∂L/∂tile_a = ∂L/∂y · (tile_b·x)^T  (tile_a gets gradient since tile_b is random)")
print()

print("=" * 80)
print("CONCLUSION:")
print("=" * 80)
print()
print("Notation이 반대이므로 초기화는 맞습니다:")
print("  tile_a = 0  (standard B = 0)")
print("  tile_b = random  (standard A = random)")
print()
print("하지만 gradient flow가 OPPOSITE입니다:")
print()
print("  Standard: B=0이지만 ∂L/∂B는 A(random)에 의존 → B updates ✓")
print("  LRTT:     tile_a=0이고 ∂L/∂tile_b는 tile_a에 의존 → tile_b blocked ❌")
print()
print("문제의 원인:")
print("  Order of multiplication이 reversed:")
print("    Standard: B·(A·x)  → ∂L/∂B depends on A")
print("    LRTT:     A·(B·x)  → ∂L/∂B depends on A")
print()
print("  Notation mapping때문에:")
print("    Standard A (random) → LRTT tile_b (random)")
print("    Standard B (zero)   → LRTT tile_a (zero)")
print()
print("  하지만 gradient는:")
print("    Standard: ∂L/∂B depends on A (random) ✓")
print("    LRTT:     ∂L/∂tile_b depends on tile_a (zero) ❌")
print()
print("해결책:")
print("  Option 1: tile_a를 random으로 초기화 (A·B ≠ 0)")
print("  Option 2: Gradient routing 수정 (tile_b → direct grad)")
print("  Option 3: Order 바꾸기 (y = C·x + B·(A·x) 로 변경)")
