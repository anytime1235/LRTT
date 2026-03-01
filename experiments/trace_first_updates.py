#!/usr/bin/env python
# coding=utf-8
"""
Trace what happens in first few training steps for both Standard LoRA and LRTT.

User's claim: Notation is reversed but equations are also reversed, so it should work fine.
Let's verify step by step.
"""

import torch

print("=" * 80)
print("FIRST TRAINING STEP ANALYSIS: Standard LoRA vs LRTT")
print("=" * 80)
print()

# Dimensions
batch, d_in, d_out, rank = 2, 4, 3, 2

print("Setup:")
print(f"  Input: [{batch}, {d_in}]")
print(f"  Output: [{batch}, {d_out}]")
print(f"  Rank: {rank}")
print()

# Standard LoRA
print("=" * 80)
print("STANDARD LoRA")
print("=" * 80)
print()

W_std = torch.randn(d_out, d_in)
A_std = torch.randn(rank, d_in)      # Random
B_std = torch.zeros(d_out, rank)     # Zero
x_std = torch.randn(batch, d_in)
labels = torch.randn(batch, d_out)

print("Initialization:")
print(f"  A [{rank}, {d_in}]: RANDOM")
print(f"  B [{d_out}, {rank}]: ZERO")
print(f"  A norm: {A_std.norm().item():.4f}")
print(f"  B norm: {B_std.norm().item():.4f}")
print()

# Forward
y_w = x_std @ W_std.t()
y_a = x_std @ A_std.t()              # [batch, rank]
y_b = y_a @ B_std.t()                # [batch, d_out] = 0 (B is zero!)
y_std = y_w + y_b
print(f"Forward: y = Wx + B(Ax)")
print(f"  Wx norm: {y_w.norm().item():.4f}")
print(f"  Ax norm: {y_a.norm().item():.4f}")
print(f"  B(Ax) norm: {y_b.norm().item():.4f} (zero, B=0)")
print(f"  y norm: {y_std.norm().item():.4f}")
print()

# Loss & gradient
loss_std = ((y_std - labels) ** 2).sum()
grad_y_std = 2 * (y_std - labels)    # [batch, d_out]

print(f"Loss: {loss_std.item():.4f}")
print()

# Gradients
print("Backward: Compute ∂L/∂A and ∂L/∂B")
print()

# ∂L/∂A = B^T · ∂L/∂y · x
grad_A_std = B_std.t() @ grad_y_std.t() @ x_std  # [rank, d_out] @ [d_out, batch] @ [batch, d_in]
print(f"∂L/∂A = B^T · ∂L/∂y · x")
print(f"  = {B_std.t().shape} @ {grad_y_std.t().shape} @ {x_std.shape}")
print(f"  = [{rank}, {d_in}]")
print(f"  norm: {grad_A_std.norm().item():.6f}")
print(f"  → B=0이므로 ∂L/∂A = 0 ❌")
print()

# ∂L/∂B = ∂L/∂y · (Ax)^T
grad_B_std = grad_y_std.t() @ y_a    # [d_out, batch] @ [batch, rank]
print(f"∂L/∂B = ∂L/∂y^T · (Ax)")
print(f"  = {grad_y_std.t().shape} @ {y_a.shape}")
print(f"  = [{d_out}, {rank}]")
print(f"  norm: {grad_B_std.norm().item():.6f}")
print(f"  → A is random이므로 Ax ≠ 0, ∂L/∂B ≠ 0 ✓")
print()

print("Step 1 Updates:")
print(f"  A: NO UPDATE (grad=0)")
print(f"  B: UPDATES (grad≠0)")
print()

# LRTT
print("=" * 80)
print("LRTT LoRA")
print("=" * 80)
print()

C_lrtt = torch.randn(d_out, d_in)
tile_a = torch.zeros(d_out, rank)    # Zero (like B)
tile_b = torch.randn(rank, d_in)     # Random (like A)
x_lrtt = torch.randn(batch, d_in)

print("Initialization:")
print(f"  tile_a [{d_out}, {rank}]: ZERO (like standard B)")
print(f"  tile_b [{rank}, {d_in}]: RANDOM (like standard A)")
print(f"  tile_a norm: {tile_a.norm().item():.4f}")
print(f"  tile_b norm: {tile_b.norm().item():.4f}")
print()

# Forward
y_c = x_lrtt @ C_lrtt.t()
y_tb = x_lrtt @ tile_b.t()           # [batch, rank]
y_ta = y_tb @ tile_a.t()             # [batch, d_out] = 0 (tile_a is zero!)
y_lrtt = y_c + y_ta
print(f"Forward: y = Cx + tile_a(tile_b·x)")
print(f"  Cx norm: {y_c.norm().item():.4f}")
print(f"  tile_b·x norm: {y_tb.norm().item():.4f}")
print(f"  tile_a(tile_b·x) norm: {y_ta.norm().item():.4f} (zero, tile_a=0)")
print(f"  y norm: {y_lrtt.norm().item():.4f}")
print()

# Loss & gradient
loss_lrtt = ((y_lrtt - labels) ** 2).sum()
grad_y_lrtt = 2 * (y_lrtt - labels)

print(f"Loss: {loss_lrtt.item():.4f}")
print()

# Gradients
print("Backward: Compute ∂L/∂tile_a and ∂L/∂tile_b")
print()

# ∂L/∂tile_a = ∂L/∂y · (tile_b·x)^T
grad_tile_a = grad_y_lrtt.t() @ y_tb  # [d_out, batch] @ [batch, rank]
print(f"∂L/∂tile_a = ∂L/∂y^T · (tile_b·x)")
print(f"  = {grad_y_lrtt.t().shape} @ {y_tb.shape}")
print(f"  = [{d_out}, {rank}]")
print(f"  norm: {grad_tile_a.norm().item():.6f}")
print(f"  → tile_b is random이므로 tile_b·x ≠ 0, ∂L/∂tile_a ≠ 0 ✓")
print()

# ∂L/∂tile_b = tile_a^T · ∂L/∂y · x
grad_tile_b = tile_a.t() @ grad_y_lrtt.t() @ x_lrtt  # [rank, d_out] @ [d_out, batch] @ [batch, d_in]
print(f"∂L/∂tile_b = tile_a^T · ∂L/∂y · x")
print(f"  = {tile_a.t().shape} @ {grad_y_lrtt.t().shape} @ {x_lrtt.shape}")
print(f"  = [{rank}, {d_in}]")
print(f"  norm: {grad_tile_b.norm().item():.6f}")
print(f"  → tile_a=0이므로 ∂L/∂tile_b = 0 ❌")
print()

print("Step 1 Updates:")
print(f"  tile_a: UPDATES (grad≠0)")
print(f"  tile_b: NO UPDATE (grad=0)")
print()

# Comparison
print("=" * 80)
print("COMPARISON")
print("=" * 80)
print()

print("Notation Mapping:")
print("  Standard A ↔ LRTT tile_b (both [rank, d_in])")
print("  Standard B ↔ LRTT tile_a (both [d_out, rank])")
print()

print("Initialization Mapping:")
print("  Standard A (random) → LRTT tile_b (random) ✓")
print("  Standard B (zero)   → LRTT tile_a (zero)   ✓")
print()

print("Step 1 Update Mapping:")
print("  Standard A: NO UPDATE")
print("  Standard B: UPDATES ✓")
print()
print("  LRTT tile_b (←Standard A): NO UPDATE")
print("  LRTT tile_a (←Standard B): UPDATES ✓")
print()

print("Pattern 일치:")
print("  Standard: Zero matrix (B) updates, Random matrix (A) doesn't")
print("  LRTT:     Zero matrix (tile_a) updates, Random matrix (tile_b) doesn't")
print()

print("=" * 80)
print("USER'S POINT IS CORRECT!")
print("=" * 80)
print()
print("Notation이 반대이고, 초기화도 반대로 매핑되어 있고,")
print("Update pattern도 반대로 되어 있습니다.")
print()
print("Standard LoRA에서 A가 처음엔 update 안되는 것처럼,")
print("LRTT에서 tile_b가 처음엔 update 안되는게 정상입니다!")
print()
print("문제는 다른 곳에 있을 수 있습니다...")
print("실제 학습 실패 원인을 다시 찾아야 합니다.")
