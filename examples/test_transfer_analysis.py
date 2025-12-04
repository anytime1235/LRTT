# -*- coding: utf-8 -*-
"""Detailed analysis of transfer mechanism."""

import torch

from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.tiles.analog import AnalogTile
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.presets.devices import IdealizedPresetDevice

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

print("\n" + "="*70)
print("TEST 1: Verify AnalogTile.update() behavior with Idealized device")
print("="*70)

# Create a simple Idealized analog tile
ideal_config = SingleRPUConfig(device=IdealizedPresetDevice())
tile = AnalogTile(out_size=8, in_size=16, rpu_config=ideal_config)
tile = tile.cuda() if DEVICE.type == 'cuda' else tile

# Set to zero
tile.set_weights(torch.zeros(8, 16).to(DEVICE))
tile.set_learning_rate(1.0)

# Get initial weights
W_before = tile.get_weights()[0].clone()
print(f"W_before norm: {W_before.norm():.6f}")

# Create test data
# PWU formula: W += -lr * d @ x^T
# x: [batch, in_size], d: [batch, out_size]
x = torch.ones(1, 16).to(DEVICE)  # [1, 16]
d = torch.ones(1, 8).to(DEVICE)   # [1, 8]

# Expected: W += -1.0 * d.T @ x = -1.0 * [8,1] @ [1,16] = -[8,16] of all 1s
expected_change = -1.0 * d.t() @ x
print(f"Expected change: all {expected_change[0,0]:.1f}, norm={expected_change.norm():.6f}")

# Apply update
tile.update(x, d)

# Get after weights
W_after = tile.get_weights()[0]
actual_change = W_after - W_before
print(f"Actual change: first elem={actual_change[0,0]:.4f}, norm={actual_change.norm():.6f}")
print(f"Match ratio: {actual_change.norm() / expected_change.norm():.6f}")

print("\n" + "="*70)
print("TEST 2: What tile_c.update(X_chunk, D_chunk_t) computes")
print("="*70)

# Reset
tile.set_weights(torch.zeros(8, 16).to(DEVICE))
W_before = tile.get_weights()[0].clone()

# In direct transfer:
# A: [d_size, rank] = [8, 4]
# B: [rank, x_size] = [4, 16]
# D_chunk = A[:, off:end] -> [8, cur]
# X_chunk = B[off:end, :] -> [cur, 16]
# D_chunk_t = D_chunk.t() -> [cur, 8]
# tile_c.update(X_chunk, D_chunk_t)
# PWU: W += -lr * D_chunk_t @ X_chunk^T
#    = -lr * [cur, 8] @ [cur, 16]^T
#    = -lr * [cur, 8] @ [16, cur]
# This is WRONG! Shape mismatch: [cur, 8] @ [16, cur] doesn't work for matrix mult

print("In _ab_weight_transfer_direct:")
print("  D_chunk = A[:, off:end]  -> shape [d_size, cur] = [8, cur]")
print("  X_chunk = B[off:end, :]  -> shape [cur, x_size] = [cur, 16]")
print("  D_chunk_t = D_chunk.t()  -> shape [cur, d_size] = [cur, 8]")
print("  Call: tile_c.update(X_chunk, D_chunk_t)")
print()
print("AnalogTile.update(x_input, d_input) expects:")
print("  x_input: [batch, in_size]")
print("  d_input: [batch, out_size]")
print("  PWU computes: W += -lr * d_input.T @ x_input")
print()
print("So with X_chunk=[cur, x_size], D_chunk_t=[cur, d_size]:")
print("  x_input = X_chunk -> [cur, x_size=16] -> batch=cur, in_size=x_size=16")
print("  d_input = D_chunk_t -> [cur, d_size=8] -> batch=cur, out_size=d_size=8")
print("  PWU: W[8,16] += -lr * D_chunk_t.T @ X_chunk")
print("      = -lr * [d_size, cur] @ [cur, x_size]")
print("      = -lr * [8, cur] @ [cur, 16]")
print("      = -lr * A[:, off:end] @ B[off:end, :]")
print("      = -lr * (A @ B)[for chunk]")
print()
print("So the formula IS correct! The issue is elsewhere.")

print("\n" + "="*70)
print("TEST 3: Simulate the exact transfer logic")
print("="*70)

# Create A, B
A = torch.randn(8, 4).to(DEVICE) * 0.1
B = torch.randn(4, 16).to(DEVICE) * 0.1

expected_AB = A @ B
print(f"Expected A @ B norm: {expected_AB.norm():.6f}")

# Reset tile
tile.set_weights(torch.zeros(8, 16).to(DEVICE))
tile.set_learning_rate(1.0)
W_before = tile.get_weights()[0].clone()

# Simulate _ab_weight_transfer_direct for rank=4, chunk_size=4 (single chunk)
rank = 4
chunk_size = 4
off = 0
end = min(off + chunk_size, rank)
cur = end - off

D_chunk = A[:, off:end].contiguous()  # [8, 4]
X_chunk = B[off:end, :].contiguous()  # [4, 16]

print(f"D_chunk shape: {D_chunk.shape}")
print(f"X_chunk shape: {X_chunk.shape}")

# Sign rule: transfer_lr > 0, so negate D
transfer_lr = 1.0
if transfer_lr > 0:
    D_chunk = -D_chunk

D_chunk_t = D_chunk.t().contiguous()  # [4, 8]
print(f"D_chunk_t shape: {D_chunk_t.shape}")

# Call update
tile.update(X_chunk, D_chunk_t)

W_after = tile.get_weights()[0]
actual_change = W_after - W_before
print(f"\nActual C change norm: {actual_change.norm():.6f}")
print(f"Expected A @ B norm: {expected_AB.norm():.6f}")
print(f"Ratio: {actual_change.norm() / expected_AB.norm():.6f}")

# Check correlation
corr = torch.nn.functional.cosine_similarity(
    actual_change.flatten().unsqueeze(0),
    expected_AB.flatten().unsqueeze(0)
).item()
print(f"Cosine similarity: {corr:.6f}")

print("\n" + "="*70)
print("TEST 4: Check if it's the device causing the issue")
print("="*70)

# The issue might be that IdealizedPresetDevice still has some noise/quantization
# Let's compute what update SHOULD do:
# W += -lr * D_chunk_t.T @ X_chunk
# D_chunk_t.T = D_chunk (since we transposed it)
# But wait, D_chunk was negated, so D_chunk_t = (-A).t() = -A.t()
# D_chunk_t.T = -A
# So: W += -lr * (-A) @ X_chunk = +lr * A @ B

theoretical = transfer_lr * A @ B
print(f"Theoretical change (with sign correction): +lr * A @ B = {theoretical.norm():.6f}")
print(f"Actual change: {actual_change.norm():.6f}")
print(f"Ratio: {actual_change.norm() / theoretical.norm():.6f}")

corr2 = torch.nn.functional.cosine_similarity(
    actual_change.flatten().unsqueeze(0),
    theoretical.flatten().unsqueeze(0)
).item()
print(f"Cosine similarity with +A@B: {corr2:.6f}")
