# -*- coding: utf-8 -*-
"""Test LRTT transfer with direct method to analyze the transfer logic."""

import torch

from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Create config
lrtt_device = PythonLRTTPreset.sixt1c_ab(
    rank=4,
    transfer_every=5,
    lora_alpha=1.0,
    dt_batch_sec=1.0,
    include_retention=False,  # No retention for clean test
    reinit_mode="decay",
    decay_factor=1.0
)
lrtt_device.forward_inject = False

rpu_config = PythonLRTTRPUConfig(device=lrtt_device)

# Create tile
lrtt_tile = LRTTSimulatorTile(
    x_size=16,  # Small for analysis
    d_size=8,
    rpu_config=rpu_config
)
lrtt_tile = lrtt_tile.cuda() if DEVICE.type == 'cuda' else lrtt_tile

# Set known A/B weights for analysis
print("\n" + "="*70)
print("Setting known A/B weights for transfer analysis")
print("="*70)

A_init = torch.randn(8, 4).to(DEVICE) * 0.1
B_init = torch.randn(4, 16).to(DEVICE) * 0.1
C_init = torch.zeros(8, 16).to(DEVICE)

lrtt_tile.tile_a.set_weights(A_init)
lrtt_tile.tile_b.set_weights(B_init)
lrtt_tile.tile_c.set_weights(C_init)

# Read back
A = lrtt_tile.tile_a.get_weights()[0]
B = lrtt_tile.tile_b.get_weights()[0]
C_before = lrtt_tile.tile_c.get_weights()[0].clone()

print(f"\nA shape: {A.shape}")  # [d_size, rank] = [8, 4]
print(f"B shape: {B.shape}")  # [rank, x_size] = [4, 16]
print(f"C shape: {C_before.shape}")  # [d_size, x_size] = [8, 16]

# Expected: C += alpha * A @ B
expected_AB = A @ B  # [8, 4] @ [4, 16] = [8, 16]
print(f"\nExpected A @ B:")
print(f"  Shape: {expected_AB.shape}")
print(f"  Norm: {expected_AB.norm():.6f}")
print(f"  Mean: {expected_AB.mean():.6f}")

# Now trigger transfer directly
print("\n" + "="*70)
print("Triggering DIRECT transfer")
print("="*70)

# Force transfer counter to trigger
lrtt_tile.controller.transfer_counter = lrtt_tile.controller.transfer_every - 1

# Check transfer_lr
print(f"transfer_lr: {lrtt_tile.controller.transfer_lr}")

# Call ab_weight_transfer with direct method (use_onehot=False)
lrtt_tile.controller.ab_weight_transfer(use_onehot=False)

# Read C after transfer
C_after = lrtt_tile.tile_c.get_weights()[0]
C_change = C_after - C_before

print(f"\nC change after DIRECT transfer:")
print(f"  Shape: {C_change.shape}")
print(f"  Norm: {C_change.norm():.6f}")
print(f"  Mean: {C_change.mean():.6f}")

print(f"\nComparison:")
print(f"  Expected ||A @ B||: {expected_AB.norm():.6f}")
print(f"  Actual ||C_change||: {C_change.norm():.6f}")
print(f"  Ratio: {C_change.norm() / expected_AB.norm():.6f}")

# Check if C_change matches expected
# Expected with transfer_lr=1.0: C_change = transfer_lr * A @ B = A @ B
correlation = torch.nn.functional.cosine_similarity(
    C_change.flatten().unsqueeze(0),
    expected_AB.flatten().unsqueeze(0)
).item()
print(f"\n  Cosine similarity between C_change and A@B: {correlation:.6f}")

# Also check alternative formulations
alt1 = A.t() @ B.t()  # Wrong
alt2 = B @ A  # Wrong
print(f"\n  Cosine sim with A.t() @ B.t(): {torch.nn.functional.cosine_similarity(C_change.flatten().unsqueeze(0), alt1.flatten().unsqueeze(0)).item():.6f}")

# Check what the PWU update formula does
# PWU: W += -lr * D @ X^T
# In code: D_chunk_t_d = A.t(), X_chunk_d = B
# So: C += -lr * A.t() @ B.t()
pwu_result = -1.0 * A.t() @ B.t()  # This is what PWU would compute if D=A.t(), X=B
print(f"  Shape of A.t() @ B.t(): {(A.t() @ B.t()).shape}")  # [4,8] @ [16,4] - shape mismatch!

print("\n" + "="*70)
print("Analysis of PWU formula:")
print("="*70)
print("PWU computes: W += -lr * D @ X^T")
print("In code:")
print("  D_chunk_t_d = A.t() -> shape [rank, d_size] = [4, 8]")
print("  X_chunk_d = B -> shape [rank, x_size] = [4, 16]")
print("  So: C += -lr * D_chunk_t_d @ X_chunk_d^T")
print("     = -lr * [4, 8] @ [4, 16]^T")
print("     = -lr * [4, 8] @ [16, 4]")
print("     = -lr * [4, 4]  <- SHAPE MISMATCH! Should be [8, 16]")
print("\nThis explains why transfer is not working correctly!")
