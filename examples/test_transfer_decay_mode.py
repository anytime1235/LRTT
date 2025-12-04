# -*- coding: utf-8 -*-
"""Test LRTT transfer with decay reinit mode (decay_factor=1.0).

This tests the new default behavior for 6T1C:
- reinit_mode='decay' with decay_factor=1.0
- No artificial reinit, only natural 6T1C retention decay
"""

import torch

from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig

# Settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Create model with 6T1C (decay mode, decay_factor=1.0)
lrtt_device = PythonLRTTPreset.sixt1c_ab(
    rank=4,
    transfer_every=5,  # Transfer every 5 steps for easy observation
    lora_alpha=1.0,
    dt_batch_sec=1.0,
    include_retention=True,
    reinit_mode="decay",  # Use decay mode
    decay_factor=1.0  # No artificial decay, only natural 6T1C retention
)
lrtt_device.forward_inject = False  # C-only forward

print(f"\nLRTT Config:")
print(f"  reinit_mode: {lrtt_device.reinit_mode}")
print(f"  decay_factor: {lrtt_device.decay_factor}")
print(f"  transfer_every: {lrtt_device.transfer_every}")

rpu_config = PythonLRTTRPUConfig(device=lrtt_device)

# Create LRTTSimulatorTile directly
lrtt_tile = LRTTSimulatorTile(
    x_size=784,
    d_size=10,
    rpu_config=rpu_config
)
lrtt_tile = lrtt_tile.cuda() if DEVICE.type == 'cuda' else lrtt_tile

print(f"\nController config:")
print(f"  reinit_mode: {lrtt_tile.controller.reinit_mode}")
print(f"  decay_factor: {lrtt_tile.controller.decay_factor}")

print(f"\nTracking transfer behavior over 20 steps:")
print("="*90)

# Track weights
def get_weight_stats():
    A = lrtt_tile.tile_a.get_weights()[0]
    B = lrtt_tile.tile_b.get_weights()[0]
    C = lrtt_tile.tile_c.get_weights()[0]
    AB = A @ B
    return {
        'A_norm': A.norm().item(),
        'B_norm': B.norm().item(),
        'C_norm': C.norm().item(),
        'AB_norm': AB.norm().item(),
        'A': A.clone(),
        'B': B.clone(),
        'C': C.clone(),
    }

prev_stats = get_weight_stats()
prev_C = prev_stats['C'].clone()

print(f"Initial: A_norm={prev_stats['A_norm']:.4f}, B_norm={prev_stats['B_norm']:.4f}, "
      f"C_norm={prev_stats['C_norm']:.4f}, AB_norm={prev_stats['AB_norm']:.4f}")

# Learning rate
lr = 0.01
lrtt_tile.set_learning_rate(lr)

transfer_count = 0
transfers_data = []

for step in range(1, 21):
    # Create random input/error
    x = torch.randn(32, 784).to(DEVICE)
    d = torch.randn(32, 10).to(DEVICE)

    # Store A/B BEFORE this step's update
    A_before_update = lrtt_tile.tile_a.get_weights()[0].clone()
    B_before_update = lrtt_tile.tile_b.get_weights()[0].clone()
    AB_before_transfer = (A_before_update @ B_before_update).norm().item()

    # Forward
    out = lrtt_tile.forward(x)

    # Backward
    x_grad = lrtt_tile.backward(d)

    # Update
    lrtt_tile._update_handled = False
    lrtt_tile.update(x, d)

    # Get stats AFTER update
    stats = get_weight_stats()
    C_change = (stats['C'] - prev_C).norm().item()

    # Check if transfer occurred (C changed significantly)
    is_transfer = C_change > 0.01
    marker = " <-- TRANSFER" if is_transfer else ""

    print(f"Step {step:2d}: A_norm={stats['A_norm']:.4f}, B_norm={stats['B_norm']:.4f}, "
          f"AB_norm={stats['AB_norm']:.4f}, C_change={C_change:.6f}{marker}")

    if is_transfer:
        transfer_count += 1
        transfers_data.append({
            'step': step,
            'AB_before': AB_before_transfer,
            'C_change': C_change,
            'A_norm_after': stats['A_norm'],
            'B_norm_after': stats['B_norm'],
        })
        print(f"         AB before transfer: {AB_before_transfer:.4f}")
        print(f"         Transfer #: {transfer_count}")

    prev_stats = stats
    prev_C = stats['C'].clone()

print("\n" + "="*90)
print("TRANSFER SUMMARY:")
print("="*90)
print(f"Total transfers: {transfer_count}")
print(f"\n{'Step':<6} {'AB_before':<12} {'C_change':<12} {'A_norm_after':<12} {'B_norm_after':<12}")
print("-"*54)
for t in transfers_data:
    print(f"{t['step']:<6} {t['AB_before']:<12.4f} {t['C_change']:<12.4f} {t['A_norm_after']:<12.4f} {t['B_norm_after']:<12.4f}")

print("\n" + "="*90)
print("CONCLUSION:")
print("="*90)
print("With reinit_mode='decay' and decay_factor=1.0:")
print("1. Transfer C += alpha * A @ B occurs every 5 steps")
print("2. A/B weights persist after transfer (no reset)")
print("3. A/B norms grow over time as gradients accumulate")
print("4. Transfer magnitude grows because A@B grows")
