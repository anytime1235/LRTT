# -*- coding: utf-8 -*-
"""Investigate LRTT transfer mechanism itself

The previous experiment showed that even Idealized device has 153% transfer error!
This suggests the error comes from the LRTT mechanism, not device non-idealities.

Possible sources:
1. Rank decomposition: C ≈ A@B (rank-64 approximation of full-rank matrix)
2. Transfer algorithm: one-hot read, pulse-based transfer
3. Numerical/quantization during transfer
4. Initial state: C might not start from A@B
"""

import torch
import torch.nn as nn

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.rpu_base import cuda

USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

print("=" * 80)
print("Investigating LRTT Transfer Mechanism")
print("=" * 80)
print()

# Create LRTT layer with Idealized device
device_config = PythonLRTTPreset.sixt1c_ab(
    rank=64,
    transfer_every=100,
    lora_alpha=1.0,
    dt_batch_sec=1.0,
    include_retention=True,
    c_device=IdealizedPresetDevice(),
)

device_config.use_onehot = True
device_config.use_sigma_delta = False

rpu_config = PythonLRTTRPUConfig(device=device_config)
rpu_config.forward.out_noise = 0.0  # Remove ALL noise
rpu_config.backward.out_noise = 0.0

analog_layer = AnalogLinear(784, 256, rpu_config=rpu_config, bias=True).to(DEVICE)

print("Configuration:")
print(f"  Rank: 64")
print(f"  C device: Idealized (perfect)")
print(f"  out_noise: 0.0 (no noise)")
print(f"  use_onehot: True")
print(f"  use_sigma_delta: False")
print()

def get_tile_weights(analog_layer):
    """Extract weights from A, B, C tiles."""
    try:
        tile = analog_layer.analog_module
        a_rows = tile.tile_a.tile.get_weights()
        b_rows = tile.tile_b.tile.get_weights()
        c_rows = tile.tile_c.tile.get_weights()

        a_weights = torch.stack([row.cpu() for row in a_rows])  # (256, 64)
        b_weights = torch.stack([row.cpu() for row in b_rows])  # (64, 784)
        c_weights = torch.stack([row.cpu() for row in c_rows])  # (256, 784)

        return a_weights, b_weights, c_weights
    except:
        return None, None, None

# Initial state (before any training)
print("=" * 80)
print("1. INITIAL STATE (before training)")
print("=" * 80)
a_w, b_w, c_w = get_tile_weights(analog_layer)

if a_w is not None:
    expected_c = torch.matmul(a_w, b_w)
    error = c_w - expected_c

    print(f"A norm: {torch.norm(a_w).item():.6f}")
    print(f"B norm: {torch.norm(b_w).item():.6f}")
    print(f"C norm: {torch.norm(c_w).item():.6f}")
    print(f"A@B norm: {torch.norm(expected_c).item():.6f}")
    print()
    print(f"Error ||C - A@B||: {torch.norm(error).item():.6f}")
    print(f"Relative error: {(torch.norm(error) / (torch.norm(expected_c) + 1e-10)).item():.6f}")
    print()
    print(f"Are C and A@B equal initially? {torch.allclose(c_w, expected_c, atol=1e-6)}")
    print()

# After initialization, check rank of C
c_rank = torch.linalg.matrix_rank(c_w).item()
ab_rank = torch.linalg.matrix_rank(expected_c).item()
print(f"Rank of C: {c_rank}")
print(f"Rank of A@B: {ab_rank}")
print()

if c_rank != ab_rank:
    print(f"⚠️  C has different rank than A@B!")
    print(f"    This suggests C is NOT initialized as A@B")
else:
    print(f"✓ C has same rank as A@B (both rank-{ab_rank})")
print()

# Train for one transfer cycle
print("=" * 80)
print("2. AFTER ONE TRANSFER (100 steps of training)")
print("=" * 80)

optimizer = AnalogSGD(analog_layer.parameters(), lr=0.1)
criterion = nn.MSELoss()

for step in range(100):
    data = torch.randn(64, 784).to(DEVICE)
    target = torch.randn(64, 256).to(DEVICE)

    optimizer.zero_grad()
    output = analog_layer(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

a_w2, b_w2, c_w2 = get_tile_weights(analog_layer)

if a_w2 is not None:
    expected_c2 = torch.matmul(a_w2, b_w2)
    error2 = c_w2 - expected_c2

    print(f"After training:")
    print(f"A norm: {torch.norm(a_w2).item():.6f}")
    print(f"B norm: {torch.norm(b_w2).item():.6f}")
    print(f"C norm: {torch.norm(c_w2).item():.6f}")
    print(f"A@B norm: {torch.norm(expected_c2).item():.6f}")
    print()
    print(f"Error ||C - A@B||: {torch.norm(error2).item():.6f}")
    print(f"Relative error: {(torch.norm(error2) / (torch.norm(expected_c2) + 1e-10)).item():.6f}")
    print()

    # Check if A and B were updated but C wasn't
    a_changed = not torch.allclose(a_w, a_w2, atol=1e-6)
    b_changed = not torch.allclose(b_w, b_w2, atol=1e-6)
    c_changed = not torch.allclose(c_w, c_w2, atol=1e-6)

    print(f"A changed: {a_changed}")
    print(f"B changed: {b_changed}")
    print(f"C changed: {c_changed}")
    print()

    if a_changed and b_changed and c_changed:
        print("✓ All tiles were updated (transfer occurred)")
    elif a_changed and b_changed and not c_changed:
        print("⚠️  A and B updated, but C did NOT change!")
        print("    This would cause large ||C - A@B|| error")
    elif not a_changed and not b_changed and not c_changed:
        print("⚠️  Nothing changed - no updates occurred")

print()
print("=" * 80)
print("3. UNDERSTANDING THE ERROR")
print("=" * 80)
print()
print("Key Questions:")
print("1. Is C initialized as A@B? Or random/zero?")
print("2. During transfer, does C become exactly A@B? Or approximate?")
print("3. Is the transfer mechanism quantized/pulsed causing discretization error?")
print()
print("From the experiments:")
print(f"  - Even Idealized device has ~153% error")
print(f"  - Device non-idealities only add ~8% more error")
print(f"  - This suggests the error is ALGORITHMIC, not from device physics")
print()
print("Hypothesis:")
print("  The transfer from A@B to C uses a pulse-based mechanism with")
print("  quantization, which introduces fundamental error regardless of")
print("  device ideality. This is an inherent limitation of analog hardware!")
print("=" * 80)
