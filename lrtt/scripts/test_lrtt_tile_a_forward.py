# -*- coding: utf-8 -*-
"""Test LRTT tile_a forward directly"""

import torch
import torch.nn as nn

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.presets.devices import EcRamPresetDevice
from aihwkit.simulator.rpu_base import cuda

USE_CUDA = cuda.is_compiled()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

print("=" * 80)
print("Test LRTT tile_a forward directly")
print("=" * 80)
print()

# Create LRTT layer
device_config = PythonLRTTPreset.sixt1c_ab(
    rank=64,
    transfer_every=100,
    dt_batch_sec=0.0,
    include_retention=False,
    c_device=EcRamPresetDevice(),
)

rpu_config = PythonLRTTRPUConfig(device=device_config)
layer = AnalogLinear(784, 256, rpu_config=rpu_config, bias=True).to(DEVICE)
opt = AnalogSGD(layer.parameters(), lr=0.1)

# Train to build up A
print("Training for 50 steps...")
for _ in range(50):
    x = torch.randn(64, 784).to(DEVICE)
    y = torch.randn(64, 256).to(DEVICE)
    opt.zero_grad()
    layer(x).sub(y).pow(2).mean().backward()
    opt.step()

print("✓ Training complete")
print()

# Get tile_a
tile_a = layer.analog_module.tile_a

print("tile_a info:")
print(f"  Type: {type(tile_a)}")
print(f"  Shape: [256, 64]")
print()

# Get weights via get_weights
a_ref_rows = tile_a.tile.get_weights()
a_ref = torch.stack([row.cpu() for row in a_ref_rows])
print(f"tile_a weights (via get_weights):")
print(f"  ||A|| = {a_ref.norm().item():.6f}")
print()

# Test forward
I = torch.eye(64, dtype=torch.float32, device=DEVICE)
e_0 = I[0].unsqueeze(0)  # [1, 64]

print(f"Testing forward with one-hot e_0:")
print(f"  Input: shape={e_0.shape}, norm={e_0.norm().item():.3f}")
print()

try:
    with torch.no_grad():
        out = tile_a.forward(e_0)
    print(f"  Output: shape={out.shape}, norm={out.norm().item():.6f}")
    print()

    expected = a_ref[:, 0]
    print(f"  Expected (A[:, 0]): norm={expected.norm().item():.6f}")
    print()

    if out.norm().item() < 1e-6:
        print("✗ OUTPUT IS ZERO!")
        print()
        print("Possible causes:")
        print("  1. tile_a has no weights (but get_weights shows weights!)")
        print("  2. tile_a.forward() is not working")
        print("  3. tile_a is in a special mode that disables forward")
        print()

        # Try calling tile.forward()
        print("Trying tile_a.tile.forward()...")
        try:
            with torch.no_grad():
                out_inner = tile_a.tile.forward(e_0)
            print(f"  tile_a.tile.forward(): norm={out_inner.norm().item():.6f}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("✓ Forward works (non-zero output)")
        error = (out.squeeze().cpu() - expected).norm().item()
        print(f"  Error: {error:.6f}")

except Exception as e:
    print(f"✗ Error calling forward: {e}")

print()
print("=" * 80)

# Try differential reading
print("Testing differential reading:")
try:
    with torch.no_grad():
        out_pos = tile_a.forward(e_0)
        out_neg = tile_a.forward(-e_0)

    print(f"  forward(+e_0): norm={out_pos.norm().item():.6f}")
    print(f"  forward(-e_0): norm={out_neg.norm().item():.6f}")

    diff = 0.5 * (out_pos - out_neg)
    print(f"  differential:  norm={diff.norm().item():.6f}")
    print()

    if diff.norm().item() < 1e-6:
        print("✗ Differential is also ZERO!")
    else:
        print("✓ Differential is non-zero")

except Exception as e:
    print(f"✗ Error: {e}")

print()
print("=" * 80)

# Check tile properties
print("Checking tile_a properties:")
print(f"  hasattr forward: {hasattr(tile_a, 'forward')}")
print(f"  hasattr backward: {hasattr(tile_a, 'backward')}")
print(f"  hasattr update: {hasattr(tile_a, 'update')}")
print()

if hasattr(tile_a, 'tile'):
    inner_tile = tile_a.tile
    print(f"  Inner tile type: {type(inner_tile)}")
    print(f"  hasattr forward: {hasattr(inner_tile, 'forward')}")
    print()

print("=" * 80)
