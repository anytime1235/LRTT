#!/usr/bin/env python3
"""
가설 3 검증: Noise management (ABS_MAX)가 rescaling하는가?

테스트: Noise management만 켜고 forward 반복
- ABS_MAX가 weight를 rescaling하는지 확인
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import SoftBoundsDevice
from aihwkit.simulator.configs.utils import NoiseManagementType, BoundManagementType

print("=" * 80)
print("TEST 3: Noise Management (ABS_MAX) 테스트")
print("=" * 80)

# Config with noise management enabled
print("\n[Setup]")
rpu_config = SingleRPUConfig(device=SoftBoundsDevice())
rpu_config.device.mult_noise = False
rpu_config.device.dw_min = 0.001
rpu_config.device.w_max = 1.0
rpu_config.device.w_min = -1.0

# CRITICAL: Enable noise management, disable bound management
rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX  # ← Noise ON
rpu_config.forward.bound_management = BoundManagementType.NONE  # ← Bound OFF

rpu_config.mapping.digital_bias = True
rpu_config.mapping.weight_scaling_omega = 1.0
rpu_config.mapping.learn_out_scaling = True

print("Config:")
print(f"  mult_noise: {rpu_config.device.mult_noise}")
print(f"  noise_management: {rpu_config.forward.noise_management}")
print(f"  bound_management: {rpu_config.forward.bound_management}")
print(f"  w_max/w_min: {rpu_config.device.w_max} / {rpu_config.device.w_min}")

# Create layer
layer = AnalogLinear.from_digital(
    torch.nn.Linear(10, 10),
    rpu_config
)
print("✓ AnalogLinear created")

# Freeze
for param in layer.parameters():
    param.requires_grad = False
print("✓ Layer frozen")

# 큰 값으로 초기화
print("\n[Initialize with large values]")
with torch.no_grad():
    weights = layer.get_weights()[0]
    weights[:] = torch.randn_like(weights) * 3.0  # 큰 값
    layer.set_weights(weights)

print(f"Initial weight stats:")
print(f"  Range: [{weights.min():.3f}, {weights.max():.3f}]")
print(f"  Max abs: {weights.abs().max():.3f}")
print(f"  Mean abs: {weights.abs().mean():.3f}")
print(f"  Mean: {weights.mean():.3f}")
print(f"  Std: {weights.std():.3f}")

# Multiple forward passes (NO optimizer, NO backward!)
print("\n" + "=" * 80)
print("FORWARD PASSES (no optimizer, no backward)")
print("=" * 80)

for i in range(5):
    w_before = layer.get_weights()[0].clone()

    # Just forward
    x = torch.randn(2, 10)
    y = layer(x)

    w_after = layer.get_weights()[0].clone()
    delta = (w_after - w_before).abs()

    print(f"\nForward {i+1}:")
    print(f"  Weight range: [{w_after.min():.4f}, {w_after.max():.4f}]")
    print(f"  Max abs: {w_after.abs().max():.3f}")
    print(f"  Max change: {delta.max():.6f}")
    print(f"  Mean change: {delta.mean():.6f}")
    print(f"  Median change: {delta.median():.6f}")
    print(f"  Num changed (>1e-8): {(delta > 1e-8).sum().item()} / {delta.numel()}")

    # Check if rescaling happened
    if delta.max() > 1e-6:
        # Calculate scale factor
        max_before = w_before.abs().max()
        max_after = w_after.abs().max()
        if max_before > 0:
            scale = max_after / max_before
            print(f"  Scale factor: {scale:.6f}")

# Conclusion
print(f"\n" + "=" * 80)
print("CONCLUSION:")
final_weights = layer.get_weights()[0]
total_change = (final_weights - weights).abs()

if total_change.max() > 1e-6:
    print("✗ WEIGHTS CHANGED during forward passes!")
    print("  → Noise management (ABS_MAX) DOES modify weights")
    print("  → This is a cause of base_layer weight changes")

    # Check if it's rescaling
    initial_max = weights.abs().max()
    final_max = final_weights.abs().max()
    print(f"\n  Initial max(abs): {initial_max:.3f}")
    print(f"  Final max(abs): {final_max:.3f}")
    if abs(final_max - 1.0) < 0.1:
        print(f"  → Weights were rescaled to have max ≈ 1.0")
else:
    print("✓ Weights did NOT change")
    print("  → Noise management is NOT the cause")
print("=" * 80)
