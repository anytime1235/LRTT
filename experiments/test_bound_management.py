#!/usr/bin/env python3
"""
가설 2 검증: Bound management가 forward 시 weight를 수정하는가?

테스트: Forward만 반복 (optimizer step 없이)
- ITERATIVE bound management가 매 forward마다 weight 수정하는지 확인
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import SoftBoundsDevice
from aihwkit.simulator.configs.utils import NoiseManagementType, BoundManagementType

print("=" * 80)
print("TEST 2: Bound Management 단독 테스트")
print("=" * 80)

# Config with bound management enabled
print("\n[Setup]")
rpu_config = SingleRPUConfig(device=SoftBoundsDevice())
rpu_config.device.mult_noise = False
rpu_config.device.dw_min = 0.001
rpu_config.device.w_max = 1.0
rpu_config.device.w_min = -1.0

# CRITICAL: Enable bound management, disable noise management
rpu_config.forward.noise_management = NoiseManagementType.NONE  # ← Noise OFF
rpu_config.forward.bound_management = BoundManagementType.ITERATIVE  # ← Bound ON

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

# 큰 값으로 초기화 (bound 초과하도록)
print("\n[Initialize with large values (exceed bounds)]")
with torch.no_grad():
    weights = layer.get_weights()[0]
    weights[:] = torch.randn_like(weights) * 2.0  # -2~2 범위 (bound 초과)
    layer.set_weights(weights)

print(f"Initial weight stats:")
print(f"  Range: [{weights.min():.3f}, {weights.max():.3f}]")
print(f"  Mean: {weights.mean():.3f}")
print(f"  Std: {weights.std():.3f}")
print(f"  Values exceeding bound (>1.0): {(weights.abs() > 1.0).sum().item()}")

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
    print(f"  Max change: {delta.max():.6f}")
    print(f"  Mean change: {delta.mean():.6f}")
    print(f"  Num changed (>1e-8): {(delta > 1e-8).sum().item()} / {delta.numel()}")
    print(f"  Values exceeding bound: {(w_after.abs() > 1.0).sum().item()}")

# Conclusion
print(f"\n" + "=" * 80)
print("CONCLUSION:")
# Check if weights changed
final_weights = layer.get_weights()[0]
initial_weights_restored = torch.randn_like(final_weights) * 2.0
total_change = (final_weights - weights).abs()

if total_change.max() > 1e-6:
    print("✗ WEIGHTS CHANGED during forward passes!")
    print("  → Bound management DOES modify weights")
    print("  → This is a cause of base_layer weight changes")
else:
    print("✓ Weights did NOT change")
    print("  → Bound management is NOT the cause")
print("=" * 80)
