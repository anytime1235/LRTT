#!/usr/bin/env python3
"""
가설 1 검증: Cached gradient가 원인인가?

테스트: Forward만 하고 backward 없이 optimizer.step() 호출
- 만약 cached gradient가 원인이라면 변화 없어야 함
- 변화가 있다면 다른 원인 (bound/noise management)
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from sixt1c_config import gen_softbounds_base_layer_config_trainable

print("=" * 80)
print("TEST 1: Gradient 캐싱 확인")
print("=" * 80)

# 모델 생성
print("\n[Setup]")
digital_layer = torch.nn.Linear(10, 10)
layer = AnalogLinear.from_digital(
    digital_layer,
    gen_softbounds_base_layer_config_trainable()
)
print("✓ AnalogLinear created")

# Freeze
for param in layer.parameters():
    param.requires_grad = False
print("✓ Layer frozen (requires_grad=False)")

# Optimizer
optimizer = AnalogSGD(layer.parameters(), lr=0.01)
optimizer.regroup_param_groups(layer)
print("✓ AnalogSGD optimizer created")

# Weight before
w_before = layer.get_weights()[0].clone()
print(f"\n[Initial weights]")
print(f"  Shape: {w_before.shape}")
print(f"  Range: [{w_before.min():.4f}, {w_before.max():.4f}]")
print(f"  Mean: {w_before.mean():.4f}")

# Forward only (NO backward)
print(f"\n[Forward pass (NO backward)]")
x = torch.randn(2, 10)
y = layer(x)
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {y.shape}")

# Check cached values
print(f"\n[Cached values check]")
print(f"  Has _cached_input? {hasattr(layer, '_cached_input')}")
if hasattr(layer, '_cached_input'):
    print(f"    Shape: {layer._cached_input.shape}")

print(f"  Has _cached_grad_output? {hasattr(layer, '_cached_grad_output')}")
if hasattr(layer, '_cached_grad_output'):
    cached_grad = layer._cached_grad_output
    print(f"    Is None? {cached_grad is None}")
    if cached_grad is not None:
        print(f"    Shape: {cached_grad.shape}")

# Check gradients
has_grad = any(p.grad is not None for p in layer.parameters())
print(f"\n[Gradient check]")
print(f"  Any parameter has gradient? {has_grad}")
if has_grad:
    for name, param in layer.named_parameters():
        if param.grad is not None:
            print(f"    {name}: grad norm = {param.grad.norm():.6f}")

# Optimizer step (without backward!)
print(f"\n[Optimizer step (NO backward beforehand)]")
optimizer.step()
print("✓ optimizer.step() called")

# Weight after
w_after = layer.get_weights()[0].clone()
print(f"\n[Weights after step]")
print(f"  Range: [{w_after.min():.4f}, {w_after.max():.4f}]")
print(f"  Mean: {w_after.mean():.4f}")

# Compare
delta = (w_after - w_before).abs()
print(f"\n[Weight change analysis]")
print(f"  Max change: {delta.max():.6f}")
print(f"  Mean change: {delta.mean():.6f}")
print(f"  Median change: {delta.median():.6f}")
print(f"  Num changed (>1e-8): {(delta > 1e-8).sum().item()} / {delta.numel()}")

# Conclusion
print(f"\n" + "=" * 80)
print("CONCLUSION:")
if delta.max() > 1e-8:
    print("✗ WEIGHTS CHANGED without gradient!")
    print("  → Gradient caching is NOT the cause")
    print("  → Bound/Noise management is likely the cause")
else:
    print("✓ Weights did NOT change")
    print("  → This doesn't happen in our case")
print("=" * 80)
