#!/usr/bin/env python3
"""
결정적 테스트: AnalogSGD.step()이 gradient 없이도 weight를 변경하는가?

이전 테스트 결과:
- Forward만: weight 변화 없음 ✓
- Optimizer.step() 없이: weight 변화 없음 ✓

이제 테스트:
- Forward + Optimizer.step() (gradient 없이)
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from sixt1c_config import gen_softbounds_base_layer_config_trainable

print("=" * 80)
print("FINAL TEST: AnalogSGD.step() with Zero Gradient")
print("=" * 80)

# 모델 생성
print("\n[Setup]")
layer = AnalogLinear.from_digital(
    torch.nn.Linear(10, 10),
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

print(f"\nConfig:")
print(f"  Config type: {type(layer.analog_module.tile).__name__}")

# Initial weights
w_initial = layer.get_weights()[0].clone()
print(f"\n[Initial weights]")
print(f"  Range: [{w_initial.min():.4f}, {w_initial.max():.4f}]")
print(f"  Mean: {w_initial.mean():.4f}")

print("\n" + "=" * 80)
print("TEST: Multiple steps with Forward + Optimizer.step() (NO backward)")
print("=" * 80)

for step in range(5):
    print(f"\n--- Step {step + 1} ---")

    w_before = layer.get_weights()[0].clone()

    # Forward (NO backward!)
    x = torch.randn(2, 10)
    y = layer(x)

    # Check gradient
    has_grad = any(p.grad is not None for p in layer.parameters())
    print(f"[Before step] Has gradient? {has_grad}")

    # Optimizer step (without backward!)
    optimizer.zero_grad()  # Ensure gradient is None
    optimizer.step()

    w_after = layer.get_weights()[0].clone()
    delta = (w_after - w_before).abs()

    print(f"[After step]")
    print(f"  Max change: {delta.max():.6f}")
    print(f"  Mean change: {delta.mean():.6f}")
    print(f"  Num changed (>1e-8): {(delta > 1e-8).sum().item()} / {delta.numel()}")

    if delta.max() > 1e-8:
        print(f"  ✗ WEIGHT CHANGED!")
        # Show which weights changed
        changed_mask = delta > 1e-8
        changed_indices = torch.nonzero(changed_mask)
        print(f"  Changed positions (first 5): {changed_indices[:5].tolist()}")
    else:
        print(f"  ✓ No change")

# Total change
w_final = layer.get_weights()[0].clone()
total_delta = (w_final - w_initial).abs()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total change over 5 steps:")
print(f"  Max: {total_delta.max():.6f}")
print(f"  Mean: {total_delta.mean():.6f}")
print(f"  Num changed: {(total_delta > 1e-8).sum().item()} / {total_delta.numel()}")

print("\n" + "=" * 80)
print("CONCLUSION:")
if total_delta.max() > 1e-8:
    print("✗ WEIGHTS CHANGED by AnalogSGD.step() alone!")
    print("  → optimizer.step() modifies weights even without gradient")
    print("  → This is the ROOT CAUSE of base_layer weight changes")
else:
    print("✓ Weights did NOT change")
    print("  → Something else is the cause (need more investigation)")
print("=" * 80)
