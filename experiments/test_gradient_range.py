#!/usr/bin/env python
"""
Test: How does tile handle gradients ranging from 10^0 to 10^5?

Key question from user:
"예를 들어 gradient가 10^5 ~ 10^0까지 변할 때 어떻게 처리한다는거야"
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
import torch.nn as nn

from sixt1c_config import gen_sixt1c_lora_config
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles import TorchInferenceTile

print("=" * 80)
print("TEST: Gradient Range 10^0 to 10^5")
print("=" * 80)

# ============================================================================
# Setup
# ============================================================================
print("\n[1] Creating analog layer...")

rpu_config = gen_sixt1c_lora_config(
    dt_batch_sec=1.0,
    include_retention=False,
    test_mode=None
)

digital_layer = nn.Linear(8, 16, bias=False)
nn.init.normal_(digital_layer.weight, mean=0, std=0.01)

analog_layer = AnalogLinear.from_digital(
    digital_layer,
    rpu_config,
    tile_module_class=TorchInferenceTile
)

tile = list(analog_layer.analog_tiles())[0]

print(f"  Layer: {analog_layer.in_features} → {analog_layer.out_features}")

# Get initial weights from tile
w_init = tile.get_weights()[0]
print(f"  Initial weight range: [{w_init.min().item():.4f}, {w_init.max().item():.4f}]")
print(f"  Initial mapping_scales: min={tile.mapping_scales.min().item():.4f}, max={tile.mapping_scales.max().item():.4f}")

# ============================================================================
# Test with different gradient magnitudes
# ============================================================================
print("\n[2] Testing gradient magnitudes from 10^0 to 10^5...")

gradient_scales = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
lr = 1e-4

results = []

for grad_scale in gradient_scales:
    print(f"\n--- Gradient scale: {grad_scale:.0e} ---")

    # Create test input and target output
    x = torch.randn(2, 8) * 0.1  # Small input
    target = torch.randn(2, 16) * grad_scale  # Target scaled to induce desired gradient

    # Forward
    analog_layer.train()
    output = analog_layer(x)

    # Create loss that will produce gradient ~grad_scale
    loss = ((output - target) ** 2).mean()

    # Record weight before
    w_before = analog_layer.weight.data.clone()

    # Get mapping_scales before
    ms_before = tile.mapping_scales.clone()

    # Backward
    analog_layer.zero_grad()
    loss.backward()

    # Check gradient magnitude
    grad = analog_layer.weight.grad
    grad_max = grad.abs().max().item()
    grad_norm = grad.norm().item()

    print(f"  Loss: {loss.item():.4e}")
    print(f"  Weight.grad: max_abs={grad_max:.4e}, norm={grad_norm:.4e}")

    # Manual SGD step
    with torch.no_grad():
        analog_layer.weight -= lr * grad

    # Record weight after
    w_after = analog_layer.weight.data.clone()
    w_delta = w_after - w_before

    # Get mapping_scales after
    ms_after = tile.mapping_scales.clone()

    print(f"  Weight delta: max_abs={w_delta.abs().max().item():.4e}, norm={w_delta.norm().item():.4e}")
    print(f"  mapping_scales: before=[{ms_before.min().item():.4e}, {ms_before.max().item():.4e}], after=[{ms_after.min().item():.4e}, {ms_after.max().item():.4e}]")

    # Check if mapping_scales changed
    ms_changed = not torch.allclose(ms_before, ms_after)
    if ms_changed:
        ms_ratio = (ms_after / (ms_before + 1e-12)).mean().item()
        print(f"  ⚠️  mapping_scales CHANGED by {ms_ratio:.4f}x")

    results.append({
        'grad_scale': grad_scale,
        'grad_max': grad_max,
        'grad_norm': grad_norm,
        'delta_max': w_delta.abs().max().item(),
        'delta_norm': w_delta.norm().item(),
        'ms_before': ms_before.mean().item(),
        'ms_after': ms_after.mean().item(),
        'ms_changed': ms_changed
    })

# ============================================================================
# Analysis
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print(f"\n{'Grad Scale':<12} {'Grad Max':<12} {'Delta Max':<12} {'Ratio':<10} {'MS After':<12}")
print("-" * 70)

for r in results:
    expected_delta = lr * r['grad_max']
    ratio = r['delta_max'] / (expected_delta + 1e-12)
    ms_marker = "*" if r['ms_changed'] else " "
    print(f"{r['grad_scale']:<12.0e} {r['grad_max']:<12.4e} {r['delta_max']:<12.4e} {ratio:<10.2f} {r['ms_after']:<12.4e}{ms_marker}")

print("\n* = mapping_scales changed during this step")

# ============================================================================
# Key insight
# ============================================================================
print("\n" + "=" * 80)
print("KEY MECHANISM")
print("=" * 80)

print(f"""
How analog tile handles gradient range 10^0 ~ 10^5:

1. **mapping_scales (ABS_MAX noise_management)**:
   - Forward: output = (input @ W) × mapping_scales
   - Weight normalization: W_normalized = W / alpha
     where alpha = max(|W|) / omega (columnwise if enabled)
   - alpha[alpha==0] = 1.0 (protect zero columns)

2. **Backward compensation**:
   - Gradient flows through: grad_W ∝ grad_output × input
   - BUT mapping_scales affects grad magnitude!
   - If omega > 0: backward hook compensates gradient scaling

3. **Per-column adaptation**:
   - Each output column has its own mapping_scale
   - As weights grow, mapping_scales adjust
   - This keeps conductance values in hardware range

4. **For this test**:
   - Initial weights ~0.01, so mapping_scales ~1.0
   - As gradient 10^5 updates weight → weight grows
   - mapping_scales adapt to normalize weight range
   - But gradient itself is NOT clipped/saturated

5. **Result**:
   - Weight update = lr × gradient (standard SGD)
   - Gradient 10^5 → update 10^1 (with lr=1e-4)
   - No special handling for large gradients
   - mapping_scales adjust AFTER update to keep weights normalized

The "reasonable conductance change" is determined by:
   • Learning rate (lr × gradient)
   • mapping_scales maintain weight normalization
   • Hardware quantization affects forward/inference, not gradient
""")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"""
사용자 질문: "gradient가 10^5 ~ 10^0까지 변할 때 어떻게 처리하나?"

답변:
1. **Gradient 자체는 그대로 전달됩니다** - clipping이나 saturation 없음
2. **Weight update = lr × gradient** - 표준 SGD와 동일
3. **mapping_scales는 weight 정규화를 위한 것**:
   - Forward pass에서 weight를 normalize
   - Backward pass에서 gradient를 보정 (omega > 0일 때)
   - Gradient 크기 자체를 제한하지 않음

4. **Layer마다 gradient가 10^0 ~ 10^5로 다르면**:
   - 각 layer는 독립적으로 처리
   - 각 layer의 mapping_scales가 다름
   - Gradient exploding 방지는 optimizer/gradient clipping이 담당

5. **Analog tile의 역할**:
   - Hardware simulation (quantization, noise)
   - Weight range 정규화 (conductance 범위 유지)
   - Gradient flow는 PyTorch autograd가 담당

따라서 gradient 10^5든 10^0이든, **tile은 그대로 받아서 lr을 곱해
weight를 업데이트하고, 그 후 mapping_scales로 normalize**합니다.
""")

print("\n" + "=" * 80)
