#!/usr/bin/env python
"""
Critical Test: Does mapping_scales depend on previous gradient or current weight?

질문: "이전 스텝의 gradient를 베이스로 조절하는건가?"
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
import torch.nn as nn

print("=" * 80)
print("TEST: mapping_scales 기준 - Previous Gradient vs Current Weight")
print("=" * 80)

# ============================================================================
# Setup: Same layer with weight normalization
# ============================================================================
class WeightNormalizedLayer(nn.Module):
    """Simulates analog tile weight normalization"""

    def __init__(self, in_features, out_features, omega=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.omega = omega
        self.mapping_scales = None

    def forward(self, x):
        # Compute mapping_scales from CURRENT weight (not gradient!)
        weight_max = self.weight.abs().max(dim=1, keepdim=True)[0]
        alpha = weight_max / self.omega
        alpha = torch.where(alpha == 0, torch.ones_like(alpha), alpha)

        self.mapping_scales = alpha.detach()

        # Normalize and forward
        w_norm = self.weight / alpha
        output = torch.matmul(x, w_norm.t()) * alpha.t()

        return output

layer = WeightNormalizedLayer(8, 16, omega=1.0)
lr = 1e-4

print(f"Layer initialized: {layer.weight.shape}")
print(f"Initial weight max: {layer.weight.abs().max().item():.4e}\n")

# ============================================================================
# Scenario 1: Large gradient → small gradient
# ============================================================================
print("=" * 80)
print("SCENARIO 1: 큰 gradient 후 작은 gradient")
print("=" * 80)

# Step 1: Very large gradient
x1 = torch.randn(2, 8) * 0.1
target1 = torch.randn(2, 16) * 1e5  # Large target → large gradient

output1 = layer(x1)
loss1 = ((output1 - target1) ** 2).mean()

layer.zero_grad()
loss1.backward()

grad1 = layer.weight.grad.clone()
grad1_max = grad1.abs().max().item()
ms1 = layer.mapping_scales.mean().item()

print(f"\nStep 1 (large gradient):")
print(f"  Gradient max: {grad1_max:.4e}")
print(f"  mapping_scales (before update): {ms1:.4e}")

# Update weight
with torch.no_grad():
    layer.weight -= lr * grad1
    w1_max = layer.weight.abs().max().item()

print(f"  Weight max (after update): {w1_max:.4e}")

# Step 2: Forward to recompute mapping_scales
with torch.no_grad():
    _ = layer(x1)
    ms1_after = layer.mapping_scales.mean().item()

print(f"  mapping_scales (after update): {ms1_after:.4e}")

# Step 2: Very SMALL gradient now
x2 = torch.randn(2, 8) * 0.1
target2 = torch.randn(2, 16) * 1e0  # Small target → small gradient

output2 = layer(x2)
loss2 = ((output2 - target2) ** 2).mean()

layer.zero_grad()
loss2.backward()

grad2 = layer.weight.grad.clone()
grad2_max = grad2.abs().max().item()
ms2 = layer.mapping_scales.mean().item()

print(f"\nStep 2 (SMALL gradient):")
print(f"  Gradient max: {grad2_max:.4e}")
print(f"  mapping_scales: {ms2:.4e}")
print(f"  Weight max (before update): {layer.weight.abs().max().item():.4e}")

# Update
with torch.no_grad():
    layer.weight -= lr * grad2
    w2_max = layer.weight.abs().max().item()

print(f"  Weight max (after update): {w2_max:.4e}")

print(f"\n🔍 Analysis:")
print(f"  Step 1 gradient: {grad1_max:.4e} → mapping_scales: {ms1_after:.4e}")
print(f"  Step 2 gradient: {grad2_max:.4e} (작음!) → mapping_scales: {ms2:.4e}")
print(f"  mapping_scales changed?: {abs(ms2 - ms1_after) > 1e-6}")

if abs(ms2 - ms1_after) < 1e-6:
    print(f"\n  ✓ mapping_scales는 gradient와 무관! (weight 크기만 봄)")
else:
    print(f"\n  ✗ mapping_scales가 변했음")

# ============================================================================
# Scenario 2: Small gradient → large gradient (reverse)
# ============================================================================
print("\n" + "=" * 80)
print("SCENARIO 2: 작은 gradient 후 큰 gradient")
print("=" * 80)

# Reset layer
layer2 = WeightNormalizedLayer(8, 16, omega=1.0)

# Step 1: Small gradient
x3 = torch.randn(2, 8) * 0.1
target3 = torch.randn(2, 16) * 1e0  # Small

output3 = layer2(x3)
loss3 = ((output3 - target3) ** 2).mean()

layer2.zero_grad()
loss3.backward()

grad3 = layer2.weight.grad.clone()
grad3_max = grad3.abs().max().item()

with torch.no_grad():
    layer2.weight -= lr * grad3

# Recompute mapping_scales
with torch.no_grad():
    _ = layer2(x3)
    ms3_after = layer2.mapping_scales.mean().item()
    w3_max = layer2.weight.abs().max().item()

print(f"\nStep 1 (small gradient):")
print(f"  Gradient max: {grad3_max:.4e}")
print(f"  Weight max (after): {w3_max:.4e}")
print(f"  mapping_scales: {ms3_after:.4e}")

# Step 2: LARGE gradient now
x4 = torch.randn(2, 8) * 0.1
target4 = torch.randn(2, 16) * 1e5  # Large!

output4 = layer2(x4)
loss4 = ((output4 - target4) ** 2).mean()

layer2.zero_grad()
loss4.backward()

grad4 = layer2.weight.grad.clone()
grad4_max = grad4.abs().max().item()
ms4 = layer2.mapping_scales.mean().item()

print(f"\nStep 2 (LARGE gradient):")
print(f"  Gradient max: {grad4_max:.4e}")
print(f"  mapping_scales (before update): {ms4:.4e}")

# Update
with torch.no_grad():
    layer2.weight -= lr * grad4
    w4_max = layer2.weight.abs().max().item()

# Recompute
with torch.no_grad():
    _ = layer2(x4)
    ms4_after = layer2.mapping_scales.mean().item()

print(f"  Weight max (after update): {w4_max:.4e}")
print(f"  mapping_scales (after update): {ms4_after:.4e}")

print(f"\n🔍 Analysis:")
print(f"  Step 1 gradient: {grad3_max:.4e} → weight: {w3_max:.4e} → ms: {ms3_after:.4e}")
print(f"  Step 2 gradient: {grad4_max:.4e} (큼!) → weight: {w4_max:.4e} → ms: {ms4_after:.4e}")
print(f"  mapping_scales increased: {ms4_after / ms3_after:.2f}x")

# ============================================================================
# Critical Test: mapping_scales는 weight만 본다
# ============================================================================
print("\n" + "=" * 80)
print("CRITICAL TEST: mapping_scales 계산 시점")
print("=" * 80)

layer3 = WeightNormalizedLayer(8, 16, omega=1.0)

# Manually set large weights
with torch.no_grad():
    layer3.weight[:] = torch.randn(16, 8) * 10.0  # Large weights

print(f"\nManually set large weights: max={layer3.weight.abs().max().item():.4e}")

# Forward pass with ZERO gradient scenario
x5 = torch.randn(2, 8) * 0.1
output5 = layer3(x5)

print(f"After forward (before any gradient):")
print(f"  mapping_scales: {layer3.mapping_scales.mean().item():.4e}")

# Now compute gradient
target5 = torch.randn(2, 16) * 1e0  # Small target
loss5 = ((output5 - target5) ** 2).mean()
layer3.zero_grad()
loss5.backward()

grad5_max = layer3.weight.grad.abs().max().item()
ms5 = layer3.mapping_scales.mean().item()

print(f"\nAfter backward (small gradient):")
print(f"  Gradient max: {grad5_max:.4e} (작음!)")
print(f"  mapping_scales: {ms5:.4e} (여전히 큼!)")

print(f"\n🔍 Conclusion:")
print(f"  Large weight ({layer3.weight.abs().max().item():.2e}) + small gradient ({grad5_max:.2e})")
print(f"  → mapping_scales는 weight 기반 ({ms5:.2e})")
print(f"  ✓ mapping_scales는 gradient를 보지 않고 weight만 본다!")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("FINAL ANSWER")
print("=" * 80)

print("""
질문: "이전 스텝의 gradient를 베이스로 조절하는건가?"

답변: **아니오!**

mapping_scales는:
  ✗ 이전 gradient 보지 않음
  ✗ 현재 gradient 보지 않음
  ✓ 현재 weight 크기만 봄

계산 시점:
  - Forward pass 시작 시 (또는 set_weights 호출 시)
  - alpha = max(|W|) / omega
  - Gradient와 완전히 독립적

작동 방식:
  Step N-1: gradient 10^5 → weight 크게 증가
  Step N:   Forward → mapping_scales = max(|W_current|) / omega (큼!)
            Backward → gradient (어떤 크기든) / mapping_scales

  따라서:
  - 이전 gradient가 크면 → weight 커짐 → mapping_scales 커짐
  - 하지만 "gradient를 기준"으로 하는 게 아니라 "weight를 기준"
  - 현재 step의 gradient가 작아도 mapping_scales는 큰 weight 기반

핵심 차이:
  "이전 gradient 기반" (X) → Step N의 gradient가 작으면 ms도 작아져야 함
  "현재 weight 기반" (O)  → Step N의 gradient가 작아도 weight가 크면 ms는 큼

예시:
  Step 1: grad=10^5 → W=10 → ms=10
  Step 2: grad=10^0 (작음!) → W는 여전히 ~10 → ms=10 (여전히 큼!)

  즉, gradient가 작아져도 mapping_scales는 안 줄어듦!
  Weight가 줄어들어야만 mapping_scales가 줄어듦.

따라서:
  mapping_scales는 "이전 gradient"가 아니라 "누적된 weight 효과"를 반영합니다.
  Gradient는 직접적인 기준이 아니고, weight가 유일한 기준입니다!
""")

print("=" * 80)
