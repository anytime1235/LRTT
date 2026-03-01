#!/usr/bin/env python
"""
Explain: How analog tile handles gradient range 10^0 ~ 10^5

핵심 질문: "gradient가 10^5 ~ 10^0까지 변할때 어떻게 처리한다는거야?"
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
import torch.nn as nn

print("=" * 80)
print("Gradient Range Handling in Analog Tiles")
print("=" * 80)

# ============================================================================
# Demonstration with simple digital layer (to understand the mechanism)
# ============================================================================
print("\n[Demo] Weight normalization and gradient compensation\n")

class SimpleAnalogSimulator(nn.Module):
    """Simplified analog tile behavior simulator"""

    def __init__(self, in_features, out_features, omega=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.omega = omega  # weight_scaling_omega
        self.mapping_scales = None
        self.columnwise = True  # weight_scaling_columnwise

    def forward(self, x):
        # Step 1: Normalize weights (like set_weights with apply_weight_scaling)
        if self.omega > 0:
            if self.columnwise:
                # Per-column normalization
                weight_max = self.weight.abs().max(dim=1, keepdim=True)[0]  # [out, 1]
            else:
                # Global normalization
                weight_max = self.weight.abs().max()

            # alpha = weight_max / omega, with protection for zeros
            alpha = weight_max / self.omega
            alpha = torch.where(alpha == 0, torch.ones_like(alpha), alpha)

            # Store mapping_scales
            self.mapping_scales = alpha.detach()

            # Normalize weights
            w_normalized = self.weight / alpha
        else:
            w_normalized = self.weight
            self.mapping_scales = torch.ones(self.weight.size(0), 1)

        # Step 2: Forward pass with normalized weights
        output = torch.matmul(x, w_normalized.t())

        # Step 3: Scale output back (restore range)
        if self.omega > 0:
            output = output * self.mapping_scales.t()

        return output

# Create layer
layer = SimpleAnalogSimulator(8, 16, omega=1.0)
print(f"Layer: {layer.weight.shape}")
print(f"Initial weight: mean={layer.weight.mean().item():.4f}, max_abs={layer.weight.abs().max().item():.4f}")

# ============================================================================
# Test with different gradient scales
# ============================================================================
print("\n[Test] Gradient scales from 10^0 to 10^5\n")

gradient_scales = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
lr = 1e-4

print(f"{'Grad Scale':<12} {'Grad Max':<15} {'Update Max':<15} {'W Max After':<15} {'Map Scale':<12}")
print("-" * 80)

for grad_scale in gradient_scales:
    # Create input and target to induce desired gradient
    x = torch.randn(4, 8) * 0.1
    target = torch.randn(4, 16) * grad_scale

    # Forward
    output = layer(x)
    loss = ((output - target) ** 2).mean()

    # Backward
    layer.zero_grad()
    loss.backward()

    # Check gradient
    grad = layer.weight.grad
    grad_max = grad.abs().max().item()

    # Manual SGD update
    with torch.no_grad():
        w_before = layer.weight.clone()
        layer.weight -= lr * grad
        update_max = (layer.weight - w_before).abs().max().item()
        w_max_after = layer.weight.abs().max().item()

    # Forward again to recompute mapping_scales
    with torch.no_grad():
        _ = layer(x)
        ms_mean = layer.mapping_scales.mean().item() if layer.mapping_scales is not None else 1.0

    print(f"{grad_scale:<12.0e} {grad_max:<15.4e} {update_max:<15.4e} {w_max_after:<15.4e} {ms_mean:<12.4e}")

# ============================================================================
# Explanation
# ============================================================================
print("\n" + "=" * 80)
print("MECHANISM EXPLANATION")
print("=" * 80)

print("""
1. **Weight Normalization (mapping_scales)**:

   Forward:
   --------
   alpha = max(|W|) / omega  (per column if columnwise)
   alpha[alpha==0] = 1.0     (protect zero columns)

   W_norm = W / alpha        (normalize to [-omega, +omega] range)
   output = input @ W_norm^T × alpha  (scale back)

   Result: Output magnitude is preserved, but W is normalized

2. **Backward Pass**:

   Gradient flows through:
   ∂L/∂W = ∂L/∂output × ∂output/∂W

   ∂output/∂W = input^T × alpha / alpha = input^T  (alpha cancels!)

   BUT: PyTorch autograd sees the W/alpha operation, so:
   ∂L/∂W = (∂L/∂output × alpha) @ input  (alpha factor in gradient!)

3. **Backward Hook Compensation** (when omega > 0):

   Without hook: grad appears larger by factor of alpha
   With hook: grad is divided by alpha to compensate

   This is what the backward hook does in periphery.py!

4. **Handling Gradient Range 10^0 ~ 10^5**:

   Scenario 1: Gradient 10^0 (small)
   - Weight update: lr × 10^0 = 1e-4 × 1 = 1e-4 (tiny)
   - Weight grows slowly
   - mapping_scales remain close to 1.0

   Scenario 2: Gradient 10^5 (large)
   - Weight update: lr × 10^5 = 1e-4 × 10^5 = 10 (large!)
   - Weight grows quickly
   - mapping_scales increase to ~10 or more
   - Forward pass: W_norm = W / 10 (keeps conductance in range)
   - Next backward: gradient compensated by /10

5. **Layer-to-Layer Variation**:

   If Layer 1 has gradient 10^5 and Layer 2 has gradient 10^0:

   Layer 1:
   - Large update → large weights
   - Large mapping_scales (e.g., 10)
   - Conductance stays in [-omega, +omega]

   Layer 2:
   - Small update → small weights
   - Small mapping_scales (e.g., 1)
   - Conductance stays in [-omega, +omega]

   Both layers maintain conductance in hardware range!

6. **Key Insight**:

   mapping_scales는:
   - Gradient 크기를 제한하지 않음
   - Weight 크기를 제한하지 않음
   - 단지 conductance (normalized weight)를 [-omega, +omega]에 유지

   따라서:
   - Gradient 10^5 → 그대로 처리됨 (lr 곱해서 update)
   - Weight가 커지면 → mapping_scales가 커져서 normalize
   - Conductance는 항상 hardware 범위 내

7. **실제 동작**:

   Step 1: Gradient 10^5 도착
   Step 2: Update = lr × gradient = 1e-4 × 10^5 = 10
   Step 3: Weight = Weight + 10 (큰 update!)
   Step 4: Next forward → mapping_scales recalculated
   Step 5: W_norm = Weight / mapping_scales (hardware 범위로)
   Step 6: Next backward → gradient / mapping_scales (보정)

""")

print("=" * 80)
print("CONCLUSION")
print("=" * 80)

print("""
질문: "gradient가 10^5 ~ 10^0까지 변할때 어떻게 처리?"

답변:

1. **Gradient는 그대로 처리됩니다** (clipping 없음)
   - Update = learning_rate × gradient
   - Gradient 10^5면 update도 크고, 10^0이면 update도 작음

2. **mapping_scales는 conductance 정규화용입니다**:
   - Forward: W_norm = W / alpha (hardware 범위로)
   - Output: output × alpha (원래 크기 복원)
   - Backward: gradient / alpha (chain rule 보정)

3. **Layer마다 독립적으로 처리됩니다**:
   - 각 layer의 weight 크기에 따라 mapping_scales 다름
   - Gradient 큰 layer → weight 크게 증가 → mapping_scales 크게
   - Gradient 작은 layer → weight 작게 증가 → mapping_scales 작게

4. **Hardware 범위 보호**:
   - Conductance는 항상 [-omega, +omega] 범위 (예: [-1, +1])
   - 하지만 effective weight (W = conductance × alpha)는 제한 없음
   - 따라서 gradient 크기와 무관하게 학습 가능

5. **실제 10^5 gradient는**:
   - 첫 step: 그대로 10^5 적용 → weight 크게 증가
   - 다음 step: mapping_scales가 커져서 normalize
   - 이후: gradient가 alpha로 보정되어 합리적 크기로

핵심: **Gradient 크기는 제한하지 않고, conductance만 normalize!**
""")

print("\n" + "=" * 80)
