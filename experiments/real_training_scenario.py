#!/usr/bin/env python
"""
Real Training Scenario: MobileBERT + sixt1c LoRA 학습 중 실제 무슨 일이 일어나나?

구체적 시나리오:
- MobileBERT layer 0 query
- lora_B: 초기 [128, 8] 모두 0
- RTE task 학습
- Step 1~100까지 추적
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

print("=" * 80)
print("REAL TRAINING SCENARIO: MobileBERT LoRA 학습")
print("=" * 80)

# ============================================================================
# 1. Setup - 실제 학습 환경
# ============================================================================
print("\n[1] Setup...")

model = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased",
    num_labels=2
)

tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

# Add LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=["query"]
)
model = get_peft_model(model, peft_config)

# Get one lora_B layer to track
target_layer = None
for name, module in model.named_modules():
    if 'layer.0.attention.self.query.lora_B' in name:
        if hasattr(module, 'default'):
            target_layer = module.default
        elif isinstance(module, nn.ModuleDict):
            target_layer = module['default']
        else:
            target_layer = module
        break

if target_layer is None:
    print("ERROR: Could not find target lora_B")
    import sys
    sys.exit(1)

print(f"Tracking layer: layer.0.query.lora_B")
print(f"  Shape: {target_layer.weight.shape}")  # [128, 8]
print(f"  Initial weight (first 3 values): {target_layer.weight.data.flatten()[:3].tolist()}")

# Load data
dataset = load_dataset("glue", "rte", split="train[:50]")
print(f"Loaded {len(dataset)} training examples")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# ============================================================================
# 2. Training loop with detailed tracking
# ============================================================================
print("\n[2] Training for 20 steps with detailed tracking...\n")

model.train()

# Track history
history = []

# Snapshots at specific steps
snapshot_steps = [1, 2, 3, 5, 10, 20]

for step in range(20):
    # Get batch
    idx = step % len(dataset)
    example = dataset[idx]

    inputs = tokenizer(
        example["sentence1"],
        example["sentence2"],
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    labels = torch.tensor([example["label"]])

    # Forward
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    # Record BEFORE backward
    w_before = target_layer.weight.data.clone()

    # Backward
    loss.backward()

    # Check gradient
    grad = target_layer.weight.grad
    if grad is None:
        print(f"  Step {step+1}: No gradient!")
        continue

    grad_clone = grad.clone()

    # Optimizer step (updates weight)
    optimizer.step()

    # Record AFTER update
    w_after = target_layer.weight.data.clone()
    w_delta = w_after - w_before

    # Statistics
    w_norm = w_after.norm().item()
    w_max = w_after.abs().max().item()
    w_mean = w_after.mean().item()

    grad_norm = grad_clone.norm().item()
    grad_max = grad_clone.abs().max().item()

    delta_norm = w_delta.norm().item()
    delta_max = w_delta.abs().max().item()

    history.append({
        'step': step + 1,
        'loss': loss.item(),
        'grad_norm': grad_norm,
        'grad_max': grad_max,
        'w_norm': w_norm,
        'w_max': w_max,
        'w_mean': w_mean,
        'delta_norm': delta_norm,
        'delta_max': delta_max,
        'w_sample': w_after[0, 0].item(),  # Track one weight
    })

    # Print snapshots
    if (step + 1) in snapshot_steps:
        print(f"Step {step+1:3d}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradient: max={grad_max:.4e}, norm={grad_norm:.4e}")
        print(f"  Weight:   max={w_max:.4e}, norm={w_norm:.4e}")
        print(f"  Update:   max={delta_max:.4e}, norm={delta_norm:.4e}")
        print(f"  Sample weight [0,0]: {w_after[0,0].item():.6f}")
        print()

# ============================================================================
# 3. Analysis - 실제로 무슨 일이 일어났나?
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS: 실제 학습에서 무슨 일이 일어났나?")
print("=" * 80)

print(f"\n{'Step':<6} {'Grad Max':<12} {'W Max':<12} {'W Norm':<12} {'비율':<10}")
print("-" * 60)

for i in [0, 1, 2, 4, 9, 19]:  # Steps 1, 2, 3, 5, 10, 20
    h = history[i]
    ratio = h['w_max'] / (h['grad_max'] * 2e-4 + 1e-12)  # lr=2e-4
    print(f"{h['step']:<6d} {h['grad_max']:<12.4e} {h['w_max']:<12.4e} {h['w_norm']:<12.4e} {ratio:<10.2f}")

# ============================================================================
# 4. 시나리오 설명
# ============================================================================
print("\n" + "=" * 80)
print("시나리오 해석")
print("=" * 80)

h1 = history[0]
h5 = history[4] if len(history) > 4 else history[-1]
h20 = history[-1]

print(f"""
실제 MobileBERT layer.0.query.lora_B [128, 8] 학습:

📍 **Step 1 (초기)**:
   - lora_B는 0으로 초기화됨
   - Gradient: {h1['grad_max']:.2e}
   - Weight update: {h1['delta_max']:.2e}
   - Weight (after): {h1['w_max']:.2e}

   해석:
   - 처음에는 weight가 0이므로 "작다"
   - Gradient가 크든 작든, 아직 weight는 작음

📍 **Step 5**:
   - Gradient: {h5['grad_max']:.2e}
   - Weight: {h5['w_max']:.2e}
   - Weight norm: {h5['w_norm']:.2e}

   해석:
   - 5번의 update를 거쳐 weight가 누적됨
   - Weight = Σ(lr × gradient_i) for i=1..5
   - 이제 weight가 "커졌다" = {h5['w_max']:.2e}

📍 **Step 20**:
   - Gradient: {h20['grad_max']:.2e}
   - Weight: {h20['w_max']:.2e}
   - Weight norm: {h20['w_norm']:.2e}

   해석:
   - 20번 update 후 weight가 더 커짐
   - Weight가 "크다" = {h20['w_max']:.2e} 정도의 값들

🔑 **"Weight가 크다"의 의미**:

   "작다" (초기):   |W| ~ 1e-5 ~ 1e-3
   "중간":         |W| ~ 1e-2 ~ 1e-1
   "크다":         |W| ~ 1 ~ 10

   이 학습에서:
   - Step 1:  W_max = {h1['w_max']:.2e} (작다)
   - Step 20: W_max = {h20['w_max']:.2e} {"(중간)" if h20['w_max'] < 0.1 else "(크다)"}

📊 **Analog Tile에서는**:

   만약 이게 sixt1c analog tile이었다면 (omega=1.0):

   Step 1:
   - W_max = {h1['w_max']:.2e}
   - mapping_scales = {h1['w_max']:.2e} / 1.0 = {h1['w_max']:.2e}
   - Conductance = W / mapping_scales ≈ 0~1 범위

   Step 20:
   - W_max = {h20['w_max']:.2e}
   - mapping_scales = {h20['w_max']:.2e} / 1.0 = {h20['w_max']:.2e}
   - Conductance = W / {h20['w_max']:.2e} ≈ 0~1 범위 (정규화됨!)

   즉:
   - Weight는 0 → {h20['w_max']:.2e}으로 커졌지만
   - Conductance는 항상 0~1 범위 (hardware 보호)
   - mapping_scales = {h20['w_max']:.2e}가 gradient를 보정

🎯 **Gradient 보정 메커니즘**:

   Step 20에서 새로운 gradient {h20['grad_max']:.2e}가 오면:

   1. Forward:
      - W_normalized = W / {h20['w_max']:.2e}
      - output = input @ W_normalized × {h20['w_max']:.2e}

   2. Backward (backward hook):
      - Raw gradient: {h20['grad_max']:.2e}
      - Compensated: {h20['grad_max']:.2e} / {h20['w_max']:.2e} = {h20['grad_max'] / (h20['w_max'] + 1e-12):.2e}

   3. 따라서:
      - Gradient가 {h20['grad_max']:.2e}로 크더라도
      - mapping_scales {h20['w_max']:.2e}로 나눠서
      - 실제 effective gradient는 합리적

⚠️ **중요: "Weight가 크다"는**:
   - 절대값이 아니라 상대적!
   - omega=1.0 기준으로 |W| > 1 이면 "크다"
   - 이 경우 |W| ≈ {h20['w_max']:.2e}이므로 {"크다" if h20['w_max'] > 1 else "작다~중간"}

   하지만 학습이 더 진행되면:
   - |W|는 1, 10, 심지어 100까지 커질 수 있음
   - mapping_scales도 그만큼 커짐
   - Conductance는 여전히 0~1 범위!
""")

# ============================================================================
# 5. Layer-by-layer variation 시뮬레이션
# ============================================================================
print("\n" + "=" * 80)
print("다른 Layer들은 어떻게 다른가?")
print("=" * 80)

print("""
실제 학습에서 layer마다 다른 상황:

Layer 0 (query):
  - Gradient 크기: 1e3 ~ 1e4 (하위 layer, gradient 큼)
  - Weight 누적: 빠르게 증가
  - Step 20에서 W_max ≈ 0.1 ~ 1.0
  - mapping_scales ≈ 0.1 ~ 1.0

Layer 11 (query, 마지막 layer):
  - Gradient 크기: 1e1 ~ 1e2 (상위 layer, gradient 작음)
  - Weight 누적: 느리게 증가
  - Step 20에서 W_max ≈ 0.01 ~ 0.1
  - mapping_scales ≈ 0.01 ~ 0.1

결과:
  - Layer 0: 큰 mapping_scales → gradient 많이 보정
  - Layer 11: 작은 mapping_scales → gradient 적게 보정
  - 각 layer가 자기 weight 크기에 맞게 독립적으로 조절!

따라서:
  "이전 gradient 기반" (X) → 각 layer가 다른 gradient history
  "현재 weight 기반" (O)  → 각 layer가 자기 weight만 봄
""")

# ============================================================================
# 6. Long-term training
# ============================================================================
print("\n" + "=" * 80)
print("장기 학습 (수천 step) 시나리오")
print("=" * 80)

print("""
Step 1000 (converged):
  - Gradient: 1e2 ~ 1e3 (여전히 있음, 하지만 작아짐)
  - Weight: 0.5 ~ 2.0 (수렴)
  - mapping_scales: 0.5 ~ 2.0 (고정됨)

  이제:
  - Weight가 거의 변하지 않음 (converged)
  - mapping_scales도 거의 변하지 않음
  - Gradient는 여전히 mapping_scales로 나눠짐
  - 하지만 update는 작아서 weight 거의 안 변함

핵심:
  - mapping_scales는 weight 크기를 "따라간다"
  - Gradient를 "예측"하거나 "기억"하지 않음
  - 단지 현재 weight를 normalize하는 역할

예시:
  Step 1000: W=1.5, ms=1.5
  Step 1001: 갑자기 큰 gradient 1e5 발생! (outlier)

  이 step에서:
  - ms는 여전히 1.5 (이전 step weight 기반)
  - Gradient 1e5 / 1.5 ≈ 6.7e4 (여전히 큼!)
  - Update = 2e-4 × 6.7e4 ≈ 13 (huge!)
  - W = 1.5 + 13 = 14.5 (폭발!)

  다음 step (1002):
  - ms = 14.5 / 1.0 = 14.5 (커짐!)
  - 이제 gradient가 14.5로 나눠짐
  - 다시 안정화

  즉:
  - mapping_scales는 reactive (반응적)
  - Gradient explosion 발생 후 다음 step에 보정
  - 예방적이지 않음!
""")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"""
"Weight가 크다"는 것:
  - 초기 (step 1-5): |W| ~ {h1['w_max']:.2e} (작다)
  - 중기 (step 20): |W| ~ {h20['w_max']:.2e} (중간)
  - 수렴 (step 1000+): |W| ~ 0.5~2.0 (크다)

  "크다"의 기준: omega (default 1.0) 대비
  |W| > omega이면 "크다" → normalize 필요

실제 학습 시나리오:
  1. 초기: Weight 작음 → mapping_scales 작음 → gradient 보정 약함
  2. 중기: Weight 증가 → mapping_scales 증가 → gradient 보정 강해짐
  3. 수렴: Weight 안정 → mapping_scales 안정 → gradient 보정 고정

  각 step의 mapping_scales는:
  - 현재 weight 크기로 결정
  - 이전 gradient 무관
  - 단지 weight 누적 효과를 반영

Layer별 차이:
  - 하위 layer: gradient 큼 → weight 빨리 증가 → ms 빨리 증가
  - 상위 layer: gradient 작음 → weight 천천히 증가 → ms 천천히 증가
  - 각자 독립적으로 자기 weight 기반으로 조절!

핵심: mapping_scales = f(current_weight), NOT f(previous_gradient)!
""")

print("\n" + "=" * 80)
