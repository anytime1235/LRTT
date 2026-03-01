#!/usr/bin/env python3
"""
최종 테스트: 실제 LoRA 구조에서 base_layer weight 변화 확인

LoRA 구조:
  output = base_layer(x) + scaling * lora_B(lora_A(x))

이 구조에서 backward + optimizer.step() 했을 때
base_layer weight가 변하는지 확인
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

from sixt1c_config import (
    gen_sixt1c_lora_config_trainable,
    gen_softbounds_base_layer_config_trainable
)
from smart_conversion import convert_base_and_lora_separately

print("=" * 80)
print("FINAL TEST: LoRA Structure - base_layer Weight Change")
print("=" * 80)

# Create minimal LoRA model
print("\n[Creating LoRA model]")
model_config = AutoConfig.from_pretrained('google/mobilebert-uncased')
model_config.num_labels = 2
model_config.num_hidden_layers = 2  # Only 2 layers for speed
print(f"  Model: mobilebert (2 layers only)")

model = AutoModelForSequenceClassification.from_pretrained(
    'google/mobilebert-uncased',
    config=model_config,
    ignore_mismatched_sizes=True
)

# Apply LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=1.0,
    target_modules=['query'],
    bias='none',
    lora_dropout=0.0,
)
model = get_peft_model(model, peft_config)
print(f"✓ LoRA applied (rank=8, alpha=1.0)")

# Convert to analog
base_config = gen_softbounds_base_layer_config_trainable()
lora_config = gen_sixt1c_lora_config_trainable()

model = convert_base_and_lora_separately(
    model,
    base_layer_config=base_config,
    lora_config=lora_config,
    lora_trainable=True
)
print(f"✓ Converted to analog")

# Freeze base_layer, classifier
for name, param in model.named_parameters():
    if 'base_layer' in name or 'classifier' in name:
        param.requires_grad = False
    elif 'lora' in name:
        param.requires_grad = True

print(f"✓ base_layer frozen, lora_A/B trainable")

# Optimizer
optimizer = AnalogSGD(model.parameters(), lr=0.001)
optimizer.regroup_param_groups(model)
print(f"✓ AnalogSGD optimizer created")

# Tokenizer and inputs
tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
inputs = tokenizer(
    ["This is a test sentence."] * 2,
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors='pt'
)
labels = torch.tensor([1, 0])
print(f"✓ Input prepared (batch=2, seq_len=128)")

# Collect base_layer modules
base_layers = {}
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'base_layer' in name and 'query' in name:
        base_layers[name] = module

print(f"\n[Found {len(base_layers)} base_layer (query) modules]")
for i, name in enumerate(list(base_layers.keys())[:3]):
    print(f"  {i}: {name}")

# Training loop
print("\n" + "=" * 80)
print("TRAINING: Forward + Backward + Optimizer.step()")
print("=" * 80)

for step in range(3):
    print(f"\n--- Step {step + 1} ---")

    # Capture weights before
    weights_before = {}
    for name, module in base_layers.items():
        w = module.get_weights()
        weights_before[name] = (w[0] if isinstance(w, tuple) else w).clone()

    # Forward
    model.train()
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    # Backward
    loss.backward()

    # Check gradients
    base_has_grad = False
    lora_has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if 'base_layer' in name:
                base_has_grad = True
            if 'lora' in name:
                lora_has_grad = True

    print(f"  Loss: {loss.item():.4f}")
    print(f"  base_layer has gradient? {base_has_grad}")
    print(f"  lora has gradient? {lora_has_grad}")

    # Optimizer step
    optimizer.step()

    # Capture weights after
    weights_after = {}
    for name, module in base_layers.items():
        w = module.get_weights()
        weights_after[name] = (w[0] if isinstance(w, tuple) else w).clone()

    # Analyze changes
    print(f"\n  Weight changes:")
    for i, (name, w_before) in enumerate(weights_before.items()):
        w_after = weights_after[name]
        delta = (w_after - w_before).abs()

        layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
        print(f"    Layer {layer_idx}: max={delta.max():.6f}, mean={delta.mean():.6f}, changed={((delta > 1e-8).sum().item())}/{delta.numel()}")

        if i >= 2:  # Show first 3 only
            break

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)

# Check if any base_layer changed
any_changed = False
for name, w_before in weights_before.items():
    w_after = weights_after[name]
    if (w_after - w_before).abs().max() > 1e-8:
        any_changed = True
        break

if any_changed:
    print("✗ base_layer weights CHANGED in LoRA structure!")
    print("  → This matches the verification script results")
    print("  → LoRA structure causes base_layer weight changes")
    print("\nPossible causes:")
    print("  1. PEFT framework's backward hooks")
    print("  2. LoRA computation graph affecting base_layer")
    print("  3. Analog tile update in LoRA context")
else:
    print("✓ base_layer weights did NOT change")
    print("  → Need further investigation")
print("=" * 80)
