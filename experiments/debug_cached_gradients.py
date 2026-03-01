#!/usr/bin/env python3
"""
디버깅 스크립트: AnalogSGD가 캐싱하는 값 확인

목적:
1. base_layer._x (cached input) 존재 여부
2. base_layer._d (cached output gradient) 존재 여부
3. Layer 0 vs Layer 1+의 차이 확인
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
print("DEBUG: Cached Input/Gradient Analysis")
print("=" * 80)

# Create minimal LoRA model (2 layers only)
print("\n[Creating LoRA model (2 layers)]")
model_config = AutoConfig.from_pretrained('google/mobilebert-uncased')
model_config.num_labels = 2
model_config.num_hidden_layers = 2

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

# Convert to analog
base_config = gen_softbounds_base_layer_config_trainable()
lora_config = gen_sixt1c_lora_config_trainable()
model = convert_base_and_lora_separately(
    model,
    base_layer_config=base_config,
    lora_config=lora_config,
    lora_trainable=True
)

# Freeze
for name, param in model.named_parameters():
    if 'base_layer' in name or 'classifier' in name:
        param.requires_grad = False
    elif 'lora' in name:
        param.requires_grad = True

print("✓ Model created and converted")

# Optimizer
optimizer = AnalogSGD(model.parameters(), lr=0.001)
optimizer.regroup_param_groups(model)
print("✓ Optimizer created")

# Collect modules
base_layers = []
lora_a_layers = []
lora_b_layers = []

for name, module in model.named_modules():
    if isinstance(module, AnalogLinear):
        if 'base_layer' in name and 'query' in name:
            base_layers.append((name, module))
        elif 'lora_A' in name and 'query' in name:
            lora_a_layers.append((name, module))
        elif 'lora_B' in name and 'query' in name:
            lora_b_layers.append((name, module))

print(f"\n[Found modules]")
print(f"  base_layer: {len(base_layers)}")
print(f"  lora_A: {len(lora_a_layers)}")
print(f"  lora_B: {len(lora_b_layers)}")

# Prepare input
tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
inputs = tokenizer(
    ["This is a test sentence."] * 2,
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors='pt'
)
labels = torch.tensor([1, 0])

print("\n" + "=" * 80)
print("STEP 1: Check BEFORE forward/backward")
print("=" * 80)

for name, module in base_layers:
    layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
    print(f"\n[Layer {layer_idx}] {name}")
    print(f"  Has _x (cached input)? {hasattr(module, '_x')}")
    print(f"  Has _d (cached gradient)? {hasattr(module, '_d')}")

print("\n" + "=" * 80)
print("STEP 2: Forward pass")
print("=" * 80)

model.train()
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")

print("\n" + "=" * 80)
print("STEP 3: Check AFTER forward, BEFORE backward")
print("=" * 80)

for name, module in base_layers:
    layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
    print(f"\n[Layer {layer_idx}] {name}")
    print(f"  Has _x? {hasattr(module, '_x')}")
    if hasattr(module, '_x'):
        print(f"    _x shape: {module._x.shape}")
        print(f"    _x norm: {module._x.norm():.4f}")
    print(f"  Has _d? {hasattr(module, '_d')}")
    if hasattr(module, '_d'):
        print(f"    _d is None? {module._d is None}")

print("\n" + "=" * 80)
print("STEP 4: Backward pass")
print("=" * 80)

loss.backward()
print("✓ Backward completed")

print("\n" + "=" * 80)
print("STEP 5: Check AFTER backward, BEFORE optimizer.step()")
print("=" * 80)

for name, module in base_layers:
    layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
    print(f"\n[Layer {layer_idx}] {name}")

    # Check _x
    print(f"  _x (cached input):")
    if hasattr(module, '_x'):
        print(f"    Exists: True")
        print(f"    Shape: {module._x.shape}")
        print(f"    Norm: {module._x.norm():.4f}")
        print(f"    Range: [{module._x.min():.4f}, {module._x.max():.4f}]")
    else:
        print(f"    Exists: False")

    # Check _d
    print(f"  _d (cached output gradient):")
    if hasattr(module, '_d'):
        print(f"    Exists: True")
        if module._d is not None:
            print(f"    Is None: False")
            print(f"    Shape: {module._d.shape}")
            print(f"    Norm: {module._d.norm():.6f}")
            print(f"    Max abs: {module._d.abs().max():.6f}")
            print(f"    Mean abs: {module._d.abs().mean():.6f}")
            print(f"    Range: [{module._d.min():.6f}, {module._d.max():.6f}]")

            # Check if it's actually zero
            if module._d.abs().max() < 1e-8:
                print(f"    → Effectively ZERO")
            else:
                print(f"    → NON-ZERO! (this will cause weight update)")
        else:
            print(f"    Is None: True")
            print(f"    → No gradient cached")
    else:
        print(f"    Exists: False")

    # Check PyTorch gradient
    print(f"  PyTorch param.grad:")
    has_grad = False
    for param in module.parameters():
        if param.grad is not None:
            has_grad = True
            print(f"    Exists: True")
            print(f"    Norm: {param.grad.norm():.6f}")
            break
    if not has_grad:
        print(f"    Exists: False (frozen as expected)")

# Also check lora layers for comparison
print("\n" + "=" * 80)
print("COMPARISON: LoRA Layers")
print("=" * 80)

for name, module in lora_a_layers[:1]:  # Just first one
    layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
    print(f"\n[lora_A Layer {layer_idx}]")

    if hasattr(module, '_x'):
        print(f"  _x: Shape {module._x.shape}, Norm {module._x.norm():.4f}")
    if hasattr(module, '_d') and module._d is not None:
        print(f"  _d: Shape {module._d.shape}, Norm {module._d.norm():.6f}")

for name, module in lora_b_layers[:1]:  # Just first one
    layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
    print(f"\n[lora_B Layer {layer_idx}]")

    if hasattr(module, '_x'):
        print(f"  _x: Shape {module._x.shape}, Norm {module._x.norm():.4f}")
    if hasattr(module, '_d') and module._d is not None:
        print(f"  _d: Shape {module._d.shape}, Norm {module._d.norm():.6f}")

print("\n" + "=" * 80)
print("STEP 6: Compute predicted weight gradient")
print("=" * 80)

for name, module in base_layers:
    layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
    print(f"\n[Layer {layer_idx}]")

    if hasattr(module, '_x') and hasattr(module, '_d') and module._d is not None:
        x = module._x
        d = module._d

        # Compute weight gradient: dL/dW = x^T @ d
        # x: [batch, in_features]
        # d: [batch, out_features]
        # grad_w: [in_features, out_features]

        grad_w = x.T @ d

        print(f"  x shape: {x.shape}")
        print(f"  d shape: {d.shape}")
        print(f"  grad_w shape: {grad_w.shape}")
        print(f"  grad_w norm: {grad_w.norm():.6f}")
        print(f"  grad_w max: {grad_w.abs().max():.6f}")
        print(f"  grad_w mean: {grad_w.abs().mean():.6f}")

        if grad_w.abs().max() > 1e-8:
            print(f"  → Will cause weight update!")

            # Estimate weight change
            lr = 0.001
            dw_min = 0.001  # SoftBounds

            # Simplified: max weight change
            max_grad = grad_w.abs().max().item()
            max_pulses = max_grad * lr / dw_min
            max_weight_change = max_pulses * dw_min

            print(f"  Estimated max weight change: ~{max_weight_change:.6f}")
        else:
            print(f"  → No weight update (grad_w ≈ 0)")
    else:
        print(f"  Cannot compute (missing _x or _d)")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Summary
print("\nSummary:")
for name, module in base_layers:
    layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'

    has_x = hasattr(module, '_x')
    has_d = hasattr(module, '_d') and module._d is not None
    d_nonzero = False

    if has_d:
        d_nonzero = module._d.abs().max() > 1e-8

    status = "✓ No update" if not (has_x and has_d and d_nonzero) else "✗ WILL UPDATE"

    print(f"  Layer {layer_idx}: _x={has_x}, _d={has_d}, d_nonzero={d_nonzero} → {status}")

print("\n" + "=" * 80)
