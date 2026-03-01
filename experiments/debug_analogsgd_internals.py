#!/usr/bin/env python3
"""
AnalogSGD 내부 메커니즘 디버깅

AnalogSGD가 실제로 어떻게 동작하는지 확인:
1. optimizer.analog_tile_array 확인
2. step() 실행 전후 비교
3. 어떤 메서드가 호출되는지 추적
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
print("DEBUG: AnalogSGD Internal Mechanism")
print("=" * 80)

# Create minimal LoRA model
print("\n[Creating LoRA model (2 layers)]")
model_config = AutoConfig.from_pretrained('google/mobilebert-uncased')
model_config.num_labels = 2
model_config.num_hidden_layers = 2

model = AutoModelForSequenceClassification.from_pretrained(
    'google/mobilebert-uncased',
    config=model_config,
    ignore_mismatched_sizes=True
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=1.0,
    target_modules=['query'],
    bias='none',
    lora_dropout=0.0,
)
model = get_peft_model(model, peft_config)

base_config = gen_softbounds_base_layer_config_trainable()
lora_config = gen_sixt1c_lora_config_trainable()
model = convert_base_and_lora_separately(
    model,
    base_layer_config=base_config,
    lora_config=lora_config,
    lora_trainable=True
)

for name, param in model.named_parameters():
    if 'base_layer' in name or 'classifier' in name:
        param.requires_grad = False
    elif 'lora' in name:
        param.requires_grad = True

print("✓ Model created")

# Create optimizer
optimizer = AnalogSGD(model.parameters(), lr=0.001)
print("✓ Optimizer created")

print("\n" + "=" * 80)
print("INSPECT: AnalogSGD attributes")
print("=" * 80)

print(f"\nOptimizer type: {type(optimizer).__name__}")
print(f"Optimizer class: {optimizer.__class__.__name__}")

# Check attributes
attrs = dir(optimizer)
important_attrs = [a for a in attrs if not a.startswith('_') and 'analog' in a.lower()]
print(f"\nAnalog-related attributes:")
for attr in important_attrs:
    print(f"  {attr}")

# Check if it has analog_tile_array or similar
if hasattr(optimizer, 'analog_tile_array'):
    print(f"\nanalog_tile_array: {optimizer.analog_tile_array}")
if hasattr(optimizer, 'tiles'):
    print(f"\ntiles: {optimizer.tiles}")

# Check param_groups
print(f"\nNumber of param_groups: {len(optimizer.param_groups)}")
for i, group in enumerate(optimizer.param_groups):
    print(f"\n  Group {i}:")
    print(f"    Keys: {group.keys()}")
    print(f"    LR: {group.get('lr', 'N/A')}")
    print(f"    Num params: {len(group.get('params', []))}")

# Now call regroup_param_groups
print("\n" + "=" * 80)
print("CALL: optimizer.regroup_param_groups(model)")
print("=" * 80)

optimizer.regroup_param_groups(model)
print("✓ regroup_param_groups() called")

# Check again
if hasattr(optimizer, 'analog_tile_array'):
    print(f"\nanalog_tile_array after regroup:")
    if optimizer.analog_tile_array is not None:
        print(f"  Type: {type(optimizer.analog_tile_array)}")
        print(f"  Length: {len(optimizer.analog_tile_array) if hasattr(optimizer.analog_tile_array, '__len__') else 'N/A'}")

# Collect analog modules
analog_modules = []
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        layer_type = 'base_layer' if 'base_layer' in name else ('lora_A' if 'lora_A' in name else 'lora_B')
        layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
        analog_modules.append((f"{layer_type} L{layer_idx}", name, module))

print(f"\n[Found {len(analog_modules)} analog modules]")

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
print("STEP 1: Capture weights BEFORE training")
print("=" * 80)

weights_before = {}
for label, name, module in analog_modules:
    w = module.get_weights()
    w_tensor = w[0] if isinstance(w, tuple) else w
    weights_before[name] = w_tensor.clone()
    print(f"  {label}: {w_tensor.shape}, mean={w_tensor.mean():.6f}")

print("\n" + "=" * 80)
print("STEP 2: Forward + Backward")
print("=" * 80)

model.train()
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")

loss.backward()
print("✓ Backward done")

# Check gradients
print("\n[Gradients after backward]")
for label, name, module in analog_modules:
    has_grad = False
    grad_norm = 0.0

    for param in module.parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            break

    print(f"  {label}: has_grad={has_grad}, norm={grad_norm:.6f}")

print("\n" + "=" * 80)
print("STEP 3: optimizer.step() - INSTRUMENTED")
print("=" * 80)

# Monkey-patch to see what's happening
original_step = optimizer.step

def instrumented_step(*args, **kwargs):
    print("\n>>> optimizer.step() called")
    print(f"    Args: {args}")
    print(f"    Kwargs: {kwargs}")

    # Check if we can access tiles
    if hasattr(optimizer, 'analog_tile_array') and optimizer.analog_tile_array is not None:
        print(f"    analog_tile_array exists, length: {len(optimizer.analog_tile_array)}")

    result = original_step(*args, **kwargs)

    print(">>> optimizer.step() finished")
    return result

optimizer.step = instrumented_step

# Call step
optimizer.step()

print("\n" + "=" * 80)
print("STEP 4: Check weights AFTER step")
print("=" * 80)

for label, name, module in analog_modules:
    w = module.get_weights()
    w_tensor = w[0] if isinstance(w, tuple) else w

    delta = (w_tensor - weights_before[name]).abs()

    print(f"\n  {label}:")
    print(f"    Max change: {delta.max():.6f}")
    print(f"    Mean change: {delta.mean():.6f}")
    print(f"    Num changed (>1e-8): {(delta > 1e-8).sum().item()}/{delta.numel()}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

base_changed = []
for label, name, module in analog_modules:
    if 'base_layer' not in label:
        continue

    w = module.get_weights()
    w_tensor = w[0] if isinstance(w, tuple) else w
    delta = (w_tensor - weights_before[name]).abs()

    if delta.max() > 1e-8:
        base_changed.append(label)
        print(f"✗ {label} CHANGED (max delta: {delta.max():.6f})")
    else:
        print(f"✓ {label} unchanged")

if base_changed:
    print(f"\n{len(base_changed)} base_layer(s) changed!")
else:
    print(f"\nNo base_layer changed")

print("=" * 80)
