#!/usr/bin/env python3
"""
TorchInferenceRPUConfig 사용 시 base_layer가 진짜 frozen인지 테스트

구조:
- base_layer: TorchInferenceRPUConfig (inference only)
- lora_A/B: SingleRPUConfig (trainable)

예상: base_layer는 업데이트 안되어야 함
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

from sixt1c_config import (
    gen_sixt1c_lora_config_trainable,
    gen_softbounds_base_layer_config  # ← TorchInferenceRPUConfig!
)
from smart_conversion import convert_base_and_lora_separately

print("=" * 80)
print("TEST: TorchInferenceRPUConfig for base_layer")
print("=" * 80)

# Create LoRA model
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

# Convert to analog
print("\n[Converting to analog]")
print("  base_layer: TorchInferenceRPUConfig (SoftBounds, INFERENCE ONLY)")
print("  lora_A/B: SingleRPUConfig (6T1C, TRAINABLE)")

base_config = gen_softbounds_base_layer_config()  # ← TorchInferenceRPUConfig
lora_config = gen_sixt1c_lora_config_trainable()  # ← SingleRPUConfig

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

print("✓ Model converted")

# Check config types
print("\n[Checking config types]")
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        tile = module.analog_module.tile if hasattr(module, 'analog_module') else module.tile

        if 'base_layer' in name:
            layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
            print(f"\nbase_layer L{layer_idx}:")
            print(f"  Tile type: {type(tile).__name__}")

            # Check if it's TorchInferenceTile
            is_inference = 'Inference' in type(tile).__name__
            print(f"  Is InferenceTile? {is_inference}")

            # Check rpu_config
            if hasattr(tile, 'rpu_config'):
                print(f"  RPU config type: {type(tile.rpu_config).__name__}")

# Optimizer
optimizer = AnalogSGD(model.parameters(), lr=0.001)
optimizer.regroup_param_groups(model)
print("\n✓ Optimizer created")

# Collect modules
base_layers = []
lora_layers = []

for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        if 'base_layer' in name:
            base_layers.append((name, module))
        elif 'lora' in name:
            lora_layers.append((name, module))

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
print("TRAINING: 3 steps")
print("=" * 80)

for step in range(3):
    print(f"\n--- Step {step + 1} ---")

    # Capture weights before
    weights_before = {}
    for name, module in base_layers:
        w = module.get_weights()
        weights_before[name] = (w[0] if isinstance(w, tuple) else w).clone()

    # Forward + Backward + Step
    model.train()
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")

    # Check weight changes
    print(f"\n  base_layer weight changes:")
    for name, module in base_layers:
        layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'

        w = module.get_weights()
        w_after = w[0] if isinstance(w, tuple) else w

        delta = (w_after - weights_before[name]).abs()

        print(f"    Layer {layer_idx}: max={delta.max():.6f}, mean={delta.mean():.6f}, changed={((delta > 1e-8).sum().item())}/{delta.numel()}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Check if any base_layer changed
any_changed = False
for name, module in base_layers:
    w = module.get_weights()
    w_after = w[0] if isinstance(w, tuple) else w

    if (w_after - weights_before[name]).abs().max() > 1e-8:
        any_changed = True
        layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
        print(f"✗ Layer {layer_idx} base_layer CHANGED")

if not any_changed:
    print("✓ ALL base_layer weights UNCHANGED!")
    print("  → TorchInferenceRPUConfig successfully prevents updates!")
else:
    print("✗ Some base_layer weights changed")
    print("  → TorchInferenceRPUConfig did NOT prevent updates")

print("=" * 80)
