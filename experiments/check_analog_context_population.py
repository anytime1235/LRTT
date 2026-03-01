#!/usr/bin/env python3
"""Check if AnalogContext is populated during forward/backward."""

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
    gen_softbounds_base_layer_config
)
from smart_conversion import convert_base_and_lora_separately

print("Creating model...")
model_config = AutoConfig.from_pretrained('google/mobilebert-uncased')
model_config.num_labels = 2
model_config.num_hidden_layers = 1

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

# Convert
base_config = gen_softbounds_base_layer_config()
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

# Find modules
lora_a = None
lora_b = None
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        if 'lora_A' in name and lora_a is None:
            lora_a = module
            lora_a_name = name
        elif 'lora_B' in name and lora_b is None:
            lora_b = module
            lora_b_name = name

print(f"\nTarget modules:")
print(f"  lora_A: {lora_a_name}")
print(f"  lora_B: {lora_b_name}")

# Get analog contexts
ctx_a = lora_a.analog_module.analog_ctx if hasattr(lora_a.analog_module, 'analog_ctx') else None
ctx_b = lora_b.analog_module.analog_ctx if hasattr(lora_b.analog_module, 'analog_ctx') else None

if ctx_a is None:
    print("\n✗ lora_A has no analog_ctx!")
    sys.exit(1)

# Setup optimizer
optimizer = AnalogSGD(model.parameters(), lr=0.001)
optimizer.regroup_param_groups(model)

# Prepare input
tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
inputs = tokenizer(
    ["Test sentence one.", "Test sentence two."],
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors='pt'
)
labels = torch.tensor([1, 0])

print("\n" + "="*80)
print("STEP 1: Check context BEFORE forward")
print("="*80)
print(f"lora_A analog_input: {len(ctx_a.analog_input) if hasattr(ctx_a, 'analog_input') else 'N/A'}")
print(f"lora_A analog_grad_output: {len(ctx_a.analog_grad_output) if hasattr(ctx_a, 'analog_grad_output') else 'N/A'}")

print("\n" + "="*80)
print("STEP 2: Forward pass")
print("="*80)
model.train()
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")

print("\n" + "="*80)
print("STEP 3: Check context AFTER forward (before backward)")
print("="*80)
print(f"lora_A analog_input: {len(ctx_a.analog_input) if hasattr(ctx_a, 'analog_input') else 'N/A'}")
print(f"lora_A analog_grad_output: {len(ctx_a.analog_grad_output) if hasattr(ctx_a, 'analog_grad_output') else 'N/A'}")

if hasattr(ctx_a, 'analog_input') and len(ctx_a.analog_input) > 0:
    inp = ctx_a.analog_input[0]
    print(f"  analog_input[0]: shape={inp.shape}, norm={inp.norm().item():.4e}")

print("\n" + "="*80)
print("STEP 4: Backward pass")
print("="*80)
loss.backward()

print("\n" + "="*80)
print("STEP 5: Check context AFTER backward")
print("="*80)
print(f"lora_A analog_input: {len(ctx_a.analog_input) if hasattr(ctx_a, 'analog_input') else 'N/A'}")
print(f"lora_A analog_grad_output: {len(ctx_a.analog_grad_output) if hasattr(ctx_a, 'analog_grad_output') else 'N/A'}")

if hasattr(ctx_a, 'analog_input') and len(ctx_a.analog_input) > 0:
    inp = ctx_a.analog_input[0]
    print(f"  analog_input[0]: shape={inp.shape}, norm={inp.norm().item():.4e}")

if hasattr(ctx_a, 'analog_grad_output') and len(ctx_a.analog_grad_output) > 0:
    grad = ctx_a.analog_grad_output[0]
    print(f"  analog_grad_output[0]: shape={grad.shape}, norm={grad.norm().item():.4e}")

print("\n" + "="*80)
print("STEP 6: Check has_gradient()")
print("="*80)
has_grad = ctx_a.has_gradient() if hasattr(ctx_a, 'has_gradient') else False
print(f"ctx_a.has_gradient(): {has_grad}")

if has_grad:
    print("  ✓ AnalogSGD will call tile.update()")
else:
    print("  ✗ AnalogSGD will SKIP tile.update() - THIS IS THE PROBLEM!")
    print("\nDiagnosis:")
    if not hasattr(ctx_a, 'analog_input') or len(ctx_a.analog_input) == 0:
        print("  → analog_input is empty - forward pass not caching inputs!")
    if not hasattr(ctx_a, 'analog_grad_output') or len(ctx_a.analog_grad_output) == 0:
        print("  → analog_grad_output is empty - backward pass not caching gradients!")

print("="*80)
