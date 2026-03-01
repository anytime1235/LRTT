#!/usr/bin/env python3
"""Debug: Check if tile.update() is being called for lora_A/B."""

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

# Find lora_A module and patch tile.update()
lora_a_module = None
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'lora_A' in name and 'query' in name:
        lora_a_module = module
        lora_a_name = name
        break

print(f"Target module: {lora_a_name}")

# Patch tile.update() to log when it's called
if hasattr(lora_a_module, 'analog_module'):
    tile = lora_a_module.analog_module.tile
    original_update = tile.update

    update_called = [False]

    def logged_update(x_input, d_input, bias=False, x_trans=False, d_trans=False):
        update_called[0] = True
        print(f"  → tile.update() CALLED!")
        print(f"      x_input shape: {x_input.shape}, norm: {x_input.norm().item():.4e}")
        print(f"      d_input shape: {d_input.shape}, norm: {d_input.norm().item():.4e}")
        result = original_update(x_input, d_input, bias, x_trans, d_trans)
        return result

    tile.update = logged_update

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

# Training step
print("\n" + "="*80)
print("TRAINING STEP")
print("="*80)

# Forward + backward
model.train()
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
print(f"Loss: {loss.item():.4f}")

loss.backward()

# Check AnalogContext
if hasattr(lora_a_module, 'analog_module'):
    if hasattr(lora_a_module.analog_module, 'analog_ctx'):
        ctx = lora_a_module.analog_module.analog_ctx
        print(f"\nAnalogContext status:")
        print(f"  analog_input cached: {len(ctx.analog_input) if hasattr(ctx, 'analog_input') else 'N/A'}")
        print(f"  analog_grad_output cached: {len(ctx.analog_grad_output) if hasattr(ctx, 'analog_grad_output') else 'N/A'}")

        if hasattr(ctx, 'analog_input') and len(ctx.analog_input) > 0:
            print(f"  → analog_input exists, optimizer should call tile.update()")
        else:
            print(f"  → analog_input empty, optimizer will skip tile.update()")

# Optimizer step
print(f"\nCalling optimizer.step()...")
print(f"  Watching for tile.update() calls...")
optimizer.step()

# Check if update was called
print(f"\n" + "="*80)
if update_called[0]:
    print("✓ tile.update() WAS CALLED by optimizer")
    print("→ Analog tile mechanism is working")
else:
    print("✗ tile.update() WAS NOT CALLED by optimizer")
    print("→ Analog tile mechanism is NOT working!")
    print("\nPossible reasons:")
    print("  - AnalogContext not populated (analog_input empty)")
    print("  - Optimizer doesn't recognize the tile")
    print("  - AnalogSGD configuration issue")
print("="*80)
