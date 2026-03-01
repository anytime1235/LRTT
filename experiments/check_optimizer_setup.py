#!/usr/bin/env python3
"""Check if AnalogSGD optimizer is properly configured."""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

from transformers import AutoConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

from sixt1c_config import (
    gen_sixt1c_lora_config_trainable,
    gen_softbounds_base_layer_config
)
from smart_conversion import convert_base_and_lora_separately

# Create model
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

print("=" * 80)
print("CHECKING OPTIMIZER SETUP")
print("=" * 80)

# Create optimizer
optimizer = AnalogSGD(model.parameters(), lr=0.001)

print("\n[BEFORE regroup_param_groups]")
print(f"Number of param_groups: {len(optimizer.param_groups)}")
for i, group in enumerate(optimizer.param_groups):
    params = group['params']
    print(f"  Group {i}: {len(params)} parameters")
    for p in params[:3]:  # Show first 3
        print(f"    - {p.shape}: {type(p).__name__}")

# Regroup
print("\n[Calling regroup_param_groups(model)]")
optimizer.regroup_param_groups(model)

print("\n[AFTER regroup_param_groups]")
print(f"Number of param_groups: {len(optimizer.param_groups)}")
for i, group in enumerate(optimizer.param_groups):
    params = group['params']
    print(f"\n  Group {i}: {len(params)} parameters, lr={group['lr']}")
    for p in params:
        print(f"    - {p.shape}: {type(p).__name__}, requires_grad={p.requires_grad}")

# Check analog tiles
print("\n" + "=" * 80)
print("ANALOG TILES IN OPTIMIZER")
print("=" * 80)

# AnalogSGD should have analog_tiles attribute
if hasattr(optimizer, 'analog_tiles'):
    analog_tiles = optimizer.analog_tiles
    print(f"\nNumber of analog_tiles: {len(analog_tiles) if analog_tiles else 0}")

    if analog_tiles:
        for i, tile_group in enumerate(analog_tiles[:3]):  # Show first 3
            print(f"\n  Tile group {i}:")
            print(f"    Type: {type(tile_group)}")
            if hasattr(tile_group, 'analog_tile'):
                print(f"    Has analog_tile: True")
                tile = tile_group.analog_tile
                print(f"      Tile type: {type(tile).__name__}")
            if hasattr(tile_group, 'learning_rate'):
                print(f"    Learning rate: {tile_group.learning_rate}")
else:
    print("\n✗ Optimizer has no 'analog_tiles' attribute")

# Check if specific lora_A tile is in optimizer
print("\n" + "=" * 80)
print("CHECKING IF LORA_A TILE IS REGISTERED")
print("=" * 80)

lora_a_module = None
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'lora_A' in name and 'query' in name:
        lora_a_module = module
        lora_a_name = name
        break

if lora_a_module:
    print(f"\nFound lora_A: {lora_a_name}")

    # Check if its AnalogContext is in optimizer
    if hasattr(lora_a_module.analog_module, 'analog_ctx'):
        ctx = lora_a_module.analog_module.analog_ctx
        print(f"  AnalogContext: {ctx}")
        print(f"  requires_grad: {ctx.requires_grad}")

        # Check if this context is in any param group
        ctx_found = False
        for i, group in enumerate(optimizer.param_groups):
            for p in group['params']:
                if p is ctx:
                    ctx_found = True
                    print(f"  ✓ Found in param_group {i}")
                    break

        if not ctx_found:
            print(f"  ✗ NOT found in any param_group!")
            print(f"     → Optimizer will NOT update this tile!")

print("\n" + "=" * 80)
