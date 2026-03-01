#!/usr/bin/env python3
"""Check if the analog tile has the learning rate set."""

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

# Find lora_A
lora_a = None
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'lora_A' in name and 'query' in name:
        lora_a = module
        lora_a_name = name
        break

print("=" * 80)
print("CHECKING TILE LEARNING RATE")
print("=" * 80)

print(f"\nModule: {lora_a_name}")

# Get tile
tile = lora_a.analog_module.tile

# Check if tile has get_learning_rate method
if hasattr(tile, 'get_learning_rate'):
    lr = tile.get_learning_rate()
    print(f"\nTile learning rate (BEFORE optimizer setup): {lr}")

# Create optimizer
optimizer = AnalogSGD(model.parameters(), lr=0.001)
optimizer.regroup_param_groups(model)

print(f"\nOptimizer learning rate: {optimizer.param_groups[0]['lr']}")

# Check tile LR after optimizer setup
if hasattr(tile, 'get_learning_rate'):
    lr = tile.get_learning_rate()
    print(f"Tile learning rate (AFTER optimizer setup): {lr}")

    if lr == 0.0:
        print(f"  ✗ Learning rate is 0! Tile will not update!")
    elif lr > 0:
        print(f"  ✓ Learning rate is set correctly")

# Check if tile has set_learning_rate method
if hasattr(tile, 'set_learning_rate'):
    print(f"\n[Testing: Manually set tile LR to 0.001]")
    tile.set_learning_rate(0.001)
    new_lr = tile.get_learning_rate()
    print(f"Tile learning rate after manual set: {new_lr}")

# Check tile meta parameters
print(f"\n{'='*80}")
print("TILE META PARAMETERS")
print(f"{'='*80}")

if hasattr(tile, 'get_meta_parameters'):
    meta = tile.get_meta_parameters()
    print(f"\nMeta parameters:")
    for key, value in meta.items():
        print(f"  {key}: {value}")

print("\n" + "=" * 80)
