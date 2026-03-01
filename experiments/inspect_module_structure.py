#!/usr/bin/env python3
"""Inspect the actual module structure to understand why tiles aren't recognized."""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear

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

print("=" * 80)
print("MODULE STRUCTURE INSPECTION")
print("=" * 80)

# Find first lora_A module
for name, module in model.named_modules():
    if 'lora_A' in name and 'query' in name and isinstance(module, AnalogLinear):
        print(f"\n[Found AnalogLinear: {name}]")
        print(f"  Type: {type(module)}")
        print(f"  Dir: {[x for x in dir(module) if not x.startswith('_')][:20]}")

        print(f"\n  Has attributes:")
        print(f"    analog_module: {hasattr(module, 'analog_module')}")
        print(f"    tile: {hasattr(module, 'tile')}")

        if hasattr(module, 'analog_module'):
            print(f"\n  analog_module attributes:")
            am = module.analog_module
            print(f"    Type: {type(am)}")
            print(f"    Has tile: {hasattr(am, 'tile')}")

            if hasattr(am, 'tile'):
                tile = am.tile
                print(f"\n  analog_module.tile attributes:")
                print(f"    Type: {type(tile)}")
                print(f"    Has rpu_config: {hasattr(tile, 'rpu_config')}")

                if hasattr(tile, 'rpu_config'):
                    print(f"    rpu_config type: {type(tile.rpu_config)}")
                    print(f"    ✓ This IS an analog tile!")
                else:
                    print(f"    Dir: {[x for x in dir(tile) if not x.startswith('_')]}")

        if hasattr(module, 'tile'):
            tile = module.tile
            print(f"\n  module.tile attributes:")
            print(f"    Type: {type(tile)}")
            print(f"    Has rpu_config: {hasattr(tile, 'rpu_config')}")

        break

print("\n" + "=" * 80)
print("PARAMETER INSPECTION")
print("=" * 80)

for name, param in model.named_parameters():
    if 'lora_A' in name and 'query' in name:
        print(f"\n{name}:")
        print(f"  Type: {type(param)}")
        print(f"  Shape: {param.shape}")
        print(f"  requires_grad: {param.requires_grad}")
        print(f"  Is AnalogContext: {'AnalogContext' in str(type(param))}")

print("=" * 80)
