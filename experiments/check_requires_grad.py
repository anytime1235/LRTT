#!/usr/bin/env python
"""Check requires_grad settings for LRTT layers"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from transformers import AutoModelForSequenceClassification
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

MODEL_NAME = "google/mobilebert-uncased"

print("Checking requires_grad settings...")

# Load and convert model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
lrtt_config = create_lrtt_lora_config(rank=8, lora_alpha=1.0, use_floating_point=False)
model = convert_model_to_lrtt_lora(model, lrtt_config, ["query", "key", "value"])

print("\n" + "=" * 80)
print("CHECKING LRTT ANALOG LAYERS")
print("=" * 80)

from aihwkit.nn import AnalogLinear

for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        print(f"\n[{name}]")
        print(f"  Type: {type(module)}")
        print(f"  weight.requires_grad: {module.weight.requires_grad}")
        if module.bias is not None:
            print(f"  bias.requires_grad: {module.bias.requires_grad}")

        # Check if we can access tile
        if hasattr(module, 'analog_module'):
            tile = module.analog_module
            if hasattr(tile, 'array'):
                tile = tile.array[0][0]
            print(f"  Tile type: {type(tile)}")
            print(f"  forward_inject: {tile.controller.forward_inject_enabled if hasattr(tile, 'controller') else 'N/A'}")

            # Check if tile weights have grad
            if hasattr(tile, 'tile_a'):
                weights_a, _ = tile.tile_a.get_weights()
                print(f"  tile_a weights shape: {weights_a.shape}")
                print(f"  tile_a weights requires_grad: {weights_a.requires_grad}")
                print(f"  tile_a weights grad: {weights_a.grad is not None if weights_a.requires_grad else 'N/A'}")
        break  # Just check one layer

print("\n" + "=" * 80)
print("CHECKING DIGITAL LAYERS (classifier)")
print("=" * 80)

for name, module in model.named_modules():
    if 'classifier' in name and hasattr(module, 'weight'):
        print(f"\n[{name}]")
        print(f"  weight.requires_grad: {module.weight.requires_grad}")
        if module.bias is not None:
            print(f"  bias.requires_grad: {module.bias.requires_grad}")
        break

print("\n" + "=" * 80)
print("SUMMARY OF ALL PARAMETERS")
print("=" * 80)

trainable = []
frozen = []

for name, param in model.named_parameters():
    if param.requires_grad:
        trainable.append((name, param.numel()))
    else:
        frozen.append((name, param.numel()))

print(f"\n Trainable parameters: {len(trainable)}")
for name, count in trainable[:10]:
    print(f"  {name}: {count:,}")
if len(trainable) > 10:
    print(f"  ... and {len(trainable) - 10} more")

print(f"\nFrozen parameters: {len(frozen)}")
print(f"  Total: {sum(c for _, c in frozen):,}")

total_trainable = sum(c for _, c in trainable)
total_frozen = sum(c for _, c in frozen)
print(f"\nTotal trainable: {total_trainable:,}")
print(f"Total frozen: {total_frozen:,}")
print(f"Trainable fraction: {total_trainable / (total_trainable + total_frozen):.2%}")
