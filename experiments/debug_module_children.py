#!/usr/bin/env python
"""Debug: Check if sub-tiles are registered as child modules"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from transformers import AutoModelForSequenceClassification
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.nn import AnalogLinear

MODEL_NAME = "google/mobilebert-uncased"

print("=" * 80)
print("DEBUGGING MODULE HIERARCHY")
print("=" * 80)

# Load and convert model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
lrtt_config = create_lrtt_lora_config(rank=8, lora_alpha=1.0, use_floating_point=False)
model = convert_model_to_lrtt_lora(model, lrtt_config, ["query"])

# Find first LRTT layer
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        print(f"\nFound LRTT layer: {name}")
        print(f"Module type: {type(module)}")

        # Get the tile
        from aihwkit.simulator.tiles.array import TileModuleArray
        if isinstance(module.analog_module, TileModuleArray):
            tile = module.analog_module.array[0][0]
        else:
            tile = module.analog_module

        print(f"\nTile type: {type(tile)}")
        print(f"Tile has controller: {hasattr(tile, 'controller')}")

        # Check if sub-tiles are attributes of the tile
        print(f"\nTile has tile_a attr: {hasattr(tile, 'tile_a')}")
        print(f"Tile has tile_b attr: {hasattr(tile, 'tile_b')}")
        print(f"Tile has tile_c attr: {hasattr(tile, 'tile_c')}")

        # Check if sub-tiles are registered as child modules
        print("\nChild modules of tile:")
        for child_name, child_module in tile.named_children():
            print(f"  - {child_name}: {type(child_module)}")

        # Check if tile_a/b/c are in the children
        has_tile_a = any('tile_a' in name for name, _ in tile.named_children())
        has_tile_b = any('tile_b' in name for name, _ in tile.named_children())
        has_tile_c = any('tile_c' in name for name, _ in tile.named_children())

        print(f"\ntile_a in children: {has_tile_a}")
        print(f"tile_b in children: {has_tile_b}")
        print(f"tile_c in children: {has_tile_c}")

        if not (has_tile_a and has_tile_b and has_tile_c):
            print("\n⚠️  WARNING: Sub-tiles are NOT registered as child modules!")
            print("This means .to(device) won't automatically move them.")

        break

print("=" * 80)
