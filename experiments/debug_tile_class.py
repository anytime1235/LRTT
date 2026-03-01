#!/usr/bin/env python
"""Debug: Check tile class for CUDA mapping"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from transformers import AutoModelForSequenceClassification
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.nn import AnalogLinear

MODEL_NAME = "google/mobilebert-uncased"

print("=" * 80)
print("DEBUGGING TILE CLASS AND CUDA MAPPING")
print("=" * 80)

# Load and convert model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
lrtt_config = create_lrtt_lora_config(rank=8, lora_alpha=1.0, use_floating_point=False)
model = convert_model_to_lrtt_lora(model, lrtt_config, ["query"])

# Find first LRTT layer
from aihwkit.simulator.tiles.array import TileModuleArray

for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        print(f"\nTesting: {name}")

        # Get the tile
        if isinstance(module.analog_module, TileModuleArray):
            tile = module.analog_module.array[0][0]
        else:
            tile = module.analog_module

        print(f"\nLRTTSimulatorTile type: {type(tile)}")
        print(f"tile has 'tile' attribute (C++ backend): {hasattr(tile, 'tile')}")

        # Check sub-tiles
        print(f"\ntile_a type: {type(tile.tile_a)}")
        print(f"tile_a has 'tile' attribute: {hasattr(tile.tile_a, 'tile')}")

        if hasattr(tile.tile_a, 'tile'):
            print(f"tile_a.tile type: {type(tile.tile_a.tile)}")
            print(f"tile_a.tile.__class__: {tile.tile_a.tile.__class__}")
            print(f"tile_a.tile.__class__.__name__: {tile.tile_a.tile.__class__.__name__}")

            # Check if in mapping
            from aihwkit.simulator.tiles.rpucuda import MAP_TILE_CLASS_TO_CUDA
            in_map = tile.tile_a.tile.__class__ in MAP_TILE_CLASS_TO_CUDA
            print(f"tile_a.tile.__class__ in MAP_TILE_CLASS_TO_CUDA: {in_map}")

            print(f"\nMAP_TILE_CLASS_TO_CUDA contents:")
            for key, value in MAP_TILE_CLASS_TO_CUDA.items():
                print(f"  {key} -> {value}")

        # Check is_cuda flag
        print(f"\ntile_a.is_cuda: {tile.tile_a.is_cuda if hasattr(tile.tile_a, 'is_cuda') else 'N/A'}")
        print(f"tile_a.device: {tile.tile_a.device if hasattr(tile.tile_a, 'device') else 'N/A'}")

        break

print("=" * 80)
