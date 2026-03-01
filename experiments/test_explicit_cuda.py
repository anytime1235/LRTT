#!/usr/bin/env python
"""Test explicit .cuda() call on tiles"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from transformers import AutoModelForSequenceClassification
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.nn import AnalogLinear

MODEL_NAME = "google/mobilebert-uncased"

print("=" * 80)
print("TESTING EXPLICIT CUDA CALL")
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

        print("\nBefore:")
        print(f"  tile_a.is_cuda: {tile.tile_a.is_cuda}")
        weights_a, _ = tile.tile_a.get_weights()
        print(f"  tile_a weights device: {weights_a.device}")

        # Try calling cuda() directly on tile_a
        print("\nCalling tile.tile_a.cuda()...")
        result = tile.tile_a.cuda()
        print(f"  Returned: {result}")

        print("\nAfter tile_a.cuda():")
        print(f"  tile_a.is_cuda: {tile.tile_a.is_cuda}")
        weights_a, _ = tile.tile_a.get_weights()
        print(f"  tile_a weights device: {weights_a.device}")

        # Try calling cuda() on the LRTTSimulatorTile
        print("\nCalling tile.cuda()...")
        result = tile.cuda()
        print(f"  Returned: {result}")

        print("\nAfter tile.cuda():")
        print(f"  tile_a.is_cuda: {tile.tile_a.is_cuda}")
        weights_a, _ = tile.tile_a.get_weights()
        print(f"  tile_a weights device: {weights_a.device}")

        break

print("=" * 80)
