#!/usr/bin/env python
"""Debug: Track device movement step-by-step"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from transformers import AutoModelForSequenceClassification
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.nn import AnalogLinear

MODEL_NAME = "google/mobilebert-uncased"

print("=" * 80)
print("DEBUGGING DEVICE MOVEMENT")
print("=" * 80)

# Load and convert model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
lrtt_config = create_lrtt_lora_config(rank=8, lora_alpha=1.0, use_floating_point=False)
model = convert_model_to_lrtt_lora(model, lrtt_config, ["query"])

# Find first LRTT layer
from aihwkit.simulator.tiles.array import TileModuleArray

for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        print(f"\nTesting device movement for: {name}")

        # Get the tile
        if isinstance(module.analog_module, TileModuleArray):
            tile = module.analog_module.array[0][0]
        else:
            tile = module.analog_module

        # Check initial state
        print("\n[BEFORE] Initial state (CPU):")
        # tile.device is a @property, should work
        try:
            print(f"  tile.device: {tile.device}")
        except Exception as e:
            print(f"  tile.device: Error - {e}")
        weights_a, _ = tile.tile_a.get_weights()
        weights_b, _ = tile.tile_b.get_weights()
        weights_c, _ = tile.tile_c.get_weights()
        print(f"  tile_a weights device: {weights_a.device}")
        print(f"  tile_b weights device: {weights_b.device}")
        print(f"  tile_c weights device: {weights_c.device}")

        # Try moving just the tile (not the whole model)
        print("\n[ACTION] Calling tile.to('cuda')...")
        tile.to('cuda')

        # Check state after moving
        print("\n[AFTER] After tile.to('cuda'):")
        try:
            print(f"  tile.device: {tile.device}")
        except Exception as e:
            print(f"  tile.device: Error - {e}")
        weights_a, _ = tile.tile_a.get_weights()
        weights_b, _ = tile.tile_b.get_weights()
        weights_c, _ = tile.tile_c.get_weights()
        print(f"  tile_a weights device: {weights_a.device}")
        print(f"  tile_b weights device: {weights_b.device}")
        print(f"  tile_c weights device: {weights_c.device}")

        # Check if sub-tiles themselves report correct device
        print("\n[DEBUG] Sub-tile device properties:")
        print(f"  tile_a.device: {tile.tile_a.device if hasattr(tile.tile_a, 'device') else 'N/A'}")
        print(f"  tile_b.device: {tile.tile_b.device if hasattr(tile.tile_b, 'device') else 'N/A'}")
        print(f"  tile_c.device: {tile.tile_c.device if hasattr(tile.tile_c, 'device') else 'N/A'}")

        # Try calling cuda() on sub-tiles directly
        print("\n[ACTION] Trying tile_a.cuda() directly...")
        tile.tile_a.cuda()
        weights_a, _ = tile.tile_a.get_weights()
        print(f"  After tile_a.cuda(): {weights_a.device}")

        print("\n[ACTION] Trying tile_b.cuda() directly...")
        tile.tile_b.cuda()
        weights_b, _ = tile.tile_b.get_weights()
        print(f"  After tile_b.cuda(): {weights_b.device}")

        print("\n[ACTION] Trying tile_c.cuda() directly...")
        tile.tile_c.cuda()
        weights_c, _ = tile.tile_c.get_weights()
        print(f"  After tile_c.cuda(): {weights_c.device}")

        break

print("\n" + "=" * 80)
