#!/usr/bin/env python
"""Debug: Understand LRTT tile structure."""

import sys
sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from transformers import AutoModelForSequenceClassification, AutoConfig
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

# Create model
config = AutoConfig.from_pretrained("google/mobilebert-uncased", num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", config=config)

# Convert
lrtt_config = create_lrtt_lora_config(rank=8, lora_alpha=1.0, use_floating_point=False)
model = convert_model_to_lrtt_lora(model, lrtt_config, ["query"])

# Find first query layer
for name, module in model.named_modules():
    if "query" in name and isinstance(module, AnalogLinear):
        print(f"Found layer: {name}")

        # Get tile
        tile = module.analog_module
        if not isinstance(tile, LRTTSimulatorTile):
            from aihwkit.simulator.tiles.array import TileModuleArray
            if isinstance(tile, TileModuleArray):
                tile = tile.array[0][0]

        print(f"\nTile type: {type(tile)}")
        print(f"Tile attributes: {[attr for attr in dir(tile) if not attr.startswith('_')][:20]}")

        # Check key attributes
        if hasattr(tile, 'lrtt_config'):
            print(f"\ntile.lrtt_config type: {type(tile.lrtt_config)}")
            print(f"tile.lrtt_config attributes: {[attr for attr in dir(tile.lrtt_config) if not attr.startswith('_')][:30]}")

            # Check if it has device
            if hasattr(tile.lrtt_config, 'device'):
                print(f"\ntile.lrtt_config.device type: {type(tile.lrtt_config.device)}")

            # Check common attributes
            if hasattr(tile.lrtt_config, 'rank'):
                print(f"\ntile.lrtt_config.rank: {tile.lrtt_config.rank}")
            if hasattr(tile.lrtt_config, 'lora_alpha'):
                print(f"tile.lrtt_config.lora_alpha: {tile.lrtt_config.lora_alpha}")
            if hasattr(tile.lrtt_config, 'forward_inject'):
                print(f"tile.lrtt_config.forward_inject: {tile.lrtt_config.forward_inject}")

        break
