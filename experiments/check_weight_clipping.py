#!/usr/bin/env python
"""Check if weights are being clipped in 6T1C device."""

import sys
import torch
sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from transformers import AutoModelForSequenceClassification, AutoConfig
from lrtt_lora_config import create_lrtt_lora_config
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

def check_clipping():
    """Check if pretrained weights are clipped."""
    print("="*80)
    print("WEIGHT CLIPPING CHECK")
    print("="*80)

    # Load pretrained model
    config = AutoConfig.from_pretrained("google/mobilebert-uncased", num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", config=config)

    # Get first query layer weight
    query_layer = model.mobilebert.encoder.layer[0].attention.self.query
    orig_weight = query_layer.weight.data.clone()

    print(f"\n[Original Pretrained Weight]")
    print(f"  Shape: {orig_weight.shape}")
    print(f"  Range: [{orig_weight.min():.4f}, {orig_weight.max():.4f}]")
    print(f"  Values > 1.0: {(orig_weight > 1.0).sum().item()}")
    print(f"  Values < -1.0: {(orig_weight < -1.0).sum().item()}")
    print(f"  Total elements: {orig_weight.numel()}")
    print(f"  % outside [-1, 1]: {((orig_weight.abs() > 1.0).sum().item() / orig_weight.numel() * 100):.2f}%")

    # Test with 6T1C device
    print(f"\n[Testing 6T1C Device]")
    config_6t1c = create_lrtt_lora_config(rank=8, lora_alpha=1.0, use_floating_point=False)

    analog_layer = AnalogLinear(
        in_features=128,
        out_features=128,
        bias=False,
        rpu_config=config_6t1c,
    )

    # Get tile
    tile = analog_layer.analog_module
    if not isinstance(tile, LRTTSimulatorTile):
        from aihwkit.simulator.tiles.array import TileModuleArray
        if isinstance(tile, TileModuleArray):
            tile = tile.array[0][0]

    # Set C tile weight (pretrained)
    print(f"\n  Setting tile_c weights...")
    tile.tile_c.set_weights(orig_weight.clone())

    # Read back
    weights_c_read, _ = tile.tile_c.get_weights()

    print(f"\n[Tile C Weight After Set]")
    print(f"  Shape: {weights_c_read.shape}")
    print(f"  Range: [{weights_c_read.min():.4f}, {weights_c_read.max():.4f}]")
    print(f"  Values > 1.0: {(weights_c_read > 1.0).sum().item()}")
    print(f"  Values < -1.0: {(weights_c_read < -1.0).sum().item()}")

    # Check clipping
    was_clipped = not torch.allclose(orig_weight, weights_c_read, atol=1e-6)
    max_diff = (orig_weight - weights_c_read).abs().max().item()

    print(f"\n[Clipping Analysis]")
    print(f"  Was clipped: {was_clipped}")
    print(f"  Max difference: {max_diff:.6f}")

    if was_clipped:
        print(f"\n  ⚠️  WEIGHTS WERE CLIPPED!")
        print(f"  This explains the huge output difference!")
        print(f"\n  Solution: Use weight_scaling to map weights into [-1, 1]")
    else:
        print(f"\n  ✓ No clipping detected")

    # Check device config
    print(f"\n[Device Config]")
    print(f"  w_min: {config_6t1c.device.unit_cell_devices[2].w_min}")
    print(f"  w_max: {config_6t1c.device.unit_cell_devices[2].w_max}")

    print("="*80)

if __name__ == "__main__":
    check_clipping()
