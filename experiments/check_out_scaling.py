#!/usr/bin/env python
"""Check out_scaling initialization."""

import sys
import torch
sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from transformers import AutoModelForSequenceClassification, AutoConfig
from lrtt_lora_config import create_lrtt_lora_config
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

def check_out_scaling():
    """Check out_scaling parameter initialization."""
    print("="*80)
    print("OUT_SCALING INITIALIZATION CHECK")
    print("="*80)

    # Test FP vs 6T1C
    for use_fp in [True, False]:
        print(f"\n{'='*80}")
        print(f"Testing {'FP-LoRA' if use_fp else '6T1C-LoRA'}")
        print('='*80)

        config = create_lrtt_lora_config(rank=8, lora_alpha=1.0, use_floating_point=use_fp)

        analog_layer = AnalogLinear(
            in_features=128,
            out_features=128,
            bias=False,
            rpu_config=config,
        )

        # Get tile
        tile = analog_layer.analog_module
        if not isinstance(tile, LRTTSimulatorTile):
            from aihwkit.simulator.tiles.array import TileModuleArray
            if isinstance(tile, TileModuleArray):
                tile = tile.array[0][0]

        # Check out_scaling
        if hasattr(tile, 'out_scaling') and tile.out_scaling is not None:
            print(f"\nout_scaling parameter:")
            print(f"  Shape: {tile.out_scaling.shape}")
            print(f"  Values: {tile.out_scaling.data}")
            print(f"  Mean: {tile.out_scaling.mean():.6f}")
            print(f"  Requires grad: {tile.out_scaling.requires_grad}")
        else:
            print(f"\nout_scaling: None")

        # Test forward pass with known input
        print(f"\nForward pass test:")
        x = torch.ones(1, 128) * 0.1  # Small constant input

        # Set all weights to known values
        tile.tile_a.set_weights(torch.zeros(128, 8))  # A=0
        tile.tile_b.set_weights(torch.ones(8, 128) * 0.01)  # B=small
        tile.tile_c.set_weights(torch.eye(128) * 0.5)  # C=identity*0.5

        with torch.no_grad():
            output = analog_layer(x)

        print(f"  Input: {x[0, :5]}")
        print(f"  Output (first 5): {output[0, :5]}")
        print(f"  Expected (C·x, since A=0): ~{(x[0] * 0.5)[:5]}")
        print(f"  Output mean: {output.mean():.6f}")
        print(f"  Expected mean: {(x.mean() * 0.5):.6f}")

        # Check if output matches expectation
        expected = x * 0.5  # Since A=0, output should be C·x = 0.5*x
        diff = (output - expected).abs().max().item()
        print(f"  Max difference from expected: {diff:.6e}")

        if diff > 1.0:
            print(f"  ⚠️  HUGE DIFFERENCE! Something is wrong.")
        elif diff > 0.01:
            print(f"  ⚠️  Moderate difference (quantization/scaling)")
        else:
            print(f"  ✓ Output matches expectation")

if __name__ == "__main__":
    check_out_scaling()
