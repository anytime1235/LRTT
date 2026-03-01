#!/usr/bin/env python
"""Debug script to check 6T1C weight initialization."""

import sys
import torch
sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from transformers import AutoModelForSequenceClassification, AutoConfig
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

def check_model_weights(model_name="google/mobilebert-uncased", use_fp=False):
    """Check if weights are initialized correctly."""
    print(f"\n{'='*80}")
    print(f"CHECKING WEIGHT INITIALIZATION (FP={use_fp})")
    print('='*80)

    # Create model
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model_orig = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Get original query weight
    orig_query_weight = model_orig.mobilebert.encoder.layer[0].attention.self.query.weight.data.clone()
    print(f"\n[Original Query Weight (layer 0)]")
    print(f"  Shape: {orig_query_weight.shape}")
    print(f"  Mean: {orig_query_weight.mean():.6f}")
    print(f"  Std: {orig_query_weight.std():.6f}")
    print(f"  Min: {orig_query_weight.min():.6f}")
    print(f"  Max: {orig_query_weight.max():.6f}")

    # Convert to LRTT-LoRA
    lrtt_config = create_lrtt_lora_config(
        rank=8, lora_alpha=1.0, use_floating_point=use_fp
    )
    model_lora = convert_model_to_lrtt_lora(model_orig, lrtt_config, ["query"])

    # Get LRTT tile
    lora_layer = model_lora.mobilebert.encoder.layer[0].attention.self.query
    tile = lora_layer.analog_module
    if not isinstance(tile, LRTTSimulatorTile):
        from aihwkit.simulator.tiles.array import TileModuleArray
        if isinstance(tile, TileModuleArray):
            tile = tile.array[0][0]

    # Check A/B/C weights
    weights_a, _ = tile.tile_a.get_weights()
    weights_b, _ = tile.tile_b.get_weights()
    weights_c, _ = tile.tile_c.get_weights()

    print(f"\n[LRTT Tile Weights]")
    print(f"\ntile_a (should be ~0 initially):")
    print(f"  Shape: {weights_a.shape}")
    print(f"  Mean: {weights_a.mean():.6f}")
    print(f"  Std: {weights_a.std():.6f}")
    print(f"  Max abs: {weights_a.abs().max():.6f}")

    print(f"\ntile_b (random Kaiming init):")
    print(f"  Shape: {weights_b.shape}")
    print(f"  Mean: {weights_b.mean():.6f}")
    print(f"  Std: {weights_b.std():.6f}")
    print(f"  Max abs: {weights_b.abs().max():.6f}")

    print(f"\ntile_c (pretrained, should match original):")
    print(f"  Shape: {weights_c.shape}")
    print(f"  Mean: {weights_c.mean():.6f}")
    print(f"  Std: {weights_c.std():.6f}")
    print(f"  Matches original: {torch.allclose(weights_c, orig_query_weight, atol=1e-6)}")
    print(f"  Max difference: {(weights_c - orig_query_weight).abs().max():.6e}")

    # Test forward pass
    print(f"\n[Forward Pass Test]")
    x = torch.randn(1, 128, 512)  # [batch, seq_len, hidden_size]

    model_orig.eval()
    model_lora.eval()

    with torch.no_grad():
        # Original model
        out_orig = model_orig.mobilebert.encoder.layer[0].attention.self.query(x)

        # LRTT-LoRA model
        out_lora = lora_layer(x)

    print(f"  Original output mean: {out_orig.mean():.6f}")
    print(f"  LRTT-LoRA output mean: {out_lora.mean():.6f}")
    print(f"  Difference: {(out_orig - out_lora).abs().max():.6e}")
    print(f"  Should be ~0 if A≈0: {(out_orig - out_lora).abs().max() < 1e-3}")

    print('='*80)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("WEIGHT INITIALIZATION DEBUG")
    print("="*80)

    # Test FP-LoRA
    check_model_weights(use_fp=True)

    # Test 6T1C-LoRA
    check_model_weights(use_fp=False)
