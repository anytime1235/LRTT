#!/usr/bin/env python
"""Debug: Check weights after model conversion."""

import sys
import torch
sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

def check_conversion(use_fp=False):
    """Check weights after conversion."""
    print(f"\n{'='*80}")
    print(f"CHECKING CONVERSION ({'FP' if use_fp else '6T1C'})")
    print('='*80)

    # Load pretrained model
    config = AutoConfig.from_pretrained("google/mobilebert-uncased", num_labels=2)
    model_orig = AutoModelForSequenceClassification.from_pretrained("google/mobilebert-uncased", config=config)

    # Get original query weight (layer 0)
    orig_query = model_orig.mobilebert.encoder.layer[0].attention.self.query
    orig_weight = orig_query.weight.data.clone()

    print(f"\n[Before Conversion]")
    print(f"  Original query weight shape: {orig_weight.shape}")
    print(f"  Original query weight mean: {orig_weight.mean():.6f}")

    # Convert to LRTT-LoRA
    lrtt_config = create_lrtt_lora_config(rank=8, lora_alpha=1.0, use_floating_point=use_fp)
    model_lora = convert_model_to_lrtt_lora(model_orig, lrtt_config, ["query"])

    # Get converted query layer (layer 0)
    lora_query = model_lora.mobilebert.encoder.layer[0].attention.self.query

    # Get tile
    tile = lora_query.analog_module
    if not isinstance(tile, LRTTSimulatorTile):
        from aihwkit.simulator.tiles.array import TileModuleArray
        if isinstance(tile, TileModuleArray):
            tile = tile.array[0][0]

    # Check weights
    weights_a, _ = tile.tile_a.get_weights()
    weights_b, _ = tile.tile_b.get_weights()
    weights_c, _ = tile.tile_c.get_weights()

    print(f"\n[After Conversion]")
    print(f"\ntile_a:")
    print(f"  Shape: {weights_a.shape}")
    print(f"  Mean: {weights_a.mean():.10f}")
    print(f"  Std: {weights_a.std():.10f}")
    print(f"  Max abs: {weights_a.abs().max():.10f}")
    print(f"  Is all zeros: {torch.allclose(weights_a, torch.zeros_like(weights_a), atol=1e-10)}")

    print(f"\ntile_b:")
    print(f"  Shape: {weights_b.shape}")
    print(f"  Mean: {weights_b.mean():.6f}")
    print(f"  Std: {weights_b.std():.6f}")

    print(f"\ntile_c:")
    print(f"  Shape: {weights_c.shape}")
    print(f"  Mean: {weights_c.mean():.6f}")
    print(f"  Matches original: {torch.allclose(weights_c, orig_weight, atol=1e-6)}")
    print(f"  Max diff: {(weights_c - orig_weight).abs().max():.10f}")

    # Test forward with simple input
    print(f"\n[Forward Pass Test]")
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
    text = "This is a test sentence."
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)

    model_orig.eval()
    model_lora.eval()

    with torch.no_grad():
        output_orig = model_orig(**inputs)
        output_lora = model_lora(**inputs)

    logits_orig = output_orig.logits
    logits_lora = output_lora.logits

    print(f"  Original logits: {logits_orig}")
    print(f"  LoRA logits: {logits_lora}")
    print(f"  Difference: {(logits_orig - logits_lora).abs().max():.6f}")

    # Since A=0, ΔW=0, so outputs should be identical
    if (logits_orig - logits_lora).abs().max() < 1e-3:
        print(f"  ✓ Outputs match (as expected, A=0 so ΔW=0)")
    else:
        print(f"  ⚠️  Outputs differ! Something is wrong.")

    return weights_a, weights_b, weights_c

if __name__ == "__main__":
    print("="*80)
    print("MODEL CONVERSION WEIGHT DEBUG")
    print("="*80)

    # Test both FP and 6T1C
    print("\n" + "="*80)
    print("1. FP-LoRA CONVERSION")
    print("="*80)
    wa_fp, wb_fp, wc_fp = check_conversion(use_fp=True)

    print("\n" + "="*80)
    print("2. 6T1C-LoRA CONVERSION")
    print("="*80)
    wa_6t1c, wb_6t1c, wc_6t1c = check_conversion(use_fp=False)

    # Compare A tiles (should both be 0)
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\nA tiles equal: {torch.allclose(wa_fp, wa_6t1c, atol=1e-10)}")
    print(f"B tiles equal: {torch.allclose(wb_fp, wb_6t1c, atol=1e-6)}")
    print(f"C tiles equal: {torch.allclose(wc_fp, wc_6t1c, atol=1e-6)}")
