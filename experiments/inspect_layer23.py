#!/usr/bin/env python
# coding=utf-8
"""
Inspect Layer 23 in Detail

Find out what makes layer 23 different from other layers.
"""

import sys
import torch
import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, set_seed

sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

SEED = 42
MODEL_NAME = "google/mobilebert-uncased"


def inspect_lora_layer(name, module):
    """Inspect LoRA layer parameters in detail."""
    print(f"\n{'='*80}")
    print(f"{name}")
    print('='*80)

    if not hasattr(module, 'analog_module'):
        print("  No analog_module found!")
        return

    analog_module = module.analog_module

    # Check each tile
    for tile_name in ['tile_a', 'tile_b', 'tile_c']:
        if not hasattr(analog_module, tile_name):
            continue

        tile = getattr(analog_module, tile_name)
        print(f"\n{tile_name}:")

        # Get weights
        try:
            weights = tile.get_weights()
            w_np = weights.detach().cpu().numpy()

            print(f"  Weights shape: {weights.shape}")
            print(f"  Weights range: [{w_np.min():.6f}, {w_np.max():.6f}]")
            print(f"  Weights mean: {w_np.mean():.6f}, std: {w_np.std():.6f}")

            # Check for abnormal values
            if np.abs(w_np).max() > 100:
                print(f"  ⚠️  WARNING: Large weight values!")

        except Exception as e:
            print(f"  ✗ Could not get weights: {e}")

        # Check out_scaling_alpha
        if hasattr(tile, 'out_scaling_alpha'):
            alpha = tile.out_scaling_alpha
            alpha_np = alpha.detach().cpu().numpy()

            print(f"  out_scaling_alpha shape: {alpha.shape}")
            print(f"  out_scaling_alpha range: [{alpha_np.min():.6f}, {alpha_np.max():.6f}]")
            print(f"  out_scaling_alpha mean: {alpha_np.mean():.6f}")

            if np.abs(alpha_np).max() > 100:
                print(f"  ⚠️  WARNING: Large scaling values!")
            if np.abs(alpha_np).min() < 0.001:
                print(f"  ⚠️  WARNING: Very small scaling values!")

        # Check mapping_scales
        if hasattr(tile, 'mapping_scales'):
            scales = tile.mapping_scales
            scales_np = scales.detach().cpu().numpy()

            print(f"  mapping_scales shape: {scales.shape}")
            print(f"  mapping_scales range: [{scales_np.min():.6f}, {scales_np.max():.6f}]")
            print(f"  mapping_scales mean: {scales_np.mean():.6f}")

            if np.abs(scales_np).max() > 1000:
                print(f"  ⚠️  WARNING: Large mapping scales!")


def main():
    print("=" * 80)
    print("INSPECT: Layer 23 vs Layer 0 (Comparison)")
    print("=" * 80)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("\nLoading and converting model...")
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )

    lrtt_config = create_lrtt_lora_config(
        rank=8,
        lora_alpha=1.22,
        output_noise_level=0.0,
        use_floating_point=False,
    )

    target_modules = ["query", "key", "value"]
    model = convert_model_to_lrtt_lora(model, lrtt_config, target_modules)
    model.to(device)

    print("\n" + "=" * 80)
    print("COMPARING LAYER 0 (NORMAL) vs LAYER 23 (EXPLOSION)")
    print("=" * 80)

    # Find and compare layers
    layer0_query = None
    layer23_query = None

    for name, module in model.named_modules():
        if "layer.0.attention.self.query" in name and "AnalogLinear" in type(module).__name__:
            layer0_query = (name, module)
        if "layer.23.attention.self.query" in name and "AnalogLinear" in type(module).__name__:
            layer23_query = (name, module)

    if layer0_query:
        inspect_lora_layer("LAYER 0 (Query) - NORMAL", layer0_query[1])

    if layer23_query:
        inspect_lora_layer("LAYER 23 (Query) - EXPLOSION", layer23_query[1])

    # Also check all query layers for pattern
    print("\n" + "=" * 80)
    print("SURVEY: out_scaling_alpha across all layers")
    print("=" * 80)

    print(f"\n{'Layer':>6} {'tile_a_max':>12} {'tile_b_max':>12} {'tile_c_max':>12}")
    print("-" * 50)

    for i in range(24):
        layer_name = f"mobilebert.encoder.layer.{i}.attention.self.query"

        for name, module in model.named_modules():
            if name == layer_name and hasattr(module, 'analog_module'):
                analog_module = module.analog_module

                alpha_a = alpha_b = alpha_c = 0.0

                if hasattr(analog_module, 'tile_a') and hasattr(analog_module.tile_a, 'out_scaling_alpha'):
                    alpha_a = analog_module.tile_a.out_scaling_alpha.abs().max().item()

                if hasattr(analog_module, 'tile_b') and hasattr(analog_module.tile_b, 'out_scaling_alpha'):
                    alpha_b = analog_module.tile_b.out_scaling_alpha.abs().max().item()

                if hasattr(analog_module, 'tile_c') and hasattr(analog_module.tile_c, 'out_scaling_alpha'):
                    alpha_c = analog_module.tile_c.out_scaling_alpha.abs().max().item()

                marker = ""
                if i == 23:
                    marker = " ← EXPLOSION LAYER"
                elif alpha_a > 10 or alpha_b > 10 or alpha_c > 10:
                    marker = " ⚠️"

                print(f"{i:>6} {alpha_a:>12.6f} {alpha_b:>12.6f} {alpha_c:>12.6f}{marker}")
                break

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
