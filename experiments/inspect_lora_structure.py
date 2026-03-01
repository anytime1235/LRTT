#!/usr/bin/env python
# coding=utf-8
"""
Inspect LRTT-LoRA Layer Structure (forward_inject=True)

Check the internal structure of AnalogLinear layers with unified A/B/C tiles.
"""

import sys
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, set_seed

sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

SEED = 42
MODEL_NAME = "google/mobilebert-uncased"

def inspect_analog_layer(name, module, depth=0):
    """Recursively inspect an AnalogLinear layer."""
    indent = "  " * depth
    print(f"{indent}[{name}]")
    print(f"{indent}  Type: {type(module).__name__}")

    # Check for analog_tile
    if hasattr(module, 'analog_tile'):
        print(f"{indent}  ✓ Has analog_tile")
        tile = module.analog_tile
        print(f"{indent}    Tile type: {type(tile).__name__}")

        # Check tile configuration
        if hasattr(tile, 'get_learning_rate'):
            print(f"{indent}    Learning rate: {tile.get_learning_rate()}")

        # Check for hidden parameters (weights stored in tile)
        if hasattr(tile, 'tile'):
            print(f"{indent}    ✓ Has tile.tile (underlying C++ tile)")

        # Check alpha_scale
        if hasattr(tile, 'alpha_scale'):
            print(f"{indent}    alpha_scale: {tile.alpha_scale}")

    # Check for explicit lora_A, lora_B (shouldn't exist with forward_inject=True)
    if hasattr(module, 'lora_A'):
        print(f"{indent}  ✓ Has lora_A (unexpected!)")
    if hasattr(module, 'lora_B'):
        print(f"{indent}  ✓ Has lora_B (unexpected!)")

    # Check for base_layer
    if hasattr(module, 'base_layer'):
        print(f"{indent}  ✓ Has base_layer")

    # Check named parameters
    params = list(module.named_parameters(recurse=False))
    if params:
        print(f"{indent}  Parameters ({len(params)}):")
        for pname, param in params:
            print(f"{indent}    - {pname}: shape={param.shape}, requires_grad={param.requires_grad}")

    # Check named buffers
    buffers = list(module.named_buffers(recurse=False))
    if buffers:
        print(f"{indent}  Buffers ({len(buffers)}):")
        for bname, buffer in buffers:
            print(f"{indent}    - {bname}: shape={buffer.shape}")

    # Check for forward_inject flag
    if hasattr(module, 'forward_inject'):
        print(f"{indent}  forward_inject: {module.forward_inject}")

    # Check children
    children = list(module.named_children())
    if children:
        print(f"{indent}  Children ({len(children)}):")
        for child_name, child_module in children:
            inspect_analog_layer(child_name, child_module, depth + 2)


def main():
    print("=" * 80)
    print("INSPECT: LRTT-LoRA Layer Structure (forward_inject=True)")
    print("=" * 80)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print("\n[1/3] Loading and converting model...")
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )

    lrtt_config = create_lrtt_lora_config(
        rank=8,
        lora_alpha=1.22,
        output_noise_level=0.0,
        use_floating_point=False,  # 6T1C mode
    )

    target_modules = ["query", "key", "value"]
    model = convert_model_to_lrtt_lora(model, lrtt_config, target_modules)
    model.to(device)

    print("\n[2/3] Inspecting LoRA layer structure...\n")

    # Find and inspect first LoRA layer
    found_layers = []
    for name, module in model.named_modules():
        # Look for AnalogLinear modules in attention
        if "attention.self" in name and "AnalogLinear" in type(module).__name__:
            found_layers.append((name, module))

    if not found_layers:
        print("❌ No AnalogLinear layers found!")
        return

    print(f"Found {len(found_layers)} AnalogLinear layers\n")
    print("=" * 80)
    print("INSPECTING FIRST 3 LAYERS:")
    print("=" * 80)

    for i, (name, module) in enumerate(found_layers[:3]):
        print(f"\n{'='*80}")
        print(f"Layer {i+1}: {name}")
        print('='*80)
        inspect_analog_layer(name, module)

    # Check analog_tile structure in detail
    print("\n" + "=" * 80)
    print("[3/3] DETAILED ANALOG TILE INSPECTION")
    print("=" * 80)

    first_name, first_module = found_layers[0]
    print(f"\nInspecting: {first_name}")

    if hasattr(first_module, 'analog_tile'):
        tile = first_module.analog_tile
        print(f"\nAnalog Tile Attributes:")

        # List all attributes
        attrs = [attr for attr in dir(tile) if not attr.startswith('_')]
        for attr in attrs[:30]:  # First 30 to avoid clutter
            try:
                value = getattr(tile, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
            except:
                pass

        # Check for specific LRTT attributes
        print("\nLRTT-Specific Attributes:")
        lrtt_attrs = [
            'alpha_scale', 'rank', 'forward_inject',
            'lora_tile_A', 'lora_tile_B', 'base_tile',
            'in_size', 'out_size', 'd_size'
        ]
        for attr in lrtt_attrs:
            if hasattr(tile, attr):
                value = getattr(tile, attr)
                print(f"  ✓ {attr}: {value}")
            else:
                print(f"  ✗ {attr}: not found")

        # Try to access weights
        print("\nWeight Access:")
        try:
            weights = tile.get_weights()
            print(f"  ✓ get_weights() works: shape={weights.shape}")
            print(f"    min={weights.min():.6f}, max={weights.max():.6f}")
            print(f"    mean={weights.mean():.6f}, std={weights.std():.6f}")
        except Exception as e:
            print(f"  ✗ get_weights() failed: {e}")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
