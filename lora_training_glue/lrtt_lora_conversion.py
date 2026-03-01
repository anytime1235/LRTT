"""
LRTT-LoRA Model Conversion Utilities

This module provides functions to convert pretrained Transformer models to
LRTT-LoRA format by replacing Linear layers with LRTT AnalogLinear layers.

Key Functions:
- identify_lora_target_layers(): Find target Linear layers to convert
- convert_linear_to_lrtt(): Convert single Linear → LRTT AnalogLinear
- convert_model_to_lrtt_lora(): Convert entire model to LRTT-LoRA

Conversion Process:
1. Find target Linear layers (e.g., query, key, value)
2. Create LRTT AnalogLinear with unified tile (A, B, C sub-tiles)
3. Copy pretrained weights to C tile (frozen)
4. Initialize A=0, B~Kaiming (trainable)
5. Set requires_grad: A/B trainable, C frozen
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig


def identify_lora_target_layers(
    model: nn.Module,
    target_modules: List[str]
) -> Dict[str, nn.Linear]:
    """
    Identify Linear layers that should be converted to LRTT-LoRA.

    Args:
        model: The model to search
        target_modules: List of substrings to match layer names
            (e.g., ["query", "key", "value"])

    Returns:
        Dict mapping layer names to Linear modules

    Example:
        >>> targets = identify_lora_target_layers(model, ["query", "value"])
        >>> print(f"Found {len(targets)} target layers")
    """
    target_layers = {}

    for name, module in model.named_modules():
        # Only consider Linear layers
        if not isinstance(module, nn.Linear):
            continue

        # Exclude special layers (embedding, classifier, etc.)
        if any(exclude in name for exclude in ["embedding", "classifier", "qa_outputs", "LayerNorm", "pooler"]):
            continue

        # Check if layer name matches any target module substring
        if any(target in name for target in target_modules):
            target_layers[name] = module

    return target_layers


def convert_linear_to_lrtt(
    linear_layer: nn.Linear,
    lrtt_config: PythonLRTTRPUConfig,
) -> AnalogLinear:
    """
    Convert a single Linear layer to LRTT AnalogLinear.

    This function:
    1. Creates LRTT AnalogLinear from Linear layer
    2. Copies pretrained weights to C tile
    3. Initializes A=0 (LoRA-style, ΔW=0 initially)
    4. Initializes B~Kaiming Normal

    Args:
        linear_layer: Original Linear layer with pretrained weights
        lrtt_config: LRTT RPU configuration

    Returns:
        LRTT AnalogLinear layer with initialized weights

    Note:
        Trainability is set in convert_model_to_lrtt_lora() after
        all layers are converted.
    """
    # Create AnalogLinear from Linear layer (pretrained weights → C tile)
    analog_layer = AnalogLinear.from_digital(linear_layer, lrtt_config)

    # Get the LRTT tile (LRTTSimulatorTile)
    # Check if analog_module is TileModuleArray or direct tile
    from aihwkit.simulator.tiles.array import TileModuleArray
    from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

    if isinstance(analog_layer.analog_module, TileModuleArray):
        # Wrapped in array (non-split layers use array[0][0])
        tile = analog_layer.analog_module.array[0][0]
    elif isinstance(analog_layer.analog_module, LRTTSimulatorTile):
        # Direct tile (LRTT uses direct tile without array wrapper)
        tile = analog_layer.analog_module
    else:
        raise TypeError(f"Unexpected analog_module type: {type(analog_layer.analog_module)}")

    # Access sub-tiles: tile_a, tile_b, tile_c
    # The C tile already has pretrained weights from from_digital()

    # Initialize A tile to 0 (LoRA-style, ensures ΔW = A@B = 0 initially)
    # Get current A weights shape: [d_size, rank]
    weights_a, _ = tile.tile_a.get_weights()  # Returns (weight, bias)
    torch.nn.init.zeros_(weights_a)
    tile.tile_a.set_weights(weights_a)

    # Initialize B tile with Kaiming Normal (random initialization)
    # Get current B weights shape: [rank, x_size]
    weights_b, _ = tile.tile_b.get_weights()
    device = lrtt_config.device
    gain = device.reinit_gain if hasattr(device, 'reinit_gain') else 0.1
    torch.nn.init.kaiming_normal_(weights_b, a=gain)
    tile.tile_b.set_weights(weights_b)

    # C tile is already initialized with pretrained weights

    return analog_layer


def get_parent_module(model: nn.Module, layer_path: str) -> Tuple[nn.Module, str]:
    """
    Get parent module and attribute name for a layer path.

    Args:
        model: Root model
        layer_path: Dot-separated path (e.g., "encoder.layer.0.attention.self.query")

    Returns:
        Tuple of (parent_module, attribute_name)

    Example:
        >>> parent, attr = get_parent_module(model, "encoder.layer.0.query")
        >>> layer = getattr(parent, attr)  # Get the layer
        >>> setattr(parent, attr, new_layer)  # Replace the layer
    """
    parts = layer_path.split('.')
    parent = model

    # Traverse to parent module
    for part in parts[:-1]:
        parent = getattr(parent, part)

    # Return parent and final attribute name
    return parent, parts[-1]


def convert_model_to_lrtt_lora(
    model: nn.Module,
    lrtt_config: PythonLRTTRPUConfig,
    target_modules: List[str],
) -> nn.Module:
    """
    Convert entire model to LRTT-LoRA format.

    This function:
    1. Identifies target Linear layers
    2. Converts each to LRTT AnalogLinear
    3. Sets trainability:
       - tile_a, tile_b (+ their out_scaling): trainable (low-rank adapters)
       - tile_c (+ its out_scaling): frozen (pretrained weights)
       - bias: frozen
       - classifier/qa_outputs: trainable

    Args:
        model: Pretrained model to convert
        lrtt_config: LRTT RPU configuration
        target_modules: List of layer name substrings to convert
            (e.g., ["query", "key", "value", "dense"])

    Returns:
        Model with LRTT-LoRA layers

    Example:
        >>> from lrtt_lora_config import create_lrtt_lora_config
        >>> config = create_lrtt_lora_config(rank=8, lora_alpha=1.0)
        >>> model = convert_model_to_lrtt_lora(
        ...     model, config, target_modules=["query", "key", "value"]
        ... )
    """
    print("=" * 80)
    print("LRTT-LORA MODEL CONVERSION")
    print("=" * 80)

    # Step 1: Identify target layers
    print(f"\n[1/4] Identifying target layers (target_modules={target_modules})...")
    target_layers = identify_lora_target_layers(model, target_modules)
    print(f"Found {len(target_layers)} target layers to convert:")
    for name in list(target_layers.keys())[:5]:  # Show first 5
        print(f"  - {name}")
    if len(target_layers) > 5:
        print(f"  ... and {len(target_layers) - 5} more")

    # Step 2: Convert each target layer to LRTT
    print(f"\n[2/4] Converting {len(target_layers)} layers to LRTT-LoRA...")
    for layer_name, linear_layer in target_layers.items():
        # Get parent module to replace the layer
        parent, attr = get_parent_module(model, layer_name)

        # Convert Linear → LRTT AnalogLinear
        analog_layer = convert_linear_to_lrtt(linear_layer, lrtt_config)

        # Replace the layer
        setattr(parent, attr, analog_layer)

    print("✓ All target layers converted to LRTT AnalogLinear")

    # Step 3: Set trainability
    print("\n[3/4] Setting trainability...")
    n_trainable_params = 0
    n_frozen_params = 0

    # Use named_parameters() to directly set trainability (like reference implementation)
    # - tile_a, tile_b: TRAINABLE (including out_scaling_alpha)
    # - tile_c: FROZEN (including out_scaling_alpha) - CRITICAL FIX
    # - classifier/qa_outputs: TRAINABLE
    # - Everything else: FROZEN
    for name, param in model.named_parameters():
        if "classifier" in name or "qa_outputs" in name:
            # classifier/qa_outputs: TRAINABLE
            param.requires_grad = True
            n_trainable_params += param.numel()
        elif "tile_a" in name or "tile_b" in name:
            # A/B tiles and their out_scaling: TRAINABLE (low-rank adapters)
            param.requires_grad = True
            n_trainable_params += param.numel()
        elif "tile_c" in name:
            if "analog_ctx" in name:
                # tile_c.analog_ctx must keep requires_grad=True so that
                # AnalogFunction.backward() captures gradient for the
                # controller's ab_weight_update (hooked update mechanism)
                param.requires_grad = True
                n_trainable_params += param.numel()
            else:
                # C tile weights and out_scaling: FROZEN (pretrained weights)
                param.requires_grad = False
                n_frozen_params += param.numel()
        elif "bias" in name:
            # All bias: FROZEN
            param.requires_grad = False
            n_frozen_params += param.numel()
        else:
            # Embeddings and other layers: FROZEN
            param.requires_grad = False
            n_frozen_params += param.numel()

    print(f"✓ Trainability set:")
    print(f"  - Trainable params: {n_trainable_params:,}")
    print(f"  - Frozen params: {n_frozen_params:,}")
    print(f"  - Trainable fraction: {n_trainable_params / (n_trainable_params + n_frozen_params):.2%}")

    # Step 4: Summary
    print("\n[4/4] Conversion summary:")
    print(f"  - Architecture: Unified LRTT tiles (A, B, C sub-tiles)")
    print(f"  - Forward: y = C·x + α·A·(B·x)")
    print(f"  - Trainable: A, B tiles (low-rank adapters)")
    print(f"  - Frozen: C tile (pretrained weights)")
    print(f"  - LoRA rank: {lrtt_config.device.rank}")
    print(f"  - LoRA alpha: {lrtt_config.device.lora_alpha}")
    print("=" * 80)

    return model


def count_lrtt_layers(model: nn.Module) -> int:
    """Count number of LRTT AnalogLinear layers in model."""
    count = 0
    for module in model.modules():
        if isinstance(module, AnalogLinear):
            count += 1
    return count


def print_model_structure(model: nn.Module, max_depth: int = 3):
    """
    Print model structure showing LRTT layers.

    Args:
        model: Model to inspect
        max_depth: Maximum nesting depth to show
    """
    print("\nModel Structure (LRTT layers marked with *):")
    print("-" * 80)

    for name, module in model.named_modules():
        depth = name.count('.')
        if depth > max_depth:
            continue

        indent = "  " * depth
        module_type = type(module).__name__

        if isinstance(module, AnalogLinear):
            tile = module.analog_module.tile
            rank = lrtt_config.device.rank if hasattr(module, 'lrtt_config') else "?"
            print(f"{indent}{name}: {module_type} * [LRTT rank={rank}]")
        elif isinstance(module, nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            print(f"{indent}{name}: {module_type} [{in_f} → {out_f}]")


if __name__ == "__main__":
    # Test conversion on a dummy model
    print("Testing LRTT-LoRA conversion...\n")

    from lrtt_lora_config import create_lrtt_lora_config

    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(768, 768)
            self.key = nn.Linear(768, 768)
            self.value = nn.Linear(768, 768)
            self.dense = nn.Linear(768, 768)
            self.classifier = nn.Linear(768, 2)

        def forward(self, x):
            return self.classifier(self.dense(self.query(x)))

    # Create model and config
    model = SimpleModel()
    config = create_lrtt_lora_config(rank=8, lora_alpha=1.0)

    # Convert to LRTT-LoRA
    model = convert_model_to_lrtt_lora(
        model, config, target_modules=["query", "key", "value"]
    )

    # Verify conversion
    n_lrtt = count_lrtt_layers(model)
    print(f"\n✓ Conversion successful: {n_lrtt} LRTT layers created")
    print("\nExpected: 3 LRTT layers (query, key, value)")
    print("Expected: 2 Digital layers (dense, classifier)")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 768)  # Batch of 2, dim 768
    try:
        output = model(x)
        print(f"✓ Forward pass successful: output shape = {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
