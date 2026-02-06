# -*- coding: utf-8 -*-
"""LRTT model utilities for GLUE tasks."""

from typing import Dict, Any, Optional
from torch.nn import Module


def print_model_stats(
    model: Module,
    rank: int = 8,
    transfer_every: int = 1000,
    total_steps: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Print model statistics with LRTT parameter breakdown.

    Args:
        model: The analog model
        rank: LRTT rank
        transfer_every: Transfer frequency
        total_steps: Total training steps (optional)
        verbose: Print per-layer details
    """
    total_params = 0
    analog_params = 0
    digital_params = 0
    lrtt_layers = 0

    for name, module in model.named_modules():
        if hasattr(module, "analog_module"):
            # This is an analog layer
            tile = module.analog_module
            if hasattr(tile, "controller"):
                # LRTT tile
                lrtt_layers += 1
                d_size = tile.d_size
                x_size = tile.x_size
                # C tile params + A tile params + B tile params
                layer_params = d_size * x_size + d_size * rank + rank * x_size
                analog_params += layer_params
                if verbose:
                    print(f"  LRTT: {name} ({d_size}x{x_size}, rank={rank})")
            else:
                # Regular analog tile
                if hasattr(tile, "get_weights"):
                    w, b = tile.get_weights()
                    analog_params += w.numel()
                    if b is not None:
                        analog_params += b.numel()
        elif hasattr(module, "weight"):
            # Digital layer
            digital_params += module.weight.numel()
            if hasattr(module, "bias") and module.bias is not None:
                digital_params += module.bias.numel()

    total_params = analog_params + digital_params

    print("\n" + "=" * 60)
    print("Model Statistics")
    print("=" * 60)
    print(f"LRTT Layers: {lrtt_layers}")
    print(f"LRTT Rank: {rank}")
    print(f"Transfer Every: {transfer_every} steps")
    print(f"Total Parameters: {total_params:,}")
    print(f"  - Analog (LRTT): {analog_params:,}")
    print(f"  - Digital: {digital_params:,}")

    if total_steps is not None:
        expected_transfers = total_steps // transfer_every
        print(f"Expected Transfers: {expected_transfers} (per layer)")
    print("=" * 60 + "\n")


def get_lrtt_transfer_stats(model: Module) -> Dict[str, Any]:
    """Get LRTT transfer statistics from model.

    Args:
        model: The analog model

    Returns:
        Dictionary mapping layer names to transfer stats
    """
    stats = {}

    for name, module in model.named_modules():
        if hasattr(module, "analog_module"):
            tile = module.analog_module
            if hasattr(tile, "controller"):
                controller = tile.controller
                stats[name] = {
                    "transfer_count": getattr(controller, "transfer_count", 0),
                    "update_count": getattr(controller, "update_count", 0),
                    "d_size": tile.d_size,
                    "x_size": tile.x_size,
                    "rank": tile.rank,
                }

    return stats
