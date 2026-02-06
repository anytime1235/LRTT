"""LoRA Merge with Analog C Tile.

This module implements LRTT with Digital A,B tiles (FloatingPointDevice)
and Analog C tile (SoftBoundsDevice with noise=0).

Key concepts:
- A, B tiles: FloatingPointDevice (exact digital computation)
- C tile: SoftBoundsDevice (analog, noise=0)
- Transfer method: "set" (exact transfer, no pulsed update noise)
- Update mode: "lora" (LoRA chain rule projection)
"""

from .config import (
    SOFTBOUNDS_NO_NOISE_CONFIG,
    create_lora_merge_config,
)

__all__ = [
    "SOFTBOUNDS_NO_NOISE_CONFIG",
    "create_lora_merge_config",
]
