# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Python-level LRTT RPU configuration.

RPU configuration classes designed specifically for Python LRTT implementation.
"""

from dataclasses import dataclass, field
from typing import Type, Any, TYPE_CHECKING

from aihwkit.simulator.configs.configs import IOManagedRPUConfig
from aihwkit.simulator.tiles.array import TileModuleArray
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.parameters.enums import RPUDataType

if TYPE_CHECKING:
    from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile


def _get_lrtt_tile_class():
    """Lazy import to avoid circular dependency."""
    from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
    return LRTTSimulatorTile


@dataclass
class PythonLRTTRPUConfig(IOManagedRPUConfig):
    """RPU Configuration for Python-level LRTT implementation.

    This configuration automatically selects the appropriate LRTT tile based on
    the device configuration (standard LRTTSimulatorTile or SpatialLRTTSimulatorTile)
    and provides all necessary parameters for Python LRTT operation.
    """

    tile_class: Type = field(default_factory=_get_lrtt_tile_class)
    """Default tile class: LRTTSimulatorTile (may be overridden by device config)."""
    
    tile_array_class: Type = TileModuleArray
    """Tile array class for multi-tile scenarios."""
    
    device: PythonLRTTDevice = field(default_factory=PythonLRTTDevice)
    """Python LRTT device configuration."""
    
    def get_default_tile_module_class(self, out_size: int = 0, in_size: int = 0) -> Type:
        """Return tile class from device configuration.

        Delegates to device.get_default_tile_module_class() if available,
        allowing SpatialPythonLRTTDevice to return SpatialLRTTSimulatorTile.
        """
        if hasattr(self.device, 'get_default_tile_module_class'):
            return self.device.get_default_tile_module_class()
        return _get_lrtt_tile_class()
    
    def create_tile(self, x_size: int, d_size: int, dtype: RPUDataType = RPUDataType.FLOAT, **kwargs):
        """Create LRTT tile with this configuration.

        Uses get_default_tile_module_class() to determine the correct tile class,
        enabling both standard LRTTSimulatorTile and SpatialLRTTSimulatorTile.

        Args:
            x_size: Input size
            d_size: Output size
            dtype: Data type for tiles
            **kwargs: Additional arguments

        Returns:
            Configured LRTT tile instance (LRTTSimulatorTile or SpatialLRTTSimulatorTile)
        """
        tile_class = self.get_default_tile_module_class(out_size=d_size, in_size=x_size)
        return tile_class(
            x_size=x_size,
            d_size=d_size,
            rpu_config=self,
            dtype=dtype,
            **kwargs
        )
    
    def validate_dimensions(self, d_size: int, x_size: int) -> None:
        """Validate that dimensions are compatible with rank.
        
        Args:
            d_size: Output dimension
            x_size: Input dimension
            
        Raises:
            ValueError: If rank is incompatible with dimensions
        """
        if self.device.rank > min(d_size, x_size):
            raise ValueError(
                f"Rank {self.device.rank} too large for dimensions {d_size}×{x_size}. "
                f"Maximum rank is {min(d_size, x_size)}"
            )
    
    def get_brief_info(self) -> str:
        """Get brief configuration info."""
        return (f"PythonLRTTRPUConfig(rank={self.device.rank}, "
                f"transfer_every={self.device.transfer_every}, "
                f"lora_alpha={self.device.lora_alpha})")
    
    def as_bindings(self) -> Any:
        """Return self - no CUDA bindings needed for Python implementation.""" 
        return self


# Convenience factory functions for common configurations

def lrtt_idealized_config(rank: int = 4, transfer_every: int = 32, lora_alpha: float = 1.0) -> PythonLRTTRPUConfig:
    """Create idealized LRTT configuration.
    
    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency  
        lora_alpha: LoRA scaling factor
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.idealized(rank=rank, transfer_every=transfer_every, lora_alpha=lora_alpha)
    return PythonLRTTRPUConfig(device=device)


def lrtt_constant_step_config(rank: int = 4, transfer_every: int = 32, dw_min: float = 0.01) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with ConstantStepDevice.
    
    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency
        dw_min: Minimum weight update step
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.constant_step(rank=rank, transfer_every=transfer_every, dw_min=dw_min)
    return PythonLRTTRPUConfig(device=device)


def lrtt_lora_style_config(rank: int = 8, lora_alpha: float = 16.0, transfer_every: int = 1) -> PythonLRTTRPUConfig:
    """Create LoRA-style LRTT configuration.
    
    Args:
        rank: LoRA rank (higher for LoRA-style)
        lora_alpha: LoRA alpha (higher for LoRA-style)
        transfer_every: Transfer frequency (1 = every step)
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.lora_style(rank=rank, lora_alpha=lora_alpha, transfer_every=transfer_every)
    return PythonLRTTRPUConfig(device=device)


def lrtt_mixed_precision_config(rank: int = 4, transfer_every: int = 16) -> PythonLRTTRPUConfig:
    """Create mixed precision LRTT configuration.
    
    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.mixed_precision(rank=rank, transfer_every=transfer_every)
    return PythonLRTTRPUConfig(device=device)


def lrtt_inference_config(rank: int = 2, lora_alpha: float = 0.5) -> PythonLRTTRPUConfig:
    """Create inference-optimized LRTT configuration.
    
    Args:
        rank: Lower rank for inference
        lora_alpha: Lower alpha for stability
        
    Returns:
        Configured PythonLRTTRPUConfig
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.inference_optimized(rank=rank, lora_alpha=lora_alpha)
    return PythonLRTTRPUConfig(device=device)


# =============================================================================
# 6T1C Device Configurations
# =============================================================================

def lrtt_sixt1c_ab_config(
    rank: int = 4,
    transfer_every: int = 32,
    lora_alpha: float = 1.0,
    dt_batch_sec: float = 1.0,
    include_retention: bool = True,
    c_device=None
) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with 6T1C A/B tiles and configurable C tile.

    A/B tiles use 6T1C devices. C tile (visible) can use any device.

    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency (steps)
        lora_alpha: LoRA scaling factor
        dt_batch_sec: Assumed time per mini-batch in seconds (for 6T1C retention)
        include_retention: Whether to include retention effects for 6T1C
        c_device: Device for C tile. If None, uses IdealizedPresetDevice.
                  Can be: PCMPresetDevice, ReRamESPresetDevice, etc.

    Returns:
        Configured PythonLRTTRPUConfig with 6T1C A/B and custom C

    Example:
        >>> from aihwkit.simulator.presets.devices import PCMPresetDevice
        >>> config = lrtt_sixt1c_ab_config(c_device=PCMPresetDevice())
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.sixt1c_ab(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        dt_batch_sec=dt_batch_sec,
        include_retention=include_retention,
        c_device=c_device
    )
    return PythonLRTTRPUConfig(device=device)


def lrtt_sixt1c_ab_pcm_config(
    rank: int = 4,
    transfer_every: int = 32,
    lora_alpha: float = 1.0,
    dt_batch_sec: float = 1.0
) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with 6T1C A/B tiles and PCM C tile.

    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency (steps)
        lora_alpha: LoRA scaling factor
        dt_batch_sec: Assumed time per mini-batch in seconds

    Returns:
        Configured PythonLRTTRPUConfig with 6T1C A/B and PCM C
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.sixt1c_ab_pcm(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        dt_batch_sec=dt_batch_sec
    )
    return PythonLRTTRPUConfig(device=device)


def lrtt_sixt1c_ab_rram_config(
    rank: int = 4,
    transfer_every: int = 32,
    lora_alpha: float = 1.0,
    dt_batch_sec: float = 1.0
) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with 6T1C A/B tiles and RRAM C tile.

    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency (steps)
        lora_alpha: LoRA scaling factor
        dt_batch_sec: Assumed time per mini-batch in seconds

    Returns:
        Configured PythonLRTTRPUConfig with 6T1C A/B and RRAM C
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.sixt1c_ab_rram(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        dt_batch_sec=dt_batch_sec
    )
    return PythonLRTTRPUConfig(device=device)


def lrtt_sixt1c_ab_ideal_config(
    rank: int = 4,
    transfer_every: int = 32,
    lora_alpha: float = 1.0,
    dt_batch_sec: float = 1.0
) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with 6T1C A/B tiles and Idealized C tile.

    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency (steps)
        lora_alpha: LoRA scaling factor
        dt_batch_sec: Assumed time per mini-batch in seconds

    Returns:
        Configured PythonLRTTRPUConfig with 6T1C A/B and Idealized C
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        dt_batch_sec=dt_batch_sec
    )
    return PythonLRTTRPUConfig(device=device)


def lrtt_sixt1c_all_config(
    rank: int = 4,
    transfer_every: int = 32,
    lora_alpha: float = 1.0,
    dt_batch_sec: float = 1.0,
    include_retention: bool = True
) -> PythonLRTTRPUConfig:
    """Create LRTT configuration with 6T1C devices for ALL tiles (A, B, C).

    All three tiles use identical 6T1C device characteristics.

    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency (steps)
        lora_alpha: LoRA scaling factor
        dt_batch_sec: Assumed time per mini-batch in seconds (for retention)
        include_retention: Whether to include retention effects

    Returns:
        Configured PythonLRTTRPUConfig with 6T1C for all tiles
    """
    from .lrtt_python import PythonLRTTPreset
    device = PythonLRTTPreset.sixt1c_all(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
        dt_batch_sec=dt_batch_sec,
        include_retention=include_retention
    )
    return PythonLRTTRPUConfig(device=device)


# Legacy compatibility layer
def migrate_from_legacy_lrtt_compound(legacy_compound) -> PythonLRTTRPUConfig:
    """Migrate from legacy LRTTTransferCompound to Python LRTT.

    Args:
        legacy_compound: LRTTTransferCompound instance

    Returns:
        Equivalent PythonLRTTRPUConfig
    """
    python_device = PythonLRTTDevice.from_legacy_lrtt_compound(legacy_compound)
    return PythonLRTTRPUConfig(device=python_device)