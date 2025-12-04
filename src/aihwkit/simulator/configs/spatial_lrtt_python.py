# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Spatial LR-TT Python configurations for parameter reduction."""

from dataclasses import dataclass
from typing import List, Optional

from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs.devices import FloatingPointDevice
from aihwkit.simulator.presets.devices import IdealizedPresetDevice


@dataclass
class SpatialPythonLRTTDevice(PythonLRTTDevice):
    """Spatial Python LRTT device configuration (LoRA-C formulation).

    Extends PythonLRTTDevice to use SpatialLRTTSimulatorTile implementing
    LoRA-C spatial decomposition.

    Parameter comparison:
    - Standard LoRA: rank × (c_out + c_in×k×k)
    - LoRA-C (Spatial): rank × k² × (c_out + c_in)
    - Typical increase: 2-3x more parameters than Standard LoRA
    - But: Higher effective rank (rank × k) for better expressiveness

    Inherits all parameters from PythonLRTTDevice including:
    - reinit_mode: Reinit strategy ('standard', 'decay', 'hybrid')
    - decay_factor: Decay factor for 'decay' and 'hybrid' modes
    """
    
    def get_default_tile_module_class(self):
        """Return the spatial LRTT tile class."""
        from aihwkit.simulator.tiles.spatial_lrtt_tile import SpatialLRTTSimulatorTile
        return SpatialLRTTSimulatorTile


class SpatialPythonLRTTPreset:
    """Preset configurations for Spatial Python LRTT."""
    
    @staticmethod
    def idealized(
        rank: int = 8,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        transfer_lr: Optional[float] = None,
        unit_cell_devices: Optional[List] = None,
        reinit_mode: str = "standard",
        decay_factor: float = 0.9
    ) -> SpatialPythonLRTTDevice:
        """Create idealized spatial LRTT device with parameter reduction.

        Args:
            rank: LoRA rank
            transfer_every: Transfer frequency
            lora_alpha: LoRA scaling factor
            transfer_lr: Transfer learning rate (defaults to lora_alpha)
            unit_cell_devices: Device list [A_device, B_device, C_device]
            reinit_mode: Reinit strategy ('standard', 'decay', 'hybrid')
            decay_factor: Decay factor for 'decay' and 'hybrid' modes

        Returns:
            Configured SpatialPythonLRTTDevice
        """
        if transfer_lr is None:
            transfer_lr = lora_alpha

        if unit_cell_devices is None:
            unit_cell_devices = [
                FloatingPointDevice(),      # A matrix: LoRA left factor
                FloatingPointDevice(),      # B matrix: LoRA right factor
                IdealizedPresetDevice(),    # C matrix: Main weights
            ]

        return SpatialPythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            transfer_lr=transfer_lr,
            unit_cell_devices=unit_cell_devices,
            units_in_mbatch=False,
            reinit_gain=0.1,
            reinit_mode=reinit_mode,
            decay_factor=decay_factor,
            correct_gradient_magnitudes=False,
            forward_inject=True
        )
        
    @staticmethod
    def floating_point(
        rank: int = 8,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        transfer_lr: Optional[float] = None,
        reinit_mode: str = "standard",
        decay_factor: float = 0.9
    ) -> SpatialPythonLRTTDevice:
        """Create floating point spatial LRTT device for debugging.

        All matrices use FloatingPointDevice for exact arithmetic.

        Args:
            rank: LoRA rank
            transfer_every: Transfer frequency
            lora_alpha: LoRA scaling factor
            transfer_lr: Transfer learning rate (defaults to lora_alpha)
            reinit_mode: Reinit strategy ('standard', 'decay', 'hybrid')
            decay_factor: Decay factor for 'decay' and 'hybrid' modes
        """
        if transfer_lr is None:
            transfer_lr = lora_alpha

        unit_cell_devices = [
            FloatingPointDevice(),  # A matrix
            FloatingPointDevice(),  # B matrix
            FloatingPointDevice(),  # C matrix
        ]

        return SpatialPythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            transfer_lr=transfer_lr,
            unit_cell_devices=unit_cell_devices,
            units_in_mbatch=False,
            reinit_gain=0.1,
            reinit_mode=reinit_mode,
            decay_factor=decay_factor,
            correct_gradient_magnitudes=False,
            forward_inject=True
        )