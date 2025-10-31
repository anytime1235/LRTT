# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Spatial LR-TT Simulator Tile for parameter reduction.

Implements spatial-wise LRTT decomposition to reduce LoRA parameter count:
- Current: A:[c_out, rank], B:[rank, c_in×k×k] → rank×(c_out + c_in×k×k)
- Spatial: A:[c_out×k, rank], B:[rank, c_in×k] → rank×k×(c_out + c_in)

Key benefit: Significant parameter reduction while maintaining analog properties.
Example: k=3, c_in=64, c_out=128 → ~18% parameter reduction
"""

from typing import Optional, Tuple, Any, Dict
import torch
from torch import Tensor
from torch.nn import Module

from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
from aihwkit.simulator.parameters.base import RPUConfigGeneric
from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.configs.configs import UnitCellRPUConfig
from aihwkit.exceptions import ConfigError, TileError


class SpatialLRTTSimulatorTile(LRTTSimulatorTile):
    """Spatial LR-TT simulator tile for parameter reduction.
    
    Wraps standard LRTT tile to provide spatial decomposition:
    - Reduces LoRA parameters from rank×(c_out + c_in×k×k) to rank×k×(c_out + c_in)
    - Maintains all analog simulation properties for speed/memory efficiency
    - Uses tensor reshaping to bridge channel-wise ↔ spatial-wise formats
    
    Architecture:
    - Physical tiles: A:[c_out×k, rank], B:[rank, c_in×k], C:[c_out×k, c_in×k]
    - Interface: Compatible with existing conv layers (channel-wise I/O)
    """
    
    def __init__(
        self,
        d_size: int,         # out_features (same as LRTTSimulatorTile interface)
        x_size: int,         # in_features (same as LRTTSimulatorTile interface)
        rpu_config: UnitCellRPUConfig,
        bias: bool = False,
        dtype: Optional[RPUDataType] = None,
        **kwargs
    ):
        """Initialize Spatial LRTT simulator tile.
        
        Args:
            d_size: Output dimension (out_features for conv = c_out)  
            x_size: Input dimension (in_features for conv = c_in*k*k)
            rpu_config: LRTT configuration
            bias: Whether to use bias
            dtype: Data type
        """
        # Infer spatial dimensions from conv layer dimensions
        # x_size = c_in * k * k, d_size = c_out
        # We need to estimate k from x_size and d_size
        
        # For conv layers: x_size = c_in * k * k, d_size = c_out
        # Assume reasonable kernel sizes (3x3 is most common)
        # Try to find k such that x_size = c_in * k * k
        
        self.c_out = d_size
        possible_k_values = [1, 3, 5, 7]  # Common kernel sizes
        
        for k in possible_k_values:
            if x_size % (k * k) == 0:
                self.c_in = x_size // (k * k)
                self.k = k
                break
        else:
            # Fallback: assume k=3 (most common)
            self.k = 3
            self.c_in = x_size // (self.k * self.k)
            
        # Calculate spatial tile dimensions for parameter reduction
        # A: [c_out×k, rank], B: [rank, c_in×k], C: [c_out×k, c_in×k]  
        spatial_d_size = self.c_out * self.k   # c_out×k
        spatial_x_size = self.c_in * self.k    # c_in×k
        
        # Initialize parent with spatial dimensions  
        super().__init__(
            d_size=spatial_d_size,
            x_size=spatial_x_size, 
            rpu_config=rpu_config,
            bias=bias,
            dtype=dtype,
            **kwargs
        )
        
        # Store original parameter counts for comparison
        self.original_params = self.rank * (self.c_out + self.c_in * self.k * self.k)
        self.spatial_params = self.rank * self.k * (self.c_out + self.c_in)
        self.param_reduction = 1.0 - (self.spatial_params / self.original_params)
        
    def _reshape_input_to_spatial(self, x_input: Tensor) -> Tensor:
        """Reshape input from channel-wise [*, c_in×k×k] to spatial [*, c_in×k].
        
        Strategy: Pool spatial dimensions to reduce c_in×k×k → c_in×k
        """
        batch_dims = x_input.shape[:-1]
        
        # Reshape [*, c_in×k×k] → [*, c_in, k×k]
        x_reshaped = x_input.view(*batch_dims, self.c_in, self.k * self.k)
        
        # Average pool: [*, c_in, k×k] → [*, c_in, k] (reduce one spatial dim)
        x_pooled = x_reshaped.view(*batch_dims, self.c_in, self.k, self.k).mean(dim=-1)
        
        # Flatten: [*, c_in, k] → [*, c_in×k]
        x_spatial = x_pooled.contiguous().view(*batch_dims, self.c_in * self.k)
        
        return x_spatial
        
    def _reshape_output_from_spatial(self, y_spatial: Tensor) -> Tensor:
        """Reshape output from spatial [*, c_out×k] to channel-wise [*, c_out].
        
        Strategy: Average over spatial dimension to get final channel output
        """
        batch_dims = y_spatial.shape[:-1]
        
        # Reshape [*, c_out×k] → [*, c_out, k] 
        y_reshaped = y_spatial.view(*batch_dims, self.c_out, self.k)
        
        # Average over spatial dimension: [*, c_out, k] → [*, c_out]
        y_output = y_reshaped.mean(dim=-1)
        
        return y_output
        
    def _reshape_error_to_spatial(self, d_input: Tensor) -> Tensor:
        """Reshape error from channel-wise [*, c_out] to spatial [*, c_out×k].
        
        Strategy: Replicate error across spatial positions
        """
        batch_dims = d_input.shape[:-1]
        
        # Expand [*, c_out] → [*, c_out, k] by replication
        d_expanded = d_input.unsqueeze(-1).expand(*batch_dims, self.c_out, self.k)
        
        # Flatten [*, c_out, k] → [*, c_out×k]
        d_spatial = d_expanded.contiguous().view(*batch_dims, self.c_out * self.k)
        
        return d_spatial
        
    def _reshape_error_from_spatial(self, x_grad_spatial: Tensor) -> Tensor:
        """Reshape gradient from spatial [*, c_in×k] to channel-wise [*, c_in×k×k].
        
        Strategy: Expand spatial gradient to full kernel size
        """
        batch_dims = x_grad_spatial.shape[:-1]
        
        # Reshape [*, c_in×k] → [*, c_in, k]
        x_grad_reshaped = x_grad_spatial.view(*batch_dims, self.c_in, self.k)
        
        # Expand [*, c_in, k] → [*, c_in, k, k] by replication
        x_grad_expanded = x_grad_reshaped.unsqueeze(-1).expand(
            *batch_dims, self.c_in, self.k, self.k
        )
        
        # Flatten [*, c_in, k, k] → [*, c_in×k×k] 
        x_grad_output = x_grad_expanded.contiguous().view(
            *batch_dims, self.c_in * self.k * self.k
        )
        
        return x_grad_output
        
    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
        tensor_view: Optional[Tuple] = None,
    ) -> Tensor:
        """Spatial LRTT forward pass with parameter reduction."""
        # Channel-wise → Spatial conversion
        x_spatial = self._reshape_input_to_spatial(x_input)
        
        # Run analog LRTT forward (reduced parameters)
        y_spatial = super().forward(
            x_spatial, bias=bias, in_trans=in_trans, out_trans=out_trans,
            is_test=is_test, non_blocking=non_blocking, tensor_view=tensor_view
        )
        
        # Spatial → Channel-wise conversion
        y_output = self._reshape_output_from_spatial(y_spatial)
        
        return y_output
        
    def backward(
        self,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Spatial LRTT backward pass with parameter reduction."""
        # Channel-wise → Spatial conversion
        d_spatial = self._reshape_error_to_spatial(d_input)
        
        # Run analog LRTT backward (reduced parameters)
        x_grad_spatial = super().backward(
            d_spatial, bias=bias, in_trans=in_trans, out_trans=out_trans,
            non_blocking=non_blocking
        )
        
        # Spatial → Channel-wise conversion
        x_grad_output = self._reshape_error_from_spatial(x_grad_spatial)
        
        return x_grad_output
        
    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> None:
        """Spatial LRTT update with parameter reduction."""
        # Convert both tensors to spatial format
        x_spatial = self._reshape_input_to_spatial(x_input)
        d_spatial = self._reshape_error_to_spatial(d_input)
        
        # Run analog LRTT update (reduced parameters)
        super().update(
            x_spatial, d_spatial, bias=bias, in_trans=in_trans,
            out_trans=out_trans, non_blocking=non_blocking
        )
        
    def get_parameter_info(self) -> Dict[str, Any]:
        """Get parameter count comparison."""
        return {
            'original_params': self.original_params,
            'spatial_params': self.spatial_params,
            'param_reduction': self.param_reduction,
            'reduction_percentage': f"{self.param_reduction * 100:.1f}%"
        }
        
    def get_spatial_component_weights(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Get LRTT component weights in spatial format.
        
        Returns:
            Tuple of (C_spatial, A_spatial, B_spatial) where:
            - C_spatial: [c_out×k, c_in×k] 
            - A_spatial: [c_out×k, rank]
            - B_spatial: [rank, c_in×k]
        """
        return self.get_lrtt_component_weights()
        
    def set_spatial_component_weights(
        self, 
        C_spatial: Tensor, 
        A_spatial: Tensor, 
        B_spatial: Tensor
    ) -> None:
        """Set LRTT component weights from spatial format."""
        self.set_lrtt_component_weights(C_spatial, A_spatial, B_spatial)
        
    def get_brief_info(self) -> str:
        """Get brief tile information."""
        return f"SpatialLRTTSimulatorTile({self.c_out}×{self.c_in}×{self.k}×{self.k}, " \
               f"rank={self.rank}, param_reduction={self.param_reduction:.1%})"
               
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f"c_out={self.c_out}, c_in={self.c_in}, k={self.k}, " \
               f"spatial_dims=({self.c_out*self.k}, {self.c_in*self.k}), rank={self.rank}, " \
               f"param_reduction={self.param_reduction:.1%}, " \
               f"transfer_every={self.transfer_every}, lora_alpha={self.lora_alpha}"