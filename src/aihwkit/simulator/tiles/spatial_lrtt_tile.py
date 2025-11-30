# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Spatial LR-TT Simulator Tile (LoRA-C formulation).

Implements spatial-wise LRTT decomposition based on LoRA-C paper:
- Standard LoRA: A:[c_out, rank], B:[rank, c_in×k×k] → rank×(c_out + c_in×k×k)
- LoRA-C (Spatial): A:[c_out×k, rank×k], B:[rank×k, c_in×k] → rank×k²×(c_out + c_in)

Key properties:
- Higher effective rank (rank × k) despite configured rank
- More parameters than Standard LoRA but better spatial decomposition
- Example: k=3, c_in=64, c_out=128, rank=8 → effective rank=24, params increase ~2.45x
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
    """Spatial LR-TT simulator tile (LoRA-C formulation).

    Implements LoRA-C spatial decomposition:
    - Standard LoRA: rank×(c_out + c_in×k×k) parameters
    - LoRA-C: rank×k²×(c_out + c_in) parameters (more parameters!)
    - Effective rank: rank × k (higher expressiveness)
    - Uses tensor reshaping to bridge channel-wise ↔ spatial-wise formats

    Architecture:
    - Physical tiles: A:[c_out×k, rank×k], B:[rank×k, c_in×k], C:[c_out×k, c_in×k]
    - Interface: Compatible with existing conv layers (channel-wise I/O)
    - rank parameter: User-configured rank (actual rank = rank × k internally)
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
        # Try kernel sizes from largest to smallest to detect the actual kernel size
        # k=1 is valid for skip connections (1×1 conv), but we prefer larger k if divisible
        possible_k_values = [7, 5, 3, 1]  # Common kernel sizes, largest first

        for k in possible_k_values:
            if x_size % (k * k) == 0:
                self.c_in = x_size // (k * k)
                self.k = k
                break
        else:
            # Fallback: assume k=3 (most common)
            self.k = 3
            self.c_in = x_size // (self.k * self.k)

        # Store base rank (user-configured rank)
        self.base_rank = rpu_config.device.rank

        # LoRA-C: rank dimension is multiplied by k
        # A: [c_out×k, rank×k], B: [rank×k, c_in×k]
        # Need to modify rpu_config to use spatial_rank = base_rank × k
        from copy import deepcopy
        spatial_rpu_config = deepcopy(rpu_config)
        spatial_rpu_config.device.rank = self.base_rank * self.k  # ← Key change!

        # Calculate spatial tile dimensions
        # A: [c_out×k, rank×k], B: [rank×k, c_in×k], C: [c_out×k, c_in×k]
        spatial_d_size = self.c_out * self.k   # c_out×k
        spatial_x_size = self.c_in * self.k    # c_in×k

        # Initialize parent with spatial dimensions and modified config
        super().__init__(
            d_size=spatial_d_size,
            x_size=spatial_x_size,
            rpu_config=spatial_rpu_config,  # ← Use modified config with rank×k
            bias=bias,
            dtype=dtype,
            **kwargs
        )

        # Now self.rank (from parent) = base_rank × k
        # Store parameter counts for comparison
        # Standard LoRA: rank × (c_out + c_in×k²)
        self.standard_lora_params = self.base_rank * (self.c_out + self.c_in * self.k * self.k)
        # LoRA-C (Spatial): rank × k² × (c_out + c_in)
        self.spatial_lora_params = self.base_rank * self.k * self.k * (self.c_out + self.c_in)
        # Parameter increase ratio (LoRA-C has MORE parameters)
        self.param_ratio = self.spatial_lora_params / self.standard_lora_params

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
        """Multi-pass spatial forward with full LoRA-C support.

        Complete LoRA-C forward: y = C@x + alpha * A@(B@x)

        Forward flow for each spatial position u:
        1. Extract input slice x_u: [batch, c_in*k]
        2. C tile forward: y_c_u = C @ x_u^T → [batch, c_out*k]
        3. B tile forward: tmp_u = B @ x_u^T → [batch, rank*k] (LoRA compression)
        4. A tile forward: y_ab_u = A @ tmp_u^T → [batch, c_out*k] (LoRA expansion)
        5. Extract rows for position u from both y_c_u and y_ab_u
        6. Accumulate: y = Σ_u(y_c_u + alpha * y_ab_u)

        This is mathematically equivalent to spatial LoRA-C convolution!
        """
        # Store input for backward (detach only - no clone to save memory)
        self._last_x_input = x_input.detach()

        if bias:
            raise TileError("Spatial LRTT does not support bias")

        # Handle input transpose
        if in_trans:
            x_input = x_input.t()

        # Get batch dimension
        batch_size = x_input.shape[0]

        # Reshape input to spatial format: [batch, c_in, k, k]
        x_4d = x_input.view(batch_size, self.c_in, self.k, self.k)

        # Accumulate outputs directly (memory efficient - no lists)
        y_c = None
        y_ab = None if self.controller.forward_inject_enabled else None

        for u in range(self.k):
            # Extract kernel row u: [batch, c_in, k] → [batch, c_in*k]
            x_u = x_4d[:, :, u, :].contiguous().view(batch_size, self.c_in * self.k)

            # === C tile forward (main convolution) ===
            y_c_u = self.tile_c.forward(x_u, in_trans=False, out_trans=False)
            y_c_u_selected = y_c_u[:, u::self.k]

            # Accumulate directly
            if y_c is None:
                y_c = y_c_u_selected
            else:
                y_c.add_(y_c_u_selected)  # In-place add

            # === A, B tile forward (LoRA term) ===
            if self.controller.forward_inject_enabled:
                tmp_u = self.tile_b.forward(x_u, in_trans=False, out_trans=False)
                y_ab_u = self.tile_a.forward(tmp_u, in_trans=False, out_trans=False)
                y_ab_u_selected = y_ab_u[:, u::self.k]

                # Accumulate directly
                if y_ab is None:
                    y_ab = y_ab_u_selected
                else:
                    y_ab.add_(y_ab_u_selected)  # In-place add

        # Combine C and LoRA terms
        if self.controller.forward_inject_enabled and y_ab is not None:
            y = y_c + self.controller.lora_alpha * y_ab
        else:
            y = y_c

        # Handle output transpose
        if out_trans:
            y = y.t()

        return y
        
    def backward(
        self,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Multi-pass spatial backward with full LoRA-C support.

        Complete LoRA-C backward: dx = C^T @ dy + alpha * B^T @ (A^T @ dy)

        Backward flow for each spatial position u:
        1. Expand gradient d_u for position u: [batch, c_out*k]
        2. C tile backward: x_grad_c_u = C^T @ d_u^T → [batch, c_in*k]
        3. If forward_inject enabled:
           - A tile backward: tmp_grad_u = A^T @ d_u^T → [batch, rank*k]
           - B tile backward: x_grad_ab_u = B^T @ tmp_grad_u^T → [batch, c_in*k]
        4. Combine: x_grad_u = x_grad_c_u + alpha * x_grad_ab_u
        5. Merge all spatial slices back to [batch, c_in*k*k]
        """
        # Store for A, B update (detach only - no clone to save memory)
        if not self.controller.forward_inject_enabled:
            self._stored_d_input = d_input.detach()
            self._stored_x_input = getattr(self, '_last_x_input', None)

        # Handle output transpose
        if out_trans:
            d_input = d_input.t()

        # Get batch dimension
        batch_size = d_input.shape[0]

        # Pre-allocate d_u buffer (reuse across loop)
        d_u = torch.zeros(batch_size, self.c_out * self.k,
                         device=d_input.device, dtype=d_input.dtype)

        # Pre-allocate output gradient buffer
        x_grad = torch.zeros(batch_size, self.c_in, self.k, self.k,
                            device=d_input.device, dtype=d_input.dtype)

        # Multi-pass over spatial positions
        for u in range(self.k):
            # Reuse d_u buffer - zero out and fill
            d_u.zero_()
            d_u[:, u::self.k] = d_input  # Place gradient at corresponding rows

            # === C tile backward (main path) ===
            x_grad_c_u = self.tile_c.backward(d_u, in_trans=False, out_trans=False)
            # x_grad_c_u: [batch, c_in*k]

            # Reshape and accumulate into x_grad
            x_grad_c_u_4d = x_grad_c_u.view(batch_size, self.c_in, self.k)
            x_grad[:, :, u, :].add_(x_grad_c_u_4d)  # In-place add to row u

            # === A, B tile backward (LoRA path) ===
            if self.controller.forward_inject_enabled:
                tmp_grad_u = self.tile_a.backward(d_u, in_trans=False, out_trans=False)
                x_grad_ab_u = self.tile_b.backward(tmp_grad_u, in_trans=False, out_trans=False)

                # Reshape and accumulate
                x_grad_ab_u_4d = x_grad_ab_u.view(batch_size, self.c_in, self.k)
                x_grad[:, :, u, :].add_(x_grad_ab_u_4d * self.controller.lora_alpha)

        # Flatten back to [batch, c_in*k*k]
        x_grad = x_grad.contiguous().view(batch_size, self.c_in * self.k * self.k)

        # Handle input transpose
        if in_trans:
            x_grad = x_grad.t()

        return x_grad
        
    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> None:
        """Multi-pass spatial update for A/B tiles.

        Update flow (matching forward/backward spatial slicing):
        1. For each spatial position u:
           - Extract input slice x_u: [batch, c_in*k]
           - Expand gradient d_u: [batch, c_out*k] with values at positions u
           - Analog tile update: A/B tiles updated with (x_u, d_u)

        Note: Since Spatial LRTT uses forward_inject=False, we update using stored gradients.
        """
        # Pick stored tensors (always use stored for Spatial LRTT)
        if (hasattr(self, '_stored_d_input') and hasattr(self, '_stored_x_input') and
            self._stored_x_input is not None and self._stored_d_input is not None):
            ux = self._stored_x_input  # [batch, c_in*k*k]
            ud = self._stored_d_input  # [batch, c_out]
            # Clear after use
            delattr(self, '_stored_x_input')
            delattr(self, '_stored_d_input')
        else:
            ux, ud = x_input, d_input

        # Get batch dimension
        batch_size = ux.shape[0]

        # Reshape input to spatial format
        x_4d = ux.view(batch_size, self.c_in, self.k, self.k)

        # Pre-allocate d_u buffer (reuse across loop)
        d_u = torch.zeros(batch_size, self.c_out * self.k,
                         device=ud.device, dtype=ud.dtype)

        # Multi-pass update over spatial positions
        for u in range(self.k):
            # Extract input slice for position u
            x_u = x_4d[:, :, u, :].contiguous().view(batch_size, self.c_in * self.k)

            # Reuse d_u buffer - zero out and fill
            d_u.zero_()
            d_u[:, u::self.k] = ud

            # Analog tile update for this spatial slice
            # This updates A and B tiles using analog outer product
            super().update(x_u, d_u, bias=bias, in_trans=in_trans,
                          out_trans=out_trans, non_blocking=non_blocking)
        
    def set_weights(self, weight: Tensor, bias: Optional[Tensor] = None) -> None:
        """Set weights with automatic reshape to spatial format.

        Args:
            weight: Conv weight [c_out, c_in, k, k] or flattened [c_out, c_in*k*k]
            bias: Bias (not supported)
        """
        if bias is not None:
            raise TileError("Spatial LRTT does not support bias")

        # Detect input format and reshape to spatial format
        if weight.dim() == 4:
            # 4D conv weight: [c_out, c_in, k, k]
            if weight.shape != (self.c_out, self.c_in, self.k, self.k):
                raise TileError(f"Weight shape {weight.shape} does not match expected "
                               f"[{self.c_out}, {self.c_in}, {self.k}, {self.k}]")
            # Reshape to spatial format: permute(0,2,1,3) then flatten
            weight_spatial = weight.permute(0, 2, 1, 3).reshape(self.c_out * self.k, self.c_in * self.k)

        elif weight.dim() == 2:
            # 2D flattened weight: [c_out, c_in*k*k]
            if weight.shape == (self.c_out, self.c_in * self.k * self.k):
                # Regular LRTT format → reshape to spatial format
                weight_4d = weight.view(self.c_out, self.c_in, self.k, self.k)
                weight_spatial = weight_4d.permute(0, 2, 1, 3).reshape(self.c_out * self.k, self.c_in * self.k)
            elif weight.shape == (self.c_out * self.k, self.c_in * self.k):
                # Already in spatial format
                weight_spatial = weight
            else:
                raise TileError(f"Weight shape {weight.shape} does not match expected formats")
        else:
            raise TileError(f"Weight must be 2D or 4D, got {weight.dim()}D")

        # Set to C tile in spatial format
        self.tile_c.set_weights(weight_spatial, None)

    def get_parameter_info(self) -> Dict[str, Any]:
        """Get parameter count comparison."""
        return {
            'base_rank': self.base_rank,
            'spatial_rank': self.rank,  # = base_rank × k
            'effective_rank': self.rank,  # Same as spatial_rank
            'standard_lora_params': self.standard_lora_params,
            'spatial_lora_params': self.spatial_lora_params,
            'param_ratio': self.param_ratio,
            'param_increase_percentage': f"{(self.param_ratio - 1.0) * 100:.1f}%"
        }
        
    def get_spatial_component_weights(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Get LRTT component weights in spatial format.

        Returns:
            Tuple of (C_spatial, A_spatial, B_spatial) where:
            - C_spatial: [c_out×k, c_in×k]
            - A_spatial: [c_out×k, rank×k]  ← rank dimension is k times larger
            - B_spatial: [rank×k, c_in×k]   ← rank dimension is k times larger
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
               f"base_rank={self.base_rank}, spatial_rank={self.rank}, " \
               f"param_ratio={self.param_ratio:.2f}x)"

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f"c_out={self.c_out}, c_in={self.c_in}, k={self.k}, " \
               f"spatial_dims=({self.c_out*self.k}, {self.c_in*self.k}), " \
               f"base_rank={self.base_rank}, spatial_rank={self.rank}, " \
               f"param_ratio={self.param_ratio:.2f}x, " \
               f"transfer_every={self.transfer_every}, lora_alpha={self.lora_alpha}"