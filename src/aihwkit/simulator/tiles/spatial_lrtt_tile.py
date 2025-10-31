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
        
    def _patch_to_blocks(self, x_input: Tensor) -> Tensor:
        """Split a conv patch [*, Cin*k*k] into k blocks of [Cin*k].
        Each block corresponds to one row (u) across the k columns (v).
        Shape:
            in : [*, Cin*k*k]
            out: [*, k, Cin*k]
        """
        batch_dims = x_input.shape[:-1]
        x4 = x_input.view(*batch_dims, self.c_in, self.k, self.k)            # [*, Cin, k_u, k_v]
        # blocks[u] = x4[:, :, u, :] → [*, Cin, k_v] ⇒ flatten to [*, Cin*k]
        x_blocks = x4.permute(*range(len(batch_dims)), -2, -3, -1)           # [*, k_u, Cin, k_v]
        x_blocks = x_blocks.contiguous().view(*batch_dims, self.k, self.c_in * self.k)
        return x_blocks
        
    def _blocks_to_patch_grad(self, xg_blocks: Tensor) -> Tensor:
        """Merge k blocks of [Cin*k] gradient back to a patch [Cin*k*k].
        Shape:
            in : [*, k, Cin*k]
            out: [*, Cin*k*k]
        """
        batch_dims = xg_blocks.shape[:-2]
        xg4 = xg_blocks.view(*batch_dims, self.k, self.c_in, self.k)         # [*, k_u, Cin, k_v]
        xg4 = xg4.permute(*range(len(batch_dims)), -2, -3, -1)               # [*, Cin, k_u, k_v]
        return xg4.contiguous().view(*batch_dims, self.c_in * self.k * self.k)
        
    def _expand_error_for_blocks(self, d_input: Tensor) -> Tensor:
        """For y = sum over (u,v) of Y_block[u][:, v], the gradient w.r.t. each
        block output is just replication (no 1/k!), because forward used SUM.
        Shape:
            in  : [*, Cout]
            out : [*, k, Cout*k]
        """
        batch_dims = d_input.shape[:-1]
        d = d_input.unsqueeze(-1).expand(*batch_dims, self.c_out, self.k)    # [*, Cout, k_v]
        d = d.contiguous().view(*batch_dims, self.c_out * self.k)            # [*, Cout*k]
        d = d.unsqueeze(-2).expand(*batch_dims, self.k, self.c_out * self.k) # [*, k_u, Cout*k]
        return d
        
        
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
        """Im2Col-like block forward: split patch into k blocks → A/B → sum over (u,v)."""
        # 1) Patch → k blocks of size [Cin*k]
        x_blocks = self._patch_to_blocks(x_input)                            # [*, k, Cin*k]
        batch_dims = x_blocks.shape[:-2]
        xb2 = x_blocks.reshape(-1, self.c_in * self.k)                        # [*, Cin*k] with * = (batch... * k)

        # 2) Run analog LRTT chain (A/B) on blocks → [*, Cout*k]
        # Note: parent's forward() will store _last_x_input = xb2, which is the spatial format
        yb2 = super().forward(xb2, bias=bias, in_trans=in_trans,
                              out_trans=out_trans, is_test=is_test,
                              non_blocking=non_blocking, tensor_view=tensor_view)

        # 3) Sum over (u, v): reshape to [batch, k_u, Cout, k_v] then sum (1,3)
        yb4 = yb2.view(*batch_dims, self.k, self.c_out, self.k)               # [*, k, Cout, k]
        y_out = yb4.sum(dim=( -3, -1 ))                                       # → [*, Cout]

        # Store original input (channel-wise patch) for potential local A,B update
        # This overwrites parent's _last_x_input which is in spatial format
        self._last_x_input = x_input.detach().clone()

        return y_out
        
    def backward(
        self,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """Backward through block-sum. Expand dY to each block's (Cout*k), run A/Bᵀ, merge."""
        # Store original upstream grad (channel-wise) for potential local update
        if not self.controller.forward_inject_enabled:
            self._stored_d_input = d_input.detach().clone()                   # [*, Cout]
            self._stored_x_input = getattr(self, '_last_x_input', None)       # [*, Cin*k*k]
            # Temporarily disable gradient storage in parent class
            self._skip_gradient_storage = True

        # 1) dY replication for all (u,v) positions (SUM in forward ⇒ replicate)
        d_blocks = self._expand_error_for_blocks(d_input)                     # [*, k, Cout*k]
        batch_dims = d_blocks.shape[:-2]
        db2 = d_blocks.reshape(-1, self.c_out * self.k)                       # [*, Cout*k]

        # 2) Run analog LRTT backward on blocks → grad wrt [Cin*k]
        xg2 = super().backward(db2, bias=bias, in_trans=in_trans,
                               out_trans=out_trans, non_blocking=non_blocking) # [*, Cin*k]
        
        # Re-enable gradient storage in parent class
        if hasattr(self, '_skip_gradient_storage'):
            delattr(self, '_skip_gradient_storage')

        # 3) Merge k block-gradients back to patch space [Cin*k*k]
        xg_blocks = xg2.view(*batch_dims, self.k, self.c_in * self.k)         # [*, k, Cin*k]
        xg_out = self._blocks_to_patch_grad(xg_blocks)                        # [*, Cin*k*k]
        return xg_out
        
    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> None:
        """Local A/B update using the same block mapping as forward/backward."""
        # 0) Pick original (stored) tensors if forward_inject=False
        if (not self.controller.forward_inject_enabled and
            hasattr(self, '_stored_d_input') and hasattr(self, '_stored_x_input') and
            self._stored_x_input is not None and self._stored_d_input is not None):
            ux = self._stored_x_input                                         # [*, Cin*k*k]
            ud = self._stored_d_input                                         # [*, Cout]
            # Clear after use
            delattr(self, '_stored_x_input')
            delattr(self, '_stored_d_input')
        else:
            ux, ud = x_input, d_input

        # 1) Patch → blocks, and dY → block errors
        x_blocks = self._patch_to_blocks(ux)                                  # [*, k, Cin*k]
        d_blocks = self._expand_error_for_blocks(ud)                          # [*, k, Cout*k]

        xb2 = x_blocks.reshape(-1, self.c_in * self.k)                        # [*, Cin*k]
        db2 = d_blocks.reshape(-1, self.c_out * self.k)                       # [*, Cout*k]

        # 2) Local analog update for A/B tiles on blocks
        super().update(xb2, db2, bias=bias, in_trans=in_trans,
                       out_trans=out_trans, non_blocking=non_blocking)
        
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