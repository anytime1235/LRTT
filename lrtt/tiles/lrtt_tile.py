# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""LR-TT Simulator Tile with Python orchestration.

Integrates LRTTController with aihwkit tile system, providing the same interface
as rpucuda_lrtt_transfer_device.cu but implemented entirely in Python.
"""

from typing import Optional, Tuple, Any, Dict
import torch
from torch import Tensor
from torch.nn import Module

from aihwkit.simulator.tiles.base import SimulatorTileWrapper, SimulatorTile
from aihwkit.simulator.tiles.analog import AnalogTile
from aihwkit.simulator.tiles.lrtt_controller import LRTTController
from aihwkit.simulator.parameters.base import RPUConfigGeneric
from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.configs.configs import SingleRPUConfig, UnitCellRPUConfig
# LRTTTransferCompound removed - using Python-level LRTT instead
from aihwkit.exceptions import ConfigError, TileError


class LRTTSimulatorTile(SimulatorTile, Module):
    """LR-TT simulator tile with Python orchestration.
    
    Implements the exact semantics of rpucuda_lrtt_transfer_device.cu using
    3 analog tiles (fastA, fastB, visible) orchestrated by LRTTController.
    
    Architecture:
    - tile_a (fastA): A matrix [d_size, rank] for LoRA left factor
    - tile_b (fastB): B matrix [rank, x_size] for LoRA right factor  
    - tile_c (visible): Main weights [d_size, x_size]
    
    Key features:
    - Rank-restricted LoRA-style updates with projections
    - Pulsed transfer with outer-product accumulation
    - Forward injection: y = C·x + α·A·(B·x)
    - Full scheduling and BL management support
    """
    
    supports_indexed: bool = False
    
    def __init__(
        self, 
        d_size: int,  # out_features from AnalogLinear
        x_size: int,  # in_features from AnalogLinear 
        rpu_config: UnitCellRPUConfig, 
        bias: bool = False,  # Added for compatibility
        dtype: Optional[RPUDataType] = None,  # Optional, get from config if not provided
        **kwargs  # Ignore extra kwargs for compatibility
    ):
        """Initialize LRTT simulator tile.
        
        Args:
            d_size: Output size (out_features from AnalogLinear)
            x_size: Input size (in_features from AnalogLinear)
            rpu_config: Must contain LRTTTransferCompound device
            dtype: Data type for tiles
            bias: Whether to use bias (currently not supported in LRTT)
        """
        Module.__init__(self)
        
        self.x_size = x_size
        self.d_size = d_size
        # Get dtype from config if not provided
        if dtype is None:
            from aihwkit.simulator.parameters.enums import RPUDataType
            dtype = RPUDataType.FLOAT  # Default to float32
        self.dtype = dtype
        self.bias = bias  # Store but don't use for now
        
        # Validate configuration - check for PythonLRTTDevice
        from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
        if not isinstance(getattr(rpu_config, 'device', None), PythonLRTTDevice):
            raise ConfigError("LRTTSimulatorTile requires a PythonLRTTDevice configuration")
            
        self.lrtt_config = rpu_config.device
        self.rank = self.lrtt_config.rank
        
        if self.rank <= 0 or self.rank > min(d_size, x_size):
            raise ConfigError(f"Invalid rank {self.rank} for dimensions {d_size}×{x_size}")
            
        # Extract LRTT parameters
        self.transfer_lr = getattr(self.lrtt_config, 'transfer_lr', 1.0)
        self.transfer_every = getattr(self.lrtt_config, 'transfer_every', 32)
        self.units_in_mbatch = getattr(self.lrtt_config, 'units_in_mbatch', False)
        self.lora_alpha = getattr(self.lrtt_config, 'lora_alpha', 1.0)
        self.reinit_gain = getattr(self.lrtt_config, 'reinit_gain', 0.1)
        self.correct_gradient_magnitudes = getattr(self.lrtt_config, 'correct_gradient_magnitudes', False)
        # Note: forward_inject flag is managed by the controller only
        self.rank_chunk = getattr(self.lrtt_config, 'rank_chunk', None)
        
        # Create individual tiles using unit cell devices
        unit_devices = self.lrtt_config.unit_cell_devices
        if len(unit_devices) < 3:
            # Replicate first device if not enough specified
            while len(unit_devices) < 3:
                unit_devices.append(unit_devices[0])
                
        # Tile A: fastA [d_size, rank]
        rpu_config_a = SingleRPUConfig(
            device=unit_devices[0],
            forward=rpu_config.forward,
            backward=rpu_config.backward,
            update=rpu_config.update,
            tile_class=AnalogTile
        )
        self.tile_a = rpu_config_a.tile_class(d_size, self.rank, rpu_config_a)
        
        # Tile B: fastB [rank, x_size] (only rank rows needed for LoRA)
        rpu_config_b = SingleRPUConfig(
            device=unit_devices[1], 
            forward=rpu_config.forward,
            backward=rpu_config.backward,
            update=rpu_config.update,
            tile_class=AnalogTile
        )
        self.tile_b = rpu_config_b.tile_class(self.rank, x_size, rpu_config_b)
        
        # Tile C: visible [d_size, x_size]  
        rpu_config_c = SingleRPUConfig(
            device=unit_devices[2],
            forward=rpu_config.forward,
            backward=rpu_config.backward, 
            update=rpu_config.update,
            tile_class=AnalogTile
        )
        self.tile_c = rpu_config_c.tile_class(d_size, x_size, rpu_config_c)
        
        # Create LRTT controller
        self.controller = LRTTController(
            tile_a=self.tile_a,
            tile_b=self.tile_b,
            tile_c=self.tile_c,
            d_size=d_size,
            x_size=x_size,
            rank=self.rank,
            transfer_lr=self.transfer_lr,
            transfer_lr_scale=getattr(self.lrtt_config, 'transfer_lr_scale', 'sqrt_rank'),
            transfer_every=self.transfer_every,
            units_in_mbatch=self.units_in_mbatch,
            lora_alpha=self.lora_alpha,
            reinit_gain=self.reinit_gain,
            reinit_mode=getattr(self.lrtt_config, 'reinit_mode', 'standard'),
            decay_factor=getattr(self.lrtt_config, 'decay_factor', 0.9),
            correct_gradient_magnitudes=self.correct_gradient_magnitudes,
            rank_chunk=self.rank_chunk,
            forward_inject=getattr(self.lrtt_config, 'forward_inject', True),
            use_onehot=getattr(self.lrtt_config, 'use_onehot', True),
            use_sigma_delta=getattr(self.lrtt_config, 'use_sigma_delta', True),
        )

        # Set reconstruction parameters from config (for forward_inject=False mode)
        self.controller.recon_lambda_a = getattr(self.lrtt_config, 'recon_lambda_a', 1e-3)
        self.controller.recon_lambda_b = getattr(self.lrtt_config, 'recon_lambda_b', 1e-3)
        self.controller.recon_use_scalar_stabilizer = getattr(self.lrtt_config, 'recon_use_scalar_stabilizer', False)
        self.controller.recon_use_exact_gram = getattr(self.lrtt_config, 'recon_use_exact_gram', False)
        self.controller.recon_ema_beta = getattr(self.lrtt_config, 'recon_ema_beta', 0.9)
        self.controller.recon_lr_scale = getattr(self.lrtt_config, 'recon_lr_scale', 1.0)
        self.controller.recon_clip_norm = getattr(self.lrtt_config, 'recon_clip_norm', 10.0)
        self.controller.recon_use_clip_norm = getattr(self.lrtt_config, 'recon_use_clip_norm', False)

        # Initialize LRTT weights
        self.controller.reinit()
        
        # Hook individual tile updates to route through controller
        self._hook_tile_updates()
        
    def _hook_tile_updates(self) -> None:
        """Hook individual tile update methods to route through controller.
        
        When AnalogSGD calls update on individual tiles, we intercept
        and route through the controller for proper LRTT updates.
        """
        # Store original update methods
        self.tile_a._orig_update = self.tile_a.update
        self.tile_b._orig_update = self.tile_b.update
        self.tile_c._orig_update = self.tile_c.update
        
        # Track if we've already handled this batch
        self._update_handled = False
        
        def hooked_update(tile_name):
            def update_wrapper(x_input, d_input, *args, **kwargs):
                # Prevent double updates - only handle once per batch
                if self._update_handled:
                    return None
                    
                # Tile C gets the full inputs, use those for LRTT update
                if tile_name == 'tile_c':
                    self._update_handled = True  # Mark as handled
                    
                    # Get learning rate
                    lr = self.tile_c.get_learning_rate()
                    
                    # Route through controller for proper LRTT update
                    self.controller.ab_weight_update(
                        x=x_input,  # This is the full [batch, x_size] input
                        d=d_input,  # This is the full [batch, d_size] gradient
                        lr=lr,
                        in_trans=False,
                        out_trans=False
                    )
                    
                    # Check for transfer
                    if self.controller.should_transfer():
                        self.controller.ab_weight_transfer(
                            use_onehot=self.controller.use_onehot,
                            use_sigma_delta=self.controller.use_sigma_delta
                        )
                    
                # Don't call original update on any tile - LRTT handles all updates
                return None
            return update_wrapper
        
        # Replace update methods
        self.tile_a.update = hooked_update('tile_a')
        self.tile_b.update = hooked_update('tile_b')
        self.tile_c.update = hooked_update('tile_c')
        
    def _reset_update_flag(self) -> None:
        """Reset the update handled flag for next batch."""
        self._update_handled = False
    
    def get_tensor_view(self, ndim: int, dim: Optional[int] = None) -> tuple:
        """Return the tensor view for ndim vector at dim.
        
        Args:
            ndim: number of dimensions
            dim: the dimension to set to -1
            
        Returns:
            Tuple of ones with the `dim` index sets to -1
        """
        if dim is None:
            dim = 0 if getattr(self, 'out_trans', False) else ndim - 1
        tensor_view = [1] * ndim
        tensor_view[dim] = -1
        return tuple(tensor_view)
        
    def forward(
        self,
        x_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
        tensor_view: Optional[Tuple] = None,  # Added for array compatibility
    ) -> Tensor:
        """Forward pass with LRTT forward injection.
        
        Args:
            x_input: Input tensor 
            bias: Bias flag (not supported)
            in_trans: Input transposed
            out_trans: Output transposed
            is_test: Test mode (affects forward injection)
            non_blocking: Non-blocking flag
            
        Returns:
            Output tensor
        """
        # Reset update flag for this forward pass
        self._reset_update_flag()
        
        if bias:
            raise TileError("LRTT does not support bias")
            
        # Store input for potential local A,B update when forward_inject=False
        self._last_x_input = x_input.detach().clone()
            
        # Single source of truth: Use controller's forward_inject_enabled flag only
        # This avoids confusion from multiple forward_inject flags
        if self.controller.forward_inject_enabled:
            return self.controller.forward_inject(x_input, out_trans=out_trans, in_trans=in_trans)
        else:
            # Fallback to visible-only forward when disabled
            # Handle transpose manually since AnalogTile doesn't support transpose flags
            x = x_input.t() if in_trans else x_input
            y = self.tile_c.forward(x)
            return y.t() if out_trans else y
    
    def backward(
        self,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> Tensor:
        """LRTT backward pass using only analog tile operations.
        
        Computes: 
        - If forward_inject_enabled: x_grad = C^T @ d + α * B^T @ (A^T @ d)
        - If forward_inject_disabled: x_grad = C^T @ d (for upstream), but store gradients for A,B local update
        All operations use tile.backward() to ensure proper analog constraints.
        """
        if bias:
            raise TileError("LRTT does not support bias")
            
        # 1) Input to batch-first
        d_bf = d_input.t() if in_trans else d_input  # [batch, d_size]
        
        # 2) Always compute C gradient for upstream propagation
        xg_c = self.tile_c.backward(d_bf)   # [batch, x_size]
        
        if self.controller.forward_inject_enabled:
            # Full LRTT backward: C^T·d + α * B^T·(A^T·d)
            da = self.tile_a.backward(d_bf)     # [batch, rank]
            xg_ab = self.tile_b.backward(da)    # [batch, x_size]
            x_grad = xg_c + self.lora_alpha * xg_ab
        else:
            # forward_inject=False: Upstream gets C-only gradient
            # But store gradients for local A,B update (unless skipped by child class)
            da = self.tile_a.backward(d_bf)     # [batch, rank] - for local update
            xg_ab = self.tile_b.backward(da)    # [batch, x_size] - for local update
            
            # Store for local A,B update during update() call (unless child class handles it)
            if not hasattr(self, '_skip_gradient_storage'):
                self._stored_d_input = d_input.detach().clone()
                self._stored_x_input = getattr(self, '_last_x_input', None)
            
            # Return only C gradient for upstream propagation
            x_grad = xg_c
        
        # 3) Output transpose
        return x_grad.t() if out_trans else x_grad
    
    def update(
        self,
        x_input: Tensor,
        d_input: Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> None:
        """LRTT update: A/B LoRA updates + periodic transfer.
        
        Args:
            x_input: Input tensor
            d_input: Error tensor
            bias: Bias flag (not supported)
            in_trans: Input transposed
            out_trans: Output transposed  
            non_blocking: Non-blocking flag
        """
        if bias:
            raise TileError("LRTT does not support bias")
            
        # Prevent double updates
        if self._update_handled:
            return None
        self._update_handled = True
        
        # Get current learning rate (assuming all tiles have same LR)
        lr = self.get_learning_rate()
        
        # For forward_inject=False, use stored gradients for local A,B update
        if not self.controller.forward_inject_enabled and hasattr(self, '_stored_d_input') and hasattr(self, '_stored_x_input'):
            if self._stored_x_input is not None and self._stored_d_input is not None:
                # Use stored inputs/gradients for A,B local update
                update_x = self._stored_x_input
                update_d = self._stored_d_input
                
                # Clear stored gradients after use
                delattr(self, '_stored_d_input')
                delattr(self, '_stored_x_input')
            else:
                # Fallback to current inputs if stored inputs are not available
                update_x = x_input
                update_d = d_input
        else:
            # Normal case: use current inputs
            update_x = x_input
            update_d = d_input
        
        # Perform A/B LoRA-style updates with projections
        self.controller.ab_weight_update(
            x=update_x,
            d=update_d, 
            lr=lr,
            in_trans=in_trans,
            out_trans=out_trans
        )
        
        # Check for transfer
        if self.controller.should_transfer():
            self.controller.ab_weight_transfer(
                use_onehot=self.controller.use_onehot,
                use_sigma_delta=self.controller.use_sigma_delta
            )

    def get_weights(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get visible weights (source of truth), matching CUDA semantics.
        
        Returns:
            Tuple of (visible_weights, None)
        """
        # CRITICAL: Return visible weights only, not effective weights
        # This matches CUDA where visible (C) is the source of truth
        return self.tile_c.get_weights()
    
    def get_effective_weights(self) -> Tuple[Tensor, Optional[Tensor]]:
        """Get effective LRTT weights: W_eff = W_visible + α * A @ B.
        
        This is a separate method for when effective weights are explicitly needed.
        
        Returns:
            Tuple of (effective_weights, None)
        """
        from aihwkit.linalg.lrtt_kernels import compose_lrtt_weights
        
        # Get individual component weights
        visible_weights = self.tile_c.get_weights()[0]  # [d_size, x_size]
        A_weights = self.tile_a.get_weights()[0]        # [d_size, rank]
        B_weights = self.tile_b.get_weights()[0]        # [rank, x_size]
        
        # Compose effective weights
        W_eff = compose_lrtt_weights(
            visible_weights, A_weights, B_weights, 
            self.lora_alpha, self.rank
        )
        
        return W_eff, None
        
    def set_weights(self, weight: Tensor, bias: Optional[Tensor] = None) -> None:
        """Set visible weights (source of truth), A/B remain unchanged.
        
        This matches CUDA where visible weights are the primary storage.
        
        Args:
            weight: Weight tensor [d_size, x_size]
            bias: Bias tensor (not supported)
        """
        if bias is not None:
            raise TileError("LRTT does not support bias")
            
        # Set visible weights only, preserve A/B state
        self.tile_c.set_weights(weight, None)
        
    def get_lrtt_component_weights(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Get individual LRTT component weights.
        
        Returns:
            Tuple of (visible_weights, A_weights, B_lr_weights)
        """
        visible_weights = self.tile_c.get_weights()[0]  # [d_size, x_size]
        A_weights = self.tile_a.get_weights()[0]        # [d_size, rank]
        B_lr = self.tile_b.get_weights()[0]             # [rank, x_size]
        
        return visible_weights, A_weights, B_lr
        
    def set_lrtt_component_weights(
        self, 
        visible: Tensor, 
        A: Tensor, 
        B_lr: Tensor
    ) -> None:
        """Set individual LRTT component weights.
        
        Args:
            visible: Visible weights [d_size, x_size]
            A: A weights [d_size, rank]
            B_lr: B weights [rank, x_size] (will be placed in first rank rows)
        """
        # Set visible weights
        self.tile_c.set_weights(visible, None)
        
        # Set A weights
        self.tile_a.set_weights(A, None)
        
        # Set B weights (B tile is already [rank, x_size], no expansion needed)
        self.tile_b.set_weights(B_lr, None)
        
    def get_x_size(self) -> int:
        """Get input size."""
        return self.x_size
        
    def get_d_size(self) -> int:
        """Get output size."""
        return self.d_size
        
    def get_learning_rate(self) -> float:
        """Get learning rate from visible tile."""
        return self.tile_c.get_learning_rate()
        
    def set_learning_rate(self, learning_rate: float) -> None:
        """Set learning rate for all tiles."""
        self.tile_a.set_learning_rate(learning_rate)
        self.tile_b.set_learning_rate(learning_rate)
        self.tile_c.set_learning_rate(learning_rate)
        
    def get_hidden_parameters(self) -> Tensor:
        """Get concatenated hidden parameters from all tiles."""
        params_a = self.tile_a.get_hidden_parameters()
        params_b = self.tile_b.get_hidden_parameters() 
        params_c = self.tile_c.get_hidden_parameters()
        
        return torch.cat([params_a, params_b, params_c])
        
    def set_hidden_parameters(self, data: Tensor) -> None:
        """Set hidden parameters for all tiles."""
        # Split data based on tile parameter counts
        params_a = self.tile_a.get_hidden_parameters()
        params_b = self.tile_b.get_hidden_parameters()
        params_c = self.tile_c.get_hidden_parameters()
        
        size_a = params_a.numel()
        size_b = params_b.numel()
        size_c = params_c.numel()
        
        if data.numel() != size_a + size_b + size_c:
            raise TileError(f"Hidden parameter size mismatch: expected {size_a + size_b + size_c}, got {data.numel()}")
            
        self.tile_a.set_hidden_parameters(data[:size_a])
        self.tile_b.set_hidden_parameters(data[size_a:size_a + size_b])
        self.tile_c.set_hidden_parameters(data[size_a + size_b:])
        
    def decay_weights(self, alpha: float = 0.0) -> None:
        """Apply weight decay to all tiles."""
        self.tile_a.decay_weights(alpha)
        self.tile_b.decay_weights(alpha) 
        self.tile_c.decay_weights(alpha)
        
    def diffuse_weights(self, alpha: float = 0.0) -> None:
        """Apply weight diffusion to all tiles."""
        self.tile_a.diffuse_weights(alpha)
        self.tile_b.diffuse_weights(alpha)
        self.tile_c.diffuse_weights(alpha)

    def post_update_step(self) -> None:
        """Operators that need to be called once per mini-batch.

        Called by AnalogSGD after weight updates. Applies decay and diffusion
        to all three tiles (A, B, C) if enabled by their device configs.

        Note:
            - C tile (6T1C): Has lifetime parameter for retention decay
            - A/B tiles: May be ideal devices (no decay) or capacitor-based
        """
        # Apply decay to each tile based on its device config
        # tile_a
        if hasattr(self.tile_a, 'rpu_config') and self.tile_a.rpu_config.device.requires_decay():
            self.tile_a.decay_weights()
        # tile_b
        if hasattr(self.tile_b, 'rpu_config') and self.tile_b.rpu_config.device.requires_decay():
            self.tile_b.decay_weights()
        # tile_c (6T1C - typically has decay)
        if hasattr(self.tile_c, 'rpu_config') and self.tile_c.rpu_config.device.requires_decay():
            self.tile_c.decay_weights()

        # Apply diffusion if needed
        if hasattr(self.tile_a, 'rpu_config') and self.tile_a.rpu_config.device.requires_diffusion():
            self.tile_a.diffuse_weights()
        if hasattr(self.tile_b, 'rpu_config') and self.tile_b.rpu_config.device.requires_diffusion():
            self.tile_b.diffuse_weights()
        if hasattr(self.tile_c, 'rpu_config') and self.tile_c.rpu_config.device.requires_diffusion():
            self.tile_c.diffuse_weights()

    def clip_weights(self, clip_type: str = "", sigma: float = 0.0) -> None:
        """Apply weight clipping to all tiles.""" 
        self.tile_a.clip_weights(clip_type, sigma)
        self.tile_b.clip_weights(clip_type, sigma)
        self.tile_c.clip_weights(clip_type, sigma)
        
    def reset_columns(self, start_column_idx: int = 0, num_columns: int = 1, sigma: float = 1.0) -> None:
        """Reset columns in visible tile."""
        # Only reset visible tile columns (A/B managed by controller)
        self.tile_c.reset_columns(start_column_idx, num_columns, sigma)
        
    def get_brief_info(self) -> str:
        """Get brief tile information."""
        return f"LRTTSimulatorTile({self.d_size}, {self.x_size}, rank={self.rank})"
        
    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return f"d_size={self.d_size}, x_size={self.x_size}, rank={self.rank}, " \
               f"transfer_every={self.transfer_every}, lora_alpha={self.lora_alpha}"
               
    def get_controller_state(self) -> Dict[str, Any]:
        """Get LRTT controller state for debugging/monitoring."""
        return self.controller.get_state_dict()
        
    def manual_transfer(self) -> None:
        """Manually trigger A⊗B -> visible transfer (for testing)."""
        self.controller.ab_weight_transfer(
            use_onehot=self.controller.use_onehot,
            use_sigma_delta=self.controller.use_sigma_delta
        )
    
    def _infer_device_from_self(self) -> torch.device:
        """Infer device from submodule parameters/buffers."""
        # Check parameters
        for p in self.parameters(recurse=True):
            return p.device
        # Check buffers
        for b in self.buffers(recurse=True):
            return b.device
        # Default to CUDA if available
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def to(self, *args, **kwargs):
        """Move to device and synchronize controller."""
        super().to(*args, **kwargs)
        
        # Extract device from arguments
        device = kwargs.get('device', None)
        if device is None and len(args) > 0:
            if isinstance(args[0], torch.device):
                device = args[0]
            elif isinstance(args[0], str):
                device = torch.device(args[0])
        
        # If still no device, infer from self
        if device is None:
            device = self._infer_device_from_self()
        
        # Synchronize controller device
        if hasattr(self, 'controller'):
            self.controller.set_device(device)
            
        return self
    
    def cuda(self, device=None):
        """Move to CUDA and synchronize controller."""
        super().cuda(device=device)
        
        # Determine CUDA device
        if device is None:
            cuda_device = torch.device('cuda')
        else:
            cuda_device = torch.device(f'cuda:{device}')
        
        # Synchronize controller
        if hasattr(self, 'controller'):
            self.controller.set_device(cuda_device)
            
        return self
    
    def cpu(self):
        """Move to CPU and synchronize controller."""
        super().cpu()
        
        # Synchronize controller
        if hasattr(self, 'controller'):
            self.controller.set_device(torch.device('cpu'))

        return self

    def _apply(self, fn):
        """Override _apply to synchronize controller device after parameter moves.

        This is called by Module.cuda(), Module.to(), etc. when moving parameters.
        We need to update the controller device after all child modules are moved.
        """
        # Call parent _apply first to move all parameters/buffers
        result = super()._apply(fn)

        # After all moves, synchronize controller device with the actual tile device
        if hasattr(self, 'controller'):
            # Infer device from the moved parameters
            device = self._infer_device_from_self()
            self.controller.set_device(device)

        return result