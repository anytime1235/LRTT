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
from aihwkit.simulator.tiles.floating_point import FloatingPointTile
from aihwkit.simulator.tiles.lrtt_controller import LRTTController
from aihwkit.simulator.configs.devices import FloatingPointDevice, SoftBoundsDevice
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
        dtype: Optional[
            RPUDataType
        ] = None,  # Optional, get from config if not provided
        **kwargs,  # Ignore extra kwargs for compatibility
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
        self.bias = bias  # Passed to tile_c for digital_bias support

        # Validate configuration - check for PythonLRTTDevice
        from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

        if not isinstance(getattr(rpu_config, "device", None), PythonLRTTDevice):
            raise ConfigError(
                "LRTTSimulatorTile requires a PythonLRTTDevice configuration"
            )

        self.lrtt_config = rpu_config.device
        self.rank = self.lrtt_config.rank

        if self.rank <= 0 or self.rank > min(d_size, x_size):
            raise ConfigError(
                f"Invalid rank {self.rank} for dimensions {d_size}×{x_size}"
            )

        # Extract LRTT parameters
        self.transfer_lr = getattr(self.lrtt_config, "transfer_lr", 1.0)
        self.transfer_every = getattr(self.lrtt_config, "transfer_every", 32)
        self.units_in_mbatch = getattr(self.lrtt_config, "units_in_mbatch", False)
        self.lora_alpha = getattr(self.lrtt_config, "lora_alpha", 1.0)
        self.reinit_gain = getattr(self.lrtt_config, "reinit_gain", 0.1)
        self.correct_gradient_magnitudes = getattr(
            self.lrtt_config, "correct_gradient_magnitudes", False
        )
        # Note: forward_inject flag is managed by the controller only
        self.rank_chunk = getattr(self.lrtt_config, "rank_chunk", None)

        # Create individual tiles using unit cell devices
        unit_devices = self.lrtt_config.unit_cell_devices
        if len(unit_devices) < 3:
            # Replicate first device if not enough specified
            while len(unit_devices) < 3:
                unit_devices.append(unit_devices[0])

        # Helper function to select tile class based on device type
        def get_tile_class(device):
            if isinstance(device, FloatingPointDevice):
                return FloatingPointTile
            else:
                return AnalogTile

        # Helper function to create UpdateParameters with separate A/B scaling
        def create_update_params(base_update, tile_type):
            """Create UpdateParameters with optional tile-specific scaling.

            Args:
                base_update: Base UpdateParameters from rpu_config
                tile_type: 'a', 'b', or 'c'

            Returns:
                UpdateParameters with tile-specific scaling if configured
            """
            from copy import deepcopy
            from aihwkit.simulator.configs import UpdateParameters

            # Get tile-specific scaling overrides from lrtt_config
            a_x = getattr(self.lrtt_config, "a_x_scaling", None)
            a_d = getattr(self.lrtt_config, "a_d_scaling", None)
            b_x = getattr(self.lrtt_config, "b_x_scaling", None)
            b_d = getattr(self.lrtt_config, "b_d_scaling", None)
            c_bl = getattr(self.lrtt_config, "c_desired_bl", None)

            # Check if any tile-specific config is set
            has_tile_specific = any(x is not None for x in [a_x, a_d, b_x, b_d, c_bl])

            if not has_tile_specific:
                # No tile-specific config, use base update params
                return base_update

            # Create a copy and apply tile-specific overrides
            # We need to copy all fields manually
            update_copy = UpdateParameters(
                desired_bl=base_update.desired_bl,
                pulse_type=base_update.pulse_type,
                use_manual_scaling=base_update.use_manual_scaling,
                manual_x_scaling=base_update.manual_x_scaling,
                manual_d_scaling=base_update.manual_d_scaling,
                update_bl_management=base_update.update_bl_management,
                update_management=base_update.update_management,
            )

            # Apply tile-specific overrides
            if tile_type == "a":
                if a_x is not None:
                    update_copy.manual_x_scaling = a_x
                if a_d is not None:
                    update_copy.manual_d_scaling = a_d
            elif tile_type == "b":
                if b_x is not None:
                    update_copy.manual_x_scaling = b_x
                if b_d is not None:
                    update_copy.manual_d_scaling = b_d
            elif tile_type == "c":
                # C tile: apply separate BL for transfer
                if c_bl is not None:
                    update_copy.desired_bl = c_bl
                # C tile uses set_weights (not pulse update), but needs valid scaling
                # values for PulsedDevice config validation
                update_copy.use_manual_scaling = False
                if update_copy.manual_x_scaling is None:
                    update_copy.manual_x_scaling = 1.0
                if update_copy.manual_d_scaling is None:
                    update_copy.manual_d_scaling = 1.0

            return update_copy

        # A/B tile IO: perfect (no DAC/ADC) if ab_io_perfect is set
        ab_io_perfect = getattr(self.lrtt_config, 'ab_io_perfect', False)
        if ab_io_perfect:
            from aihwkit.simulator.parameters.io import IOParameters
            ab_forward = IOParameters(is_perfect=True)
            ab_backward = IOParameters(is_perfect=True)
        else:
            ab_forward = rpu_config.forward
            ab_backward = rpu_config.backward

        # Tile A: fastA [d_size, rank]
        tile_class_a = get_tile_class(unit_devices[0])
        update_a = create_update_params(rpu_config.update, "a")
        rpu_config_a = SingleRPUConfig(
            device=unit_devices[0],
            forward=ab_forward,
            backward=ab_backward,
            update=update_a,
            tile_class=tile_class_a,
        )
        self.tile_a = rpu_config_a.tile_class(d_size, self.rank, rpu_config_a)

        # Tile B: fastB [rank, x_size] (only rank rows needed for LoRA)
        tile_class_b = get_tile_class(unit_devices[1])
        update_b = create_update_params(rpu_config.update, "b")
        rpu_config_b = SingleRPUConfig(
            device=unit_devices[1],
            forward=ab_forward,
            backward=ab_backward,
            update=update_b,
            tile_class=tile_class_b,
        )
        self.tile_b = rpu_config_b.tile_class(self.rank, x_size, rpu_config_b)

        # Tile C: visible [d_size, x_size] - always uses SoftBoundsDevice (noise-free)
        c_device = SoftBoundsDevice(
            dw_min=0.001,
            w_max=1.0,
            w_min=-1.0,
            dw_min_dtod=0.0,
            dw_min_std=0.0,
            up_down=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            write_noise_std=0.0,
            mult_noise=True,
        )
        update_c = create_update_params(rpu_config.update, "c")

        # Combined out_scaling: disable tile_c's individual learn_out_scaling
        # so combined_out_scaling_alpha can scale the full output symmetrically
        self._combined_out_scaling_enabled = getattr(
            self.lrtt_config, 'combined_out_scaling', False
        )
        if self._combined_out_scaling_enabled:
            from copy import deepcopy
            mapping_c = deepcopy(rpu_config.mapping)
            mapping_c.learn_out_scaling = False
        else:
            mapping_c = rpu_config.mapping

        rpu_config_c = SingleRPUConfig(
            device=c_device,
            forward=rpu_config.forward,
            backward=rpu_config.backward,
            update=update_c,
            tile_class=AnalogTile,
            mapping=mapping_c,
        )
        # Pass bias to tile_c for digital_bias support
        # When bias=True, tile_c will have digital_bias=True and create self.bias Parameter
        self.tile_c = rpu_config_c.tile_class(
            d_size, x_size, rpu_config_c, bias=self.bias
        )

        # Create LRTT controller with all parameters
        self.controller = LRTTController(
            tile_a=self.tile_a,
            tile_b=self.tile_b,
            tile_c=self.tile_c,
            d_size=d_size,
            x_size=x_size,
            rank=self.rank,
            transfer_lr=self.transfer_lr,
            transfer_lr_scale=getattr(self.lrtt_config, "transfer_lr_scale", 1.0),
            transfer_every=self.transfer_every,
            units_in_mbatch=self.units_in_mbatch,
            lora_alpha=self.lora_alpha,
            reinit_gain=self.reinit_gain,
            reinit_mode=getattr(self.lrtt_config, "reinit_mode", "standard"),
            decay_factor=getattr(self.lrtt_config, "decay_factor", 1.0),
            a_init_mode=getattr(
                self.lrtt_config, "a_init_mode", "zero"
            ),  # A matrix initialization mode
            b_init_mode=getattr(
                self.lrtt_config, "b_init_mode", "kaiming"
            ),  # B matrix initialization mode
            correct_gradient_magnitudes=self.correct_gradient_magnitudes,
            rank_chunk=self.rank_chunk,
            forward_inject=getattr(self.lrtt_config, "forward_inject", False),
            num_reads=getattr(self.lrtt_config, "num_reads", 1),
            multi_read_mode=getattr(self.lrtt_config, "multi_read_mode", "average"),
            update_mode=getattr(self.lrtt_config, "update_mode", "lora"),
            transfer_method=getattr(self.lrtt_config, "transfer_method", "onehot"),
            dynamic_te=getattr(self.lrtt_config, "dynamic_te", False),
            dynamic_te_power=getattr(self.lrtt_config, "dynamic_te_power", 1.0),
            dynamic_te_min=getattr(self.lrtt_config, "dynamic_te_min", None),
            dynamic_te_max=getattr(self.lrtt_config, "dynamic_te_max", None),
            te_warmup_schedule=getattr(self.lrtt_config, "te_warmup_schedule", None),
            te_warmup_steps=getattr(self.lrtt_config, "te_warmup_steps", 0),
        )

        # Apply post-init settings from config._post_init
        kwargs = self.lrtt_config.to_controller_kwargs()
        post_init = kwargs.get("_post_init", {})

        # Transfer mode & calibration
        self.controller.transfer_mode = post_init.get("transfer_mode", "pilot")
        self.controller.transfer_micro_steps = post_init.get("transfer_micro_steps", 1)
        self.controller.transfer_pilot_frac = post_init.get(
            "transfer_pilot_frac", 1.0 / 16.0
        )
        self.controller.sd_quantum = post_init.get("sd_quantum", None)

        # Read noise reduction
        self.controller.read_n_avg = post_init.get("read_n_avg", 1)
        self.controller.differential_read = post_init.get("differential_read", False)

        # AGC settings
        self.controller.agc_enabled = post_init.get("agc_enabled", False)
        self.controller.agc_margin = post_init.get("agc_margin", 0.85)
        self.controller.agc_max_iters = post_init.get("agc_max_iters", 6)

        # Two-amplitude settings
        self.controller.two_amp_enabled = post_init.get("two_amp_enabled", False)
        self.controller.two_amp_ratio = post_init.get("two_amp_ratio", 0.5)

        # Reconstruction parameters (for update_mode='reconstruction')
        self.controller.recon_lambda_a = post_init.get("recon_lambda_a", 1e-3)
        self.controller.recon_lambda_b = post_init.get("recon_lambda_b", 1e-3)
        self.controller.recon_use_scalar_stabilizer = post_init.get(
            "recon_use_scalar_stabilizer", False
        )
        self.controller.recon_use_exact_gram = post_init.get(
            "recon_use_exact_gram", False
        )
        self.controller.recon_exact_gram_every = post_init.get(
            "recon_exact_gram_every", 0
        )
        self.controller.recon_ema_beta = post_init.get("recon_ema_beta", 0.9)
        self.controller.recon_lr_scale = post_init.get("recon_lr_scale", 1.0)
        self.controller.recon_clip_norm = post_init.get("recon_clip_norm", 10.0)
        self.controller.recon_use_clip_norm = post_init.get(
            "recon_use_clip_norm", False
        )

        # Debug logging settings
        self.controller.log_ab_scaling = post_init.get("log_ab_scaling", False)
        self.controller.log_ab_scaling_every = post_init.get("log_ab_scaling_every", 10)

        # Combined out_scaling: shared learnable parameter for full LRTT output
        # y = combined_out_scaling * [C·x + bias + α·A·(B·x)]
        if self._combined_out_scaling_enabled:
            from torch.nn import Parameter
            self.combined_out_scaling_alpha = Parameter(
                torch.ones(d_size, dtype=torch.float32)
            )
        else:
            self.combined_out_scaling_alpha = None

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

        # Warmup: CudaAnalogTile requires a small-batch first update call
        # to initialize internal buffers. Without this, large-batch updates
        # (batch >= 2048) silently produce zero weight changes.
        self._warmup_tile_updates()

        # Track if we've already handled this batch
        self._update_handled = False

        def hooked_update(tile_name):
            def update_wrapper(x_input, d_input, *args, **kwargs):
                # Prevent double updates - only handle once per batch
                if self._update_handled:
                    return None

                # Tile C gets the full inputs, use those for LRTT update
                if tile_name == "tile_c":
                    self._update_handled = True  # Mark as handled

                    # Get learning rate
                    lr = self.tile_c.get_learning_rate()

                    # Route through controller for proper LRTT update
                    self.controller.ab_weight_update(
                        x=x_input,  # This is the full [batch, x_size] input
                        d=d_input,  # This is the full [batch, d_size] gradient
                        lr=lr,
                        in_trans=False,
                        out_trans=False,
                    )

                    # Check for transfer
                    if self.controller.should_transfer():
                        self.controller.ab_weight_transfer()

                # Don't call original update on any tile - LRTT handles all updates
                return None

            return update_wrapper

        # Replace update methods
        self.tile_a.update = hooked_update("tile_a")
        self.tile_b.update = hooked_update("tile_b")
        self.tile_c.update = hooked_update("tile_c")

    def _warmup_tile_updates(self) -> None:
        """Warmup CudaAnalogTile internal buffers with a small-batch update.

        CudaAnalogTile.update() silently fails for batch sizes >= 2048
        on the first call. A single small-batch call initializes internal
        buffers, after which large-batch updates work correctly.
        """
        _warmup_count = 0
        for tile in [self.tile_a, self.tile_b, self.tile_c]:
            w_orig = tile.get_weights()
            weight = w_orig[0]
            d_size, x_size = weight.shape
            device = weight.device

            lr_orig = tile.get_learning_rate()
            tile.set_learning_rate(1e-10)  # Tiny LR to minimize weight perturbation

            x_warmup = torch.randn(1, x_size, device=device)
            d_warmup = torch.randn(1, d_size, device=device)
            tile._orig_update(x_warmup, d_warmup)

            # Restore original weights and learning rate
            tile.set_weights(weight, bias=w_orig[1])
            tile.set_learning_rate(lr_orig)
            _warmup_count += 1

        if _warmup_count > 0 and not hasattr(self, '_warmup_logged'):
            self._warmup_logged = True
            print(f"  [WARMUP] Initialized {_warmup_count} tiles for LRTT update")

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
            dim = 0 if getattr(self, "out_trans", False) else ndim - 1
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

        # Note: bias parameter here is for interface compatibility only.
        # Actual bias is handled by tile_c's digital_bias (set at __init__)
        # tile_c.forward() automatically adds bias when digital_bias=True

        # Store input for potential local A,B update when forward_inject=False
        self._last_x_input = x_input.detach().clone()

        # Single source of truth: Use controller's forward_inject_enabled flag only
        # This avoids confusion from multiple forward_inject flags
        if self.controller.forward_inject_enabled:
            y = self.controller.forward_inject(
                x_input, out_trans=out_trans, in_trans=in_trans
            )
        else:
            # Fallback to visible-only forward when disabled
            # Handle transpose manually since AnalogTile doesn't support transpose flags
            x = x_input.t() if in_trans else x_input
            y = self.tile_c.forward(x)
            if out_trans:
                y = y.t()

        # Apply combined out_scaling (symmetric for both C and LoRA paths)
        if self.combined_out_scaling_alpha is not None:
            tv = self.get_tensor_view(y.dim())
            y = y * self.combined_out_scaling_alpha.view(*tv)

        return y

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

        Note: bias parameter is for interface compatibility only.
        Bias gradients are not computed here (digital_bias is handled separately).
        """
        # 1) Input to batch-first
        d_bf = d_input.t() if in_trans else d_input  # [batch, d_size]

        # 2) Always compute C gradient for upstream propagation
        xg_c = self.tile_c.backward(d_bf)  # [batch, x_size]

        if self.controller.forward_inject_enabled:
            # Full LRTT backward: C^T·d + α * B^T·(A^T·d)
            da = self.tile_a.backward(d_bf)  # [batch, rank]
            xg_ab = self.tile_b.backward(da)  # [batch, x_size]
            x_grad = xg_c + self.lora_alpha * xg_ab
        else:
            # forward_inject=False: Upstream gets C-only gradient
            # But store gradients for local A,B update (unless skipped by child class)
            da = self.tile_a.backward(d_bf)  # [batch, rank] - for local update
            xg_ab = self.tile_b.backward(da)  # [batch, x_size] - for local update

            # Store for local A,B update during update() call (unless child class handles it)
            if not hasattr(self, "_skip_gradient_storage"):
                self._stored_d_input = d_input.detach().clone()
                self._stored_x_input = getattr(self, "_last_x_input", None)

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
            bias: Bias flag (for interface compatibility, bias updates handled by tile_c)
            in_trans: Input transposed
            out_trans: Output transposed
            non_blocking: Non-blocking flag

        Note: bias parameter is for interface compatibility only.
        Digital bias updates are handled automatically by tile_c's optimizer.
        """
        # Prevent double updates
        if self._update_handled:
            return None
        self._update_handled = True

        # Get current learning rate (assuming all tiles have same LR)
        lr = self.get_learning_rate()

        # For forward_inject=False, use stored gradients for local A,B update
        if (
            not self.controller.forward_inject_enabled
            and hasattr(self, "_stored_d_input")
            and hasattr(self, "_stored_x_input")
        ):
            if self._stored_x_input is not None and self._stored_d_input is not None:
                # Use stored inputs/gradients for A,B local update
                update_x = self._stored_x_input
                update_d = self._stored_d_input

                # Clear stored gradients after use
                delattr(self, "_stored_d_input")
                delattr(self, "_stored_x_input")
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
            x=update_x, d=update_d, lr=lr, in_trans=in_trans, out_trans=out_trans
        )

        # Check for transfer
        if self.controller.should_transfer():
            self.controller.ab_weight_transfer()

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
        A_weights = self.tile_a.get_weights()[0]  # [d_size, rank]
        B_weights = self.tile_b.get_weights()[0]  # [rank, x_size]

        # Compose effective weights
        W_eff = compose_lrtt_weights(
            visible_weights, A_weights, B_weights, self.lora_alpha, self.rank
        )

        return W_eff, None

    def set_weights(
        self, weight: Tensor, bias: Optional[Tensor] = None, **kwargs
    ) -> None:
        """Set visible weights (and bias) on tile_c.

        When tile_c is created with bias=True, it supports digital_bias:
        - bias is stored as a PyTorch Parameter in tile_c.bias
        - forward() automatically adds bias to output
        - update() and transfer only affect weights, not bias

        Args:
            weight: Weight tensor [d_size, x_size]
            bias: Bias tensor [d_size] (stored in tile_c.bias if digital_bias=True)
            **kwargs: Additional arguments (passed to tile_c.set_weights)
        """
        # Pass bias to tile_c - it will handle digital_bias internally
        # If tile_c.digital_bias=True, bias is stored in tile_c.bias Parameter
        # If tile_c.digital_bias=False, bias is ignored
        self.tile_c.set_weights(weight, bias, **kwargs)

    def get_lrtt_component_weights(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Get individual LRTT component weights.

        Returns:
            Tuple of (visible_weights, A_weights, B_lr_weights)
        """
        visible_weights = self.tile_c.get_weights()[0]  # [d_size, x_size]
        A_weights = self.tile_a.get_weights()[0]  # [d_size, rank]
        B_lr = self.tile_b.get_weights()[0]  # [rank, x_size]

        return visible_weights, A_weights, B_lr

    def set_lrtt_component_weights(
        self, visible: Tensor, A: Tensor, B_lr: Tensor
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
            raise TileError(
                f"Hidden parameter size mismatch: expected {size_a + size_b + size_c}, got {data.numel()}"
            )

        self.tile_a.set_hidden_parameters(data[:size_a])
        self.tile_b.set_hidden_parameters(data[size_a : size_a + size_b])
        self.tile_c.set_hidden_parameters(data[size_a + size_b :])

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
            - Controller decay_factor is applied every step (if reinit_mode="decay")
        """
        # Apply decay to each tile based on its device config (lifetime decay)
        # tile_a
        if (
            hasattr(self.tile_a, "rpu_config")
            and self.tile_a.rpu_config.device.requires_decay()
        ):
            self.tile_a.decay_weights()
        # tile_b
        if (
            hasattr(self.tile_b, "rpu_config")
            and self.tile_b.rpu_config.device.requires_decay()
        ):
            self.tile_b.decay_weights()
        # tile_c (6T1C - typically has decay)
        if (
            hasattr(self.tile_c, "rpu_config")
            and self.tile_c.rpu_config.device.requires_decay()
        ):
            self.tile_c.decay_weights()

        # Apply controller's decay_factor every step (if reinit_mode="decay")
        self.controller.apply_step_decay()

        # Apply diffusion if needed
        if (
            hasattr(self.tile_a, "rpu_config")
            and self.tile_a.rpu_config.device.requires_diffusion()
        ):
            self.tile_a.diffuse_weights()
        if (
            hasattr(self.tile_b, "rpu_config")
            and self.tile_b.rpu_config.device.requires_diffusion()
        ):
            self.tile_b.diffuse_weights()
        if (
            hasattr(self.tile_c, "rpu_config")
            and self.tile_c.rpu_config.device.requires_diffusion()
        ):
            self.tile_c.diffuse_weights()

        # CRITICAL: Reset sub-tile analog contexts to prevent memory leak
        # Each sub-tile accumulates analog_input/analog_grad_output during backward
        # These lists must be cleared after each optimizer step
        for tile in [self.tile_a, self.tile_b, self.tile_c]:
            if hasattr(tile, "analog_ctx") and tile.analog_ctx is not None:
                tile.analog_ctx.reset()

    def clip_weights(self, clip_type: str = "", sigma: float = 0.0) -> None:
        """Apply weight clipping to all tiles."""
        self.tile_a.clip_weights(clip_type, sigma)
        self.tile_b.clip_weights(clip_type, sigma)
        self.tile_c.clip_weights(clip_type, sigma)

    def reset_columns(
        self, start_column_idx: int = 0, num_columns: int = 1, sigma: float = 1.0
    ) -> None:
        """Reset columns in visible tile."""
        # Only reset visible tile columns (A/B managed by controller)
        self.tile_c.reset_columns(start_column_idx, num_columns, sigma)

    def get_brief_info(self) -> str:
        """Get brief tile information."""
        return f"LRTTSimulatorTile({self.d_size}, {self.x_size}, rank={self.rank})"

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"d_size={self.d_size}, x_size={self.x_size}, rank={self.rank}, "
            f"transfer_every={self.transfer_every}, lora_alpha={self.lora_alpha}"
        )

    def get_controller_state(self) -> Dict[str, Any]:
        """Get LRTT controller state for debugging/monitoring."""
        return self.controller.get_state_dict()

    def manual_transfer(self, method: Optional[str] = None) -> None:
        """Manually trigger A⊗B -> visible transfer (for testing).

        Args:
            method: Transfer method override.
                   "onehot" - One-hot reading (pulsed update)
                   "direct" - Direct weight access (pulsed update)
                   "set" - Exact weight setting (no pulsed update)
                   If None, use controller's transfer_method setting.
        """
        self.controller.ab_weight_transfer(method=method)

    def _infer_device_from_self(self) -> torch.device:
        """Infer device from submodule parameters/buffers."""
        # Check parameters
        for p in self.parameters(recurse=True):
            return p.device
        # Check buffers
        for b in self.buffers(recurse=True):
            return b.device
        # Default to CUDA if available
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to(self, *args, **kwargs):
        """Move to device and synchronize controller."""
        super().to(*args, **kwargs)

        # Extract device from arguments
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            if isinstance(args[0], torch.device):
                device = args[0]
            elif isinstance(args[0], str):
                device = torch.device(args[0])

        # If still no device, infer from self
        if device is None:
            device = self._infer_device_from_self()

        # Synchronize controller device
        if hasattr(self, "controller"):
            self.controller.set_device(device)

        return self

    def cuda(self, device=None):
        """Move to CUDA and synchronize controller."""
        super().cuda(device=device)

        # Determine CUDA device
        if device is None:
            cuda_device = torch.device("cuda")
        else:
            cuda_device = torch.device(f"cuda:{device}")

        # Synchronize controller
        if hasattr(self, "controller"):
            self.controller.set_device(cuda_device)

        return self

    def cpu(self):
        """Move to CPU and synchronize controller."""
        super().cpu()

        # Synchronize controller
        if hasattr(self, "controller"):
            self.controller.set_device(torch.device("cpu"))

        return self
