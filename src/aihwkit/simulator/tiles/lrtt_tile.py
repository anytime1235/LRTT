# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""LR-TT Simulator Tile with Python orchestration.

Integrates LRTTController with aihwkit tile system, providing the same interface
as rpucuda_lrtt_transfer_device.cu but implemented entirely in Python.
"""

from typing import Optional, Tuple, Any, Dict
import math
import torch
from torch import Tensor
from torch.nn import Module

from aihwkit.simulator.tiles.base import SimulatorTileWrapper, SimulatorTile
from aihwkit.simulator.tiles.analog import AnalogTile
from aihwkit.simulator.tiles.floating_point import FloatingPointTile
from aihwkit.simulator.tiles.lrtt_controller import LRTTController
from aihwkit.simulator.configs.devices import FloatingPointDevice
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
        self.reinit_gain = getattr(self.lrtt_config, "reinit_gain", 1.0)
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
            transfer_method = getattr(self.lrtt_config, 'transfer_method', 'direct')
            c_needs_nwd = (tile_type == "c" and transfer_method == "set")
            has_tile_specific = any(x is not None for x in [a_x, a_d, b_x, b_d, c_bl])

            if not has_tile_specific and not c_needs_nwd:
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

                # For "set" transfer: use NoneWithDevice for deterministic FP
                # update with weight clipping (must be set at construction time)
                if c_needs_nwd:
                    from aihwkit.simulator.parameters.enums import PulseType
                    update_copy.pulse_type = PulseType.NONE_WITH_DEVICE

            return update_copy

        # Tile A/B creation: use mapping_ab from device config
        from aihwkit.simulator.parameters.mapping import MappingParameter
        tile_class_a = get_tile_class(unit_devices[0])
        update_a = create_update_params(rpu_config.update, "a")
        tile_class_b = get_tile_class(unit_devices[1])
        update_b = create_update_params(rpu_config.update, "b")

        mapping_ab = getattr(self.lrtt_config, 'mapping_ab', MappingParameter())

        # A/B tile IO: optionally remove ADC/DAC between B and A projections
        from copy import deepcopy
        no_adc_ab_proj = getattr(self.lrtt_config, 'no_adc_ab_projection', False)

        backward_a = rpu_config.backward
        if no_adc_ab_proj:
            backward_a = deepcopy(rpu_config.backward)
            backward_a.out_res = -1  # Remove ADC at A backward output (between A and B)

        rpu_config_a = SingleRPUConfig(
            device=unit_devices[0],
            forward=rpu_config.forward,
            backward=backward_a,
            update=update_a,
            tile_class=tile_class_a,
            mapping=mapping_ab,
        )
        self.tile_a = rpu_config_a.tile_class(d_size, self.rank, rpu_config_a)

        forward_b = rpu_config.forward
        if no_adc_ab_proj:
            forward_b = deepcopy(rpu_config.forward)
            forward_b.out_res = -1  # Remove ADC at B forward output

        rpu_config_b = SingleRPUConfig(
            device=unit_devices[1],
            forward=forward_b,
            backward=rpu_config.backward,
            update=update_b,
            tile_class=tile_class_b,
            mapping=mapping_ab,
        )
        self.tile_b = rpu_config_b.tile_class(self.rank, x_size, rpu_config_b)

        # Tile C: visible [d_size, x_size] - use mapping_c from device config
        tile_class_c = get_tile_class(unit_devices[2])
        update_c = create_update_params(rpu_config.update, "c")
        mapping_c = getattr(self.lrtt_config, 'mapping_c', MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=True,
            out_scaling_columnwise=True,
        ))
        rpu_config_c = SingleRPUConfig(
            device=unit_devices[2],
            forward=rpu_config.forward,
            backward=rpu_config.backward,
            update=update_c,
            tile_class=tile_class_c,
            mapping=mapping_c,
        )
        # Pass bias to tile_c for digital_bias support
        # When bias=True, tile_c will have digital_bias=True and create self.bias Parameter
        self.tile_c = rpu_config_c.tile_class(
            d_size, x_size, rpu_config_c, bias=self.bias
        )

        # Freeze/unfreeze bias in tile_c based on config
        # (C tile weights are already untrainable via update hooks)
        # Note: out_scaling trainability is controlled by mapping_c.learn_out_scaling
        _train_bias = getattr(self.lrtt_config, 'train_c_bias', False)
        for name, param in self.tile_c.named_parameters():
            if 'bias' in name:
                param.requires_grad = _train_bias

        # Create LRTT controller with all parameters
        self.controller = LRTTController(
            tile_a=self.tile_a,
            tile_b=self.tile_b,
            tile_c=self.tile_c,
            d_size=d_size,
            x_size=x_size,
            rank=self.rank,
            transfer_lr=self.transfer_lr,
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
            fast_lr=getattr(self.lrtt_config, "fast_lr", 1.0),
            scale_transfer_lr=getattr(self.lrtt_config, "scale_transfer_lr", True),
            transfer_fast_lr_ref=getattr(self.lrtt_config, "transfer_fast_lr_ref", "geomean"),
            rank_chunk=self.rank_chunk,
            forward_inject=getattr(self.lrtt_config, "forward_inject", False),
            dynamic_te=getattr(self.lrtt_config, "dynamic_te", False),
            dynamic_te_power=getattr(self.lrtt_config, "dynamic_te_power", 1.0),
            dynamic_te_min=getattr(self.lrtt_config, "dynamic_te_min", None),
            dynamic_te_max=getattr(self.lrtt_config, "dynamic_te_max", None),
            te_warmup_schedule=getattr(self.lrtt_config, "te_warmup_schedule", None),
            te_warmup_steps=getattr(self.lrtt_config, "te_warmup_steps", 0),
            num_reads=getattr(self.lrtt_config, "num_reads", 1),
            multi_read_mode=getattr(self.lrtt_config, "multi_read_mode", "average"),
            update_mode=getattr(self.lrtt_config, "update_mode", "lora"),
            transfer_method=getattr(self.lrtt_config, "transfer_method", "onehot"),
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

        # Auto-scale settings (from post_init)
        self.controller.auto_scale_mode = post_init.get("auto_scale_mode", "none")
        self.controller.auto_momentum = post_init.get("auto_momentum", 0.99)
        # EMA is lazily initialized by controller._lazy_init_ema(x.device) on first update

        # Granularity for separate auto-scale mode
        if self.controller.auto_scale_mode == "separate":
            dw_min_a = getattr(self.lrtt_config.unit_cell_devices[0], "dw_min", 1.0)
            dw_min_b = getattr(self.lrtt_config.unit_cell_devices[1], "dw_min", 1.0)
            desired_bl = rpu_config.update.desired_bl
            self.controller.gran_a = desired_bl * dw_min_a
            self.controller.gran_b = desired_bl * dw_min_b

        # Initialize LRTT weights
        self.controller.reinit()

        # Hook individual tile updates to route through controller
        self._hook_tile_updates()

    def _hook_tile_updates(self) -> None:
        """Hook individual tile update methods to route through controller.

        When AnalogSGD calls update on individual tiles, we intercept
        and route through the controller for proper LRTT updates.

        Two modes depending on forward_inject_enabled:

        forward_inject=False (original):
            All 3 tiles hooked. tile_a/tile_b → no-op, tile_c triggers
            ab_weight_update() which manually computes XB=B·x, DA=A^T·d
            and calls tile_a/tile_b._orig_update with lr_eff = lr*α.

        forward_inject=True (autograd-driven):
            Autograd already populates tile_a/tile_b AnalogContexts with
            correct LoRA gradient components:
              tile_a: x_input=g=B·x, d_input=α·d
              tile_b: x_input=x,     d_input=α·A^T·d
            However, α is baked into d_input (from chain rule through
            y = y_c + α·y_ab). For faithful stochastic pulse simulation,
            we must remove α from d and put it in lr instead:
              tile_a: orig_update(g, d) with lr = lr_base * α
              tile_b: orig_update(x, A^T·d) with lr = lr_base * α
            tile_c → no-op (C is frozen).
        """
        # Store original update methods
        self.tile_a._orig_update = self.tile_a.update
        self.tile_b._orig_update = self.tile_b.update
        self.tile_c._orig_update = self.tile_c.update

        # Back-references so the optimizer patch can find the controller
        # from any sub-tile's AnalogContext.analog_tile
        self.tile_a._lrtt_controller = self.controller
        self.tile_b._lrtt_controller = self.controller
        self.tile_c._lrtt_controller = self.controller
        self.tile_a._lrtt_tile_name = 'tile_a'
        self.tile_b._lrtt_tile_name = 'tile_b'
        self.tile_c._lrtt_tile_name = 'tile_c'

        # Track if we've already handled this batch
        self._update_handled = False

        if self.controller.forward_inject_enabled:
            self._hook_tile_updates_fi()
        else:
            self._hook_tile_updates_nfi()

    def _hook_tile_updates_fi(self) -> None:
        """Hook setup for forward_inject=True.

        tile_a/tile_b: rescale d_input (remove α), use last_lr_eff from controller,
                       then call orig_update for stochastic pulse update.
        tile_c: no-op (C frozen), handle transfer counter.

        Note: auto_scale_mode != 'none' is not supported with forward_inject=True
        because EMA cannot be updated without _ab_weight_update_lora().
        """
        ctrl = self.controller
        alpha = ctrl.lora_alpha

        # Guard: auto_scale not supported in FI path
        if ctrl.auto_scale_mode != "none":
            raise ValueError(
                f"auto_scale_mode='{ctrl.auto_scale_mode}' is not supported with "
                f"forward_inject=True. Use auto_scale_mode='none' or forward_inject=False."
            )

        def hooked_ab(tile, tile_name):
            def update_wrapper(x_input, d_input, *args, **kwargs):
                # Remove α from gradient, move it to learning rate
                d_rescaled = d_input / alpha

                # Use last_lr_eff_a/b from controller (set by fast_lr in 'none' mode)
                lr_base = tile.get_learning_rate()
                if ctrl._is_hardware_mode():
                    lr_eff = 1.0
                else:
                    lr_eff = ctrl.last_lr_eff_a if tile_name == "tile_a" else ctrl.last_lr_eff_b

                tile.set_learning_rate(lr_eff)
                tile._orig_update(x_input, d_rescaled, *args, **kwargs)
                tile.set_learning_rate(lr_base)

                # Track update count
                if tile_name == "tile_a":
                    ctrl.num_a_updates += 1
                else:
                    ctrl.num_b_updates += 1
                return None
            return update_wrapper

        def hooked_c_noop(x_input, d_input, *args, **kwargs):
            # C is frozen — no weight update
            # Track lr_sgd for transfer LR computation
            ctrl._last_lr_sgd = self.tile_c.get_learning_rate()
            # Handle transfer counter and dynamic TE
            lr = self.tile_c.get_learning_rate()
            ctrl._update_dynamic_te(lr)
            m_batch = x_input.shape[0]
            ctrl._last_m_batch = m_batch
            ctrl.transfer_counter += m_batch
            if ctrl.should_transfer():
                ctrl.ab_weight_transfer()
            return None

        self.tile_a.update = hooked_ab(self.tile_a, "tile_a")
        self.tile_b.update = hooked_ab(self.tile_b, "tile_b")
        self.tile_c.update = hooked_c_noop

    def _hook_tile_updates_nfi(self) -> None:
        """Hook setup for forward_inject=False (original behavior).

        tile_a/tile_b → no-op, tile_c triggers ab_weight_update.
        """
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
                        x=x_input,
                        d=d_input,
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

        # Single source of truth: Use controller's forward_inject_enabled flag only
        # This avoids confusion from multiple forward_inject flags
        if self.controller.forward_inject_enabled:
            return self.controller.forward_inject(
                x_input, out_trans=out_trans, in_trans=in_trans
            )
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
            # forward_inject=False: Upstream gets C-only gradient.
            # A,B updates are handled entirely by ab_weight_update() which does
            # its own tile_a.backward() and tile_b.forward() projections.
            # update() receives x_input/d_input directly — no need to store here.
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

        # Perform A/B LoRA-style updates with projections
        # ab_weight_update does its own tile_b.forward() and tile_a.backward()
        # internally, so x_input and d_input are used directly.
        self.controller.ab_weight_update(
            x=x_input, d=d_input, lr=lr, in_trans=in_trans, out_trans=out_trans
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
