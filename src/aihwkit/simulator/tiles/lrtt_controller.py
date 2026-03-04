# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""LR-TT Controller: Pure Python orchestrator for 3-tile LRTT (fastA, fastB, visible).

Implements the exact semantics from rpucuda_lrtt_transfer_device.cu as a pure Python
orchestrator on top of aihwkit tiles. Operates on A, B, visible (C) tile stack with:
- Rank-restricted LoRA-style updates
- Pulsed transfer with outer-product accumulation
- Forward injection with W_eff composition
- Full BL-management and scheduling support
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Dict, Any
import math

from aihwkit.simulator.tiles.analog import AnalogTileWithoutPeriphery
from aihwkit.simulator.parameters.enums import PulseType


class LRTTController:
    """LR-TT controller orchestrating 3 analog tiles: fastA, fastB, visible (C).

    Replicates rpucuda_lrtt_transfer_device.cu behavior with:
    - tile_a: FastA weights [d_size, rank] for LoRA left factor
    - tile_b: FastB weights [rank, x_size] for LoRA right factor
    - tile_c: Visible weights [d_size, x_size] for main matrix C

    Core operations:
    1. reinit(): A=0, B~Kaiming (first rank rows), optional C init
    2. ab_weight_update(): LoRA-style pulsed updates with projections
    3. ab_weight_transfer(): A⊗B -> C transfer, then reinit
    4. forward_inject(): y = C·x + α·A·(B·x) composition
    """

    def __init__(
        self,
        tile_a: AnalogTileWithoutPeriphery,   # fastA [d_size, rank]
        tile_b: AnalogTileWithoutPeriphery,   # fastB [rank, x_size]
        tile_c: AnalogTileWithoutPeriphery,   # visible [d_size, x_size]
        d_size: int,
        x_size: int,
        rank: int,
        *,
        transfer_lr: float = 1.0,
        transfer_every: int = 32,
        dynamic_te: bool = False,
        dynamic_te_power: float = 1.0,
        dynamic_te_min: Optional[int] = None,
        dynamic_te_max: Optional[int] = None,
        te_warmup_schedule: Optional[List[int]] = None,
        te_warmup_steps: int = 0,
        units_in_mbatch: bool = False,
        lora_alpha: float = 1.0,
        reinit_gain: float = 1.0,
        reinit_mode: str = "standard",
        a_init_mode: str = "zero",  # "zero" or "kaiming"
        b_init_mode: str = "kaiming",  # "kaiming" or "zero"
        correct_gradient_magnitudes: bool = False,
        fast_lr: float = 1.0,
        scale_transfer_lr: bool = True,
        transfer_fast_lr_ref: str = "geomean",
        rank_chunk: Optional[int] = None,
        ab_bl_mgmt: Optional[Dict[str, Any]] = None,
        transfer_bl_mgmt: Optional[Dict[str, Any]] = None,
        forward_inject: bool = False,
        num_reads: int = 1,
        multi_read_mode: str = "average",
        update_mode: str = "lora",  # "lora" or "reconstruction"
        transfer_method: str = "onehot",  # "onehot", "direct", or "set"
        transfer_rank_schedule: str = "all",  # "all" or "round_robin"
        transfer_ranks_per_step: int = 1,  # ranks per transfer in round_robin mode
        fi_continuous_alpha: bool = False,  # α = eff_transfer_lr when FI=True
        device: Optional[torch.device] = None,  # Explicit device to avoid get_weights()
        dtype: torch.dtype = torch.float32      # Explicit dtype
    ):
        """Initialize LR-TT controller.

        Args:
            tile_a: FastA tile for A matrix [d_size, rank]
            tile_b: FastB tile for B matrix [rank, x_size]
            tile_c: Visible tile for C matrix [d_size, x_size]
            d_size: Output dimension
            x_size: Input dimension
            rank: LoRA rank (must be <= min(d_size, x_size))
            transfer_lr: Transfer learning rate scalar
            transfer_every: Transfer frequency (steps or samples)
            units_in_mbatch: Whether transfer_every counts samples vs steps
            lora_alpha: LoRA scaling factor α
            reinit_gain: Kaiming uniform initialization multiplier (1.0 = standard PyTorch default)
            reinit_mode: Reinit strategy after transfer:
                        "standard" - A=0, B=Kaiming (original LRTT)
                        "decay" - no reinit, 6T1C capacitor decay handles weight reduction
                        "hybrid" - A=0, B unchanged (6T1C capacitor decay handles B)
                        "orthogonal_zero" - A=0, B=Random Orthogonal (FROZEN)
                        "orthogonal_decay" - A unchanged, B=Random Orthogonal (FROZEN)
                        "zero_orthogonal_zero" - A=0, B=0 every transfer (write noise varies)
                        "zero_orthogonal_decay" - A unchanged, B=0 every transfer (write noise varies)
            correct_gradient_magnitudes: Divide transfer LR by fast effective LR reference
            rank_chunk: Chunk size for transfer (None = full rank)
            ab_bl_mgmt: BL management for A/B updates {update_bl_management, update_management, desired_BL}
            transfer_bl_mgmt: BL management for transfers
            forward_inject: Enable forward injection optimization
            num_reads: Number of reads per rank during one-hot transfer (default 1)
            multi_read_mode: How to handle multiple reads: 'average' or 'per_read'
            update_mode: A/B update mode: 'lora' (chain rule) or 'reconstruction' (TikiTaka-style)
            transfer_method: Transfer method: "onehot", "direct", or "set" (exact weight setting)
            device: Explicit device (if None, safely inferred from tiles using tiny dummy forward)
                   Strongly recommended to pass the tile device explicitly for best performance
            dtype: Explicit dtype for tensors
        """
        if rank <= 0 or rank > min(d_size, x_size):
            raise ValueError(f"Invalid rank {rank} for dimensions {d_size}×{x_size}")

        self.tile_a = tile_a
        self.tile_b = tile_b
        self.tile_c = tile_c

        self.d_size = d_size
        self.x_size = x_size
        self.rank = rank

        # LRTT parameters
        self.transfer_lr = transfer_lr

        self.transfer_every = transfer_every
        self.units_in_mbatch = units_in_mbatch
        self.lora_alpha = lora_alpha
        self.reinit_gain = reinit_gain
        self.reinit_mode = reinit_mode
        self.a_init_mode = a_init_mode
        self.b_init_mode = b_init_mode
        self.correct_gradient_magnitudes = correct_gradient_magnitudes
        self.fast_lr = fast_lr
        self.scale_transfer_lr = scale_transfer_lr
        self.transfer_fast_lr_ref = transfer_fast_lr_ref
        self.auto_scale_mode = "none"    # Set from post_init
        self.auto_momentum = 0.99        # Set from post_init
        # EMA state: lazy init (first update creates 0-dim CUDA tensors)
        self.m_x = None
        self.m_d = None
        self.m_xb = None    # separate mode only
        self.m_da = None     # separate mode only
        self._autoscale_ema_initialized = False
        self.gran_a = 1.0    # Set from lrtt_tile.py for separate mode
        self.gran_b = 1.0
        self.last_lr_eff_a = 1.0   # Python float (after .item())
        self.last_lr_eff_b = 1.0
        self._last_lr_sgd = 1.0
        self.rank_chunk = rank_chunk or rank
        self.forward_inject_enabled = forward_inject
        self.fi_continuous_alpha = fi_continuous_alpha

        # Dynamic TE state
        self.dynamic_te = dynamic_te
        self.dynamic_te_power = dynamic_te_power
        self.te_base = transfer_every  # TE_0 (immutable)
        self.dynamic_te_min = dynamic_te_min if dynamic_te_min is not None else transfer_every
        self.dynamic_te_max = dynamic_te_max if dynamic_te_max is not None else transfer_every * 10
        self.lr_peak: Optional[float] = None  # Auto-tracked peak LR

        # TE warmup ramp state
        self.te_warmup_schedule = te_warmup_schedule  # e.g. [32, 64, 128, 230]
        self.te_warmup_steps = te_warmup_steps
        self.te_step_counter = 0  # Counts _update_dynamic_te calls

        # If warmup schedule is set, start TE at the first value
        if self.te_warmup_schedule and len(self.te_warmup_schedule) > 0:
            self.transfer_every = self.te_warmup_schedule[0]

        self.num_reads = num_reads
        self.multi_read_mode = multi_read_mode
        self.update_mode = update_mode
        self.transfer_method = transfer_method
        self.transfer_rank_schedule = transfer_rank_schedule
        self.transfer_ranks_per_step = transfer_ranks_per_step
        self._rr_rank_idx: int = 0  # Round-robin pointer: next transfer start rank

        # BL management settings
        self.ab_bl_mgmt = ab_bl_mgmt or {}
        self.transfer_bl_mgmt = transfer_bl_mgmt or {}

        # Diagnostics (off by default for speed; fine scripts enable this)
        self.enable_diagnostics: bool = False

        # Counters and state
        self.transfer_counter = 0
        self._last_m_batch = 1  # Last mini-batch size (for units_in_mbatch)
        self.num_a_updates = 0
        self.num_b_updates = 0
        self.num_transfers = 0
        self.last_transfer_delta = None
        self.last_actual_delta = None

        # Cached buffers for efficiency
        self._x_b_buffer: Optional[Tensor] = None
        self._d_a_buffer: Optional[Tensor] = None
        self._pad_buffer_a: Optional[Tensor] = None
        self._pad_buffer_b: Optional[Tensor] = None

        # Cached C tile bounds (set once on first transfer, immutable after d2d init)
        self._c_bounds_cached: bool = False
        self._c_max_bound: Optional[Tensor] = None  # None means no clipping needed
        self._c_min_bound: Optional[Tensor] = None

        # Device info - infer from tiles if not provided
        if device is None:
            # Safely infer device from tile using a tiny dummy forward
            device = self._infer_device_from_tile()
        self.device = device
        self.dtype = dtype

        # Track initialization state with flags to avoid weight norm checks
        self._c_initialized = True
        self._tiles_initialized = False
        self._b_frozen = False  # Set to True for orthogonal mode

        # Transfer robustness knobs (safe defaults)
        self.transfer_micro_steps: int = 1          # M: micro-transfer 반복 횟수
        self.transfer_centering: bool = False       # 행/열 평균 제거 (기본 off - gradient 왜곡 방지)
        self.transfer_normalize: bool = False       # 랭크별 ℓ2 정규화 (기본 off - gradient 왜곡 방지)

        # Transfer mode: "pilot" (gamma calibration) or "sigma_delta" (ΣΔ quantization)
        self.transfer_mode: str = "off"             # {"pilot", "sigma_delta", "off"}
        self.transfer_gamma_mode: str = "off"       # Legacy alias for transfer_mode (deprecated)
        self.transfer_pilot_frac: float = 1.0/16.0  # 파일럿 전송 lr = |transfer_lr| * frac

        # Sigma-Delta (ΣΔ) state for "sigma_delta" mode
        self.sd_quantum: Optional[float] = None     # g: unit quantum for rank-wise pulses (None -> derive per transfer)
        self.sd_acc: Optional[Tensor] = None        # h_k residuals [rank], persistent across transfers

        # --- Read noise reduction (oversampling) ---
        self.read_n_avg: int = 1                    # Oversampling count (1=disabled, 4/8=recommended)
        self.differential_read: bool = False        # Use +e/-e differential (True) or +e only (False)

        # --- AGC (Automatic Gain Control) settings ---
        self.agc_enabled: bool = False              # Enable AGC for read amplitude optimization
        self.agc_margin: float = 0.85               # Target output bound fraction (avoid clipping)
        self.agc_max_iters: int = 6                 # Max iterations for AGC binary search

        # --- Two-Amplitude differential read settings ---
        self.two_amp_enabled: bool = False          # Enable two-amplitude differential read (odd offset removal)
        self.two_amp_ratio: float = 0.5             # Ratio α1/α2 for two-amplitude method

        # === Reconstruction update parameters (for update_mode='reconstruction') ===
        self.recon_lambda_a: float = 1e-3
        self.recon_lambda_b: float = 1e-3
        self.recon_use_scalar_stabilizer: bool = False
        self.recon_use_exact_gram: bool = False
        self.recon_exact_gram_every: int = 0  # 0 = disabled, N = every N steps
        self.recon_ema_beta: float = 0.9
        self.recon_lr_scale: float = 1.0
        self.recon_clip_norm: float = 10.0
        self.recon_use_clip_norm: bool = False

        # EMA state for scalar stabilizer (sA = ||A||^2/rank, sB = ||B||^2/rank)
        self._ema_sA: float = 0.0
        self._ema_sB: float = 0.0
        self._ema_initialized: bool = False
        self._recon_step_counter: int = 0

        # Transfer one-hot vectors cache
        self._transfer_vec_a: Optional[Tensor] = None

        # === Debug Logging Settings ===
        self.log_ab_scaling: bool = False
        """Enable logging of x,d max values during A/B updates."""

        self.log_ab_scaling_every: int = 10
        """Log x,d max values every N steps (only when log_ab_scaling=True)."""

        self._ab_update_step: int = 0
        """Internal counter for A/B update logging."""

        # === Hardware Mode Flag ===
        self._hardware_mode: Optional[bool] = None
        """Cache for hardware mode detection (use_manual_scaling=True)."""

    @property
    def effective_alpha(self) -> float:
        """Return the forward-injection scaling factor α.

        When fi_continuous_alpha is True and forward injection is enabled,
        α equals the effective transfer learning rate (transfer_lr, or
        transfer_lr * lr_sgd when scale_transfer_lr is True).
        Otherwise falls back to the static lora_alpha.
        """
        if self.fi_continuous_alpha and self.forward_inject_enabled:
            lr_tr = self.transfer_lr
            if self.scale_transfer_lr:
                lr_tr *= self._last_lr_sgd
            return lr_tr
        return self.lora_alpha

    def _is_hardware_mode(self) -> bool:
        """Check if tiles are in hardware mode (use_manual_scaling=True).

        In hardware mode, learning rate cannot be configured - weight updates
        are purely determined by pulse coincidences and dw_min.

        Returns:
            True if use_manual_scaling=True in tile config, False otherwise.
        """
        if self._hardware_mode is not None:
            return self._hardware_mode

        # Check tile_a's rpu_config for use_manual_scaling
        try:
            if hasattr(self.tile_a, 'rpu_config') and hasattr(self.tile_a.rpu_config, 'update'):
                self._hardware_mode = getattr(self.tile_a.rpu_config.update, 'use_manual_scaling', False)
            else:
                self._hardware_mode = False
        except Exception:
            self._hardware_mode = False

        return self._hardware_mode

    def _ensure_sd_state(self) -> None:
        """Ensure ΣΔ state tensors exist on the right device/dtype."""
        if self.sd_acc is None or self.sd_acc.numel() != self.rank or self.sd_acc.device != self.device:
            self.sd_acc = torch.zeros(self.rank, device=self.device, dtype=self.dtype)

    def _diff_read(self, tile, e: Tensor, amp: float, mode: str, read_n_avg: int) -> Tensor:
        """Differential read with amplitude scaling and averaging.

        Computes: d(amp) = 0.5 * (f(+amp·e) - f(-amp·e)) averaged over read_n_avg readings.

        This removes DC offset and even-order distortions through differential reading,
        and reduces stochastic noise by √read_n_avg through averaging.

        Args:
            tile: Analog tile to read from (tile_a or tile_b)
            e: One-hot vector [1, rank]
            amp: Input amplitude scaling factor
            mode: "fwd" for forward pass, "bwd" for backward pass
            read_n_avg: Number of readings to average (noise reduction by √N)

        Returns:
            Differential read result: 0.5*(f(+amp·e) - f(-amp·e)) averaged
        """
        acc = None
        for _ in range(read_n_avg):
            if mode == "fwd":
                yp = tile.forward(amp * e)
                ym = tile.forward(-amp * e)
            else:  # "bwd"
                yp = tile.backward(amp * e)
                ym = tile.backward(-amp * e)
            d = 0.5 * (yp - ym)
            acc = d if acc is None else (acc + d)
        return acc / float(read_n_avg)

    def _pick_amp_agc(self, tile, e: Tensor, mode: str, margin: float = 0.85, max_iters: int = 6) -> float:
        """Automatic Gain Control: pick amplitude to maximize SNR without clipping.

        Uses binary search to find the largest amplitude that keeps output within
        margin * out_bound, maximizing signal strength while avoiding ADC saturation.

        Args:
            tile: Analog tile to probe
            e: One-hot vector [1, rank]
            mode: "fwd" for forward pass, "bwd" for backward pass
            margin: Target fraction of out_bound (0.85 = 85%)
            max_iters: Maximum binary search iterations

        Returns:
            Optimal amplitude for the tile read operation
        """
        # Get IO parameters from tile config
        # FloatingPointTile doesn't have forward/backward IO parameters
        io = getattr(tile.rpu_config, 'forward' if mode == "fwd" else 'backward', None)
        if io is None:
            return 1.0  # FloatingPointTile: 기본 amplitude 1.0 반환
        out_bound = float(getattr(io, "out_bound", 0.0) or 0.0)
        inp_bound = float(getattr(io, "inp_bound", 1.0) or 1.0)

        # Start with maximum allowed input amplitude
        amp = min(1.0, inp_bound)

        # If no output bound defined, just use max input
        if out_bound <= 0.0:
            return amp

        for _ in range(max_iters):
            # Probe with current amplitude
            if mode == "fwd":
                yp = tile.forward(amp * e)
                ym = tile.forward(-amp * e)
            else:
                yp = tile.backward(amp * e)
                ym = tile.backward(-amp * e)

            raw_max = float(torch.max(yp.abs().amax(), ym.abs().amax()).item())

            # Binary search: decrease if clipping, increase if too low
            if raw_max > margin * out_bound and amp > 1e-4:
                amp *= 0.5
                continue
            if raw_max < 0.2 * out_bound and amp < inp_bound:
                amp = min(amp * 2.0, inp_bound)
                continue
            break

        return amp

    def _two_amp_read(self, tile, e: Tensor, mode: str, read_n_avg: int = 8, margin: float = 0.85) -> tuple:
        """Two-amplitude differential read to cancel odd offset.

        The one-hot output model is: d(α) = α·w_k + b_odd
        where w_k is the desired weight column and b_odd is odd-order distortion.

        Using two amplitudes α1, α2:
          d(α1) = α1·w_k + b_odd
          d(α2) = α2·w_k + b_odd
        Solving: w_k = (d(α2) - d(α1)) / (α2 - α1)

        Args:
            tile: Analog tile to read from
            e: One-hot vector [1, rank]
            mode: "fwd" or "bwd"
            read_n_avg: Number of readings per amplitude level (noise reduction by √N)
            margin: AGC margin for amplitude selection

        Returns:
            (w_hat: estimated weight, b_odd: estimated odd offset, (a1, a2): amplitudes used)
        """
        # Find optimal high amplitude using AGC
        a2 = self._pick_amp_agc(tile, e, mode=mode, margin=margin, max_iters=self.agc_max_iters)

        # Low amplitude is a fraction of high amplitude
        a1 = self.two_amp_ratio * a2

        # Read at both amplitudes
        d1 = self._diff_read(tile, e, amp=a1, mode=mode, read_n_avg=read_n_avg)
        d2 = self._diff_read(tile, e, amp=a2, mode=mode, read_n_avg=read_n_avg)

        # Solve for weight (cancel odd offset)
        denom = a2 - a1
        w_hat = (d2 - d1) / max(denom, 1e-12)

        # Estimate odd offset
        b_odd = d1 - a1 * w_hat

        return w_hat.squeeze(0), b_odd.squeeze(0), (a1, a2)

    def _infer_device_from_tile(self) -> torch.device:
        """Safely infer device from tile.

        This is now a simple fallback since device should be explicitly synchronized
        via set_device() when tiles are moved.
        """
        # Check tile backend type
        if hasattr(self.tile_c, 'tile'):
            tile_str = str(type(self.tile_c.tile).__name__)
            if 'Cuda' in tile_str or 'CUDA' in tile_str:
                return torch.device('cuda')

        # Default based on CUDA availability
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_tile_device(self) -> torch.device:
        """Get device that tiles expect for operations."""
        # OPTIMIZATION: Return cached device instead of using get_weights()
        return self.device

    def _get_tile_dtype(self) -> torch.dtype:
        """Get common dtype from tiles."""
        # OPTIMIZATION: Return cached dtype instead of checking tiles
        return self.dtype

    def _ensure_buffers(self, batch_size: int) -> None:
        """Ensure scratch buffers are allocated for given batch size."""
        if (self._x_b_buffer is None or
            self._x_b_buffer.size(-1) != batch_size):

            # Use cached device
            device = self.device

            # Projection buffers
            self._x_b_buffer = torch.zeros(
                self.rank, batch_size, device=device, dtype=self.dtype
            )
            self._d_a_buffer = torch.zeros(
                self.rank, batch_size, device=device, dtype=self.dtype
            )

            # CRITICAL FIX: Padding buffers must match tile input dimensions
            # A tile expects [x_size, batch] inputs (not d_size!)
            # B tile expects [d_size, batch] for errors
            self._x_pad = torch.zeros(
                self.x_size, batch_size, device=device, dtype=self.dtype
            )
            self._d_pad = torch.zeros(
                self.d_size, batch_size, device=device, dtype=self.dtype
            )

    def apply_step_decay(self) -> None:
        """No-op. Kept for backward compatibility.

        Decay is handled by 6T1C capacitor decay in post_update_step().
        """
        pass

    def reinit(self) -> None:
        """Reinit A,B matrices based on reinit_mode, a_init_mode, and b_init_mode.

        Reinit modes:
        - "standard": Always reinitialize A and B
        - "decay": First time initialize, no reinit after transfer (6T1C capacitor decay handles it)
        - "hybrid": A=0, B unchanged (6T1C capacitor decay handles B)
        - "orthogonal_zero": A=0, B=Random Orthogonal (FROZEN)
        - "orthogonal_decay": A unchanged, B=Random Orthogonal (FROZEN) (6T1C decay handles A)
        - "zero_orthogonal_zero": A=0, B=0 every transfer (write noise varies each time)
        - "zero_orthogonal_decay": A unchanged, B=0 every transfer (write noise varies each time)

        A initialization (controlled by a_init_mode):
        - "zero": A=0 (LoRA-style, ensures ΔW=0 initially)
        - "kaiming": A=Kaiming Normal (random initialization)

        B initialization (controlled by b_init_mode):
        - "kaiming": B=Kaiming Normal (standard LoRA initialization)
        - "zero": B=0 (ensures ΔW=0 initially)
        """
        with torch.no_grad():
            # Helper: set raw weights on inner tile directly (bypasses periphery)
            def _set_raw(tile, w):
                tile.tile.set_weights(w)

            def _get_raw_gpu(tile):
                return tile.tile.get_weights().to(self.device)

            if self.reinit_mode == "standard":
                # A matrix: Use a_init_mode
                if self.a_init_mode == "kaiming":
                    A_init = torch.empty(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                    nn.init.kaiming_uniform_(A_init, a=math.sqrt(5))
                    A_init *= self.reinit_gain
                else:  # "zero"
                    A_init = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                _set_raw(self.tile_a, A_init)

                # B matrix: Use b_init_mode
                if self.b_init_mode == "zero":
                    B_init = torch.zeros(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                else:  # "kaiming"
                    B_init = torch.empty(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                    nn.init.kaiming_uniform_(B_init, a=math.sqrt(5))
                    B_init *= self.reinit_gain
                _set_raw(self.tile_b, B_init)

            elif self.reinit_mode == "decay":
                # Decay mode: First time initialize, no reinit after transfer
                # 6T1C capacitor decay handles weight reduction naturally
                if not self._tiles_initialized:
                    # First time: Use a_init_mode for A initialization
                    if self.a_init_mode == "kaiming":
                        A_init = torch.empty(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                        nn.init.kaiming_uniform_(A_init, a=math.sqrt(5))
                        A_init *= self.reinit_gain
                    else:  # "zero"
                        A_init = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                    _set_raw(self.tile_a, A_init)

                    # B matrix: Use b_init_mode
                    if self.b_init_mode == "zero":
                        B_init = torch.zeros(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                    else:  # "kaiming"
                        B_init = torch.empty(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                        nn.init.kaiming_uniform_(B_init, a=math.sqrt(5))
                        B_init *= self.reinit_gain
                    _set_raw(self.tile_b, B_init)
                # else: no reinit, 6T1C capacitor decay handles it

            elif self.reinit_mode == "hybrid":
                # A=0, B unchanged (6T1C capacitor decay handles B reduction)
                A_zeros = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                _set_raw(self.tile_a, A_zeros)

                if not self._tiles_initialized:
                    # First time: Use b_init_mode
                    if self.b_init_mode == "zero":
                        B_init = torch.zeros(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                    else:  # "kaiming"
                        B_init = torch.empty(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                        nn.init.kaiming_uniform_(B_init, a=math.sqrt(5))
                        B_init *= self.reinit_gain
                    _set_raw(self.tile_b, B_init)
                # else: B unchanged, 6T1C capacitor decay handles it

            elif self.reinit_mode == "orthogonal_zero":
                # B = Random Orthogonal (FROZEN), A = 0 after each transfer
                A_zeros = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                _set_raw(self.tile_a, A_zeros)

                if not self._tiles_initialized:
                    random_matrix = torch.randn(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                    Q, R = torch.linalg.qr(random_matrix.t())
                    B_orthogonal = Q.t() * math.sqrt(self.x_size / self.rank)
                    _set_raw(self.tile_b, B_orthogonal)
                    self._b_frozen = True
                # else: B is frozen, don't change it

            elif self.reinit_mode == "orthogonal_decay":
                # B = Random Orthogonal (FROZEN), A unchanged (6T1C decay handles it)
                if not self._tiles_initialized:
                    A_zeros = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                    _set_raw(self.tile_a, A_zeros)

                    random_matrix = torch.randn(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                    Q, R = torch.linalg.qr(random_matrix.t())
                    B_orthogonal = Q.t() * math.sqrt(self.x_size / self.rank)
                    _set_raw(self.tile_b, B_orthogonal)
                    self._b_frozen = True
                # else: A unchanged, 6T1C capacitor decay handles it

            elif self.reinit_mode == "zero_orthogonal_zero":
                # A=0, B=0 every transfer (non-trainable)
                # B is re-set to zero each transfer so device write noise differs each time
                A_zeros = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                _set_raw(self.tile_a, A_zeros)
                B_zeros = torch.zeros(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                _set_raw(self.tile_b, B_zeros)
                self._b_frozen = True

            elif self.reinit_mode == "zero_orthogonal_decay":
                # A unchanged (6T1C decay handles it), B=0 every transfer (non-trainable)
                # B is re-set to zero each transfer so device write noise differs each time
                if not self._tiles_initialized:
                    A_zeros = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                    _set_raw(self.tile_a, A_zeros)
                # else: A unchanged, 6T1C capacitor decay handles it
                B_zeros = torch.zeros(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                _set_raw(self.tile_b, B_zeros)
                self._b_frozen = True

            else:
                raise ValueError(f"Unknown reinit_mode: {self.reinit_mode}. "
                                 f"Must be 'standard', 'decay', 'hybrid', 'orthogonal_zero', "
                                 f"'orthogonal_decay', 'zero_orthogonal_zero', or 'zero_orthogonal_decay'")

        # Apply device clipping if available
        if hasattr(self.tile_a, 'clip_weights'):
            self.tile_a.clip_weights()
        if hasattr(self.tile_b, 'clip_weights'):
            self.tile_b.clip_weights()

        # OPTIMIZATION: Use flag instead of reading C weights for norm check
        if self.forward_inject_enabled and not self._c_initialized:
            # Kaiming uniform init (same as PyTorch nn.Linear default)
            C_init = torch.empty(self.d_size, self.x_size, device=self.device, dtype=self.dtype)
            nn.init.kaiming_uniform_(C_init, a=math.sqrt(5))
            C_init *= self.reinit_gain
            self.tile_c.set_weights(C_init)
            if hasattr(self.tile_c, 'clip_weights'):
                self.tile_c.clip_weights()
            self._c_initialized = True

        # Reset counters
        self.transfer_counter = 0
        self._tiles_initialized = True

    def _reinit_ranks(self, rank_indices: List[int]) -> None:
        """Reinit only the specified ranks of A and B after transfer.

        If all ranks are selected, delegates to full reinit().
        Otherwise, performs per-rank read-modify-write on A columns and B rows.

        Args:
            rank_indices: List of rank indices to reinitialize.
        """
        # Full rank case → delegate to existing reinit()
        if len(rank_indices) == self.rank:
            self.reinit()
            return

        with torch.no_grad():
            def _get_raw(tile):
                return tile.tile.get_weights().to(self.device)

            def _set_raw(tile, w):
                tile.tile.set_weights(w)

            A_full = _get_raw(self.tile_a)  # [d_size, rank]
            B_full = _get_raw(self.tile_b)  # [rank, x_size]

            for k in rank_indices:
                if self.reinit_mode == "standard":
                    # A[:, k] = 0, B[k, :] = Kaiming
                    if self.a_init_mode == "kaiming":
                        col = torch.empty(self.d_size, 1, device=self.device, dtype=self.dtype)
                        nn.init.kaiming_uniform_(col, a=math.sqrt(5))
                        A_full[:, k] = col.squeeze(1) * self.reinit_gain
                    else:
                        A_full[:, k] = 0.0

                    if self.b_init_mode == "zero":
                        B_full[k, :] = 0.0
                    else:
                        row = torch.empty(1, self.x_size, device=self.device, dtype=self.dtype)
                        nn.init.kaiming_uniform_(row, a=math.sqrt(5))
                        B_full[k, :] = row.squeeze(0) * self.reinit_gain

                elif self.reinit_mode == "decay":
                    # No reinit — 6T1C capacitor decay handles it
                    pass

                elif self.reinit_mode == "hybrid":
                    # A[:, k] = 0, B unchanged
                    A_full[:, k] = 0.0

                elif self.reinit_mode == "orthogonal_zero":
                    # A[:, k] = 0, B frozen
                    A_full[:, k] = 0.0

                elif self.reinit_mode == "orthogonal_decay":
                    # A unchanged, B frozen — no-op
                    pass

                elif self.reinit_mode == "zero_orthogonal_zero":
                    # A[:, k] = 0, B[k, :] = 0
                    A_full[:, k] = 0.0
                    B_full[k, :] = 0.0

                elif self.reinit_mode == "zero_orthogonal_decay":
                    # A unchanged, B[k, :] = 0
                    B_full[k, :] = 0.0

            _set_raw(self.tile_a, A_full)
            _set_raw(self.tile_b, B_full)

        # Apply device clipping if available
        if hasattr(self.tile_a, 'clip_weights'):
            self.tile_a.clip_weights()
        if hasattr(self.tile_b, 'clip_weights'):
            self.tile_b.clip_weights()

        # Reset counters
        self.transfer_counter = 0

    def ab_weight_update(
        self,
        x: Tensor,
        d: Tensor,
        lr: float,
        in_trans: bool = False,
        out_trans: bool = False
    ) -> None:
        """Update A and B with rank-r gradient approximation.

        When update_mode='lora': Uses LoRA chain rule (original LRTT behavior).
        When update_mode='reconstruction': Uses gradient reconstruction (TikiTaka-style).

        Args:
            x: Input tensor
            d: Error tensor
            lr: Learning rate
            in_trans: Whether x is transposed
            out_trans: Whether d is transposed
        """
        # Branch based on update_mode
        if self.update_mode == "reconstruction":
            return self._ab_weight_update_reconstruction(x, d, lr, in_trans, out_trans)
        else:
            return self._ab_weight_update_lora(x, d, lr, in_trans, out_trans)

    def _lazy_init_ema(self, device: torch.device) -> None:
        """Lazily initialize EMA tensors on the correct device."""
        if self._autoscale_ema_initialized:
            return
        self.m_x = torch.tensor(0.0, device=device)
        self.m_d = torch.tensor(0.0, device=device)
        if self.auto_scale_mode == "separate":
            self.m_xb = torch.tensor(0.0, device=device)
            self.m_da = torch.tensor(0.0, device=device)
        self._autoscale_ema_initialized = True

    def _update_autoscale_ema(
        self, x: Tensor, d: Tensor, XB: Tensor, DA: Tensor, m_batch: int
    ) -> None:
        """Update EMA statistics and compute lr_eff_a/b for auto_scale_mode.

        This is extracted from _ab_weight_update_lora so it can also be called
        from the forward_inject=True hook path.

        Args:
            x: Input activations [batch, x_size]
            d: Gradient w.r.t. output [batch, d_size]
            XB: B·x projection [batch, rank]
            DA: A^T·d projection [batch, rank]
            m_batch: Batch size (for tau calculation)
        """
        tau = (1.0 - self.auto_momentum) / m_batch
        self._lazy_init_ema(x.device)

        if self.auto_scale_mode == "shared":
            self.m_x = (1 - tau) * self.m_x + tau * x.abs().amax()
            self.m_d = (1 - tau) * self.m_d + tau * d.abs().amax()
            denom = self.m_x * self.m_d
            fast_lr_t = x.new_tensor(self.fast_lr)
            lr_eff = torch.where(denom > 0, fast_lr_t / denom, fast_lr_t).item()
            self.last_lr_eff_a = lr_eff
            self.last_lr_eff_b = lr_eff
        elif self.auto_scale_mode == "separate":
            self.m_x = (1 - tau) * self.m_x + tau * x.abs().amax()
            self.m_d = (1 - tau) * self.m_d + tau * d.abs().amax()
            self.m_xb = (1 - tau) * self.m_xb + tau * XB.abs().amax()
            self.m_da = (1 - tau) * self.m_da + tau * DA.abs().amax()
            denom_a = self.m_xb * self.m_d
            fa = x.new_tensor(self.fast_lr * self.gran_a)
            self.last_lr_eff_a = torch.where(denom_a > 0, fa / denom_a, fa).item()
            denom_b = self.m_x * self.m_da
            fb = x.new_tensor(self.fast_lr * self.gran_b)
            self.last_lr_eff_b = torch.where(denom_b > 0, fb / denom_b, fb).item()

    def _ab_weight_update_lora(
        self,
        x: Tensor,
        d: Tensor,
        lr: float,
        in_trans: bool = False,
        out_trans: bool = False
    ) -> None:
        """LoRA chain rule update for A and B (original LRTT).

        Uses tile forward/backward for projections and tile update for weight changes.
        """
        # 0) Normalize to [batch, feat] format
        if in_trans:
            x = x.t()
        if out_trans:
            d = d.t()

        # 1) Projections (analog path — IO resolution controlled per-tile)
        with torch.no_grad():
            XB = self.tile_b.forward(x)     # [batch, rank] = B·X
            DA = self.tile_a.backward(d)    # [batch, rank] = A^T·D

        # 2) lr_eff_a, lr_eff_b via fast_lr + auto_scale
        self._last_lr_sgd = lr
        m_batch = x.shape[0]

        if self._is_hardware_mode():
            self.last_lr_eff_a = self.last_lr_eff_b = 1.0
        elif self.auto_scale_mode == "none":
            self.last_lr_eff_a = self.last_lr_eff_b = self.fast_lr
        else:
            self._update_autoscale_ema(x, d, XB, DA, m_batch)

        lr_eff_a = self.last_lr_eff_a
        lr_eff_b = self.last_lr_eff_b

        # === Debug Logging: x,d max values for A and B tiles ===
        self._ab_update_step += 1
        if self.log_ab_scaling and (self._ab_update_step % self.log_ab_scaling_every == 1):
            # A tile update uses: x=XB (projection), d=d (gradient)
            a_x_max = XB.abs().max().item()
            a_d_max = d.abs().max().item()
            # B tile update uses: x=x (input), d=DA (projection)
            b_x_max = x.abs().max().item()
            b_d_max = DA.abs().max().item()

            print(f"  [AB-Scaling Step {self._ab_update_step}] "
                  f"A: x_max={a_x_max:.4f}, d_max={a_d_max:.4f} | "
                  f"B: x_max={b_x_max:.4f}, d_max={b_d_max:.4f} | "
                  f"lr_eff_a={lr_eff_a:.6f}, lr_eff_b={lr_eff_b:.6f}")

        # 3) ΔA = -lr_eff_a · D^T · (B·X) → tile_a.update(XB, d)
        lr_a_old = self.tile_a.get_learning_rate()
        self.tile_a.set_learning_rate(lr_eff_a)
        if hasattr(self.tile_a, '_orig_update'):
            self.tile_a._orig_update(XB, d)
        else:
            self.tile_a.update(XB, d)
        self.tile_a.set_learning_rate(lr_a_old)
        self.num_a_updates += 1

        # 4) ΔB = -lr_eff_b · (A^T·D)^T · X → tile_b.update(x, DA)
        # Skip B update if B is frozen (orthogonal mode)
        if not self._b_frozen:
            lr_b_old = self.tile_b.get_learning_rate()
            self.tile_b.set_learning_rate(lr_eff_b)
            if hasattr(self.tile_b, '_orig_update'):
                self.tile_b._orig_update(x, DA)
            else:
                self.tile_b.update(x, DA)
            self.tile_b.set_learning_rate(lr_b_old)
            self.num_b_updates += 1

        # 5) Dynamic TE update (before counter check)
        self._update_dynamic_te(lr)

        # 6) Counter (always += m_batch, matching TikiTaka convention)
        self._last_m_batch = m_batch
        self.transfer_counter += m_batch

    def _ab_weight_update_reconstruction(
        self,
        x: Tensor,
        d: Tensor,
        lr: float,
        in_trans: bool = False,
        out_trans: bool = False
    ) -> None:
        """TikiTaka-style gradient reconstruction update for A and B.

        When forward_inject=False, the forward pass uses only C (y = Cx), so A,B
        don't appear in the loss. Instead of using LoRA chain rule, we treat A,B
        as "gradient buffers" and minimize:

            L_rec(A,B) = 1/2 ||AB + G||_F^2 + (λA/2)||A||_F^2 + (λB/2)||B||_F^2

        where G = D^T @ X is the ideal gradient for C. This makes AB ≈ -G,
        so that `C += transfer_lr * AB` implements SGD descent: C -= transfer_lr * G.

        Gradients:
            ∂L_rec/∂A = A(BB^T) + GB^T + λA*A
            ∂L_rec/∂B = (A^TA)B + A^TG + λB*B

        Hardware mapping:
            - Hebbian terms (GB^T, A^TG):
                XB = tile_b.forward(X)
                tile_a.update(XB, D)  → A -= lr * D^T @ XB = A -= lr * GB^T
                DA = tile_a.backward(D)
                tile_b.update(X, DA)  → B -= lr * DA^T @ X = B -= lr * A^TG

            - Stabilizer terms: Use scalar approximation or exact Gram
        """
        # 0) Normalize to [batch, feat] format
        if in_trans:
            x = x.t()
        if out_trans:
            d = d.t()

        # 1) Effective learning rate (no lora_alpha in reconstruction mode)
        lr_rec = lr * self.recon_lr_scale

        # 2) Projections for Hebbian terms
        with torch.no_grad():
            XB = self.tile_b.forward(x)  # [batch, rank]
            DA = self.tile_a.backward(d)  # [batch, rank]

        # 3) Hebbian updates: A -= lr * GB^T, B -= lr * A^TG
        # A update: A -= lr_rec * D^T @ XB = A -= lr_rec * GB^T
        lr_a_old = self.tile_a.get_learning_rate()
        self.tile_a.set_learning_rate(lr_rec)
        if hasattr(self.tile_a, '_orig_update'):
            self.tile_a._orig_update(XB, d)
        else:
            self.tile_a.update(XB, d)
        self.tile_a.set_learning_rate(lr_a_old)
        self.num_a_updates += 1

        # B update: B -= lr_rec * DA^T @ X = B -= lr_rec * A^TG
        # Skip if B is frozen (orthogonal mode)
        if not self._b_frozen:
            lr_b_old = self.tile_b.get_learning_rate()
            self.tile_b.set_learning_rate(lr_rec)
            if hasattr(self.tile_b, '_orig_update'):
                self.tile_b._orig_update(x, DA)
            else:
                self.tile_b.update(x, DA)
            self.tile_b.set_learning_rate(lr_b_old)
            self.num_b_updates += 1

        # 4) Stabilizer terms (prevent ||A@B|| growth)
        self._apply_reconstruction_stabilizer(lr_rec)

        # 5) Safety norm clipping (fallback) - only if enabled
        if self.recon_use_clip_norm:
            self._clip_ab_norms(max_norm=self.recon_clip_norm)

        # 6) Dynamic TE update (before counter check)
        self._update_dynamic_te(lr)

        # 7) Counter (always += m_batch, matching TikiTaka convention)
        m_batch = x.shape[0]
        self._last_m_batch = m_batch
        self.transfer_counter += m_batch

    def _apply_reconstruction_stabilizer(self, lr_rec: float) -> None:
        """Apply stabilizer terms for reconstruction update.

        Uses forward/backward to read weights and update() to apply decay,
        avoiding expensive get_weights()/set_weights() calls.
        """
        with torch.no_grad():
            device = self.device

            # Read A using forward: tile_a.forward(I) = I @ A.T = A.T
            I_rank = torch.eye(self.rank, device=device, dtype=self.dtype)
            A_T = self.tile_a.forward(I_rank)  # [rank, d_size] = A.T

            # Read B using backward: tile_b.backward(I) = I @ B = B
            B_read = self.tile_b.backward(I_rank)  # [rank, x_size] = B

            # Compute norms for scalar stabilizer
            A_norm_sq = torch.sum(A_T ** 2).item()
            B_norm_sq = torch.sum(B_read ** 2).item()
            sA = A_norm_sq / self.rank  # tr(A^TA)/rank
            sB = B_norm_sq / self.rank  # tr(BB^T)/rank

            # Update EMA estimates
            if not self._ema_initialized:
                self._ema_sA = sA
                self._ema_sB = sB
                self._ema_initialized = True
            else:
                beta = self.recon_ema_beta
                self._ema_sA = beta * self._ema_sA + (1 - beta) * sA
                self._ema_sB = beta * self._ema_sB + (1 - beta) * sB

            # Determine whether to use exact Gram this step
            self._recon_step_counter += 1
            use_exact_this_step = self.recon_use_exact_gram or (
                self.recon_exact_gram_every > 0 and
                self._recon_step_counter % self.recon_exact_gram_every == 0
            )

            if use_exact_this_step:
                # Exact Gram matrix stabilizer
                A_rank = A_T.t()  # [d_size, rank]
                B_rank = B_read   # [rank, x_size]

                BBT = B_rank @ B_rank.t()  # [rank, rank]
                ATA = A_rank.t() @ A_rank  # [rank, rank]

                # Combined stabilizer + L2: d_A = A @ BBT + λA * A = A @ (BBT + λA*I)
                BBT_reg = BBT + self.recon_lambda_a * I_rank
                ATA_reg = ATA + self.recon_lambda_b * I_rank

                # Apply A -= lr_rec * A @ BBT_reg using tile.update
                d_A = (A_rank @ BBT_reg).t()  # [rank, d_size]
                lr_old_a = self.tile_a.get_learning_rate()
                self.tile_a.set_learning_rate(lr_rec)
                self.tile_a.update(I_rank, d_A.t())
                self.tile_a.set_learning_rate(lr_old_a)

                if not self._b_frozen:
                    # B -= lr_rec * ATA_reg @ B
                    lr_old_b = self.tile_b.get_learning_rate()
                    self.tile_b.set_learning_rate(lr_rec)
                    self.tile_b.update(B_rank, ATA_reg.t())
                    self.tile_b.set_learning_rate(lr_old_b)

            elif self.recon_use_scalar_stabilizer:
                # Scalar approximation: BB^T ≈ sB*I, A^TA ≈ sA*I
                # Exponential decay (always positive, no sign flip risk)
                exp_arg_A = lr_rec * (self._ema_sB + self.recon_lambda_a)
                exp_arg_B = lr_rec * (self._ema_sA + self.recon_lambda_b)

                # Clamp exponent to prevent extreme decay
                exp_arg_A = min(exp_arg_A, 3.0)
                exp_arg_B = min(exp_arg_B, 3.0)

                decay_A = math.exp(-exp_arg_A)
                decay_B = math.exp(-exp_arg_B)

                # Apply A *= decay using tile.update: A -= (1-decay)*A
                c_A = 1.0 - decay_A
                if c_A > 1e-8:
                    lr_old_a = self.tile_a.get_learning_rate()
                    self.tile_a.set_learning_rate(c_A)
                    self.tile_a.update(I_rank, A_T.t())
                    self.tile_a.set_learning_rate(lr_old_a)

                if not self._b_frozen:
                    c_B = 1.0 - decay_B
                    if c_B > 1e-8:
                        lr_old_b = self.tile_b.get_learning_rate()
                        self.tile_b.set_learning_rate(c_B)
                        self.tile_b.update(B_read, I_rank)
                        self.tile_b.set_learning_rate(lr_old_b)

    def _clip_ab_norms(self, max_norm: float = 10.0) -> None:
        """Clip A and B norms to prevent explosion.

        Uses forward/backward to read weights and update() to apply scaling.
        """
        with torch.no_grad():
            device = self.device

            # Read A using forward: tile_a.forward(I) = I @ A.T = A.T
            I_rank_a = torch.eye(self.rank, device=device, dtype=self.dtype)
            A_T = self.tile_a.forward(I_rank_a)  # [rank, d_size] = A.T
            A_norm = torch.norm(A_T).item()

            if A_norm > max_norm:
                scale = max_norm / A_norm
                c = 1.0 - scale
                lr_old = self.tile_a.get_learning_rate()
                self.tile_a.set_learning_rate(c)
                self.tile_a.update(I_rank_a, A_T.t())
                self.tile_a.set_learning_rate(lr_old)

            # Read B using backward: tile_b.backward(I) = I @ B = B
            I_rank_b = torch.eye(self.rank, device=device, dtype=self.dtype)
            B_read = self.tile_b.backward(I_rank_b)
            B_norm = torch.norm(B_read).item()

            if B_norm > max_norm and not self._b_frozen:
                scale = max_norm / B_norm
                c = 1.0 - scale
                lr_old = self.tile_b.get_learning_rate()
                self.tile_b.set_learning_rate(c)
                self.tile_b.update(B_read, I_rank_b)
                self.tile_b.set_learning_rate(lr_old)

    def _compute_effective_transfer_lr(self) -> float:
        """Compute effective transfer LR with optional SGD scaling and gradient correction.

        Returns:
            Effective transfer learning rate for this transfer step.
        """
        lr_tr = self.transfer_lr
        if self.scale_transfer_lr:
            lr_tr *= self._last_lr_sgd
        if self.correct_gradient_magnitudes:
            if self.transfer_fast_lr_ref == "A":
                ref = self.last_lr_eff_a
            elif self.transfer_fast_lr_ref == "B":
                ref = self.last_lr_eff_b
            else:  # "geomean"
                ref = math.sqrt(self.last_lr_eff_a * self.last_lr_eff_b)
            if ref > 0.0:
                lr_tr /= ref
        return lr_tr

    def ab_weight_transfer(self, method: Optional[str] = None) -> None:
        """Memory-optimized A⊗B -> visible transfer, then reinit.

        Transfer: C += transfer_lr * (A @ B)

        Supports per-rank round-robin scheduling via transfer_rank_schedule:
        - "all": Transfer all ranks at once (default, original behavior)
        - "round_robin": Cycle through ranks, transferring transfer_ranks_per_step
          ranks each time. After transfer, only the transferred ranks are reinit'd.

        Args:
            method: Transfer method override.
                   "onehot" - One-hot reading (analog-realistic pulsed update)
                   "direct" - Direct weight access (pulsed update)
                   "set" - Exact weight setting (no pulsed update, precise)
                   If None, use self.transfer_method setting.
        """
        # Compute effective transfer LR once for all sub-methods
        self._eff_transfer_lr = self._compute_effective_transfer_lr()

        # Use instance setting if not specified
        if method is None:
            method = self.transfer_method

        # Compute rank indices to transfer
        if self.transfer_rank_schedule == "round_robin":
            n = self.transfer_ranks_per_step
            rank_indices = [(self._rr_rank_idx + i) % self.rank for i in range(n)]
        else:
            rank_indices = list(range(self.rank))

        # Dispatch to transfer method
        if method == "onehot":
            self._ab_weight_transfer_onehot(rank_indices)
        elif method == "direct":
            self._ab_weight_transfer_direct(rank_indices)
        elif method == "set":
            self._ab_weight_transfer_set(rank_indices)
        else:
            raise ValueError(f"Unknown transfer method: {method}. Use 'onehot', 'direct', or 'set'.")

        # Reinit only the transferred ranks
        self._reinit_ranks(rank_indices)

        # Advance round-robin pointer
        if self.transfer_rank_schedule == "round_robin":
            self._rr_rank_idx = (self._rr_rank_idx + self.transfer_ranks_per_step) % self.rank

    def _ab_weight_transfer_direct(self, rank_indices: List[int]) -> None:
        """Original transfer implementation using direct weight access.

        Args:
            rank_indices: List of rank indices to transfer.
        """
        with torch.no_grad():
            # Get weights (they come in the tile's native device)
            A_weights = self.tile_a.get_weights()[0]  # [d_size, rank]
            B_weights = self.tile_b.get_weights()[0]  # [rank, x_size]

            A_lr = A_weights[:, rank_indices]  # [d_size, len(rank_indices)]
            B_lr = B_weights[rank_indices, :]  # [len(rank_indices), x_size]

            # Transfer in chunks to manage memory
            lr_eff = abs(self._eff_transfer_lr)
            old_lr = self.tile_c.get_learning_rate()
            self.tile_c.set_learning_rate(lr_eff)

            n_sel = len(rank_indices)
            chunk_size = min(self.rank_chunk, n_sel)
            for off in range(0, n_sel, chunk_size):
                end = min(off + chunk_size, n_sel)

                # Pack chunks (keep on same device as tiles)
                D_chunk = A_lr[:, off:end].contiguous()  # [d_size, cur]
                X_chunk = B_lr[off:end, :].contiguous()  # [cur, x_size]

                # Sign rule: PWU computes W += -lr * D @ X^T, we want W += +transfer_lr * D @ X^T
                if self._eff_transfer_lr > 0:
                    D_chunk = -D_chunk

                # Use controller's device (single source of truth)
                dev = self.device
                X_chunk_d = X_chunk.contiguous().to(dev, non_blocking=True)
                D_chunk_t_d = D_chunk.t().contiguous().to(dev, non_blocking=True)

                # Pulsed update to C tile
                if hasattr(self.tile_c, '_orig_update'):
                    self.tile_c._orig_update(X_chunk_d, D_chunk_t_d)
                else:
                    self.tile_c.update(X_chunk_d, D_chunk_t_d)

                del X_chunk_d, D_chunk_t_d

            self.tile_c.set_learning_rate(old_lr)
        self.num_transfers += 1

    def _ab_weight_transfer_set(self, rank_indices: List[int]) -> None:
        """Deterministic FP transfer via tile.update() with NoneWithDevice.

        C tile must be constructed with PulseType.NONE_WITH_DEVICE (set in
        lrtt_tile.py create_update_params). This gives pure FP update with
        device weight clipping, avoiding get_weights/set_weights CPU roundtrips.

        Args:
            rank_indices: List of rank indices to transfer.
        """
        with torch.no_grad():
            # Read raw A, B weights and scale factors
            A_raw = self.tile_a.tile.get_weights().to(self.device)
            B_raw = self.tile_b.tile.get_weights().to(self.device)
            alpha_a = self.tile_a.get_scales()
            alpha_b = self.tile_b.get_scales()
            alpha_c = self.tile_c.get_scales()

            A_lr = A_raw[:, rank_indices]
            if alpha_a is not None:
                A_lr = A_lr * alpha_a.view(-1, 1)
            B_lr = B_raw[rank_indices, :]
            if alpha_b is not None:
                B_lr = B_lr * alpha_b.view(-1, 1)

            if self.enable_diagnostics:
                self.last_transfer_delta = (self._eff_transfer_lr * (A_lr @ B_lr)).detach()

            # Undo alpha_c so update writes correct raw delta
            if alpha_c is not None:
                A_adj = A_lr / alpha_c.view(-1, 1)
            else:
                A_adj = A_lr

            # PWU: W += -lr * d^T @ x  →  we want W += +lr * A_adj @ B_lr
            # So pass d = -A_adj^T, x = B_lr
            old_lr = self.tile_c.tile.get_learning_rate()
            self.tile_c.tile.set_learning_rate(abs(self._eff_transfer_lr))
            self.tile_c.tile.update(
                B_lr.contiguous(),
                (-A_adj).t().contiguous(),
                False, False, False, False,
            )
            self.tile_c.tile.set_learning_rate(old_lr)

        self.num_transfers += 1

    def _read_ab_onehot_symmetric(self, rank_indices: Optional[List[int]] = None) -> tuple:
        """± one-hot differential read with optional AGC and two-amplitude modes.

        Three operation modes based on settings:
        1. Basic (default): Simple ± differential with oversampling (read_n_avg)
        2. AGC: Automatic gain control for optimal amplitude (agc_enabled=True)
        3. Two-Amplitude: Cancel odd offset using two amplitudes (two_amp_enabled=True)

        Uses self.read_n_avg for oversampling (noise reduction by √N).
        - read_n_avg=1: Standard single reading (original behavior)
        - read_n_avg>1: Average N independent readings to reduce stochastic noise

        Also supports legacy num_reads for backward compatibility.

        Returns:
            (A_cols: [d_size, rank], B_rows: [rank, x_size])
        """
        # Ensure one-hot cache exists on correct device
        if self._transfer_vec_a is None or self._transfer_vec_a.device != self.device:
            self._transfer_vec_a = torch.eye(
                self.rank, dtype=self.dtype, device=self.device
            )

        I = self._transfer_vec_a

        # Default: all ranks
        if rank_indices is None:
            rank_indices = list(range(self.rank))

        # Determine read count (new read_n_avg takes precedence over legacy num_reads)
        read_count = self.read_n_avg if self.read_n_avg > 1 else self.num_reads

        # Fast path: batch processing (no AGC, no two_amp, single read)
        if read_count == 1 and not self.agc_enabled and not self.two_amp_enabled:
            I_sel = I[rank_indices]  # [len(rank_indices), rank]
            if self.differential_read:
                A_p = self.tile_a.forward(I_sel)    # [n_sel, d_size]
                A_m = self.tile_a.forward(-I_sel)   # [n_sel, d_size]
                B_p = self.tile_b.backward(I_sel)   # [n_sel, x_size]
                B_m = self.tile_b.backward(-I_sel)  # [n_sel, x_size]
                A_cols = (0.5 * (A_p - A_m)).T  # [d_size, n_sel]
                B_rows = 0.5 * (B_p - B_m)      # [n_sel, x_size]
            else:
                A_cols = self.tile_a.forward(I_sel).T   # [d_size, n_sel]
                B_rows = self.tile_b.backward(I_sel)    # [n_sel, x_size]
            return A_cols, B_rows

        # Slow path: per-rank processing (for AGC, two_amp, or multi-read)
        A_cols = []
        B_rows = []

        for k in rank_indices:
            e = I[k].unsqueeze(0)  # [1, rank], +one-hot

            if self.two_amp_enabled:
                # Two-amplitude read to cancel odd offset
                a_k, _, _ = self._two_amp_read(
                    self.tile_a, e, mode="fwd",
                    read_n_avg=read_count, margin=self.agc_margin
                )
                b_k, _, _ = self._two_amp_read(
                    self.tile_b, e, mode="bwd",
                    read_n_avg=read_count, margin=self.agc_margin
                )
            elif self.agc_enabled:
                # AGC mode: find optimal amplitude, then differential read
                amp_a = self._pick_amp_agc(self.tile_a, e, mode="fwd",
                                           margin=self.agc_margin, max_iters=self.agc_max_iters)
                amp_b = self._pick_amp_agc(self.tile_b, e, mode="bwd",
                                           margin=self.agc_margin, max_iters=self.agc_max_iters)

                a_k = self._diff_read(self.tile_a, e, amp=amp_a, mode="fwd", read_n_avg=read_count).squeeze(0)
                b_k = self._diff_read(self.tile_b, e, amp=amp_b, mode="bwd", read_n_avg=read_count).squeeze(0)
            elif read_count == 1:
                if self.differential_read:
                    # Differential read: cancels DC offset (2x tile ops)
                    a_p = self.tile_a.forward(e).squeeze(0)   # [d_size]
                    a_m = self.tile_a.forward(-e).squeeze(0)  # [d_size]
                    b_p = self.tile_b.backward(e).squeeze(0)  # [x_size]
                    b_m = self.tile_b.backward(-e).squeeze(0) # [x_size]
                    a_k = 0.5 * (a_p - a_m)
                    b_k = 0.5 * (b_p - b_m)
                else:
                    # Simple read: faster but includes DC offset (1x tile ops)
                    a_k = self.tile_a.forward(e).squeeze(0)   # [d_size]
                    b_k = self.tile_b.backward(e).squeeze(0)  # [x_size]
            else:
                # Multi-read averaging with amplitude=1.0
                a_k = self._diff_read(self.tile_a, e, amp=1.0, mode="fwd", read_n_avg=read_count).squeeze(0)
                b_k = self._diff_read(self.tile_b, e, amp=1.0, mode="bwd", read_n_avg=read_count).squeeze(0)

            A_cols.append(a_k)
            B_rows.append(b_k)

        A_cols = torch.stack(A_cols, dim=1)  # [d_size, rank]
        B_rows = torch.stack(B_rows, dim=0)  # [rank, x_size]
        return A_cols, B_rows

    def _center_and_normalize(self, A_cols: Tensor, B_rows: Tensor, eps: float = 1e-8) -> tuple:
        """(선택) 행/열 평균 제거 + 랭크별 ℓ2 정규화.

        Args:
            A_cols: [d_size, rank]
            B_rows: [rank, x_size]
            eps: Numerical stability epsilon

        Returns:
            (A_cols_processed, B_rows_processed)
        """
        if self.transfer_centering:
            A_cols = A_cols - A_cols.mean(dim=0, keepdim=True)
            B_rows = B_rows - B_rows.mean(dim=1, keepdim=True)

        if self.transfer_normalize:
            for k in range(A_cols.shape[1]):
                ak = A_cols[:, k]
                bk = B_rows[k, :]
                na = ak.norm()
                nb = bk.norm()
                if na > eps:
                    A_cols[:, k] = ak / na
                if nb > eps:
                    B_rows[k, :] = bk / nb

        return A_cols, B_rows

    def _ze_norm2_via_gram(self, A_cols: Tensor, B_rows: Tensor) -> float:
        """||Σ_k a_k⊗b_k||_F^2 = sum_{i,j} (a_i^T a_j)*(b_i^T b_j).

        Computes Frobenius norm squared of the outer product sum efficiently
        using Gram matrices without materializing the full [d_size, x_size] matrix.
        """
        G_A = A_cols.t() @ A_cols      # [rank, rank]
        G_B = B_rows @ B_rows.t()      # [rank, rank]
        return (G_A * G_B).sum().item()

    def _ab_weight_transfer_onehot(self, rank_indices: List[int]) -> None:
        """One-hot 기반 전송 (모드에 따라 off, pilot, sigma_delta 방식 선택).

        Transfer modes:
        - "off": No calibration, simple direct transfer (default)
        - "pilot": Pilot-based γ calibration for scale correction
        - "sigma_delta": ΣΔ quantization with residual accumulation

        Args:
            rank_indices: List of rank indices to transfer.
        """
        # 모드 결정 (transfer_mode 우선, 없으면 transfer_gamma_mode 사용)
        mode = getattr(self, 'transfer_mode', None) or self.transfer_gamma_mode

        if mode == "sigma_delta":
            self._ab_weight_transfer_onehot_sigma_delta(rank_indices)
        elif mode == "pilot":
            self._ab_weight_transfer_onehot_pilot(rank_indices)
        else:  # "off" (default)
            self._ab_weight_transfer_onehot_off(rank_indices)

    def _ab_weight_transfer_onehot_pilot(self, rank_indices: List[int]) -> None:
        """One-hot 기반 전송 (± 차분, 파일럿 γ 보정, micro-transfer 포함).

        Hardware-friendly transfer using:
        1. ± one-hot symmetric reading to cancel DC/even-order distortions
        2. Pilot-based γ calibration for scale correction
        3. Micro-transfer for smoother accumulation
        """
        with torch.no_grad():
            # 0) 준비: one-hot 캐시, LR/노이즈 백업
            if self._transfer_vec_a is None:
                self._transfer_vec_a = torch.eye(
                    self.rank, dtype=self.dtype, device=self.device
                )

            old_lr_c = self.tile_c.get_learning_rate()

            # A/B/C 읽기·계측 동안 out_noise=0으로 (out_res 등은 유지)
            # A, B 타일이 FloatingPointTile일 경우 forward/backward 속성이 없음
            old_out_a = getattr(getattr(self.tile_a.rpu_config, 'forward', None), 'out_noise', None)
            old_out_b_f = getattr(getattr(self.tile_b.rpu_config, 'forward', None), 'out_noise', None)
            old_out_b_b = getattr(getattr(self.tile_b.rpu_config, 'backward', None), 'out_noise', None)
            old_out_c = getattr(getattr(self.tile_c.rpu_config, 'forward', None), 'out_noise', None)

            if old_out_a is not None:
                self.tile_a.rpu_config.forward.out_noise = 0.0
            if old_out_b_f is not None:
                self.tile_b.rpu_config.forward.out_noise = 0.0
            if old_out_b_b is not None:
                self.tile_b.rpu_config.backward.out_noise = 0.0
            if old_out_c is not None:
                self.tile_c.rpu_config.forward.out_noise = 0.0

            try:
                # 1) ± one-hot 차분 읽기
                A_cols, B_rows = self._read_ab_onehot_symmetric(rank_indices)

                # (선택) 중심화/정규화
                A_cols, B_rows = self._center_and_normalize(A_cols, B_rows)

                # 2) 파일럿 기반 γ 산출 (스칼라 캘리브레이션)
                Z_norm2 = self._ze_norm2_via_gram(A_cols, B_rows) + 1e-12
                pilot_frac = max(1e-4, float(self.transfer_pilot_frac))
                M_total = max(2, int(self.transfer_micro_steps))
                lr_abs = abs(self._eff_transfer_lr)

                # 파일럿 lr
                lr_pilot = lr_abs * pilot_frac
                gamma = 1.0  # Default gamma

                if self.transfer_gamma_mode == "pilot":
                    # 파일럿 전송으로 실제 스케일 측정
                    num_acc = 0.0

                    def _forward_c(x_row):
                        """Forward C tile with input row vector."""
                        y = self.tile_c.forward(x_row.unsqueeze(0))  # [1, d_size]
                        return y.squeeze(0)

                    for ki in range(A_cols.shape[1]):
                        a_k = A_cols[:, ki]
                        b_k = B_rows[ki, :]

                        # 전송 전 응답
                        y0 = _forward_c(b_k)

                        # 서명 규칙: PWU는 W += -lr * D @ X^T
                        D_k = -a_k if self._eff_transfer_lr > 0 else a_k
                        X_k = b_k

                        # 파일럿 전송 1회
                        self.tile_c.set_learning_rate(lr_pilot)
                        if hasattr(self.tile_c, '_orig_update'):
                            self.tile_c._orig_update(X_k.unsqueeze(0), D_k.unsqueeze(0))
                        else:
                            self.tile_c.update(X_k.unsqueeze(0), D_k.unsqueeze(0))

                        # 전송 후 변화량 계측
                        y1 = _forward_c(b_k)
                        dy = y1 - y0
                        num_acc += (dy * a_k).sum().item()

                    # γ 계산 (측정된 변화 / 예상 변화)
                    gamma = num_acc / (lr_pilot * Z_norm2)
                    # 방어적 클램핑 (극단값/계측 오차 대비)
                    gamma = float(torch.clamp(
                        torch.tensor(gamma, device=self.device), 0.05, 20.0
                    ).item())

                # 3) 본 전송: 남은 lr를 M_rest로 나누어 micro-transfer 수행
                lr_remain = lr_abs * gamma - (lr_pilot if self.transfer_gamma_mode == "pilot" else 0.0)
                lr_remain = max(0.0, lr_remain)
                M_rest = max(1, M_total - 1) if self.transfer_gamma_mode == "pilot" else M_total

                # multi_read_mode에 따른 전송 방식 결정
                num_reads = self.num_reads
                if self.multi_read_mode == "per_read" and num_reads > 1:
                    # per_read 모드: 각 읽기마다 transfer (lr를 num_reads로 나눔)
                    lr_step = lr_remain / (M_rest * num_reads) if M_rest > 0 else 0.0
                    self.tile_c.set_learning_rate(lr_step if lr_step > 0 else 1e-12)

                    I = self._transfer_vec_a
                    for k in rank_indices:
                        e = I[k].unsqueeze(0)

                        for _ in range(num_reads):
                            # 각 읽기마다 새로 읽기
                            a_p = self.tile_a.forward(e).squeeze(0)
                            a_m = self.tile_a.forward(-e).squeeze(0)
                            b_p = self.tile_b.backward(e).squeeze(0)
                            b_m = self.tile_b.backward(-e).squeeze(0)
                            a_k = 0.5 * (a_p - a_m)
                            b_k = 0.5 * (b_p - b_m)

                            D_k = -a_k if self._eff_transfer_lr > 0 else a_k
                            X_k = b_k

                            # M_rest 회 micro-transfer
                            for _ in range(M_rest):
                                if hasattr(self.tile_c, '_orig_update'):
                                    self.tile_c._orig_update(X_k.unsqueeze(0), D_k.unsqueeze(0))
                                else:
                                    self.tile_c.update(X_k.unsqueeze(0), D_k.unsqueeze(0))
                else:
                    # average 모드 (기본): 이미 평균된 A_cols, B_rows 사용
                    lr_step = lr_remain / M_rest if M_rest > 0 else 0.0
                    self.tile_c.set_learning_rate(lr_step if lr_step > 0 else 1e-12)

                    for ki in range(A_cols.shape[1]):
                        a_k = A_cols[:, ki]
                        b_k = B_rows[ki, :]
                        D_k = -a_k if self._eff_transfer_lr > 0 else a_k
                        X_k = b_k

                        # M_rest 회 micro-transfer
                        for _ in range(M_rest):
                            if hasattr(self.tile_c, '_orig_update'):
                                self.tile_c._orig_update(X_k.unsqueeze(0), D_k.unsqueeze(0))
                            else:
                                self.tile_c.update(X_k.unsqueeze(0), D_k.unsqueeze(0))

                # 디버그 로깅 (LRTT_SILENT가 설정되지 않은 경우에만)
                import os
                if self.num_transfers < 1 and not os.environ.get("LRTT_SILENT"):
                    pilot_info = f"lr_pilot={lr_pilot:.3e} " if self.transfer_gamma_mode == "pilot" else ""
                    print(f"[LRTT onehot] gamma={gamma:.3f} {pilot_info}"
                          f"lr_remain={lr_remain:.3e} Znorm2={Z_norm2:.3e}")

            finally:
                # 5) 복구 + 후처리
                self.tile_c.set_learning_rate(old_lr_c)
                if old_out_a is not None:
                    self.tile_a.rpu_config.forward.out_noise = old_out_a
                if old_out_b_f is not None:
                    self.tile_b.rpu_config.forward.out_noise = old_out_b_f
                if old_out_b_b is not None:
                    self.tile_b.rpu_config.backward.out_noise = old_out_b_b
                if old_out_c is not None:
                    self.tile_c.rpu_config.forward.out_noise = old_out_c

        self.num_transfers += 1

    def _ab_weight_transfer_onehot_off(self, rank_indices: List[int]) -> None:
        """One-hot 기반 전송 (off 모드: calibration 없이 직접 전송).

        Simple transfer without pilot calibration or sigma-delta:
        - gamma = 1.0 (fixed)
        - Uses full transfer_lr
        - No calibration overhead
        """
        with torch.no_grad():
            # 1) 준비: one-hot 캐시, LR/노이즈 백업
            if self._transfer_vec_a is None:
                self._transfer_vec_a = torch.eye(
                    self.rank, dtype=self.dtype, device=self.device
                )

            old_lr_c = self.tile_c.get_learning_rate()

            # A/B 읽기 동안 out_noise=0
            old_out_a = getattr(getattr(self.tile_a.rpu_config, 'forward', None), 'out_noise', None)
            old_out_b_f = getattr(getattr(self.tile_b.rpu_config, 'forward', None), 'out_noise', None)
            old_out_b_b = getattr(getattr(self.tile_b.rpu_config, 'backward', None), 'out_noise', None)

            if old_out_a is not None:
                self.tile_a.rpu_config.forward.out_noise = 0.0
            if old_out_b_f is not None:
                self.tile_b.rpu_config.forward.out_noise = 0.0
            if old_out_b_b is not None:
                self.tile_b.rpu_config.backward.out_noise = 0.0

            try:
                # 2) ± one-hot 차분 읽기
                A_cols, B_rows = self._read_ab_onehot_symmetric(rank_indices)
                A_cols, B_rows = self._center_and_normalize(A_cols, B_rows)

                # 3) off 모드: gamma=1.0, 전체 lr 사용
                lr_abs = abs(self._eff_transfer_lr)
                M_total = max(1, int(self.transfer_micro_steps))
                lr_step = lr_abs / M_total
                self.tile_c.set_learning_rate(lr_step if lr_step > 0 else 1e-12)

                # 4) micro-transfer
                for ki in range(A_cols.shape[1]):
                    a_k = A_cols[:, ki]
                    b_k = B_rows[ki, :]
                    D_k = -a_k if self._eff_transfer_lr > 0 else a_k
                    X_k = b_k

                    for _ in range(M_total):
                        if hasattr(self.tile_c, '_orig_update'):
                            self.tile_c._orig_update(X_k.unsqueeze(0), D_k.unsqueeze(0))
                        else:
                            self.tile_c.update(X_k.unsqueeze(0), D_k.unsqueeze(0))

            finally:
                # 5) 복구
                self.tile_c.set_learning_rate(old_lr_c)
                if old_out_a is not None:
                    self.tile_a.rpu_config.forward.out_noise = old_out_a
                if old_out_b_f is not None:
                    self.tile_b.rpu_config.forward.out_noise = old_out_b_f
                if old_out_b_b is not None:
                    self.tile_b.rpu_config.backward.out_noise = old_out_b_b

        self.num_transfers += 1

    def _ab_weight_transfer_onehot_sigma_delta(self, rank_indices: List[int]) -> None:
        """One-hot 기반 전송 (ΣΔ 방식: 랭크별 적분기 h_k + 고정 quantum g).

        Sigma-Delta quantization:
        - 원하는 스칼라 투영: δ_k := |transfer_lr| (랭크별 동일 스칼라)
        - ΣΔ 1차: h_k <- h_k + δ_k; n_k <- round(h_k / g); h_k <- h_k - n_k*g
        - sign rule: tile.update는 W += -lr * D @ X^T → transfer_lr>0일 때 D=-a_k
        - g(quantum): sd_quantum 사용. None이면 g := |transfer_lr| / micro_steps
        """
        with torch.no_grad():
            # 준비: one-hot 캐시, LR/노이즈 백업
            if self._transfer_vec_a is None:
                self._transfer_vec_a = torch.eye(self.rank, dtype=self.dtype, device=self.device)

            old_lr_c = self.tile_c.get_learning_rate()

            # A/B/C 읽기 동안 out_noise=0
            # A, B 타일이 FloatingPointTile일 경우 forward/backward 속성이 없음
            old_out_a = getattr(getattr(self.tile_a.rpu_config, 'forward', None), 'out_noise', None)
            old_out_b_f = getattr(getattr(self.tile_b.rpu_config, 'forward', None), 'out_noise', None)
            old_out_b_b = getattr(getattr(self.tile_b.rpu_config, 'backward', None), 'out_noise', None)
            old_out_c = getattr(getattr(self.tile_c.rpu_config, 'forward', None), 'out_noise', None)

            if old_out_a is not None:
                self.tile_a.rpu_config.forward.out_noise = 0.0
            if old_out_b_f is not None:
                self.tile_b.rpu_config.forward.out_noise = 0.0
            if old_out_b_b is not None:
                self.tile_b.rpu_config.backward.out_noise = 0.0
            if old_out_c is not None:
                self.tile_c.rpu_config.forward.out_noise = 0.0

            try:
                # 1) ± one-hot 차분 읽기: A_cols[d, r], B_rows[r, x]
                A_cols, B_rows = self._read_ab_onehot_symmetric(rank_indices)

                # (선택) 중심화/정규화
                A_cols, B_rows = self._center_and_normalize(A_cols, B_rows)

                # 2) ΣΔ 상태/파라미터 확보
                self._ensure_sd_state()
                lr_abs = float(abs(self._eff_transfer_lr))
                g = float(self.sd_quantum) if (self.sd_quantum is not None and self.sd_quantum > 0.0) \
                    else max(lr_abs / float(max(1, int(self.transfer_micro_steps))), 1e-12)

                # 랭크별 목표 스칼라 δ_k := |_eff_transfer_lr| (선택된 rank만)
                # 3) ΣΔ 적분/정수화: h_k 누적 -> 정수 펄스 n_k, 잔여 갱신
                ri = rank_indices  # selected ranks for this transfer step
                self.sd_acc[ri] = self.sd_acc[ri] + lr_abs  # h_k += δ_k
                n_sel_float = self.sd_acc[ri] / g
                n_sel = torch.round(n_sel_float).to(torch.int64)
                self.sd_acc[ri] = self.sd_acc[ri] - n_sel.to(self.dtype) * g

                # 4) C에 정수 펄스 n_k만큼 전송
                # sign rule: _eff_transfer_lr>0 이면 D=-a_k
                sign = -1.0 if (self._eff_transfer_lr > 0) else 1.0

                # unit pulse의 lr = g 로 통일
                self.tile_c.set_learning_rate(g)

                nonzero = int((n_sel != 0).sum().item())
                max_rep = int(n_sel.abs().max().item()) if nonzero > 0 else 0

                for col_idx in range(len(ri)):
                    reps = int(n_sel[col_idx].item())
                    if reps == 0:
                        continue

                    a_k = (sign * A_cols[:, col_idx]).unsqueeze(0)  # [1, d]
                    b_k = B_rows[col_idx, :].unsqueeze(0)          # [1, x]

                    # 양수/음수 reps 모두 지원: reps<0이면 부호를 D로 흡수
                    if reps < 0:
                        a_k = -a_k
                        reps = -reps

                    # reps 번 unit 업데이트
                    for _ in range(reps):
                        if hasattr(self.tile_c, '_orig_update'):
                            self.tile_c._orig_update(b_k, a_k)
                        else:
                            self.tile_c.update(b_k, a_k)

                # 디버그 (초기 몇 회만)
                if self.num_transfers < 1:  # 거의 출력 안 함
                    res_max = float(self.sd_acc.abs().max().item())
                    print(f"[ΣΔ transfer] g={g:.3e}, nonzero_ranks={nonzero}, max_reps={max_rep}, "
                          f"residual_max<=g/2? {res_max <= 0.5*g + 1e-12} (res_max={res_max:.3e})")

            finally:
                # 복구
                self.tile_c.set_learning_rate(old_lr_c)
                if old_out_a is not None:
                    self.tile_a.rpu_config.forward.out_noise = old_out_a
                if old_out_b_f is not None:
                    self.tile_b.rpu_config.forward.out_noise = old_out_b_f
                if old_out_b_b is not None:
                    self.tile_b.rpu_config.backward.out_noise = old_out_b_b
                if old_out_c is not None:
                    self.tile_c.rpu_config.forward.out_noise = old_out_c

        self.num_transfers += 1

    def forward_inject(
        self,
        x: Tensor,                    # [x_size, m] or [batch, x_size]
        out_trans: bool = False,
        in_trans: bool = False
    ) -> Tensor:
        """Forward inject: y = C·x + lora_alpha * A·(B·x).

        Returns y = C·x + α * A·(B·x) under these rules:
        - If forward_inject_enabled=False or rank=0: visible-only (y = C·x)
        - Default analog-hybrid: y_vis = C·x, g = B·x, y_ab = A·g, y = y_vis + α*y_ab
        - Fallback (transposed): digital composition W_eff = C + α*(A_lr @ B_lr), then W_eff @ x

        Args:
            x: Input tensor [x_size, m] or [batch, x_size]
            out_trans: Output transposed flag
            in_trans: Input transposed flag

        Returns:
            Output tensor [d_size, m] or [batch, d_size]
        """
        # Initialize tiles on first forward if needed
        if not self._tiles_initialized:
            self.reinit()

        # Handle disabled forward injection
        if not self.forward_inject_enabled or self.rank == 0:
            return self.tile_c.forward(x, in_trans=in_trans, out_trans=out_trans)

        # Use unified analog path for all cases (including transpose)
        return self._forward_inject_analog_unified(x, in_trans=in_trans, out_trans=out_trans)

    def _forward_inject_digital_fallback(
        self,
        x: Tensor,
        out_trans: bool,
        in_trans: bool
    ) -> Tensor:
        """Digital fallback: compose W_eff then single forward pass.

        WARNING: This path creates large GPU tensors and can cause OOM!
        The unified analog path should be used instead whenever possible.
        """
        # WARNING: get_weights() can cause memory issues with large models
        C_weights = self.tile_c.get_weights()[0]   # [d_size, x_size]
        A_lr = self.tile_a.get_weights()[0]        # [d_size, rank]
        B_lr = self.tile_b.get_weights()[0]        # [rank, x_size]

        # WARNING: This creates a large intermediate tensor W_eff
        W_eff = C_weights + self.effective_alpha * (A_lr @ B_lr)

        # Set temporary weights and forward
        original_weights = C_weights.clone()
        self.tile_c.set_weights(W_eff)

        try:
            result = self.tile_c.forward(x, bias=False, in_trans=in_trans, out_trans=out_trans)
        finally:
            # Restore original weights
            self.tile_c.set_weights(original_weights)

        return result

    def _forward_inject_analog_hybrid(self, x: Tensor) -> Tensor:
        """Analog-hybrid path using direct weight computation (deterministic).

        Rcolaces non-deterministic tile forward operations with direct matrix computation:
        y = x @ (C^T + α * B^T @ A^T)

        This ensures consistent forward pass behavior for training stability.
        """
        # Get component weights directly
        C_weights = self.tile_c.get_weights()[0]  # [d_size, x_size]
        A_weights = self.tile_a.get_weights()[0][:, :self.rank]  # [d_size, rank]
        B_weights = self.tile_b.get_weights()[0][:self.rank, :]  # [rank, x_size]

        # Compute effective weight matrix: W_eff = C^T + α * B^T @ A^T
        W_eff = C_weights.t() + self.effective_alpha * (B_weights.t() @ A_weights.t())

        # Ensure same device as input
        W_eff = W_eff.to(x.device)

        # Forward pass: y = x @ W_eff
        result = x @ W_eff  # [batch, x_size] @ [x_size, d_size] = [batch, d_size]

        return result

    def _forward_inject_analog_unified(
        self,
        x: Tensor,
        in_trans: bool,
        out_trans: bool
    ) -> Tensor:
        """Unified analog path using proper tile forward operations.

        Uses analog tile forward operations in the correct B→A→C order.
        This ensures analog read constraints (noise/clipping) are applied
        and AnalogSGD's input/error caches work correctly.
        """
        # 1) Normalize input to batch-first
        x_bf = x.t() if in_trans else x  # [batch, x_size]

        # 2) Analog read order guaranteed: B → A → C
        g = self.tile_b.forward(x_bf)      # [batch, rank]
        y_ab = self.tile_a.forward(g)      # [batch, d_size]
        y_c = self.tile_c.forward(x_bf)    # [batch, d_size]

        # 3) Composition
        y = y_c + self.effective_alpha * y_ab   # [batch, d_size]

        # 4) Output transpose
        return y.t() if out_trans else y

    def _update_dynamic_te(self, lr_current: float) -> None:
        """Update transfer_every based on warmup schedule and LR decay ratio.

        Phase 1 (Warmup ramp): TE steps through te_warmup_schedule in equal intervals.
        Phase 2 (Peak LR): TE fixed at te_base.
        Phase 3 (LR decay): TE increases proportionally to LR ratio.

        Formula (Phase 3): TE(t) = clip(te_min, te_max, round(TE_0 * (lr_peak / lr_current)^p))
        """
        self.te_step_counter += 1

        # Phase 1: Warmup ramp (step-wise schedule)
        if (self.te_warmup_schedule and self.te_warmup_steps > 0
                and self.te_step_counter <= self.te_warmup_steps):
            n_phases = len(self.te_warmup_schedule)
            phase_len = self.te_warmup_steps / n_phases
            phase_idx = min(int((self.te_step_counter - 1) / phase_len), n_phases - 1)
            self.transfer_every = self.te_warmup_schedule[phase_idx]
            # Still track lr_peak during warmup
            if lr_current > 0 and (self.lr_peak is None or lr_current > self.lr_peak):
                self.lr_peak = lr_current
            return

        if not self.dynamic_te:
            self.transfer_every = self.te_base
            return

        if lr_current <= 0:
            return

        # Phase 2: Peak tracking (lr still increasing after warmup)
        if self.lr_peak is None or lr_current > self.lr_peak:
            self.lr_peak = lr_current
            self.transfer_every = self.te_base
            return

        # Phase 3: Decay — increase TE proportionally
        ratio = self.lr_peak / lr_current  # >= 1.0
        te_proposed = round(self.te_base * (ratio ** self.dynamic_te_power))
        te_proposed = max(self.dynamic_te_min, min(self.dynamic_te_max, te_proposed))

        # Monotonic non-decreasing
        self.transfer_every = max(self.transfer_every, te_proposed)

    def should_transfer(self) -> bool:
        """Check if transfer should occur based on counter and schedule.

        Uses the same modulo-based boundary crossing check as TikiTaka
        (see transfer.py and rpu_transfer_device.cpp):
        - Counter always increments by m_batch (never reset).
        - effective_te = transfer_every * m_batch when units_in_mbatch=True.
        - Transfer fires when the current batch crosses an effective_te boundary.
        This avoids premature transfer when the last mini-batch is smaller.
        """
        effective_te = self.transfer_every
        if self.units_in_mbatch:
            effective_te = int(math.ceil(self.transfer_every * self._last_m_batch))
        if effective_te <= 0:
            return False
        rest_count = ((self.transfer_counter - self._last_m_batch) % effective_te) + self._last_m_batch
        return rest_count >= effective_te

    def reset_transfer_counter(self) -> None:
        """Reset transfer counter (called after transfer)."""
        self.transfer_counter = 0

    def get_state_dict(self) -> Dict[str, Any]:
        """Get controller state for serialization."""
        return {
            'transfer_counter': self.transfer_counter,
            'num_a_updates': self.num_a_updates,
            'num_b_updates': self.num_b_updates,
            'num_transfers': self.num_transfers,
            'd_size': self.d_size,
            'x_size': self.x_size,
            'rank': self.rank,
            'transfer_lr': self.transfer_lr,
            'transfer_every': self.transfer_every,
            'units_in_mbatch': self.units_in_mbatch,
            'lora_alpha': self.lora_alpha,
            'reinit_gain': self.reinit_gain,
            'reinit_mode': self.reinit_mode,
            'forward_inject_enabled': self.forward_inject_enabled,
            'lr_peak': self.lr_peak,
            'te_step_counter': self.te_step_counter,
            'transfer_rank_schedule': self.transfer_rank_schedule,
            'transfer_ranks_per_step': self.transfer_ranks_per_step,
            '_rr_rank_idx': self._rr_rank_idx,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load controller state from serialization."""
        # Handle backward compatibility for old 'forward_inject' key
        if 'forward_inject' in state_dict and 'forward_inject_enabled' not in state_dict:
            state_dict['forward_inject_enabled'] = state_dict.pop('forward_inject')

        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def set_device(self, device: torch.device) -> None:
        """Set device and clear buffers for reallocation.

        Args:
            device: Target device (CPU or CUDA)
        """
        self.device = torch.device(device)
        # Clear buffers so they get reallocated on the new device
        self._x_b_buffer = None
        self._d_a_buffer = None
        self._x_pad = None
        self._d_pad = None
        # Clear transfer vectors (one-hot reading)
        self._transfer_vec_a = None
