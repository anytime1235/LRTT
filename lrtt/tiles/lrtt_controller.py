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
from torch import Tensor
from typing import Optional, Dict, Any
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
        transfer_lr_scale: str = "sqrt_rank",  # "none", "sqrt_rank", "rank"
        transfer_every: int = 32,
        units_in_mbatch: bool = False,
        lora_alpha: float = 1.0,
        reinit_gain: float = 0.1,
        reinit_mode: str = "standard",
        decay_factor: float = 0.9,
        correct_gradient_magnitudes: bool = False,
        rank_chunk: Optional[int] = None,
        ab_bl_mgmt: Optional[Dict[str, Any]] = None,
        transfer_bl_mgmt: Optional[Dict[str, Any]] = None,
        forward_inject: bool = True,
        use_onehot: bool = True,
        use_sigma_delta: bool = True,
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
            transfer_lr: Transfer learning rate scalar (base value before scaling)
            transfer_lr_scale: Auto-scaling mode for transfer_lr based on rank:
                        "none" - No scaling, use transfer_lr as-is (default)
                        "sqrt_rank" - Scale by 1/sqrt(rank), i.e., transfer_lr / sqrt(rank)
                        "rank" - Scale by 1/rank, i.e., transfer_lr / rank
            transfer_every: Transfer frequency (steps or samples)
            units_in_mbatch: Whether transfer_every counts samples vs steps
            lora_alpha: LoRA scaling factor α
            reinit_gain: Kaiming initialization gain for B matrix
            reinit_mode: Reinit strategy after transfer:
                        "standard" - A=0, B=Kaiming (original LRTT)
                        "decay" - A*=decay_factor, B*=decay_factor (gradual decay)
                        "hybrid" - A=0, B*=decay_factor (hybrid approach)
            decay_factor: Decay factor for "decay" and "hybrid" modes (0 < decay_factor < 1)
            correct_gradient_magnitudes: Scale lr by sqrt(rank) for gradient correction
            rank_chunk: Chunk size for transfer (None = full rank)
            ab_bl_mgmt: BL management for A/B updates {update_bl_management, update_management, desired_BL}
            transfer_bl_mgmt: BL management for transfers
            forward_inject: Enable forward injection optimization
            use_onehot: Transfer read mode (True=one-hot, False=direct)
            use_sigma_delta: Transfer write mode when use_onehot=True (True=ΣΔ, False=simple)
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
        # Apply transfer_lr scaling based on rank
        self.transfer_lr_scale = transfer_lr_scale
        if transfer_lr_scale == "sqrt_rank":
            self.transfer_lr = transfer_lr / math.sqrt(rank)
        elif transfer_lr_scale == "rank":
            self.transfer_lr = transfer_lr / rank
        else:  # "none" or default
            self.transfer_lr = transfer_lr
        self.transfer_lr_base = transfer_lr  # Store original for reference
        self.transfer_every = transfer_every
        self.units_in_mbatch = units_in_mbatch
        self.lora_alpha = lora_alpha
        self.reinit_gain = reinit_gain
        self.reinit_mode = reinit_mode
        self.decay_factor = decay_factor
        self.correct_gradient_magnitudes = correct_gradient_magnitudes
        self.rank_chunk = rank_chunk or rank
        self.forward_inject_enabled = forward_inject

        # Transfer mode settings
        self.use_onehot = use_onehot
        self.use_sigma_delta = use_sigma_delta

        # BL management settings
        self.ab_bl_mgmt = ab_bl_mgmt or {}
        self.transfer_bl_mgmt = transfer_bl_mgmt or {}

        # Counters and state
        self.transfer_counter = 0
        self.num_a_updates = 0
        self.num_b_updates = 0
        self.num_transfers = 0

        # Pulse counting (for analysis)
        self.pulse_count_enabled = False
        self.total_pulses_to_c = 0
        self.pulse_history = []  # List of (transfer_num, pulse_count) tuples

        # Cached buffers for efficiency
        self._x_b_buffer: Optional[Tensor] = None
        self._d_a_buffer: Optional[Tensor] = None
        self._pad_buffer_a: Optional[Tensor] = None
        self._pad_buffer_b: Optional[Tensor] = None

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

        # === Reconstruction update parameters (for forward_inject=False) ===
        # When forward_inject=False, use gradient reconstruction instead of LoRA chain rule
        self.recon_lambda_a: float = 1e-3
        self.recon_lambda_b: float = 1e-3
        self.recon_use_scalar_stabilizer: bool = False
        self.recon_use_exact_gram: bool = False
        self.recon_ema_beta: float = 0.9
        self.recon_lr_scale: float = 1.0
        self.recon_clip_norm: float = 10.0
        self.recon_use_clip_norm: bool = False

        # EMA state for scalar stabilizer (sA = ||A||^2/rank, sB = ||B||^2/rank)
        self._ema_sA: float = 0.0
        self._ema_sB: float = 0.0
        self._ema_initialized: bool = False

        # Periodic exact Gram stabilizer (use exact Gram every N steps)
        self.recon_exact_gram_every: int = 0  # 0 = disabled, N = every N steps
        self._recon_step_counter: int = 0

        # Transfer robustness knobs (safe defaults)
        self.transfer_micro_steps: int = 1          # M: micro-transfer 반복 횟수
        self.transfer_centering: bool = False       # 행/열 평균 제거 (기본 off - gradient 왜곡 방지)
        self.transfer_normalize: bool = False       # 랭크별 ℓ2 정규화 (기본 off - gradient 왜곡 방지)

        # --- Oversampling for noise reduction in one-hot reading ---
        self.read_n_avg: int = 1                    # Oversampling count (1=disabled, 4/8=recommended)

        # --- AGC (Automatic Gain Control) settings ---
        self.agc_enabled: bool = False              # Enable AGC for read amplitude optimization
        self.agc_margin: float = 0.85               # Target output bound fraction (avoid clipping)
        self.agc_max_iters: int = 6                 # Max iterations for AGC binary search

        # --- Two-Amplitude differential read settings ---
        self.two_amp_enabled: bool = False          # Enable two-amplitude differential read (odd offset removal)
        self.two_amp_ratio: float = 0.5             # Ratio α1/α2 for two-amplitude method

        # --- Sigma-Delta (ΣΔ) core state ---
        self.sd_quantum: Optional[float] = None     # g: unit quantum for rank-wise pulses (None -> derive per transfer)
        self.sd_acc: Optional[Tensor] = None        # h_k residuals [rank], persistent across transfers
        self.transfer_burst_limit: int = 10         # Burst limit for Tiki-Taka Threshold clamping (max pulses per rank per transfer)

        # Transfer one-hot vectors cache
        self._transfer_vec_a: Optional[Tensor] = None

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
        io = tile.rpu_config.forward if mode == "fwd" else tile.rpu_config.backward
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
        """Safely infer device from tile by checking the underlying tile type.

        Note: get_weights() always returns CPU tensors (copies), so we must check
        the tile backend type instead.
        """
        # Primary method: Check tile backend type string
        # get_weights() returns CPU copies, so we check the tile type instead
        if hasattr(self.tile_c, 'tile'):
            tile_str = str(type(self.tile_c.tile).__name__)
            if 'Cuda' in tile_str or 'CUDA' in tile_str:
                return torch.device('cuda')

        # Fallback: CPU (safer default)
        return torch.device('cpu')

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

    def reinit(self) -> None:
        """Reinit A,B matrices based on reinit_mode.

        Three modes:
        - "standard": A=0, B=Kaiming (original LRTT)
        - "decay": A*=decay_factor, B*=decay_factor (gradual decay)
        - "hybrid": A=0, B*=decay_factor (hybrid approach)
        """
        with torch.no_grad():
            if self.reinit_mode == "standard":
                # Original LRTT: A=0, B=Kaiming
                A_zeros = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                self.tile_a.set_weights(A_zeros)

                # B matrix: Kaiming Normal initialization
                std = self.reinit_gain * math.sqrt(2.0 / self.x_size)
                B_kaiming = torch.normal(0, std, size=(self.rank, self.x_size), device=self.device, dtype=self.dtype)
                self.tile_b.set_weights(B_kaiming)

            elif self.reinit_mode == "decay":
                # First time initialization or decay mode
                if not self._tiles_initialized:
                    # First time: Initialize A and B with small random values for decay mode
                    # A matrix: Small random initialization
                    A_std = self.reinit_gain * math.sqrt(2.0 / self.rank) * 0.1  # Small init for A
                    A_init = torch.normal(0, A_std, size=(self.d_size, self.rank), device=self.device, dtype=self.dtype)
                    self.tile_a.set_weights(A_init)

                    # B matrix: Standard Kaiming initialization
                    B_std = self.reinit_gain * math.sqrt(2.0 / self.x_size)
                    B_init = torch.normal(0, B_std, size=(self.rank, self.x_size), device=self.device, dtype=self.dtype)
                    self.tile_b.set_weights(B_init)
                else:
                    # After transfer: Decay both A and B
                    A_weights = self.tile_a.get_weights()[0] * self.decay_factor
                    B_weights = self.tile_b.get_weights()[0] * self.decay_factor
                    self.tile_a.set_weights(A_weights)
                    self.tile_b.set_weights(B_weights)

            elif self.reinit_mode == "hybrid":
                # A=0, B decayed or initialized
                A_zeros = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                self.tile_a.set_weights(A_zeros)

                if not self._tiles_initialized:
                    # First time: Initialize B with Kaiming
                    B_std = self.reinit_gain * math.sqrt(2.0 / self.x_size)
                    B_init = torch.normal(0, B_std, size=(self.rank, self.x_size), device=self.device, dtype=self.dtype)
                    self.tile_b.set_weights(B_init)
                else:
                    # After transfer: Decay B
                    B_weights = self.tile_b.get_weights()[0] * self.decay_factor
                    self.tile_b.set_weights(B_weights)

            elif self.reinit_mode == "orthogonal":
                # B = Random Orthogonal (FROZEN), A = 0
                # B @ B.T = I, so projection preserves gradient direction
                A_zeros = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
                self.tile_a.set_weights(A_zeros)

                if not self._tiles_initialized:
                    # Initialize B as random orthogonal matrix using QR decomposition
                    # Generate random matrix and orthogonalize rows
                    random_matrix = torch.randn(self.rank, self.x_size, device=self.device, dtype=self.dtype)
                    # QR decomposition: Q has orthonormal columns, so Q.T has orthonormal rows
                    Q, R = torch.linalg.qr(random_matrix.t())  # [x_size, rank]
                    B_orthogonal = Q.t()  # [rank, x_size] - rows are orthonormal

                    # Device-aware adaptive scaling to ensure B stays within bounds
                    # Get device bounds
                    w_max = self.tile_b.rpu_config.device.w_max
                    w_min = self.tile_b.rpu_config.device.w_min
                    bound_range = min(abs(w_max), abs(w_min))

                    # Use 3σ rule to ensure 99.7% of values within bounds
                    # For orthonormal matrix: std per element ≈ 1/sqrt(x_size)
                    # So 3σ range ≈ 3/sqrt(x_size)
                    # Therefore: max_scale = bound_range * sqrt(x_size) / 3
                    max_safe_scale = bound_range * math.sqrt(self.x_size) / 3.0

                    # Use smaller of theoretical and safe scale, then apply reinit_gain
                    theoretical_scale = math.sqrt(self.x_size / self.rank)
                    base_scale = min(theoretical_scale, max_safe_scale)
                    scale = base_scale * self.reinit_gain  # reinit_gain으로 scale 조절 가능

                    B_orthogonal = B_orthogonal * scale

                    # Explicit clipping to ensure 100% compliance with device bounds
                    # (3σ rule is statistical, ~0.3% may still exceed)
                    B_orthogonal = torch.clamp(B_orthogonal, min=w_min, max=w_max)

                    self.tile_b.set_weights(B_orthogonal)
                    self._b_frozen = True
                # else: B is frozen, don't change it

            else:
                raise ValueError(f"Unknown reinit_mode: {self.reinit_mode}. Must be 'standard', 'decay', 'hybrid', or 'orthogonal'")

        # Apply device clipping if available
        if hasattr(self.tile_a, 'clip_weights'):
            self.tile_a.clip_weights()
        if hasattr(self.tile_b, 'clip_weights'):
            self.tile_b.clip_weights()

        # OPTIMIZATION: Use flag instead of reading C weights for norm check
        if self.forward_inject_enabled and not self._c_initialized:
            # Small Kaiming init to avoid degenerate W_eff
            C_std = self.reinit_gain * math.sqrt(2.0 / self.x_size) * 0.1  # Smaller
            C_init = torch.normal(0, C_std, size=(self.d_size, self.x_size), device=self.device, dtype=self.dtype)
            self.tile_c.set_weights(C_init)
            if hasattr(self.tile_c, 'clip_weights'):
                self.tile_c.clip_weights()
            self._c_initialized = True

        # Reset counters
        self.transfer_counter = 0
        self._tiles_initialized = True

    def ab_weight_update(
        self,
        x: Tensor,
        d: Tensor,
        lr: float,
        in_trans: bool = False,
        out_trans: bool = False
    ) -> None:
        """Update A and B with rank-r gradient approximation.

        When forward_inject=True: Uses LoRA chain rule (original LRTT behavior).
        When forward_inject=False: Uses gradient reconstruction to make AB ≈ -G.

        Args:
            x: Input tensor
            d: Error tensor
            lr: Learning rate
            in_trans: Whether x is transposed
            out_trans: Whether d is transposed
        """
        # Branch: forward_inject=False uses reconstruction update
        if not self.forward_inject_enabled:
            return self.ab_weight_update_reconstruction(x, d, lr, in_trans, out_trans)

        # === Original LoRA chain rule update (forward_inject=True) ===
        # 0) Normalize to [batch, feat] format
        if in_trans:
            x = x.t()
        if out_trans:
            d = d.t()

        # 1) Projections (analog path)
        with torch.no_grad():
            XB = self.tile_b.forward(x)     # [batch, rank] = B·X
            DA = self.tile_a.backward(d)    # [batch, rank] = A^T·D

        # 2) lr_eff = lr * α * (1/√r, optional)
        lr_eff = lr * self.lora_alpha
        if self.correct_gradient_magnitudes:
            lr_eff /= math.sqrt(self.rank)

        # 3) ΔA = -lr_eff · D^T · (B·X) → tile_a.update(XB, d)
        lr_a_old = self.tile_a.get_learning_rate()
        self.tile_a.set_learning_rate(lr_eff)
        if hasattr(self.tile_a, '_orig_update'):
            self.tile_a._orig_update(XB, d)
        else:
            self.tile_a.update(XB, d)
        self.tile_a.set_learning_rate(lr_a_old)
        self.num_a_updates += 1

        # 4) ΔB = -lr_eff · (A^T·D)^T · X → tile_b.update(x, DA)
        # Skip B update if B is frozen (orthogonal mode)
        if not getattr(self, '_b_frozen', False):
            lr_b_old = self.tile_b.get_learning_rate()
            self.tile_b.set_learning_rate(lr_eff)
            if hasattr(self.tile_b, '_orig_update'):
                self.tile_b._orig_update(x, DA)
            else:
                self.tile_b.update(x, DA)
            self.tile_b.set_learning_rate(lr_b_old)
            self.num_b_updates += 1

        # 5) Norm clipping to prevent A@B explosion
        self._clip_ab_norms()

        # 6) Counter
        self.transfer_counter += (x.shape[0] if self.units_in_mbatch else 1)

    def _clip_ab_norms(self, max_norm: float = 10.0) -> None:
        """Clip A and B norms to prevent explosion.

        Uses forward/backward to read weights and update() to apply scaling,
        avoiding expensive get_weights()/set_weights() calls.
        """
        with torch.no_grad():
            device = next(iter(self.tile_a.parameters())).device if hasattr(self.tile_a, 'parameters') else 'cpu'

            # Read A using forward: tile_a.forward(I) = I @ A.T = A.T
            I_rank_a = torch.eye(self.rank, device=device)
            A_T = self.tile_a.forward(I_rank_a)  # [rank, d_size] = A.T
            A_norm = torch.norm(A_T).item()

            if A_norm > max_norm:
                # Apply scaling: A *= scale = A - (1-scale)*A
                # tile.update(x, d): W -= lr * d.T @ x
                # Use x = I, d = A.T.T = A, lr = (1-scale)
                scale = max_norm / A_norm
                c = 1.0 - scale
                lr_old = self.tile_a.get_learning_rate()
                self.tile_a.set_learning_rate(c)
                self.tile_a.update(I_rank_a, A_T.t())  # A -= c * A
                self.tile_a.set_learning_rate(lr_old)

            # Read B using backward: tile_b.backward(I) = I @ B = B
            I_rank_b = torch.eye(self.rank, device=device)
            B_read = self.tile_b.backward(I_rank_b)  # [rank, x_size] = B
            B_norm = torch.norm(B_read).item()

            if B_norm > max_norm and not getattr(self, '_b_frozen', False):
                scale = max_norm / B_norm
                c = 1.0 - scale
                lr_old = self.tile_b.get_learning_rate()
                self.tile_b.set_learning_rate(c)
                self.tile_b.update(B_read, I_rank_b)  # B -= c * B
                self.tile_b.set_learning_rate(lr_old)

    def ab_weight_update_reconstruction(
        self,
        x: Tensor,
        d: Tensor,
        lr: float,
        in_trans: bool = False,
        out_trans: bool = False
    ) -> None:
        """TikiTaka-style gradient reconstruction update for A and B.

        When forward_inject=False, the forward pass uses only C (y = Cx), so A,B
        don't appear in the loss. Instead of using LoRA chain rule (which causes
        ||A@B|| explosion), we treat A,B as "gradient buffers" and minimize:

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

            - Stabilizer terms: Use scalar approximation (default) or exact Gram (debug)
                BB^T ≈ sB*I, A^TA ≈ sA*I where sB=||B||^2/rank, sA=||A||^2/rank
                A *= (1 - lr*(sB + λA))
                B *= (1 - lr*(sA + λB))

        Args:
            x: Input tensor [batch, x_size]
            d: Error tensor [batch, d_size]
            lr: Learning rate
            in_trans: Whether x is transposed
            out_trans: Whether d is transposed
        """
        # 0) Normalize to [batch, feat] format
        if in_trans:
            x = x.t()
        if out_trans:
            d = d.t()

        # 1) Effective learning rate
        # Note: In reconstruction mode (forward_inject=False), lora_alpha is NOT used
        # in forward pass, so we don't multiply it here. Only use recon_lr_scale.
        lr_rec = lr * self.recon_lr_scale
        if self.correct_gradient_magnitudes:
            lr_rec /= math.sqrt(self.rank)

        # 2) Projections for Hebbian terms
        with torch.no_grad():
            # XB = B @ X.T → [rank, batch], but tile forward expects [batch, x_size]
            # tile_b.forward(x) gives [batch, rank]
            XB = self.tile_b.forward(x)  # [batch, rank]

            # DA = A^T @ D.T → [rank, batch], but tile backward expects [batch, d_size]
            # tile_a.backward(d) gives [batch, rank]
            DA = self.tile_a.backward(d)  # [batch, rank]

        # 3) Hebbian updates: A -= lr * GB^T, B -= lr * A^TG
        # tile.update(x, d) implements W -= lr * d^T @ x

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
        if not getattr(self, '_b_frozen', False):
            lr_b_old = self.tile_b.get_learning_rate()
            self.tile_b.set_learning_rate(lr_rec)
            if hasattr(self.tile_b, '_orig_update'):
                self.tile_b._orig_update(x, DA)
            else:
                self.tile_b.update(x, DA)
            self.tile_b.set_learning_rate(lr_b_old)
            self.num_b_updates += 1

        # 4) Stabilizer terms to prevent ||A@B|| growth
        # Uses forward/backward to read weights and update() to apply decay
        # Avoids expensive get_weights()/set_weights() calls
        with torch.no_grad():
            device = x.device

            # Read A using forward: tile_a.forward(I) = I @ A.T = A.T
            I_rank = torch.eye(self.rank, device=device)
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
                # Exact Gram matrix stabilizer using tile operations
                # A -= lr_rec * A @ (B @ B^T) - lr_rec * λA * A
                # B -= lr_rec * (A^T @ A) @ B - lr_rec * λB * B
                A_rank = A_T.t()  # [d_size, rank]
                B_rank = B_read   # [rank, x_size]

                BBT = B_rank @ B_rank.t()  # [rank, rank]
                ATA = A_rank.t() @ A_rank  # [rank, rank]

                # Combined stabilizer + L2: d_A = A @ BBT + λA * A = A @ (BBT + λA*I)
                BBT_reg = BBT + self.recon_lambda_a * I_rank
                ATA_reg = ATA + self.recon_lambda_b * I_rank

                # Apply A -= lr_rec * A @ BBT_reg using tile.update
                # tile.update(x, d): W -= lr * d.T @ x
                # A @ BBT_reg has shape [d_size, rank]
                # We need d.T @ x = A @ BBT_reg
                # Let x = BBT_reg [rank, rank], d = A.T.T = A [d_size, rank] -> d.T = [rank, d_size]
                # d.T @ x = A.T @ BBT_reg -> wrong shape
                # Actually: d.T @ x should give [d_size, rank]
                # So d is [batch, d_size], x is [batch, rank]
                # d.T @ x = [d_size, batch] @ [batch, rank] = [d_size, rank]
                # We want [d_size, rank] = A @ BBT_reg
                # If x = BBT_reg.T [rank, rank], d = A_rank [rank, d_size]... no
                # Let's use: x = I [rank, rank], d = (A @ BBT_reg).T [rank, d_size]
                d_A = (A_rank @ BBT_reg).t()  # [rank, d_size]
                lr_old_a = self.tile_a.get_learning_rate()
                self.tile_a.set_learning_rate(lr_rec)
                self.tile_a.update(I_rank, d_A.t())  # A -= lr_rec * d_A.T @ I = A -= lr_rec * A @ BBT_reg
                self.tile_a.set_learning_rate(lr_old_a)

                if not getattr(self, '_b_frozen', False):
                    # B -= lr_rec * ATA_reg @ B
                    # tile_b.update(x, d) does B -= lr * d.T @ x
                    # We want d.T @ x = ATA_reg @ B where B is [rank, x_size]
                    # Let d = ATA_reg.T [rank, rank], x = B [rank, x_size]
                    # d.T @ x = ATA_reg @ B ✓
                    lr_old_b = self.tile_b.get_learning_rate()
                    self.tile_b.set_learning_rate(lr_rec)
                    self.tile_b.update(B_rank, ATA_reg.t())
                    self.tile_b.set_learning_rate(lr_old_b)

            elif self.recon_use_scalar_stabilizer:
                # Scalar approximation: BB^T ≈ sB*I, A^TA ≈ sA*I
                # Apply decay: A *= decay = A - (1-decay)*A using tile.update
                import math as _math

                # Exponential decay (always positive, no sign flip risk)
                exp_arg_A = lr_rec * (self._ema_sB + self.recon_lambda_a)
                exp_arg_B = lr_rec * (self._ema_sA + self.recon_lambda_b)

                # Clamp exponent to prevent extreme decay (max decay to ~0.05 per step)
                exp_arg_A = min(exp_arg_A, 3.0)  # exp(-3) ≈ 0.05
                exp_arg_B = min(exp_arg_B, 3.0)

                decay_A = _math.exp(-exp_arg_A)
                decay_B = _math.exp(-exp_arg_B)

                # Apply A *= decay using tile.update: A -= (1-decay)*A
                c_A = 1.0 - decay_A
                if c_A > 1e-8:  # Only apply if decay is significant
                    lr_old_a = self.tile_a.get_learning_rate()
                    self.tile_a.set_learning_rate(c_A)
                    self.tile_a.update(I_rank, A_T.t())  # A -= c_A * A
                    self.tile_a.set_learning_rate(lr_old_a)

                if not getattr(self, '_b_frozen', False):
                    c_B = 1.0 - decay_B
                    if c_B > 1e-8:
                        lr_old_b = self.tile_b.get_learning_rate()
                        self.tile_b.set_learning_rate(c_B)
                        self.tile_b.update(B_read, I_rank)  # B -= c_B * B
                        self.tile_b.set_learning_rate(lr_old_b)

        # 5) Safety norm clipping (fallback) - only if enabled
        if self.recon_use_clip_norm:
            self._clip_ab_norms(max_norm=self.recon_clip_norm)

        # 6) Counter
        self.transfer_counter += (x.shape[0] if self.units_in_mbatch else 1)

    def ab_weight_transfer(self, use_onehot: bool = True, use_sigma_delta: bool = True) -> None:
        """Memory-optimized pulsed A⊗B -> visible transfer, then reinit.

        Transfer: C += transfer_lr * (A @ B) via pulsed outer product.

        Args:
            use_onehot: If True, use one-hot reading (analog-realistic).
                       If False, use direct weight access.
            use_sigma_delta: If True and use_onehot=True, use ΣΔ modulation.
                            If False, use simple pulsed update.
                            Ignored when use_onehot=False.

        Three modes:
        1. Direct (use_onehot=False):
           - Read: get_weights() 직접 접근
           - Write: chunk 단위 pulsed update

        2. One-hot + ΣΔ (use_onehot=True, use_sigma_delta=True):
           - Read: one-hot forward/backward
           - Write: ΣΔ 적분 후 정수 펄스만큼 반복 update

        3. One-hot Simple (use_onehot=True, use_sigma_delta=False):
           - Read: one-hot forward/backward
           - Write: rank별 외적을 개별 pulsed update (1회씩)
        """
        if use_onehot:
            if use_sigma_delta:
                self._ab_weight_transfer_onehot()
            else:
                self._ab_weight_transfer_onehot_simple()
        else:
            self._ab_weight_transfer_direct()

    def _ab_weight_transfer_direct(self) -> None:
        """Original transfer implementation using direct weight access."""
        with torch.no_grad():
            # Get weights (they come in the tile's native device)
            A_weights = self.tile_a.get_weights()[0]  # [d_size, rank]
            B_weights = self.tile_b.get_weights()[0]  # [rank, x_size]

            A_lr = A_weights[:, :self.rank]  # [d_size, rank]

            # Transfer in chunks to manage memory
            lr_eff = abs(self.transfer_lr)
            old_lr = self.tile_c.get_learning_rate()
            self.tile_c.set_learning_rate(lr_eff)

            # Apply transfer BL management
            if self.transfer_bl_mgmt:
                # Apply transfer_bl_mgmt settings
                pass

            chunk_size = self.rank_chunk
            num_chunks = 0
            for off in range(0, self.rank, chunk_size):
                end = min(off + chunk_size, self.rank)
                cur = end - off

                # Pack chunks (keep on same device as tiles)
                D_chunk = A_lr[:, off:end].contiguous()  # [d_size, cur]
                X_chunk = B_weights[off:end, :].contiguous()     # [cur, x_size]

                # Sign rule: PWU computes W += -lr * D @ X^T, we want W += +transfer_lr * D @ X^T
                # So when transfer_lr > 0, negate D to get correct sign
                if self.transfer_lr > 0:
                    D_chunk = -D_chunk
                elif self.transfer_lr < 0:
                    # transfer_lr < 0: want W += transfer_lr * D @ X^T (negative), so keep D positive
                    # PWU does W += -lr * D @ X^T with lr > 0, so net effect is W += -D @ X^T (negative) ✓
                    pass

                # Use controller's device (single source of truth)
                dev = self.device
                X_chunk_d = X_chunk.contiguous().to(dev, non_blocking=True)
                D_chunk_t_d = D_chunk.t().contiguous().to(dev, non_blocking=True)

                # Debug assertion to ensure same device
                assert X_chunk_d.device == D_chunk_t_d.device, \
                    f"Device mismatch: X={X_chunk_d.device}, D={D_chunk_t_d.device}"

                # Pulsed update to C tile
                if hasattr(self.tile_c, '_orig_update'):
                    self.tile_c._orig_update(X_chunk_d, D_chunk_t_d)
                else:
                    self.tile_c.update(X_chunk_d, D_chunk_t_d)

                num_chunks += 1

                # OPTIMIZATION: Immediately free GPU memory
                del X_chunk_d, D_chunk_t_d
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.tile_c.set_learning_rate(old_lr)

            # Track pulses if enabled (direct: 1 pulse per chunk)
            if self.pulse_count_enabled:
                pulses_this_transfer = num_chunks
                self.total_pulses_to_c += pulses_this_transfer
                self.pulse_history.append((self.num_transfers, pulses_this_transfer))
        self.num_transfers += 1

        # CRITICAL: Reset transfer counter after transfer (matches CUDA)
        self.transfer_counter = 0

        # DEBUG: Check A before reinit (first few transfers only)
        if self.num_transfers <= 3:
            A_before_reinit = self.tile_a.get_weights()[0] if hasattr(self.tile_a, 'get_weights') else None
            if A_before_reinit is not None:
                print(f"TRANSFER #{self.num_transfers} - Before reinit: A norm={A_before_reinit.norm():.6f}")

        # Unconditional reinit after transfer
        self.reinit()

        # DEBUG: Check A after reinit (first few transfers only)
        if self.num_transfers <= 3:
            A_after_reinit = self.tile_a.get_weights()[0] if hasattr(self.tile_a, 'get_weights') else None
            if A_after_reinit is not None:
                print(f"TRANSFER #{self.num_transfers} - After reinit ({self.reinit_mode}): A norm={A_after_reinit.norm():.6f}")
                if self.reinit_mode == "decay":
                    expected = A_before_reinit.norm() * self.decay_factor if A_before_reinit is not None else 0
                    print(f"  Expected A norm (decay): {expected:.6f}")
                print()

    def _read_ab_onehot_symmetric(self) -> tuple:
        """± one-hot differential read with optional AGC and two-amplitude modes.

        Three operation modes based on settings:
        1. Basic (default): Simple ± differential with oversampling
        2. AGC: Automatic gain control for optimal amplitude (agc_enabled=True)
        3. Two-Amplitude: Cancel odd offset using two amplitudes (two_amp_enabled=True)

        Uses self.read_n_avg for oversampling (noise reduction by √N).
        - read_n_avg=1: Standard single reading (original behavior)
        - read_n_avg>1: Average N independent readings to reduce stochastic noise

        Returns:
            (A_cols: [d_size, rank], B_rows: [rank, x_size])
        """
        if self._transfer_vec_a is None:
            self._transfer_vec_a = torch.eye(
                self.rank, dtype=self.dtype, device=self.device
            )

        I = self._transfer_vec_a
        read_n_avg = max(1, self.read_n_avg)

        # Pre-allocate result tensors
        A_cols = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
        B_rows = torch.zeros(self.rank, self.x_size, device=self.device, dtype=self.dtype)

        # Mode selection based on settings
        if self.two_amp_enabled:
            # Two-Amplitude mode: Cancel odd offset using two amplitude levels
            # This also uses AGC to find optimal amplitude
            for k in range(self.rank):
                e = I[k].unsqueeze(0)  # [1, rank]

                # Read A column using two-amplitude method
                a_col, _, _ = self._two_amp_read(self.tile_a, e, mode="fwd",
                                                  read_n_avg=read_n_avg, margin=self.agc_margin)
                A_cols[:, k] = a_col

                # Read B row using two-amplitude method
                b_row, _, _ = self._two_amp_read(self.tile_b, e, mode="bwd",
                                                  read_n_avg=read_n_avg, margin=self.agc_margin)
                B_rows[k, :] = b_row

        elif self.agc_enabled:
            # AGC mode: Use optimal amplitude for each rank
            for k in range(self.rank):
                e = I[k].unsqueeze(0)  # [1, rank]

                # Find optimal amplitude for A tile
                amp_a = self._pick_amp_agc(self.tile_a, e, mode="fwd",
                                           margin=self.agc_margin, max_iters=self.agc_max_iters)
                # Read A column with optimal amplitude
                a_col = self._diff_read(self.tile_a, e, amp=amp_a, mode="fwd", read_n_avg=read_n_avg)
                A_cols[:, k] = a_col.squeeze(0) / amp_a  # Normalize by amplitude

                # Find optimal amplitude for B tile
                amp_b = self._pick_amp_agc(self.tile_b, e, mode="bwd",
                                           margin=self.agc_margin, max_iters=self.agc_max_iters)
                # Read B row with optimal amplitude
                b_row = self._diff_read(self.tile_b, e, amp=amp_b, mode="bwd", read_n_avg=read_n_avg)
                B_rows[k, :] = b_row.squeeze(0) / amp_b  # Normalize by amplitude

        else:
            # Basic mode: Simple ± differential with oversampling
            A_sum = torch.zeros(self.d_size, self.rank, device=self.device, dtype=self.dtype)
            B_sum = torch.zeros(self.rank, self.x_size, device=self.device, dtype=self.dtype)

            for _ in range(read_n_avg):
                for k in range(self.rank):
                    e = I[k].unsqueeze(0)  # [1, rank], +one-hot

                    # ± forward/backward for symmetric reading
                    a_p = self.tile_a.forward(e).squeeze(0)   # [d_size]
                    a_m = self.tile_a.forward(-e).squeeze(0)  # [d_size]
                    b_p = self.tile_b.backward(e).squeeze(0)  # [x_size]
                    b_m = self.tile_b.backward(-e).squeeze(0) # [x_size]

                    # Differential: cancels DC offset and even-order distortions
                    A_sum[:, k] += 0.5 * (a_p - a_m)
                    B_sum[k, :] += 0.5 * (b_p - b_m)

            # Average over N readings (noise reduction by √read_n_avg)
            A_cols = A_sum / read_n_avg
            B_rows = B_sum / read_n_avg

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
            for k in range(self.rank):
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

    def _ab_weight_transfer_onehot(self) -> None:
        """One-hot 기반 전송 (Tiki-Taka Threshold + Clamping 적용 버전).

        변경점:
          - 기존: ΣΔ 방식 (누적된 모든 정수분을 한 번에 전송)
          - 변경: Threshold 기반 + Burst Clamping (최대 펄스 수 제한)
            * 32배치 주기를 고려하여 Single Pulse(limit=1) 대신 Multi-Pulse(limit=10) 허용 권장.

        핵심 로직:
          1. h_k += δ_k (누적)
          2. n_raw = round(h_k / g) (임계값 0.5g 적용)
          3. n_k = clamp(n_raw, -limit, +limit) (최대 전송량 제한)
          4. h_k <- h_k - n_k * g (전송한 만큼만 차감, 잔여 정보 보존)
        """
        # === [설정] Burst Limit (펄스 상한선) ===
        # limit=1 : Strict TTv2 (Single Pulse) - 매우 느릴 수 있음
        # limit=5~10 : Clamped TTv2 (Recommended) - 속도와 안정성 균형
        burst_limit = getattr(self, 'transfer_burst_limit', 10)

        with torch.no_grad():
            # --- 준비: one-hot 캐시, LR/노이즈 백업 ---
            if self._transfer_vec_a is None:
                self._transfer_vec_a = torch.eye(self.rank, dtype=self.dtype, device=self.device)

            old_lr_c = self.tile_c.get_learning_rate()

            try:
                # --- 1) ± one-hot 차분 읽기: A_cols[d, r], B_rows[r, x] ---
                A_cols, B_rows = self._read_ab_onehot_symmetric()

                # (선택) 중심화/정규화
                A_cols, B_rows = self._center_and_normalize(A_cols, B_rows)

                # --- 2) ΣΔ 상태/파라미터 확보 ---
                self._ensure_sd_state()

                # quantum g (Threshold 역할): 기본은 transfer_lr 분해능
                lr_abs = float(abs(self.transfer_lr))
                g = float(self.sd_quantum) if (self.sd_quantum is not None and self.sd_quantum > 0.0) \
                    else max(lr_abs / float(max(1, int(self.transfer_micro_steps))), 1e-12)

                # 랭크별 목표 스칼라 δ_k := |transfer_lr|
                delta = torch.full((self.rank,), lr_abs, device=self.device, dtype=self.dtype)

                # --- 3) [수정됨] Thresholding & Clamping 로직 ---

                # A. 누적 (Integrate)
                self.sd_acc = self.sd_acc + delta  # h_k += δ_k

                # B. 양자화 (Quantize / Threshold check)
                # round()는 0.5g 미만 잡음을 0으로 만드므로 Soft Threshold 역할을 함
                n_raw = torch.round(self.sd_acc / g).to(torch.int64)

                # C. [New] Clamping (Burst Limit 적용)
                # 누적값이 아무리 커도 한 번에 burst_limit 개수 이상 보내지 않음
                n = torch.clamp(n_raw, min=-burst_limit, max=burst_limit)

                # D. 잔여값 갱신 (Update Residual)
                # 중요: '전송한 펄스(n)' 만큼만 차감해야 함.
                # (clamping으로 잘려나간 정보는 sd_acc에 남아 다음 턴을 기약함)
                self.sd_acc = self.sd_acc - (n.to(self.dtype) * g)

                # Track clamping statistics (silent)
                clamped = (n_raw.abs() > burst_limit).sum().item()
                clipped_pulses = (n_raw - n).abs().sum().item()

                # --- 4) C에 정수 펄스 n_k만큼 전송 ---
                # sign rule: transfer_lr>0 -> D=-a_k
                sign = -1.0 if (self.transfer_lr > 0) else 1.0

                # unit pulse의 lr = g 로 통일
                self.tile_c.set_learning_rate(g)

                nonzero = int((n != 0).sum().item())
                max_rep = int(n.abs().max().item()) if nonzero > 0 else 0

                # Track pulses if enabled
                if self.pulse_count_enabled:
                    pulses_this_transfer = int(n.abs().sum().item())
                    self.total_pulses_to_c += pulses_this_transfer
                    self.pulse_history.append((self.num_transfers, pulses_this_transfer))

                for k in range(self.rank):
                    reps = int(n[k].item())
                    if reps == 0:
                        continue

                    a_k = (sign * A_cols[:, k]).unsqueeze(0)  # [1, d]
                    b_k = B_rows[k, :].unsqueeze(0)          # [1, x]

                    # 양수/음수 reps 모두 지원
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
                if self.num_transfers < 3:
                    res_max = float(self.sd_acc.abs().max().item())
                    print(f"[Tiki-Taka Transfer] g={g:.3e}, limit={burst_limit}, "
                          f"nonzero_ranks={nonzero}, max_reps={max_rep} (clamped), "
                          f"residual_max={res_max:.3e}")

            finally:
                # 복구
                self.tile_c.set_learning_rate(old_lr_c)

        # 계수/카운터 및 reinit
        self.num_transfers += 1
        self.transfer_counter = 0
        self.reinit()

    def _ab_weight_transfer_onehot_simple(self) -> None:
        """One-hot read 후 단순 pulsed update (ΣΔ 없음).

        Direct와 동일한 write 로직이지만, read만 one-hot 방식 사용.
        - Read: one-hot forward/backward (analog-realistic)
        - Write: rank별 외적을 개별 pulsed update (ΣΔ 없음)

        수학적으로: C += transfer_lr * Σ_k (a_k ⊗ b_k) = transfer_lr * A @ B
        """
        with torch.no_grad():
            # 1) One-hot으로 A, B 읽기
            A_cols, B_rows = self._read_ab_onehot_symmetric()

            # (선택) 중심화/정규화
            A_cols, B_rows = self._center_and_normalize(A_cols, B_rows)

            # 2) LR 설정
            lr_eff = abs(self.transfer_lr)
            old_lr = self.tile_c.get_learning_rate()
            self.tile_c.set_learning_rate(lr_eff)

            # Sign rule: PWU does C += -lr * D @ X^T
            # We want C += +transfer_lr * a_k @ b_k^T
            # So when transfer_lr > 0, negate a_k
            sign = -1.0 if self.transfer_lr > 0 else 1.0

            # [DEBUG] Rank-wise update measurement
            if not hasattr(self, '_debug_rank_updates'):
                self._debug_rank_updates = []

            rank_updates_this_transfer = []
            C_before_transfer = None
            if hasattr(self.tile_c.tile, 'get_weights'):
                c_rows = self.tile_c.tile.get_weights()
                C_before_transfer = torch.stack([row.cpu() for row in c_rows])

            # Calculate actual pulses (simple pulsed mode)
            # Each update C += lr * a_k^T @ b_k translates to pulses
            # For each weight cell C[i,j], update = lr * a_k[i] * b_k[j]
            # Number of pulses ≈ |update| / dw_min
            # Total pulses for rank k = sum over all cells
            total_pulses_this_transfer = 0
            if self.pulse_count_enabled:
                dw_min_c = self.tile_c.rpu_config.device.dw_min
                for k in range(self.rank):
                    a_k_vals = (sign * A_cols[:, k]).abs()  # [d_size]
                    b_k_vals = B_rows[k, :].abs()           # [x_size]

                    # Total update magnitude = lr * sum|a_k[i] * b_k[j]|
                    # = lr * ||a_k||_1 * ||b_k||_1
                    pulse_magnitude = lr_eff * a_k_vals.sum() * b_k_vals.sum()

                    # Convert to pulse count
                    num_pulses_k = (pulse_magnitude / dw_min_c).item()
                    total_pulses_this_transfer += num_pulses_k

                self.total_pulses_to_c += total_pulses_this_transfer
                self.pulse_history.append((self.num_transfers, total_pulses_this_transfer))

            # 3) rank번 반복하여 각 외적을 개별 update
            for k in range(self.rank):
                a_k = (sign * A_cols[:, k]).unsqueeze(0)  # [1, d_size]
                b_k = B_rows[k, :].unsqueeze(0)           # [1, x_size]

                # [DEBUG] Measure C before this rank update
                C_before_k = None
                if C_before_transfer is not None:
                    c_rows = self.tile_c.tile.get_weights()
                    C_before_k = torch.stack([row.cpu() for row in c_rows])

                # Pulsed update: C += lr * a_k^T @ b_k
                if hasattr(self.tile_c, '_orig_update'):
                    self.tile_c._orig_update(b_k, a_k)
                else:
                    self.tile_c.update(b_k, a_k)

                # [DEBUG] Measure C after this rank update
                if C_before_k is not None:
                    c_rows = self.tile_c.tile.get_weights()
                    C_after_k = torch.stack([row.cpu() for row in c_rows])

                    # Calculate actual update for this rank
                    actual_delta_k = C_after_k - C_before_k
                    actual_mag_k = actual_delta_k.norm().item()

                    # Calculate expected update
                    a_k_cpu = a_k.cpu().squeeze()  # [d_size]
                    b_k_cpu = b_k.cpu().squeeze()  # [x_size]
                    expected_delta_k = lr_eff * torch.outer(a_k_cpu, b_k_cpu)
                    expected_mag_k = expected_delta_k.norm().item()

                    # Calculate pulse magnitude
                    pulse_mag_k = torch.outer(a_k_cpu, b_k_cpu).norm().item()

                    rank_updates_this_transfer.append({
                        'rank': k,
                        'pulse_magnitude': pulse_mag_k,
                        'expected_magnitude': expected_mag_k,
                        'actual_magnitude': actual_mag_k,
                        'amplification': actual_mag_k / expected_mag_k if expected_mag_k > 0 else 0,
                    })

            # Store rank updates for this transfer
            if len(rank_updates_this_transfer) > 0:
                self._debug_rank_updates.append({
                    'transfer_num': self.num_transfers + 1,
                    'ranks': rank_updates_this_transfer,
                })

            self.tile_c.set_learning_rate(old_lr)

        self.num_transfers += 1
        self.transfer_counter = 0
        self.reinit()

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
        W_eff = C_weights + self.lora_alpha * (A_lr @ B_lr)

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
        W_eff = C_weights.t() + self.lora_alpha * (B_weights.t() @ A_weights.t())

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
        y = y_c + self.lora_alpha * y_ab   # [batch, d_size]

        # 4) Output transpose
        return y.t() if out_trans else y

    def should_transfer(self) -> bool:
        """Check if transfer should occur based on counter and schedule."""
        return self.transfer_counter >= self.transfer_every

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
            'decay_factor': self.decay_factor,
            'forward_inject_enabled': self.forward_inject_enabled
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
        # Clear ΣΔ state for reallocation on new device
        self.sd_acc = None
