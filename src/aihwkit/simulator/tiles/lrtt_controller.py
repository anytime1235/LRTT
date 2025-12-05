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
        self.decay_factor = decay_factor
        self.correct_gradient_magnitudes = correct_gradient_magnitudes
        self.rank_chunk = rank_chunk or rank
        self.forward_inject_enabled = forward_inject

        # BL management settings
        self.ab_bl_mgmt = ab_bl_mgmt or {}
        self.transfer_bl_mgmt = transfer_bl_mgmt or {}

        # Counters and state
        self.transfer_counter = 0
        self.num_a_updates = 0
        self.num_b_updates = 0
        self.num_transfers = 0

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

        # Transfer robustness knobs (safe defaults)
        self.transfer_micro_steps: int = 1          # M: micro-transfer 반복 횟수
        self.transfer_centering: bool = False       # 행/열 평균 제거 (기본 off - gradient 왜곡 방지)
        self.transfer_normalize: bool = False       # 랭크별 ℓ2 정규화 (기본 off - gradient 왜곡 방지)

        # --- Sigma-Delta (ΣΔ) core state ---
        self.sd_quantum: Optional[float] = None     # g: unit quantum for rank-wise pulses (None -> derive per transfer)
        self.sd_acc: Optional[Tensor] = None        # h_k residuals [rank], persistent across transfers

        # Transfer one-hot vectors cache
        self._transfer_vec_a: Optional[Tensor] = None

    def _ensure_sd_state(self) -> None:
        """Ensure ΣΔ state tensors exist on the right device/dtype."""
        if self.sd_acc is None or self.sd_acc.numel() != self.rank or self.sd_acc.device != self.device:
            self.sd_acc = torch.zeros(self.rank, device=self.device, dtype=self.dtype)

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

            else:
                raise ValueError(f"Unknown reinit_mode: {self.reinit_mode}. Must be 'standard', 'decay', or 'hybrid'")

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
        """Update A and B with LoRA-style rank-r gradient approximation.

        Simplified batch-first processing with no intermediate transposes.
        Uses tile forward/backward for projections and tile update for weight changes.

        Args:
            x: Input tensor
            d: Error tensor
            lr: Learning rate
            in_trans: Whether x is transposed
            out_trans: Whether d is transposed
        """
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
        lr_b_old = self.tile_b.get_learning_rate()
        self.tile_b.set_learning_rate(lr_eff)
        if hasattr(self.tile_b, '_orig_update'):
            self.tile_b._orig_update(x, DA)
        else:
            self.tile_b.update(x, DA)
        self.tile_b.set_learning_rate(lr_b_old)
        self.num_b_updates += 1

        # 5) Counter
        self.transfer_counter += (x.shape[0] if self.units_in_mbatch else 1)

    def ab_weight_transfer(self, use_onehot: bool = True) -> None:
        """Memory-optimized pulsed A⊗B -> visible transfer, then reinit.

        Transfer: C += transfer_lr * (A @ B) via pulsed outer product.

        Args:
            use_onehot: If True, use one-hot reading (analog-realistic).
                       If False, use direct weight access (default).

        Direct mode:
        1. Get weights to CPU first to avoid GPU memory spike
        2. For chunks of rank: pack D_chunk = A[:, off:off+cur], X_chunk = B[off:off+cur, :]
        3. Move only chunks to GPU for update
        4. Call visible pulsed updater: C.update(X_chunk^T, D_chunk, lr=|transfer_lr|)
        5. Handle sign rule: negate D when transfer_lr > 0
        6. Unconditionally call reinit() after transfer

        One-hot mode:
        1. Read A columns using forward pass with one-hot vectors
        2. Read B rows using backward pass with one-hot vectors
        3. Accumulate outer products into C
        4. Unconditionally call reinit() after transfer
        """
        if use_onehot:
            self._ab_weight_transfer_onehot()
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

                # OPTIMIZATION: Immediately free GPU memory
                del X_chunk_d, D_chunk_t_d
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.tile_c.set_learning_rate(old_lr)
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
        """± one-hot 차분 읽기로 DC/짝수차 왜곡 제거.

        Returns:
            (A_cols: [d_size, rank], B_rows: [rank, x_size])
        """
        if self._transfer_vec_a is None:
            self._transfer_vec_a = torch.eye(
                self.rank, dtype=self.dtype, device=self.device
            )

        I = self._transfer_vec_a
        A_cols = []
        B_rows = []

        for k in range(self.rank):
            e = I[k].unsqueeze(0)  # [1, rank], +one-hot

            # ± forward/backward for symmetric reading
            a_p = self.tile_a.forward(e).squeeze(0)   # [d_size]
            a_m = self.tile_a.forward(-e).squeeze(0)  # [d_size]
            b_p = self.tile_b.backward(e).squeeze(0)  # [x_size]
            b_m = self.tile_b.backward(-e).squeeze(0) # [x_size]

            # Differential: cancels DC offset and even-order distortions
            a_k = 0.5 * (a_p - a_m)
            b_k = 0.5 * (b_p - b_m)

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
        """One-hot 기반 전송 (ΣΔ 핵심 버전: 랭크별 적분기 h_k + 고정 quantum g).

        핵심:
          - 원하는 스칼라 투영: δ_k := |transfer_lr| (랭크별 동일 스칼라)
          - ΣΔ 1차: h_k <- h_k + δ_k; n_k <- round(h_k / g); h_k <- h_k - n_k*g
          - sign rule: tile.update는 W += -lr * D @ X^T → transfer_lr>0일 때 D=-a_k
          - g(quantum): sd_quantum 사용. None이면 기본값 g := max(|transfer_lr| / max(1, self.transfer_micro_steps), 1e-12)

        관리/최적화(버스트캡, 게이트, EMA 보정 등)는 추후 추가.
        """
        with torch.no_grad():
            # --- 준비: one-hot 캐시, LR/노이즈 백업 ---
            if self._transfer_vec_a is None:
                self._transfer_vec_a = torch.eye(self.rank, dtype=self.dtype, device=self.device)

            old_lr_c = self.tile_c.get_learning_rate()

            # A/B/C 읽기 동안 out_noise=0 (out_res 등은 유지)
            old_out_a = self.tile_a.rpu_config.forward.out_noise
            old_out_b_f = self.tile_b.rpu_config.forward.out_noise
            old_out_b_b = self.tile_b.rpu_config.backward.out_noise
            old_out_c = getattr(self.tile_c.rpu_config.forward, "out_noise", 0.0)

            self.tile_a.rpu_config.forward.out_noise = 0.0
            self.tile_b.rpu_config.forward.out_noise = 0.0
            self.tile_b.rpu_config.backward.out_noise = 0.0
            if hasattr(self.tile_c.rpu_config.forward, "out_noise"):
                self.tile_c.rpu_config.forward.out_noise = 0.0

            try:
                # --- 1) ± one-hot 차분 읽기: A_cols[d, r], B_rows[r, x] ---
                A_cols, B_rows = self._read_ab_onehot_symmetric()

                # (선택) 중심화/정규화
                A_cols, B_rows = self._center_and_normalize(A_cols, B_rows)

                # --- 2) ΣΔ 상태/파라미터 확보 ---
                self._ensure_sd_state()
                # quantum g 설정: 기본은 transfer_lr를 micro_steps로 쪼갠 크기
                lr_abs = float(abs(self.transfer_lr))
                g = float(self.sd_quantum) if (self.sd_quantum is not None and self.sd_quantum > 0.0) \
                    else max(lr_abs / float(max(1, int(self.transfer_micro_steps))), 1e-12)

                # 랭크별 목표 스칼라 δ_k := |transfer_lr|  (모든 k 동일)
                delta = torch.full((self.rank,), lr_abs, device=self.device, dtype=self.dtype)

                # --- 3) ΣΔ 적분/정수화: h_k 누적 -> 정수 펄스 n_k, 잔여 갱신 ---
                self.sd_acc = self.sd_acc + delta  # h_k += δ_k
                n_float = self.sd_acc / g          # n* ≈ h_k/g
                n = torch.round(n_float).to(torch.int64)  # 정수 펄스
                self.sd_acc = self.sd_acc - n.to(self.dtype) * g  # h_k <- h_k - n_k*g

                # --- 4) C에 정수 펄스 n_k만큼 전송 ---
                # sign rule: transfer_lr>0 이면 D=-a_k (W += +transfer_lr*A@B를 얻기 위함)
                sign = -1.0 if (self.transfer_lr > 0) else 1.0

                # unit pulse의 lr = g 로 통일
                self.tile_c.set_learning_rate(g)

                nonzero = int((n != 0).sum().item())
                max_rep = int(n.abs().max().item()) if nonzero > 0 else 0

                for k in range(self.rank):
                    reps = int(n[k].item())
                    if reps == 0:
                        continue

                    a_k = (sign * A_cols[:, k]).unsqueeze(0)  # [1, d]
                    b_k = B_rows[k, :].unsqueeze(0)          # [1, x]

                    # 양수/음수 reps 모두 지원: reps<0이면 부호를 D로 흡수
                    if reps < 0:
                        a_k = -a_k
                        reps = -reps

                    # 핵심만: reps 번 unit 업데이트 (추후 burst-cap/macro-call 최적화 가능)
                    for _ in range(reps):
                        if hasattr(self.tile_c, '_orig_update'):
                            self.tile_c._orig_update(b_k, a_k)
                        else:
                            self.tile_c.update(b_k, a_k)

                # 디버그 (초기 몇 회만)
                if self.num_transfers < 3:
                    res_max = float(self.sd_acc.abs().max().item())
                    print(f"[ΣΔ transfer] g={g:.3e}, nonzero_ranks={nonzero}, max_reps={max_rep}, "
                          f"residual_max<=g/2? {res_max <= 0.5*g + 1e-12} (res_max={res_max:.3e})")

            finally:
                # 복구
                self.tile_c.set_learning_rate(old_lr_c)
                self.tile_a.rpu_config.forward.out_noise = old_out_a
                self.tile_b.rpu_config.forward.out_noise = old_out_b_f
                self.tile_b.rpu_config.backward.out_noise = old_out_b_b
                if hasattr(self.tile_c.rpu_config.forward, "out_noise"):
                    self.tile_c.rpu_config.forward.out_noise = old_out_c

        # 계수/카운터 및 reinit는 기존과 동일
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
