# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""Python-level LRTT configuration classes.

Pure Python LRTT configurations designed specifically for our Python-level
implementation, eliminating CUDA dependencies and rpucuda bindings.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import warnings

from aihwkit.simulator.configs.devices import PulsedDevice, ConstantStepDevice, LinearStepDevice
from aihwkit.simulator.parameters.enums import RPUDataType
from aihwkit.simulator.parameters.helpers import _PrintableMixin


@dataclass  
class PythonLRTTDevice(_PrintableMixin):
    """Python-level LRTT device configuration.
    
    Designed specifically for Python LRTT implementation without CUDA dependencies.
    Maps directly to LRTTController parameters.
    """
    
    # === Core LRTT Parameters ===
    rank: int = 4
    """LoRA rank dimension r. Must be > 0 and <= min(d_size, x_size)."""
    
    transfer_every: int = 32
    """Transfer frequency: every N steps (or samples if units_in_mbatch=True)."""
    
    transfer_lr: float = 1.0
    """Transfer learning rate scalar applied during A⊗B -> visible transfer."""

    transfer_lr_scale: float = 1.0
    """Scaling factor for transfer_lr. Effective transfer_lr = transfer_lr * transfer_lr_scale.
    - 1.0: No scaling (default)
    - < 1.0: Reduce transfer learning rate
    - > 1.0: Increase transfer learning rate
    """

    lora_alpha: float = 1.0
    """LoRA scaling factor α in W_eff = W_visible + α * A @ B."""
    
    reinit_gain: float = 0.1
    """Kaiming initialization gain for B matrix after transfer."""

    reinit_mode: str = "standard"
    """Reinit strategy after transfer:
    - 'standard': A=0, B=Kaiming (original LRTT)
    - 'decay': A*=decay_factor, B*=decay_factor (gradual decay)
    - 'hybrid': A=0, B*=decay_factor (hybrid approach)
    - 'orthogonal': A=0, B=Random Orthogonal (FROZEN). B @ B.T = I for projection.
    """

    decay_factor: float = 0.9
    """Decay factor for 'decay' and 'hybrid' reinit modes (0 < decay_factor < 1)."""

    a_init_mode: str = "zero"
    """A matrix initialization mode for first reinit:
    - 'zero': A=0 (LoRA-style, ensures ΔW=0 initially)
    - 'kaiming': A=Kaiming Normal (random initialization)
    """

    b_init_mode: str = "kaiming"
    """B matrix initialization mode for first reinit:
    - 'kaiming': B=Kaiming Normal (standard LoRA initialization)
    - 'zero': B=0 (ensures ΔW=0 initially)
    """

    # === Transfer Read Settings ===
    num_reads: int = 1
    """Number of reads per rank during one-hot transfer.
    Higher values reduce analog read noise but increase transfer time.
    Default is 1 (single read, original behavior)."""

    multi_read_mode: str = "average"
    """How to handle multiple reads (only when num_reads > 1):
    - 'average': Read num_reads times, average, then transfer once.
                 Reduces read noise (1/sqrt(N)), write noise unchanged.
    - 'per_read': Transfer after each read with lr/num_reads.
                  Reduces read noise but write noise may increase (N writes).
    Default is 'average'."""

    # === Transfer Mode & Calibration ===
    transfer_mode: str = "off"
    """Transfer calibration mode:
    - 'pilot': Pilot-based γ calibration. Sends a small pilot transfer first,
               measures actual vs expected, computes γ correction factor.
    - 'sigma_delta': ΣΔ quantization. Accumulates target lr in a residual h_k,
                     quantizes to integer pulses n_k = round(h_k/g), stores
                     residual for next transfer. Long-term accurate but
                     per-transfer variance higher.
    - 'off': No calibration, direct transfer with transfer_lr.
    Default is 'off'."""

    transfer_micro_steps: int = 1
    """M: Number of micro-transfer steps per transfer.
    - 1: Fast, single update per rank (default, recommended for speed)
    - 4+: Slower but more realistic analog simulation with noise averaging
    Higher values give smoother pulse accumulation but increase transfer time.
    For 'pilot' mode: lr_remain is split across M_rest = M - 1 steps.
    For 'sigma_delta' mode: g = |transfer_lr| / M if sd_quantum is not set."""

    transfer_pilot_frac: float = 0.0625  # 1/16
    """Fraction of |transfer_lr| used for pilot in 'pilot' mode.
    Smaller values give more accurate γ estimation but less total transfer.
    Default is 1/16 = 0.0625."""

    sd_quantum: Optional[float] = None
    """Unit quantum g for ΣΔ mode. If None, derived as |transfer_lr| / micro_steps.
    Each rank accumulates target lr and quantizes to n_k * g pulses."""

    # === Read Noise Reduction Settings ===
    read_n_avg: int = 1
    """Oversampling count for one-hot reads. Each read is averaged N times.
    - 1: Standard single reading (original behavior)
    - 4-8: Recommended for noise reduction (√N improvement)
    Default is 1."""

    differential_read: bool = False
    """Use differential read (+e, -e) for DC offset cancellation.
    - True: Read with +e and -e, use 0.5*(f(+e) - f(-e)). More accurate but 2x slower.
    - False: Read with +e only. Faster but includes DC offset.
    Default is False (faster)."""

    # === AGC (Automatic Gain Control) Settings ===
    agc_enabled: bool = False
    """Enable AGC for optimal read amplitude selection.
    Uses binary search to find largest amplitude without ADC clipping."""

    agc_margin: float = 0.85
    """Target output bound fraction for AGC (avoid clipping).
    0.85 = 85% of out_bound. Default is 0.85."""

    agc_max_iters: int = 6
    """Max iterations for AGC binary search. Default is 6."""

    # === Two-Amplitude Differential Read Settings ===
    two_amp_enabled: bool = False
    """Enable two-amplitude differential read for odd offset removal.
    Uses two amplitudes to cancel odd-order distortions."""

    two_amp_ratio: float = 0.5
    """Ratio α1/α2 for two-amplitude method. Default is 0.5."""

    # === Update Mode Settings ===
    update_mode: str = "lora"
    """A/B update mode:
    - 'lora': LoRA chain rule (original LRTT, requires forward_inject=True)
    - 'reconstruction': TikiTaka-style gradient reconstruction (for forward_inject=False)
    - 'selector_reconstruction' (alias 'selector_v2'): LRTT-v2 row-coordinate selector
      reconstruction. tile_a is unused; B accumulates only the selected row block of the
      gradient via a fixed one-hot row selector S_R (non-trainable). Pair with
      transfer_method='blockwise'.
    Default is 'lora'."""

    # === Reconstruction Update Parameters (for update_mode='reconstruction') ===
    recon_lambda_a: float = 1e-3
    """L2 regularization coefficient for A in reconstruction loss."""

    recon_lambda_b: float = 1e-3
    """L2 regularization coefficient for B in reconstruction loss."""

    recon_use_scalar_stabilizer: bool = False
    """Use scalar approximation for stabilizer terms (BB^T ≈ sB*I, A^TA ≈ sA*I).
    Disabled by default as orthogonal reinit + transfer provides natural stability."""

    recon_use_exact_gram: bool = False
    """Use exact Gram matrix (BB^T, A^TA) for stabilizer terms.
    Only for debugging - expensive O(rank^2) computation."""

    recon_exact_gram_every: int = 0
    """Use exact Gram every N steps (0 = disabled). For periodic exact stabilization."""

    recon_ema_beta: float = 0.9
    """EMA decay for tracking sA, sB norms (0.9~0.99 recommended)."""

    recon_lr_scale: float = 1.0
    """Additional learning rate scale for reconstruction updates (0.1~1.0)."""

    recon_clip_norm: float = 10.0
    """Max norm for A,B clipping (safety fallback). Only used if recon_use_clip_norm=True."""

    recon_use_clip_norm: bool = False
    """Enable norm clipping for A,B. Disabled by default as orthogonal reinit provides stability."""

    # === Transfer Method Settings ===
    transfer_method: str = "onehot"
    """Transfer method:
    - "onehot": One-hot transfer (rank-by-rank differential read, pulsed update)
    - "direct": Direct transfer (matrix multiply A @ B, pulsed update)
    - "set": Exact transfer (set_weights directly, no pulsed update noise)
    - "blockwise": LRTT-v2 blockwise transfer. Updates only the selected C row block
      via a single pulsed outer product C[R, :] += transfer_lr * B, then resets B and
      advances the selector. Pair with update_mode='selector_reconstruction'.
    Default is "onehot"."""

    # === LRTT-v2 Selector Settings ===
    selector_axis: str = "row"
    """Selector axis. Currently only 'row' is implemented (S_R selects d_size rows)."""

    selector_policy: str = "shuffled_cycle"
    """Selector schedule:
    - 'cyclic': sequential row blocks 0..b-1, b..2b-1, ...
    - 'shuffled_cycle': random permutation of rows, advance one block at a time, reshuffle
      after a full cycle.
    - 'random': independent random sample of b rows per transfer (requires
      selector_random_unbiased_scale=True for an unbiased estimator).
    Default 'shuffled_cycle' guarantees full coverage with low variance."""

    selector_block_size: Optional[int] = None
    """Selector block size b. None defaults to rank. Must equal tile_b's rank dimension
    because tile_b is allocated as [rank, x_size]."""

    selector_seed: int = 0
    """Seed for the selector RNG (CPU torch.Generator)."""

    selector_allow_partial_block: bool = False
    """If True, the final block of a cycle may be smaller than block_size; remaining
    slots are masked out via selector_valid_mask. If False, d_size must be divisible by
    block_size."""

    selector_random_unbiased_scale: bool = False
    """When selector_policy='random', multiply transfer magnitude by d_size/active_b
    to obtain an unbiased estimator of the full gradient. Off by default (cyclic
    schedules already guarantee full coverage)."""

    selector_reset_b_on_advance: bool = True
    """If True, B is zeroed via _reset_b_buffer() after each blockwise transfer. Required
    for coordinate consistency: B[k] is the residual for the current row i_k, and after
    selector advance the new row j_k must start from a clean state."""

    # === LRTT-v2 Capacitor Stabilizer Settings ===
    cap_stabilizer_enabled: bool = True
    """Enable capacitor-leak/range stabilizer for the B buffer in LRTT-v2."""

    cap_rho: float = 1.0
    """B leakage factor per update step: B <- cap_rho * B (then accumulate). 1.0 disables
    leakage. 0.99~0.999 model 6T1C capacitor retention within a transfer window."""

    cap_compensate_transfer: bool = True
    """If True, multiply transfer magnitude by kappa_rho(tau) = tau*(1-rho)/(1-rho^tau) to
    compensate for the geometric attenuation of B under leakage. Recovers unbiased
    full-gradient transfer for constant-gradient windows."""

    cap_compensation_max: float = 4.0
    """Safety clamp for kappa_rho. Prevents runaway compensation when tau is large."""

    cap_monitor_every: int = 0
    """Run RMS monitor + soft-clip every N B-updates. 0 disables monitoring (rely on
    natural leakage)."""

    cap_target_rms: Optional[float] = None
    """Target RMS for B (informational only). If None, no target tracking."""

    cap_max_rms: Optional[float] = None
    """If set, applies soft-clipping when measured B RMS exceeds this threshold."""

    cap_soft_clip: bool = True
    """If True, soft-clip via a multiplicative scale on persistent weights. If False, the
    monitor only logs RMS (no action)."""

    cap_reset_mode: str = "set_zero"
    """B reset semantics:
    - 'set_zero': tile_b.set_weights(zeros) (correct for simulator/FloatingPoint).
    - 'hard_reset': reserved for future hardware backend hard-reset primitive.
    - 'none': skip B reset (NOT recommended; only for ablations)."""

    # === SRA-LRTT-v2 (Stochastic Reset-Anchor) Settings ===
    sra_anchor_source: str = "reset_columns"
    """Anchor generation source for A_q within each transfer window:
    - 'reset_columns': call tile_a.reset_columns(0, rank, 1.0) and read the resulting
      device state (main hardware-native source; uses LinearStepDevice reset/reset_std).
    - 'set_zero_write_noise': set_weights(zeros) and rely on apply_write_noise_on_set=True
      to produce a stochastic apparent state (ablation; usually too small in 6T1C).
    - 'explicit_gaussian': sample A_raw ~ N(0, target_rms^2) directly (sanity check / FP).
    - 'pulse_scramble': reset to zero-centered state then apply controlled pulse updates
      to seed independent randomness via dw_min_std (alternative source)."""

    sra_anchor_target_rms: Optional[float] = None
    """Target RMS of the scaled anchor A_q. None -> 1/sqrt(rank), giving E[A A^T] ≈ I."""

    sra_anchor_gain_max: float = 1.0e3
    """Clamp for the scalar gain g_q after RMS normalization. Prevents runaway when
    raw RMS is near the floor."""

    sra_anchor_min_rms: float = 1.0e-12
    """Minimum raw RMS before declaring the generated anchor degenerate (gain=0)."""

    sra_use_cached_anchor: bool = True
    """If True, cache A_raw and A_scaled on the controller after each reset so the
    SAME A_q is used for both B accumulation and C transfer within a window. This
    self-correlated anchor is what makes SRA an unbiased TT estimator."""

    sra_use_analog_projection: bool = False
    """If True, use tile_a.backward(d) * sra_anchor_gain to compute d_proj (with A
    read-path nonidealities). If False (default), use cached d @ A_scaled (clean)."""

    sra_resample_on_transfer: bool = True
    """Re-sample A_q after every transfer (the typical SRA behavior)."""

    sra_reset_b_on_transfer: bool = True
    """Reset B to zero after every transfer (residual buffer cleanup)."""

    sra_b_reset_mode: str = "set_zero"
    """B reset semantics under SRA: 'set_zero' | 'reset_columns' | 'none'.
    Default 'set_zero' achieves exact zero on FP/simulator. 'reset_columns' uses the
    device reset distribution. 'none' is for ablations only."""

    sra_pulse_scramble_steps: int = 0
    """Number of controlled random-pulse-scramble passes after reset. 0 disables.
    Used only when sra_anchor_source == 'pulse_scramble'."""

    sra_pulse_scramble_lr: float = 1.0
    """Learning-rate scalar applied during pulse-scramble anchor generation."""

    sra_seed: int = 0
    """CPU torch.Generator seed for deterministic 'explicit_gaussian' and
    'pulse_scramble' anchor sources."""

    # === Advanced Parameters ===
    units_in_mbatch: bool = False
    """If True, transfer_every counts samples; if False, counts steps."""
    
    correct_gradient_magnitudes: bool = False
    """If True, scale learning rate by sqrt(rank) for gradient correction."""
    
    forward_inject: bool = False
    """Enable forward injection optimization: W_eff composition."""
    
    rank_chunk: Optional[int] = None
    """Chunk size for transfer (None = use full rank). For memory management."""
    
    columns_mode: bool = True
    """Transfer mode: True=columns (forward), False=rows (backward)."""
    
    # === Device Configuration ===
    unit_cell_devices: List[PulsedDevice] = field(default_factory=lambda: [
        ConstantStepDevice(dw_min=0.01, w_min=-1.0, w_max=1.0),
        ConstantStepDevice(dw_min=0.01, w_min=-1.0, w_max=1.0), 
        ConstantStepDevice(dw_min=0.01, w_min=-1.0, w_max=1.0)
    ])
    """Device configurations for [fastA, fastB, visible] tiles."""
    
    # === BL Management (Simplified for Python) ===
    ab_bl_mgmt: Optional[Dict[str, Any]] = None
    """BL management settings for A/B updates (optional)."""

    transfer_bl_mgmt: Optional[Dict[str, Any]] = None
    """BL management settings for transfers (optional)."""

    # === Separate A/B Scaling Parameters ===
    # When use_manual_scaling=True in UpdateParameters, these override global scaling
    a_x_scaling: Optional[float] = None
    """X (input) scaling factor for A tile. If None, uses global manual_x_scaling."""

    a_d_scaling: Optional[float] = None
    """D (gradient) scaling factor for A tile. If None, uses global manual_d_scaling."""

    b_x_scaling: Optional[float] = None
    """X (input) scaling factor for B tile. If None, uses global manual_x_scaling."""

    b_d_scaling: Optional[float] = None
    """D (gradient) scaling factor for B tile. If None, uses global manual_d_scaling."""

    # === Separate BL (Bit Length) Settings ===
    c_desired_bl: Optional[int] = None
    """Desired BL for C tile (transfer). If None, uses global desired_bl from UpdateParameters.
    Typically set higher (e.g., 31) for more accurate transfer to C tile."""

    # === Debug Logging ===
    log_ab_scaling: bool = False
    """Enable logging of x,d max values during A/B updates."""

    log_ab_scaling_every: int = 10
    """Log x,d max values every N steps (only when log_ab_scaling=True)."""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate rank
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
            
        # Validate transfer parameters
        if self.transfer_every <= 0:
            raise ValueError(f"transfer_every must be positive, got {self.transfer_every}")
            
        if self.transfer_lr <= 0:
            raise ValueError(f"transfer_lr must be positive, got {self.transfer_lr}")
            
        # Validate LoRA parameters
        if self.lora_alpha < 0:
            raise ValueError(f"lora_alpha must be non-negative, got {self.lora_alpha}")
            
        if self.reinit_gain < 0:
            raise ValueError(f"reinit_gain must be non-negative, got {self.reinit_gain}")

        # Validate reinit_mode
        valid_modes = ["standard", "decay", "hybrid", "orthogonal_zero", "orthogonal_decay"]
        if self.reinit_mode not in valid_modes:
            raise ValueError(f"reinit_mode must be one of {valid_modes}, got '{self.reinit_mode}'")

        # Validate decay_factor
        if not (0 < self.decay_factor <= 1):
            raise ValueError(f"decay_factor must be in (0, 1], got {self.decay_factor}")

        # Validate num_reads
        if self.num_reads < 1:
            raise ValueError(f"num_reads must be >= 1, got {self.num_reads}")

        # Validate multi_read_mode
        valid_read_modes = ["average", "per_read"]
        if self.multi_read_mode not in valid_read_modes:
            raise ValueError(f"multi_read_mode must be one of {valid_read_modes}, got '{self.multi_read_mode}'")

        # Validate transfer_lr_scale (must be a positive float)
        if self.transfer_lr_scale <= 0:
            raise ValueError(f"transfer_lr_scale must be > 0, got {self.transfer_lr_scale}")

        # Validate read_n_avg
        if self.read_n_avg < 1:
            raise ValueError(f"read_n_avg must be >= 1, got {self.read_n_avg}")

        # Validate AGC settings
        if not (0 < self.agc_margin <= 1.0):
            raise ValueError(f"agc_margin must be in (0, 1], got {self.agc_margin}")
        if self.agc_max_iters < 1:
            raise ValueError(f"agc_max_iters must be >= 1, got {self.agc_max_iters}")

        # Validate two_amp_ratio
        if not (0 < self.two_amp_ratio < 1.0):
            raise ValueError(f"two_amp_ratio must be in (0, 1), got {self.two_amp_ratio}")

        # Validate update_mode
        valid_update_modes = [
            "lora", "reconstruction", "selector_reconstruction", "selector_v2",
            "stochastic_reset_anchor", "sra_reconstruction",
            "random_anchor_reconstruction", "sra_v2",
        ]
        if self.update_mode not in valid_update_modes:
            raise ValueError(f"update_mode must be one of {valid_update_modes}, got '{self.update_mode}'")

        # Validate transfer_method
        valid_transfer_methods = ["onehot", "direct", "set", "blockwise", "stochastic_anchor", "sra"]
        if self.transfer_method not in valid_transfer_methods:
            raise ValueError(f"transfer_method must be one of {valid_transfer_methods}, got '{self.transfer_method}'")

        # Validate LRTT-v2 selector parameters
        if self.selector_axis != "row":
            raise ValueError(f"selector_axis must be 'row' (only row selector supported), got '{self.selector_axis}'")
        valid_selector_policies = ["cyclic", "shuffled_cycle", "random"]
        if self.selector_policy not in valid_selector_policies:
            raise ValueError(f"selector_policy must be one of {valid_selector_policies}, got '{self.selector_policy}'")
        if self.selector_block_size is not None and self.selector_block_size <= 0:
            raise ValueError(f"selector_block_size must be positive or None, got {self.selector_block_size}")

        # Validate LRTT-v2 capacitor parameters
        if not (0 < self.cap_rho <= 1):
            raise ValueError(f"cap_rho must be in (0, 1], got {self.cap_rho}")
        if self.cap_compensation_max < 1.0:
            raise ValueError(f"cap_compensation_max must be >= 1.0, got {self.cap_compensation_max}")
        if self.cap_monitor_every < 0:
            raise ValueError(f"cap_monitor_every must be >= 0, got {self.cap_monitor_every}")
        valid_cap_reset_modes = ["set_zero", "hard_reset", "none"]
        if self.cap_reset_mode not in valid_cap_reset_modes:
            raise ValueError(f"cap_reset_mode must be one of {valid_cap_reset_modes}, got '{self.cap_reset_mode}'")

        # Validate SRA-LRTT-v2 parameters
        valid_sra_sources = ["reset_columns", "set_zero_write_noise", "explicit_gaussian", "pulse_scramble"]
        if self.sra_anchor_source not in valid_sra_sources:
            raise ValueError(
                f"sra_anchor_source must be one of {valid_sra_sources}, got '{self.sra_anchor_source}'"
            )
        valid_sra_b_reset = ["set_zero", "reset_columns", "none"]
        if self.sra_b_reset_mode not in valid_sra_b_reset:
            raise ValueError(
                f"sra_b_reset_mode must be one of {valid_sra_b_reset}, got '{self.sra_b_reset_mode}'"
            )
        # sra_anchor_target_rms == 0 IS allowed: it is the σ_A=0 negative control
        # (A_q := 0  ⇒  A_q^T G = 0  ⇒  B does not accumulate). Only strictly
        # negative values are rejected.
        if self.sra_anchor_target_rms is not None and self.sra_anchor_target_rms < 0:
            raise ValueError(
                f"sra_anchor_target_rms must be >= 0 or None "
                f"(0 = negative control), got {self.sra_anchor_target_rms}"
            )
        if self.sra_anchor_gain_max <= 0:
            raise ValueError(f"sra_anchor_gain_max must be positive, got {self.sra_anchor_gain_max}")
        if self.sra_anchor_min_rms <= 0:
            raise ValueError(f"sra_anchor_min_rms must be positive, got {self.sra_anchor_min_rms}")
        if self.sra_pulse_scramble_steps < 0:
            raise ValueError(
                f"sra_pulse_scramble_steps must be >= 0, got {self.sra_pulse_scramble_steps}"
            )
        if self.sra_pulse_scramble_lr <= 0:
            raise ValueError(
                f"sra_pulse_scramble_lr must be positive, got {self.sra_pulse_scramble_lr}"
            )

        # SRA-LRTT-v2 cross-validation: A/B are optimizer state only, not forward-visible.
        sra_update_modes = (
            "stochastic_reset_anchor", "sra_reconstruction",
            "random_anchor_reconstruction", "sra_v2",
        )
        if self.update_mode in sra_update_modes:
            if self.forward_inject:
                raise ValueError(
                    "SRA-LRTT-v2 requires forward_inject=False; A/B are optimizer states only "
                    "and must not be visible in the forward path."
                )
            if self.transfer_method not in ("stochastic_anchor", "sra"):
                raise ValueError(
                    f"SRA update_mode '{self.update_mode}' requires transfer_method "
                    f"in ('stochastic_anchor', 'sra'), got '{self.transfer_method}'."
                )

        # Validate reconstruction parameters
        if self.recon_lambda_a < 0:
            raise ValueError(f"recon_lambda_a must be non-negative, got {self.recon_lambda_a}")
        if self.recon_lambda_b < 0:
            raise ValueError(f"recon_lambda_b must be non-negative, got {self.recon_lambda_b}")
        if not (0 < self.recon_ema_beta < 1):
            raise ValueError(f"recon_ema_beta must be in (0, 1), got {self.recon_ema_beta}")
        if self.recon_lr_scale <= 0:
            raise ValueError(f"recon_lr_scale must be positive, got {self.recon_lr_scale}")
        if self.recon_clip_norm <= 0:
            raise ValueError(f"recon_clip_norm must be positive, got {self.recon_clip_norm}")

        # Validate transfer_mode
        valid_transfer_modes = ["pilot", "sigma_delta", "off"]
        if self.transfer_mode not in valid_transfer_modes:
            raise ValueError(f"transfer_mode must be one of {valid_transfer_modes}, got '{self.transfer_mode}'")

        # Validate transfer_micro_steps
        if self.transfer_micro_steps < 1:
            raise ValueError(f"transfer_micro_steps must be >= 1, got {self.transfer_micro_steps}")

        # Validate transfer_pilot_frac
        if not (0.0 < self.transfer_pilot_frac < 1.0):
            raise ValueError(f"transfer_pilot_frac must be in (0, 1), got {self.transfer_pilot_frac}")

        # Validate sd_quantum
        if self.sd_quantum is not None and self.sd_quantum <= 0:
            raise ValueError(f"sd_quantum must be positive or None, got {self.sd_quantum}")

        # Validate rank_chunk
        if self.rank_chunk is not None and self.rank_chunk <= 0:
            raise ValueError(f"rank_chunk must be positive or None, got {self.rank_chunk}")
            
        # Validate unit cell devices
        if len(self.unit_cell_devices) != 3:
            raise ValueError(f"Must provide exactly 3 unit_cell_devices for [fastA, fastB, visible], got {len(self.unit_cell_devices)}")
            
        # Set default rank_chunk
        if self.rank_chunk is None:
            self.rank_chunk = self.rank

        # Default selector_block_size = rank (LRTT-v2 invariant: block_size == tile_b rank dim)
        if self.selector_block_size is None:
            self.selector_block_size = self.rank
        if self.selector_block_size != self.rank:
            raise ValueError(
                f"selector_block_size ({self.selector_block_size}) must equal rank ({self.rank}); "
                "tile_b is allocated as [rank, x_size] and the selector block must match."
            )

        # Initialize BL management if not provided
        if self.ab_bl_mgmt is None:
            self.ab_bl_mgmt = {}
        if self.transfer_bl_mgmt is None:
            self.transfer_bl_mgmt = {}
    
    def get_device_for_tile(self, tile_type: str) -> PulsedDevice:
        """Get device configuration for specific tile type.
        
        Args:
            tile_type: 'fastA', 'fastB', or 'visible'
            
        Returns:
            Device configuration for the specified tile
        """
        tile_map = {'fastA': 0, 'fastB': 1, 'visible': 2}
        if tile_type not in tile_map:
            raise ValueError(f"Unknown tile_type '{tile_type}', must be one of {list(tile_map.keys())}")
            
        return self.unit_cell_devices[tile_map[tile_type]]
    
    def to_controller_kwargs(self) -> Dict[str, Any]:
        """Convert to LRTTController constructor arguments.

        Returns:
            Dictionary of arguments for LRTTController.__init__()
        """
        kwargs = {
            'transfer_lr': self.transfer_lr,
            'transfer_lr_scale': self.transfer_lr_scale,
            'transfer_every': self.transfer_every,
            'units_in_mbatch': self.units_in_mbatch,
            'lora_alpha': self.lora_alpha,
            'reinit_gain': self.reinit_gain,
            'reinit_mode': self.reinit_mode,
            'decay_factor': self.decay_factor,
            'correct_gradient_magnitudes': self.correct_gradient_magnitudes,
            'rank_chunk': self.rank_chunk,
            'ab_bl_mgmt': self.ab_bl_mgmt,
            'transfer_bl_mgmt': self.transfer_bl_mgmt,
            'forward_inject': self.forward_inject,
            'num_reads': self.num_reads,
            'multi_read_mode': self.multi_read_mode,
            'update_mode': self.update_mode,
            'transfer_method': self.transfer_method,
            # LRTT-v2 selector (direct kwargs: needed at __init__ for selector state setup)
            'selector_axis': self.selector_axis,
            'selector_policy': self.selector_policy,
            'selector_block_size': self.selector_block_size,
            'selector_seed': self.selector_seed,
            'selector_allow_partial_block': self.selector_allow_partial_block,
            'selector_random_unbiased_scale': self.selector_random_unbiased_scale,
            'selector_reset_b_on_advance': self.selector_reset_b_on_advance,
            # LRTT-v2 capacitor toggles (direct kwargs: gate update behavior)
            'cap_stabilizer_enabled': self.cap_stabilizer_enabled,
            'cap_compensate_transfer': self.cap_compensate_transfer,
        }
        # Post-init settings (set on controller after creation)
        kwargs['_post_init'] = {
            # Transfer mode & calibration
            'transfer_mode': self.transfer_mode,
            'transfer_micro_steps': self.transfer_micro_steps,
            'transfer_pilot_frac': self.transfer_pilot_frac,
            'sd_quantum': self.sd_quantum,
            # Read noise reduction
            'read_n_avg': self.read_n_avg,
            'differential_read': self.differential_read,
            # AGC settings
            'agc_enabled': self.agc_enabled,
            'agc_margin': self.agc_margin,
            'agc_max_iters': self.agc_max_iters,
            # Two-amplitude settings
            'two_amp_enabled': self.two_amp_enabled,
            'two_amp_ratio': self.two_amp_ratio,
            # Reconstruction parameters
            'recon_lambda_a': self.recon_lambda_a,
            'recon_lambda_b': self.recon_lambda_b,
            'recon_use_scalar_stabilizer': self.recon_use_scalar_stabilizer,
            'recon_use_exact_gram': self.recon_use_exact_gram,
            'recon_exact_gram_every': self.recon_exact_gram_every,
            'recon_ema_beta': self.recon_ema_beta,
            'recon_lr_scale': self.recon_lr_scale,
            'recon_clip_norm': self.recon_clip_norm,
            'recon_use_clip_norm': self.recon_use_clip_norm,
            # Separate A/B scaling
            'a_x_scaling': self.a_x_scaling,
            'a_d_scaling': self.a_d_scaling,
            'b_x_scaling': self.b_x_scaling,
            'b_d_scaling': self.b_d_scaling,
            # Debug logging
            'log_ab_scaling': self.log_ab_scaling,
            'log_ab_scaling_every': self.log_ab_scaling_every,
            # Separate BL for C tile
            'c_desired_bl': self.c_desired_bl,
            # LRTT-v2 capacitor tuning (post_init: tunable knobs, mirror recon_*)
            'cap_rho': self.cap_rho,
            'cap_compensation_max': self.cap_compensation_max,
            'cap_monitor_every': self.cap_monitor_every,
            'cap_target_rms': self.cap_target_rms,
            'cap_max_rms': self.cap_max_rms,
            'cap_soft_clip': self.cap_soft_clip,
            'cap_reset_mode': self.cap_reset_mode,
            # SRA-LRTT-v2 (Stochastic Reset-Anchor) settings
            'sra_anchor_source': self.sra_anchor_source,
            'sra_anchor_target_rms': self.sra_anchor_target_rms,
            'sra_anchor_gain_max': self.sra_anchor_gain_max,
            'sra_anchor_min_rms': self.sra_anchor_min_rms,
            'sra_use_cached_anchor': self.sra_use_cached_anchor,
            'sra_use_analog_projection': self.sra_use_analog_projection,
            'sra_resample_on_transfer': self.sra_resample_on_transfer,
            'sra_reset_b_on_transfer': self.sra_reset_b_on_transfer,
            'sra_b_reset_mode': self.sra_b_reset_mode,
            'sra_pulse_scramble_steps': self.sra_pulse_scramble_steps,
            'sra_pulse_scramble_lr': self.sra_pulse_scramble_lr,
            'sra_seed': self.sra_seed,
        }
        return kwargs
    
    @classmethod
    def from_legacy_lrtt_compound(cls, legacy_compound) -> 'PythonLRTTDevice':
        """Create from legacy LRTTTransferCompound for migration.
        
        Args:
            legacy_compound: LRTTTransferCompound instance
            
        Returns:
            Equivalent PythonLRTTDevice
        """
        return cls(
            rank=getattr(legacy_compound, 'rank', 4),
            transfer_every=getattr(legacy_compound, 'transfer_every', 32),
            transfer_lr=getattr(legacy_compound, 'transfer_lr', 1.0),
            lora_alpha=getattr(legacy_compound, 'lora_alpha', 1.0),
            reinit_gain=getattr(legacy_compound, 'reinit_gain', 0.1),
            units_in_mbatch=getattr(legacy_compound, 'units_in_mbatch', False),
            correct_gradient_magnitudes=getattr(legacy_compound, 'correct_gradient_magnitudes', False),
            forward_inject=getattr(legacy_compound, 'forward_inject', False),
            rank_chunk=getattr(legacy_compound, 'rank_chunk', None),
            unit_cell_devices=getattr(legacy_compound, 'unit_cell_devices', [ConstantStepDevice()] * 3)
        )


@dataclass
class PythonLRTTPreset(_PrintableMixin):
    """Preset configurations for common LRTT use cases."""
    
    @staticmethod
    def idealized(rank: int = 4, transfer_every: int = 32, lora_alpha: float = 1.0) -> PythonLRTTDevice:
        """Idealized LRTT with minimal noise and perfect devices.
        
        Args:
            rank: LoRA rank
            transfer_every: Transfer frequency
            lora_alpha: LoRA scaling factor
            
        Returns:
            Idealized PythonLRTTDevice configuration
        """
        from aihwkit.simulator.presets.devices import IdealizedPresetDevice
        
        ideal_device = IdealizedPresetDevice()
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=0.1,
            forward_inject=False,
            unit_cell_devices=[ideal_device, ideal_device, ideal_device]
        )
    
    @staticmethod
    def constant_step(rank: int = 4, transfer_every: int = 32, dw_min: float = 0.01) -> PythonLRTTDevice:
        """LRTT with ConstantStepDevice for all tiles.
        
        Args:
            rank: LoRA rank
            transfer_every: Transfer frequency  
            dw_min: Minimum weight update step
            
        Returns:
            ConstantStep PythonLRTTDevice configuration
        """
        device = ConstantStepDevice(
            dw_min=dw_min,
            dw_min_dtod=0.0,
            up_down_dtod=0.0,
            w_min=-1.0,
            w_max=1.0
        )
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=1.0,
            reinit_gain=0.1,
            forward_inject=False,
            unit_cell_devices=[device, device, device]
        )
    
    @staticmethod  
    def lora_style(rank: int = 8, lora_alpha: float = 16.0, transfer_every: int = 1) -> PythonLRTTDevice:
        """LoRA-style configuration with frequent transfers.
        
        Similar to standard LoRA but with analog tiles and periodic consolidation.
        
        Args:
            rank: LoRA rank (typically higher for LoRA-style)
            lora_alpha: LoRA alpha (typically higher: α = 16, 32)
            transfer_every: Transfer frequency (1 = every step)
            
        Returns:
            LoRA-style PythonLRTTDevice configuration
        """
        from aihwkit.simulator.presets.devices import IdealizedPresetDevice
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=0.05,  # Smaller reinit for frequent transfers
            forward_inject=False,
            correct_gradient_magnitudes=True,  # Better scaling for higher ranks
            unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice(), IdealizedPresetDevice()]
        )
    
    @staticmethod
    def mixed_precision(rank: int = 4, transfer_every: int = 16) -> PythonLRTTDevice:
        """Mixed precision: high precision visible, lower precision A/B.
        
        Args:
            rank: LoRA rank
            transfer_every: Transfer frequency
            
        Returns:
            Mixed precision PythonLRTTDevice configuration
        """
        # Lower precision for A/B (faster updates)
        low_precision = ConstantStepDevice(dw_min=0.05, w_min=-0.8, w_max=0.8)
        
        # Higher precision for visible (stable storage)
        high_precision = ConstantStepDevice(dw_min=0.001, w_min=-2.0, w_max=2.0)
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=1.0,
            reinit_gain=0.1,
            forward_inject=False,
            unit_cell_devices=[low_precision, low_precision, high_precision]
        )
    
    @staticmethod
    def inference_optimized(rank: int = 2, lora_alpha: float = 0.5) -> PythonLRTTDevice:
        """Inference-optimized configuration with forward injection.
        
        Args:
            rank: Lower rank for faster inference
            lora_alpha: Lower alpha for stability
            
        Returns:
            Inference-optimized PythonLRTTDevice configuration
        """
        from aihwkit.simulator.presets.devices import IdealizedPresetDevice
        
        return PythonLRTTDevice(
            rank=rank,
            transfer_every=1,  # Transfer immediately
            lora_alpha=lora_alpha,
            reinit_gain=0.0,  # No reinit needed for inference
            forward_inject=False,  # Essential for inference
            columns_mode=True,  # Optimized mode
            unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice(), IdealizedPresetDevice()]
        )

    @staticmethod
    def floating_ab_softbound_c(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dw_min: float = 0.001,
        lifetime: float = 0.0,
    ) -> 'PythonLRTTDevice':
        """LRTT with FloatingPoint A/B tiles and SoftBounds C tile.

        - A, B tiles: FloatingPointDevice (exact arithmetic, no noise)
        - C tile: SoftBoundsReferenceDevice (noise=0, w_max=1, w_min=-1)

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dw_min: Minimum weight update step for C tile
            lifetime: Retention lifetime for C tile (0 = no decay)

        Returns:
            PythonLRTTDevice configuration with FloatingPoint A/B and SoftBounds C
        """
        from aihwkit.simulator.configs.devices import (
            FloatingPointDevice,
            SoftBoundsReferenceDevice,
        )

        # SoftBounds C tile with COMPLETE noise removal
        c_device = SoftBoundsReferenceDevice(
            # Weight bounds
            w_max=1.0,
            w_min=-1.0,
            dw_min=dw_min,

            # ===== ALL NOISE = 0 (completely removed) =====
            dw_min_std=0.0,         # cycle-to-cycle noise
            write_noise_std=0.0,    # write noise
            diffusion=0.0,          # diffusion noise

            # Device-to-device variation = 0
            dw_min_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            up_down=0.0,
            up_down_dtod=0.0,

            # Lifetime (configurable)
            lifetime=lifetime,
            lifetime_dtod=0.0,

            # Slope variations = 0
            slope_up_dtod=0.0,
            slope_down_dtod=0.0,
        )

        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=0.1,
            forward_inject=False,
            unit_cell_devices=[
                FloatingPointDevice(),  # A: exact arithmetic
                FloatingPointDevice(),  # B: exact arithmetic
                c_device,               # C: SoftBounds (noise=0)
            ]
        )

    @staticmethod
    def sixt1c_ab(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dt_batch_sec: float = 1.0,
        include_retention: bool = True,
        c_device: Optional[PulsedDevice] = None,
        reinit_mode: str = "decay",
        decay_factor: float = 1.0
    ) -> 'PythonLRTTDevice':
        """LRTT with 6T1C devices for A/B tiles and configurable C tile.

        A/B tiles use 6T1C (6 Transistors, 1 Capacitor) devices based on
        experimental measurements. C tile (visible) can use any device.

        6T1C Device Characteristics (A/B tiles):
            - ~1000 conductance states per direction
            - Capacitor-based weight storage with exponential decay
            - Time constant τ ≈ 775 min (12.9 hours)
            - Decay target: 0V

        Update Model (LinearStepDevice):
            - dw_min = 0.001981
            - gamma_up = -0.1678 (slight saturation)
            - gamma_down = +0.1410 (near-linear)

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dt_batch_sec: Assumed time per mini-batch in seconds (for 6T1C retention)
            include_retention: Whether to include retention effects for 6T1C
            c_device: Device for C tile (visible). If None, uses IdealizedPresetDevice.
                      Can be any PulsedDevice: IdealizedPresetDevice, PCM, RRAM, etc.
            reinit_mode: Reinit strategy after transfer ('standard', 'decay', 'hybrid').
                         Default 'decay' for 6T1C to allow natural retention decay.
            decay_factor: Decay factor for reinit (default 1.0 = no artificial reinit,
                          only natural 6T1C retention decay affects A/B weights).

        Returns:
            PythonLRTTDevice configuration with 6T1C A/B and custom C device

        Example:
            >>> from aihwkit.simulator.presets.devices import PCMPresetDevice, ReRamESPresetDevice
            >>> # 6T1C A/B with PCM C tile
            >>> device = PythonLRTTPreset.sixt1c_ab(c_device=PCMPresetDevice())
            >>> # 6T1C A/B with RRAM C tile
            >>> device = PythonLRTTPreset.sixt1c_ab(c_device=ReRamESPresetDevice())
        """
        import math

        # Calculate lifetime from physical τ for 6T1C
        TAU_SEC = 46505.0  # Physical time constant: 775.1 min = 46505 sec
        if include_retention and dt_batch_sec > 0:
            delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
            lifetime = 1.0 / delta
        else:
            lifetime = 0.0  # No retention

        # Create 6T1C device for A/B tiles (LinearStepDevice)
        sixt1c_device = LinearStepDevice(
            # Core update parameters (fitted from 6T1C data)
            dw_min=0.001981,
            up_down=0.0,
            w_max=1.0,
            w_min=-1.0,
            gamma_up=-0.1678,
            gamma_down=0.1410,
            mult_noise=True,

            # Device-to-device variation
            dw_min_dtod=0.1,
            up_down_dtod=0.01,
            w_max_dtod=0.05,
            w_min_dtod=0.05,
            gamma_up_dtod=0.05,
            gamma_down_dtod=0.05,

            # Cycle-to-cycle variation
            dw_min_std=0.3,
            write_noise_std=0.0182,

            # LinearStepDevice specific
            mean_bound_reference=True,

            # Retention (capacitor leakage)
            lifetime=lifetime,
            lifetime_dtod=0.1 if include_retention else 0.0,
            reset=0.0,  # Decay toward 0V
            reset_dtod=0.0,
        )

        # C tile device: use provided device or default to Idealized with optimized dw_min
        # dw_min=0.001 gives ~96% transfer accuracy with cosine_sim=0.97
        # (default 0.0002 only transfers ~24% due to stochastic PWU limitations)
        if c_device is None:
            from aihwkit.simulator.presets.devices import IdealizedPresetDevice
            c_device = IdealizedPresetDevice(
                dw_min=0.001,  # Optimized for accurate transfer
                dw_min_std=0.0,  # No noise for clean transfer
                dw_min_dtod=0.0,
            )

        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=0.1,
            reinit_mode=reinit_mode,
            decay_factor=decay_factor,
            forward_inject=False,
            unit_cell_devices=[sixt1c_device, sixt1c_device, c_device]
        )

    @staticmethod
    def sixt1c_ab_pcm(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dt_batch_sec: float = 1.0
    ) -> 'PythonLRTTDevice':
        """LRTT with 6T1C A/B tiles and PCM C tile.

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dt_batch_sec: Assumed time per mini-batch in seconds

        Returns:
            PythonLRTTDevice with 6T1C A/B and PCM C
        """
        from aihwkit.simulator.presets.devices import PCMPresetDevice
        return PythonLRTTPreset.sixt1c_ab(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            dt_batch_sec=dt_batch_sec,
            c_device=PCMPresetDevice()
        )

    @staticmethod
    def sixt1c_ab_rram(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dt_batch_sec: float = 1.0
    ) -> 'PythonLRTTDevice':
        """LRTT with 6T1C A/B tiles and RRAM C tile.

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dt_batch_sec: Assumed time per mini-batch in seconds

        Returns:
            PythonLRTTDevice with 6T1C A/B and RRAM C
        """
        from aihwkit.simulator.presets.devices import ReRamESPresetDevice
        return PythonLRTTPreset.sixt1c_ab(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            dt_batch_sec=dt_batch_sec,
            c_device=ReRamESPresetDevice()
        )

    @staticmethod
    def sixt1c_ab_ideal(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dt_batch_sec: float = 1.0
    ) -> 'PythonLRTTDevice':
        """LRTT with 6T1C A/B tiles and Idealized C tile.

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dt_batch_sec: Assumed time per mini-batch in seconds

        Returns:
            PythonLRTTDevice with 6T1C A/B and Idealized C
        """
        from aihwkit.simulator.presets.devices import IdealizedPresetDevice
        return PythonLRTTPreset.sixt1c_ab(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            dt_batch_sec=dt_batch_sec,
            c_device=IdealizedPresetDevice()
        )

    @staticmethod
    def sixt1c_all(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dt_batch_sec: float = 1.0,
        include_retention: bool = True
    ) -> 'PythonLRTTDevice':
        """LRTT with 6T1C devices for ALL tiles (A, B, and C).

        All three tiles use identical 6T1C device characteristics.

        Args:
            rank: LoRA rank dimension
            transfer_every: Transfer frequency (steps)
            lora_alpha: LoRA scaling factor
            dt_batch_sec: Assumed time per mini-batch in seconds (for retention)
            include_retention: Whether to include retention effects

        Returns:
            PythonLRTTDevice configuration with 6T1C for all tiles
        """
        import math

        # Calculate lifetime from physical τ
        TAU_SEC = 46505.0
        if include_retention and dt_batch_sec > 0:
            delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
            lifetime = 1.0 / delta
        else:
            lifetime = 0.0

        # Create 6T1C device (same for A, B, C)
        sixt1c_device = LinearStepDevice(
            dw_min=0.001981,
            up_down=0.0,
            w_max=1.0,
            w_min=-1.0,
            gamma_up=-0.1678,
            gamma_down=0.1410,
            mult_noise=True,
            dw_min_dtod=0.1,
            up_down_dtod=0.01,
            w_max_dtod=0.05,
            w_min_dtod=0.05,
            gamma_up_dtod=0.05,
            gamma_down_dtod=0.05,
            dw_min_std=0.3,
            write_noise_std=0.0182,
            mean_bound_reference=True,
            lifetime=lifetime,
            lifetime_dtod=0.1 if include_retention else 0.0,
            reset=0.0,
            reset_dtod=0.0,
        )

        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
            reinit_gain=0.1,
            forward_inject=False,
            unit_cell_devices=[sixt1c_device, sixt1c_device, sixt1c_device]
        )

    @staticmethod
    def sixt1c_sra_all(
        rank: int = 4,
        transfer_every: int = 32,
        dt_batch_sec: float = 1.0,
        include_retention: bool = True,
        anchor_source: str = "reset_columns",
        reset_std: float = 0.01,
        c_dw_min: float = 0.001981,
    ) -> 'PythonLRTTDevice':
        """SRA-LRTT-v2 with LinearStepDevice 6T1C-like A/B/C tiles.

        Constructs SEPARATE A/B/C device instances (not aliased) so that A may
        carry apply_write_noise_on_set=True (for the set_zero_write_noise anchor
        ablation) while B and C keep apply_write_noise_on_set=False (so that
        B reset and C pretrained-init are exact / noise-free).

        Args:
            rank: LoRA rank.
            transfer_every: Transfer frequency.
            dt_batch_sec: Per-batch wall-clock seconds (for retention model).
            include_retention: If True, model 6T1C lifetime decay.
            anchor_source: SRA anchor source. 'reset_columns' is the main hardware
                source; 'set_zero_write_noise' / 'pulse_scramble' / 'explicit_gaussian'
                are alternates / ablations.
            reset_std: stddev of LinearStepDevice reset distribution (used by
                'reset_columns' source).
            c_dw_min: dw_min for C tile (transfer pulses).
        """
        import math
        TAU_SEC = 46505.0
        if include_retention and dt_batch_sec > 0:
            delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
            lifetime = 1.0 / delta
        else:
            lifetime = 0.0

        def make_dev(apply_noise_on_set: bool, dw_min: float = 0.001981):
            return LinearStepDevice(
                dw_min=dw_min,
                up_down=0.0,
                w_max=1.0,
                w_min=-1.0,
                gamma_up=-0.1678,
                gamma_down=0.1410,
                mult_noise=True,
                dw_min_dtod=0.1,
                up_down_dtod=0.01,
                w_max_dtod=0.05,
                w_min_dtod=0.05,
                gamma_up_dtod=0.05,
                gamma_down_dtod=0.05,
                dw_min_std=0.3,
                write_noise_std=0.0182,
                apply_write_noise_on_set=apply_noise_on_set,
                mean_bound_reference=True,
                lifetime=lifetime,
                lifetime_dtod=0.1 if include_retention else 0.0,
                reset=0.0,
                reset_std=reset_std,
                reset_dtod=0.0,
            )

        # A: may use set_zero_write_noise → allow write noise on set.
        a_dev = make_dev(apply_noise_on_set=(anchor_source == "set_zero_write_noise"))
        # B: reset must be as close to exact zero as possible.
        b_dev = make_dev(apply_noise_on_set=False)
        # C: core transfer is pulsed via update(); set_weights() at init must not add noise.
        c_dev = make_dev(apply_noise_on_set=False, dw_min=c_dw_min)

        return PythonLRTTDevice(
            rank=rank,
            transfer_every=transfer_every,
            transfer_lr=1.0,
            lora_alpha=1.0,
            reinit_gain=0.0,
            reinit_mode="standard",
            a_init_mode="zero",
            b_init_mode="zero",
            forward_inject=False,
            update_mode="stochastic_reset_anchor",
            transfer_method="stochastic_anchor",
            cap_stabilizer_enabled=True,
            cap_rho=1.0,
            cap_compensate_transfer=True,
            sra_anchor_source=anchor_source,
            sra_anchor_target_rms=None,
            sra_b_reset_mode="set_zero",
            unit_cell_devices=[a_dev, b_dev, c_dev],
        )


# =============================================================================
# 6T1C Device Utility Functions
# =============================================================================

def get_6t1c_lifetime_for_dt_batch(dt_batch_sec: float) -> float:
    """Calculate AIHWKit lifetime parameter for 6T1C given dt_batch.

    The 6T1C capacitor has a physical time constant τ = 46505 seconds (775.1 min).
    The AIHWKit lifetime parameter depends on the dt_batch assumption.

    Args:
        dt_batch_sec: Assumed time per mini-batch in seconds.

    Returns:
        Lifetime parameter for AIHWKit configuration.

    Example:
        >>> # For 1 second per batch
        >>> lifetime = get_6t1c_lifetime_for_dt_batch(1.0)
        >>> print(f"lifetime = {lifetime:.0f}")
        lifetime = 46506

        >>> # For 1 minute per batch
        >>> lifetime = get_6t1c_lifetime_for_dt_batch(60.0)
        >>> print(f"lifetime = {lifetime:.0f}")
        lifetime = 776
    """
    import math
    TAU_SEC = 46505.0  # Physical time constant in seconds
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    return 1.0 / delta