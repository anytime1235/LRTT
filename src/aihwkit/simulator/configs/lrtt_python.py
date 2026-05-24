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
from aihwkit.simulator.parameters.mapping import MappingParameter


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
    """Transfer frequency: every N mini-batches (if units_in_mbatch=True)
    or every N mat-vec operations (if units_in_mbatch=False)."""
    
    transfer_lr: float = 1.0
    """Transfer learning rate scalar applied during A⊗B -> visible transfer."""

    scale_transfer_lr: bool = True
    """If True, multiply transfer_lr by the current SGD learning rate (lr_sgd).
    Effective: lr_tr = transfer_lr * lr_sgd (when True) or transfer_lr (when False)."""

    transfer_fast_lr_ref: str = "geomean"
    """Reference for correct_gradient_magnitudes transfer LR division:
    - 'A': divide by lr_eff_a
    - 'B': divide by lr_eff_b
    - 'geomean': divide by sqrt(lr_eff_a * lr_eff_b)
    Only used when correct_gradient_magnitudes=True."""

    lora_alpha: float = 1.0
    """LoRA scaling factor α in W_eff = W_visible + α * A @ B.
    Used only for forward-injection scaling (y = Cx + α·ABx).
    No longer affects A/B learning rate — use fast_lr instead."""

    fi_continuous_alpha: bool = False
    """When True and forward_inject=True, override lora_alpha with transfer_lr
    (or transfer_lr * lr_sgd when scale_transfer_lr=True) so that the forward
    injection scaling α equals the effective transfer learning rate every step.
    This ensures continuity: the α used in y = Cx + α·ABx matches the
    lr used in C += lr·AB transfer."""

    fast_lr: float = 1.0
    """Fixed learning rate constant for A/B weight updates (replaces lr*lora_alpha).
    In auto_scale_mode='none': lr_eff = fast_lr.
    In auto_scale_mode='shared'/'separate': fast_lr is normalized by signal EMA."""

    auto_scale_mode: str = "none"
    """Auto-scale mode for A/B learning rate normalization:
    - 'none': lr_eff = fast_lr (no signal-based scaling)
    - 'shared': lr_eff = fast_lr / (μ_x · μ_d), shared across A and B
    - 'separate': lr_eff_a and lr_eff_b computed independently with granularity"""

    auto_momentum: float = 0.99
    """EMA momentum for auto-scale signal tracking. τ = (1 - auto_momentum) / m_batch.
    Higher values = smoother EMA, slower adaptation. Must be in (0, 1)."""

    reinit_gain: float = 1.0
    """Kaiming uniform initialization gain (multiplier). 1.0 = standard PyTorch nn.Linear default."""

    reinit_mode: str = "standard"
    """Reinit strategy after transfer:
    - 'standard': A=0, B=Kaiming (original LRTT)
    - 'decay': no reinit, 6T1C capacitor decay handles weight reduction
    - 'hybrid': A=0, B unchanged (6T1C capacitor decay handles B)
    - 'orthogonal_zero': A=0, B=Random Orthogonal (FROZEN)
    - 'orthogonal_decay': A unchanged, B=Random Orthogonal (FROZEN)
    - 'gauss_b_zero': A=0, B=fresh Gaussian draw every transfer (B frozen between transfers; B device's reset_std = σ)
    - 'gauss_b_decay': A first-time only (6T1C decay), B=fresh Gaussian every transfer
    - 'gauss_a_zero': A=fresh Gaussian every transfer, B=0 every transfer (mirror of gauss_b_zero)
    - 'gauss_a_decay': A=fresh Gaussian every transfer, B=0 first-init only (then 6T1C decay) (mirror of gauss_b_decay)
    - 'selector_b_zero': A=0, B=fresh row-coordinate selector (rank one-hot rows over x_size) every transfer
    - 'selector_b_decay': A first-init only (then 6T1C decay), B=fresh row-coordinate selector every transfer
    - 'selector_a_zero': A=fresh column-coordinate selector (rank one-hot cols over d_size) every transfer, B=0
    - 'selector_a_decay': A=fresh selector every transfer, B=0 first-init only (then 6T1C decay) (mirror of selector_b_decay)
    - 'sparse_b_zero': A=0, B=sparse ±1 Rademacher (each entry ±1 w.p. b_density/2 each, 0 w.p. 1-b_density) every transfer
    - 'sparse_a_zero': A=sparse ±1 Rademacher (each entry ±1 w.p. a_density/2 each, 0 w.p. 1-a_density) every transfer, B=0 (mirror of sparse_b_zero)
    - 'binary_b_zero': A=0, B=Bernoulli {0,1} (each entry 1 w.p. b_density, 0 w.p. 1-b_density) every transfer
    - 'binary_a_zero': A=Bernoulli {0,1} (each entry 1 w.p. a_density, 0 w.p. 1-a_density) every transfer, B=0 (mirror of binary_b_zero)
    """

    a_density: float = 1.0
    """Density (fraction of nonzero entries) for sparse_a_zero. p=1 → dense ±1 Rademacher, p<1 → sparser pattern."""

    b_density: float = 1.0
    """Density (fraction of nonzero entries) for sparse_b_zero. p=1 → dense ±1 Rademacher, p<1 → sparser pattern."""

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
    Default is "onehot"."""

    # === Transfer Rank Scheduling ===
    transfer_rank_schedule: str = "all"
    """Transfer rank scheduling mode:
    - "all": Transfer all ranks at once (default, original behavior)
    - "round_robin": Cycle through ranks, transferring a subset each time
    Default is "all"."""

    transfer_ranks_per_step: int = 1
    """Number of ranks to transfer per transfer event in round_robin mode.
    Only used when transfer_rank_schedule="round_robin".
    Must be >= 1 and <= rank. Default is 1."""

    # === Advanced Parameters ===
    units_in_mbatch: bool = False
    """If True, transfer_every is in mini-batch units (TE=1 → every batch).
    If False, transfer_every is in mat-vec units (TE=1 → every sample).
    Matches TikiTaka convention."""

    no_adc_ab_projection: bool = False
    """If True, remove ADC/DAC quantization between A/B projections.
    B output and A backward output pass through at full precision."""
    
    correct_gradient_magnitudes: bool = False
    """If True, divide transfer LR by the fast effective LR (lr_eff_a/b or geomean).
    This corrects transfer magnitude relative to fast A/B update speed.
    The old A/B sqrt(rank) correction has been removed."""
    
    forward_inject: bool = False
    """Enable forward injection optimization: W_eff composition."""

    ab_pulse_type: str = "default"
    """Pulse type for A/B tile weight updates:
    - 'default': Use RPUConfig's default pulse type (typically StochasticCompressed)
    - 'none': Pure floating point update (no pulses, no device effects)
    - 'none_with_device': Floating point update with device effects (weight clipping)
    - 'stochastic_compressed': Stochastic bit lines, +/- in same pass
    - 'mean_count': Coincidence based in prob (p_a * p_b)
    - 'deterministic_implicit': Deterministic coincidences with quantized input/error
    """

    # === C Tile Parameter Training ===
    train_c_bias: bool = False
    """Allow C tile bias to be trainable. Default False (frozen).
    Set True for scratch training where bias needs to be learned."""

    # === Dynamic Transfer Every ===
    dynamic_te: bool = False
    """Enable dynamic transfer_every that increases as LR decays.
    TE(t) = clip(te_min, te_max, round(TE_0 * (lr_peak / lr_current)^p)).
    Only activates in LR decay phase (lr_current < lr_peak)."""

    dynamic_te_power: float = 1.0
    """Power p for dynamic TE. 1.0=exact LR inverse, 0.5=smoother."""

    dynamic_te_min: Optional[int] = None
    """Minimum TE floor. None=use TE_0 (never decrease below initial value)."""

    dynamic_te_max: Optional[int] = None
    """Maximum TE ceiling. None=10*TE_0."""

    te_warmup_schedule: Optional[List[int]] = None
    """Step-wise TE warmup schedule, e.g. [32, 64, 128, 230].
    During warmup, TE transitions through these values in equal intervals.
    The last value should match transfer_every (TE_0). None=no warmup ramp."""

    te_warmup_steps: int = 0
    """Number of steps for TE warmup ramp. The schedule is divided equally."""

    # === Tile Mapping Parameters ===
    mapping_ab: MappingParameter = field(default_factory=lambda: MappingParameter(
        weight_scaling_omega=0.0,
        learn_out_scaling=False,
    ))
    """A/B tile mapping (default for both A and B). Default: no weight scaling
    (omega=0), no learnable out_scaling. Overridden per-tile by mapping_a / mapping_b
    when those are not None."""

    mapping_a: Optional[MappingParameter] = None
    """Optional override for A tile mapping. None → use mapping_ab."""

    mapping_b: Optional[MappingParameter] = None
    """Optional override for B tile mapping. None → use mapping_ab."""

    mapping_c: MappingParameter = field(default_factory=lambda: MappingParameter(
        weight_scaling_omega=1.0,
        weight_scaling_columnwise=True,
        learn_out_scaling=True,
        out_scaling_columnwise=True,
    ))
    """C tile mapping. Default: pretrained weight scaling + trainable columnwise out_scaling."""

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
    a_desired_bl: Optional[int] = None
    """Desired BL for A tile updates. If None, uses global desired_bl from UpdateParameters."""

    b_desired_bl: Optional[int] = None
    """Desired BL for B tile updates. If None, uses global desired_bl from UpdateParameters."""

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
        # Validate fast_lr and auto-scale
        if self.fast_lr <= 0:
            raise ValueError(f"fast_lr must be positive, got {self.fast_lr}")
        if self.auto_scale_mode not in ["none", "shared", "separate"]:
            raise ValueError(f"auto_scale_mode must be 'none', 'shared', or 'separate', got '{self.auto_scale_mode}'")
        if not (0.0 < self.auto_momentum < 1.0):
            raise ValueError(f"auto_momentum must be in (0, 1), got {self.auto_momentum}")
        if self.transfer_fast_lr_ref not in ["A", "B", "geomean"]:
            raise ValueError(f"transfer_fast_lr_ref must be 'A', 'B', or 'geomean', got '{self.transfer_fast_lr_ref}'")

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
        valid_modes = ["standard", "decay", "hybrid", "orthogonal_zero", "orthogonal_decay",
                       "gauss_b_zero", "gauss_b_decay", "gauss_a_zero", "gauss_a_decay",
                       "selector_b_zero", "selector_b_decay", "selector_a_zero", "selector_a_decay",
                       "sparse_a_zero", "sparse_b_zero",
                       "binary_a_zero", "binary_b_zero"]
        if self.reinit_mode not in valid_modes:
            raise ValueError(f"reinit_mode must be one of {valid_modes}, got '{self.reinit_mode}'")

        # Validate density params
        if not (0.0 < self.a_density <= 1.0):
            raise ValueError(f"a_density must be in (0, 1], got {self.a_density}")
        if not (0.0 < self.b_density <= 1.0):
            raise ValueError(f"b_density must be in (0, 1], got {self.b_density}")

        # Validate num_reads
        if self.num_reads < 1:
            raise ValueError(f"num_reads must be >= 1, got {self.num_reads}")

        # Validate multi_read_mode
        valid_read_modes = ["average", "per_read"]
        if self.multi_read_mode not in valid_read_modes:
            raise ValueError(f"multi_read_mode must be one of {valid_read_modes}, got '{self.multi_read_mode}'")

        # Validate dynamic TE parameters
        if self.dynamic_te_power <= 0:
            raise ValueError(f"dynamic_te_power must be positive, got {self.dynamic_te_power}")
        if self.dynamic_te_min is not None and self.dynamic_te_min <= 0:
            raise ValueError(f"dynamic_te_min must be positive or None, got {self.dynamic_te_min}")
        if self.dynamic_te_max is not None and self.dynamic_te_max <= 0:
            raise ValueError(f"dynamic_te_max must be positive or None, got {self.dynamic_te_max}")
        if (self.dynamic_te_min is not None and self.dynamic_te_max is not None
                and self.dynamic_te_min > self.dynamic_te_max):
            raise ValueError(f"dynamic_te_min ({self.dynamic_te_min}) must be <= dynamic_te_max ({self.dynamic_te_max})")

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
        valid_update_modes = ["lora", "reconstruction"]
        if self.update_mode not in valid_update_modes:
            raise ValueError(f"update_mode must be one of {valid_update_modes}, got '{self.update_mode}'")

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

        # Validate transfer_rank_schedule
        valid_rank_schedules = ["all", "round_robin"]
        if self.transfer_rank_schedule not in valid_rank_schedules:
            raise ValueError(f"transfer_rank_schedule must be one of {valid_rank_schedules}, got '{self.transfer_rank_schedule}'")

        # Validate transfer_ranks_per_step
        if self.transfer_ranks_per_step < 1:
            raise ValueError(f"transfer_ranks_per_step must be >= 1, got {self.transfer_ranks_per_step}")
        if self.transfer_ranks_per_step > self.rank:
            raise ValueError(f"transfer_ranks_per_step ({self.transfer_ranks_per_step}) must be <= rank ({self.rank})")

        # Validate rank_chunk
        if self.rank_chunk is not None and self.rank_chunk <= 0:
            raise ValueError(f"rank_chunk must be positive or None, got {self.rank_chunk}")
            
        # Validate unit cell devices
        if len(self.unit_cell_devices) != 3:
            raise ValueError(f"Must provide exactly 3 unit_cell_devices for [fastA, fastB, visible], got {len(self.unit_cell_devices)}")
            
        # Set default rank_chunk
        if self.rank_chunk is None:
            self.rank_chunk = self.rank
            
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
            'transfer_every': self.transfer_every,
            'units_in_mbatch': self.units_in_mbatch,
            'lora_alpha': self.lora_alpha,
            'reinit_gain': self.reinit_gain,
            'reinit_mode': self.reinit_mode,
            'correct_gradient_magnitudes': self.correct_gradient_magnitudes,
            'fast_lr': self.fast_lr,
            'scale_transfer_lr': self.scale_transfer_lr,
            'transfer_fast_lr_ref': self.transfer_fast_lr_ref,
            'rank_chunk': self.rank_chunk,
            'ab_bl_mgmt': self.ab_bl_mgmt,
            'transfer_bl_mgmt': self.transfer_bl_mgmt,
            'forward_inject': self.forward_inject,
            'dynamic_te': self.dynamic_te,
            'dynamic_te_power': self.dynamic_te_power,
            'dynamic_te_min': self.dynamic_te_min,
            'dynamic_te_max': self.dynamic_te_max,
            'te_warmup_schedule': self.te_warmup_schedule,
            'te_warmup_steps': self.te_warmup_steps,
            'num_reads': self.num_reads,
            'multi_read_mode': self.multi_read_mode,
            'update_mode': self.update_mode,
            'transfer_method': self.transfer_method,
            'transfer_rank_schedule': self.transfer_rank_schedule,
            'transfer_ranks_per_step': self.transfer_ranks_per_step,
            'fi_continuous_alpha': self.fi_continuous_alpha,
        }
        # Post-init settings (set on controller after creation)
        kwargs['_post_init'] = {
            # Auto-scale settings
            'auto_scale_mode': self.auto_scale_mode,
            'auto_momentum': self.auto_momentum,
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
            reinit_gain=getattr(legacy_compound, 'reinit_gain', 1.0),
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
            reinit_gain=1.0,
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
            reinit_gain=1.0,
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
            reinit_gain=1.0,
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
            reinit_gain=1.0,
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
            reinit_gain=1.0,
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
            mult_noise=False,

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
            reinit_gain=1.0,
            reinit_mode=reinit_mode,
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
            mult_noise=False,
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
            reinit_gain=1.0,
            forward_inject=False,
            unit_cell_devices=[sixt1c_device, sixt1c_device, sixt1c_device]
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