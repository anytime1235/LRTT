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

    transfer_lr_scale: str = "sqrt_rank"
    """Auto-scaling mode for transfer_lr based on rank:
    - 'none': No scaling, use transfer_lr as-is
    - 'sqrt_rank': Scale by 1/sqrt(rank), i.e., transfer_lr / sqrt(rank) (default)
    - 'rank': Scale by 1/rank, i.e., transfer_lr / rank
    """

    lora_alpha: float = 1.0
    """LoRA scaling factor α in W_eff = W_visible + α * A @ B."""
    
    reinit_gain: float = 0.1
    """Kaiming initialization gain for B matrix after transfer."""

    reinit_mode: str = "orthogonal"
    """Reinit strategy after transfer:
    - 'orthogonal': A=0, B=orthogonal matrix via QR (default, best stability)
    - 'standard': A=0, B=Kaiming (original LRTT)
    - 'decay': A*=decay_factor, B*=decay_factor (gradual decay)
    - 'hybrid': A=0, B*=decay_factor (hybrid approach)
    """

    decay_factor: float = 0.9
    """Decay factor for 'decay' and 'hybrid' reinit modes (0 < decay_factor < 1)."""
    
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

    # === Transfer Mode Parameters ===
    use_onehot: bool = True
    """Transfer read mode:
    - True: Use one-hot forward/backward for analog-realistic reading (default)
    - False: Use direct weight access via get_weights()
    """

    use_sigma_delta: bool = True
    """Transfer write mode (only used when use_onehot=True):
    - True: Use ΣΔ modulation with integer pulses (default)
    - False: Use simple pulsed update (rank times, 1 update per rank)
    """
    
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

    # === Reconstruction Update Parameters (for forward_inject=False) ===
    # When forward_inject=False, A,B act as "gradient buffers" storing low-rank
    # approximation of gradient G = D^T @ X. Goal: minimize L_rec = ||AB + G||^2
    # so that AB ≈ -G, making C += transfer_lr*AB equivalent to SGD descent.

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

    recon_ema_beta: float = 0.9
    """EMA decay for tracking sA, sB norms (0.9~0.99 recommended)."""

    recon_lr_scale: float = 1.0
    """Additional learning rate scale for reconstruction updates (0.1~1.0)."""

    recon_clip_norm: float = 10.0
    """Max norm for A,B clipping (safety fallback). Only used if recon_use_clip_norm=True."""

    recon_use_clip_norm: bool = False
    """Enable norm clipping for A,B. Disabled by default as orthogonal reinit provides stability."""
    
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
        valid_modes = ["standard", "decay", "hybrid", "orthogonal"]
        if self.reinit_mode not in valid_modes:
            raise ValueError(f"reinit_mode must be one of {valid_modes}, got '{self.reinit_mode}'")

        # Validate decay_factor
        if not (0 < self.decay_factor <= 1):
            raise ValueError(f"decay_factor must be in (0, 1], got {self.decay_factor}")
            
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
        return {
            'transfer_lr': self.transfer_lr,
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
            'use_onehot': self.use_onehot,
            'use_sigma_delta': self.use_sigma_delta,
        }
    
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
            forward_inject=getattr(legacy_compound, 'forward_inject', True),
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
            forward_inject=False,
            columns_mode=True,  # Optimized mode
            unit_cell_devices=[IdealizedPresetDevice(), IdealizedPresetDevice(), IdealizedPresetDevice()]
        )

    @staticmethod
    def sixt1c_ab(
        rank: int = 4,
        transfer_every: int = 32,
        lora_alpha: float = 1.0,
        dt_batch_sec: float = 1.0,
        include_retention: bool = True,
        c_device: Optional[PulsedDevice] = None,
        reinit_mode: str = "orthogonal",
        decay_factor: float = 0.9
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
            reinit_mode: Reinit strategy after transfer ('orthogonal', 'standard', 'decay', 'hybrid').
                         Default 'orthogonal' for best stability via QR decomposition.
            decay_factor: Decay factor for 'decay'/'hybrid' modes (default 0.9).

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