# -*- coding: utf-8 -*-
"""LRTT configuration helper for GLUE/SQuAD sweep tasks."""

import sys
import os

# Add LRTT src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LRTT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(LRTT_ROOT, "src"))

from aihwkit.simulator.configs.lrtt_config import (
    lrtt_sixt1c_ab_ideal_config,
    lrtt_idealized_config,
    PythonLRTTRPUConfig,
)


def create_lrtt_config(
    rank: int,
    transfer_every: int,
    transfer_lr: float,
    lora_alpha: float = 1.0,
    reinit_gain: float = 0.1,
    reinit_mode: str = "standard",
    forward_inject: bool = False,
    transfer_method: str = "onehot",
    update_mode: str = "lora",
    # Auto-scale (A/B update LR normalization)
    auto_scale: bool = False,
    auto_scale_momentum: float = 0.99,
    # Transfer EMA scaling (transfer magnitude normalization)
    transfer_ema_scale: bool = False,
    transfer_ema_momentum: float = 0.99,
    transfer_ema_target_norm: float = 0.0,
) -> PythonLRTTRPUConfig:
    """Create LRTT configuration for GLUE/SQuAD tasks.

    Args:
        rank: LRTT rank (similar to LoRA rank)
        transfer_every: Transfer frequency in steps
        transfer_lr: Transfer learning rate scalar
        lora_alpha: LoRA scaling factor (default 1.0)
        reinit_gain: Kaiming initialization gain for B matrix after transfer
        reinit_mode: Reinit strategy ('standard', 'decay', 'hybrid')
        forward_inject: Enable forward injection optimization
        transfer_method: Transfer method ('onehot', 'direct', 'set')
        update_mode: A/B update mode ('lora', 'reconstruction')
        auto_scale: Normalize lr_eff by EMA(|x|_max * |d|_max)
        auto_scale_momentum: EMA momentum for auto_scale (default 0.99)
        transfer_ema_scale: Normalize transfer_lr by EMA(||A@B||_F)
        transfer_ema_momentum: Transfer norm EMA momentum (default 0.99)
        transfer_ema_target_norm: Target norm (0 = auto-calibrate from first transfer)

    Returns:
        Configured PythonLRTTRPUConfig
    """
    # Use the 6T1C A/B with idealized C configuration
    rpu_config = lrtt_sixt1c_ab_ideal_config(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=lora_alpha,
    )

    # Apply additional settings
    rpu_config.device.transfer_lr = transfer_lr
    rpu_config.device.reinit_gain = reinit_gain
    rpu_config.device.reinit_mode = reinit_mode
    rpu_config.device.forward_inject = forward_inject
    rpu_config.device.transfer_method = transfer_method
    rpu_config.device.update_mode = update_mode

    # Auto-scale and transfer EMA scaling
    rpu_config.device.auto_scale = auto_scale
    rpu_config.device.auto_scale_momentum = auto_scale_momentum
    rpu_config.device.transfer_ema_scale = transfer_ema_scale
    rpu_config.device.transfer_ema_momentum = transfer_ema_momentum
    rpu_config.device.transfer_ema_target_norm = transfer_ema_target_norm

    # Configure weight scaling for C tile to prevent clipping of pretrained weights
    # This scales weights into tile bounds (w_max=1.0) instead of hard clipping
    # which would destroy pretrained weight information and cause loss explosion
    rpu_config.mapping.weight_scaling_omega = 1.0        # Scale to ±1.0 range
    rpu_config.mapping.weight_scaling_columnwise = True  # Per-column scaling
    rpu_config.mapping.learn_out_scaling = True          # Learnable output scaling
    rpu_config.mapping.out_scaling_columnwise = True     # Per-column output scaling

    return rpu_config


def get_glue_preset_config(
    rank: int = 8,
    transfer_every: int = 1000,
    lora_alpha: float = 32.0,
    use_ideal: bool = False,
) -> PythonLRTTRPUConfig:
    """Get LRTT configuration preset for GLUE tasks.

    Args:
        rank: LRTT rank (similar to LoRA rank)
        transfer_every: Transfer frequency in steps
        lora_alpha: LoRA scaling factor
        use_ideal: If True, use fully idealized config

    Returns:
        Configured PythonLRTTRPUConfig
    """
    if use_ideal:
        rpu_config = lrtt_idealized_config(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
        )
    else:
        rpu_config = lrtt_sixt1c_ab_ideal_config(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
        )

    # Configure weight scaling for C tile to prevent clipping of pretrained weights
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config
