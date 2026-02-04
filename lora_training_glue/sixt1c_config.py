"""
Sixt1c (6T1C) configuration for LoRA inference.

This module provides RPU configuration for 6T1C (LinearStepDevice) analog hardware
with retention/lifetime parameters. Designed for inference-only scenarios without
transfer mechanisms.

Key characteristics:
- LinearStepDevice with 6T1C physical parameters
- Lifetime from TAU_SEC = 46505.0 seconds (775.1 minutes)
- write_noise_std = 0.0 (as requested)
- No PCM noise model (uses device retention instead)
"""

import math
from aihwkit.simulator.configs import TorchInferenceRPUConfig
from aihwkit.simulator.presets.utils import IOParameters
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    WeightClipType,
    WeightRemapType,
)
from aihwkit.simulator.configs.devices import LinearStepDevice


# 6T1C Physical Parameters (from LRTT)
TAU_SEC = 46505.0  # Time constant in seconds (775.1 minutes)
DW_MIN = 0.001981  # Minimum weight change
GAMMA_UP = -0.1678  # Nonlinearity for positive updates
GAMMA_DOWN = 0.1410  # Nonlinearity for negative updates


def gen_sixt1c_inference_config(
    dt_batch_sec: float = 1.0,
    include_retention: bool = True,
    write_noise_std: float = 0.0,
    output_noise_level: float = 0.0,
):
    """
    Generate Sixt1c (6T1C) RPU config for LoRA inference.

    This configuration uses LinearStepDevice with 6T1C characteristics.
    Retention is handled via the lifetime parameter instead of PCM drift.

    Args:
        dt_batch_sec: Time step between batches in seconds (default: 1.0)
        include_retention: Whether to include retention decay (default: True)
        write_noise_std: Write noise standard deviation (default: 0.0)
        output_noise_level: Output noise level for forward pass (default: 0.0)

    Returns:
        TorchInferenceRPUConfig configured for 6T1C device characteristics
    """
    rpu_config = TorchInferenceRPUConfig()

    # Mapping configuration
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    # Weight modifier configuration
    rpu_config.modifier.std_dev = 0.0  # No additional weight noise for sixt1c
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL

    # Remap configuration
    rpu_config.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC

    # Forward pass configuration
    rpu_config.forward = IOParameters()
    rpu_config.forward.out_noise = output_noise_level
    rpu_config.forward.is_perfect = False
    rpu_config.forward.inp_res = 1 / (2**8 - 2)  # 8-bit input resolution
    rpu_config.forward.out_res = 1 / (2**8 - 2)  # 8-bit output resolution

    # Clipping configuration
    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 3

    # No PCM noise model for sixt1c (retention handled by lifetime)
    rpu_config.noise_model = None
    rpu_config.drift_compensation = None

    return rpu_config


def calculate_lifetime(dt_batch_sec: float, tau_sec: float = TAU_SEC) -> float:
    """
    Calculate the lifetime parameter for 6T1C retention decay.

    The lifetime parameter controls exponential weight decay during inference.
    For 6T1C devices, decay follows: w(t) = w(0) * exp(-t/tau)

    Args:
        dt_batch_sec: Time step between batches in seconds
        tau_sec: Time constant (default: 46505.0 seconds)

    Returns:
        Lifetime value for use in LinearStepDevice configuration
    """
    if dt_batch_sec <= 0:
        return 0.0

    delta = 1 - math.exp(-dt_batch_sec / tau_sec)
    if delta <= 0:
        return 0.0

    lifetime = 1.0 / delta
    return lifetime


def get_sixt1c_device_params(
    include_retention: bool = True,
    dt_batch_sec: float = 1.0,
    write_noise_std: float = 0.0,
):
    """
    Get 6T1C device parameters for LinearStepDevice configuration.

    Args:
        include_retention: Whether to include retention decay
        dt_batch_sec: Time step between batches in seconds
        write_noise_std: Write noise standard deviation

    Returns:
        Dictionary of device parameters
    """
    # Calculate lifetime for retention
    if include_retention and dt_batch_sec > 0:
        lifetime = calculate_lifetime(dt_batch_sec)
        lifetime_dtod = 0.1  # Device-to-device variation in lifetime
    else:
        lifetime = 0.0
        lifetime_dtod = 0.0

    return {
        'dw_min': DW_MIN,
        'gamma_up': GAMMA_UP,
        'gamma_down': GAMMA_DOWN,
        'lifetime': lifetime,
        'lifetime_dtod': lifetime_dtod,
        'write_noise_std': write_noise_std,
        'reset': 0.0,  # Decay toward 0V
    }


# Pre-configured settings for common use cases
SIXT1C_DEFAULT_PARAMS = {
    'dt_batch_sec': 1.0,
    'include_retention': True,
    'write_noise_std': 0.0,
}
