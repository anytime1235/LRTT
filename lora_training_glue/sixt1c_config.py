"""
Sixt1c (6T1C) configuration for LoRA layers.

This module provides RPU configuration for 6T1C (LinearStepDevice) analog hardware
with retention/lifetime parameters. Based on LRTT's PythonLRTTPreset.sixt1c_ab() configuration.

Key characteristics:
- LinearStepDevice with 6T1C physical parameters from LRTT
- Lifetime from TAU_SEC = 46505.0 seconds (775.1 minutes)
- Device-to-device and cycle-to-cycle variations included
- Retention is applied via apply_sixt1c_lora_retention() function
"""

import math
from aihwkit.simulator.configs import TorchInferenceRPUConfig
from aihwkit.simulator.presets.utils import IOParameters
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    WeightClipType,
    WeightRemapType,
)


# 6T1C Physical Parameters (from LRTT)
TAU_SEC = 46505.0  # Time constant in seconds (775.1 minutes)
DW_MIN = 0.001981  # Minimum weight change
GAMMA_UP = -0.1678  # Nonlinearity for positive updates
GAMMA_DOWN = 0.1410  # Nonlinearity for negative updates


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


def gen_sixt1c_lora_config(
    dt_batch_sec: float = 1.0,
    include_retention: bool = True,
    output_noise_level: float = 0.0,
):
    """
    Generate sixt1c RPU config for LoRA layers.
    Based on LRTT's PythonLRTTPreset.sixt1c_ab() configuration.

    Note: TorchInferenceRPUConfig doesn't directly support LinearStepDevice.
    The 6T1C retention effect is applied via apply_sixt1c_lora_retention() function
    during evaluation, which uses the TAU_SEC time constant.

    Args:
        dt_batch_sec: Time step between batches in seconds (default: 1.0)
        include_retention: Whether to include retention decay (default: True)
        output_noise_level: Output noise level for forward pass (default: 0.0)

    Returns:
        TorchInferenceRPUConfig configured for 6T1C characteristics
    """
    rpu_config = TorchInferenceRPUConfig()

    # Mapping configuration — disable weight scaling/remap for LoRA layers.
    # CHANNELWISE_SYMMETRIC remap normalizes weights to [-1, 1] per channel,
    # which causes NaN for lora_B (zero-initialized: max=0 → division by zero).
    # weight_scaling_omega/columnwise similarly distorts small LoRA weights.
    # These settings are designed for PCM base_layer, not LoRA adapters.
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 0.0  # Disable weight scaling
    rpu_config.mapping.weight_scaling_columnwise = False
    rpu_config.mapping.learn_out_scaling = False
    rpu_config.mapping.out_scaling_columnwise = False

    # Weight modifier configuration - no additional noise for sixt1c
    # (device variations are captured in apply_sixt1c_lora_retention)
    rpu_config.modifier.std_dev = 0.0
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL

    # No remap for LoRA layers — prevents NaN from zero-initialized lora_B
    rpu_config.remap.type = WeightRemapType.NONE

    # Forward pass configuration — keep 8-bit quantization for hardware realism
    rpu_config.forward = IOParameters()
    rpu_config.forward.out_noise = output_noise_level
    rpu_config.forward.is_perfect = False
    rpu_config.forward.inp_res = 1 / (2**8 - 2)  # 8-bit input resolution
    rpu_config.forward.out_res = 1 / (2**8 - 2)  # 8-bit output resolution

    # No clipping for LoRA layers — LAYER_GAUSSIAN clipping is designed for
    # PCM base_layer hardware constraints, not for LoRA adapters.
    # LoRA lora_B is initialized to zeros (std=0), so LAYER_GAUSSIAN would
    # clip all weights to 0 and prevent any learning.
    rpu_config.clip.type = WeightClipType.NONE

    # No PCM noise model for sixt1c (retention handled by apply_sixt1c_lora_retention)
    rpu_config.noise_model = None
    rpu_config.drift_compensation = None

    return rpu_config


def get_sixt1c_device_params(
    include_retention: bool = True,
    dt_batch_sec: float = 1.0,
):
    """
    Get 6T1C device parameters for LinearStepDevice configuration.
    These parameters are used by apply_sixt1c_lora_retention() function.

    Args:
        include_retention: Whether to include retention decay
        dt_batch_sec: Time step between batches in seconds

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
        'write_noise_std': 0.0,
        'reset': 0.0,
    }


# Pre-configured settings for common use cases
SIXT1C_DEFAULT_PARAMS = {
    'dt_batch_sec': 1.0,
    'include_retention': True,
}
