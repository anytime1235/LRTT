"""
Sixt1c (6T1C) configuration for LoRA layers.

This module provides RPU configuration for 6T1C (LinearStepDevice) analog hardware
with retention/lifetime parameters. Based on LRTT's PythonLRTTPreset.sixt1c_ab() configuration.

Key characteristics:
- LinearStepDevice with 6T1C physical parameters from LRTT (via tikitaka_config.py)
- Lifetime from TAU_SEC = 46505.0 seconds (775.1 minutes)
- Device-to-device and cycle-to-cycle variations included
- Uses UnitCellRPUConfig with ChoppedTransferCompound (2-tile) for memory efficiency

OOM 해결:
- 기존 SingleRPUConfig + AnalogTile은 backward pass에서 모든 intermediate states 저장 → OOM
- 2-tile ChoppedTransferCompound (transfer_every=0) 구조로 전환하여 메모리 효율 개선
- gamma=0: Slow tile만 forward에 사용
- transfer_every=0: transfer 완전 비활성화 (sixt1c와 동일 동작)
"""

import math
from aihwkit.simulator.configs import SingleRPUConfig, TorchInferenceRPUConfig, UnitCellRPUConfig
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.utils import WeightClipType, WeightRemapType
from aihwkit.simulator.parameters.enums import NoiseManagementType, BoundManagementType
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.training import UpdateParameters

# Reuse the exact same LinearStepDevice config from tikitaka_config.py
from tikitaka_config import create_sixt1c_device, create_softbounds_device_noise_free, SIXT1C_LIFETIME


# Re-export constants from tikitaka_config for backwards compatibility
TAU_SEC = SIXT1C_LIFETIME  # 46506.0 seconds (775.1 minutes)


def calculate_lifetime(dt_batch_sec: float, tau_sec: float = TAU_SEC) -> float:
    """
    Calculate the lifetime parameter for 6T1C retention decay.

    The lifetime parameter controls exponential weight decay during inference.
    For 6T1C devices, decay follows: w(t) = w(0) * exp(-t/tau)

    Args:
        dt_batch_sec: Time step between batches in seconds
        tau_sec: Time constant (default: 46506.0 seconds)

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
    lifetime: float = SIXT1C_LIFETIME,
) -> SingleRPUConfig:
    """
    Generate sixt1c RPU config for LoRA layers (legacy, memory-inefficient).
    Uses the exact same LinearStepDevice configuration as LRTT/TikiTaka.

    WARNING: This config uses SingleRPUConfig which causes OOM on large models.
    Use gen_sixt1c_2tile_config() instead for memory-efficient training.

    Args:
        lifetime: Retention lifetime (default: 46506.0, same as LRTT)

    Returns:
        SingleRPUConfig with LinearStepDevice for 6T1C training
    """
    # Use the exact same device config as tikitaka_config.py
    device = create_sixt1c_device(lifetime=lifetime)
    rpu_config = SingleRPUConfig(device=device)
    return rpu_config


def gen_sixt1c_2tile_config(
    lifetime: float = SIXT1C_LIFETIME,
    fast_lr: float = 1.0,
) -> UnitCellRPUConfig:
    """
    Generate sixt1c RPU config using 2-tile ChoppedTransferCompound structure.

    This is the memory-efficient version that solves OOM issues.
    Uses the same device configuration as TikiTaka v2, but with transfer disabled.

    Structure:
    - Fast tile: 6T1C LinearStepDevice - receives SGD updates
    - Slow tile: SoftBoundsDevice (noise=0) - used for forward pass
    - transfer_every=0: No transfer (sixt1c와 동일 동작)
    - gamma=0: Only Slow tile visible in forward

    Why this solves OOM:
    - SingleRPUConfig + AnalogTile stores all intermediate states for backward
    - 2-tile structure distributes computation across Fast/Slow tiles
    - ChoppedTransferCompound has more efficient memory management

    Args:
        lifetime: Retention lifetime for 6T1C device (default: 46506.0)
        fast_lr: Learning rate multiplier for Fast tile (default: 1.0)

    Returns:
        UnitCellRPUConfig with ChoppedTransferCompound for memory-efficient training
    """
    # Fast Tile: 6T1C LinearStepDevice (receives updates)
    fast_device = create_sixt1c_device(lifetime=lifetime)

    # Slow Tile: SoftBoundsDevice (noise=0, used for forward)
    slow_device = create_softbounds_device_noise_free()

    # Transfer IO parameters (minimal, since transfer is disabled)
    transfer_io = IOParameters(
        noise_management=NoiseManagementType.NONE,
        bound_management=BoundManagementType.NONE,
    )

    # Transfer update parameters
    transfer_update = UpdateParameters(
        desired_bl=1,
        update_bl_management=False,
        update_management=False,
    )

    # Create 2-tile config with no transfer (sixt1c behavior)
    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            # [Fast, Slow] - Fast receives updates, Slow is visible
            unit_cell_devices=[fast_device, slow_device],
            # Transfer disabled (sixt1c와 동일)
            transfer_every=0,  # 0 = completely disabled
            units_in_mbatch=False,  # mat-vec units
            n_reads_per_transfer=1,
            transfer_columns=True,
            # gamma=0: Only Slow tile visible in forward
            gamma=0.0,
            # Learning rates
            transfer_lr=1.0,  # Not used since transfer_every=0
            fast_lr=fast_lr,
            scale_transfer_lr=True,
            # Auto scale disabled (sixt1c uses fixed scale)
            auto_scale=False,
            auto_granularity=0.0,
            buffer_granularity=1.0,
            auto_momentum=0.0,
            # No chopped input (sixt1c behavior)
            in_chop_prob=0.0,
            in_chop_random=False,
            # Transfer IO and update settings
            transfer_forward=transfer_io,
            transfer_update=transfer_update,
        )
    )

    return rpu_config


def gen_softbounds_base_config() -> TorchInferenceRPUConfig:
    """
    Generate SoftBounds-like RPU config for base_layer (noise=0).
    Uses TorchInferenceRPUConfig for memory efficiency (inference-only).

    Base layer is frozen, so training capability is not needed.
    This saves significant GPU memory compared to SingleRPUConfig.

    Returns:
        TorchInferenceRPUConfig with no noise (SoftBounds-like behavior)
    """
    rpu_config = TorchInferenceRPUConfig()

    # Digital bias, standard mapping
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True

    # No noise for SoftBounds-like behavior
    rpu_config.modifier.std_dev = 0.0

    # No remap (preserve original weights)
    rpu_config.remap.type = WeightRemapType.NONE

    # No clipping
    rpu_config.clip.type = WeightClipType.NONE

    # No PCM noise model (clean inference)
    rpu_config.noise_model = None
    rpu_config.drift_compensation = None

    return rpu_config


# Pre-configured settings for common use cases
SIXT1C_DEFAULT_PARAMS = {
    'lifetime': SIXT1C_LIFETIME,
}
