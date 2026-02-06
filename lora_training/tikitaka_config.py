# -*- coding: utf-8 -*-
"""TikiTaka v2 configuration helper for GLUE tasks.

TikiTaka v2 uses a 2-tile structure with ChoppedTransferCompound:
- Fast tile (A): 6T1C LinearStepDevice - receives SGD updates
- Slow tile (C): SoftBoundsDevice (noise=0) - stable weight storage

Key differences from v1:
- ChoppedTransferCompound instead of TransferCompound
- units_in_mbatch=False (mat-vec units)
- auto_scale=True with auto_granularity
- in_chop_prob for chopped updates

Trial 17 Best Hyperparameters:
- learning_rate: 1.31e-04
- transfer_lr: 7.36
- transfer_every: 160
- fast_lr: 0.862
- auto_granularity: 305.91
- in_chop_prob: 0.020
"""

import sys
import os

# Add LRTT src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LRTT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(LRTT_ROOT, "src"))

from aihwkit.simulator.configs import UnitCellRPUConfig, SoftBoundsDevice
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.parameters.enums import (
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.training import UpdateParameters
from aihwkit.simulator.parameters.inference import DriftParameter


# Trial 17 Best Hyperparameters
TIKITAKA_V2_DEFAULT_PARAMS = {
    "learning_rate": 1.31e-04,
    "transfer_lr": 7.36,
    "transfer_every": 160,
    "fast_lr": 0.862,
    "auto_granularity": 305.91,
    "in_chop_prob": 0.020,
}

# 6T1C lifetime (dt_batch_sec=1.0 기준, lrtt_python.py sixt1c_ab와 동일)
SIXT1C_LIFETIME = 46506.0


def create_sixt1c_device(
    dw_min: float = 0.001981,
    gamma_up: float = -0.1678,
    gamma_down: float = 0.1410,
    lifetime: float = SIXT1C_LIFETIME,
) -> LinearStepDevice:
    """Create a 6T1C device configuration matching lrtt_python.py sixt1c_ab.

    6T1C Device Characteristics:
        - ~1000 conductance states per direction
        - Capacitor-based weight storage with exponential decay
        - Time constant τ ≈ 775 min (12.9 hours)

    Args:
        dw_min: Minimum weight update step
        gamma_up: Up-pulse nonlinearity (negative = decreases with weight)
        gamma_down: Down-pulse nonlinearity (positive = decreases with weight)
        lifetime: Retention lifetime (default: 46506, same as lrtt_python.py sixt1c_ab)

    Returns:
        Configured LinearStepDevice for 6T1C behavior
    """
    return LinearStepDevice(
        # Core update parameters (fitted from 6T1C data)
        dw_min=dw_min,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=gamma_up,
        gamma_down=gamma_down,
        mult_noise=True,
        # Device-to-device variation (same as lrtt_python.py sixt1c_ab)
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        # Cycle-to-cycle variation
        dw_min_std=0.3,
        write_noise_std=0.0,  # No write noise (different from lrtt_python.py)
        # LinearStepDevice specific
        mean_bound_reference=True,
        # Retention (capacitor leakage)
        lifetime=lifetime,
        lifetime_dtod=0.1,
        reset=0.0,
        reset_dtod=0.0,
    )


def create_softbounds_device_noise_free() -> SoftBoundsDevice:
    """Create a SoftBoundsDevice with all noise sources disabled.

    Returns:
        Configured SoftBoundsDevice with noise=0
    """
    return SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )


def create_softbounds_device_with_drift(
    nu: float = 0.05,
    nu_dtod: float = 0.02,
    t_0: float = 1.0,
) -> SoftBoundsDevice:
    """Create a SoftBoundsDevice with PCM-like drift for fair comparison.

    This device has the same noise-free characteristics as the standard
    SoftBoundsDevice, but with drift parameters enabled to simulate
    PCM-like conductance drift over time.

    Args:
        nu: Mean drift exponent (default: 0.05, typical PCM value)
        nu_dtod: Device-to-device variation in nu (default: 0.02)
        t_0: Reference time for drift (default: 1.0 second)

    Returns:
        Configured SoftBoundsDevice with PCM-like drift
    """
    return SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=True,
        drift=DriftParameter(nu=nu, nu_dtod=nu_dtod, t_0=t_0),
    )


def create_tikitaka_v2_config(
    transfer_every: int = 160,
    transfer_lr: float = 7.36,
    fast_lr: float = 0.862,
    auto_granularity: float = 305.91,
    in_chop_prob: float = 0.020,
    gamma: float = 0.0,
    auto_scale: bool = True,
    units_in_mbatch: bool = False,
    buffer_granularity: float = 1.0,
    auto_momentum: float = 0.99,
    in_chop_random: bool = True,
) -> UnitCellRPUConfig:
    """Create TikiTaka v2 config with 6T1C Fast + SoftBounds Slow.

    Uses ChoppedTransferCompound with Trial 17 optimized hyperparameters.

    TikiTaka v2 learning rule:
    - Fast tile (A): 6T1C LinearStepDevice receives SGD updates
    - Slow tile (C): SoftBoundsDevice stores accumulated weights (visible for forward)
    - Periodic chopped transfer: Fast -> Slow via one-hot column read/write
    - auto_scale automatically adjusts weight scale
    - in_chop_prob controls chopped update probability

    Args:
        transfer_every: Transfer frequency in mini-batches (default: 160)
        transfer_lr: Learning rate for transfer (default: 7.36)
        fast_lr: Learning rate multiplier for Fast tile updates (default: 0.862)
        auto_granularity: Granularity for auto scaling (default: 305.91)
        in_chop_prob: Probability for chopped input (default: 0.020)
        gamma: Weight mixing ratio (0 = only Slow visible, 1 = only Fast visible)
        auto_scale: Enable auto scaling (default: True)
        units_in_mbatch: Use mini-batch units (default: False for mat-vec)
        buffer_granularity: Buffer granularity (default: 1.0)
        auto_momentum: Momentum for auto scaling (default: 0.99)
        in_chop_random: Randomize chopped input (default: True)

    Returns:
        Configured UnitCellRPUConfig for TikiTaka v2 training
    """
    # Fast Tile (A): 6T1C LinearStepDevice
    fast_device = create_sixt1c_device()

    # Slow Tile (C): SoftBoundsDevice (noise=0)
    slow_device = create_softbounds_device_noise_free()

    # Transfer IO parameters
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

    # Create TikiTaka v2 config using ChoppedTransferCompound
    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            # [Fast, Slow] - Fast receives updates, Slow is visible
            unit_cell_devices=[fast_device, slow_device],
            # Transfer settings
            transfer_every=transfer_every,
            units_in_mbatch=units_in_mbatch,  # False = mat-vec units
            n_reads_per_transfer=1,
            transfer_columns=True,
            # gamma=0 means only Slow tile is visible in forward
            gamma=gamma,
            # Transfer learning rates
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=True,
            # Auto scale settings (key v2 feature)
            auto_scale=auto_scale,
            auto_granularity=auto_granularity,
            buffer_granularity=buffer_granularity,
            auto_momentum=auto_momentum,
            # Chopped input settings (key v2 feature)
            in_chop_prob=in_chop_prob,
            in_chop_random=in_chop_random,
            # Transfer IO and update settings
            transfer_forward=transfer_io,
            transfer_update=transfer_update,
        )
    )

    return rpu_config


def create_tikitaka_v2_config_with_drift(
    transfer_every: int = 160,
    transfer_lr: float = 7.36,
    fast_lr: float = 0.862,
    auto_granularity: float = 305.91,
    in_chop_prob: float = 0.020,
    gamma: float = 0.0,
    auto_scale: bool = True,
    units_in_mbatch: bool = False,
    buffer_granularity: float = 1.0,
    auto_momentum: float = 0.99,
    in_chop_random: bool = True,
    nu: float = 0.05,
    nu_dtod: float = 0.02,
    t_0: float = 1.0,
) -> UnitCellRPUConfig:
    """Create TikiTaka v2 config with drift-enabled Slow tile for fair comparison.

    This is the same as create_tikitaka_v2_config() but uses a SoftBoundsDevice
    with PCM-like drift parameters for the Slow tile, enabling drift evaluation
    comparable to PCM/sixt1c experiments.

    Args:
        transfer_every: Transfer frequency in mini-batches (default: 160)
        transfer_lr: Learning rate for transfer (default: 7.36)
        fast_lr: Learning rate multiplier for Fast tile updates (default: 0.862)
        auto_granularity: Granularity for auto scaling (default: 305.91)
        in_chop_prob: Probability for chopped input (default: 0.020)
        gamma: Weight mixing ratio (0 = only Slow visible, 1 = only Fast visible)
        auto_scale: Enable auto scaling (default: True)
        units_in_mbatch: Use mini-batch units (default: False for mat-vec)
        buffer_granularity: Buffer granularity (default: 1.0)
        auto_momentum: Momentum for auto scaling (default: 0.99)
        in_chop_random: Randomize chopped input (default: True)
        nu: Mean drift exponent for Slow tile (default: 0.05)
        nu_dtod: Device-to-device variation in nu (default: 0.02)
        t_0: Reference time for drift (default: 1.0 second)

    Returns:
        Configured UnitCellRPUConfig for TikiTaka v2 training with drift support
    """
    # Fast Tile (A): 6T1C LinearStepDevice (same as standard)
    fast_device = create_sixt1c_device()

    # Slow Tile (C): SoftBoundsDevice with PCM-like drift
    slow_device = create_softbounds_device_with_drift(nu=nu, nu_dtod=nu_dtod, t_0=t_0)

    # Transfer IO parameters
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

    # Create TikiTaka v2 config using ChoppedTransferCompound
    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            # [Fast, Slow] - Fast receives updates, Slow is visible
            unit_cell_devices=[fast_device, slow_device],
            # Transfer settings
            transfer_every=transfer_every,
            units_in_mbatch=units_in_mbatch,  # False = mat-vec units
            n_reads_per_transfer=1,
            transfer_columns=True,
            # gamma=0 means only Slow tile is visible in forward
            gamma=gamma,
            # Transfer learning rates
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=True,
            # Auto scale settings (key v2 feature)
            auto_scale=auto_scale,
            auto_granularity=auto_granularity,
            buffer_granularity=buffer_granularity,
            auto_momentum=auto_momentum,
            # Chopped input settings (key v2 feature)
            in_chop_prob=in_chop_prob,
            in_chop_random=in_chop_random,
            # Transfer IO and update settings
            transfer_forward=transfer_io,
            transfer_update=transfer_update,
        )
    )

    return rpu_config


def get_tikitaka_v2_glue_config(
    transfer_every: int = None,
    transfer_lr: float = None,
    fast_lr: float = None,
    auto_granularity: float = None,
    in_chop_prob: float = None,
) -> UnitCellRPUConfig:
    """Get TikiTaka v2 configuration for GLUE tasks with Trial 17 defaults.

    Args:
        transfer_every: Override default transfer_every (160)
        transfer_lr: Override default transfer_lr (7.36)
        fast_lr: Override default fast_lr (0.862)
        auto_granularity: Override default auto_granularity (305.91)
        in_chop_prob: Override default in_chop_prob (0.020)

    Returns:
        Configured UnitCellRPUConfig for TikiTaka v2
    """
    params = TIKITAKA_V2_DEFAULT_PARAMS.copy()

    if transfer_every is not None:
        params["transfer_every"] = transfer_every
    if transfer_lr is not None:
        params["transfer_lr"] = transfer_lr
    if fast_lr is not None:
        params["fast_lr"] = fast_lr
    if auto_granularity is not None:
        params["auto_granularity"] = auto_granularity
    if in_chop_prob is not None:
        params["in_chop_prob"] = in_chop_prob

    return create_tikitaka_v2_config(
        transfer_every=params["transfer_every"],
        transfer_lr=params["transfer_lr"],
        fast_lr=params["fast_lr"],
        auto_granularity=params["auto_granularity"],
        in_chop_prob=params["in_chop_prob"],
    )


if __name__ == "__main__":
    # Test configuration creation
    print("Testing TikiTaka v2 configuration...")

    print("\nDefault parameters (Trial 17 Best):")
    for key, value in TIKITAKA_V2_DEFAULT_PARAMS.items():
        print(f"  {key}: {value}")

    print("\nTile Configuration:")
    print("  Fast tile (A): 6T1C LinearStepDevice (noise=0)")
    print("  Slow tile (C): SoftBoundsDevice (noise=0)")

    config = create_tikitaka_v2_config()
    print("\nTikiTaka v2 config:")
    print(config)

    glue_config = get_tikitaka_v2_glue_config()
    print("\nTikiTaka v2 GLUE config (default params):")
    print(glue_config)

    print("\nConfiguration test passed!")
