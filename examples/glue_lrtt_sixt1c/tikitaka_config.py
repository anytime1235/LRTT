# -*- coding: utf-8 -*-
"""TikiTaka configuration helper for GLUE tasks.

TikiTaka uses a 2-tile structure:
- Fast tile (A): 6T1C device - receives SGD updates
- Slow tile (C): SoftBounds device - stable weight storage

Forward: y = gamma*A @ x + (1-gamma)*C @ x
Transfer: One-hot vector transfer from Fast -> Slow periodically
"""

import sys
import os

# Add LRTT src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LRTT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(LRTT_ROOT, "src"))

from aihwkit.simulator.configs import UnitCellRPUConfig
from aihwkit.simulator.configs.compounds import TransferCompound
from aihwkit.simulator.configs.devices import (
    LinearStepDevice,
    SoftBoundsReferenceDevice,
)
from aihwkit.simulator.parameters.enums import (
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.simulator.parameters.training import UpdateParameters


def create_sixt1c_device(
    dw_min: float = 0.001981,
    gamma_up: float = -0.1678,
    gamma_down: float = 0.1410,
    dw_min_dtod: float = 0.1,
    dw_min_std: float = 0.3,
    write_noise_std: float = 0.0182,
    lifetime: float = 0.0,
) -> LinearStepDevice:
    """Create a 6T1C device configuration.

    Args:
        dw_min: Minimum weight update step
        gamma_up: Up-pulse nonlinearity (negative = decreases with weight)
        gamma_down: Down-pulse nonlinearity (positive = decreases with weight)
        dw_min_dtod: Device-to-device variation of dw_min
        dw_min_std: Cycle-to-cycle variation of dw_min
        write_noise_std: Write noise standard deviation
        lifetime: Retention lifetime (0 = no decay)

    Returns:
        Configured LinearStepDevice for 6T1C behavior
    """
    return LinearStepDevice(
        dw_min=dw_min,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        mult_noise=False,
        gamma_up=gamma_up,
        gamma_down=gamma_down,
        dw_min_dtod=dw_min_dtod,
        up_down_dtod=0.01,
        w_max_dtod=0.1,
        w_min_dtod=0.1,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=dw_min_std,
        write_noise_std=write_noise_std,
        mean_bound_reference=True,
        lifetime=lifetime,
        lifetime_dtod=0.3,
    )


def create_softbounds_slow_device(
    dw_min: float = 0.001,
    noise_free: bool = True,
) -> SoftBoundsReferenceDevice:
    """Create a SoftBounds device for Slow tile (stable weight storage).

    Args:
        dw_min: Minimum weight update step
        noise_free: If True, disable all noise sources

    Returns:
        Configured SoftBoundsReferenceDevice
    """
    if noise_free:
        return SoftBoundsReferenceDevice(
            dw_min=dw_min,
            up_down=0.0,
            w_max=1.0,
            w_min=-1.0,
            mult_noise=False,
            dw_min_dtod=0.0,
            up_down_dtod=0.0,
            w_max_dtod=0.0,
            w_min_dtod=0.0,
            dw_min_std=0.0,
            write_noise_std=0.0,
            diffusion=0.0,
            lifetime=0.0,
        )
    else:
        return SoftBoundsReferenceDevice(
            dw_min=dw_min,
            up_down=0.0,
            w_max=1.0,
            w_min=-1.0,
        )


def tikitaka_sixt1c_softbounds_config(
    transfer_every: int = 32,
    transfer_lr: float = 1.0,
    fast_lr: float = 1.0,
    gamma: float = 0.0,
    n_reads_per_transfer: int = 1,
    # 6T1C Fast tile parameters
    sixt1c_dw_min: float = 0.001981,
    sixt1c_gamma_up: float = -0.1678,
    sixt1c_gamma_down: float = 0.1410,
    sixt1c_dw_min_std: float = 0.3,
    sixt1c_write_noise_std: float = 0.0182,
    sixt1c_lifetime: float = 0.0,
    # SoftBounds Slow tile parameters
    slow_dw_min: float = 0.001,
    slow_noise_free: bool = True,
    # IO parameters
    inp_res: float = 0.0,  # 0 = infinite resolution
    out_res: float = 0.0,
) -> UnitCellRPUConfig:
    """Create TikiTaka config with 6T1C Fast + SoftBounds Slow tiles.

    TikiTaka learning rule:
    - Fast tile receives SGD updates
    - Slow tile stores accumulated weights (visible for forward)
    - Periodic transfer: Fast -> Slow via one-hot column read/write

    Args:
        transfer_every: Transfer frequency in mini-batches
        transfer_lr: Learning rate for transfer (relative to SGD LR)
        fast_lr: Learning rate multiplier for Fast tile updates
        gamma: Weight mixing ratio (0 = only Slow visible, 1 = only Fast visible)
        n_reads_per_transfer: Number of columns transferred per transfer event

        sixt1c_*: Parameters for 6T1C Fast tile
        slow_*: Parameters for SoftBounds Slow tile

        inp_res: Input resolution (0 = infinite)
        out_res: Output resolution (0 = infinite)

    Returns:
        Configured UnitCellRPUConfig for TikiTaka training
    """
    # Create Fast tile (6T1C)
    fast_device = create_sixt1c_device(
        dw_min=sixt1c_dw_min,
        gamma_up=sixt1c_gamma_up,
        gamma_down=sixt1c_gamma_down,
        dw_min_std=sixt1c_dw_min_std,
        write_noise_std=sixt1c_write_noise_std,
        lifetime=sixt1c_lifetime,
    )

    # Create Slow tile (SoftBounds)
    slow_device = create_softbounds_slow_device(
        dw_min=slow_dw_min,
        noise_free=slow_noise_free,
    )

    # IO parameters for transfer
    transfer_io = IOParameters(
        noise_management=NoiseManagementType.NONE,
        bound_management=BoundManagementType.NONE,
    )

    # Create TikiTaka config using TransferCompound
    rpu_config = UnitCellRPUConfig(
        device=TransferCompound(
            # [Fast, Slow] - Fast receives updates, Slow is visible
            unit_cell_devices=[fast_device, slow_device],
            # Transfer settings
            transfer_every=transfer_every,
            units_in_mbatch=True,
            n_reads_per_transfer=n_reads_per_transfer,
            # gamma=0 means only Slow tile is visible in forward
            gamma=gamma,
            # Transfer learning rates
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=True,
            # Use column-wise transfer (standard TikiTaka)
            transfer_columns=True,
            # Transfer IO settings
            transfer_forward=transfer_io,
        )
    )

    # Set IO resolution if specified
    if inp_res > 0:
        rpu_config.forward.inp_res = inp_res
        rpu_config.backward.inp_res = inp_res
    if out_res > 0:
        rpu_config.forward.out_res = out_res
        rpu_config.backward.out_res = out_res

    return rpu_config


def tikitaka_idealized_config(
    transfer_every: int = 32,
    transfer_lr: float = 1.0,
    fast_lr: float = 1.0,
    gamma: float = 0.0,
) -> UnitCellRPUConfig:
    """Create idealized TikiTaka config (no noise, no nonlinearity).

    Useful for baseline comparisons.

    Args:
        transfer_every: Transfer frequency in mini-batches
        transfer_lr: Learning rate for transfer
        fast_lr: Learning rate multiplier for Fast tile
        gamma: Weight mixing ratio

    Returns:
        Configured UnitCellRPUConfig for idealized TikiTaka
    """
    # Idealized Fast tile (linear, no noise)
    fast_device = LinearStepDevice(
        dw_min=0.001,
        gamma_up=0.0,  # Linear
        gamma_down=0.0,  # Linear
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
    )

    # Idealized Slow tile (no noise)
    slow_device = create_softbounds_slow_device(
        dw_min=0.001,
        noise_free=True,
    )

    transfer_io = IOParameters(
        noise_management=NoiseManagementType.NONE,
        bound_management=BoundManagementType.NONE,
    )

    return UnitCellRPUConfig(
        device=TransferCompound(
            unit_cell_devices=[fast_device, slow_device],
            transfer_every=transfer_every,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            gamma=gamma,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=True,
            transfer_columns=True,
            transfer_forward=transfer_io,
        )
    )


# Convenience function for GLUE experiments
def get_tikitaka_glue_config(
    transfer_every: int = 32,
    fast_lr: float = 1.0,
    use_ideal: bool = False,
) -> UnitCellRPUConfig:
    """Get TikiTaka configuration for GLUE tasks.

    Args:
        transfer_every: Transfer frequency (32 is typical for TikiTaka)
        fast_lr: Fast tile learning rate multiplier
        use_ideal: If True, use idealized (noise-free) config

    Returns:
        Configured UnitCellRPUConfig
    """
    if use_ideal:
        return tikitaka_idealized_config(
            transfer_every=transfer_every,
            fast_lr=fast_lr,
        )
    else:
        return tikitaka_sixt1c_softbounds_config(
            transfer_every=transfer_every,
            fast_lr=fast_lr,
        )


if __name__ == "__main__":
    # Test configuration creation
    print("Testing TikiTaka configuration...")

    config = tikitaka_sixt1c_softbounds_config()
    print("\n6T1C + SoftBounds TikiTaka config:")
    print(config)

    ideal_config = tikitaka_idealized_config()
    print("\nIdealized TikiTaka config:")
    print(ideal_config)

    print("\nConfiguration test passed!")
