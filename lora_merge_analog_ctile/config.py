"""Configuration for LoRA Merge with Analog C Tile.

This module provides configuration classes and factory functions for creating
LRTT configurations with:
- Digital A, B tiles (FloatingPointDevice)
- Analog C tile (SoftBoundsDevice with noise=0)
- One-hot based transfer method

The key idea is:
1. A, B tiles use FloatingPointDevice for exact gradient computation
2. C tile uses SoftBoundsDevice (analog) with no noise for weight storage
3. Forward: y = C(x) (forward_inject=False, original LRTT style)
4. Periodic transfer: C += transfer_lr * (A @ B) using one-hot method
5. After transfer: A=0, B=Kaiming (standard reinit)
"""

from aihwkit.simulator.configs.devices import FloatingPointDevice, SoftBoundsDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


# SoftBounds configuration with no noise (same as sweep_softbounds_lifetime.py)
SOFTBOUNDS_NO_NOISE_CONFIG = {
    'dw_min': 0.001,
    'w_max': 1.0,
    'w_min': -1.0,
    'dw_min_dtod': 0.0,
    'dw_min_std': 0.0,
    'up_down': 0.0,
    'up_down_dtod': 0.0,
    'w_max_dtod': 0.0,
    'w_min_dtod': 0.0,
    'write_noise_std': 0.0,
    'mult_noise': True,
}


def create_lora_merge_config(
    rank: int,
    transfer_every: int,
    transfer_lr: float,
    lora_alpha: float = 1.0,
    reinit_gain: float = 0.1,
    reinit_mode: str = "standard",
) -> PythonLRTTRPUConfig:
    """Create LRTT config with Digital A,B and Analog C.

    Architecture:
    - A tile [d_size, rank]: FloatingPointDevice (digital, exact computation)
    - B tile [rank, x_size]: FloatingPointDevice (digital, exact computation)
    - C tile [d_size, x_size]: SoftBoundsDevice (analog, noise=0)

    Training flow:
    1. Forward: y = C(x) (forward_inject=False, C tile only)
    2. Backward: gradients computed through C
    3. A,B Update: LoRA chain rule projection (digital, exact)
       - XB = B @ x (FloatingPoint - exact)
       - DA = A.T @ d (FloatingPoint - exact)
       - A -= lr * alpha * d.T @ XB (exact update)
       - B -= lr * alpha * DA.T @ x (exact update)
    4. Transfer (every transfer_every steps):
       - C += transfer_lr * (A @ B) using one-hot method
       - C_new = clamp(C, -1, 1) (analog bounds)
    5. Reinit: A=0, B=Kaiming (standard) or decay (configurable)

    Args:
        rank: LoRA rank dimension (r)
        transfer_every: Transfer frequency (steps)
        transfer_lr: Transfer learning rate for A @ B -> C
        lora_alpha: LoRA scaling factor (default 1.0)
        reinit_gain: Kaiming initialization gain for B (default 0.1)
        reinit_mode: Reinit strategy after transfer:
                     - "standard": A=0, B=Kaiming (original LRTT)
                     - "decay": A*=factor, B*=factor (gradual decay)
                     - "hybrid": A=0, B*=factor

    Returns:
        PythonLRTTRPUConfig configured for digital A,B and analog C
    """
    # A, B tiles: FloatingPointDevice (digital, exact computation)
    ab_device = FloatingPointDevice()

    # C tile: SoftBoundsDevice (analog, noise=0)
    c_device = SoftBoundsDevice(**SOFTBOUNDS_NO_NOISE_CONFIG)

    # Create PythonLRTTDevice with custom device configuration
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        lora_alpha=lora_alpha,
        reinit_gain=reinit_gain,
        reinit_mode=reinit_mode,
        # Device configuration: [A, B, C]
        unit_cell_devices=[ab_device, ab_device, c_device],
        # Transfer method: "onehot" for one-hot vector based transfer
        transfer_method="onehot",
        # Transfer mode: "off" (no calibration)
        transfer_mode="off",
        # Forward inject enabled: y = C(x) + α * A(B(x)) (ReLoRA style)
        forward_inject=True,
        # Update mode: LoRA chain rule
        update_mode="lora",
    )

    return PythonLRTTRPUConfig(device=device_config)


def create_lora_merge_config_decay(
    rank: int,
    transfer_every: int,
    transfer_lr: float,
    lora_alpha: float = 1.0,
    decay_factor: float = 0.9,
) -> PythonLRTTRPUConfig:
    """Create LRTT config with decay mode for gradual A,B weight decay.

    Similar to create_lora_merge_config but uses "decay" reinit mode
    where A and B are decayed by decay_factor instead of being reset.

    Args:
        rank: LoRA rank dimension
        transfer_every: Transfer frequency (steps)
        transfer_lr: Transfer learning rate
        lora_alpha: LoRA scaling factor (default 1.0)
        decay_factor: Factor to multiply A,B by after transfer (default 0.9)

    Returns:
        PythonLRTTRPUConfig with decay mode
    """
    ab_device = FloatingPointDevice()
    c_device = SoftBoundsDevice(**SOFTBOUNDS_NO_NOISE_CONFIG)

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        transfer_lr=transfer_lr,
        lora_alpha=lora_alpha,
        reinit_mode="decay",
        decay_factor=decay_factor,
        unit_cell_devices=[ab_device, ab_device, c_device],
        transfer_method="onehot",
        transfer_mode="off",
        forward_inject=True,
        update_mode="lora",
    )

    return PythonLRTTRPUConfig(device=device_config)
