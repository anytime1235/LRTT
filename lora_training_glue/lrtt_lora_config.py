"""
LRTT-LoRA Configuration - Unified LRTT Tile for LoRA Training

This module creates LRTT configurations that replicate PEFT LoRA behavior
using the unified LRTTSimulatorTile (3 sub-tiles: A, B, C).

Key Design:
- forward_inject=True: Enables LoRA composition y = C·x + α·A·(B·x)
- transfer disabled: A·B remain separate (no collapse to C)
- A/B tiles: 6T1C LinearStepDevice (trainable)
- C tile: SoftBoundsDevice (frozen, holds pretrained weights)

Architecture:
    LRTTSimulatorTile (single unified tile):
    ├─ tile_a: LinearStepDevice 6T1C [d_size, rank] - TRAINABLE
    ├─ tile_b: LinearStepDevice 6T1C [rank, x_size] - TRAINABLE
    └─ tile_c: SoftBoundsDevice [d_size, x_size]    - FROZEN

Forward: y = C·x + α·A·(B·x) (forward_inject=True)
Gradient: LRTTController's native backward hook
"""

from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs.utils import (
    NoiseManagementType,
    BoundManagementType,
)

# ============================================================================
# 6T1C Physical Parameters (from sixt1c_config.py)
# ============================================================================
DW_MIN = 0.001981
GAMMA_UP = -0.1678
GAMMA_DOWN = 0.1410


def create_lrtt_lora_config(
    rank: int = 8,
    lora_alpha: float = 1.0,
    output_noise_level: float = 0.0,
    use_floating_point: bool = False,
) -> PythonLRTTRPUConfig:
    """
    Create LRTT configuration that replicates PEFT LoRA behavior.

    This function creates a unified LRTT tile with:
    - forward_inject=True: Enables y = C·x + α·A·(B·x)
    - transfer disabled: A/B remain separate
    - A/B tiles: 6T1C (or FloatingPoint if use_floating_point=True) trainable
    - C tile: SoftBounds frozen (pretrained weights)

    Settings match mobilebert_squad_lrtt_scratch.py except:
    - forward_inject=True (vs False in reference)
    - transfer disabled (vs periodic transfer in reference)

    Args:
        rank: LoRA rank dimension (default: 8)
        lora_alpha: LoRA scaling factor α (default: 1.0)
        output_noise_level: Output noise (default: 0.0 for deterministic)
        use_floating_point: Use FloatingPoint for A/B tiles instead of 6T1C (default: False)
            When True, this creates FP-LoRA mode for exact arithmetic (for testing)

    Returns:
        PythonLRTTRPUConfig configured for LRTT-LoRA training

    Example:
        >>> # Standard 6T1C mode
        >>> config = create_lrtt_lora_config(rank=8, lora_alpha=1.0)
        >>> # FP-LoRA mode (exact arithmetic, for testing)
        >>> config = create_lrtt_lora_config(rank=8, lora_alpha=1.0, use_floating_point=True)
    """
    # ========================================================================
    # A/B Tiles: 6T1C LinearStepDevice (or FloatingPoint for FP-LoRA mode)
    # ========================================================================
    if use_floating_point:
        # FP-LoRA mode: FloatingPoint device for exact arithmetic (testing)
        # Note: Weight clipping is handled in FPLoRATrainer to prevent explosion
        from aihwkit.simulator.configs.devices import FloatingPointDevice
        ab_device = FloatingPointDevice()
    else:
        # Standard mode: 6T1C LinearStepDevice (trainable)
        ab_device = LinearStepDevice(
            # Core 6T1C parameters
            dw_min=DW_MIN,
            w_max=1.0,
            w_min=-1.0,
            up_down=0.0,
            gamma_up=GAMMA_UP,
            gamma_down=GAMMA_DOWN,
            mult_noise=True,  # LRTT_setup.txt (FIXED)

            # Device-to-device variation (LRTT_setup.txt values)
            dw_min_dtod=0.1,
            up_down_dtod=0.01,
            w_max_dtod=0.05,
            w_min_dtod=0.05,
            gamma_up_dtod=0.05,
            gamma_down_dtod=0.05,

            # Cycle-to-cycle variation (LRTT_setup.txt values)
            dw_min_std=0.3,
            write_noise_std=0.0,

            # LinearStepDevice specific
            mean_bound_reference=True,

            # No retention during training
            lifetime=0.0,
            lifetime_dtod=0.0,
            reset=0.0,
            reset_dtod=0.0,
        )

    # ========================================================================
    # C Tile: SoftBoundsDevice (frozen, holds pretrained weights)
    # ========================================================================
    c_device = SoftBoundsDevice(
        # Weight bounds
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        up_down=0.0,
        mult_noise=False,  # Deterministic (FIXED)

        # Device-to-device variation (disabled)
        dw_min_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        up_down_dtod=0.0,

        # Cycle-to-cycle variation (disabled)
        dw_min_std=0.0,
        write_noise_std=0.0,

        # No diffusion or retention
        diffusion=0.0,
        lifetime=0.0,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )

    # ========================================================================
    # LRTT Device Configuration
    # ========================================================================
    lrtt_device = PythonLRTTDevice(
        # === Core LoRA Parameters ===
        rank=rank,
        lora_alpha=lora_alpha,

        # === Forward Composition (KEY for LoRA) ===
        forward_inject=True,  # Enable y = C·x + α·A·(B·x)

        # === Transfer Control (DISABLE transfer) ===
        transfer_every=10000000,  # Effectively disabled
        transfer_mode="off",      # No calibration
        transfer_lr=1.0,          # Must be positive (but transfer won't happen)
        transfer_lr_scale=1.0,

        # === Initialization ===
        a_init_mode="zero",       # A=0 (LoRA-style, ΔW=0 initially)
        b_init_mode="kaiming",    # B~Kaiming (random initialization)

        # === Reinit (not used since transfer disabled, but set for consistency) ===
        reinit_mode="decay",      # Decay mode
        decay_factor=1.0,         # No decay (keep A/B weights)
        reinit_gain=0.1,

        # === Update Mode ===
        update_mode="lora",       # LoRA chain rule (requires forward_inject=True)

        # === Noise Settings ===
        correct_gradient_magnitudes=False,  # No gradient correction

        # === Device Tiles ===
        unit_cell_devices=[ab_device, ab_device, c_device],
    )

    # ========================================================================
    # RPU Configuration (Mapping, Forward IO)
    # ========================================================================
    rpu_config = PythonLRTTRPUConfig(device=lrtt_device)

    # Mapping (FIXED - must match sixt1c_config.py)
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    # Forward IO (FIXED - LRTT_setup.txt values)
    rpu_config.forward.inp_res = 1 / (2**7 - 2)  # 7-bit (LRTT_setup.txt)
    rpu_config.forward.out_res = 1 / (2**9 - 2)  # 9-bit (LRTT_setup.txt)
    rpu_config.forward.out_noise = 0.0  # Always 0 for deterministic (matches reference)
    rpu_config.forward.is_perfect = False
    rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX
    rpu_config.forward.bound_management = BoundManagementType.ITERATIVE

    # Backward IO (FIXED - matches Forward IO per LRTT_setup.txt line 24)
    rpu_config.backward.inp_res = 1 / (2**7 - 2)  # 7-bit (same as forward)
    rpu_config.backward.out_res = 1 / (2**9 - 2)  # 9-bit (same as forward)
    rpu_config.backward.out_noise = 0.0  # Always 0 for deterministic
    rpu_config.backward.noise_management = NoiseManagementType.ABS_MAX
    rpu_config.backward.bound_management = BoundManagementType.ITERATIVE

    # Update (FIXED - LRTT_setup.txt lines 33-37)
    rpu_config.update.desired_bl = 31
    rpu_config.update.update_bl_management = True
    rpu_config.update.update_management = True

    return rpu_config


def print_lrtt_lora_config_summary(config: PythonLRTTRPUConfig):
    """Print LRTT-LoRA configuration summary."""
    device = config.device

    print("=" * 80)
    print("LRTT-LORA CONFIGURATION")
    print("=" * 80)
    print("\n[ARCHITECTURE]")
    print("  Unified LRTTSimulatorTile (3 sub-tiles):")
    print("    tile_a: 6T1C LinearStepDevice (trainable)")
    print("    tile_b: 6T1C LinearStepDevice (trainable)")
    print("    tile_c: SoftBoundsDevice (frozen)")
    print("\n[LORA PARAMETERS]")
    print(f"  rank: {device.rank}")
    print(f"  lora_alpha: {device.lora_alpha}")
    print(f"  forward_inject: {device.forward_inject} (enables LoRA composition)")
    print("\n[TRANSFER CONTROL]")
    print(f"  transfer_every: {device.transfer_every} (effectively disabled)")
    print(f"  transfer_mode: {device.transfer_mode}")
    print(f"  transfer_lr: {device.transfer_lr}")
    print("\n[INITIALIZATION]")
    print(f"  a_init_mode: {device.a_init_mode} (A=0, ΔW=0 initially)")
    print(f"  b_init_mode: {device.b_init_mode} (B~Kaiming)")
    print("\n[UPDATE MODE]")
    print(f"  update_mode: {device.update_mode} (LoRA chain rule)")
    print("\n[MAPPING - FIXED]")
    print(f"  weight_scaling_omega: {config.mapping.weight_scaling_omega}")
    print(f"  learn_out_scaling: {config.mapping.learn_out_scaling}")
    print(f"  out_scaling_columnwise: {config.mapping.out_scaling_columnwise}")
    print("\n[FORWARD IO - FIXED]")
    print(f"  noise_management: {config.forward.noise_management}")
    print(f"  bound_management: {config.forward.bound_management}")
    print(f"  out_noise: {config.forward.out_noise}")
    print("=" * 80)


if __name__ == "__main__":
    # Test configuration creation
    print("Testing LRTT-LoRA configuration...\n")

    config = create_lrtt_lora_config(rank=8, lora_alpha=1.0, output_noise_level=0.0)
    print_lrtt_lora_config_summary(config)

    print("\n✓ LRTT-LoRA config created successfully")
    print("\nForward equation: y = C·x + α·A·(B·x)")
    print("  C: Frozen pretrained weights")
    print("  A, B: Trainable low-rank adapters")
    print("  α: LoRA scaling factor")
