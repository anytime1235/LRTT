# -*- coding: utf-8 -*-
"""Test 6T1C retention decay in LRTT tiles.

This script tests whether the 6T1C device's retention (natural decay)
is being applied correctly in the LRTT implementation.

6T1C characteristics:
- Time constant τ = 775.1 min = 46505 sec
- Decay toward 0 (reset=0)
- For dt_batch=1sec: lifetime = 46506 (very slow decay)
"""

import math
import torch
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.tiles.analog import AnalogTile
from aihwkit.simulator.configs import SingleRPUConfig

# Check CUDA
from aihwkit.simulator.rpu_base import cuda
USE_CUDA = cuda.is_compiled()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# 6T1C parameters
TAU_SEC = 46505.0  # Physical time constant: 775.1 min
DT_BATCH_SEC = 1.0


def calculate_lifetime(dt_batch_sec):
    """Calculate AIHWKit lifetime from dt_batch."""
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    return 1.0 / delta


def create_6t1c_device(dt_batch_sec=1.0, include_retention=True):
    """Create 6T1C LinearStepDevice."""
    if include_retention and dt_batch_sec > 0:
        lifetime = calculate_lifetime(dt_batch_sec)
    else:
        lifetime = 0.0  # No retention

    return LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=False,  # Disable noise for cleaner test
        dw_min_dtod=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        gamma_up_dtod=0.0,
        gamma_down_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lifetime,
        lifetime_dtod=0.0,
        reset=0.0,  # Decay toward 0
        reset_dtod=0.0,
    )


def test_retention_decay():
    """Test 6T1C retention decay over time."""
    print("\n" + "="*60)
    print("Testing 6T1C Retention Decay")
    print("="*60)

    # Calculate expected decay
    lifetime = calculate_lifetime(DT_BATCH_SEC)
    expected_decay_per_step = 1.0 / lifetime  # decay_rate per forward call

    print(f"\n6T1C Parameters:")
    print(f"  Physical τ: {TAU_SEC} sec ({TAU_SEC/60:.1f} min)")
    print(f"  dt_batch: {DT_BATCH_SEC} sec")
    print(f"  AIHWKit lifetime: {lifetime:.1f}")
    print(f"  Expected decay per step: {expected_decay_per_step:.6f}")
    print(f"  Expected decay per 1000 steps: {1 - (1-expected_decay_per_step)**1000:.4f}")

    # Create tile with 6T1C device
    device = create_6t1c_device(DT_BATCH_SEC, include_retention=True)
    rpu_config = SingleRPUConfig(device=device)

    # Small tile for testing
    in_size = 10
    out_size = 10
    tile = AnalogTile(out_size, in_size, rpu_config)

    if USE_CUDA:
        tile = tile.cuda()

    # Set initial weights to known values
    initial_weights = torch.ones(out_size, in_size) * 0.5
    tile.set_weights(initial_weights)

    # Get initial weights
    w0 = tile.get_weights()[0].clone()
    print(f"\nInitial weights mean: {w0.mean().item():.6f}")

    # Run forward passes (this should trigger retention decay)
    x_input = torch.randn(1, in_size).to(DEVICE)

    steps_to_test = [100, 500, 1000, 5000, 10000]

    print(f"\nDecay over forward passes:")
    print("-"*60)
    print(f"{'Steps':<10} {'Weights Mean':<15} {'Decay %':<15} {'Expected Decay %':<15}")
    print("-"*60)

    current_step = 0
    for target_steps in steps_to_test:
        # Run forward passes
        for _ in range(target_steps - current_step):
            tile.forward(x_input)
        current_step = target_steps

        # Get current weights
        w_current = tile.get_weights()[0]
        actual_decay = 1 - (w_current.mean().item() / w0.mean().item())
        expected_decay = 1 - (1 - expected_decay_per_step)**target_steps

        print(f"{target_steps:<10} {w_current.mean().item():<15.6f} {actual_decay*100:<15.4f} {expected_decay*100:<15.4f}")

    print("-"*60)

    # Test without retention
    print("\n" + "="*60)
    print("Testing WITHOUT Retention (lifetime=0)")
    print("="*60)

    device_no_ret = create_6t1c_device(DT_BATCH_SEC, include_retention=False)
    rpu_config_no_ret = SingleRPUConfig(device=device_no_ret)
    tile_no_ret = AnalogTile(out_size, in_size, rpu_config_no_ret)

    if USE_CUDA:
        tile_no_ret = tile_no_ret.cuda()

    tile_no_ret.set_weights(initial_weights)
    w0_no_ret = tile_no_ret.get_weights()[0].clone()

    # Run 10000 forward passes
    for _ in range(10000):
        tile_no_ret.forward(x_input)

    w_final_no_ret = tile_no_ret.get_weights()[0]
    decay_no_ret = 1 - (w_final_no_ret.mean().item() / w0_no_ret.mean().item())

    print(f"\nAfter 10000 steps WITHOUT retention:")
    print(f"  Initial weights mean: {w0_no_ret.mean().item():.6f}")
    print(f"  Final weights mean: {w_final_no_ret.mean().item():.6f}")
    print(f"  Decay: {decay_no_ret*100:.4f}%")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if actual_decay > 0.001:  # More than 0.1% decay
        print("✓ Retention decay IS being applied")
    else:
        print("✗ Retention decay is NOT being applied (or very slow)")


def test_retention_with_different_dt():
    """Test retention with different dt_batch values."""
    print("\n" + "="*60)
    print("Testing Retention with Different dt_batch Values")
    print("="*60)

    dt_values = [1.0, 10.0, 60.0, 600.0, 3600.0]  # 1s, 10s, 1min, 10min, 1hour

    in_size = 10
    out_size = 10
    initial_weights = torch.ones(out_size, in_size) * 0.5
    x_input = torch.randn(1, in_size).to(DEVICE)

    print(f"\n{'dt_batch (sec)':<15} {'Lifetime':<15} {'Decay/1000 steps (%)':<20}")
    print("-"*50)

    for dt in dt_values:
        device = create_6t1c_device(dt, include_retention=True)
        rpu_config = SingleRPUConfig(device=device)
        tile = AnalogTile(out_size, in_size, rpu_config)

        if USE_CUDA:
            tile = tile.cuda()

        tile.set_weights(initial_weights)
        w0 = tile.get_weights()[0].mean().item()

        # Run 1000 forward passes
        for _ in range(1000):
            tile.forward(x_input)

        w_final = tile.get_weights()[0].mean().item()
        decay_pct = (1 - w_final/w0) * 100
        lifetime = calculate_lifetime(dt)

        print(f"{dt:<15.1f} {lifetime:<15.1f} {decay_pct:<20.4f}")


if __name__ == "__main__":
    test_retention_decay()
    test_retention_with_different_dt()
