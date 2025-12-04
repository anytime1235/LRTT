# -*- coding: utf-8 -*-
"""Test LRTT transfer accuracy with different configurations.

Experiments with:
- set_weights vs pulsed update (one-hot)
- Different BL (bit length) values
- Different noise/bound management settings
- is_perfect True/False
"""

import torch
import pandas as pd
from itertools import product

from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import (
    BoundManagementType, NoiseManagementType, WeightNoiseType, UpdateParameters
)
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.configs.devices import IdealDevice
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

# ==============================================================================
# Configuration
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Tile dimensions
D_SIZE = 64
X_SIZE = 128
RANK = 32

# Test parameters to sweep
BL_VALUES = [31]
AB_SCALE_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
DEVICE_TYPES = ['idealized']  # IdealizedPresetDevice only

# Fixed parameters
IS_PERFECT = False
TRANSFER_LR = 2.0
LORA_ALPHA = 2.0


# ==============================================================================
# Helper functions
# ==============================================================================

def create_lrtt_config(
    bl: int,
    device_type: str
):
    """Create LRTT configuration with specified parameters."""

    # Device selection
    if device_type == 'ideal':
        unit_devices = [IdealDevice(), IdealDevice(), IdealDevice()]
    else:
        unit_devices = [
            IdealizedPresetDevice(),
            IdealizedPresetDevice(),
            IdealizedPresetDevice()
        ]

    device_config = PythonLRTTDevice(
        rank=RANK,
        transfer_every=10000,  # Large value to control transfer manually
        lora_alpha=LORA_ALPHA,
        transfer_lr=TRANSFER_LR,
        forward_inject=False,
        correct_gradient_magnitudes=False,
        unit_cell_devices=unit_devices
    )

    mapping = MappingParameter(
        weight_scaling_omega=0.6,
        digital_bias=True,
        max_input_size=512,
        max_output_size=512
    )

    # Forward/backward IO - fixed settings (is_perfect=False)
    forward_io = IOParameters(
        inp_res=0.007937,
        inp_bound=1.0,
        inp_noise=0.0,
        out_res=0.001961,
        out_bound=12.0,
        out_noise=0.06,
        w_noise=0.0,
        w_noise_type=WeightNoiseType.NONE,
        bound_management=BoundManagementType.ITERATIVE,
        noise_management=NoiseManagementType.ABS_MAX,
        is_perfect=IS_PERFECT,
        max_bm_factor=1000,
    )

    # Update parameters - fixed to True/True
    update_params = UpdateParameters(
        desired_bl=bl,
        update_bl_management=True,
        update_management=True,
    )

    return PythonLRTTRPUConfig(
        device=device_config,
        mapping=mapping,
        forward=forward_io,
        backward=forward_io,
        update=update_params
    )


def test_transfer_set_weights(config, A_init, B_init):
    """Test transfer using set_weights (direct method).

    This bypasses tile.update() and directly sets C = C + transfer_lr * A @ B.
    """
    tile = LRTTSimulatorTile(d_size=D_SIZE, x_size=X_SIZE, rpu_config=config)
    if DEVICE.type == "cuda":
        tile.cuda()

    controller = tile.controller

    # Initialize
    controller.tile_a.set_weights(A_init.clone())
    controller.tile_b.set_weights(B_init.clone())
    C_init = torch.zeros(D_SIZE, X_SIZE, device=DEVICE)
    controller.tile_c.set_weights(C_init)

    # Read weights
    A_read = controller.tile_a.get_weights()[0][:, :RANK]
    B_read = controller.tile_b.get_weights()[0][:RANK, :]
    C_before = controller.tile_c.get_weights()[0].clone()

    # Expected result
    expected_delta = TRANSFER_LR * (A_read @ B_read)

    # Transfer using set_weights
    C_new = C_before + expected_delta
    controller.tile_c.set_weights(C_new)

    # Measure
    C_after = controller.tile_c.get_weights()[0]
    actual_delta = C_after - C_before

    # Metrics
    ratio = actual_delta.norm() / (expected_delta.norm() + 1e-10)
    correlation = (actual_delta * expected_delta).sum() / (
        actual_delta.norm() * expected_delta.norm() + 1e-10
    )
    mse = ((actual_delta - expected_delta) ** 2).mean()

    return {
        'method': 'set_weights',
        'expected_norm': expected_delta.norm().item(),
        'actual_norm': actual_delta.norm().item(),
        'ratio': ratio.item(),
        'correlation': correlation.item(),
        'mse': mse.item()
    }


def test_transfer_onehot(config, A_init, B_init):
    """Test transfer using one-hot pulsed update method."""
    tile = LRTTSimulatorTile(d_size=D_SIZE, x_size=X_SIZE, rpu_config=config)
    if DEVICE.type == "cuda":
        tile.cuda()

    controller = tile.controller
    controller.num_transfers = 100  # Disable debug output

    # Initialize
    controller.tile_a.set_weights(A_init.clone())
    controller.tile_b.set_weights(B_init.clone())
    C_init = torch.zeros(D_SIZE, X_SIZE, device=DEVICE)
    controller.tile_c.set_weights(C_init)

    # Read weights for expected calculation
    A_read = controller.tile_a.get_weights()[0][:, :RANK]
    B_read = controller.tile_b.get_weights()[0][:RANK, :]
    C_before = controller.tile_c.get_weights()[0].clone()

    # Expected result (using get_weights, not forward)
    expected_delta = TRANSFER_LR * (A_read @ B_read)

    # Transfer using one-hot method
    controller.ab_weight_transfer(use_onehot=True)

    # Note: reinit() is called inside ab_weight_transfer, but C should be updated
    C_after = controller.tile_c.get_weights()[0]
    actual_delta = C_after - C_before

    # Metrics
    ratio = actual_delta.norm() / (expected_delta.norm() + 1e-10)
    correlation = (actual_delta * expected_delta).sum() / (
        actual_delta.norm() * expected_delta.norm() + 1e-10
    )
    mse = ((actual_delta - expected_delta) ** 2).mean()

    return {
        'method': 'onehot',
        'expected_norm': expected_delta.norm().item(),
        'actual_norm': actual_delta.norm().item(),
        'ratio': ratio.item(),
        'correlation': correlation.item(),
        'mse': mse.item()
    }


def test_transfer_direct(config, A_init, B_init):
    """Test transfer using direct chunk-based update method."""
    tile = LRTTSimulatorTile(d_size=D_SIZE, x_size=X_SIZE, rpu_config=config)
    if DEVICE.type == "cuda":
        tile.cuda()

    controller = tile.controller
    controller.num_transfers = 100  # Disable debug output

    # Initialize
    controller.tile_a.set_weights(A_init.clone())
    controller.tile_b.set_weights(B_init.clone())
    C_init = torch.zeros(D_SIZE, X_SIZE, device=DEVICE)
    controller.tile_c.set_weights(C_init)

    # Read weights for expected calculation
    A_read = controller.tile_a.get_weights()[0][:, :RANK]
    B_read = controller.tile_b.get_weights()[0][:RANK, :]
    C_before = controller.tile_c.get_weights()[0].clone()

    # Expected result
    expected_delta = TRANSFER_LR * (A_read @ B_read)

    # Transfer using direct method
    controller.ab_weight_transfer(use_onehot=False)

    C_after = controller.tile_c.get_weights()[0]
    actual_delta = C_after - C_before

    # Metrics
    ratio = actual_delta.norm() / (expected_delta.norm() + 1e-10)
    correlation = (actual_delta * expected_delta).sum() / (
        actual_delta.norm() * expected_delta.norm() + 1e-10
    )
    mse = ((actual_delta - expected_delta) ** 2).mean()

    return {
        'method': 'direct',
        'expected_norm': expected_delta.norm().item(),
        'actual_norm': actual_delta.norm().item(),
        'ratio': ratio.item(),
        'correlation': correlation.item(),
        'mse': mse.item()
    }


# ==============================================================================
# Main experiment
# ==============================================================================

def run_experiments():
    """Run all experiments and collect results."""

    # Fixed random seed for reproducibility
    torch.manual_seed(42)
    # Base random tensors (will be scaled)
    A_base = torch.randn(D_SIZE, RANK, device=DEVICE)
    B_base = torch.randn(RANK, X_SIZE, device=DEVICE)

    results = []

    # Full sweep
    total_configs = len(BL_VALUES) * len(AB_SCALE_VALUES) * len(DEVICE_TYPES)
    print(f"\nRunning {total_configs} configurations x 3 methods = {total_configs * 3} experiments")
    print(f"is_perfect = {IS_PERFECT} (fixed)")
    print("=" * 80)

    config_idx = 0
    for bl, ab_scale, device_type in product(
        BL_VALUES, AB_SCALE_VALUES, DEVICE_TYPES
    ):
        config_idx += 1

        # Scale A and B
        A_init = A_base * ab_scale
        B_init = B_base * ab_scale

        config_name = f"BL={bl}, ab_scale={ab_scale}, dev={device_type}"
        print(f"\n[{config_idx}/{total_configs}] {config_name}")

        try:
            config = create_lrtt_config(bl, device_type)

            # Test all three methods: set_weights, onehot, direct (chunk-based)
            for test_func in [test_transfer_set_weights, test_transfer_onehot, test_transfer_direct]:
                result = test_func(config, A_init, B_init)
                result.update({
                    'bl': bl,
                    'ab_scale': ab_scale,
                    'device_type': device_type
                })
                results.append(result)

                print(f"  {result['method']:12s}: ratio={result['ratio']:.4f}, corr={result['correlation']:.4f}, mse={result['mse']:.6f}")

        except Exception as e:
            print(f"  ERROR: {e}")

    return pd.DataFrame(results)


def analyze_results(df):
    """Analyze and summarize results with clean tables."""

    print("\n" + "=" * 100)
    print(f"RESULTS SUMMARY (IdealizedPresetDevice, is_perfect={IS_PERFECT}, update_bl_mgmt=True, update_mgmt=True)")
    print("=" * 100)

    # =========================================================================
    # Table 1: All methods comparison by ab_scale and BL
    # =========================================================================
    print("\n" + "-" * 100)
    print("TABLE 1: All Methods by ab_scale and BL")
    print("-" * 100)
    print(f"{'ab_scale':<10} {'BL':<6} {'method':<14} {'ratio':<10} {'corr':<10} {'mse':<12} {'exp_norm':<12}")
    print("-" * 100)

    for ab_scale in AB_SCALE_VALUES:
        for bl in BL_VALUES:
            for method in ['set_weights', 'onehot', 'direct']:
                subset = df[(df['ab_scale'] == ab_scale) & (df['bl'] == bl) & (df['method'] == method)]
                if len(subset) > 0:
                    row = subset.iloc[0]
                    print(f"{ab_scale:<10} {bl:<6} {method:<14} "
                          f"{row['ratio']:<10.4f} {row['correlation']:<10.4f} {row['mse']:<12.6f} {row['expected_norm']:<12.4f}")
            print()  # blank line between BL groups

    # =========================================================================
    # Table 2: Summary by ab_scale (averaged over BL)
    # =========================================================================
    print("\n" + "-" * 100)
    print("TABLE 2: Summary by ab_scale (averaged over BL)")
    print("-" * 100)
    print(f"{'ab_scale':<10} {'method':<14} {'avg_ratio':<12} {'avg_corr':<12} {'avg_mse':<12}")
    print("-" * 100)

    for ab_scale in AB_SCALE_VALUES:
        for method in ['set_weights', 'onehot', 'direct']:
            subset = df[(df['ab_scale'] == ab_scale) & (df['method'] == method)]
            if len(subset) > 0:
                print(f"{ab_scale:<10} {method:<14} {subset['ratio'].mean():<12.4f} "
                      f"{subset['correlation'].mean():<12.4f} {subset['mse'].mean():<12.6f}")
        print()

    # =========================================================================
    # Table 3: Summary by BL (averaged over ab_scale)
    # =========================================================================
    print("\n" + "-" * 100)
    print("TABLE 3: Summary by BL (averaged over ab_scale)")
    print("-" * 100)
    print(f"{'BL':<6} {'method':<14} {'avg_ratio':<12} {'avg_corr':<12} {'avg_mse':<12}")
    print("-" * 100)

    for bl in BL_VALUES:
        for method in ['set_weights', 'onehot', 'direct']:
            subset = df[(df['bl'] == bl) & (df['method'] == method)]
            if len(subset) > 0:
                print(f"{bl:<6} {method:<14} {subset['ratio'].mean():<12.4f} "
                      f"{subset['correlation'].mean():<12.4f} {subset['mse'].mean():<12.6f}")
        print()


if __name__ == "__main__":
    print("=" * 80)
    print("LRTT Transfer Accuracy Test")
    print("=" * 80)
    print(f"Tile: {D_SIZE} x {X_SIZE}, Rank: {RANK}")
    print(f"Transfer LR: {TRANSFER_LR}, LoRA Alpha: {LORA_ALPHA}")

    df = run_experiments()
    analyze_results(df)

    # Save results
    csv_path = "lrtt_transfer_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
