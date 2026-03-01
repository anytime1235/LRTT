#!/usr/bin/env python3
"""Test if A weights update with different learning rates."""

import sys
sys.path.insert(0, '/data/LRTT_transformer/src')

import torch
import torch.nn as nn
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam


def create_sixt1c_config(rank=4, lora_alpha=2.0):
    """Create sixt1c_lora config with dw_min=0.001981."""

    ab_device = LinearStepDevice(
        dw_min=0.001981,  # 6T1C characteristic
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.0,  # No noise for testing
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        gamma_up_dtod=0.0,
        gamma_down_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=0.0,
    )

    c_device = SoftBoundsDevice(
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

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=1000000,
        lora_alpha=lora_alpha,
        reinit_gain=0.1,
        reinit_mode="hybrid",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.001
    device_config.units_in_mbatch = True
    device_config.forward_inject = True
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


def test_a_update_with_lr(lr: float, n_steps: int = 5):
    """Test if A updates with given learning rate."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    torch.manual_seed(42)
    d_size, x_size = 64, 32
    rank = 4

    digital_model = nn.Linear(x_size, d_size, bias=False)
    rpu_config = create_sixt1c_config(rank=rank, lora_alpha=2.0)
    analog_model = convert_to_analog(digital_model, rpu_config)
    analog_model = analog_model.to(device)

    # Get controller
    lrtt_tile = None
    for m in analog_model.modules():
        if hasattr(m, 'controller'):
            lrtt_tile = m
            break

    controller = lrtt_tile.controller

    # Get initial A and B weights
    A_init, _ = controller.tile_a.get_weights()
    A_init_norm = A_init.abs().sum().item()
    B_init, _ = controller.tile_b.get_weights()

    # Create optimizer with specified lr
    optimizer = AnalogAdam(analog_model.parameters(), lr=lr)

    # Simple training loop
    for step in range(n_steps):
        optimizer.zero_grad()

        # Random input and target
        x = torch.randn(8, x_size, device=device)
        target = torch.randn(8, d_size, device=device)

        # Forward
        output = analog_model(x)
        loss = ((output - target) ** 2).mean()

        # Backward
        loss.backward()

        # Step
        optimizer.step()

    # Get final A weights
    A_final, _ = controller.tile_a.get_weights()
    A_final_norm = A_final.abs().sum().item()
    A_change = (A_final - A_init).abs().sum().item()

    # Check B change
    B_final, _ = controller.tile_b.get_weights()
    B_change = (B_final - B_init).abs().sum().item()

    return {
        'lr': lr,
        'A_init_norm': A_init_norm,
        'A_final_norm': A_final_norm,
        'A_change': A_change,
        'B_change': B_change,
        'A_updated': A_change > 1e-6,
        'B_updated': B_change > 1e-6,
        'final_loss': loss.item(),
    }


def main():
    print("=" * 70)
    print("Testing A weight updates with different learning rates")
    print("dw_min = 0.001981")
    print("=" * 70)

    # Test different learning rates
    lr_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]

    results = []
    for lr in lr_values:
        result = test_a_update_with_lr(lr, n_steps=10)
        results.append(result)

        a_status = "✓" if result['A_updated'] else "✗"
        b_status = "✓" if result['B_updated'] else "✗"
        print(f"\nlr={lr:.0e}: A{a_status} B{b_status}")
        print(f"  A_change: {result['A_change']:.6f}")
        print(f"  B_change: {result['B_change']:.6f}")
        print(f"  final_loss: {result['final_loss']:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'lr':<10} {'A':<5} {'B':<5} {'A Change':<12} {'B Change':<12} {'Loss':<10}")
    print("-" * 60)
    for r in results:
        a_up = "YES" if r['A_updated'] else "NO"
        b_up = "YES" if r['B_updated'] else "NO"
        print(f"{r['lr']:<10.0e} {a_up:<5} {b_up:<5} {r['A_change']:<12.6f} {r['B_change']:<12.6f} {r['final_loss']:<10.4f}")

    # Find threshold
    updated_lrs = [r['lr'] for r in results if r['A_updated']]
    if updated_lrs:
        min_lr = min(updated_lrs)
        print(f"\n→ Minimum lr for A update: {min_lr:.0e}")
    else:
        print("\n→ A did not update at any tested lr!")


if __name__ == "__main__":
    main()
