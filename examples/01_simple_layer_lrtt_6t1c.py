# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example: LRTT with 6T1C (6 Transistors, 1 Capacitor) devices.

This example demonstrates using LRTT (Low-Rank Tensor-Train) with 6T1C
capacitor-based synaptic devices. The 6T1C device parameters are based on
experimental measurements from 6T1C_result.xlsx.

6T1C Device Characteristics:
    - ~1000 conductance states per direction
    - Capacitor-based weight storage with exponential decay
    - Time constant τ ≈ 775 min (12.9 hours)
    - Decay target: 0V

This example compares:
    1. Idealized LRTT (no device non-idealities)
    2. 6T1C LRTT without retention (update characteristics only)
    3. 6T1C LRTT with retention (full device model)
"""
# pylint: disable=invalid-name

import torch
from torch import Tensor
from torch.nn.functional import mse_loss

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import (
    PythonLRTTRPUConfig,
    lrtt_sixt1c_config,
    lrtt_sixt1c_no_retention_config,
)
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.rpu_base import cuda

# Set seed for reproducibility
torch.manual_seed(42)


def create_sum_data(n_samples=10):
    """Create data where output is related to input sum."""
    x_data = torch.rand(n_samples, 4)
    y_data = torch.zeros(n_samples, 2)
    for i in range(n_samples):
        sum_val = x_data[i].sum()
        y_data[i, 0] = sum_val * 1.5
        y_data[i, 1] = sum_val * 0.5
    return x_data, y_data


def train_model(model, x, y, n_epochs=200, lr=0.1, name="Model"):
    """Train model and return final loss."""
    opt = AnalogSGD(model.parameters(), lr=lr)
    opt.regroup_param_groups(model)

    losses = []
    for epoch in range(n_epochs):
        pred = model(x)
        loss = mse_loss(pred, y)
        losses.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 50 == 0:
            print(f"  [{name}] Epoch {epoch:3d}: Loss = {loss:.6f}")

    return losses


def main():
    print("=" * 70)
    print("LRTT with 6T1C Device Example")
    print("=" * 70)

    # Create training data
    x, y = create_sum_data(20)

    # Move to CUDA if available
    device = torch.device('cuda' if cuda.is_compiled() else 'cpu')
    x = x.to(device)
    y = y.to(device)

    print(f"\nDevice: {device}")
    print(f"Training data shape: {x.shape}")
    print("-" * 70)

    # ==========================================================================
    # Configuration 1: Idealized LRTT (baseline)
    # ==========================================================================
    print("\n[1] Idealized LRTT (baseline)")
    print("-" * 40)

    device_cfg_ideal = PythonLRTTPreset.idealized(
        rank=2,
        transfer_every=100,
        lora_alpha=2.0
    )
    device_cfg_ideal.transfer_lr = device_cfg_ideal.lora_alpha
    device_cfg_ideal.correct_gradient_magnitudes = True

    rpu_config_ideal = PythonLRTTRPUConfig(device=device_cfg_ideal)
    model_ideal = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config_ideal).to(device)

    losses_ideal = train_model(model_ideal, x, y, name="Idealized")

    # ==========================================================================
    # Configuration 2: 6T1C LRTT without retention
    # ==========================================================================
    print("\n[2] 6T1C LRTT (no retention)")
    print("-" * 40)

    device_cfg_6t1c_no_ret = PythonLRTTPreset.sixt1c_no_retention(
        rank=2,
        transfer_every=100,
        lora_alpha=2.0
    )
    device_cfg_6t1c_no_ret.transfer_lr = device_cfg_6t1c_no_ret.lora_alpha
    device_cfg_6t1c_no_ret.correct_gradient_magnitudes = True

    rpu_config_6t1c_no_ret = PythonLRTTRPUConfig(device=device_cfg_6t1c_no_ret)
    model_6t1c_no_ret = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config_6t1c_no_ret).to(device)

    losses_6t1c_no_ret = train_model(model_6t1c_no_ret, x, y, name="6T1C-NoRet")

    # ==========================================================================
    # Configuration 3: 6T1C LRTT with retention (dt_batch = 1 sec)
    # ==========================================================================
    print("\n[3] 6T1C LRTT (with retention, dt_batch=1s)")
    print("-" * 40)

    device_cfg_6t1c_ret = PythonLRTTPreset.sixt1c(
        rank=2,
        transfer_every=100,
        lora_alpha=2.0,
        dt_batch_sec=1.0,
        include_retention=True
    )
    device_cfg_6t1c_ret.transfer_lr = device_cfg_6t1c_ret.lora_alpha
    device_cfg_6t1c_ret.correct_gradient_magnitudes = True

    rpu_config_6t1c_ret = PythonLRTTRPUConfig(device=device_cfg_6t1c_ret)
    model_6t1c_ret = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config_6t1c_ret).to(device)

    losses_6t1c_ret = train_model(model_6t1c_ret, x, y, name="6T1C-Ret")

    # ==========================================================================
    # Configuration 4: 6T1C LRTT with fast decay (dt_batch = 10 min)
    # ==========================================================================
    print("\n[4] 6T1C LRTT (fast decay, dt_batch=10min)")
    print("-" * 40)

    device_cfg_6t1c_fast = PythonLRTTPreset.sixt1c_fast_decay(
        rank=2,
        transfer_every=100,
        lora_alpha=2.0,
        dt_batch_sec=600.0  # 10 minutes
    )
    device_cfg_6t1c_fast.transfer_lr = device_cfg_6t1c_fast.lora_alpha
    device_cfg_6t1c_fast.correct_gradient_magnitudes = True

    rpu_config_6t1c_fast = PythonLRTTRPUConfig(device=device_cfg_6t1c_fast)
    model_6t1c_fast = AnalogLinear(4, 2, bias=False, rpu_config=rpu_config_6t1c_fast).to(device)

    losses_6t1c_fast = train_model(model_6t1c_fast, x, y, name="6T1C-Fast")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    results = [
        ("Idealized", losses_ideal),
        ("6T1C (no retention)", losses_6t1c_no_ret),
        ("6T1C (retention, 1s)", losses_6t1c_ret),
        ("6T1C (fast decay, 10min)", losses_6t1c_fast),
    ]

    print(f"\n{'Configuration':<30} {'Initial Loss':>15} {'Final Loss':>15} {'Reduction':>12}")
    print("-" * 75)

    for name, losses in results:
        initial = losses[0]
        final = losses[-1]
        reduction = (initial - final) / initial * 100
        print(f"{name:<30} {initial:>15.6f} {final:>15.6f} {reduction:>11.1f}%")

    print("-" * 75)

    # ==========================================================================
    # 6T1C Device Info
    # ==========================================================================
    print("\n" + "=" * 70)
    print("6T1C DEVICE PARAMETERS")
    print("=" * 70)
    print("""
    Update Model (LinearStepDevice):
    ─────────────────────────────────────────
      dw_min:      0.001981  (weight step size)
      gamma_up:    -0.1678   (UP nonlinearity)
      gamma_down:  +0.1410   (DOWN nonlinearity)

    Retention Model:
    ─────────────────────────────────────────
      Physical τ:   775.1 min (46505 sec)
      Decay target: 0V (reset = 0.0)

    Lifetime at different dt_batch:
    ─────────────────────────────────────────
      dt_batch = 1 sec  -> lifetime = 46506
      dt_batch = 1 min  -> lifetime = 776
      dt_batch = 10 min -> lifetime = 78
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
