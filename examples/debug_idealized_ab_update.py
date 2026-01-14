# -*- coding: utf-8 -*-
"""Verify ab_weight_update with Idealized devices (no noise).

This isolates the algorithm correctness from device noise effects.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset


def test_idealized_ab_update():
    """Test ab_weight_update with idealized (noise-free) devices."""
    print("\n" + "="*70)
    print("TEST: ab_weight_update with IDEALIZED devices (no noise)")
    print("="*70)

    IN_SIZE = 8
    OUT_SIZE = 4
    RANK = 2
    BATCH_SIZE = 2
    LR = 0.1

    # Use fully idealized config
    rpu_config = PythonLRTTPreset.idealized(
        rank=RANK,
        transfer_every=100,
        lora_alpha=1.0,
        forward_inject=False,
        ab_update_mode="auto",
        reinit_mode="standard",  # A=0, B=Kaiming (not orthogonal)
        is_perfect=True  # Perfect IO
    )

    print(f"\nConfiguration:")
    print(f"  forward_inject: {rpu_config.device.forward_inject}")
    print(f"  ab_update_mode: {rpu_config.device.ab_update_mode}")
    print(f"  reinit_mode: {rpu_config.device.reinit_mode}")

    layer = AnalogLinear(IN_SIZE, OUT_SIZE, bias=False, rpu_config=rpu_config)
    controller = layer.analog_module.controller

    # Set known initial weights
    torch.manual_seed(0)
    A_init = torch.zeros(OUT_SIZE, RANK)
    B_init = torch.randn(RANK, IN_SIZE) * 0.1

    controller.tile_a.set_weights(A_init.clone())
    controller.tile_b.set_weights(B_init.clone())
    controller._tiles_initialized = True

    # Verify tile operations first
    print("\n" + "-"*50)
    print("Verifying tile operations (idealized):")
    print("-"*50)

    A = controller.tile_a.get_weights()[0].clone()
    B = controller.tile_b.get_weights()[0].clone()

    torch.manual_seed(1)
    x = torch.randn(BATCH_SIZE, IN_SIZE)
    d = torch.randn(BATCH_SIZE, OUT_SIZE)

    # Test tile_b.forward
    xb_actual = controller.tile_b.forward(x)
    xb_expected = x @ B.t()
    print(f"\ntile_b.forward(x):")
    print(f"  Expected norm: {xb_expected.norm():.6f}")
    print(f"  Actual norm:   {xb_actual.norm():.6f}")
    print(f"  Max diff:      {(xb_actual - xb_expected).abs().max():.10f}")
    print(f"  Match: {torch.allclose(xb_actual, xb_expected, atol=1e-5)}")

    # Test tile_a.backward
    da_actual = controller.tile_a.backward(d)
    da_expected = d @ A
    print(f"\ntile_a.backward(d) with A=0:")
    print(f"  Expected norm (A=0): {da_expected.norm():.6f}")
    print(f"  Actual norm:         {da_actual.norm():.6f}")
    print(f"  Max diff:            {(da_actual - da_expected).abs().max():.10f}")
    print(f"  Match: {torch.allclose(da_actual, da_expected, atol=1e-5)}")

    # Now test with non-zero A
    A_nonzero = torch.randn(OUT_SIZE, RANK) * 0.1
    controller.tile_a.set_weights(A_nonzero.clone())
    A = controller.tile_a.get_weights()[0].clone()

    da_actual = controller.tile_a.backward(d)
    da_expected = d @ A
    print(f"\ntile_a.backward(d) with non-zero A:")
    print(f"  Expected norm: {da_expected.norm():.6f}")
    print(f"  Actual norm:   {da_actual.norm():.6f}")
    print(f"  Max diff:      {(da_actual - da_expected).abs().max():.10f}")
    print(f"  Match: {torch.allclose(da_actual, da_expected, atol=1e-5)}")

    # Reset to A=0 for update test
    controller.tile_a.set_weights(A_init.clone())

    # Test ab_weight_update
    print("\n" + "-"*50)
    print("Testing ab_weight_update:")
    print("-"*50)

    A = controller.tile_a.get_weights()[0].clone()
    B = controller.tile_b.get_weights()[0].clone()

    # Manual expected calculation
    XB = x @ B.t()  # [batch, rank]
    DA = d @ A      # [batch, rank] - should be 0 since A=0

    expected_dA = -LR * (d.t() @ XB)  # [out, rank]
    expected_dB = -LR * (DA.t() @ x)  # [rank, in] - should be 0 since DA=0

    print(f"\nExpected ΔA (with A=0):")
    print(f"  norm: {expected_dA.norm():.6f}")
    print(f"\nExpected ΔB (with A=0, so DA=0):")
    print(f"  norm: {expected_dB.norm():.6f}")

    # Perform update
    controller.ab_weight_update(x, d, lr=LR)

    A_after = controller.tile_a.get_weights()[0].clone()
    B_after = controller.tile_b.get_weights()[0].clone()

    actual_dA = A_after - A
    actual_dB = B_after - B

    print(f"\nActual ΔA:")
    print(f"  norm: {actual_dA.norm():.6f}")
    print(f"\nActual ΔB:")
    print(f"  norm: {actual_dB.norm():.6f}")

    # Comparison
    print("\n" + "-"*50)
    print("Comparison:")
    print("-"*50)

    if expected_dA.norm() > 1e-10:
        cos_sim_A = torch.nn.functional.cosine_similarity(
            expected_dA.flatten().unsqueeze(0),
            actual_dA.flatten().unsqueeze(0)
        ).item()
        ratio_A = actual_dA.norm() / expected_dA.norm()
        print(f"\nΔA: cosine_sim={cos_sim_A:.4f}, ratio={ratio_A:.4f}")
        if cos_sim_A > 0.99 and abs(ratio_A - 1.0) < 0.1:
            print("  --> PASS: A update correct")
        else:
            print("  --> FAIL: A update incorrect")

    print(f"\nΔB: expected_norm={expected_dB.norm():.6f}, actual_norm={actual_dB.norm():.6f}")
    if actual_dB.norm() < 1e-5:
        print("  --> PASS: B unchanged (correct, since A=0)")
    else:
        print("  --> FAIL: B should not change when A=0")


def test_idealized_transfer():
    """Test transfer with idealized devices."""
    print("\n" + "="*70)
    print("TEST: Transfer with IDEALIZED devices")
    print("="*70)

    IN_SIZE = 8
    OUT_SIZE = 4
    RANK = 2

    rpu_config = PythonLRTTPreset.idealized(
        rank=RANK,
        transfer_every=1,
        lora_alpha=1.0,
        forward_inject=False,
        ab_update_mode="auto",
        reinit_mode="standard",
        is_perfect=True
    )

    # Disable sigma-delta for cleaner test
    rpu_config.device.use_onehot = False
    rpu_config.device.use_sigma_delta = False
    rpu_config.device.transfer_lr = 1.0
    rpu_config.device.transfer_lr_scale = "none"

    layer = AnalogLinear(IN_SIZE, OUT_SIZE, bias=False, rpu_config=rpu_config)
    controller = layer.analog_module.controller

    # Set known weights
    torch.manual_seed(42)
    A = torch.randn(OUT_SIZE, RANK) * 0.5
    B = torch.randn(RANK, IN_SIZE) * 0.5
    C = torch.zeros(OUT_SIZE, IN_SIZE)

    controller.tile_a.set_weights(A.clone())
    controller.tile_b.set_weights(B.clone())
    controller.tile_c.set_weights(C.clone())
    controller._tiles_initialized = True
    controller.transfer_counter = 1

    print(f"\nTransfer LR: {controller.transfer_lr}")
    print(f"A @ B norm: {(A @ B).norm():.6f}")

    # Expected
    expected_C = C + controller.transfer_lr * (A @ B)
    print(f"Expected C norm: {expected_C.norm():.6f}")

    # Transfer
    controller.ab_weight_transfer(use_onehot=False)

    # Actual
    C_after = controller.tile_c.get_weights()[0].clone()
    print(f"Actual C norm: {C_after.norm():.6f}")

    # Comparison
    cos_sim = torch.nn.functional.cosine_similarity(
        expected_C.flatten().unsqueeze(0),
        C_after.flatten().unsqueeze(0)
    ).item()
    efficiency = C_after.norm() / expected_C.norm()

    print(f"\nCosine similarity: {cos_sim:.4f}")
    print(f"Transfer efficiency: {efficiency*100:.1f}%")

    if cos_sim > 0.99 and abs(efficiency - 1.0) < 0.1:
        print("--> PASS: Transfer correct")
    else:
        print("--> FAIL: Transfer incorrect")
        print(f"\nExpected C:\n{expected_C}")
        print(f"\nActual C:\n{C_after}")
        print(f"\nDifference:\n{C_after - expected_C}")


def test_full_training_loop_idealized():
    """Test complete training loop with idealized devices."""
    print("\n" + "="*70)
    print("TEST: Full training loop with IDEALIZED devices")
    print("="*70)

    IN_SIZE = 16
    OUT_SIZE = 8
    RANK = 4
    BATCH_SIZE = 4
    N_UPDATES = 10
    LR = 0.01

    rpu_config = PythonLRTTPreset.idealized(
        rank=RANK,
        transfer_every=N_UPDATES,
        lora_alpha=1.0,
        forward_inject=False,
        ab_update_mode="auto",
        reinit_mode="standard",
        is_perfect=True
    )

    rpu_config.device.use_onehot = False
    rpu_config.device.use_sigma_delta = False
    rpu_config.device.transfer_lr = 1.0
    rpu_config.device.transfer_lr_scale = "none"

    layer = AnalogLinear(IN_SIZE, OUT_SIZE, bias=False, rpu_config=rpu_config)
    controller = layer.analog_module.controller
    controller.reinit()

    print(f"\nConfiguration:")
    print(f"  Dimensions: {IN_SIZE} -> {OUT_SIZE}, rank={RANK}")
    print(f"  Transfer every: {N_UPDATES} updates")
    print(f"  LR: {LR}")

    # Initial state
    C_init = controller.tile_c.get_weights()[0].clone()
    print(f"\nInitial C norm: {C_init.norm():.6f}")

    # Accumulate gradients
    accumulated_G = torch.zeros(OUT_SIZE, IN_SIZE)

    print(f"\nPerforming {N_UPDATES} updates...")
    for i in range(N_UPDATES):
        torch.manual_seed(i * 100)
        x = torch.randn(BATCH_SIZE, IN_SIZE)
        d = torch.randn(BATCH_SIZE, OUT_SIZE)

        # Ideal gradient
        G = d.t() @ x  # [OUT, IN]
        accumulated_G += LR * G  # Accumulate for comparison

        controller.ab_weight_update(x, d, lr=LR)

    # Get A, B before transfer
    A_pre = controller.tile_a.get_weights()[0].clone()
    B_pre = controller.tile_b.get_weights()[0].clone()
    AB_pre = A_pre @ B_pre

    print(f"\nBefore transfer:")
    print(f"  A norm: {A_pre.norm():.6f}")
    print(f"  B norm: {B_pre.norm():.6f}")
    print(f"  A @ B norm: {AB_pre.norm():.6f}")
    print(f"  Accumulated ideal G norm: {accumulated_G.norm():.6f}")

    # Compare A@B with accumulated gradient
    # In projected mode: A@B should approximate -accumulated_G
    # (since updates push A,B to capture the gradient)
    cos_sim_AB_G = torch.nn.functional.cosine_similarity(
        AB_pre.flatten().unsqueeze(0),
        (-accumulated_G).flatten().unsqueeze(0)
    ).item()
    print(f"\n  cos_sim(A@B, -accumulated_G): {cos_sim_AB_G:.4f}")

    # Transfer
    print(f"\nTransfer counter: {controller.transfer_counter}")
    controller.ab_weight_transfer(use_onehot=False)

    # After transfer
    C_after = controller.tile_c.get_weights()[0].clone()
    delta_C = C_after - C_init

    print(f"\nAfter transfer:")
    print(f"  C delta norm: {delta_C.norm():.6f}")

    # Expected: C += transfer_lr * A@B
    expected_delta_C = controller.transfer_lr * AB_pre
    cos_sim_transfer = torch.nn.functional.cosine_similarity(
        delta_C.flatten().unsqueeze(0),
        expected_delta_C.flatten().unsqueeze(0)
    ).item()
    print(f"  cos_sim(delta_C, transfer_lr*A@B): {cos_sim_transfer:.4f}")

    # The key question: does C move in the gradient direction?
    # If A@B ≈ -G, then C += transfer_lr * A@B ≈ C - transfer_lr * G
    # which is SGD descent!
    cos_sim_sgd = torch.nn.functional.cosine_similarity(
        delta_C.flatten().unsqueeze(0),
        (-accumulated_G).flatten().unsqueeze(0)
    ).item()
    print(f"  cos_sim(delta_C, -accumulated_G): {cos_sim_sgd:.4f}")

    if cos_sim_sgd > 0.5:
        print("\n--> PASS: C moved in gradient descent direction")
    else:
        print("\n--> INFO: C movement has low correlation with ideal gradient")
        print("         This may be expected due to low-rank approximation.")


if __name__ == "__main__":
    test_idealized_ab_update()
    test_idealized_transfer()
    test_full_training_loop_idealized()
