# -*- coding: utf-8 -*-
"""Deep debug: ab_weight_update implementation verification.

Detailed analysis of:
1. Why B update ratio is ~1000x expected?
2. Why transfer cosine similarity is only 0.37?
3. orthogonal reinit mode behavior
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset


def debug_ab_weight_update_math():
    """Debug the mathematical correctness of ab_weight_update."""
    print("\n" + "="*70)
    print("DEBUG: ab_weight_update mathematical verification")
    print("="*70)

    # Simple setup
    IN_SIZE = 8
    OUT_SIZE = 4
    RANK = 2
    BATCH_SIZE = 2
    LR = 0.1

    # Create config
    rpu_config = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=RANK,
        transfer_every=100,
        lora_alpha=1.0,
        dt_batch_sec=0.0,
        ab_update_mode="auto",
        forward_inject=False
    )

    layer = AnalogLinear(IN_SIZE, OUT_SIZE, bias=False, rpu_config=rpu_config)
    controller = layer.analog_module.controller

    # Set known initial weights
    torch.manual_seed(0)
    A_init = torch.zeros(OUT_SIZE, RANK)  # A starts at 0 (orthogonal mode)
    B_init = torch.randn(RANK, IN_SIZE) * 0.1

    controller.tile_a.set_weights(A_init.clone())
    controller.tile_b.set_weights(B_init.clone())
    controller._tiles_initialized = True

    print(f"\nDimensions: IN={IN_SIZE}, OUT={OUT_SIZE}, RANK={RANK}, BATCH={BATCH_SIZE}")
    print(f"LR={LR}")

    # Get initial weights
    A = controller.tile_a.get_weights()[0].clone()
    B = controller.tile_b.get_weights()[0].clone()

    print(f"\nInitial A [{A.shape}]:\n{A}")
    print(f"\nInitial B [{B.shape}]:\n{B}")

    # Create known input and gradient
    torch.manual_seed(1)
    x = torch.randn(BATCH_SIZE, IN_SIZE)  # [2, 8]
    d = torch.randn(BATCH_SIZE, OUT_SIZE)  # [2, 4]

    print(f"\nInput x [{x.shape}]:\n{x}")
    print(f"\nGradient d [{d.shape}]:\n{d}")

    # === Manual calculation of expected update ===
    # Projected mode (no lora_alpha):
    # XB = tile_b.forward(x) = x @ B^T = [batch, rank]
    # DA = tile_a.backward(d) = d @ A = [batch, rank]
    #
    # tile_a.update(XB, d): A -= lr * d^T @ XB
    # tile_b.update(x, DA): B -= lr * DA^T @ x

    print("\n" + "-"*50)
    print("Expected calculation (projected mode):")
    print("-"*50)

    # XB = B @ x^T (tile forward: input @ W^T)
    # For tile_b.forward(x): x is [batch, in], output is x @ B^T = [batch, rank]
    XB = x @ B.t()  # [batch, rank]
    print(f"\nXB = x @ B^T [{XB.shape}]:\n{XB}")

    # DA = A^T @ d^T (tile backward: error @ W)
    # For tile_a.backward(d): d is [batch, out], output is d @ A = [batch, rank]
    DA = d @ A  # [batch, rank]
    print(f"\nDA = d @ A [{DA.shape}]:\n{DA}")

    # Expected A update: A -= lr * d^T @ XB
    expected_dA = -LR * (d.t() @ XB)  # [out, batch] @ [batch, rank] = [out, rank]
    print(f"\nExpected ΔA = -lr * d^T @ XB [{expected_dA.shape}]:\n{expected_dA}")
    print(f"Expected ΔA norm: {expected_dA.norm():.6f}")

    # Expected B update: B -= lr * DA^T @ x
    expected_dB = -LR * (DA.t() @ x)  # [rank, batch] @ [batch, in] = [rank, in]
    print(f"\nExpected ΔB = -lr * DA^T @ x [{expected_dB.shape}]:\n{expected_dB}")
    print(f"Expected ΔB norm: {expected_dB.norm():.6f}")

    # Note: DA = d @ A, but A is all zeros, so DA should be zeros!
    print(f"\n*** Note: A is all zeros, so DA = d @ A should be zeros! ***")
    print(f"DA norm: {DA.norm():.6f}")

    # === Actual update ===
    print("\n" + "-"*50)
    print("Actual ab_weight_update:")
    print("-"*50)

    controller.ab_weight_update(x, d, lr=LR, in_trans=False, out_trans=False)

    A_after = controller.tile_a.get_weights()[0].clone()
    B_after = controller.tile_b.get_weights()[0].clone()

    actual_dA = A_after - A
    actual_dB = B_after - B

    print(f"\nActual ΔA:\n{actual_dA}")
    print(f"Actual ΔA norm: {actual_dA.norm():.6f}")

    print(f"\nActual ΔB:\n{actual_dB}")
    print(f"Actual ΔB norm: {actual_dB.norm():.6f}")

    # === Comparison ===
    print("\n" + "-"*50)
    print("Comparison:")
    print("-"*50)

    print(f"\nΔA comparison:")
    print(f"  Expected norm: {expected_dA.norm():.6f}")
    print(f"  Actual norm:   {actual_dA.norm():.6f}")
    if expected_dA.norm() > 1e-10:
        print(f"  Ratio: {actual_dA.norm()/expected_dA.norm():.4f}")
        cos_sim_A = torch.nn.functional.cosine_similarity(
            expected_dA.flatten().unsqueeze(0),
            actual_dA.flatten().unsqueeze(0)
        ).item()
        print(f"  Cosine similarity: {cos_sim_A:.4f}")

    print(f"\nΔB comparison:")
    print(f"  Expected norm: {expected_dB.norm():.6f}")
    print(f"  Actual norm:   {actual_dB.norm():.6f}")
    if expected_dB.norm() > 1e-10:
        print(f"  Ratio: {actual_dB.norm()/expected_dB.norm():.4f}")
        cos_sim_B = torch.nn.functional.cosine_similarity(
            expected_dB.flatten().unsqueeze(0),
            actual_dB.flatten().unsqueeze(0)
        ).item()
        print(f"  Cosine similarity: {cos_sim_B:.4f}")
    else:
        print(f"  Expected is ~0 (A was zeros), actual is: {actual_dB.norm():.6f}")

    # === Check what tile.forward/backward actually do ===
    print("\n" + "-"*50)
    print("Verifying tile.forward / tile.backward semantics:")
    print("-"*50)

    # Reset
    controller.tile_a.set_weights(A_init.clone())
    controller.tile_b.set_weights(B_init.clone())

    # Test tile_b.forward
    xb_actual = controller.tile_b.forward(x)  # Should be x @ B^T
    xb_expected = x @ B_init.t()
    print(f"\ntile_b.forward(x):")
    print(f"  Expected (x @ B^T): norm={xb_expected.norm():.6f}")
    print(f"  Actual:             norm={xb_actual.norm():.6f}")
    print(f"  Match: {torch.allclose(xb_actual, xb_expected, atol=1e-5)}")

    # Test tile_a.backward
    da_actual = controller.tile_a.backward(d)  # Should be d @ A
    da_expected = d @ A_init
    print(f"\ntile_a.backward(d):")
    print(f"  Expected (d @ A): norm={da_expected.norm():.6f}")
    print(f"  Actual:           norm={da_actual.norm():.6f}")
    print(f"  Match: {torch.allclose(da_actual, da_expected, atol=1e-5)}")

    # === The issue: when A=0, DA=0, but B still gets updated! ===
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("="*70)
    print("""
When A is initialized to zeros (orthogonal mode):
  - DA = d @ A = 0 (since A=0)
  - Expected ΔB = -lr * DA^T @ x = 0
  - But actual ΔB ≠ 0 !

This suggests the tile.update() is doing something different,
or there's noise/quantization in the device model.
""")


def debug_transfer_direction():
    """Debug why transfer direction has low cosine similarity."""
    print("\n" + "="*70)
    print("DEBUG: Transfer direction verification")
    print("="*70)

    IN_SIZE = 8
    OUT_SIZE = 4
    RANK = 2
    LR = 0.1
    TRANSFER_LR = 1.0

    # Create config with direct transfer (no sigma-delta noise)
    rpu_config = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=RANK,
        transfer_every=1,  # Transfer after 1 update
        lora_alpha=1.0,
        dt_batch_sec=0.0,
        ab_update_mode="auto",
        forward_inject=False
    )

    # Modify transfer settings for cleaner test
    rpu_config.device.transfer_lr = TRANSFER_LR
    rpu_config.device.use_onehot = False  # Direct weight access
    rpu_config.device.use_sigma_delta = False

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
    controller.transfer_counter = 1  # Ensure transfer happens

    print(f"\nA [{A.shape}]:\n{A}")
    print(f"\nB [{B.shape}]:\n{B}")
    print(f"\nC (zeros) [{C.shape}]")

    # Expected A @ B
    AB = A @ B
    print(f"\nA @ B [{AB.shape}]:\n{AB}")
    print(f"A @ B norm: {AB.norm():.6f}")

    # Expected C after transfer
    transfer_lr = controller.transfer_lr
    print(f"\nTransfer LR (after scaling): {transfer_lr:.6f}")
    expected_C = C + transfer_lr * AB
    print(f"Expected C = 0 + {transfer_lr} * A@B")
    print(f"Expected C norm: {expected_C.norm():.6f}")

    # Perform transfer
    print("\n>>> Calling ab_weight_transfer(use_onehot=False) <<<")
    controller.ab_weight_transfer(use_onehot=False, use_sigma_delta=False)

    # Get actual C
    C_after = controller.tile_c.get_weights()[0].clone()
    print(f"\nActual C after transfer:\n{C_after}")
    print(f"Actual C norm: {C_after.norm():.6f}")

    # Comparison
    delta_C = C_after - C
    print(f"\nΔC (actual):\n{delta_C}")
    print(f"ΔC norm: {delta_C.norm():.6f}")

    print(f"\nExpected ΔC ({transfer_lr} * A@B):\n{transfer_lr * AB}")

    cos_sim = torch.nn.functional.cosine_similarity(
        expected_C.flatten().unsqueeze(0),
        C_after.flatten().unsqueeze(0)
    ).item()
    print(f"\nCosine similarity (expected vs actual C): {cos_sim:.4f}")

    # Element-wise comparison
    if expected_C.norm() > 1e-10:
        ratio = delta_C / (transfer_lr * AB + 1e-10)
        print(f"\nElement-wise ratio (actual/expected):")
        print(f"  Mean: {ratio.mean():.4f}")
        print(f"  Std:  {ratio.std():.4f}")


def debug_orthogonal_reinit():
    """Debug orthogonal reinit mode behavior."""
    print("\n" + "="*70)
    print("DEBUG: Orthogonal reinit mode")
    print("="*70)

    IN_SIZE = 8
    OUT_SIZE = 4
    RANK = 2

    rpu_config = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=RANK,
        transfer_every=1,
        lora_alpha=1.0,
        dt_batch_sec=0.0,
    )

    layer = AnalogLinear(IN_SIZE, OUT_SIZE, bias=False, rpu_config=rpu_config)
    controller = layer.analog_module.controller

    print(f"reinit_mode: {controller.reinit_mode}")

    # First reinit
    controller.reinit()

    A1 = controller.tile_a.get_weights()[0].clone()
    B1 = controller.tile_b.get_weights()[0].clone()

    print(f"\nAfter first reinit:")
    print(f"A norm: {A1.norm():.6f} (should be ~0)")
    print(f"B norm: {B1.norm():.6f}")

    # Check B orthogonality: B @ B^T should be proportional to I
    BBT = B1 @ B1.t()
    print(f"\nB @ B^T (should be ≈ scaled I):\n{BBT}")

    # Second reinit
    controller._tiles_initialized = True
    controller.reinit()

    A2 = controller.tile_a.get_weights()[0].clone()
    B2 = controller.tile_b.get_weights()[0].clone()

    print(f"\nAfter second reinit:")
    print(f"A norm: {A2.norm():.6f} (should be ~0)")
    print(f"B norm: {B2.norm():.6f}")

    # Check if B changed
    B_changed = not torch.allclose(B1, B2, atol=1e-5)
    print(f"\nB changed after second reinit: {B_changed}")

    if not B_changed:
        print("  Note: In orthogonal mode, B is NOT reinitialized after first init!")
        print("  This preserves the orthogonal structure.")


if __name__ == "__main__":
    debug_ab_weight_update_math()
    debug_transfer_direction()
    debug_orthogonal_reinit()
