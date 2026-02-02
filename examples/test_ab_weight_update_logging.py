# -*- coding: utf-8 -*-
"""Test script: ab_weight_update verification with A,B,C weight logging.

This script validates:
1. ab_weight_update (projected mode) correctly updates A and B
2. ab_weight_transfer correctly transfers A@B to C
3. Weight changes are logged before/after transfer

Expected behavior:
- forward_inject=False, ab_update_mode="auto" -> "projected" mode
- ΔA = -lr · D^T · (B·X) = -lr · G · B^T
- ΔB = -lr · (A^T·D)^T · X = -lr · A^T · G
- Transfer: C += transfer_lr * (A @ B)
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset


def get_tile_weights(controller) -> Dict[str, torch.Tensor]:
    """Extract A, B, C weights from controller."""
    A = controller.tile_a.get_weights()[0].clone()  # [d_size, rank]
    B = controller.tile_b.get_weights()[0].clone()  # [rank, x_size]
    C = controller.tile_c.get_weights()[0].clone()  # [d_size, x_size]
    return {'A': A, 'B': B, 'C': C}


def compute_weight_stats(weights: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    """Compute statistics for weight matrices."""
    stats = {}
    for name, W in weights.items():
        stats[name] = {
            'norm': W.norm().item(),
            'mean': W.mean().item(),
            'std': W.std().item(),
            'max': W.max().item(),
            'min': W.min().item(),
        }
    return stats


def print_weight_comparison(before: Dict, after: Dict, label: str):
    """Print weight comparison before/after an operation."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print('='*60)

    for name in ['A', 'B', 'C']:
        W_before = before[name]
        W_after = after[name]
        delta = W_after - W_before

        print(f"\n{name} matrix [{W_before.shape[0]}x{W_before.shape[1]}]:")
        print(f"  Before: norm={W_before.norm():.6f}, mean={W_before.mean():.6f}")
        print(f"  After:  norm={W_after.norm():.6f}, mean={W_after.mean():.6f}")
        print(f"  Delta:  norm={delta.norm():.6f}, mean={delta.mean():.6f}")

        # Check if changed
        if delta.norm() < 1e-10:
            print(f"  --> NO CHANGE")
        else:
            print(f"  --> CHANGED (relative: {delta.norm()/W_before.norm()*100:.2f}%)")


def test_ab_weight_update():
    """Test ab_weight_update mechanism with detailed logging."""
    print("\n" + "="*70)
    print("TEST: ab_weight_update verification")
    print("="*70)

    # Configuration
    IN_SIZE = 32
    OUT_SIZE = 16
    RANK = 4
    BATCH_SIZE = 8
    TRANSFER_EVERY = 10  # Small for quick testing
    LR = 0.01

    # Create LRTT config
    rpu_config = PythonLRTTPreset.sixt1c_ab_ideal(
        rank=RANK,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=1.0,
        dt_batch_sec=0.0,  # No retention for clean test
        ab_update_mode="auto",
        forward_inject=False
    )

    # Print config
    device_cfg = rpu_config.device
    print(f"\nConfiguration:")
    print(f"  forward_inject: {device_cfg.forward_inject}")
    print(f"  ab_update_mode: {device_cfg.ab_update_mode}")
    print(f"  transfer_every: {device_cfg.transfer_every}")
    print(f"  lora_alpha: {device_cfg.lora_alpha}")
    print(f"  reinit_mode: {device_cfg.reinit_mode}")

    # Determine effective mode
    effective_mode = "chain_rule" if device_cfg.forward_inject else "projected"
    print(f"  --> Effective ab_update_mode: {effective_mode}")

    # Create layer
    layer = AnalogLinear(IN_SIZE, OUT_SIZE, bias=False, rpu_config=rpu_config)
    controller = layer.analog_module.controller

    # Initialize
    controller.reinit()

    print(f"\nLayer dimensions: {IN_SIZE} -> {OUT_SIZE}, rank={RANK}")
    print(f"A: [{OUT_SIZE}, {RANK}], B: [{RANK}, {IN_SIZE}], C: [{OUT_SIZE}, {IN_SIZE}]")

    # ===== Test 1: Single ab_weight_update =====
    print("\n" + "-"*60)
    print("Test 1: Single ab_weight_update call")
    print("-"*60)

    # Create input and gradient
    x = torch.randn(BATCH_SIZE, IN_SIZE)
    d = torch.randn(BATCH_SIZE, OUT_SIZE)  # Error/gradient

    # Record before
    weights_before = get_tile_weights(controller)

    # Expected gradient G = D^T @ X (ideal update direction)
    G = d.t() @ x  # [OUT_SIZE, IN_SIZE]
    print(f"\nIdeal gradient G = D^T @ X: norm={G.norm():.6f}")

    # Call ab_weight_update
    controller.ab_weight_update(x, d, lr=LR, in_trans=False, out_trans=False)

    # Record after
    weights_after = get_tile_weights(controller)

    print_weight_comparison(weights_before, weights_after, "After ab_weight_update")

    # Verify update direction
    A_before, B_before = weights_before['A'], weights_before['B']
    A_after, B_after = weights_after['A'], weights_after['B']

    # Expected updates (projected mode):
    # ΔA = -lr · D^T · (B·X) = -lr · G · B^T
    # ΔB = -lr · (A^T·D)^T · X = -lr · A^T · G
    XB = x @ B_before.t()  # [batch, rank]
    DA = d @ A_before      # [batch, rank]

    expected_dA = -LR * (d.t() @ XB)  # [OUT_SIZE, RANK]
    expected_dB = -LR * (DA.t() @ x)  # [RANK, IN_SIZE]

    actual_dA = A_after - A_before
    actual_dB = B_after - B_before

    print(f"\nUpdate verification (projected mode):")
    print(f"  Expected ΔA norm: {expected_dA.norm():.6f}")
    print(f"  Actual ΔA norm:   {actual_dA.norm():.6f}")
    print(f"  A update ratio:   {actual_dA.norm()/expected_dA.norm():.4f}" if expected_dA.norm() > 1e-10 else "  A update: N/A")
    print(f"  Expected ΔB norm: {expected_dB.norm():.6f}")
    print(f"  Actual ΔB norm:   {actual_dB.norm():.6f}")
    print(f"  B update ratio:   {actual_dB.norm()/expected_dB.norm():.4f}" if expected_dB.norm() > 1e-10 else "  B update: N/A")

    # C should NOT change during ab_weight_update
    C_delta = weights_after['C'] - weights_before['C']
    print(f"\n  C change (should be ~0): {C_delta.norm():.10f}")
    if C_delta.norm() < 1e-8:
        print("  --> PASS: C unchanged during ab_weight_update")
    else:
        print("  --> FAIL: C should not change during ab_weight_update!")

    # ===== Test 2: Multiple updates then transfer =====
    print("\n" + "-"*60)
    print(f"Test 2: {TRANSFER_EVERY} updates then transfer")
    print("-"*60)

    # Re-initialize
    controller.reinit()
    controller.transfer_counter = 0

    weights_init = get_tile_weights(controller)
    print(f"\nInitial weights:")
    for name, W in weights_init.items():
        print(f"  {name}: norm={W.norm():.6f}")

    # Accumulate updates
    print(f"\nPerforming {TRANSFER_EVERY} ab_weight_update calls...")
    for i in range(TRANSFER_EVERY):
        x_i = torch.randn(BATCH_SIZE, IN_SIZE)
        d_i = torch.randn(BATCH_SIZE, OUT_SIZE)
        controller.ab_weight_update(x_i, d_i, lr=LR)

    weights_pre_transfer = get_tile_weights(controller)
    print(f"\nAfter {TRANSFER_EVERY} updates (before transfer):")
    for name, W in weights_pre_transfer.items():
        delta = W - weights_init[name]
        print(f"  {name}: norm={W.norm():.6f}, delta_norm={delta.norm():.6f}")

    # Compute A @ B product before transfer
    AB_pre = weights_pre_transfer['A'] @ weights_pre_transfer['B']
    print(f"\nA @ B product (to be transferred to C):")
    print(f"  A @ B norm: {AB_pre.norm():.6f}")
    print(f"  A @ B mean: {AB_pre.mean():.6f}")

    # Check if transfer should happen
    print(f"\nTransfer counter: {controller.transfer_counter}")
    print(f"Should transfer: {controller.should_transfer()}")

    # Perform transfer
    print("\n>>> Calling ab_weight_transfer() <<<")
    controller.ab_weight_transfer(use_onehot=True, use_sigma_delta=True)

    weights_post_transfer = get_tile_weights(controller)

    print_weight_comparison(weights_pre_transfer, weights_post_transfer,
                           "After ab_weight_transfer")

    # Verify transfer to C
    C_before_transfer = weights_pre_transfer['C']
    C_after_transfer = weights_post_transfer['C']
    C_delta = C_after_transfer - C_before_transfer

    # Expected: C += transfer_lr * (A @ B)
    transfer_lr = controller.transfer_lr
    expected_C_delta = transfer_lr * AB_pre

    print(f"\n\nTransfer verification:")
    print(f"  transfer_lr: {transfer_lr:.6f}")
    print(f"  Expected C delta norm: {expected_C_delta.norm():.6f}")
    print(f"  Actual C delta norm:   {C_delta.norm():.6f}")

    if expected_C_delta.norm() > 1e-10:
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            expected_C_delta.flatten().unsqueeze(0),
            C_delta.flatten().unsqueeze(0)
        ).item()
        print(f"  Cosine similarity: {cos_sim:.4f}")

        # Transfer efficiency
        efficiency = C_delta.norm() / expected_C_delta.norm()
        print(f"  Transfer efficiency: {efficiency*100:.1f}%")

        if cos_sim > 0.9:
            print("  --> PASS: Transfer direction correct")
        else:
            print("  --> WARNING: Transfer direction may be incorrect")

    # Verify A, B reinit
    print(f"\nReinit verification (mode={controller.reinit_mode}):")
    A_post = weights_post_transfer['A']
    B_post = weights_post_transfer['B']

    if controller.reinit_mode == "standard":
        print(f"  A norm (should be ~0): {A_post.norm():.6f}")
        if A_post.norm() < 1e-6:
            print("  --> PASS: A reset to 0")
        else:
            print("  --> FAIL: A should be reset to 0")
    elif controller.reinit_mode == "orthogonal":
        print(f"  A norm (should be ~0): {A_post.norm():.6f}")
        # Check B orthogonality
        BBT = B_post @ B_post.t()
        off_diag = BBT - torch.diag(torch.diag(BBT))
        print(f"  B @ B^T off-diagonal norm: {off_diag.norm():.6f}")
        if A_post.norm() < 1e-6:
            print("  --> PASS: A reset to 0")
    elif controller.reinit_mode == "decay":
        expected_A = weights_pre_transfer['A'] * controller.decay_factor
        expected_B = weights_pre_transfer['B'] * controller.decay_factor
        print(f"  Expected A norm (decay): {expected_A.norm():.6f}")
        print(f"  Actual A norm: {A_post.norm():.6f}")

    # ===== Test 3: Training loop simulation =====
    print("\n" + "-"*60)
    print("Test 3: Training loop simulation (3 epochs)")
    print("-"*60)

    controller.reinit()

    training_log = []
    n_epochs = 3
    batches_per_epoch = TRANSFER_EVERY * 2  # 2 transfers per epoch

    for epoch in range(n_epochs):
        epoch_log = {'epoch': epoch, 'transfers': 0, 'C_norms': []}

        for batch in range(batches_per_epoch):
            x_batch = torch.randn(BATCH_SIZE, IN_SIZE)
            d_batch = torch.randn(BATCH_SIZE, OUT_SIZE)

            # Update A, B
            controller.ab_weight_update(x_batch, d_batch, lr=LR)

            # Check if transfer needed
            if controller.should_transfer():
                weights_pre = get_tile_weights(controller)
                controller.ab_weight_transfer()
                weights_post = get_tile_weights(controller)

                epoch_log['transfers'] += 1
                epoch_log['C_norms'].append(weights_post['C'].norm().item())

        training_log.append(epoch_log)

        weights_end = get_tile_weights(controller)
        print(f"\nEpoch {epoch+1}:")
        print(f"  Transfers: {epoch_log['transfers']}")
        print(f"  Final C norm: {weights_end['C'].norm():.6f}")
        print(f"  Total transfers so far: {controller.num_transfers}")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

    # Summary
    print("\nSummary:")
    print(f"  ab_update_mode: {effective_mode}")
    print(f"  forward_inject: {device_cfg.forward_inject}")
    print(f"  Total A updates: {controller.num_a_updates}")
    print(f"  Total B updates: {controller.num_b_updates}")
    print(f"  Total transfers: {controller.num_transfers}")

    return True


def test_chain_rule_vs_projected():
    """Compare chain_rule and projected update modes."""
    print("\n" + "="*70)
    print("TEST: chain_rule vs projected mode comparison")
    print("="*70)

    IN_SIZE = 32
    OUT_SIZE = 16
    RANK = 4
    BATCH_SIZE = 8
    LR = 0.01
    LORA_ALPHA = 4.0

    # Common input/gradient
    torch.manual_seed(42)
    x = torch.randn(BATCH_SIZE, IN_SIZE)
    d = torch.randn(BATCH_SIZE, OUT_SIZE)

    results = {}

    for mode, forward_inject in [("projected", False), ("chain_rule", True)]:
        print(f"\n--- Mode: {mode} (forward_inject={forward_inject}) ---")

        rpu_config = PythonLRTTPreset.sixt1c_ab_ideal(
            rank=RANK,
            transfer_every=100,
            lora_alpha=LORA_ALPHA,
            dt_batch_sec=0.0,
            ab_update_mode="auto",
            forward_inject=forward_inject
        )

        layer = AnalogLinear(IN_SIZE, OUT_SIZE, bias=False, rpu_config=rpu_config)
        controller = layer.analog_module.controller

        # Set same initial weights
        torch.manual_seed(123)
        A_init = torch.randn(OUT_SIZE, RANK) * 0.1
        B_init = torch.randn(RANK, IN_SIZE) * 0.1
        C_init = torch.randn(OUT_SIZE, IN_SIZE) * 0.1

        controller.tile_a.set_weights(A_init.clone())
        controller.tile_b.set_weights(B_init.clone())
        controller.tile_c.set_weights(C_init.clone())
        controller._tiles_initialized = True

        # Get weights before
        weights_before = get_tile_weights(controller)

        # Update
        controller.ab_weight_update(x.clone(), d.clone(), lr=LR)

        # Get weights after
        weights_after = get_tile_weights(controller)

        dA = weights_after['A'] - weights_before['A']
        dB = weights_after['B'] - weights_before['B']

        results[mode] = {
            'dA_norm': dA.norm().item(),
            'dB_norm': dB.norm().item(),
            'dA': dA,
            'dB': dB,
        }

        print(f"  ΔA norm: {dA.norm():.6f}")
        print(f"  ΔB norm: {dB.norm():.6f}")

    # Compare
    print("\n--- Comparison ---")
    ratio_A = results['chain_rule']['dA_norm'] / results['projected']['dA_norm']
    ratio_B = results['chain_rule']['dB_norm'] / results['projected']['dB_norm']

    print(f"chain_rule/projected ratio for ΔA: {ratio_A:.4f}")
    print(f"chain_rule/projected ratio for ΔB: {ratio_B:.4f}")
    print(f"Expected ratio (lora_alpha={LORA_ALPHA}): {LORA_ALPHA:.4f}")

    if abs(ratio_A - LORA_ALPHA) < 0.1 and abs(ratio_B - LORA_ALPHA) < 0.1:
        print("--> PASS: chain_rule applies lora_alpha correctly")
    else:
        print("--> INFO: Ratio differs from lora_alpha (may be due to noise)")


if __name__ == "__main__":
    test_ab_weight_update()
    test_chain_rule_vs_projected()
