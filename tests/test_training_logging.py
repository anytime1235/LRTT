"""Test actual training with A, B, C weight logging.

Logs weights at each step to verify:
1. A, B updates happen correctly during training
2. Transfer moves AB to C correctly
3. A, B get reinitialized after transfer
"""

import torch
import sys
sys.path.insert(0, '/root/LRTT/src')

from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile


def log_weights(controller, step, label=""):
    """Log current A, B, C weights."""
    A = controller.tile_a.get_weights()[0]
    B = controller.tile_b.get_weights()[0]
    C = controller.tile_c.get_weights()[0]
    AB = A @ B

    print(f"\n{'='*60}")
    print(f"Step {step}: {label}")
    print(f"{'='*60}")
    print(f"||A|| = {torch.norm(A).item():.6f}")
    print(f"||B|| = {torch.norm(B).item():.6f}")
    print(f"||AB|| = {torch.norm(AB).item():.6f}")
    print(f"||C|| = {torch.norm(C).item():.6f}")

    # Show first row of each matrix
    print(f"\nA[0,:] = {A[0,:].tolist()}")
    print(f"B[:,0] = {B[:,0].tolist()}")
    print(f"C[0,:3] = {C[0,:3].tolist()}")

    return {
        'A': A.clone(),
        'B': B.clone(),
        'C': C.clone(),
        'AB': AB.clone(),
        'norm_A': torch.norm(A).item(),
        'norm_B': torch.norm(B).item(),
        'norm_AB': torch.norm(AB).item(),
        'norm_C': torch.norm(C).item(),
    }


def test_training_with_logging():
    """Run training steps with detailed weight logging."""
    print("=" * 70)
    print("LRTT Training Step Logging Test")
    print("=" * 70)

    torch.manual_seed(42)

    d_size, x_size, rank = 8, 8, 2
    batch_size = 4
    lr = 0.1
    transfer_every = 3  # Transfer every 3 steps

    config = PythonLRTTPreset.idealized(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=1.0,
        forward_inject=False,
        ab_update_mode="projected",
        reinit_mode="standard",
        is_perfect=True,
    )

    tile = LRTTSimulatorTile(d_size, x_size, config)
    controller = tile.controller

    print(f"\nConfiguration:")
    print(f"  d_size={d_size}, x_size={x_size}, rank={rank}")
    print(f"  transfer_every={transfer_every}")
    print(f"  ab_update_mode={controller.ab_update_mode}")
    print(f"  forward_inject={controller.forward_inject_enabled}")
    print(f"  reinit_mode={controller.reinit_mode}")
    print(f"  transfer_lr={controller.transfer_lr}")

    # Set non-zero initial weights for A, B
    A_init = torch.randn(d_size, rank) * 0.1
    B_init = torch.randn(rank, x_size) * 0.1
    C_init = torch.zeros(d_size, x_size)

    controller.tile_a.set_weights(A_init)
    controller.tile_b.set_weights(B_init)
    controller.tile_c.set_weights(C_init)

    # Reset transfer counter
    controller.transfer_counter = 0
    controller.num_transfers = 0

    history = []
    history.append(log_weights(controller, 0, "Initial"))

    # Run training steps
    num_steps = 10

    for step in range(1, num_steps + 1):
        # Generate random input and gradient
        x = torch.randn(batch_size, x_size)
        d = torch.randn(batch_size, d_size)  # Error/delta

        # Compute expected gradient for reference
        G = d.T @ x

        # Check if transfer will happen this step (after this update)
        will_transfer = (controller.transfer_counter + 1) >= transfer_every

        # Log before update
        if will_transfer:
            pre_transfer = log_weights(controller, step, f"BEFORE Transfer (counter={controller.transfer_counter})")

        # Use tile.update() which handles both AB update and transfer
        tile._update_handled = False  # Reset flag for new update
        tile.update(x, d)

        # Log after update
        if will_transfer:
            post_transfer = log_weights(controller, step, f"AFTER Transfer (num_transfers={controller.num_transfers})")

            # Analyze transfer
            print(f"\n--- Transfer Analysis ---")
            delta_C = post_transfer['C'] - pre_transfer['C']
            expected_delta_C = controller.transfer_lr * pre_transfer['AB']
            transfer_error = torch.norm(delta_C - expected_delta_C) / (torch.norm(expected_delta_C) + 1e-8)

            print(f"||ΔC|| = {torch.norm(delta_C).item():.6f}")
            print(f"||expected ΔC|| = {torch.norm(expected_delta_C).item():.6f}")
            print(f"Transfer error: {transfer_error.item():.6e}")
            print(f"A reinitialized: ||A_after|| = {post_transfer['norm_A']:.6f} (was {pre_transfer['norm_A']:.6f})")
            print(f"B reinitialized: ||B_after|| = {post_transfer['norm_B']:.6f} (was {pre_transfer['norm_B']:.6f})")
        else:
            # Just log normal update
            current = log_weights(controller, step, f"After update (counter={controller.transfer_counter})")
            history.append(current)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total steps: {num_steps}")
    print(f"Total transfers: {controller.num_transfers}")
    print(f"Total A updates: {controller.num_a_updates}")
    print(f"Total B updates: {controller.num_b_updates}")

    final = log_weights(controller, "Final", "Final State")

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


def test_single_update_detailed():
    """Test a single update with detailed math verification."""
    print("\n" + "=" * 70)
    print("Single Update Detailed Verification")
    print("=" * 70)

    torch.manual_seed(123)

    d_size, x_size, rank = 4, 4, 2
    batch_size = 2
    lr = 0.1

    config = PythonLRTTPreset.idealized(
        rank=rank,
        transfer_every=1000,  # No transfer during this test
        forward_inject=False,
        ab_update_mode="projected",
        reinit_mode="standard",
        is_perfect=True,
    )

    tile = LRTTSimulatorTile(d_size, x_size, config)
    controller = tile.controller

    # Set known initial weights
    A_init = torch.tensor([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
    ], dtype=torch.float32)

    B_init = torch.tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ], dtype=torch.float32)

    controller.tile_a.set_weights(A_init.clone())
    controller.tile_b.set_weights(B_init.clone())

    # Simple input and error
    x = torch.ones(batch_size, x_size)
    d = torch.ones(batch_size, d_size)

    print("\n--- Initial State ---")
    print(f"A:\n{A_init}")
    print(f"B:\n{B_init}")
    print(f"x: {x[0].tolist()}")
    print(f"d: {d[0].tolist()}")

    # Compute expected update
    G = d.T @ x  # [d_size, x_size]
    expected_dA = -lr * (G @ B_init.T)  # [d_size, rank]
    expected_dB = -lr * (A_init.T @ G)  # [rank, x_size]

    print(f"\n--- Expected Updates ---")
    print(f"G = d.T @ x:\n{G}")
    print(f"expected ΔA = -lr * G @ B.T:\n{expected_dA}")
    print(f"expected ΔB = -lr * A.T @ G:\n{expected_dB}")

    # Perform update
    controller.ab_weight_update(x, d, lr)

    # Get actual updates
    A_after = controller.tile_a.get_weights()[0]
    B_after = controller.tile_b.get_weights()[0]

    actual_dA = A_after - A_init
    actual_dB = B_after - B_init

    print(f"\n--- Actual Updates ---")
    print(f"actual ΔA:\n{actual_dA}")
    print(f"actual ΔB:\n{actual_dB}")

    # Verify
    dA_error = torch.norm(actual_dA - expected_dA).item()
    dB_error = torch.norm(actual_dB - expected_dB).item()

    print(f"\n--- Verification ---")
    print(f"||actual_dA - expected_dA|| = {dA_error:.6e}")
    print(f"||actual_dB - expected_dB|| = {dB_error:.6e}")

    if dA_error < 1e-5 and dB_error < 1e-5:
        print("\n✓ Update matches expected formula exactly!")
    else:
        print("\n✗ Update does not match expected formula")

    print(f"\n--- Final State ---")
    print(f"A_after:\n{A_after}")
    print(f"B_after:\n{B_after}")


if __name__ == "__main__":
    test_single_update_detailed()
    print("\n\n")
    test_training_with_logging()
