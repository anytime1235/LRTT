"""
Test 2: A/B Tile Update Equivalence

Verifies that PEFT LoRA and FP-LoRA produce identical weight updates
after one training step.

Expected:
- A/B weight changes match within 1e-6
- C weights unchanged (frozen)
"""

import sys
import torch
import torch.nn as nn

# Add paths
sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
from aihwkit.optim import AnalogSGD
from lrtt_lora_config import create_lrtt_lora_config


def create_peft_lora_layer(in_features, out_features, rank, lora_alpha):
    """Create a PEFT-style LoRA layer manually (using PyTorch)."""
    # Base layer (frozen)
    base_layer = nn.Linear(in_features, out_features, bias=False)

    # LoRA adapters
    lora_A = nn.Linear(in_features, rank, bias=False)  # [rank, in_features]
    lora_B = nn.Linear(rank, out_features, bias=False)  # [out_features, rank]

    return base_layer, lora_A, lora_B


def create_fplora_layer(in_features, out_features, rank, lora_alpha):
    """Create FP-LoRA layer using FloatingPoint devices."""
    # Create FP-LoRA config
    config = create_lrtt_lora_config(
        rank=rank,
        lora_alpha=lora_alpha,
        output_noise_level=0.0,
        use_floating_point=True,  # CRITICAL: FloatingPoint mode for exact arithmetic
    )

    # Create AnalogLinear layer
    analog_layer = AnalogLinear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        rpu_config=config,
    )

    return analog_layer


def set_identical_weights(base_layer, lora_A, lora_B, analog_layer):
    """Initialize PEFT and FP-LoRA with identical weights."""
    # Generate random weights
    W = torch.randn(base_layer.out_features, base_layer.in_features) * 0.02
    A = torch.randn(lora_A.out_features, lora_A.in_features) * 0.02
    B = torch.randn(lora_B.out_features, lora_B.in_features) * 0.02

    # Set PEFT weights
    base_layer.weight.data.copy_(W)
    lora_A.weight.data.copy_(A)
    lora_B.weight.data.copy_(B)

    # Freeze base layer
    base_layer.weight.requires_grad = False

    # Set FP-LoRA weights
    # Extract the LRTT tile
    tile = analog_layer.analog_module
    if not isinstance(tile, LRTTSimulatorTile):
        from aihwkit.simulator.tiles.array import TileModuleArray
        if isinstance(tile, TileModuleArray):
            tile = tile.array[0][0]

    # Set tile weights: C=W, tile_a=lora_B, tile_b=lora_A
    # Note: In LRTT, tile_b is applied first (input→rank), tile_a second (rank→output)
    # In PEFT: lora_A is applied first (input→rank), lora_B second (rank→output)
    # So: tile_a ← lora_B, tile_b ← lora_A
    tile.tile_c.set_weights(W.clone())  # C tile (frozen, pretrained)
    tile.tile_a.set_weights(B.clone())  # A tile = lora_B (rank → output)
    tile.tile_b.set_weights(A.clone())  # B tile = lora_A (input → rank)

    return W, A, B


def peft_forward(x, base_layer, lora_A, lora_B, lora_alpha):
    """Manual PEFT LoRA forward: y = W·x + α·B·(A·x)

    In standard LoRA:
    - lora_A: [rank, in_features] - applied first (input → rank)
    - lora_B: [out_features, rank] - applied second (rank → output)
    """
    # Base output: W·x
    y_base = base_layer(x)

    # LoRA path: α·B·(A·x)
    g = lora_A(x)  # A·x: [batch, in_features] → [batch, rank]
    y_lora = lora_B(g)  # B·(A·x): [batch, rank] → [batch, out_features]
    y_lora = lora_alpha * y_lora

    # Combined output
    y = y_base + y_lora
    return y


def main():
    print("=" * 80)
    print("TEST 2: A/B TILE UPDATE EQUIVALENCE")
    print("=" * 80)
    print()

    # Configuration
    in_features = 256
    out_features = 128
    rank = 8
    lora_alpha = 1.0
    batch_size = 4
    lr = 0.01

    print(f"Configuration:")
    print(f"  in_features: {in_features}")
    print(f"  out_features: {out_features}")
    print(f"  rank: {rank}")
    print(f"  lora_alpha: {lora_alpha}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {lr}")
    print()

    # Step 1: Create layers
    print("[1/8] Creating PEFT and FP-LoRA layers...", end=" ")
    base_layer, lora_A, lora_B = create_peft_lora_layer(in_features, out_features, rank, lora_alpha)
    analog_layer = create_fplora_layer(in_features, out_features, rank, lora_alpha)
    print("✓")

    # Step 2: Initialize with identical weights
    print("[2/8] Initializing with identical weights...", end=" ")
    W, A, B = set_identical_weights(base_layer, lora_A, lora_B, analog_layer)

    # Enable training mode for analog layer
    analog_layer.train()

    print("✓")

    # Step 3: Capture initial weights
    print("[3/8] Capturing initial weights...", end=" ")
    A_before = lora_A.weight.data.clone()
    B_before = lora_B.weight.data.clone()
    W_before = base_layer.weight.data.clone()

    tile = analog_layer.analog_module
    if not isinstance(tile, LRTTSimulatorTile):
        from aihwkit.simulator.tiles.array import TileModuleArray
        if isinstance(tile, TileModuleArray):
            tile = tile.array[0][0]

    A_before_fplora, _ = tile.tile_a.get_weights()
    B_before_fplora, _ = tile.tile_b.get_weights()
    W_before_fplora, _ = tile.tile_c.get_weights()
    A_before_fplora = A_before_fplora.clone()
    B_before_fplora = B_before_fplora.clone()
    W_before_fplora = W_before_fplora.clone()
    print("✓")

    # Step 4: Create optimizers
    print("[4/8] Creating optimizers...", end=" ")
    # PEFT: Standard SGD
    optimizer_peft = torch.optim.SGD([lora_A.weight, lora_B.weight], lr=lr)

    # FP-LoRA: AnalogSGD
    optimizer_fplora = AnalogSGD(analog_layer.parameters(), lr=lr)
    optimizer_fplora.regroup_param_groups(analog_layer)  # CRITICAL: Regroup for analog tiles
    print("✓")

    # Step 5: Run training step (PEFT)
    print("[5/8] Running PEFT training step...", end=" ")
    x = torch.randn(batch_size, in_features)
    target = torch.randn(batch_size, out_features)

    # Forward
    y_peft = peft_forward(x, base_layer, lora_A, lora_B, lora_alpha)
    loss_peft = ((y_peft - target) ** 2).mean()

    # Backward
    optimizer_peft.zero_grad()
    loss_peft.backward()

    # Step
    optimizer_peft.step()
    print("✓")

    # Step 6: Run training step (FP-LoRA)
    print("[6/8] Running FP-LoRA training step...", end=" ")
    # Use same input and target (requires_grad for input)
    x_fplora = x.clone().requires_grad_(True)
    target_fplora = target.clone()

    # Forward
    y_fplora = analog_layer(x_fplora)
    loss_fplora = ((y_fplora - target_fplora) ** 2).mean()

    # Backward
    optimizer_fplora.zero_grad()
    loss_fplora.backward()

    # Step
    optimizer_fplora.step()
    print("✓")

    # Step 7: Extract updated weights
    print("[7/8] Extracting weight changes...", end=" ")
    # PEFT
    A_after_peft = lora_A.weight.data.clone()
    B_after_peft = lora_B.weight.data.clone()
    W_after_peft = base_layer.weight.data.clone()

    # FP-LoRA
    A_after_fplora, _ = tile.tile_a.get_weights()
    B_after_fplora, _ = tile.tile_b.get_weights()
    W_after_fplora, _ = tile.tile_c.get_weights()

    # Compute deltas
    delta_A_peft = A_after_peft - A_before
    delta_B_peft = B_after_peft - B_before
    delta_W_peft = W_after_peft - W_before

    delta_A_fplora = A_after_fplora - A_before_fplora
    delta_B_fplora = B_after_fplora - B_before_fplora
    delta_W_fplora = W_after_fplora - W_before_fplora
    print("✓")

    # Step 8: Compare weight changes
    print("[8/8] Comparing weight changes...", end=" ")
    # Remember: tile_a ← lora_B, tile_b ← lora_A
    A_matches = torch.allclose(delta_A_peft, delta_B_fplora, atol=1e-6)  # lora_A vs tile_b
    B_matches = torch.allclose(delta_B_peft, delta_A_fplora, atol=1e-6)  # lora_B vs tile_a
    W_frozen = torch.allclose(W_after_fplora, W_before_fplora, atol=1e-9)

    max_diff_A = (delta_A_peft - delta_B_fplora).abs().max().item()  # lora_A vs tile_b
    max_diff_B = (delta_B_peft - delta_A_fplora).abs().max().item()  # lora_B vs tile_a
    max_diff_W = (W_after_fplora - W_before_fplora).abs().max().item()
    print("✓")

    print()
    print("-" * 80)
    print("RESULTS:")
    print("-" * 80)
    print(f"  Loss (PEFT): {loss_peft.item():.6f}")
    print(f"  Loss (FP-LoRA): {loss_fplora.item():.6f}")
    print()
    print(f"  Δ(lora_A vs tile_b) max difference: {max_diff_A:.3e}")
    print(f"  Δ(lora_A vs tile_b) match (atol=1e-6): {A_matches}")
    print()
    print(f"  Δ(lora_B vs tile_a) max difference: {max_diff_B:.3e}")
    print(f"  Δ(lora_B vs tile_a) match (atol=1e-6): {B_matches}")
    print()
    print(f"  ΔW (tile_c) max change: {max_diff_W:.3e}")
    print(f"  W frozen (no change): {W_frozen}")
    print()

    # Debug info if mismatch
    if not (A_matches and B_matches):
        print("DEBUG INFO:")
        print(f"  Δ(lora_A) PEFT (first 5x5):\n{delta_A_peft[:5, :5]}")
        print(f"  Δ(tile_b) FP-LoRA (first 5x5):\n{delta_B_fplora[:5, :5]}")
        print()
        print(f"  Δ(lora_B) PEFT (first 5x5):\n{delta_B_peft[:5, :5]}")
        print(f"  Δ(tile_a) FP-LoRA (first 5x5):\n{delta_A_fplora[:5, :5]}")
        print()

    if not W_frozen:
        print("WARNING: C tile (W) was NOT frozen!")
        print(f"  ΔW max change: {max_diff_W:.3e}")
        print()

    # Final result
    all_pass = A_matches and B_matches and W_frozen
    if all_pass:
        print("✓ TEST PASSED: Weight updates are identical")
        print()
        print("=" * 80)
        return 0
    else:
        print("✗ TEST FAILED: Weight updates differ")
        print()
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
