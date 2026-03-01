"""
Test 1: Forward Output Equivalence

Verifies that PEFT LoRA and FP-LoRA produce identical forward outputs
when initialized with the same weights.

Expected: torch.allclose(y_peft, y_fplora, atol=1e-6) = True
"""

import sys
import torch
import torch.nn as nn

# Add paths
sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
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
    print("TEST 1: FORWARD OUTPUT EQUIVALENCE")
    print("=" * 80)
    print()

    # Configuration
    in_features = 256
    out_features = 128
    rank = 8
    lora_alpha = 1.0
    batch_size = 4

    print(f"Configuration:")
    print(f"  in_features: {in_features}")
    print(f"  out_features: {out_features}")
    print(f"  rank: {rank}")
    print(f"  lora_alpha: {lora_alpha}")
    print(f"  batch_size: {batch_size}")
    print()

    # Step 1: Create PEFT LoRA layer
    print("[1/5] Creating PEFT LoRA layer...", end=" ")
    base_layer, lora_A, lora_B = create_peft_lora_layer(in_features, out_features, rank, lora_alpha)
    print("✓")

    # Step 2: Create FP-LoRA layer
    print("[2/5] Creating FP-LoRA layer...", end=" ")
    analog_layer = create_fplora_layer(in_features, out_features, rank, lora_alpha)
    print("✓")

    # Step 3: Initialize with identical weights
    print("[3/5] Initializing with identical weights...", end=" ")
    W, A, B = set_identical_weights(base_layer, lora_A, lora_B, analog_layer)
    print("✓")

    # Verify weights match
    tile = analog_layer.analog_module
    if not isinstance(tile, LRTTSimulatorTile):
        from aihwkit.simulator.tiles.array import TileModuleArray
        if isinstance(tile, TileModuleArray):
            tile = tile.array[0][0]

    W_fplora, _ = tile.tile_c.get_weights()
    tile_a_weights, _ = tile.tile_a.get_weights()  # Should match lora_B
    tile_b_weights, _ = tile.tile_b.get_weights()  # Should match lora_A

    assert torch.allclose(W, W_fplora, atol=1e-8), "C weights don't match!"
    assert torch.allclose(B, tile_a_weights, atol=1e-8), "tile_a (lora_B) weights don't match!"
    assert torch.allclose(A, tile_b_weights, atol=1e-8), "tile_b (lora_A) weights don't match!"
    print(f"  Weight verification: C, tile_a (lora_B), tile_b (lora_A) all match ✓")

    # Step 4: Run forward pass
    print("[4/5] Running forward pass...", end=" ")
    x = torch.randn(batch_size, in_features)

    # PEFT forward
    y_peft = peft_forward(x, base_layer, lora_A, lora_B, lora_alpha)

    # FP-LoRA forward
    y_fplora = analog_layer(x)

    print("✓")

    # Step 5: Compare outputs
    print("[5/5] Comparing outputs...", end=" ")
    matches = torch.allclose(y_peft, y_fplora, atol=1e-6)
    max_diff = (y_peft - y_fplora).abs().max().item()
    print("✓")

    print()
    print("-" * 80)
    print("RESULTS:")
    print("-" * 80)
    print(f"  PEFT output shape: {y_peft.shape}")
    print(f"  FP-LoRA output shape: {y_fplora.shape}")
    print(f"  Max difference: {max_diff:.3e}")
    print(f"  torch.allclose(atol=1e-6): {matches}")
    print()

    # Debug info if mismatch
    if not matches:
        print("DEBUG INFO:")
        print(f"  PEFT output (first 5): {y_peft[0, :5]}")
        print(f"  FP-LoRA output (first 5): {y_fplora[0, :5]}")
        print(f"  Difference (first 5): {(y_peft - y_fplora)[0, :5]}")
        print()

        # Check intermediate values
        print("Checking intermediate computations...")
        g_peft = lora_A(x)  # A·x (intermediate activation)

        # Get FP-LoRA intermediate (B·x)
        tile = analog_layer.analog_module
        if not isinstance(tile, LRTTSimulatorTile):
            from aihwkit.simulator.tiles.array import TileModuleArray
            if isinstance(tile, TileModuleArray):
                tile = tile.array[0][0]

        print(f"  forward_inject: {tile.forward_inject}")
        print(f"  lora_alpha: {tile.lora_alpha}")
        print()

    # Final result
    if matches:
        print("✓ TEST PASSED: Forward outputs are identical")
        print()
        print("=" * 80)
        return 0
    else:
        print("✗ TEST FAILED: Forward outputs differ")
        print()
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
