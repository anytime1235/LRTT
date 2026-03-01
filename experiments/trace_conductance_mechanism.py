#!/usr/bin/env python
"""
Deep dive: How does analog tile convert gradient → conductance change?

Question: What determines the "reasonable" conductance change from 10^5 gradient?
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from sixt1c_config import gen_sixt1c_lora_config
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles import TorchInferenceTile

from peft import LoraConfig, get_peft_model

print("=" * 80)
print("TRACE: Conductance Update Mechanism")
print("=" * 80)

# ============================================================================
# 1. Setup simple analog layer for controlled test
# ============================================================================
print("\n[1] Creating standalone analog layer...")

rpu_config = gen_sixt1c_lora_config(
    dt_batch_sec=1.0,
    include_retention=False,
    test_mode=None
)

# Simple layer for testing
digital_layer = nn.Linear(8, 128, bias=False)
# Initialize with small values for testing
nn.init.normal_(digital_layer.weight, mean=0, std=0.01)

analog_layer = AnalogLinear.from_digital(
    digital_layer,
    rpu_config,
    tile_module_class=TorchInferenceTile
)

tile = list(analog_layer.analog_tiles())[0]

print(f"  Tile: {type(tile)}")
print(f"  Shape: {tile.out_size} x {tile.in_size}")

# Check tile's learning rate
if hasattr(tile, 'get_learning_rate'):
    try:
        tile_lr = tile.get_learning_rate()
        print(f"  Tile learning rate: {tile_lr}")
    except (TypeError, AttributeError) as e:
        print(f"  Tile learning rate: not set yet ({e})")
else:
    print(f"  No get_learning_rate() method")

# Check RPU config
print(f"\n  RPU Config:")
print(f"    inp_res: {rpu_config.forward.inp_res:.6f}")
print(f"    out_res: {rpu_config.forward.out_res:.6f}")
print(f"    is_perfect: {rpu_config.forward.is_perfect}")
print(f"    noise_management: {rpu_config.forward.noise_management}")

# ============================================================================
# 2. Manually test tile.update() with known inputs
# ============================================================================
print("\n[2] Manual tile update test...")

# Create simple test inputs
batch_size = 4
x_test = torch.randn(batch_size, 8) * 0.1  # Small input
d_test = torch.randn(batch_size, 128) * 1e4  # Large gradient ~10^4

print(f"\n  Test inputs:")
print(f"    x: shape={x_test.shape}, max_abs={x_test.abs().max().item():.4e}")
print(f"    d: shape={d_test.shape}, max_abs={d_test.abs().max().item():.4e}")

# Expected update (digital): w -= lr * (d.t() @ x) / batch
lr_test = 1e-4
expected_update_digital = -lr_test * (d_test.t() @ x_test) / batch_size
print(f"\n  Expected (digital SGD, lr={lr_test}):")
print(f"    update norm: {expected_update_digital.norm().item():.4e}")
print(f"    update max_abs: {expected_update_digital.abs().max().item():.4e}")

# Read weight before
w_before = tile.get_weights()[0].clone()
print(f"\n  Weight before:")
print(f"    norm: {w_before.norm().item():.4e}")
print(f"    max_abs: {w_before.abs().max().item():.4e}")

# Set tile learning rate explicitly
tile.set_learning_rate(lr_test)
print(f"\n  Set tile learning rate: {lr_test}")

# Call tile.update() directly
# Note: TorchInferenceTile's update computes: w -= lr * outer(x, d)
print(f"\n  Calling tile.update(x_test, d_test)...")

# Patch to capture pre_update processing
if hasattr(tile, 'pre_update'):
    x_processed, d_processed = tile.pre_update(x_test, 1, d_test, 1)
    print(f"\n  After pre_update:")
    print(f"    x_processed max_abs: {x_processed.abs().max().item():.4e}")
    print(f"    d_processed max_abs: {d_processed.abs().max().item():.4e}")

    # Check if they're different
    if not torch.equal(x_test, x_processed):
        print(f"    x was scaled by: {(x_processed.abs().max() / (x_test.abs().max() + 1e-12)).item():.4f}")
    if not torch.equal(d_test, d_processed):
        print(f"    d was scaled by: {(d_processed.abs().max() / (d_test.abs().max() + 1e-12)).item():.4f}")
else:
    print(f"  No pre_update() method")

# Actually update
tile.update(x_test, d_test)

# Read weight after
w_after = tile.get_weights()[0].clone()
w_delta = w_after - w_before

print(f"\n  Weight after:")
print(f"    norm: {w_after.norm().item():.4e}")
print(f"    max_abs: {w_after.abs().max().item():.4e}")

print(f"\n  Actual weight delta:")
print(f"    norm: {w_delta.norm().item():.4e}")
print(f"    max_abs: {w_delta.abs().max().item():.4e}")

# ============================================================================
# 3. Compare expected vs actual
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

ratio_norm = w_delta.norm().item() / (expected_update_digital.norm().item() + 1e-12)
ratio_max = w_delta.abs().max().item() / (expected_update_digital.abs().max().item() + 1e-12)

print(f"\n  Expected (digital SGD): norm={expected_update_digital.norm().item():.4e}")
print(f"  Actual (tile.update):   norm={w_delta.norm().item():.4e}")
print(f"  Ratio: {ratio_norm:.2f}x")

print(f"\n  Expected max_abs: {expected_update_digital.abs().max().item():.4e}")
print(f"  Actual max_abs:   {w_delta.abs().max().item():.4e}")
print(f"  Ratio: {ratio_max:.2f}x")

# ============================================================================
# 4. Hypothesis: batch accumulation vs averaging
# ============================================================================
print("\n" + "=" * 80)
print("HYPOTHESIS TEST: Batch Accumulation")
print("=" * 80)

# Hypothesis: tile.update() accumulates over batch without averaging
# Expected (no averaging): w -= lr * sum_over_batch(outer(x[i], d[i]))
expected_no_avg = -lr_test * (d_test.t() @ x_test)  # No division by batch_size

print(f"\n  Hypothesis: tile accumulates without dividing by batch_size")
print(f"    Expected (no avg): norm={expected_no_avg.norm().item():.4e}")
print(f"    Actual:            norm={w_delta.norm().item():.4e}")

ratio_no_avg = w_delta.norm().item() / (expected_no_avg.norm().item() + 1e-12)
print(f"    Ratio: {ratio_no_avg:.4f}x")

if abs(ratio_no_avg - 1.0) < 0.1:
    print(f"\n  ✓ CONFIRMED: Tile does NOT average over batch!")
    print(f"    Conductance update = -lr × Σ(d[i] ⊗ x[i])")
    print(f"    This is different from typical SGD which averages.")
else:
    print(f"\n  ✗ Still {ratio_no_avg:.2f}x difference, other factors involved")

# ============================================================================
# 5. Check mapping_scales effect
# ============================================================================
print("\n" + "=" * 80)
print("MAPPING_SCALES EFFECT")
print("=" * 80)

if hasattr(tile, 'mapping_scales') and tile.mapping_scales is not None:
    ms = tile.mapping_scales
    print(f"\n  mapping_scales:")
    print(f"    Shape: {ms.shape}")
    print(f"    Range: [{ms.min().item():.4e}, {ms.max().item():.4e}]")
    print(f"    Mean: {ms.mean().item():.4e}")

    # If mapping_scales != 1, gradient is divided by it
    if not torch.allclose(ms, torch.ones_like(ms)):
        print(f"\n  → Gradient d is divided by mapping_scales in pre_update()")
        print(f"    Effective scaling factor: 1/{ms.mean().item():.4e}")
    else:
        print(f"\n  → mapping_scales all 1.0, no scaling effect")
else:
    print(f"\n  No mapping_scales")

if hasattr(tile, 'mapping_lr_scale'):
    print(f"\n  mapping_lr_scale: {tile.mapping_lr_scale:.4e}")

# ============================================================================
# 6. Test with different learning rates
# ============================================================================
print("\n" + "=" * 80)
print("LEARNING RATE SENSITIVITY TEST")
print("=" * 80)

print(f"\nTesting if tile respects learning rate parameter...")

# Reset to known state
tile.set_weights(w_before, None)

# Test with different lr
lr_test2 = 1e-3  # 10x larger
tile.set_learning_rate(lr_test2)

w_before2 = tile.get_weights()[0].clone()
tile.update(x_test, d_test)
w_after2 = tile.get_weights()[0].clone()
w_delta2 = w_after2 - w_before2

print(f"\n  lr={lr_test}: delta norm = {w_delta.norm().item():.4e}")
print(f"  lr={lr_test2}: delta norm = {w_delta2.norm().item():.4e}")
print(f"  Ratio (should be ~10x): {(w_delta2.norm() / (w_delta.norm() + 1e-12)).item():.2f}x")

lr_sensitive = abs((w_delta2.norm() / (w_delta.norm() + 1e-12)).item() - 10.0) < 2.0
if lr_sensitive:
    print(f"\n  ✓ Tile respects learning rate parameter")
else:
    print(f"\n  ✗ Tile behavior not linear with learning rate")

# ============================================================================
# 7. Summary
# ============================================================================
print("\n" + "=" * 80)
print("CONDUCTANCE UPDATE MECHANISM")
print("=" * 80)

print(f"""
How analog tile converts gradient 10^5 to conductance change:

1. **Input scaling (pre_update)**:
   - x: scaled by input_range (if set)
   - d: divided by mapping_scales × mapping_lr_scale
   - For zero-init weights: mapping_scales=1.0 → no scaling

2. **Batch accumulation (NOT averaging)**:
   - Digital SGD: Δw = -lr × (1/batch) × Σ(d[i] ⊗ x[i])
   - Analog tile: Δw = -lr × Σ(d[i] ⊗ x[i])  [no 1/batch!]
   - This explains batch_size={batch_size} → {batch_size}x difference

3. **Learning rate**:
   - Tile respects lr parameter: Δw ∝ lr
   - But optimizer lr and tile lr can be different!

4. **Hardware constraints** (if not is_perfect):
   - inp_res, out_res: quantize inputs/outputs
   - noise_management: scale activations for range
   - But these don't directly scale gradient

5. **Result**:
   - Gradient d = 10^5
   - Learning rate lr = 1e-4
   - Batch size = {batch_size}
   - Update = lr × batch × d × x ≈ 1e-4 × {batch_size} × 10^5 × 0.1 = ~{batch_size}
   - Actual update max: {w_delta.abs().max().item():.2e}

The "reasonable" conductance change is determined by:
- Learning rate (controlled by optimizer)
- Batch accumulation (not averaged)
- Mapping scales (from weight normalization)
- NOT by hardware noise/quantization (those affect forward/inference)
""")

print("\n" + "=" * 80)
print("TRACE COMPLETE")
print("=" * 80)
