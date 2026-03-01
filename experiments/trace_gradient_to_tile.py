#!/usr/bin/env python
"""
Trace gradient values through to actual tile updates.

Question: If gradient is ~10^5, how does it translate to actual A/B tile weight updates?
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# Import sixt1c config
from sixt1c_config import gen_sixt1c_lora_config

# Import analog conversion utilities
from related_functions import (
    convert_to_analog_lora,
    find_lora_modules
)

print("=" * 80)
print("TRACE: Gradient → Tile Update")
print("=" * 80)

# ============================================================================
# 1. Setup: Load model and add sixt1c LoRA
# ============================================================================
print("\n[1] Loading MobileBERT with sixt1c LoRA...")

model = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased",
    num_labels=2
)

tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

# Add digital LoRA first
from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.0,  # No dropout for cleaner analysis
    target_modules=["query", "key", "value"]
)
model = get_peft_model(model, peft_config)

# Convert to analog
rpu_config = gen_sixt1c_lora_config(
    dt_batch_sec=1.0,
    include_retention=False,  # Disable retention for simplicity
    test_mode=None  # Use default: omega=1.0, backward hook active
)

print("  Converting to analog with sixt1c config...")
convert_to_analog_lora(
    model,
    rpu_config,
    transfer_lr=1.0,
    transfer_every=1000,  # Don't transfer during this test
    lora_alpha=32,
    correct_gradient_magnitudes=False  # Keep gradient as-is for tracing
)

# Find a target layer to trace
lora_modules = find_lora_modules(model)
target_layer = None
for name, module in lora_modules:
    if 'query' in name and 'layer.0' in name:  # First layer, query
        target_layer = (name, module)
        break

if target_layer is None:
    print("ERROR: Could not find target layer")
    sys.exit(1)

layer_name, layer_module = target_layer
print(f"  Target layer: {layer_name}")
print(f"  Layer type: {type(layer_module)}")

# Check if it has lrtt_controller
if not hasattr(layer_module, 'lrtt_controller'):
    print("ERROR: Layer does not have lrtt_controller")
    sys.exit(1)

controller = layer_module.lrtt_controller
print(f"  Controller: rank={controller.rank}, lora_alpha={controller.lora_alpha}")
print(f"  correct_gradient_magnitudes={controller.correct_gradient_magnitudes}")

# ============================================================================
# 2. Hook into lrtt_controller.ab_weight_update to capture values
# ============================================================================
print("\n[2] Installing hooks to capture update values...")

captured_data = {
    'step': 0,
    'x_input': None,      # Input to forward pass
    'd_output': None,     # Gradient from output (d)
    'XB': None,           # tile_b.forward(x) for A update
    'DA': None,           # tile_a.backward(d) for B update
    'lr_eff': None,       # Effective learning rate
    'A_before': None,     # A tile weights before update
    'B_before': None,     # B tile weights before update
    'A_after': None,      # A tile weights after update
    'B_after': None,      # B tile weights after update
}

# Patch ab_weight_update
original_ab_update = controller.ab_weight_update

def traced_ab_update(x, d, bias_grad=None, in_trans=False, out_trans=False, lr=1.0):
    """Wrapper to capture all intermediate values"""

    # Store inputs
    captured_data['x_input'] = x.detach().clone()
    captured_data['d_output'] = d.detach().clone()

    # Read A, B before update
    with torch.no_grad():
        I_rank = torch.eye(controller.rank, device=x.device, dtype=x.dtype)
        A_T = controller.tile_a.forward(I_rank)  # [rank, d_size]
        captured_data['A_before'] = A_T.t().clone()  # [d_size, rank]

        B_read = controller.tile_b.backward(I_rank)  # [rank, x_size]
        captured_data['B_before'] = B_read.clone()

    # Compute intermediate projections (mimics controller logic)
    with torch.no_grad():
        XB = controller.tile_b.forward(x)  # [batch, rank]
        DA = controller.tile_a.backward(d)  # [batch, rank]
        captured_data['XB'] = XB.clone()
        captured_data['DA'] = DA.clone()

        # Compute effective lr (from lrtt_controller.py:683-685)
        lr_eff = lr * controller.lora_alpha
        if controller.correct_gradient_magnitudes:
            lr_eff /= (controller.rank ** 0.5)
        captured_data['lr_eff'] = lr_eff

    # Call original update
    result = original_ab_update(x, d, bias_grad, in_trans, out_trans, lr)

    # Read A, B after update
    with torch.no_grad():
        I_rank = torch.eye(controller.rank, device=x.device, dtype=x.dtype)
        A_T = controller.tile_a.forward(I_rank)
        captured_data['A_after'] = A_T.t().clone()

        B_read = controller.tile_b.backward(I_rank)
        captured_data['B_after'] = B_read.clone()

    captured_data['step'] += 1
    return result

# Apply patch
controller.ab_weight_update = traced_ab_update

# ============================================================================
# 3. Run one training step
# ============================================================================
print("\n[3] Running one training step...")

# Load one example
dataset = load_dataset("glue", "rte", split="train[:1]")
example = dataset[0]

inputs = tokenizer(
    example["sentence1"],
    example["sentence2"],
    padding="max_length",
    max_length=128,
    truncation=True,
    return_tensors="pt"
)
labels = torch.tensor([example["label"]])

# Setup optimizer
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)  # Simple SGD for clarity

# Forward pass
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss

print(f"  Loss: {loss.item():.6f}")

# Backward pass
loss.backward()

# Check gradient on layer's weight (if accessible)
if hasattr(layer_module, 'weight') and layer_module.weight.grad is not None:
    grad_norm = layer_module.weight.grad.norm().item()
    grad_max = layer_module.weight.grad.abs().max().item()
    print(f"  Layer weight.grad: norm={grad_norm:.2e}, max={grad_max:.2e}")

# Optimizer step (this triggers tile updates via backward hooks)
optimizer.step()

# ============================================================================
# 4. Analyze captured data
# ============================================================================
print("\n" + "=" * 80)
print("CAPTURED DATA ANALYSIS")
print("=" * 80)

if captured_data['step'] == 0:
    print("\nWARNING: No update was captured! The layer may not have been updated.")
    print("This could happen if:")
    print("  - The layer has no gradient flow")
    print("  - The update hook was not triggered")
    sys.exit(1)

print(f"\nNumber of updates captured: {captured_data['step']}")

# Input/output dimensions
x_input = captured_data['x_input']
d_output = captured_data['d_output']
print(f"\n[Input x] shape={x_input.shape}")
print(f"  Range: [{x_input.min().item():.4e}, {x_input.max().item():.4e}]")
print(f"  Norm: {x_input.norm().item():.4e}")

print(f"\n[Output gradient d] shape={d_output.shape}")
print(f"  Range: [{d_output.min().item():.4e}, {d_output.max().item():.4e}]")
print(f"  Norm: {d_output.norm().item():.4e}")
print(f"  Max abs: {d_output.abs().max().item():.4e}")

# Projections for A/B updates
XB = captured_data['XB']
DA = captured_data['DA']
print(f"\n[XB = B·x] (for A update) shape={XB.shape}")
print(f"  Range: [{XB.min().item():.4e}, {XB.max().item():.4e}]")
print(f"  Max abs: {XB.abs().max().item():.4e}")

print(f"\n[DA = A^T·d] (for B update) shape={DA.shape}")
print(f"  Range: [{DA.min().item():.4e}, {DA.max().item():.4e}]")
print(f"  Max abs: {DA.abs().max().item():.4e}")

# Effective learning rate
lr_eff = captured_data['lr_eff']
print(f"\n[Effective LR] lr_eff = {lr_eff:.6e}")

# Weight changes
A_before = captured_data['A_before']
A_after = captured_data['A_after']
B_before = captured_data['B_before']
B_after = captured_data['B_after']

A_delta = A_after - A_before
B_delta = B_after - B_before

print(f"\n[A tile] shape={A_before.shape}")
print(f"  Before: norm={A_before.norm().item():.4e}, max_abs={A_before.abs().max().item():.4e}")
print(f"  After:  norm={A_after.norm().item():.4e}, max_abs={A_after.abs().max().item():.4e}")
print(f"  Delta:  norm={A_delta.norm().item():.4e}, max_abs={A_delta.abs().max().item():.4e}")

print(f"\n[B tile] shape={B_before.shape}")
print(f"  Before: norm={B_before.norm().item():.4e}, max_abs={B_before.abs().max().item():.4e}")
print(f"  After:  norm={B_after.norm().item():.4e}, max_abs={B_after.abs().max().item():.4e}")
print(f"  Delta:  norm={B_delta.norm().item():.4e}, max_abs={B_delta.abs().max().item():.4e}")

# ============================================================================
# 5. Expected vs Actual update
# ============================================================================
print("\n" + "=" * 80)
print("EXPECTED vs ACTUAL UPDATE")
print("=" * 80)

# Expected: ΔA = -lr_eff * d^T @ XB  (in controller: tile_a.update(XB, d))
# Expected: ΔB = -lr_eff * DA^T @ x  (in controller: tile_b.update(x, DA))

with torch.no_grad():
    # For A: ΔA = -lr * d^T @ XB
    # Note: tile.update(x, d) computes: w -= lr * outer(x, d)
    # So A_update computes: A -= lr * outer(XB, d)
    #   = A -= lr * XB @ d^T  [after transpose]
    # Let's compute expected change

    # tile_a.update(XB, d) with XB:[batch, rank], d:[batch, d_size]
    # Computes: A -= lr_eff * (XB^T @ d)  [internally transposed]
    # A is [d_size, rank], so: A -= lr_eff * d^T @ XB
    expected_A_delta = -lr_eff * (d_output.t() @ XB)  # [d_size, rank]

    # tile_b.update(x, DA) with x:[batch, x_size], DA:[batch, rank]
    # Computes: B -= lr_eff * (x^T @ DA)
    # B is [rank, x_size], so: B -= lr_eff * DA^T @ x
    expected_B_delta = -lr_eff * (DA.t() @ x_input)  # [rank, x_size]

    print(f"\n[Expected A delta (simple)]")
    print(f"  = -lr_eff * d^T @ XB")
    print(f"  = -{lr_eff:.4e} * ({d_output.shape} @ {XB.shape})")
    print(f"  Norm: {expected_A_delta.norm().item():.4e}")
    print(f"  Max abs: {expected_A_delta.abs().max().item():.4e}")

    print(f"\n[Actual A delta]")
    print(f"  Norm: {A_delta.norm().item():.4e}")
    print(f"  Max abs: {A_delta.abs().max().item():.4e}")

    print(f"\n[Ratio] actual/expected (A)")
    ratio_A_norm = A_delta.norm().item() / (expected_A_delta.norm().item() + 1e-12)
    ratio_A_max = A_delta.abs().max().item() / (expected_A_delta.abs().max().item() + 1e-12)
    print(f"  Norm ratio: {ratio_A_norm:.4f}")
    print(f"  Max ratio: {ratio_A_max:.4f}")

    print(f"\n[Expected B delta (simple)]")
    print(f"  = -lr_eff * DA^T @ x")
    print(f"  Norm: {expected_B_delta.norm().item():.4e}")
    print(f"  Max abs: {expected_B_delta.abs().max().item():.4e}")

    print(f"\n[Actual B delta]")
    print(f"  Norm: {B_delta.norm().item():.4e}")
    print(f"  Max abs: {B_delta.abs().max().item():.4e}")

    print(f"\n[Ratio] actual/expected (B)")
    ratio_B_norm = B_delta.norm().item() / (expected_B_delta.norm().item() + 1e-12)
    ratio_B_max = B_delta.abs().max().item() / (expected_B_delta.abs().max().item() + 1e-12)
    print(f"  Norm ratio: {ratio_B_norm:.4f}")
    print(f"  Max ratio: {ratio_B_max:.4f}")

# ============================================================================
# 6. Key Question: If d is ~10^5, why doesn't A explode?
# ============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)

print(f"""
If gradient d has values ~10^5, but actual weight update is reasonable,
this could be due to:

1. **Periphery scaling**: ABS_MAX noise_management scales inputs/outputs
   - Forward: input scaled by 1/max(|input|), output restored
   - Backward: gradient scaled similarly, but backward hook compensates
   - Check: mapping_scales on tiles

2. **Learning rate in optimizer**: SGD lr was {1e-4:.4e}
   - Even if d=10^5, with lr=1e-4, update = 10^5 * 1e-4 = 10

3. **Tile internal scaling**: Tiles may apply additional scaling
   - dw_min, weight range normalization

From captured data:
  - d max: {d_output.abs().max().item():.4e}
  - lr_eff: {lr_eff:.6e}
  - Expected update scale: {(d_output.abs().max().item() * lr_eff):.4e}
  - Actual A delta max: {A_delta.abs().max().item():.4e}
  - Actual B delta max: {B_delta.abs().max().item():.4e}

The ratio actual/expected shows how much additional scaling happens inside tiles.
""")

# Check if tiles have mapping_scales
if hasattr(controller.tile_a, 'mapping_scales'):
    if controller.tile_a.mapping_scales is not None:
        print(f"\nTile A mapping_scales: {controller.tile_a.mapping_scales}")
    else:
        print(f"\nTile A mapping_scales: None")
else:
    print(f"\nTile A has no mapping_scales attribute")

if hasattr(controller.tile_b, 'mapping_scales'):
    if controller.tile_b.mapping_scales is not None:
        print(f"Tile B mapping_scales: {controller.tile_b.mapping_scales}")
    else:
        print(f"Tile B mapping_scales: None")
else:
    print(f"Tile B has no mapping_scales attribute")

print("\n" + "=" * 80)
print("TRACE COMPLETE")
print("=" * 80)
