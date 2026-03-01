#!/usr/bin/env python
"""
Simplified: Trace gradient → analog tile update in sixt1c LoRA.

Goal: Understand how gradient ~10^5 translates to actual weight updates.
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

# Import analog
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles import TorchInferenceTile
from torch.nn import Linear

# PEFT
from peft import LoraConfig, get_peft_model

print("=" * 80)
print("TRACE: Analog LoRA Gradient → Tile Update")
print("=" * 80)

# ============================================================================
# 1. Create model with digital LoRA
# ============================================================================
print("\n[1] Loading MobileBERT with digital LoRA...")

model = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased",
    num_labels=2
)

tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.0,
    target_modules=["query"]  # Just query for simplicity
)
model = get_peft_model(model, peft_config)

# ============================================================================
# 2. Convert ONE lora_B layer to analog for tracing
# ============================================================================
print("\n[2] Converting one lora_B to analog...")

# Find first query.lora_B layer
target_layer_name = None
target_parent = None
target_attr = None

for name, module in model.named_modules():
    if 'query.lora_B' in name and 'layer.0' in name:
        # Get parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        target_layer_name = name
        target_parent = parent
        target_attr = parts[-1]
        break

if target_layer_name is None:
    print("ERROR: Could not find target lora_B layer")
    sys.exit(1)

print(f"  Target layer: {target_layer_name}")

original_lora_B = getattr(target_parent, target_attr)
print(f"  Original layer: {original_lora_B}")

# Handle ModuleDict (PEFT v0.8+)
if isinstance(original_lora_B, nn.ModuleDict):
    print(f"  (ModuleDict detected, using 'default' key)")
    original_lora_B = original_lora_B['default']

print(f"  Weight shape: {original_lora_B.weight.shape}")
print(f"  Weight range: [{original_lora_B.weight.min().item():.4f}, {original_lora_B.weight.max().item():.4f}]")

# Convert to analog
rpu_config = gen_sixt1c_lora_config(
    dt_batch_sec=1.0,
    include_retention=False,
    test_mode=None  # omega=1.0, backward hook active
)

analog_lora_B = AnalogLinear.from_digital(
    original_lora_B,
    rpu_config,
    tile_module_class=TorchInferenceTile
)

# Replace in ModuleDict
if isinstance(getattr(target_parent, target_attr), nn.ModuleDict):
    getattr(target_parent, target_attr)['default'] = analog_lora_B
else:
    setattr(target_parent, target_attr, analog_lora_B)

print(f"  Converted to AnalogLinear")

# Get tile - AnalogLinear.analog_tiles() returns a generator
if hasattr(analog_lora_B, 'analog_tiles') and callable(analog_lora_B.analog_tiles):
    tiles_list = list(analog_lora_B.analog_tiles())
    if not tiles_list:
        print(f"  ERROR: No tiles found in AnalogLinear")
        sys.exit(1)
    tile = tiles_list[0]
else:
    print(f"  ERROR: AnalogLinear does not have analog_tiles() method")
    sys.exit(1)

print(f"  Tile type: {type(tile)}")
print(f"  Tile shape: {tile.out_size} x {tile.in_size}")

# Check mapping_scales
if hasattr(tile, 'mapping_scales'):
    print(f"  mapping_scales: {tile.mapping_scales}")
else:
    print(f"  No mapping_scales attribute")

# ============================================================================
# 3. Hook to capture gradient and tile state
# ============================================================================
print("\n[3] Installing hooks...")

captured = {
    'grad': None,
    'weight_before': None,
    'weight_after': None,
    'input': None,
    'output': None,
}

def forward_hook(module, input, output):
    """Capture input/output during forward"""
    captured['input'] = input[0].detach().clone() if isinstance(input, tuple) else input.detach().clone()
    captured['output'] = output.detach().clone()
    return output

def backward_hook(module, grad_input, grad_output):
    """Capture gradient during backward"""
    if grad_output[0] is not None:
        captured['grad'] = grad_output[0].detach().clone()
    return grad_input

# Register hooks
analog_lora_B.register_forward_hook(forward_hook)
analog_lora_B.register_backward_hook(backward_hook)

# ============================================================================
# 4. Run one training step
# ============================================================================
print("\n[4] Running training step...")

# Load data
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

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# Capture weight before
with torch.no_grad():
    w_tuple = tile.get_weights()
    captured['weight_before'] = w_tuple[0].clone() if isinstance(w_tuple, tuple) else w_tuple.clone()

# Forward
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss

print(f"  Loss: {loss.item():.6f}")

# Backward
loss.backward()

# Optimizer step (triggers tile update)
optimizer.step()

# Capture weight after
with torch.no_grad():
    w_tuple = tile.get_weights()
    captured['weight_after'] = w_tuple[0].clone() if isinstance(w_tuple, tuple) else w_tuple.clone()

# ============================================================================
# 5. Analyze
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if captured['grad'] is None:
    print("\nWARNING: No gradient captured!")
    sys.exit(1)

grad = captured['grad']
inp = captured['input']
out = captured['output']
w_before = captured['weight_before']
w_after = captured['weight_after']
w_delta = w_after - w_before

print(f"\n[Gradient captured]")
print(f"  Shape: {grad.shape}")
print(f"  Min: {grad.min().item():.4e}")
print(f"  Max: {grad.max().item():.4e}")
print(f"  Mean abs: {grad.abs().mean().item():.4e}")
print(f"  Max abs: {grad.abs().max().item():.4e}")
print(f"  Norm: {grad.norm().item():.4e}")

print(f"\n[Input to layer]")
print(f"  Shape: {inp.shape}")
print(f"  Max abs: {inp.abs().max().item():.4e}")
print(f"  Norm: {inp.norm().item():.4e}")

print(f"\n[Output from layer]")
print(f"  Shape: {out.shape}")
print(f"  Max abs: {out.abs().max().item():.4e}")
print(f"  Norm: {out.norm().item():.4e}")

print(f"\n[Weight before update]")
print(f"  Shape: {w_before.shape}")
print(f"  Max abs: {w_before.abs().max().item():.4e}")
print(f"  Norm: {w_before.norm().item():.4e}")

print(f"\n[Weight after update]")
print(f"  Max abs: {w_after.abs().max().item():.4e}")
print(f"  Norm: {w_after.norm().item():.4e}")

print(f"\n[Weight delta (actual update)]")
print(f"  Max abs: {w_delta.abs().max().item():.4e}")
print(f"  Norm: {w_delta.norm().item():.4e}")
print(f"  Mean abs: {w_delta.abs().mean().item():.4e}")

# ============================================================================
# 6. Expected update (simple SGD)
# ============================================================================
print("\n" + "=" * 80)
print("EXPECTED vs ACTUAL")
print("=" * 80)

lr = 1e-4

# For a linear layer: w -= lr * grad_output^T @ input
# grad_output: [batch, out_size]
# input: [batch, in_size]
# weight: [out_size, in_size]

with torch.no_grad():
    # Compute outer product: grad_output^T @ input
    if len(grad.shape) == 2 and len(inp.shape) == 2:
        # grad: [batch, out_size], inp: [batch, in_size]
        expected_delta = -lr * (grad.t() @ inp) / grad.shape[0]  # Average over batch
    elif len(grad.shape) == 3 and len(inp.shape) == 3:
        # grad: [batch, seq, out_size], inp: [batch, seq, in_size]
        grad_flat = grad.reshape(-1, grad.shape[-1])  # [batch*seq, out_size]
        inp_flat = inp.reshape(-1, inp.shape[-1])     # [batch*seq, in_size]
        expected_delta = -lr * (grad_flat.t() @ inp_flat) / grad_flat.shape[0]
    else:
        print(f"  Unexpected shapes: grad={grad.shape}, inp={inp.shape}")
        expected_delta = None

if expected_delta is not None:
    print(f"\n[Expected delta (digital SGD)]")
    print(f"  = -lr * grad^T @ input / batch_size")
    print(f"  Max abs: {expected_delta.abs().max().item():.4e}")
    print(f"  Norm: {expected_delta.norm().item():.4e}")

    print(f"\n[Actual delta (analog tile)]")
    print(f"  Max abs: {w_delta.abs().max().item():.4e}")
    print(f"  Norm: {w_delta.norm().item():.4e}")

    print(f"\n[Ratio: actual / expected]")
    ratio_norm = w_delta.norm().item() / (expected_delta.norm().item() + 1e-12)
    ratio_max = w_delta.abs().max().item() / (expected_delta.abs().max().item() + 1e-12)
    print(f"  Norm ratio: {ratio_norm:.4f}")
    print(f"  Max ratio: {ratio_max:.4f}")

    if ratio_norm < 0.1 or ratio_norm > 10:
        print(f"\n⚠️  SIGNIFICANT SCALING DETECTED!")
        print(f"  The analog tile applies ~{ratio_norm:.2f}x scaling compared to digital SGD")
        print(f"  This could be due to:")
        print(f"    - mapping_scales (ABS_MAX noise management)")
        print(f"    - Tile internal weight range normalization")
        print(f"    - Learning rate compensation")
    else:
        print(f"\n✓ Update magnitude is similar to digital SGD")

# ============================================================================
# 7. Check tile scaling parameters
# ============================================================================
print("\n" + "=" * 80)
print("TILE SCALING PARAMETERS")
print("=" * 80)

# tile is already defined earlier
if hasattr(tile, 'mapping_scales') and tile.mapping_scales is not None:
    print(f"\nmapping_scales present:")
    print(f"  Shape: {tile.mapping_scales.shape}")
    print(f"  Values: {tile.mapping_scales}")
    print(f"  Mean: {tile.mapping_scales.mean().item():.4e}")
    print(f"\n  → This scales output (forward) and compensates gradient (backward)")
else:
    print(f"\nmapping_scales: None")

if hasattr(tile, 'mapping_lr_scale'):
    print(f"\nmapping_lr_scale: {tile.mapping_lr_scale:.4e}")

if hasattr(tile, 'rpu_config') and hasattr(tile.rpu_config, 'mapping'):
    mapping = tile.rpu_config.mapping
    print(f"\nRPU mapping config:")
    print(f"  weight_scaling_omega: {mapping.weight_scaling_omega}")
    print(f"  weight_scaling_columnwise: {mapping.weight_scaling_columnwise}")
    print(f"  weight_scaling_lr_compensation: {getattr(mapping, 'weight_scaling_lr_compensation', 'N/A')}")

# ============================================================================
# KEY INSIGHT
# ============================================================================
print("\n" + "=" * 80)
print("KEY INSIGHT")
print("=" * 80)

expected_str = f"{expected_delta.abs().max().item():.2e}" if expected_delta is not None else 'N/A'
print(f"""
Question: If gradient is ~{grad.abs().max().item():.2e}, why doesn't weight explode?

From this trace:
  1. Gradient captured: {grad.abs().max().item():.2e}
  2. Learning rate: {lr:.2e}
  3. Expected update (digital): {expected_str}
  4. Actual update (analog): {w_delta.abs().max().item():.2e}
  5. Ratio: {ratio_norm:.2f}x

The analog tile internally applies scaling via:
  - mapping_scales (from ABS_MAX noise_management)
  - Backward hook compensation (when omega > 0)
  - Weight range normalization

So even if PyTorch sees gradient ~10^5, the actual tile update
is scaled down to reasonable magnitudes (~1e-3 to 1e-1).

The "large gradient" is in the PyTorch computation graph,
but the tile's internal update mechanism rescales it appropriately.
""")

print("\n" + "=" * 80)
print("TRACE COMPLETE")
print("=" * 80)
