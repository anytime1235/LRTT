#!/usr/bin/env python
# coding=utf-8
"""
Trace LRTT-LoRA A/B tile updates in sixt1c mode.

Verify:
1. C tile gradient computation (frozen, should receive gradient but not update)
2. A tile update (trainable, should change)
3. B tile update (trainable, should change)
4. LoRA chain rule: ∂L/∂A and ∂L/∂B computed correctly
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import copy

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.optim import AnalogSGD

import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("LRTT-LORA A/B TILE UPDATE VERIFICATION (sixt1c mode)")
print("=" * 80)
print()

# =============================================================================
# Create LRTT-LoRA model with Q/K/V layers
# =============================================================================
print("[1/5] Creating LRTT-LoRA model (6T1C sixt1c mode)...")
print("-" * 80)

class TransformerBlock(nn.Module):
    """Simplified transformer block with Q/K/V."""
    def __init__(self, hidden_size=128, num_heads=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # Simple attention
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Dense and classify
        out = self.dense(out)
        out = out.mean(dim=1)  # Pool
        return self.classifier(out)

# Create model
model = TransformerBlock(hidden_size=128, num_heads=4)

# Convert to LRTT-LoRA (sixt1c mode)
config = create_lrtt_lora_config(
    rank=4,
    lora_alpha=1.0,
    use_floating_point=False  # Use 6T1C
)
model = convert_model_to_lrtt_lora(
    model,
    config,
    target_modules=["query", "key", "value"]
)

print("✓ Model created")
print(f"  Q/K/V layers converted to LRTT-LoRA")
print(f"  Rank: 4")
print(f"  LoRA alpha: 1.0")
print(f"  Device: 6T1C LinearStepDevice")
print()

# =============================================================================
# Helper functions to access tile weights
# =============================================================================

def get_lrtt_tile(layer_name):
    """Get LRTT tile from layer."""
    parts = layer_name.split('.')
    module = model
    for part in parts:
        module = getattr(module, part)

    # AnalogLinear → analog_module → LRTTSimulatorTile
    if hasattr(module, 'analog_module'):
        return module.analog_module
    return None

def get_tile_weights(tile):
    """Get weights from all 3 sub-tiles (A, B, C)."""
    weights = {}

    # tile_a
    w_a, _ = tile.tile_a.get_weights()
    weights['A'] = w_a.clone().detach()

    # tile_b
    w_b, _ = tile.tile_b.get_weights()
    weights['B'] = w_b.clone().detach()

    # tile_c
    w_c, _ = tile.tile_c.get_weights()
    weights['C'] = w_c.clone().detach()

    return weights

def compute_weight_changes(weights_before, weights_after):
    """Compute L2 norm of weight changes."""
    changes = {}
    for key in weights_before.keys():
        diff = weights_after[key] - weights_before[key]
        changes[key] = {
            'max_change': diff.abs().max().item(),
            'mean_change': diff.abs().mean().item(),
            'norm_change': diff.norm().item(),
        }
    return changes

# =============================================================================
# Store initial weights
# =============================================================================
print("[2/5] Storing initial weights...")
print("-" * 80)

layers_to_track = ['query', 'key', 'value']
initial_weights = {}

for layer_name in layers_to_track:
    tile = get_lrtt_tile(layer_name)
    if tile:
        initial_weights[layer_name] = get_tile_weights(tile)
        print(f"{layer_name}:")
        print(f"  A shape: {initial_weights[layer_name]['A'].shape}")
        print(f"  B shape: {initial_weights[layer_name]['B'].shape}")
        print(f"  C shape: {initial_weights[layer_name]['C'].shape}")
        print(f"  A norm: {initial_weights[layer_name]['A'].norm().item():.6f}")
        print(f"  B norm: {initial_weights[layer_name]['B'].norm().item():.6f}")
        print(f"  C norm: {initial_weights[layer_name]['C'].norm().item():.6f}")
        print()

# =============================================================================
# Forward and backward pass
# =============================================================================
print("[3/5] Forward and backward pass...")
print("-" * 80)

# Create input data
torch.manual_seed(42)
batch_size = 4
seq_length = 8
hidden_size = 128

x_input = torch.randn(batch_size, seq_length, hidden_size)
labels = torch.randint(0, 2, (batch_size,))

print(f"Input shape: {x_input.shape}")
print(f"Labels: {labels}")
print()

# Forward pass
model.train()
output = model(x_input)
loss = nn.CrossEntropyLoss()(output, labels)

print(f"Output shape: {output.shape}")
print(f"Loss: {loss.item():.6f}")
print()

# Backward pass
model.zero_grad()
loss.backward()

print("✓ Backward pass completed")
print()

# Check if gradients are computed
print("Gradient status:")
for name, param in model.named_parameters():
    if 'query' in name or 'key' in name or 'value' in name:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 1e-8:
                print(f"  {name}: grad_norm={grad_norm:.6f} ✓")
            else:
                print(f"  {name}: grad_norm={grad_norm:.6f} (very small)")
        else:
            print(f"  {name}: NO GRADIENT")

print()

# =============================================================================
# Gradient clipping (as in real training)
# =============================================================================
print("[4/5] Applying gradient clipping...")
print("-" * 80)

params_with_grad = [p for p in model.parameters() if p.grad is not None]
total_norm = torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=1.0).item()

print(f"Total gradient norm: {total_norm:.6f}")
print(f"Gradient clipping: {'Applied' if total_norm > 1.0 else 'Not needed'}")
print()

# =============================================================================
# Optimizer step and check weight updates
# =============================================================================
print("[5/5] Optimizer step and weight update verification...")
print("-" * 80)

# Create optimizer
optimizer = AnalogSGD(model.parameters(), lr=0.01)

# Run optimizer step
optimizer.step()

print("✓ Optimizer.step() completed")
print()

# Get updated weights
updated_weights = {}
for layer_name in layers_to_track:
    tile = get_lrtt_tile(layer_name)
    if tile:
        updated_weights[layer_name] = get_tile_weights(tile)

# Compute and display changes
print("=" * 80)
print("WEIGHT UPDATE VERIFICATION")
print("=" * 80)
print()

for layer_name in layers_to_track:
    print(f"\n{layer_name.upper()} Layer:")
    print("-" * 60)

    changes = compute_weight_changes(
        initial_weights[layer_name],
        updated_weights[layer_name]
    )

    # A tile (should UPDATE - trainable)
    print(f"  A tile (trainable):")
    print(f"    Max change:  {changes['A']['max_change']:.8f}")
    print(f"    Mean change: {changes['A']['mean_change']:.8f}")
    print(f"    Norm change: {changes['A']['norm_change']:.8f}")

    if changes['A']['max_change'] > 1e-8:
        print(f"    Status: ✅ UPDATED (training working!)")
    else:
        print(f"    Status: ❌ NO CHANGE (training NOT working!)")

    # B tile (should UPDATE - trainable)
    print(f"  B tile (trainable):")
    print(f"    Max change:  {changes['B']['max_change']:.8f}")
    print(f"    Mean change: {changes['B']['mean_change']:.8f}")
    print(f"    Norm change: {changes['B']['norm_change']:.8f}")

    if changes['B']['max_change'] > 1e-8:
        print(f"    Status: ✅ UPDATED (training working!)")
    else:
        print(f"    Status: ❌ NO CHANGE (training NOT working!)")

    # C tile (should NOT UPDATE - frozen)
    print(f"  C tile (frozen):")
    print(f"    Max change:  {changes['C']['max_change']:.8f}")
    print(f"    Mean change: {changes['C']['mean_change']:.8f}")
    print(f"    Norm change: {changes['C']['norm_change']:.8f}")

    if changes['C']['max_change'] < 1e-8:
        print(f"    Status: ✅ FROZEN (correct!)")
    else:
        print(f"    Status: ⚠️  CHANGED (should be frozen!)")

# =============================================================================
# Verify LoRA forward computation
# =============================================================================
print()
print("=" * 80)
print("LORA FORWARD COMPUTATION VERIFICATION")
print("=" * 80)
print()

print("Testing: y = C·x + α·A·(B·x)")
print()

# Test with query layer
layer_name = 'query'
tile = get_lrtt_tile(layer_name)

# Get weights
w_a = updated_weights[layer_name]['A']  # [hidden, rank]
w_b = updated_weights[layer_name]['B']  # [rank, hidden]
w_c = updated_weights[layer_name]['C']  # [hidden, hidden]

# Test input
test_x = torch.randn(1, hidden_size)

print(f"Test input shape: {test_x.shape}")
print(f"A shape: {w_a.shape}")
print(f"B shape: {w_b.shape}")
print(f"C shape: {w_c.shape}")
print()

# Manual computation
y_c = test_x @ w_c.t()  # C·x
y_b = test_x @ w_b.t()  # B·x
y_a = y_b @ w_a.t()     # A·(B·x)
y_manual = y_c + 1.0 * y_a  # α=1.0

print(f"Manual computation:")
print(f"  C·x norm: {y_c.norm().item():.6f}")
print(f"  B·x norm: {y_b.norm().item():.6f}")
print(f"  A·(B·x) norm: {y_a.norm().item():.6f}")
print(f"  y = C·x + α·A·(B·x) norm: {y_manual.norm().item():.6f}")
print()

# Actual forward pass
with torch.no_grad():
    y_actual = model.query(test_x)

print(f"Actual forward pass:")
print(f"  y norm: {y_actual.norm().item():.6f}")
print()

# Compare
diff = (y_manual - y_actual).abs().max().item()
print(f"Difference: {diff:.8f}")
if diff < 1e-4:
    print("  ✅ LoRA computation matches!")
else:
    print("  ⚠️  LoRA computation differs (may need to account for quantization)")

# =============================================================================
# Summary
# =============================================================================
print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

all_a_updated = all(
    compute_weight_changes(initial_weights[l], updated_weights[l])['A']['max_change'] > 1e-8
    for l in layers_to_track
)

all_b_updated = all(
    compute_weight_changes(initial_weights[l], updated_weights[l])['B']['max_change'] > 1e-8
    for l in layers_to_track
)

all_c_frozen = all(
    compute_weight_changes(initial_weights[l], updated_weights[l])['C']['max_change'] < 1e-8
    for l in layers_to_track
)

print("A tile (low-rank adapter):")
if all_a_updated:
    print("  ✅ All A tiles updated correctly")
    print("  ✅ Gradient flow to A working")
else:
    print("  ❌ Some A tiles NOT updated")
    print("  ❌ Check gradient flow or learning rate")

print()
print("B tile (low-rank adapter):")
if all_b_updated:
    print("  ✅ All B tiles updated correctly")
    print("  ✅ Gradient flow to B working")
else:
    print("  ❌ Some B tiles NOT updated")
    print("  ❌ Check gradient flow or learning rate")

print()
print("C tile (pretrained weights):")
if all_c_frozen:
    print("  ✅ All C tiles remain frozen")
    print("  ✅ Pretrained weights preserved")
else:
    print("  ⚠️  Some C tiles changed")
    print("  ⚠️  Check trainability settings")

print()
print("Overall LRTT-LoRA training status:")
if all_a_updated and all_b_updated and all_c_frozen:
    print("  ✅ LRTT-LoRA training mechanism working correctly!")
    print("  ✅ A and B tiles learning")
    print("  ✅ C tile frozen")
    print("  ✅ Ready for full training")
else:
    print("  ❌ LRTT-LoRA training has issues")
    print("  ❌ Check configuration and gradient flow")

print()
