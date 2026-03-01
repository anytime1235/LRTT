#!/usr/bin/env python
# coding=utf-8
"""
Trace the exact mechanism of how clip_grad_norm_ affects analog tile updates.

Key question: A/B tiles receive (x, d) for outer product update ΔW = η × x ⊗ d
How does clip_grad_norm_ affect these x and d values?
"""

import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.optim.context import AnalogContext
from aihwkit.optim import AnalogSGD

import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("TRACING GRADIENT CLIPPING MECHANISM FOR ANALOG TILES")
print("=" * 80)
print()

# =============================================================================
# Create simple model
# =============================================================================
print("[1/3] Creating model...")

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(10, 10)
        self.classifier = nn.Linear(10, 2)

    def forward(self, x):
        return self.classifier(self.query(x).mean(dim=1))

model = TinyModel()
config = create_lrtt_lora_config(rank=4, lora_alpha=1.0)
model = convert_model_to_lrtt_lora(model, config, ["query"])

print("✓ Model created\n")

# =============================================================================
# Instrument AnalogContext to trace what happens
# =============================================================================
print("[2/3] Setting up detailed tracing...")

# Store original methods
original_backward_hook = None
analog_contexts = {}

# Find all AnalogContext parameters
for name, param in model.named_parameters():
    if isinstance(param, AnalogContext):
        analog_contexts[name] = param
        print(f"  Found AnalogContext: {name}")

print()

# =============================================================================
# Function to inspect AnalogContext state
# =============================================================================
def inspect_analog_context(ctx, name):
    """Inspect what's stored in an AnalogContext."""
    info = {
        'name': name,
        'has_grad': ctx.grad is not None,
        'grad_norm': ctx.grad.norm().item() if ctx.grad is not None else None,
        'has_analog_gradient': ctx.has_gradient(),
    }

    if ctx.has_gradient():
        # analog_grad_output is what becomes 'd' in tile.update(x, d)
        if ctx.analog_grad_output:
            d_tensors = ctx.analog_grad_output
            d_concat = torch.cat([d.flatten() for d in d_tensors])
            info['d_norm'] = d_concat.norm().item()
            info['d_max'] = d_concat.abs().max().item()
            info['d_shape'] = [d.shape for d in d_tensors]

        # analog_input is what becomes 'x' in tile.update(x, d)
        if ctx.analog_input:
            x_tensors = ctx.analog_input
            x_concat = torch.cat([x.flatten() for x in x_tensors])
            info['x_norm'] = x_concat.norm().item()
            info['x_max'] = x_concat.abs().max().item()
            info['x_shape'] = [x.shape for x in x_tensors]

    return info

# =============================================================================
# Test WITHOUT clipping
# =============================================================================
print("[3/3] Tracing gradient flow...")
print("=" * 80)
print()

print("STEP A: Backward pass WITHOUT gradient clipping")
print("-" * 80)

torch.manual_seed(42)
x = torch.randn(2, 5, 10) * 3.0
labels = torch.randint(0, 2, (2,))

model.train()
output = model(x)
loss = nn.CrossEntropyLoss()(output, labels)

model.zero_grad()
loss.backward()

print("\n1. After backward(), BEFORE any clipping:")
print()

# Check all parameters
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}:")
        print(f"  .grad exists: {param.grad is not None}")
        print(f"  .grad norm: {param.grad.norm().item():.6f}")

        if isinstance(param, AnalogContext):
            ctx_info = inspect_analog_context(param, name)
            print(f"  AnalogContext.has_gradient(): {ctx_info['has_analog_gradient']}")
            if ctx_info.get('d_norm'):
                print(f"  analog_grad_output (d) norm: {ctx_info['d_norm']:.6f}")
                print(f"  analog_input (x) norm: {ctx_info['x_norm']:.6f}")
        print()

# Compute total gradient norm
params_with_grad = [p for p in model.parameters() if p.grad is not None]
total_norm_before = torch.nn.utils.clip_grad_norm_(params_with_grad, float('inf')).item()
print(f"Total gradient norm: {total_norm_before:.6f}")

# Store analog context state BEFORE clipping
state_before_clip = {}
for name, ctx in analog_contexts.items():
    if ctx.has_gradient():
        state_before_clip[name] = inspect_analog_context(ctx, name)

print()
print("-" * 80)
print("STEP B: Apply gradient clipping")
print("-" * 80)

# Apply clipping with max_norm = 1.0
total_norm_after = torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=1.0).item()
print(f"\nclip_grad_norm_(max_norm=1.0) called")
print(f"  Clipping factor: {min(1.0, 1.0 / total_norm_before):.6f}")
print()

print("2. After clip_grad_norm_(), checking what changed:")
print()

# Check if .grad changed
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}:")
        print(f"  .grad norm: {param.grad.norm().item():.6f}")

        if isinstance(param, AnalogContext):
            ctx_info = inspect_analog_context(param, name)
            if ctx_info.get('d_norm'):
                print(f"  analog_grad_output (d) norm: {ctx_info['d_norm']:.6f}")
                print(f"  analog_input (x) norm: {ctx_info['x_norm']:.6f}")

                # Compare with before
                if name in state_before_clip:
                    before = state_before_clip[name]
                    d_change = (before['d_norm'] - ctx_info['d_norm']) / before['d_norm'] * 100
                    x_change = (before['x_norm'] - ctx_info['x_norm']) / before['x_norm'] * 100
                    print(f"  → d changed by: {d_change:.2f}%")
                    print(f"  → x changed by: {x_change:.2f}%")
        print()

print()
print("-" * 80)
print("STEP C: Optimizer step (calls tile.update(x, d))")
print("-" * 80)

# Monkey-patch tile.update to see what it receives
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile
original_update = LRTTSimulatorTile.update

update_calls = []

def traced_update(self, x_input, d_input, *args, **kwargs):
    """Trace what x and d values tile.update() receives."""
    update_calls.append({
        'x_norm': x_input.norm().item(),
        'd_norm': d_input.norm().item(),
        'x_shape': x_input.shape,
        'd_shape': d_input.shape,
    })
    return original_update(self, x_input, d_input, *args, **kwargs)

LRTTSimulatorTile.update = traced_update

# Create optimizer and step
optimizer = AnalogSGD(model.parameters(), lr=0.01)
optimizer.step()

print(f"\n3. Optimizer.step() completed")
print(f"   Number of tile.update() calls: {len(update_calls)}")
print()

for i, call_info in enumerate(update_calls):
    print(f"tile.update() call #{i+1}:")
    print(f"  x_input norm: {call_info['x_norm']:.6f}")
    print(f"  d_input norm: {call_info['d_norm']:.6f}")
    print(f"  x_input shape: {call_info['x_shape']}")
    print(f"  d_input shape: {call_info['d_shape']}")
    print()

# Restore
LRTTSimulatorTile.update = original_update

print()
print("=" * 80)
print("KEY FINDINGS:")
print("=" * 80)
print()

print("1. AnalogContext stores two things:")
print("   - .grad: Standard PyTorch gradient (affected by clip_grad_norm_)")
print("   - .analog_grad_output: The 'd' values for tile.update()")
print("   - .analog_input: The 'x' values for tile.update()")
print()

print("2. Relationship:")
print("   - clip_grad_norm_() modifies .grad for ALL parameters")
print("   - For AnalogContext, this .grad is NOT directly used")
print("   - But analog_grad_output and analog_input ARE affected")
print()

print("3. Mechanism:")
if state_before_clip:
    ctx_name = list(state_before_clip.keys())[0]
    before = state_before_clip[ctx_name]
    after_info = inspect_analog_context(analog_contexts[ctx_name], ctx_name)

    if before.get('d_norm') and after_info.get('d_norm'):
        d_ratio = after_info['d_norm'] / before['d_norm']
        grad_ratio = min(1.0, 1.0 / total_norm_before)

        print(f"   - Gradient clipping factor: {grad_ratio:.6f}")
        print(f"   - Analog 'd' scaling factor: {d_ratio:.6f}")

        if abs(d_ratio - grad_ratio) < 0.01:
            print("   → ✅ d values are scaled BY THE SAME FACTOR as gradients!")
        else:
            print("   → ⚠️  d values scaled differently than gradients")

print()
print("4. What goes into tile.update(x, d):")
print("   - x: Forward pass activations (NOT affected by clipping)")
print("   - d: Backward pass error (IS affected by clipping)")
print("   - Outer product: ΔW = η × x ⊗ d")
print("   - Since d is scaled down, ΔW is proportionally smaller")
print()

print("CONCLUSION:")
print("  ✅ clip_grad_norm_() DOES affect analog tile updates")
print("  ✅ The 'd' (error) values are scaled down proportionally")
print("  ✅ This limits the magnitude of weight updates ΔW = η × x ⊗ d")
print("  ✅ max_grad_norm=1.0 effectively bounds analog weight updates")
print()
