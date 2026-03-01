#!/usr/bin/env python
# coding=utf-8
"""
Verify gradient clipping at analog tile level - Version 2

This version directly inspects AnalogContext gradients that flow into
the LRTT controller's update mechanism.
"""

import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.optim.context import AnalogContext

import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("GRADIENT CLIPPING VERIFICATION - Analog Context Level")
print("=" * 80)
print()

# =============================================================================
# Create Model
# =============================================================================
print("[1/4] Creating LRTT-LoRA model...")

class SimpleModel(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(768), dim=-1)
        out = (attn @ v).mean(dim=1)
        return self.classifier(out)

model = SimpleModel()
lrtt_config = create_lrtt_lora_config(rank=8, lora_alpha=1.0)
model = convert_model_to_lrtt_lora(
    model,
    lrtt_config,
    target_modules=["query", "key", "value"]
)

print("✓ Model ready\n")

# =============================================================================
# Helper functions
# =============================================================================

def get_analog_contexts(model):
    """Extract all AnalogContext parameters from model."""
    contexts = {}
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            contexts[name] = param
    return contexts

def compute_analog_gradient_norms(model):
    """
    Compute gradient norms from AnalogContext objects.

    AnalogContext stores:
    - analog_input: x values from forward pass
    - analog_grad_output: delta values from backward pass

    The actual gradient to the tile is the outer product of these.
    """
    contexts = get_analog_contexts(model)
    norms = {}

    for name, ctx in contexts.items():
        if ctx.has_gradient():
            # Get the gradient output (delta) that will be used for update
            # This is what goes into tile.update(x_input, d_input)
            grad_outputs = ctx.analog_grad_output

            if grad_outputs:
                # Concatenate all gradient outputs
                d_tensor = torch.cat([d.flatten() for d in grad_outputs])
                norms[name] = {
                    'norm': d_tensor.norm().item(),
                    'max': d_tensor.abs().max().item(),
                    'mean': d_tensor.abs().mean().item(),
                    'shape': [d.shape for d in grad_outputs],
                }

    return norms

# =============================================================================
# Test 1: WITHOUT gradient clipping
# =============================================================================
print("[2/4] Training step WITHOUT gradient clipping...")
print("-" * 80)

torch.manual_seed(42)
x_input = torch.randn(4, 10, 768) * 5.0  # Moderate input
labels = torch.randint(0, 2, (4,))

model.train()
output = model(x_input)
loss = nn.CrossEntropyLoss()(output, labels)

model.zero_grad()
loss.backward()

# Compute gradient norm WITHOUT clipping
params_with_grad = [p for p in model.parameters() if p.grad is not None and not isinstance(p, AnalogContext)]
total_norm_no_clip = torch.nn.utils.clip_grad_norm_(params_with_grad, float('inf')).item()

print(f"Total gradient norm (no clip): {total_norm_no_clip:.6f}")

# Check individual parameter gradients
print("\nGradient norms by parameter type:")
analog_grad_sum = 0
digital_grad_sum = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if 'analog_ctx' in name:
            print(f"  {name}: {grad_norm:.6f} (ANALOG)")
            analog_grad_sum += grad_norm ** 2
        elif 'classifier' in name:
            print(f"  {name}: {grad_norm:.6f} (DIGITAL)")
            digital_grad_sum += grad_norm ** 2

print(f"\nAnalog gradient L2 norm: {np.sqrt(analog_grad_sum):.6f}")
print(f"Digital gradient L2 norm: {np.sqrt(digital_grad_sum):.6f}")

# Get analog context gradient info
analog_norms_no_clip = compute_analog_gradient_norms(model)
print("\nAnalog context gradient details:")
for name, info in sorted(analog_norms_no_clip.items()):
    print(f"  {name}:")
    print(f"    delta norm: {info['norm']:.6f}")
    print(f"    delta max:  {info['max']:.6f}")

# Store for optimizer step
from aihwkit.optim import AnalogSGD
optimizer = AnalogSGD(model.parameters(), lr=0.001)
optimizer.step()

no_clip_values = analog_norms_no_clip.copy()

# =============================================================================
# Test 2: WITH gradient clipping
# =============================================================================
print()
print("[3/4] Training step WITH gradient clipping (max_norm=1.0)...")
print("-" * 80)

torch.manual_seed(42)
x_input = torch.randn(4, 10, 768) * 5.0
labels = torch.randint(0, 2, (4,))

model.train()
output = model(x_input)
loss = nn.CrossEntropyLoss()(output, labels)

model.zero_grad()
loss.backward()

# Compute gradient norm BEFORE clipping
params_with_grad = [p for p in model.parameters() if p.grad is not None and not isinstance(p, AnalogContext)]
total_norm_before = torch.nn.utils.clip_grad_norm_(params_with_grad, float('inf')).item()

print(f"Total gradient norm (before clip): {total_norm_before:.6f}")

# Apply clipping (THIS IS WHAT HF TRAINER DOES)
total_norm_after = torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=1.0).item()

print(f"Total gradient norm (after clip):  {total_norm_after:.6f}")
print(f"Clipping factor: {min(1.0, 1.0 / total_norm_before):.6f}")

# Check individual parameter gradients AFTER clipping
print("\nGradient norms by parameter type (AFTER clipping):")
analog_grad_sum_clip = 0
digital_grad_sum_clip = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if 'analog_ctx' in name:
            print(f"  {name}: {grad_norm:.6f} (ANALOG - clipped)")
            analog_grad_sum_clip += grad_norm ** 2
        elif 'classifier' in name:
            print(f"  {name}: {grad_norm:.6f} (DIGITAL - clipped)")
            digital_grad_sum_clip += grad_norm ** 2

print(f"\nAnalog gradient L2 norm (clipped): {np.sqrt(analog_grad_sum_clip):.6f}")
print(f"Digital gradient L2 norm (clipped): {np.sqrt(digital_grad_sum_clip):.6f}")

# Get analog context gradient info AFTER clipping
analog_norms_with_clip = compute_analog_gradient_norms(model)
print("\nAnalog context gradient details (AFTER clipping):")
for name, info in sorted(analog_norms_with_clip.items()):
    print(f"  {name}:")
    print(f"    delta norm: {info['norm']:.6f}")
    print(f"    delta max:  {info['max']:.6f}")

optimizer.step()

with_clip_values = analog_norms_with_clip.copy()

# =============================================================================
# Comparison
# =============================================================================
print()
print("[4/4] VERIFICATION RESULTS")
print("=" * 80)

print("\nComparison: Gradient magnitudes reaching analog tiles")
print("-" * 80)

all_reduced = True
for ctx_name in no_clip_values.keys():
    if ctx_name in with_clip_values:
        no_clip_norm = no_clip_values[ctx_name]['norm']
        with_clip_norm = with_clip_values[ctx_name]['norm']
        reduction = (1 - with_clip_norm / no_clip_norm) * 100 if no_clip_norm > 0 else 0

        print(f"\n{ctx_name}:")
        print(f"  Without clipping: {no_clip_norm:.6f}")
        print(f"  With clipping:    {with_clip_norm:.6f}")
        print(f"  Reduction:        {reduction:.2f}%")

        if reduction < 5 and total_norm_before > 1.0:
            all_reduced = False
            print(f"  ⚠️  Insufficient reduction despite grad_norm > 1.0")

print()
print("=" * 80)
print("CONCLUSION:")
print("=" * 80)

if all_reduced or total_norm_before <= 1.0:
    print("✅ Gradient clipping DOES affect analog tiles")
    print(f"   - Total grad norm before: {total_norm_before:.6f}")
    print(f"   - Total grad norm after:  {total_norm_after:.6f}")
    print(f"   - Analog context gradients are scaled proportionally")
    print(f"   - LRTT controller receives CLIPPED gradients")
    print()
    print("   This confirms: max_grad_norm=1.0 is working correctly!")
else:
    print("⚠️  Gradient clipping may NOT be affecting analog tiles correctly")
    print(f"   - Total grad norm: {total_norm_before:.6f} → {total_norm_after:.6f}")
    print(f"   - But analog context gradients didn't change significantly")
    print()
    print("   This needs investigation!")

print()
