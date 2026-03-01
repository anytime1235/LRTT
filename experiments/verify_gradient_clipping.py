#!/usr/bin/env python
# coding=utf-8
"""
Verify that gradient clipping actually affects analog tile updates.

This script:
1. Creates a simple LRTT-LoRA model
2. Simulates training step with gradient clipping
3. Intercepts gradients at analog tile update
4. Verifies gradients are clipped before analog update
"""

import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("GRADIENT CLIPPING VERIFICATION FOR ANALOG TILES")
print("=" * 80)
print()

# =============================================================================
# Step 1: Create a simple model with LRTT-LoRA
# =============================================================================
print("[1/5] Creating LRTT-LoRA model...")

class SimpleTransformerLayer(nn.Module):
    """Minimal transformer layer for testing."""
    def __init__(self, hidden_size=768):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # Simple forward pass
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(768), dim=-1)
        out = attn @ v
        out = self.dense(out)
        out = out.mean(dim=1)  # Pool
        return self.classifier(out)

# Create model
model = SimpleTransformerLayer(hidden_size=768)

# Convert to LRTT-LoRA (only Q/K/V, not dense)
lrtt_config = create_lrtt_lora_config(rank=8, lora_alpha=1.0)
model = convert_model_to_lrtt_lora(
    model,
    lrtt_config,
    target_modules=["query", "key", "value"]
)

print(f"✓ Model created with LRTT-LoRA layers")
print()

# =============================================================================
# Step 2: Prepare to intercept analog tile updates
# =============================================================================
print("[2/5] Setting up gradient interception...")

# Store intercepted gradients
intercepted_gradients = {}

# Monkey-patch the analog tile update to intercept gradients
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

original_update = LRTTSimulatorTile.update

def intercepted_update(self, x_input, d_input, *args, **kwargs):
    """Intercept gradients going into analog tile update."""
    # Store the gradient (d_input) for inspection
    layer_name = None
    for name, module in model.named_modules():
        if hasattr(module, 'analog_module'):
            if isinstance(module.analog_module, LRTTSimulatorTile):
                if module.analog_module is self:
                    layer_name = name
                    break
            elif hasattr(module.analog_module, 'array'):
                # TileModuleArray case
                try:
                    if module.analog_module.array[0][0] is self:
                        layer_name = name
                        break
                except:
                    pass

    if layer_name:
        # d_input is the gradient (delta) for this layer
        grad_norm = d_input.norm().item()
        intercepted_gradients[layer_name] = {
            'grad_norm': grad_norm,
            'grad_shape': d_input.shape,
            'grad_max': d_input.abs().max().item(),
            'grad_mean': d_input.abs().mean().item(),
        }

    # Call original update
    return original_update(self, x_input, d_input, *args, **kwargs)

# Apply monkey patch
LRTTSimulatorTile.update = intercepted_update

print("✓ Gradient interception enabled")
print()

# =============================================================================
# Step 3: Simulate training with LARGE gradients (no clipping)
# =============================================================================
print("[3/5] Training step WITHOUT gradient clipping...")
print("-" * 80)

# Create input that will produce large gradients
torch.manual_seed(42)
x_large = torch.randn(4, 10, 768) * 10.0  # Large input → large gradients
labels = torch.randint(0, 2, (4,))

# Forward pass
model.train()
output = model(x_large)
loss = nn.CrossEntropyLoss()(output, labels)

# Backward pass (NO gradient clipping)
model.zero_grad()
loss.backward()

# Compute gradient norm BEFORE any clipping
params_with_grad = [p for p in model.parameters() if p.grad is not None]
total_grad_norm_before = torch.nn.utils.clip_grad_norm_(
    params_with_grad, max_norm=float('inf')
).item()

print(f"Total gradient norm (pre-clip): {total_grad_norm_before:.4f}")

# Now do optimizer step WITHOUT clipping
from aihwkit.optim import AnalogSGD
optimizer = AnalogSGD(model.parameters(), lr=0.001)
intercepted_gradients.clear()
optimizer.step()

print(f"\nGradients received by analog tiles (NO clipping):")
for layer_name, grad_info in sorted(intercepted_gradients.items()):
    print(f"  {layer_name}:")
    print(f"    Gradient norm: {grad_info['grad_norm']:.4f}")
    print(f"    Gradient max:  {grad_info['grad_max']:.4f}")
    print(f"    Gradient mean: {grad_info['grad_mean']:.4f}")

no_clip_norms = {name: info['grad_norm'] for name, info in intercepted_gradients.items()}

# =============================================================================
# Step 4: Simulate training with gradient clipping
# =============================================================================
print()
print("[4/5] Training step WITH gradient clipping (max_norm=1.0)...")
print("-" * 80)

# Same input
torch.manual_seed(42)
x_large = torch.randn(4, 10, 768) * 10.0
labels = torch.randint(0, 2, (4,))

# Forward pass
output = model(x_large)
loss = nn.CrossEntropyLoss()(output, labels)

# Backward pass
model.zero_grad()
loss.backward()

# Apply gradient clipping (SAME AS HF TRAINER)
params_with_grad = [p for p in model.parameters() if p.grad is not None]
total_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
    params_with_grad, max_norm=float('inf')
).item()
print(f"Total gradient norm (before clipping): {total_grad_norm_before_clip:.4f}")

# NOW CLIP
total_grad_norm_after = torch.nn.utils.clip_grad_norm_(
    params_with_grad, max_norm=1.0
).item()
print(f"Total gradient norm (after clipping):  {total_grad_norm_after:.4f}")
print(f"Clipping applied: {total_grad_norm_before_clip > 1.0}")

# Optimizer step with CLIPPED gradients
intercepted_gradients.clear()
optimizer.step()

print(f"\nGradients received by analog tiles (WITH clipping):")
for layer_name, grad_info in sorted(intercepted_gradients.items()):
    print(f"  {layer_name}:")
    print(f"    Gradient norm: {grad_info['grad_norm']:.4f}")
    print(f"    Gradient max:  {grad_info['grad_max']:.4f}")
    print(f"    Gradient mean: {grad_info['grad_mean']:.4f}")

clip_norms = {name: info['grad_norm'] for name, info in intercepted_gradients.items()}

# =============================================================================
# Step 5: Verify clipping effect
# =============================================================================
print()
print("[5/5] Verification Results")
print("=" * 80)

clipping_worked = True
for layer_name in no_clip_norms.keys():
    no_clip = no_clip_norms[layer_name]
    with_clip = clip_norms[layer_name]
    reduction = (no_clip - with_clip) / no_clip * 100 if no_clip > 0 else 0

    print(f"\n{layer_name}:")
    print(f"  No clipping:   grad_norm = {no_clip:.6f}")
    print(f"  With clipping: grad_norm = {with_clip:.6f}")
    print(f"  Reduction:     {reduction:.2f}%")

    # Check if clipping actually reduced the gradient
    if no_clip > 1.0 and with_clip >= no_clip * 0.95:
        print(f"  ⚠️  WARNING: Clipping didn't reduce gradient significantly!")
        clipping_worked = False

print()
print("=" * 80)
print("CONCLUSION:")
print("=" * 80)

if clipping_worked and all(clip_norms[k] < no_clip_norms[k] for k in clip_norms):
    print("✅ SUCCESS: Gradient clipping IS applied to analog tiles")
    print("   - Gradients are clipped BEFORE optimizer.step()")
    print("   - Analog tiles receive CLIPPED gradients")
    print("   - max_grad_norm=1.0 logic is working correctly")
else:
    print("❌ FAILURE: Gradient clipping NOT working as expected")
    print("   - Analog tiles may receive unclipped gradients")
    print("   - Need to investigate HF Trainer integration")

print()
print("Key insight:")
print(f"  Total model grad_norm before clip: {total_grad_norm_before_clip:.4f}")
print(f"  Total model grad_norm after clip:  {total_grad_norm_after:.4f}")
print(f"  Individual tile grad_norms are proportionally scaled")
print()

# Restore original update method
LRTTSimulatorTile.update = original_update
