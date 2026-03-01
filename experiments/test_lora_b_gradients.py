"""
Test if gradients flow to LoRA B during backprop
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from transformers import AutoTokenizer
from aihwkit.optim import AnalogSGD

device = torch.device("cuda")
model = create_glue_model('sst2', device, ['query'], fp_lora=False, lora_alpha=0.01)
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

print("=" * 80)
print("TEST: Gradient Flow to LoRA B")
print("=" * 80)

# Find first LoRA B layer
lora_b_layer = None
for name, module in model.named_modules():
    if 'layer.0.attention.self.query' in name and 'lora_B' in name and 'default' in name:
        lora_b_layer = module
        lora_b_name = name
        break

print(f"\nTesting LoRA B: {lora_b_name}")
print(f"Type: {type(lora_b_layer)}")

# Get initial weights
initial_weights = lora_b_layer.get_weights()
if isinstance(initial_weights, tuple):
    initial_weights = initial_weights[0]

print(f"\nInitial LoRA B weights:")
print(f"  Shape: {initial_weights.shape}")
print(f"  Mean: {initial_weights.mean().item():.6f}")
print(f"  Std: {initial_weights.std().item():.6f}")
print(f"  Max: {initial_weights.abs().max().item():.6f}")

# Setup for training
model.train()
optimizer = AnalogSGD(model.parameters(), lr=0.1)  # High LR for strong signal
optimizer.regroup_param_groups(model)

criterion = nn.CrossEntropyLoss()

print("\n" + "-" * 80)
print("Running 5 training steps with high LR (0.1)...")
print("-" * 80)

# Hook to capture gradients
gradient_captured = {}

def capture_gradient(name):
    def hook(grad):
        gradient_captured[name] = grad.clone()
        return grad
    return hook

# Register gradient hook on LoRA B
# Note: AnalogLinear doesn't have .weight parameter, so we can't hook directly
# We'll check weight changes instead

for step in range(5):
    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = torch.tensor([1]).to(device)

    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = criterion(outputs.logits, labels)

    print(f"\nStep {step + 1}:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits: {outputs.logits[0]}")

    loss.backward()

    # Check if analog tile received gradients
    # (AnalogSGD should have updated the tile's pending_grad)

    optimizer.step()

# Get final weights
final_weights = lora_b_layer.get_weights()
if isinstance(final_weights, tuple):
    final_weights = final_weights[0]

print("\n" + "-" * 80)
print("After training:")
print("-" * 80)

print(f"\nFinal LoRA B weights:")
print(f"  Mean: {final_weights.mean().item():.6f}")
print(f"  Std: {final_weights.std().item():.6f}")
print(f"  Max: {final_weights.abs().max().item():.6f}")

# Check weight change
weight_diff = (final_weights - initial_weights).abs()
max_change = weight_diff.max().item()
mean_change = weight_diff.mean().item()

print(f"\nWeight changes:")
print(f"  Max change: {max_change:.6f}")
print(f"  Mean change: {mean_change:.6f}")

print("\n" + "=" * 80)
if max_change > 1e-6:
    print("✓ LoRA B weights ARE updating!")
    print(f"  Weight update magnitude: {max_change:.6f}")
else:
    print("✗ LoRA B weights NOT updating!")
    print("  Possible issues:")
    print("    1. Gradients not reaching LoRA B")
    print("    2. AnalogSGD not updating the tile")
    print("    3. Tile configuration preventing updates")
print("=" * 80)

# Additional diagnostic: Check if LoRA B is marked as trainable
print("\n" + "-" * 80)
print("LoRA B trainability check:")
print("-" * 80)

# Check analog tile configuration
if hasattr(lora_b_layer, 'analog_module'):
    tile = lora_b_layer.analog_module
    print(f"\nAnalog tile type: {type(tile)}")
    if hasattr(tile, 'rpu_config'):
        print(f"RPU config: {type(tile.rpu_config)}")
        print(f"Device: {type(tile.rpu_config.device)}")

    # Check if tile is in training mode
    print(f"Tile training mode: {tile.training}")

# Check if weight is frozen
print(f"\nLoRA B layer training mode: {lora_b_layer.training}")

# Try to directly check if gradients were applied
print("\n" + "-" * 80)
print("Optimizer parameter groups:")
print("-" * 80)

for i, group in enumerate(optimizer.param_groups):
    if 'analog_tile_id' in group:
        print(f"\nGroup {i}:")
        print(f"  Analog tiles: {len(group.get('analog_tiles', []))}")
        # Check if our lora_B is in this group
        for tile_id, tile in enumerate(group.get('analog_tiles', [])):
            if tile is lora_b_layer.analog_module:
                print(f"  ✓ Found LoRA B tile in group {i}, tile {tile_id}")
