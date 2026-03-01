#!/usr/bin/env python
"""
Diagnose gradient flow with forward_inject=True
"""
import sys
import torch
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from transformers import AutoModelForSequenceClassification
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogLinear

MODEL_NAME = "google/mobilebert-uncased"
BATCH_SIZE = 8
SEQ_LENGTH = 128

print("=" * 80)
print("GRADIENT FLOW DIAGNOSTIC TEST")
print("=" * 80)

device = torch.device("cuda")

# Create small dummy batch
input_ids = torch.randint(0, 30522, (BATCH_SIZE, SEQ_LENGTH)).to(device)
attention_mask = torch.ones(BATCH_SIZE, SEQ_LENGTH).to(device)
labels = torch.randint(0, 2, (BATCH_SIZE,)).to(device)

# Load and convert model
print("\n[1] Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

print("[2] Creating LRTT config with forward_inject=True...")
lrtt_config = create_lrtt_lora_config(
    rank=8,
    lora_alpha=1.0,
    output_noise_level=0.0,
    use_floating_point=False,
)

print(f"    Config forward_inject: {lrtt_config.device.forward_inject}")

print("[3] Converting to LRTT-LoRA...")
model = convert_model_to_lrtt_lora(model, lrtt_config, ["query", "key", "value"])
model.to(device)

# Find first query layer
print("\n[4] Inspecting first query layer...")
first_query = None
first_query_name = None
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        first_query = module
        first_query_name = name
        break

print(f"    Layer: {first_query_name}")
print(f"    Type: {type(first_query)}")

# Access tile
if hasattr(first_query, 'analog_module'):
    tile_module = first_query.analog_module
    if hasattr(tile_module, 'array'):
        tile = tile_module.array[0][0]
    else:
        tile = tile_module

    print(f"    Tile type: {type(tile)}")
    print(f"    forward_inject: {tile.controller.forward_inject_enabled}")

    # Get analog contexts
    ctx_a = tile.tile_a.get_analog_ctx()
    ctx_b = tile.tile_b.get_analog_ctx()
    ctx_c = tile.tile_c.get_analog_ctx()

    print(f"\n    Before forward:")
    print(f"      ctx_a.has_gradient(): {ctx_a.has_gradient()}")
    print(f"      ctx_b.has_gradient(): {ctx_b.has_gradient()}")
    print(f"      ctx_c.has_gradient(): {ctx_c.has_gradient()}")
    print(f"      ctx_a.analog_input: {len(ctx_a.analog_input)} items")
    print(f"      ctx_b.analog_input: {len(ctx_b.analog_input)} items")
    print(f"      ctx_c.analog_input: {len(ctx_c.analog_input)} items")

# Create optimizer
print("\n[5] Creating optimizer...")
optimizer = AnalogSGD(model.parameters(), lr=5e-4)
optimizer.regroup_param_groups(model)

# Single training step
print("\n[6] Running single training step...")
model.train()

optimizer.zero_grad()

# Forward
print("    [6.1] Forward pass...")
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
print(f"    Loss: {loss.item():.4f}")

if torch.isnan(loss):
    print("    ✗ Loss is NaN!")
    sys.exit(1)

# Check after forward, before backward
print(f"\n    After forward, before backward:")
print(f"      ctx_a.has_gradient(): {ctx_a.has_gradient()}")
print(f"      ctx_b.has_gradient(): {ctx_b.has_gradient()}")
print(f"      ctx_c.has_gradient(): {ctx_c.has_gradient()}")
print(f"      ctx_a.analog_input: {len(ctx_a.analog_input)} items")
print(f"      ctx_b.analog_input: {len(ctx_b.analog_input)} items")
print(f"      ctx_c.analog_input: {len(ctx_c.analog_input)} items")

# Backward
print("    [6.2] Backward pass...")
loss.backward()

# Check after backward
print(f"\n    After backward:")
print(f"      ctx_a.has_gradient(): {ctx_a.has_gradient()}")
print(f"      ctx_b.has_gradient(): {ctx_b.has_gradient()}")
print(f"      ctx_c.has_gradient(): {ctx_c.has_gradient()}")
print(f"      ctx_a.analog_input: {len(ctx_a.analog_input)} items")
print(f"      ctx_b.analog_input: {len(ctx_b.analog_input)} items")
print(f"      ctx_c.analog_input: {len(ctx_c.analog_input)} items")
print(f"      ctx_a.analog_grad_output: {len(ctx_a.analog_grad_output)} items")
print(f"      ctx_b.analog_grad_output: {len(ctx_b.analog_grad_output)} items")
print(f"      ctx_c.analog_grad_output: {len(ctx_c.analog_grad_output)} items")

# Check standard PyTorch gradients
print("\n    Checking standard PyTorch parameter gradients:")
has_any_grad = False
for name, param in model.named_parameters():
    if param.grad is not None and 'query' in name:
        has_any_grad = True
        print(f"      {name}: grad norm = {param.grad.norm().item():.6f}")

if not has_any_grad:
    print("      ✗ No standard PyTorch gradients found!")

# Optimizer step
print("\n    [6.3] Optimizer step...")
optimizer.step()

# Check after optimizer step
print(f"\n    After optimizer step:")
print(f"      ctx_a.has_gradient(): {ctx_a.has_gradient()}")
print(f"      ctx_b.has_gradient(): {ctx_b.has_gradient()}")
print(f"      ctx_c.has_gradient(): {ctx_c.has_gradient()}")
print(f"      ctx_a.analog_input: {len(ctx_a.analog_input)} items")
print(f"      ctx_b.analog_input: {len(ctx_b.analog_input)} items")
print(f"      ctx_c.analog_input: {len(ctx_c.analog_input)} items")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if ctx_a.has_gradient() or ctx_b.has_gradient() or ctx_c.has_gradient():
    print("✓ Analog contexts received gradients!")
    print("  This means the backward hook WAS called.")
else:
    print("✗ No gradients in analog contexts!")
    print("  This means the backward hook was NOT called.")
    print("  Possible reasons:")
    print("    1. Output tensor doesn't require grad")
    print("    2. Hook registration failed")
    print("    3. Gradient computation stopped before reaching the hook")

print("=" * 80)
