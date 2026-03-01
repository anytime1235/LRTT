#!/usr/bin/env python3
"""
Diagnose why lora_A/B are not updating properly

We know:
- base_layer is properly frozen ✓
- lora_A/B have requires_grad=True
- Some weights change (241/1024) but max change = 0.000000
- This suggests updates are too small

Check:
1. Gradient magnitudes for lora_A/B parameters
2. AnalogContext cached values (analog_input, analog_grad_output)
3. Tile update mechanism
4. Learning rate and scaling effects
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

from sixt1c_config import (
    gen_sixt1c_lora_config_trainable,
    gen_softbounds_base_layer_config
)
from smart_conversion import convert_base_and_lora_separately

print("=" * 80)
print("DIAGNOSE LORA UPDATE ISSUE")
print("=" * 80)

# Create model
print("\n[Creating model]")
model_config = AutoConfig.from_pretrained('google/mobilebert-uncased')
model_config.num_labels = 2
model_config.num_hidden_layers = 2

model = AutoModelForSequenceClassification.from_pretrained(
    'google/mobilebert-uncased',
    config=model_config,
    ignore_mismatched_sizes=True
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=1.0,
    target_modules=['query'],
    bias='none',
    lora_dropout=0.0,
)
model = get_peft_model(model, peft_config)

# Convert to analog
base_config = gen_softbounds_base_layer_config()
lora_config = gen_sixt1c_lora_config_trainable()

model = convert_base_and_lora_separately(
    model,
    base_layer_config=base_config,
    lora_config=lora_config,
    lora_trainable=True
)

# Freeze
for name, param in model.named_parameters():
    if 'base_layer' in name or 'classifier' in name:
        param.requires_grad = False
    elif 'lora' in name:
        param.requires_grad = True

print("✓ Model created and converted")

# Setup optimizer
optimizer = AnalogSGD(model.parameters(), lr=0.001)
optimizer.regroup_param_groups(model)
print("✓ Optimizer created (lr=0.001)")

# Prepare input
tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
inputs = tokenizer(
    ["This is a test sentence."] * 2,
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors='pt'
)
labels = torch.tensor([1, 0])

# Find first lora_A and lora_B modules
lora_a_module = None
lora_b_module = None
lora_a_name = None
lora_b_name = None

for name, module in model.named_modules():
    if isinstance(module, AnalogLinear):
        if 'lora_A' in name and 'query' in name and lora_a_module is None:
            lora_a_module = module
            lora_a_name = name
        elif 'lora_B' in name and 'query' in name and lora_b_module is None:
            lora_b_module = module
            lora_b_name = name

print(f"\nTarget modules:")
print(f"  lora_A: {lora_a_name}")
print(f"  lora_B: {lora_b_name}")

# ============================================================================
# TEST 1: Check PyTorch gradients
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: PyTorch Gradient Magnitudes")
print("=" * 80)

model.train()
optimizer.zero_grad()
outputs = model(**inputs, labels=labels)
loss = outputs.loss
print(f"\nLoss: {loss.item():.4f}")

loss.backward()

print("\n[Checking param.grad for lora_A/B]")
for name, param in model.named_parameters():
    if 'lora' in name and 'query' in name and param.requires_grad:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_max = param.grad.abs().max().item()
            grad_mean = param.grad.abs().mean().item()
            print(f"\n{name}:")
            print(f"  grad norm: {grad_norm:.4e}")
            print(f"  grad max:  {grad_max:.4e}")
            print(f"  grad mean: {grad_mean:.4e}")
        else:
            print(f"\n{name}: grad is None ✗")

# ============================================================================
# TEST 2: Check AnalogContext (analog_input, analog_grad_output)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: AnalogContext Cached Values")
print("=" * 80)

def check_analog_context(module, name):
    """Check analog_input and analog_grad_output in the tile."""
    if hasattr(module, 'analog_module'):
        tile = module.analog_module.tile
    else:
        tile = module.tile

    # Check if tile has rpu_config (analog tile)
    if not hasattr(tile, 'rpu_config'):
        print(f"\n{name}: Not an analog tile (no rpu_config)")
        return

    # Try to access analog context
    # In AnalogSGD, it accesses: analog_ctx.analog_input and analog_ctx.analog_grad_output
    # These are stored on the AnalogContext which wraps the tile

    print(f"\n{name}:")
    print(f"  Tile type: {type(tile).__name__}")
    print(f"  Has rpu_config: {hasattr(tile, 'rpu_config')}")

    # Check if module has analog_ctx
    if hasattr(module, 'analog_ctx'):
        ctx = module.analog_ctx
        print(f"  Has analog_ctx: True")
        print(f"    analog_input cached: {len(ctx.analog_input) if hasattr(ctx, 'analog_input') else 'N/A'}")
        print(f"    analog_grad_output cached: {len(ctx.analog_grad_output) if hasattr(ctx, 'analog_grad_output') else 'N/A'}")

        if hasattr(ctx, 'analog_input') and len(ctx.analog_input) > 0:
            inp = ctx.analog_input[0]
            print(f"    analog_input[0] shape: {inp.shape}")
            print(f"    analog_input[0] norm: {inp.norm().item():.4e}")

        if hasattr(ctx, 'analog_grad_output') and len(ctx.analog_grad_output) > 0:
            grad = ctx.analog_grad_output[0]
            print(f"    analog_grad_output[0] shape: {grad.shape}")
            print(f"    analog_grad_output[0] norm: {grad.norm().item():.4e}")
            print(f"    analog_grad_output[0] max: {grad.abs().max().item():.4e}")
    else:
        print(f"  Has analog_ctx: False ✗")

# Already did backward, so context should be populated
check_analog_context(lora_a_module, "lora_A")
check_analog_context(lora_b_module, "lora_B")

# ============================================================================
# TEST 3: Capture weights before/after optimizer.step()
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Weight Changes After optimizer.step()")
print("=" * 80)

# Capture weights before
w_a_before = lora_a_module.get_weights()
w_a_before = (w_a_before[0] if isinstance(w_a_before, tuple) else w_a_before).clone()

w_b_before = lora_b_module.get_weights()
w_b_before = (w_b_before[0] if isinstance(w_b_before, tuple) else w_b_before).clone()

print("\n[Before optimizer.step()]")
print(f"lora_A weight norm: {w_a_before.norm().item():.4f}")
print(f"lora_A weight range: [{w_a_before.min().item():.4f}, {w_a_before.max().item():.4f}]")
print(f"lora_B weight norm: {w_b_before.norm().item():.4f}")
print(f"lora_B weight range: [{w_b_before.min().item():.4f}, {w_b_before.max().item():.4f}]")

# Take optimizer step
print("\n[Calling optimizer.step()]")
optimizer.step()

# Capture weights after
w_a_after = lora_a_module.get_weights()
w_a_after = (w_a_after[0] if isinstance(w_a_after, tuple) else w_a_after).clone()

w_b_after = lora_b_module.get_weights()
w_b_after = (w_b_after[0] if isinstance(w_b_after, tuple) else w_b_after).clone()

print("\n[After optimizer.step()]")
print(f"lora_A weight norm: {w_a_after.norm().item():.4f}")
print(f"lora_B weight norm: {w_b_after.norm().item():.4f}")

# Compute deltas
delta_a = (w_a_after - w_a_before).abs()
delta_b = (w_b_after - w_b_before).abs()

print("\n[Weight changes]")
print(f"lora_A:")
print(f"  max change:  {delta_a.max().item():.6e}")
print(f"  mean change: {delta_a.mean().item():.6e}")
print(f"  num changed (>1e-8): {(delta_a > 1e-8).sum().item()} / {delta_a.numel()}")
print(f"  num changed (>1e-6): {(delta_a > 1e-6).sum().item()} / {delta_a.numel()}")

print(f"lora_B:")
print(f"  max change:  {delta_b.max().item():.6e}")
print(f"  mean change: {delta_b.mean().item():.6e}")
print(f"  num changed (>1e-8): {(delta_b > 1e-8).sum().item()} / {delta_b.numel()}")
print(f"  num changed (>1e-6): {(delta_b > 1e-6).sum().item()} / {delta_b.numel()}")

# ============================================================================
# TEST 4: Check learning rate and update scale
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: Learning Rate and Update Scale Analysis")
print("=" * 80)

# Expected update magnitude: lr * grad
# With lr=0.001, if grad~0.01, update should be ~1e-5

print("\nExpected update magnitude:")
print(f"  Learning rate: 0.001")

# Get grad from lora_A weight parameter
lora_a_weight_param = None
for name, param in model.named_parameters():
    if 'lora_A' in name and 'query' in name and 'weight' in name and param.requires_grad:
        lora_a_weight_param = param
        break

if lora_a_weight_param is not None and lora_a_weight_param.grad is not None:
    grad_magnitude = lora_a_weight_param.grad.abs().max().item()
    expected_update = 0.001 * grad_magnitude
    print(f"  lora_A grad max: {grad_magnitude:.4e}")
    print(f"  Expected update: lr × grad = {expected_update:.4e}")
    print(f"  Actual update: {delta_a.max().item():.4e}")

    ratio = delta_a.max().item() / expected_update if expected_update > 0 else 0
    print(f"  Ratio (actual/expected): {ratio:.4f}")

    if ratio < 0.01:
        print(f"  ⚠️ Update is {100*(1-ratio):.1f}% smaller than expected!")

# ============================================================================
# TEST 5: Check tile configuration
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: Tile Configuration Details")
print("=" * 80)

def print_rpu_config(module, name):
    """Print RPU config details."""
    if hasattr(module, 'analog_module'):
        tile = module.analog_module.tile
    else:
        tile = module.tile

    if not hasattr(tile, 'rpu_config'):
        print(f"\n{name}: Not an analog tile")
        return

    config = tile.rpu_config
    print(f"\n{name}:")
    print(f"  Config type: {type(config).__name__}")

    # Check device
    if hasattr(config, 'device'):
        device = config.device
        print(f"  Device type: {type(device).__name__}")
        if hasattr(device, 'dw_min'):
            print(f"  dw_min: {device.dw_min}")

    # Check mapping
    if hasattr(config, 'mapping'):
        mapping = config.mapping
        print(f"  Mapping:")
        print(f"    weight_scaling_omega: {mapping.weight_scaling_omega}")
        print(f"    learn_out_scaling: {mapping.learn_out_scaling}")

    # Check forward
    if hasattr(config, 'forward'):
        fwd = config.forward
        print(f"  Forward:")
        print(f"    inp_res: {fwd.inp_res:.6f}")
        print(f"    out_res: {fwd.out_res:.6f}")
        print(f"    noise_management: {fwd.noise_management}")
        print(f"    bound_management: {fwd.bound_management}")

print_rpu_config(lora_a_module, "lora_A")
print_rpu_config(lora_b_module, "lora_B")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print("\nKey findings:")
print(f"1. PyTorch gradients flowing to lora_A/B: {'✓' if lora_a_weight_param.grad is not None else '✗'}")
print(f"2. lora_A max weight change: {delta_a.max().item():.6e}")
print(f"3. lora_B max weight change: {delta_b.max().item():.6e}")

if delta_a.max().item() < 1e-6:
    print("\n⚠️ PROBLEM: Weight updates are too small (<1e-6)")
    print("\nPossible causes:")
    print("- Quantization settings (inp_res, out_res) may be limiting precision")
    print("- Device dw_min setting may be too large")
    print("- Learning rate may be too small")
    print("- Gradient scaling issue in analog tile update")

print("=" * 80)
