#!/usr/bin/env python3
"""Check what get_weights() actually returns - tile weights or scaled weights?"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear

from sixt1c_config import (
    gen_sixt1c_lora_config_trainable,
    gen_softbounds_base_layer_config
)
from smart_conversion import convert_base_and_lora_separately

# Create minimal model
model_config = AutoConfig.from_pretrained('google/mobilebert-uncased')
model_config.num_labels = 2
model_config.num_hidden_layers = 1

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

# Convert
base_config = gen_softbounds_base_layer_config()
lora_config = gen_sixt1c_lora_config_trainable()

model = convert_base_and_lora_separately(
    model,
    base_layer_config=base_config,
    lora_config=lora_config,
    lora_trainable=True
)

# Find lora_A
lora_a = None
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'lora_A' in name and 'query' in name:
        lora_a = module
        lora_a_name = name
        break

print("=" * 80)
print("CHECKING get_weights() OUTPUT")
print("=" * 80)

print(f"\nModule: {lora_a_name}")

# Method 1: module.get_weights()
w1 = lora_a.get_weights()
w1 = w1[0] if isinstance(w1, tuple) else w1

print(f"\n[Method 1] module.get_weights():")
print(f"  Shape: {w1.shape}")
print(f"  Norm: {w1.norm().item():.4f}")
print(f"  Range: [{w1.min().item():.4f}, {w1.max().item():.4f}]")

# Method 2: analog_module.tile.get_weights()
w2 = lora_a.analog_module.tile.get_weights()

print(f"\n[Method 2] analog_module.tile.get_weights():")
print(f"  Shape: {w2.shape}")
print(f"  Norm: {w2.norm().item():.4f}")
print(f"  Range: [{w2.min().item():.4f}, {w2.max().item():.4f}]")

# Check if they're the same
print(f"\n[Comparison]:")
diff = (w1 - w2).abs()
print(f"  Max difference: {diff.max().item():.6e}")
print(f"  Are they the same? {torch.allclose(w1, w2)}")

# Check out_scaling_alpha
if hasattr(lora_a.analog_module, 'out_scaling_alpha'):
    alpha = lora_a.analog_module.out_scaling_alpha
    print(f"\n[out_scaling_alpha]:")
    print(f"  Shape: {alpha.shape}")
    print(f"  Values: {alpha.detach().cpu().numpy()}")
    print(f"  Mean: {alpha.mean().item():.6f}")
    print(f"  Range: [{alpha.min().item():.6f}, {alpha.max().item():.6f}]")

    # Compute scaled weights manually
    w2_scaled = w2 * alpha.view(1, -1)  # Broadcast alpha columnwise
    print(f"\n[Manual scaling: tile_weights * alpha]:")
    print(f"  Shape: {w2_scaled.shape}")
    print(f"  Norm: {w2_scaled.norm().item():.4f}")
    print(f"  Max diff from get_weights(): {(w1 - w2_scaled).abs().max().item():.6e}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if torch.allclose(w1, w2):
    print("\n✓ get_weights() returns RAW tile weights (no scaling)")
    print("  → Weight changes we measure ARE the analog tile weight changes")
else:
    print("\n✓ get_weights() returns SCALED weights (tile × alpha)")
    print("  → Weight changes we measure include BOTH tile AND alpha changes")
    print("  → To measure pure tile updates, use analog_module.tile.get_weights()")

print("=" * 80)
