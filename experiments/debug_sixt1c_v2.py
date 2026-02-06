#!/usr/bin/env python
"""
Debug script v2 - Check LoRA A and B separately
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear

from sixt1c_config import gen_sixt1c_lora_config
from related_functions import convert_lora_layers_only_to_analog

print("=" * 60)
print("SIXT1C DEBUG v2 - LoRA A vs B")
print("=" * 60)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

# Apply LoRA
peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1,
                         target_modules=["query"])  # Just one for simplicity
model = get_peft_model(model, peft_config)

# Check LoRA A and B BEFORE conversion
print("\n[BEFORE] LoRA weights:")
for name, param in model.named_parameters():
    if 'layer.0.attention.self.query' in name and 'weight' in name:
        if 'lora_A' in name:
            print(f"\nLoRA A: {name}")
            print(f"  shape: {param.shape}")
            print(f"  min: {param.min().item():.6f}, max: {param.max().item():.6f}")
            print(f"  mean: {param.mean().item():.6f}, std: {param.std().item():.6f}")
        elif 'lora_B' in name:
            print(f"\nLoRA B: {name}")
            print(f"  shape: {param.shape}")
            print(f"  min: {param.min().item():.6f}, max: {param.max().item():.6f}")
            print(f"  mean: {param.mean().item():.6f}, std: {param.std().item():.6f}")
            print(f"  all zeros: {(param == 0).all().item()}")

# Convert to analog
print("\n" + "=" * 60)
sixt1c_config = gen_sixt1c_lora_config()
print(f"Remap type: {sixt1c_config.remap.type}")
print(f"Clip type: {sixt1c_config.clip.type}")
print("=" * 60)

model = convert_lora_layers_only_to_analog(model, sixt1c_config)
model.to("cuda")

# Check LoRA A and B AFTER conversion
print("\n[AFTER] LoRA weights:")
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear):
        if 'layer.0.attention.self.query' in name:
            tile = module.analog_module.tile
            weights = tile.get_weights()

            layer_type = "LoRA A" if 'lora_A' in name else "LoRA B"
            print(f"\n{layer_type}: {name}")
            print(f"  shape: {weights.shape}")
            print(f"  min: {weights.min().item():.6f}, max: {weights.max().item():.6f}")
            print(f"  mean: {weights.mean().item():.6f}, std: {weights.std().item():.6f}")
            print(f"  all zeros: {(weights == 0).all().item()}")
            print(f"  has NaN: {torch.isnan(weights).any().item()}")
            print(f"  has Inf: {torch.isinf(weights).any().item()}")

            # Check out_scaling_alpha if exists
            if hasattr(module.analog_module, 'out_scaling_alpha'):
                alpha = module.analog_module.out_scaling_alpha
                print(f"  out_scaling_alpha: {alpha}")

# Check intermediate activations
print("\n" + "=" * 60)
print("Testing forward pass with hooks...")
print("=" * 60)

activations = {}

def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations[name] = {
                'min': output.min().item(),
                'max': output.max().item(),
                'mean': output.mean().item(),
                'has_nan': torch.isnan(output).any().item(),
                'has_inf': torch.isinf(output).any().item(),
            }
    return hook

# Register hooks on LoRA layers
hooks = []
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear):
        if 'layer.0.attention.self.query' in name:
            h = module.register_forward_hook(hook_fn(name))
            hooks.append(h)

# Forward pass
model.eval()
dummy_input = tokenizer("Test", return_tensors="pt", padding="max_length", max_length=32)
dummy_input = {k: v.to("cuda") for k, v in dummy_input.items()}

with torch.no_grad():
    outputs = model(**dummy_input)

# Print activations
print("\nActivations after LoRA layers:")
for name, act in activations.items():
    layer_type = "LoRA A" if 'lora_A' in name else "LoRA B"
    print(f"\n{layer_type}: {name}")
    print(f"  min: {act['min']:.4f}, max: {act['max']:.4f}")
    print(f"  mean: {act['mean']:.4f}")
    print(f"  has_nan: {act['has_nan']}, has_inf: {act['has_inf']}")

# Remove hooks
for h in hooks:
    h.remove()

print("\nFinal logits:", outputs.logits)
print("=" * 60)
