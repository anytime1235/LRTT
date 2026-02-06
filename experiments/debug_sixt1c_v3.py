#!/usr/bin/env python
"""
Debug script v3 - Check base_layer (PCM) weights
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear

from run_glue import gen_rpu_config  # PCM config
from sixt1c_config import gen_sixt1c_lora_config
from related_functions import convert_to_analog, convert_lora_layers_only_to_analog, list_linear_layers

print("=" * 60)
print("SIXT1C DEBUG v3 - Base Layer (PCM)")
print("=" * 60)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

# Check base model weights BEFORE any conversion
print("\n[1] Base model weights BEFORE conversion:")
for name, param in model.named_parameters():
    if 'layer.0.attention.self.query' in name and 'weight' in name:
        print(f"\n{name}:")
        print(f"  shape: {param.shape}")
        print(f"  min: {param.min().item():.6f}, max: {param.max().item():.6f}")
        print(f"  mean: {param.mean().item():.6f}, std: {param.std().item():.6f}")
        break

# Apply LoRA
peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1,
                         target_modules=["query"])
model = get_peft_model(model, peft_config)

# Get PCM config
pcm_config = gen_rpu_config(output_noise_level=0.0, pcm_model="PCM_Gmax25")
print(f"\n[2] PCM Config:")
print(f"  remap.type: {pcm_config.remap.type}")
print(f"  clip.type: {pcm_config.clip.type}")
print(f"  clip.sigma: {pcm_config.clip.sigma}")

# Convert ALL layers to PCM analog first (simulating sixt1c step 1)
print("\n[3] Converting ALL layers to PCM analog...")
from aihwkit.nn.conversion import convert_to_analog

# Get the base model without LoRA wrappers for conversion check
# Actually in sixt1c mode, only base_layer is converted to PCM
# Let me check the actual conversion flow

# Simpler test: just check what PCM remap does to weights
print("\n[4] Testing PCM conversion on a single Linear layer...")
test_layer = torch.nn.Linear(128, 128)
print(f"  Before - min: {test_layer.weight.min().item():.6f}, max: {test_layer.weight.max().item():.6f}")

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.tiles.inference_torch import TorchInferenceTile

analog_test = AnalogLinear.from_digital(test_layer, pcm_config, tile_module_class=TorchInferenceTile)
weights = analog_test.analog_module.tile.get_weights()
print(f"  After - min: {weights.min().item():.6f}, max: {weights.max().item():.6f}")
print(f"  Ratio: {weights.max().item() / test_layer.weight.max().item():.2f}x")

# Check out_scaling_alpha
if hasattr(analog_test.analog_module, 'out_scaling_alpha'):
    alpha = analog_test.analog_module.out_scaling_alpha
    print(f"  out_scaling_alpha: min={alpha.min().item():.4f}, max={alpha.max().item():.4f}")

# Test forward pass
print("\n[5] Testing forward pass through analog layer...")
test_input = torch.randn(1, 128)
with torch.no_grad():
    digital_out = test_layer(test_input)
    analog_out = analog_test(test_input)

print(f"  Digital output: min={digital_out.min().item():.4f}, max={digital_out.max().item():.4f}")
print(f"  Analog output: min={analog_out.min().item():.4f}, max={analog_out.max().item():.4f}")
print(f"  Ratio: {analog_out.abs().max().item() / digital_out.abs().max().item():.2f}x")

# Now check what happens with the full model
print("\n" + "=" * 60)
print("[6] Full sixt1c conversion test")
print("=" * 60)

# Fresh model
model2 = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=2
)
model2 = get_peft_model(model2, peft_config)

# Digital forward pass first
model2.eval()
dummy_input = tokenizer("Test", return_tensors="pt", padding="max_length", max_length=32)
with torch.no_grad():
    digital_logits = model2(**dummy_input).logits
print(f"\nDigital logits: {digital_logits}")

# Now do sixt1c conversion (simplified - just PCM on all layers for this test)
print("\nConverting to analog (PCM config)...")
from related_functions import convert_selected_layers_to_analog
model2 = convert_selected_layers_to_analog(model2, pcm_config)
model2.eval()

with torch.no_grad():
    analog_logits = model2(**dummy_input).logits
print(f"Analog logits: {analog_logits}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
