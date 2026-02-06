#!/usr/bin/env python
"""
Debug script to investigate sixt1c LoRA conversion issues.
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear

# Import sixt1c config
from sixt1c_config import gen_sixt1c_lora_config
from related_functions import convert_lora_layers_only_to_analog, list_lora_layers

print("=" * 60)
print("SIXT1C LORA DEBUG")
print("=" * 60)

# 1. Load model
print("\n[1] Loading MobileBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased",
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

# 2. Apply LoRA
print("\n[2] Applying LoRA...")
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],  # Simplified for debugging
)
model = get_peft_model(model, peft_config)

# 3. Check LoRA weights BEFORE conversion
print("\n[3] LoRA weights BEFORE analog conversion:")
lora_layers = list_lora_layers(model)
print(f"Found {len(lora_layers)} LoRA layers")

for name, param in model.named_parameters():
    if 'lora_A' in name or 'lora_B' in name:
        if 'weight' in name:
            w = param.data
            print(f"  {name}:")
            print(f"    shape: {w.shape}")
            print(f"    min: {w.min().item():.6f}, max: {w.max().item():.6f}")
            print(f"    mean: {w.mean().item():.6f}, std: {w.std().item():.6f}")
            print(f"    zeros: {(w == 0).sum().item()} / {w.numel()}")
        break  # Just show first one

# 4. Get sixt1c config and print it
print("\n[4] Sixt1c RPU Config:")
sixt1c_config = gen_sixt1c_lora_config(
    dt_batch_sec=1.0,
    include_retention=True,
    output_noise_level=0.0,
)
print(f"  remap.type: {sixt1c_config.remap.type}")
print(f"  clip.type: {sixt1c_config.clip.type}")
print(f"  clip.sigma: {sixt1c_config.clip.sigma if hasattr(sixt1c_config.clip, 'sigma') else 'N/A'}")
print(f"  forward.inp_res: {sixt1c_config.forward.inp_res}")
print(f"  forward.out_res: {sixt1c_config.forward.out_res}")
print(f"  mapping.weight_scaling_omega: {sixt1c_config.mapping.weight_scaling_omega}")

# 5. Convert to analog
print("\n[5] Converting LoRA layers to analog...")
model = convert_lora_layers_only_to_analog(model, sixt1c_config)
model.to("cuda")

# 6. Check weights AFTER conversion
print("\n[6] LoRA weights AFTER analog conversion:")
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear):
        if 'lora_A' in name or 'lora_B' in name:
            tile = module.analog_module.tile
            weights = tile.get_weights()
            print(f"  {name}:")
            print(f"    shape: {weights.shape}")
            print(f"    min: {weights.min().item():.6f}, max: {weights.max().item():.6f}")
            print(f"    mean: {weights.mean().item():.6f}, std: {weights.std().item():.6f}")
            print(f"    zeros: {(weights == 0).sum().item()} / {weights.numel()}")
            print(f"    has NaN: {torch.isnan(weights).any().item()}")
            print(f"    has Inf: {torch.isinf(weights).any().item()}")
            break  # Just show first one

# 7. Test forward pass
print("\n[7] Testing forward pass...")
model.train()
dummy_input = tokenizer("This is a test sentence.", return_tensors="pt", padding="max_length", max_length=128)
dummy_input = {k: v.to("cuda") for k, v in dummy_input.items()}
dummy_labels = torch.tensor([0]).to("cuda")

try:
    outputs = model(**dummy_input, labels=dummy_labels)
    loss = outputs.loss
    logits = outputs.logits
    print(f"  Loss: {loss.item()}")
    print(f"  Loss is NaN: {torch.isnan(loss).item()}")
    print(f"  Loss is Inf: {torch.isinf(loss).item()}")
    print(f"  Logits: {logits}")
    print(f"  Logits has NaN: {torch.isnan(logits).any().item()}")
    print(f"  Logits has Inf: {torch.isinf(logits).any().item()}")
except Exception as e:
    print(f"  Forward pass ERROR: {e}")

# 8. Test backward pass
print("\n[8] Testing backward pass...")
try:
    loss.backward()
    print("  Backward pass completed")

    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None and ('lora_A' in name or 'lora_B' in name):
            grad = param.grad
            print(f"  {name}:")
            print(f"    grad shape: {grad.shape}")
            print(f"    grad min: {grad.min().item():.6f}, max: {grad.max().item():.6f}")
            print(f"    grad has NaN: {torch.isnan(grad).any().item()}")
            print(f"    grad has Inf: {torch.isinf(grad).any().item()}")
            break  # Just show first one
except Exception as e:
    print(f"  Backward pass ERROR: {e}")

# 9. Test multiple forward passes (simulating training)
print("\n[9] Testing multiple forward passes...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

for step in range(5):
    optimizer.zero_grad()
    outputs = model(**dummy_input, labels=dummy_labels)
    loss = outputs.loss
    print(f"  Step {step}: loss={loss.item():.4f}, is_nan={torch.isnan(loss).item()}")

    if torch.isnan(loss):
        print("  NaN detected! Stopping.")
        break

    loss.backward()

    # Check grad norm
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"    grad_norm={total_norm:.4f}, is_nan={np.isnan(total_norm)}")

    if np.isnan(total_norm):
        print("  NaN gradient detected! Stopping.")
        break

    optimizer.step()

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
