#!/usr/bin/env python3
"""Test with higher learning rate to see if updates become visible."""

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

# Create model
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

# Freeze
for name, param in model.named_parameters():
    if 'base_layer' in name or 'classifier' in name:
        param.requires_grad = False
    elif 'lora' in name:
        param.requires_grad = True

# Find modules
lora_a = lora_b = None
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        if 'lora_A' in name and lora_a is None:
            lora_a = module
        elif 'lora_B' in name and lora_b is None:
            lora_b = module

# Test multiple learning rates
learning_rates = [0.001, 0.01, 0.1, 1.0]

tokenizer = AutoTokenizer.from_pretrained('google/mobilebert-uncased')
inputs = tokenizer(
    ["Test sentence one.", "Test sentence two."],
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors='pt'
)
labels = torch.tensor([1, 0])

print("=" * 80)
print("TESTING DIFFERENT LEARNING RATES")
print("=" * 80)

for lr in learning_rates:
    print(f"\n{'='*80}")
    print(f"Learning Rate: {lr}")
    print(f"{'='*80}")

    # Reset model weights to initial state
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

    model = convert_base_and_lora_separately(
        model,
        base_layer_config=base_config,
        lora_config=lora_config,
        lora_trainable=True
    )

    for name, param in model.named_parameters():
        if 'base_layer' in name or 'classifier' in name:
            param.requires_grad = False
        elif 'lora' in name:
            param.requires_grad = True

    # Find modules again
    lora_a = lora_b = None
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear) and 'query' in name:
            if 'lora_A' in name and lora_a is None:
                lora_a = module
            elif 'lora_B' in name and lora_b is None:
                lora_b = module

    # Setup optimizer with this LR
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)

    # Capture before
    w_a_before = lora_a.get_weights()
    w_a_before = (w_a_before[0] if isinstance(w_a_before, tuple) else w_a_before).clone()

    w_b_before = lora_b.get_weights()
    w_b_before = (w_b_before[0] if isinstance(w_b_before, tuple) else w_b_before).clone()

    # Training step
    model.train()
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    # Capture after
    w_a_after = lora_a.get_weights()
    w_a_after = (w_a_after[0] if isinstance(w_a_after, tuple) else w_a_after).clone()

    w_b_after = lora_b.get_weights()
    w_b_after = (w_b_after[0] if isinstance(w_b_after, tuple) else w_b_after).clone()

    # Check changes
    delta_a = (w_a_after - w_a_before).abs()
    delta_b = (w_b_after - w_b_before).abs()

    print(f"Loss: {loss.item():.4f}")
    print(f"\nlora_A:")
    print(f"  max: {delta_a.max().item():.6e}")
    print(f"  mean: {delta_a.mean().item():.6e}")
    print(f"  changed (>1e-6): {(delta_a > 1e-6).sum().item()} / {delta_a.numel()}")
    print(f"  changed (>dw_min=0.002): {(delta_a > 0.002).sum().item()} / {delta_a.numel()}")

    print(f"\nlora_B:")
    print(f"  max: {delta_b.max().item():.6e}")
    print(f"  mean: {delta_b.mean().item():.6e}")
    print(f"  changed (>1e-6): {(delta_b > 1e-6).sum().item()} / {delta_b.numel()}")
    print(f"  changed (>dw_min=0.002): {(delta_b > 0.002).sum().item()} / {delta_b.numel()}")

    # Verdict
    if delta_a.max().item() > 1e-3 and delta_b.max().item() > 1e-3:
        print(f"\n✓ Good updates with lr={lr}")
    elif delta_a.max().item() > 1e-6 or delta_b.max().item() > 1e-6:
        print(f"\n⚠️  Small updates with lr={lr}")
    else:
        print(f"\n✗ No updates with lr={lr}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nRecommendation: Use the learning rate that gives")
print("updates > dw_min (0.001981) for reliable training.")
print("=" * 80)
