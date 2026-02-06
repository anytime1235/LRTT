#!/usr/bin/env python
"""
Debug script v4 - Why are digital logits so huge?
"""

import os
os.environ["WANDB_DISABLED"] = "true"

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model

print("=" * 60)
print("DEBUG v4 - Digital Model Logits Investigation")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
dummy_input = tokenizer("Test sentence.", return_tensors="pt", padding="max_length", max_length=32)

# Test 1: Base model without LoRA
print("\n[1] Base MobileBERT (no LoRA):")
model1 = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=2
)
model1.eval()
with torch.no_grad():
    out1 = model1(**dummy_input)
print(f"  Logits: {out1.logits}")
print(f"  Logits range: [{out1.logits.min().item():.2f}, {out1.logits.max().item():.2f}]")

# Test 2: With LoRA (lora_alpha=32, r=8)
print("\n[2] MobileBERT + LoRA (alpha=32, r=8):")
model2 = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=2
)
peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.0,
                         target_modules=["query", "key", "value"])
model2 = get_peft_model(model2, peft_config)
model2.eval()
with torch.no_grad():
    out2 = model2(**dummy_input)
print(f"  Logits: {out2.logits}")
print(f"  Logits range: [{out2.logits.min().item():.2f}, {out2.logits.max().item():.2f}]")
print(f"  LoRA scaling: alpha/r = 32/8 = 4")

# Test 3: With LoRA (lora_alpha=8, r=8) - scaling = 1
print("\n[3] MobileBERT + LoRA (alpha=8, r=8) - scaling=1:")
model3 = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=2
)
peft_config3 = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0,
                         target_modules=["query", "key", "value"])
model3 = get_peft_model(model3, peft_config3)
model3.eval()
with torch.no_grad():
    out3 = model3(**dummy_input)
print(f"  Logits: {out3.logits}")
print(f"  Logits range: [{out3.logits.min().item():.2f}, {out3.logits.max().item():.2f}]")

# Test 4: Check LoRA output contribution
print("\n[4] Checking LoRA output contribution:")
print("  Since LoRA B is initialized to 0, LoRA output should be 0")
print("  Therefore logits should be same as base model...")

# Check if logits are same
print(f"\n  Base model logits: {out1.logits}")
print(f"  LoRA model logits: {out2.logits}")
print(f"  Difference: {(out2.logits - out1.logits).abs().max().item():.6f}")

# Test 5: Check intermediate activations
print("\n[5] Checking intermediate activations:")
model5 = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=2
)
peft_config5 = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.0,
                         target_modules=["query"])
model5 = get_peft_model(model5, peft_config5)
model5.eval()

# Hook to capture activations
activations = {}
def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output
    return hook

# Register hooks
hooks = []
for name, module in model5.named_modules():
    if 'layer.0.attention' in name and 'lora' not in name.lower():
        if hasattr(module, 'weight') or 'self' in name:
            h = module.register_forward_hook(hook_fn(name))
            hooks.append(h)

with torch.no_grad():
    out5 = model5(**dummy_input)

print(f"  Captured {len(activations)} activations")
for name, act in list(activations.items())[:5]:
    if isinstance(act, torch.Tensor):
        print(f"  {name}: shape={act.shape}, range=[{act.min().item():.2f}, {act.max().item():.2f}]")
    elif isinstance(act, tuple) and len(act) > 0:
        a = act[0]
        print(f"  {name}: shape={a.shape}, range=[{a.min().item():.2f}, {a.max().item():.2f}]")

# Remove hooks
for h in hooks:
    h.remove()

# Test 6: Check pooler and classifier
print("\n[6] Checking pooler and classifier:")
model6 = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=2
)
model6.eval()

# Get pooler output
with torch.no_grad():
    outputs = model6.mobilebert(**dummy_input)
    hidden_states = outputs.last_hidden_state
    pooled = model6.mobilebert.pooler(hidden_states)
    pooled_dropped = model6.dropout(pooled)
    logits = model6.classifier(pooled_dropped)

print(f"  Hidden states: shape={hidden_states.shape}, range=[{hidden_states.min().item():.4f}, {hidden_states.max().item():.4f}]")
print(f"  Pooled output: shape={pooled.shape}, range=[{pooled.min().item():.4f}, {pooled.max().item():.4f}]")
print(f"  Classifier weight: shape={model6.classifier.weight.shape}, range=[{model6.classifier.weight.min().item():.4f}, {model6.classifier.weight.max().item():.4f}]")
print(f"  Final logits: {logits}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
