#!/usr/bin/env python3
"""
최종 테스트: 업데이트된 config로 base_layer frozen 확인

검증 항목:
1. Config 설정이 올바른지 확인
2. base_layer weight가 변하지 않는지 확인
3. lora_A/B는 정상 업데이트되는지 확인
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
print("FINAL TEST: Updated Config Verification")
print("=" * 80)

# Create LoRA model
print("\n[Creating LoRA model (2 layers)]")
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

# Convert to analog with NEW config
print("\n[Converting to analog with UPDATED config]")
print("  base_layer: gen_softbounds_base_layer_config() - TorchInferenceRPUConfig")
print("  lora_A/B: gen_sixt1c_lora_config_trainable() - SingleRPUConfig")

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

print("✓ Model converted")

# Verify config settings
print("\n" + "=" * 80)
print("STEP 1: Verify Config Settings")
print("=" * 80)

print("\n[base_layer config verification]")
print(f"  Type: TorchInferenceRPUConfig")
print(f"  Expected settings:")
print(f"    - modifier.std_dev: 0.0 (noise OFF)")
print(f"    - forward.out_noise: 0.0 (noise OFF)")
print(f"    - forward.is_perfect: False")
print(f"    - remap.type: CHANNELWISE_SYMMETRIC")
print(f"    - clip.type: LAYER_GAUSSIAN, sigma=3")
print(f"    - inp_res/out_res: 8-bit quantization")
print(f"    - noise_model: None")

# Check actual config
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'base_layer' in name and 'query' in name:
        print(f"\n[Actual config from base_layer]")
        if hasattr(module, 'analog_module'):
            tile = module.analog_module.tile
        else:
            tile = module.tile

        # Check if it's inference tile
        tile_type = type(tile).__name__
        print(f"  Tile type: {tile_type}")

        if 'Inference' in tile_type:
            print(f"  ✓ Using InferenceTile (correct)")
        else:
            print(f"  ✗ NOT using InferenceTile (wrong!)")

        break

# Optimizer
optimizer = AnalogSGD(model.parameters(), lr=0.001)
optimizer.regroup_param_groups(model)
print("\n✓ Optimizer created")

# Collect modules
base_layers = []
lora_a_layers = []
lora_b_layers = []

for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        if 'base_layer' in name:
            base_layers.append((name, module))
        elif 'lora_A' in name:
            lora_a_layers.append((name, module))
        elif 'lora_B' in name:
            lora_b_layers.append((name, module))

print(f"\nFound modules:")
print(f"  base_layer: {len(base_layers)}")
print(f"  lora_A: {len(lora_a_layers)}")
print(f"  lora_B: {len(lora_b_layers)}")

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

print("\n" + "=" * 80)
print("STEP 2: Training Test (3 steps)")
print("=" * 80)

all_results = {
    'base_layer': [],
    'lora_A': [],
    'lora_B': []
}

for step in range(3):
    print(f"\n--- Step {step + 1} ---")

    # Capture weights before
    weights_before = {}
    for name, module in base_layers + lora_a_layers + lora_b_layers:
        w = module.get_weights()
        weights_before[name] = (w[0] if isinstance(w, tuple) else w).clone()

    # Forward + Backward + Step
    model.train()
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")

    # Check weight changes
    print(f"\n  Weight changes:")

    # base_layer
    for name, module in base_layers:
        layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
        w = module.get_weights()
        w_after = w[0] if isinstance(w, tuple) else w
        delta = (w_after - weights_before[name]).abs()

        max_change = delta.max().item()
        mean_change = delta.mean().item()
        num_changed = (delta > 1e-8).sum().item()

        all_results['base_layer'].append({
            'step': step + 1,
            'layer': layer_idx,
            'max': max_change,
            'mean': mean_change,
            'num_changed': num_changed
        })

        status = "✓" if max_change < 1e-8 else "✗"
        print(f"    base_layer L{layer_idx}: max={max_change:.6f}, changed={num_changed}/{delta.numel()} {status}")

    # lora_A (first one only)
    if lora_a_layers:
        name, module = lora_a_layers[0]
        layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
        w = module.get_weights()
        w_after = w[0] if isinstance(w, tuple) else w
        delta = (w_after - weights_before[name]).abs()

        max_change = delta.max().item()
        num_changed = (delta > 1e-8).sum().item()

        all_results['lora_A'].append({
            'step': step + 1,
            'max': max_change,
            'num_changed': num_changed
        })

        status = "✓" if max_change > 1e-6 else "✗"
        print(f"    lora_A L{layer_idx}: max={max_change:.6f}, changed={num_changed}/{delta.numel()} {status}")

    # lora_B (first one only)
    if lora_b_layers:
        name, module = lora_b_layers[0]
        layer_idx = name.split('layer.')[1].split('.')[0] if 'layer.' in name else '?'
        w = module.get_weights()
        w_after = w[0] if isinstance(w, tuple) else w
        delta = (w_after - weights_before[name]).abs()

        max_change = delta.max().item()
        num_changed = (delta > 1e-8).sum().item()

        all_results['lora_B'].append({
            'step': step + 1,
            'max': max_change,
            'num_changed': num_changed
        })

        status = "✓" if max_change > 1e-6 else "✗"
        print(f"    lora_B L{layer_idx}: max={max_change:.6f}, changed={num_changed}/{delta.numel()} {status}")

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

# Analyze results
base_all_frozen = all(r['max'] < 1e-8 for r in all_results['base_layer'])
lora_a_all_updated = all(r['max'] > 1e-6 for r in all_results['lora_A']) if all_results['lora_A'] else False
lora_b_all_updated = all(r['max'] > 1e-6 for r in all_results['lora_B']) if all_results['lora_B'] else False

print("\n[base_layer - Should be FROZEN]")
if base_all_frozen:
    print("  ✓✓✓ ALL base_layer weights FROZEN across all steps!")
    print("  → TorchInferenceRPUConfig successfully prevents updates!")
else:
    print("  ✗✗✗ Some base_layer weights changed!")
    for r in all_results['base_layer']:
        if r['max'] > 1e-8:
            print(f"      Step {r['step']}, Layer {r['layer']}: max change = {r['max']:.6f}")

print("\n[lora_A - Should be UPDATED]")
if lora_a_all_updated:
    print("  ✓ lora_A updating correctly")
else:
    print("  ✗ lora_A NOT updating (problem!)")
    for r in all_results['lora_A']:
        print(f"      Step {r['step']}: max change = {r['max']:.6f}")

print("\n[lora_B - Should be UPDATED]")
if lora_b_all_updated:
    print("  ✓ lora_B updating correctly")
else:
    print("  ✗ lora_B NOT updating (problem!)")
    for r in all_results['lora_B']:
        print(f"      Step {r['step']}: max change = {r['max']:.6f}")

print("\n" + "=" * 80)
print("OVERALL VERDICT")
print("=" * 80)

if base_all_frozen and lora_a_all_updated and lora_b_all_updated:
    print("\n🎉🎉🎉 SUCCESS! 🎉🎉🎉")
    print("\n✓ base_layer: FROZEN (no updates)")
    print("✓ lora_A/B: TRAINABLE (updating correctly)")
    print("\n→ The issue has been RESOLVED!")
    print("→ Safe to run experiments with this config!")
else:
    print("\n❌ FAILED ❌")
    if not base_all_frozen:
        print("\n✗ base_layer still updating (issue NOT resolved)")
    if not lora_a_all_updated:
        print("✗ lora_A not updating")
    if not lora_b_all_updated:
        print("✗ lora_B not updating")
    print("\n→ Need to investigate further!")

print("=" * 80)
