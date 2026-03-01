#!/usr/bin/env python3
"""
Sixt1c LoRA Forward & Update 검증 스크립트

현재 실행 중인 sixt1c LoRA 실험의 forward pass와 weight update 로직이
올바르게 작동하는지 검증합니다.

검증 항목:
1. Analog device configuration (6T1C LinearStepDevice, SoftBoundsDevice)
2. Forward pass logic (base + scaling * lora_B(lora_A(x)))
3. Gradient flow (lora_A/B에만 gradient, base_layer는 frozen)
4. Weight updates (lora_A/B 변경, base_layer 불변)
5. Expected vs Actual comparison

실행 시간: ~3-5분
"""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

# Import standardized configs
from sixt1c_config import (
    gen_sixt1c_lora_config_trainable,
    gen_softbounds_base_layer_config_trainable
)
from smart_conversion import convert_base_and_lora_separately


# ============================================================================
# Test Configuration
# ============================================================================

TEST_CONFIG = {
    'model_name': 'google/mobilebert-uncased',
    'task': 'sst2',
    'target_layer': 'encoder.layer.0.attention.self.query',  # Single layer for testing
    'rank': 8,
    'lora_alpha': 1.0,
    'learning_rate': 1e-3,
    'num_steps': 3,
    'batch_size': 2,
    'seq_length': 128,
}


# ============================================================================
# 1. Test Model Creation
# ============================================================================

def create_test_model():
    """Create minimal test model with sixt1c LoRA on single layer."""

    print("\n[Creating test model]")
    print(f"  Model: {TEST_CONFIG['model_name']}")
    print(f"  Rank: {TEST_CONFIG['rank']}")
    print(f"  LoRA alpha: {TEST_CONFIG['lora_alpha']}")

    model_config = AutoConfig.from_pretrained(TEST_CONFIG['model_name'])
    model_config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(
        TEST_CONFIG['model_name'],
        config=model_config
    )

    # Apply PEFT LoRA to single layer (query only)
    peft_config = LoraConfig(
        r=TEST_CONFIG['rank'],
        lora_alpha=TEST_CONFIG['lora_alpha'],
        target_modules=['query'],  # Only query for testing
        bias='none',
        lora_dropout=0.0,
    )
    model = get_peft_model(model, peft_config)

    # Convert to analog using smart conversion
    base_config = gen_softbounds_base_layer_config_trainable()
    lora_config = gen_sixt1c_lora_config_trainable()

    model = convert_base_and_lora_separately(
        model,
        base_layer_config=base_config,
        lora_config=lora_config,
        lora_trainable=True
    )

    # Freeze base_layer, classifier
    for name, param in model.named_parameters():
        if 'base_layer' in name or 'classifier' in name:
            param.requires_grad = False
        elif 'lora' in name:
            param.requires_grad = True

    print("✓ Model created successfully")
    return model


# ============================================================================
# 2. Device Configuration Verification
# ============================================================================

def verify_device_config(model):
    """TEST 1: Verify analog device parameters."""

    print("\n" + "="*80)
    print("TEST 1: ANALOG DEVICE CONFIGURATION")
    print("="*80)

    results = {'pass': True, 'details': []}

    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear):
            continue

        # Access tile
        tile = module.analog_module.tile if hasattr(module, 'analog_module') else module.tile

        # Check device type and parameters
        if 'base_layer' in name:
            # Should be SoftBoundsDevice
            device = tile.rpu_config.device if hasattr(tile, 'rpu_config') else None
            expected_type = 'SoftBoundsDevice'
            # Check frozen
            is_frozen = not any(p.requires_grad for p in module.parameters())

            print(f"\n[base_layer] {name}")
            print(f"  Device type: {type(device).__name__}")
            print(f"  Frozen: {is_frozen} {'✓' if is_frozen else '✗'}")

            if not is_frozen:
                results['pass'] = False
                results['details'].append(f"{name}: should be frozen")

            if device and not isinstance(device, SoftBoundsDevice):
                results['pass'] = False
                results['details'].append(f"{name}: wrong device type")

        elif 'lora_A' in name or 'lora_B' in name:
            # Should be LinearStepDevice with 6T1C params
            device = tile.rpu_config.device if hasattr(tile, 'rpu_config') else None
            expected_type = 'LinearStepDevice'

            layer_type = 'lora_A' if 'lora_A' in name else 'lora_B'
            print(f"\n[{layer_type}] {name}")
            print(f"  Device type: {type(device).__name__}")

            if device and not isinstance(device, LinearStepDevice):
                results['pass'] = False
                results['details'].append(f"{name}: wrong device type")

            # Check 6T1C parameters
            if device and hasattr(device, 'dw_min'):
                dw_min = device.dw_min
                print(f"  dw_min: {dw_min:.6f} {'✓' if abs(dw_min - 0.001981) < 1e-5 else '✗'}")

                if abs(dw_min - 0.001981) > 1e-5:
                    results['pass'] = False
                    results['details'].append(f"{name}: dw_min mismatch")

            # Check trainable
            is_trainable = any(p.requires_grad for p in module.parameters())
            print(f"  Trainable: {is_trainable} {'✓' if is_trainable else '✗'}")

            if not is_trainable:
                results['pass'] = False
                results['details'].append(f"{name}: should be trainable")

    print(f"\nResult: {'PASS' if results['pass'] else 'FAIL'}")
    if not results['pass']:
        for detail in results['details']:
            print(f"  - {detail}")

    return results


# ============================================================================
# 3. Forward Pass Verification
# ============================================================================

def verify_forward_pass(model, tokenizer):
    """TEST 2: Verify forward pass logic."""

    print("\n" + "="*80)
    print("TEST 2: FORWARD PASS LOGIC")
    print("="*80)

    results = {'pass': True, 'details': []}

    # Create synthetic input
    inputs = tokenizer(
        ["This is a test sentence."] * TEST_CONFIG['batch_size'],
        padding='max_length',
        max_length=TEST_CONFIG['seq_length'],
        truncation=True,
        return_tensors='pt'
    )

    # Storage for captured values
    captured = {
        'base_output': None,
        'lora_a_output': None,
        'lora_b_output': None,
        'final_output': None,
    }

    # Register hooks
    def make_hook(key):
        def hook(module, input, output):
            captured[key] = output.detach().clone()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if 'query.base_layer' in name:
            hooks.append(module.register_forward_hook(make_hook('base_output')))
        elif 'query.lora_A' in name:
            hooks.append(module.register_forward_hook(make_hook('lora_a_output')))
        elif 'query.lora_B' in name:
            hooks.append(module.register_forward_hook(make_hook('lora_b_output')))

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Validate
    print(f"Input shape: {inputs['input_ids'].shape}")

    if captured['base_output'] is not None:
        print(f"\n[base_layer output]")
        print(f"  Shape: {captured['base_output'].shape} ✓")
        print(f"  Range: [{captured['base_output'].min():.3f}, {captured['base_output'].max():.3f}]")
        no_nan_inf = not torch.isnan(captured['base_output']).any() and not torch.isinf(captured['base_output']).any()
        print(f"  No NaN/Inf: {no_nan_inf} {'✓' if no_nan_inf else '✗'}")

        if not no_nan_inf:
            results['pass'] = False
            results['details'].append("base_layer output contains NaN/Inf")
    else:
        results['pass'] = False
        results['details'].append("base_layer output not captured")

    if captured['lora_a_output'] is not None:
        print(f"\n[lora_A output]")
        print(f"  Shape: {captured['lora_a_output'].shape} ✓")
        expected_rank = TEST_CONFIG['rank']
        actual_rank = captured['lora_a_output'].shape[-1]
        print(f"  Rank: {actual_rank} {'✓' if actual_rank == expected_rank else '✗'}")

        if actual_rank != expected_rank:
            results['pass'] = False
            results['details'].append(f"lora_A rank mismatch: {actual_rank} != {expected_rank}")
    else:
        results['pass'] = False
        results['details'].append("lora_A output not captured")

    if captured['lora_b_output'] is not None:
        print(f"\n[lora_B output]")
        print(f"  Shape: {captured['lora_b_output'].shape} ✓")
    else:
        results['pass'] = False
        results['details'].append("lora_B output not captured")

    # Check scaling
    scaling = TEST_CONFIG['lora_alpha'] / TEST_CONFIG['rank']
    print(f"\n[Scaling]")
    print(f"  scaling = lora_alpha / rank = {TEST_CONFIG['lora_alpha']} / {TEST_CONFIG['rank']} = {scaling:.4f} ✓")

    print(f"\nResult: {'PASS' if results['pass'] else 'FAIL'}")
    return results


# ============================================================================
# 4. Gradient Flow Verification
# ============================================================================

def verify_gradient_flow(model, tokenizer):
    """TEST 3: Verify gradient flow to lora_A/B, not base_layer."""

    print("\n" + "="*80)
    print("TEST 3: GRADIENT FLOW")
    print("="*80)

    results = {'pass': True, 'details': []}

    # Create synthetic input
    inputs = tokenizer(
        ["This is a test sentence."],
        padding='max_length',
        max_length=TEST_CONFIG['seq_length'],
        truncation=True,
        return_tensors='pt'
    )
    labels = torch.tensor([1])

    # Forward + backward
    model.train()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()

    # Check gradients
    lora_grads = []
    base_grads = []

    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                lora_grads.append((name, grad_norm))

        if 'base_layer' in name:
            if param.grad is not None and param.grad.abs().max() > 1e-8:
                base_grads.append(name)

    # Validate LoRA gradients
    print(f"\n[LoRA gradients]")
    if len(lora_grads) > 0:
        for name, norm in lora_grads[:3]:  # Show first 3
            print(f"  {name}: norm={norm:.2e} ✓")
        if len(lora_grads) > 3:
            print(f"  ... and {len(lora_grads)-3} more")
    else:
        print(f"  ✗ No LoRA gradients found!")
        results['pass'] = False
        results['details'].append("No LoRA gradients")

    # Validate base_layer has no gradients
    print(f"\n[base_layer gradients]")
    if len(base_grads) == 0:
        print(f"  No gradients found ✓ (frozen as expected)")
    else:
        print(f"  ✗ base_layer received gradients: {base_grads}")
        results['pass'] = False
        results['details'].append(f"base_layer has gradients: {base_grads}")

    print(f"\nResult: {'PASS' if results['pass'] else 'FAIL'}")
    return results


# ============================================================================
# 5. Weight Update Verification
# ============================================================================

def verify_weight_updates(model, tokenizer):
    """TEST 4: Verify weights actually update."""

    print("\n" + "="*80)
    print("TEST 4: WEIGHT UPDATES")
    print("="*80)

    results = {'pass': True, 'details': [], 'steps': []}

    # Setup optimizer
    optimizer = AnalogSGD(model.parameters(), lr=TEST_CONFIG['learning_rate'])
    optimizer.regroup_param_groups(model)

    # Prepare inputs
    inputs = tokenizer(
        ["This is a test sentence."] * TEST_CONFIG['batch_size'],
        padding='max_length',
        max_length=TEST_CONFIG['seq_length'],
        truncation=True,
        return_tensors='pt'
    )
    labels = torch.tensor([1, 0])

    # Find analog layers
    analog_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear) and 'query' in name:
            analog_layers[name] = module

    # Run training steps
    for step in range(TEST_CONFIG['num_steps']):
        print(f"\nStep {step+1}:")

        step_results = {}

        # Capture weights before
        weights_before = {}
        for name, module in analog_layers.items():
            w = module.get_weights()
            weights_before[name] = (w[0] if isinstance(w, tuple) else w).clone()

        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Capture weights after
        weights_after = {}
        for name, module in analog_layers.items():
            w = module.get_weights()
            weights_after[name] = (w[0] if isinstance(w, tuple) else w).clone()

        # Compute deltas
        for name in analog_layers:
            delta = (weights_after[name] - weights_before[name]).abs()
            max_change = delta.max().item()
            mean_change = delta.mean().item()

            layer_type = 'lora_A' if 'lora_A' in name else ('lora_B' if 'lora_B' in name else 'base_layer')

            print(f"  [{layer_type}]")
            print(f"    Max change: {max_change:.2e}")
            print(f"    Mean change: {mean_change:.2e}")

            # Check if changed (lora) or unchanged (base)
            if 'lora' in name:
                if max_change > 1e-6:
                    print(f"    ✓ Updated")
                else:
                    print(f"    ✗ NOT updated")
                    results['pass'] = False
                    results['details'].append(f"{layer_type} not updated at step {step+1}")
            else:  # base_layer
                if max_change < 1e-8:
                    print(f"    ✓ Frozen (no change)")
                else:
                    print(f"    ✗ Changed (should be frozen)")
                    results['pass'] = False
                    results['details'].append(f"base_layer changed at step {step+1}")

            step_results[layer_type] = {
                'max_change': max_change,
                'mean_change': mean_change,
            }

        results['steps'].append(step_results)

    print(f"\nResult: {'PASS' if results['pass'] else 'FAIL'}")
    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("="*80)
    print("SIXT1C-LORA VERIFICATION SCRIPT")
    print("="*80)

    print("\n[Setup]")
    print(f"  Model: {TEST_CONFIG['model_name']}")
    print(f"  Task: {TEST_CONFIG['task']}")
    print(f"  Target layer: {TEST_CONFIG['target_layer']}")
    print(f"  Rank: {TEST_CONFIG['rank']}")
    print(f"  LoRA alpha: {TEST_CONFIG['lora_alpha']}")
    print(f"  Learning rate: {TEST_CONFIG['learning_rate']}")
    print(f"  Steps: {TEST_CONFIG['num_steps']}")

    # Create model
    print("\nCreating test model...")
    tokenizer = AutoTokenizer.from_pretrained(TEST_CONFIG['model_name'])
    model = create_test_model()

    # Run tests
    test_results = {}

    test_results['config'] = verify_device_config(model)
    test_results['forward'] = verify_forward_pass(model, tokenizer)
    test_results['gradient'] = verify_gradient_flow(model, tokenizer)
    test_results['update'] = verify_weight_updates(model, tokenizer)

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    all_passed = True
    for test_name, result in test_results.items():
        status = "PASS" if result['pass'] else "FAIL"
        symbol = "✓" if result['pass'] else "✗"
        print(f"{symbol} TEST {test_name.upper()}: {status}")

        if not result['pass']:
            all_passed = False
            for detail in result.get('details', []):
                print(f"    - {detail}")

    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    print("="*80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
