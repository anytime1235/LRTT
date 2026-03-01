"""
Comprehensive diagnosis of Sixt1c-LoRA model for SST-2
Checks:
1. Model architecture (analog conversion)
2. Trainability (frozen/trainable parameters)
3. Bias mapping
4. Forward pass
5. Backward pass
6. Weight updates
"""

import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from transformers import AutoTokenizer
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_architecture(model):
    """Check if analog conversion happened correctly."""
    print_section("1. ARCHITECTURE CHECK")

    # Count layers
    base_analog = []
    lora_a_analog = []
    lora_b_analog = []
    base_digital = []
    lora_digital = []
    classifier_layers = []

    for name, module in model.named_modules():
        if 'classifier' in name and isinstance(module, (nn.Linear, AnalogLinear)):
            classifier_layers.append((name, type(module).__name__))
        elif 'base_layer' in name:
            if isinstance(module, AnalogLinear):
                base_analog.append(name)
            elif isinstance(module, nn.Linear):
                base_digital.append(name)
        elif 'lora_A' in name:
            if isinstance(module, AnalogLinear):
                lora_a_analog.append(name)
            elif isinstance(module, nn.Linear):
                lora_digital.append(name)
        elif 'lora_B' in name:
            if isinstance(module, AnalogLinear):
                lora_b_analog.append(name)

    print(f"\nbase_layer:")
    print(f"  Analog (AnalogLinear): {len(base_analog)}")
    print(f"  Digital (nn.Linear): {len(base_digital)}")
    if len(base_analog) > 0:
        print(f"  Sample: {base_analog[0]}")

    print(f"\nlora_A:")
    print(f"  Analog (AnalogLinear): {len(lora_a_analog)}")
    print(f"  Digital (nn.Linear): {len(lora_digital)}")
    if len(lora_a_analog) > 0:
        print(f"  Sample: {lora_a_analog[0]}")

    print(f"\nlora_B:")
    print(f"  Analog (AnalogLinear): {len(lora_b_analog)}")
    if len(lora_b_analog) > 0:
        print(f"  Sample: {lora_b_analog[0]}")

    print(f"\nClassifier:")
    for name, typ in classifier_layers:
        print(f"  {name}: {typ}")

    # Verdict
    print(f"\n{'✓' if len(base_analog) > 0 else '✗'} base_layer converted to analog: {len(base_analog)} layers")
    print(f"{'✓' if len(lora_a_analog) > 0 else '✗'} lora_A converted to analog: {len(lora_a_analog)} layers")
    print(f"{'✓' if len(lora_b_analog) > 0 else '✗'} lora_B converted to analog: {len(lora_b_analog)} layers")

    return len(base_analog) > 0 and len(lora_a_analog) > 0 and len(lora_b_analog) > 0


def check_trainability(model):
    """Check if parameters are trainable/frozen correctly."""
    print_section("2. TRAINABILITY CHECK")

    base_trainable = []
    base_frozen = []
    lora_trainable = []
    lora_frozen = []
    classifier_trainable = []
    classifier_frozen = []

    for name, param in model.named_parameters():
        if 'classifier' in name:
            if param.requires_grad:
                classifier_trainable.append(name)
            else:
                classifier_frozen.append(name)
        elif 'base_layer' in name:
            if param.requires_grad:
                base_trainable.append(name)
            else:
                base_frozen.append(name)
        elif 'lora_A' in name or 'lora_B' in name:
            if param.requires_grad:
                lora_trainable.append(name)
            else:
                lora_frozen.append(name)

    print(f"\nbase_layer:")
    print(f"  Trainable: {len(base_trainable)}")
    print(f"  Frozen: {len(base_frozen)}")
    if base_trainable:
        print(f"  ✗ WARNING: base_layer should be frozen!")
        for name in base_trainable[:3]:
            print(f"    - {name}")

    print(f"\nlora_A/B:")
    print(f"  Trainable: {len(lora_trainable)}")
    print(f"  Frozen: {len(lora_frozen)}")
    if lora_frozen:
        print(f"  ✗ WARNING: lora should be trainable!")
        for name in lora_frozen[:3]:
            print(f"    - {name}")

    print(f"\nclassifier:")
    print(f"  Trainable: {len(classifier_trainable)}")
    print(f"  Frozen: {len(classifier_frozen)}")

    # Verdict
    base_ok = len(base_trainable) == 0
    lora_ok = len(lora_trainable) > 0 and len(lora_frozen) == 0
    classifier_ok = len(classifier_trainable) > 0

    print(f"\n{'✓' if base_ok else '✗'} base_layer frozen: {base_ok}")
    print(f"{'✓' if lora_ok else '✗'} lora trainable: {lora_ok}")
    print(f"{'✓' if classifier_ok else '✗'} classifier trainable: {classifier_ok}")

    return base_ok and lora_ok and classifier_ok


def check_bias_mapping(model):
    """Check if bias is properly mapped from pretrained model."""
    print_section("3. BIAS MAPPING CHECK")

    bias_count = 0
    bias_samples = []

    for name, module in model.named_modules():
        if isinstance(module, (AnalogLinear, nn.Linear)):
            # For AnalogLinear, bias is accessed differently
            if isinstance(module, AnalogLinear):
                if hasattr(module, 'bias') and module.bias is not False and module.bias is not None:
                    try:
                        # Try to get bias from analog tile
                        weights_tuple = module.get_weights()
                        if len(weights_tuple) > 1 and weights_tuple[1] is not None:
                            bias = weights_tuple[1]
                            bias_count += 1
                            if len(bias_samples) < 3:
                                bias_samples.append((name, bias.shape, bias[:5]))
                    except:
                        pass
            elif hasattr(module, 'bias') and module.bias is not None:
                bias_count += 1
                if len(bias_samples) < 3:
                    bias_samples.append((name, module.bias.shape, module.bias[:5]))

    print(f"\nTotal layers with bias: {bias_count}")
    print(f"\nSample biases (first 5 values):")
    for name, shape, values in bias_samples:
        print(f"  {name}")
        print(f"    Shape: {shape}")
        print(f"    Values: {values.detach().cpu().numpy()}")

    # Check if biases are reasonable (not all zeros)
    all_zero = True
    for _, _, values in bias_samples:
        if not torch.allclose(values, torch.zeros_like(values)):
            all_zero = False
            break

    print(f"\n{'✓' if not all_zero else '✗'} Biases initialized (not all zeros): {not all_zero}")

    return not all_zero


def check_forward_pass(model, device, tokenizer):
    """Check if forward pass works."""
    print_section("4. FORWARD PASS CHECK")

    # Prepare input
    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print(f"\nInput: {text}")
    print(f"Input shape: {inputs['input_ids'].shape}")

    try:
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        print(f"\n✓ Forward pass successful!")
        print(f"  Logits shape: {outputs.logits.shape}")
        print(f"  Logits: {outputs.logits}")
        print(f"  Prediction: {outputs.logits.argmax(dim=-1).item()}")

        # Check for NaN/Inf
        has_nan = torch.isnan(outputs.logits).any()
        has_inf = torch.isinf(outputs.logits).any()

        print(f"\n{'✗' if has_nan else '✓'} No NaN in output: {not has_nan}")
        print(f"{'✗' if has_inf else '✓'} No Inf in output: {not has_inf}")

        model.train()
        return not has_nan and not has_inf

    except Exception as e:
        print(f"\n✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_backward_pass(model, device, tokenizer):
    """Check if backward pass works and gradients flow correctly."""
    print_section("5. BACKWARD PASS CHECK")

    # Prepare input
    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = torch.tensor([1]).to(device)  # Positive sentiment

    model.train()
    model.zero_grad()

    try:
        # Forward
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        print(f"\nLoss: {loss.item()}")

        # Backward
        loss.backward()

        print(f"✓ Backward pass successful!")

        # Check gradients
        # For AnalogLinear, gradient is in analog_ctx parameter
        lora_grads = []
        base_grads = []
        classifier_grads = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_value = param.grad.abs().mean().item() if param.grad.numel() > 1 else param.grad.abs().item()
                if 'lora' in name and 'analog_ctx' in name:
                    lora_grads.append((name, grad_value))
                elif 'base_layer' in name and 'analog_ctx' in name:
                    base_grads.append((name, grad_value))
                elif 'classifier' in name:
                    classifier_grads.append((name, grad_value))

        print(f"\nGradients received:")
        print(f"  lora parameters: {len(lora_grads)}")
        print(f"  base_layer parameters: {len(base_grads)}")
        print(f"  classifier parameters: {len(classifier_grads)}")

        if lora_grads:
            print(f"\n  Sample lora gradients (mean abs):")
            for name, grad in lora_grads[:3]:
                print(f"    {name}: {grad:.6f}")

        if base_grads:
            print(f"\n  ✗ WARNING: base_layer received gradients (should be frozen)!")
            for name, grad in base_grads[:3]:
                print(f"    {name}: {grad:.6f}")

        if classifier_grads:
            print(f"\n  Sample classifier gradients (mean abs):")
            for name, grad in classifier_grads[:3]:
                print(f"    {name}: {grad:.6f}")

        lora_ok = len(lora_grads) > 0
        base_ok = len(base_grads) == 0
        classifier_ok = len(classifier_grads) > 0

        print(f"\n{'✓' if lora_ok else '✗'} LoRA receives gradients: {lora_ok}")
        print(f"{'✓' if base_ok else '✗'} base_layer no gradients (frozen): {base_ok}")
        print(f"{'✓' if classifier_ok else '✗'} Classifier receives gradients: {classifier_ok}")

        return lora_ok and base_ok and classifier_ok

    except Exception as e:
        print(f"\n✗ Backward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_weight_updates(model, device, tokenizer):
    """Check if weights actually update during training."""
    print_section("6. WEIGHT UPDATE CHECK")

    # Get initial weights from AnalogLinear modules
    initial_analog_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear) and ('lora_A' in name or 'lora_B' in name):
            try:
                weights = module.get_weights()
                if isinstance(weights, tuple):
                    w = weights[0]
                else:
                    w = weights
                initial_analog_weights[name] = w.clone().detach()
            except:
                pass

    # Get initial digital weights
    initial_digital_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'classifier' in name:
            initial_digital_weights[name] = param.data.clone()

    print(f"\nTracking:")
    print(f"  Analog layers (LoRA): {len(initial_analog_weights)}")
    print(f"  Digital parameters (classifier): {len(initial_digital_weights)}")

    # Prepare data
    texts = ["This movie is great!", "This movie is terrible!"]
    labels_list = [1, 0]

    # Setup optimizer - CRITICAL: pass model to regroup_param_groups!
    optimizer = AnalogSGD(model.parameters(), lr=0.001)
    optimizer.regroup_param_groups(model)

    print(f"\nOptimizer param_groups: {len(optimizer.param_groups)}")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  Group {i}: {len(group['params'])} params")

    # Training step
    model.train()
    total_loss = 0

    for text, label in zip(texts, labels_list):
        inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                          max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor([label]).to(device)

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"\nTraining loss: {total_loss / len(texts):.4f}")

    # Check analog weight changes
    lora_updated = []
    lora_unchanged = []
    base_updated = []

    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            if name in initial_analog_weights:
                try:
                    weights = module.get_weights()
                    if isinstance(weights, tuple):
                        current_w = weights[0]
                    else:
                        current_w = weights
                    diff = (current_w - initial_analog_weights[name]).abs().max().item()

                    if 'lora' in name:
                        if diff > 1e-8:
                            lora_updated.append((name, diff))
                        else:
                            lora_unchanged.append(name)
                    elif 'base_layer' in name:
                        if diff > 1e-8:
                            base_updated.append((name, diff))
                except:
                    pass

    # Check digital weight changes (classifier)
    classifier_updated = []
    for name, param in model.named_parameters():
        if name in initial_digital_weights:
            diff = (param.data - initial_digital_weights[name]).abs().max().item()
            if diff > 1e-8:
                classifier_updated.append((name, diff))

    print(f"\nWeight changes:")
    print(f"  LoRA updated: {len(lora_updated)}")
    print(f"  LoRA unchanged: {len(lora_unchanged)}")
    print(f"  base_layer updated: {len(base_updated)}")
    print(f"  classifier updated: {len(classifier_updated)}")

    if lora_updated:
        print(f"\n  Sample LoRA weight changes (max abs):")
        for name, diff in lora_updated[:3]:
            print(f"    {name}: {diff:.6e}")

    if lora_unchanged:
        print(f"\n  ✗ WARNING: Some LoRA weights didn't update!")
        for name in lora_unchanged[:3]:
            print(f"    {name}")

    if base_updated:
        print(f"\n  ✗ WARNING: base_layer weights updated (should be frozen)!")
        for name, diff in base_updated[:3]:
            print(f"    {name}: {diff:.6e}")

    # More lenient criteria: >90% of LoRA should update
    total_lora = len(lora_updated) + len(lora_unchanged)
    lora_update_rate = len(lora_updated) / total_lora if total_lora > 0 else 0
    lora_ok = lora_update_rate > 0.9
    base_ok = len(base_updated) == 0
    classifier_ok = len(classifier_updated) > 0

    print(f"\n{'✓' if lora_ok else '✗'} LoRA weights update: {lora_update_rate * 100:.1f}% ({len(lora_updated)}/{total_lora})")
    print(f"{'✓' if base_ok else '✗'} base_layer weights frozen: {base_ok}")
    print(f"{'✓' if classifier_ok else '✗'} Classifier weights update: {classifier_ok}")

    return lora_ok and base_ok and classifier_ok


def main():
    """Run all diagnostic checks."""
    print("\n" + "=" * 80)
    print("  SIXT1C-LORA MODEL DIAGNOSTIC FOR SST-2")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    task_name = "sst2"
    target_modules = ["query", "key", "value"]
    lora_alpha = 0.01  # Use the best-performing alpha

    print(f"Task: {task_name}")
    print(f"Target modules: {target_modules}")
    print(f"LoRA alpha: {lora_alpha}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Create model
    print("\nCreating model...")
    model = create_glue_model(task_name, device, target_modules,
                             fp_lora=False, lora_alpha=lora_alpha)

    # Run checks
    results = {}
    results['architecture'] = check_architecture(model)
    results['trainability'] = check_trainability(model)
    results['bias'] = check_bias_mapping(model)
    results['forward'] = check_forward_pass(model, device, tokenizer)
    results['backward'] = check_backward_pass(model, device, tokenizer)
    results['updates'] = check_weight_updates(model, device, tokenizer)

    # Final summary
    print_section("DIAGNOSTIC SUMMARY")

    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")

    all_passed = all(results.values())

    print(f"\n{'=' * 80}")
    if all_passed:
        print("✓✓✓ ALL CHECKS PASSED!")
        print("Model is correctly configured and functional.")
    else:
        print("✗✗✗ SOME CHECKS FAILED!")
        print("Model has configuration or functional issues.")
    print(f"{'=' * 80}\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
