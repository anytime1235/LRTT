"""
Deep test of Analog LoRA functionality
Tests:
1. LoRA contribution in forward pass
2. LoRA scaling (lora_alpha / r)
3. Learning effect (before/after training)
4. Comparison: Digital LoRA vs Analog LoRA
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
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def get_layer_output(model, layer_name, input_tensor):
    """Hook to capture layer output."""
    outputs = {}

    def hook_fn(module, input, output):
        outputs['output'] = output.detach().clone()

    # Find layer
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break

    if target_module is None:
        return None

    # Register hook
    handle = target_module.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(**input_tensor)

    handle.remove()
    return outputs.get('output', None)


def test_lora_contribution(model, device, tokenizer):
    """Test if LoRA contributes to forward pass."""
    print_section("TEST 1: LoRA Contribution to Forward Pass")

    # Prepare input
    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Find first LoRA layer
    lora_layers = []
    for name, module in model.named_modules():
        if 'lora_A' in name or 'lora_B' in name:
            if isinstance(module, AnalogLinear):
                lora_layers.append(name)

    if not lora_layers:
        print("✗ No LoRA analog layers found!")
        return False

    print(f"\nFound {len(lora_layers)} analog LoRA layers")
    print(f"Testing first layer: {lora_layers[0]}")

    # Get output with LoRA enabled
    model.eval()
    with torch.no_grad():
        output_with_lora = model(**inputs).logits

    print(f"\nOutput with LoRA:")
    print(f"  Shape: {output_with_lora.shape}")
    print(f"  Values: {output_with_lora}")
    print(f"  Mean: {output_with_lora.mean().item():.4f}")
    print(f"  Std: {output_with_lora.std().item():.4f}")

    # Try to disable LoRA and compare
    # (PEFT LoRA can be disabled by setting scaling to 0)
    print("\n[Attempting to disable LoRA for comparison...]")

    # Save original scaling values
    original_scalings = {}
    for name, module in model.named_modules():
        if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
            original_scalings[name] = module.scaling['default']
            module.scaling['default'] = 0.0  # Disable LoRA

    with torch.no_grad():
        output_without_lora = model(**inputs).logits

    # Restore scaling
    for name, module in model.named_modules():
        if name in original_scalings:
            module.scaling['default'] = original_scalings[name]

    print(f"\nOutput WITHOUT LoRA (scaling=0):")
    print(f"  Values: {output_without_lora}")
    print(f"  Mean: {output_without_lora.mean().item():.4f}")
    print(f"  Std: {output_without_lora.std().item():.4f}")

    # Compare
    diff = (output_with_lora - output_without_lora).abs()
    print(f"\nDifference:")
    print(f"  Max diff: {diff.max().item():.6f}")
    print(f"  Mean diff: {diff.mean().item():.6f}")

    has_contribution = diff.max().item() > 1e-6

    if has_contribution:
        print(f"\n✓ LoRA IS contributing to forward pass")
        print(f"  Contribution magnitude: {diff.mean().item():.6f}")
    else:
        print(f"\n✗ LoRA NOT contributing (difference too small)")

    return has_contribution


def test_lora_weights_magnitude(model):
    """Check LoRA weight magnitudes."""
    print_section("TEST 2: LoRA Weight Magnitudes")

    lora_a_weights = []
    lora_b_weights = []

    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            try:
                weights = module.get_weights()
                if isinstance(weights, tuple):
                    w = weights[0]
                else:
                    w = weights

                if 'lora_A' in name:
                    lora_a_weights.append((name, w))
                elif 'lora_B' in name:
                    lora_b_weights.append((name, w))
            except:
                pass

    print(f"\nLoRA A layers: {len(lora_a_weights)}")
    if lora_a_weights:
        w = lora_a_weights[0][1]
        print(f"  Sample: {lora_a_weights[0][0]}")
        print(f"  Shape: {w.shape}")
        print(f"  Mean: {w.mean().item():.6f}")
        print(f"  Std: {w.std().item():.6f}")
        print(f"  Max: {w.abs().max().item():.6f}")

    print(f"\nLoRA B layers: {len(lora_b_weights)}")
    if lora_b_weights:
        w = lora_b_weights[0][1]
        print(f"  Sample: {lora_b_weights[0][0]}")
        print(f"  Shape: {w.shape}")
        print(f"  Mean: {w.mean().item():.6f}")
        print(f"  Std: {w.std().item():.6f}")
        print(f"  Max: {w.abs().max().item():.6f}")

    # Check if weights are initialized (not all zeros)
    all_zero_a = all(torch.allclose(w, torch.zeros_like(w)) for _, w in lora_a_weights)
    all_zero_b = all(torch.allclose(w, torch.zeros_like(w)) for _, w in lora_b_weights)

    weights_ok = not (all_zero_a or all_zero_b)

    print(f"\n{'✓' if weights_ok else '✗'} Weights properly initialized: {weights_ok}")

    return weights_ok


def test_lora_learning(model, device, tokenizer):
    """Test if LoRA actually learns."""
    print_section("TEST 3: LoRA Learning Test")

    # Get initial weights
    initial_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear) and 'lora' in name:
            try:
                weights = module.get_weights()
                if isinstance(weights, tuple):
                    w = weights[0]
                else:
                    w = weights
                initial_weights[name] = w.clone().detach()
            except:
                pass

    print(f"\nTracking {len(initial_weights)} LoRA layers")

    # Prepare training data
    texts = ["This movie is great!"] * 10  # Repeat for stronger signal
    labels_list = [1] * 10

    # Setup optimizer
    optimizer = AnalogSGD(model.parameters(), lr=0.01)  # Higher LR for faster learning
    optimizer.regroup_param_groups(model)

    print(f"\nTraining on {len(texts)} samples (lr=0.01)...")

    # Training
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for text, label in zip(texts, labels_list):
        inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                          max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor([label]).to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(texts)
    print(f"Average loss: {avg_loss:.4f}")

    # Check weight changes
    weight_changes = []
    for name, module in model.named_modules():
        if name in initial_weights:
            try:
                weights = module.get_weights()
                if isinstance(weights, tuple):
                    current_w = weights[0]
                else:
                    current_w = weights

                diff = (current_w - initial_weights[name]).abs().max().item()
                weight_changes.append((name, diff))
            except:
                pass

    if weight_changes:
        weight_changes.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop 5 weight changes:")
        for name, diff in weight_changes[:5]:
            print(f"  {name}: {diff:.6f}")

        max_change = weight_changes[0][1]
        min_change = weight_changes[-1][1]
        avg_change = sum(d for _, d in weight_changes) / len(weight_changes)

        print(f"\nWeight change statistics:")
        print(f"  Max: {max_change:.6f}")
        print(f"  Min: {min_change:.6f}")
        print(f"  Avg: {avg_change:.6f}")

        learning_ok = avg_change > 1e-6

        print(f"\n{'✓' if learning_ok else '✗'} LoRA weights changed: {learning_ok}")

        return learning_ok
    else:
        print("\n✗ Could not track weight changes")
        return False


def test_scaling_factor(model):
    """Test LoRA scaling factor (lora_alpha / r)."""
    print_section("TEST 4: LoRA Scaling Factor")

    scaling_values = []
    rank_values = []

    for name, module in model.named_modules():
        if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
            if 'default' in module.scaling:
                scaling = module.scaling['default']
                scaling_values.append((name, scaling))

        if hasattr(module, 'r') and isinstance(module.r, dict):
            if 'default' in module.r:
                rank = module.r['default']
                rank_values.append((name, rank))

    print(f"\nFound {len(scaling_values)} layers with scaling")
    print(f"Found {len(rank_values)} layers with rank")

    if scaling_values:
        print(f"\nSample scaling values:")
        for name, scaling in scaling_values[:3]:
            print(f"  {name}: {scaling}")

    if rank_values:
        print(f"\nSample rank values:")
        for name, rank in rank_values[:3]:
            print(f"  {name}: {rank}")

    # Check if scaling is reasonable
    if scaling_values and rank_values:
        avg_scaling = sum(s for _, s in scaling_values) / len(scaling_values)
        avg_rank = sum(r for _, r in rank_values) / len(rank_values)

        print(f"\nAverage scaling: {avg_scaling:.6f}")
        print(f"Average rank: {avg_rank:.1f}")
        print(f"Expected alpha (scaling * rank): {avg_scaling * avg_rank:.6f}")

        scaling_ok = avg_scaling > 0

        print(f"\n{'✓' if scaling_ok else '✗'} Scaling configured: {scaling_ok}")

        return scaling_ok
    else:
        print("\n✗ Could not find scaling/rank info")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("  ANALOG LORA FUNCTIONALITY TEST")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    task_name = "sst2"
    target_modules = ["query", "key", "value"]
    lora_alpha = 1.0  # Changed from 0.01 to match sweep default

    print(f"Task: {task_name}")
    print(f"Target: {target_modules}")
    print(f"lora_alpha: {lora_alpha}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Create model
    print("\n[Creating model...]")
    model = create_glue_model(task_name, device, target_modules,
                             fp_lora=False, lora_alpha=lora_alpha)

    # Run tests
    results = {}
    results['contribution'] = test_lora_contribution(model, device, tokenizer)
    results['weights'] = test_lora_weights_magnitude(model)
    results['learning'] = test_lora_learning(model, device, tokenizer)
    results['scaling'] = test_scaling_factor(model)

    # Summary
    print_section("TEST SUMMARY")

    print(f"\n{'✓' if results['contribution'] else '✗'} LoRA contributes to forward: {results['contribution']}")
    print(f"{'✓' if results['weights'] else '✗'} LoRA weights initialized: {results['weights']}")
    print(f"{'✓' if results['learning'] else '✗'} LoRA learns from training: {results['learning']}")
    print(f"{'✓' if results['scaling'] else '✗'} LoRA scaling configured: {results['scaling']}")

    all_passed = all(results.values())

    print(f"\n{'=' * 80}")
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED!")
        print("Analog LoRA is functioning correctly.")
    else:
        print("✗✗✗ SOME TESTS FAILED!")
        print("Analog LoRA has functional issues.")
    print(f"{'=' * 80}\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
