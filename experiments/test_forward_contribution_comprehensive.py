"""
Comprehensive test of LoRA forward pass contribution.
Tests both at initialization and after training.
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


def test_forward_contribution(model, tokenizer, device, test_name="Test"):
    """Test if LoRA contributes to forward pass."""
    print(f"\n{'='*80}")
    print(f"{test_name}")
    print(f"{'='*80}")

    # Prepare multiple test inputs
    test_texts = [
        "This movie is great!",
        "This movie is terrible!",
        "The weather is nice today.",
        "I love this product!",
        "This is the worst experience ever.",
    ]

    all_results = []

    for i, text in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                          max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model.eval()

        # Get output WITH LoRA (normal)
        with torch.no_grad():
            output_with_lora = model(**inputs).logits.cpu()

        # Disable LoRA by setting scaling to 0
        original_scalings = {}
        for name, module in model.named_modules():
            if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
                if 'default' in module.scaling:
                    original_scalings[name] = module.scaling['default']
                    module.scaling['default'] = 0.0

        with torch.no_grad():
            output_without_lora = model(**inputs).logits.cpu()

        # Restore scaling
        for name, module in model.named_modules():
            if name in original_scalings:
                module.scaling['default'] = original_scalings[name]

        # Calculate difference
        diff = (output_with_lora - output_without_lora).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        all_results.append({
            'text': text[:30] + '...' if len(text) > 30 else text,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'with_lora': output_with_lora,
            'without_lora': output_without_lora,
        })

        print(f"\n  Input {i+1}: {text[:40]}...")
        print(f"    Output WITH LoRA: {output_with_lora.numpy()}")
        print(f"    Output WITHOUT LoRA: {output_without_lora.numpy()}")
        print(f"    Max diff: {max_diff:.6f}")
        print(f"    Mean diff: {mean_diff:.6f}")

    # Overall statistics
    avg_max_diff = np.mean([r['max_diff'] for r in all_results])
    avg_mean_diff = np.mean([r['mean_diff'] for r in all_results])

    print(f"\n  Overall Statistics:")
    print(f"    Average max diff: {avg_max_diff:.6f}")
    print(f"    Average mean diff: {avg_mean_diff:.6f}")

    has_contribution = avg_max_diff > 1e-4

    if has_contribution:
        print(f"\n  ✓ LoRA IS contributing to forward pass")
    else:
        print(f"\n  ✗ LoRA NOT contributing to forward pass")

    return has_contribution, avg_max_diff


def main():
    print("\n" + "="*80)
    print("  COMPREHENSIVE FORWARD PASS TEST")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_name = "sst2"
    target_modules = ["query", "key", "value"]
    lora_alpha = 1.0

    print(f"\nSetup:")
    print(f"  Task: {task_name}")
    print(f"  Target modules: {target_modules}")
    print(f"  lora_alpha: {lora_alpha}")
    print(f"  Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Create model
    print("\n[Creating model...]")
    model = create_glue_model(task_name, device, target_modules,
                             fp_lora=False, lora_alpha=lora_alpha)

    # Test 1: At initialization
    init_has_contribution, init_diff = test_forward_contribution(
        model, tokenizer, device,
        test_name="TEST 1: Forward Contribution at Initialization"
    )

    # Train the model
    print(f"\n{'='*80}")
    print("TRAINING MODEL")
    print(f"{'='*80}")

    texts = ["This movie is great!"] * 30
    labels_list = [1] * 30

    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    optimizer.regroup_param_groups(model)

    model.train()
    criterion = nn.CrossEntropyLoss()

    print(f"\n  Training on {len(texts)} samples...")

    for idx, (text, label) in enumerate(zip(texts, labels_list)):
        inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                          max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor([label]).to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        if (idx + 1) % 10 == 0:
            print(f"    Step {idx+1}/{len(texts)}, Loss: {loss.item():.4f}")

    print("  Training complete!")

    # Test 2: After training
    trained_has_contribution, trained_diff = test_forward_contribution(
        model, tokenizer, device,
        test_name="TEST 2: Forward Contribution After Training"
    )

    # Summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\n  At Initialization:")
    print(f"    Contributing: {init_has_contribution}")
    print(f"    Avg max diff: {init_diff:.6f}")

    print(f"\n  After Training:")
    print(f"    Contributing: {trained_has_contribution}")
    print(f"    Avg max diff: {trained_diff:.6f}")

    if trained_has_contribution:
        improvement = trained_diff / init_diff if init_diff > 0 else float('inf')
        print(f"\n  ✓✓✓ SUCCESS!")
        print(f"  LoRA forward pass is working correctly!")
        print(f"  Contribution increased by {improvement:.1f}x after training")
    else:
        print(f"\n  ✗✗✗ FAILURE!")
        print(f"  LoRA forward pass is NOT working!")

    print(f"{'='*80}\n")

    return trained_has_contribution


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
