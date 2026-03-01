"""
Test LoRA contribution AFTER training (not just at initialization).
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from transformers import AutoTokenizer
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD


def main():
    print("\n" + "="*80)
    print("  TEST: LORA CONTRIBUTION AFTER TRAINING")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_name = "sst2"
    target_modules = ["query", "key", "value"]
    lora_alpha = 1.0

    print(f"\nTask: {task_name}")
    print(f"Target: {target_modules}")
    print(f"lora_alpha: {lora_alpha}")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Create model with analog LoRA
    print("\n[Step 1] Creating model with analog LoRA...")
    model = create_glue_model(task_name, device, target_modules,
                             fp_lora=False, lora_alpha=lora_alpha)

    # Check LoRA B weights BEFORE training
    print("\n[Step 2] LoRA B weights BEFORE training:")
    lora_b_layers = []
    for name, module in model.named_modules():
        if 'lora_B' in name and isinstance(module, AnalogLinear):
            try:
                weights = module.get_weights()
                if isinstance(weights, tuple):
                    w = weights[0]
                else:
                    w = weights
                lora_b_layers.append((name, w))
            except:
                pass

    if lora_b_layers:
        w = lora_b_layers[0][1]
        print(f"  Sample: {lora_b_layers[0][0]}")
        print(f"  Mean: {w.mean().item():.6f}")
        print(f"  Std: {w.std().item():.6f}")
        print(f"  Max abs: {w.abs().max().item():.6f}")

    # Prepare training data
    print("\n[Step 3] Training model...")
    texts = ["This movie is great!"] * 20
    labels_list = [1] * 20

    optimizer = AnalogSGD(model.parameters(), lr=0.01)
    optimizer.regroup_param_groups(model)

    model.train()
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

    print("  Training complete!")

    # Check LoRA B weights AFTER training
    print("\n[Step 4] LoRA B weights AFTER training:")
    for name, module in model.named_modules():
        if name == lora_b_layers[0][0]:
            try:
                weights = module.get_weights()
                if isinstance(weights, tuple):
                    w = weights[0]
                else:
                    w = weights
                print(f"  Sample: {name}")
                print(f"  Mean: {w.mean().item():.6f}")
                print(f"  Std: {w.std().item():.6f}")
                print(f"  Max abs: {w.abs().max().item():.6f}")

                # Check if weights changed
                initial_w = lora_b_layers[0][1]
                diff = (w - initial_w).abs().max().item()
                print(f"  Max weight change: {diff:.6f}")

                if diff > 1e-6:
                    print("  ✓ Weights CHANGED during training")
                else:
                    print("  ✗ Weights DID NOT change")
                break
            except:
                pass

    # Test forward pass contribution AFTER training
    print("\n[Step 5] Testing forward pass contribution AFTER training...")

    test_text = "This movie is terrible!"
    inputs = tokenizer(test_text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    # Get output WITH LoRA (normal)
    with torch.no_grad():
        output_with_lora = model(**inputs).logits

    print(f"  Output WITH LoRA:")
    print(f"    Values: {output_with_lora}")
    print(f"    Mean: {output_with_lora.mean().item():.4f}")

    # Disable LoRA by setting scaling to 0
    original_scalings = {}
    for name, module in model.named_modules():
        if hasattr(module, 'scaling') and isinstance(module.scaling, dict):
            original_scalings[name] = module.scaling['default']
            module.scaling['default'] = 0.0

    with torch.no_grad():
        output_without_lora = model(**inputs).logits

    # Restore scaling
    for name, module in model.named_modules():
        if name in original_scalings:
            module.scaling['default'] = original_scalings[name]

    print(f"  Output WITHOUT LoRA (scaling=0):")
    print(f"    Values: {output_without_lora}")
    print(f"    Mean: {output_without_lora.mean().item():.4f}")

    # Compare
    diff = (output_with_lora - output_without_lora).abs()
    print(f"\n  Difference:")
    print(f"    Max diff: {diff.max().item():.6f}")
    print(f"    Mean diff: {diff.mean().item():.6f}")

    has_contribution = diff.max().item() > 1e-4

    print("\n" + "="*80)
    if has_contribution:
        print("✓✓✓ SUCCESS!")
        print(f"LoRA IS contributing to forward pass after training")
        print(f"Contribution magnitude: {diff.mean().item():.6f}")
    else:
        print("✗✗✗ FAILURE!")
        print("LoRA NOT contributing to forward pass (even after training)")
    print("="*80 + "\n")

    return has_contribution


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
