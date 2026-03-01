"""
Check if classifier gradients are being computed in sixt1c mode.
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from transformers import AutoTokenizer
from aihwkit.optim import AnalogSGD


def check_gradients(mode_name, fp_lora):
    """Check if gradients are computed for classifier."""
    print(f"\n{'='*80}")
    print(f"  {mode_name} MODE - GRADIENT CHECK")
    print(f"{'='*80}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Create model
    print(f"\n  Creating model (fp_lora={fp_lora})...")
    model = create_glue_model("sst2", device, ["value"],
                             fp_lora=fp_lora, lora_alpha=1.0)

    # Prepare single input
    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = torch.tensor([1]).to(device)

    # Setup optimizer
    optimizer = AnalogSGD(model.parameters(), lr=0.001)
    optimizer.regroup_param_groups(model)

    # Forward pass
    model.train()
    optimizer.zero_grad()

    outputs = model(**inputs)
    logits = outputs.logits

    print(f"\n  Forward pass:")
    print(f"    Logits: {logits.squeeze().tolist()}")
    print(f"    Logits shape: {logits.shape}")
    print(f"    Logits requires_grad: {logits.requires_grad}")

    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    print(f"\n  Loss:")
    print(f"    Value: {loss.item():.6f}")
    print(f"    Requires_grad: {loss.requires_grad}")

    # Backward pass
    print(f"\n  Running backward...")
    loss.backward()

    # Check classifier gradients
    print(f"\n  Classifier gradients:")

    classifier_has_grad = False
    for name, param in model.named_parameters():
        if 'classifier' in name:
            has_grad = param.grad is not None
            if has_grad:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_max = param.grad.abs().max().item()
                is_zero = torch.allclose(param.grad, torch.zeros_like(param.grad))

                print(f"    {name}:")
                print(f"      has_grad: {has_grad}")
                print(f"      grad norm: {grad_norm:.6f}")
                print(f"      grad mean: {grad_mean:.6f}")
                print(f"      grad max abs: {grad_max:.6f}")
                print(f"      all zeros: {is_zero}")

                if not is_zero:
                    classifier_has_grad = True
            else:
                print(f"    {name}:")
                print(f"      ✗ NO GRADIENT!")

    # Check optimizer state
    print(f"\n  Optimizer info:")
    print(f"    Number of param groups: {len(optimizer.param_groups)}")

    for i, group in enumerate(optimizer.param_groups):
        print(f"\n    Group {i}:")
        print(f"      LR: {group['lr']}")
        print(f"      Params in group: {len(group['params'])}")

        # Check if classifier is in this group
        classifier_in_group = False
        for param in group['params']:
            for name, model_param in model.named_parameters():
                if param is model_param and 'classifier' in name:
                    classifier_in_group = True
                    print(f"      ✓ Contains classifier: {name}")
                    break

        if not classifier_in_group:
            print(f"      No classifier params in this group")

    # Perform optimizer step and check weight change
    print(f"\n  Performing optimizer step...")

    initial_weights = {}
    for name, param in model.named_parameters():
        if 'classifier' in name:
            initial_weights[name] = param.clone().detach().cpu()

    optimizer.step()

    print(f"\n  Weight changes after step:")
    for name, initial_weight in initial_weights.items():
        for pname, param in model.named_parameters():
            if pname == name:
                current_weight = param.detach().cpu()
                diff = (current_weight - initial_weight).abs()

                max_change = diff.max().item()
                mean_change = diff.mean().item()

                print(f"    {name}:")
                print(f"      max change: {max_change:.8f}")
                print(f"      mean change: {mean_change:.8f}")

                if max_change > 1e-8:
                    print(f"      ✓ Weights changed!")
                else:
                    print(f"      ✗ Weights did NOT change!")
                break

    return classifier_has_grad


def main():
    print("="*80)
    print("  CLASSIFIER GRADIENT VERIFICATION")
    print("="*80)

    sixt1c_has_grad = check_gradients("SIXT1C", fp_lora=False)

    print("\n\n")

    fp_has_grad = check_gradients("FP", fp_lora=True)

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)

    print(f"\n  Sixt1c mode:")
    print(f"    Classifier has gradients: {'✓ YES' if sixt1c_has_grad else '✗ NO'}")

    print(f"\n  FP mode:")
    print(f"    Classifier has gradients: {'✓ YES' if fp_has_grad else '✗ NO'}")

    if sixt1c_has_grad and fp_has_grad:
        print("\n  ✓ Both modes have classifier gradients")
    else:
        print("\n  ✗ Problem detected!")
        if not sixt1c_has_grad:
            print("    - Sixt1c classifier has NO gradients!")
        if not fp_has_grad:
            print("    - FP classifier has NO gradients!")

    print("="*80 + "\n")

    return sixt1c_has_grad and fp_has_grad


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
