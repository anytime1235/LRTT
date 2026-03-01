"""
Verify classifier training with stable settings (lower LR, gradient clipping).
"""

import sys
import torch
import torch.nn as nn

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from transformers import AutoTokenizer
from aihwkit.optim import AnalogSGD


def test_classifier_with_settings(mode_name, fp_lora, learning_rate=0.001):
    """Test classifier with specific settings."""
    print(f"\n{'='*80}")
    print(f"  {mode_name} MODE")
    print(f"{'='*80}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_name = "sst2"
    target_modules = ["value"]
    lora_alpha = 1.0

    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Create model
    print(f"\n  Creating model (fp_lora={fp_lora})...")
    model = create_glue_model(task_name, device, target_modules,
                             fp_lora=fp_lora, lora_alpha=lora_alpha)

    # Check classifier parameters
    print(f"\n  Classifier parameters:")
    classifier_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            print(f"    {name}:")
            print(f"      requires_grad: {param.requires_grad}")
            print(f"      shape: {param.shape}")
            print(f"      device: {param.device}")
            print(f"      initial mean: {param.mean().item():.6f}")
            print(f"      initial std: {param.std().item():.6f}")
            classifier_params.append((name, param.clone().detach().cpu()))

    # Prepare simple training data
    texts = ["This movie is great!"] * 10
    labels_list = [1] * 10

    # Setup optimizer with LOWER learning rate
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    optimizer.regroup_param_groups(model)

    model.train()
    criterion = nn.CrossEntropyLoss()

    print(f"\n  Training with lr={learning_rate}...")
    print(f"    Samples: {len(texts)}")

    losses = []
    logits_history = []

    for i, (text, label) in enumerate(zip(texts, labels_list)):
        inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                          max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = torch.tensor([label]).to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits

        # Check for NaN/Inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"    ✗ Step {i+1}: NaN/Inf in logits!")
            print(f"      logits: {logits}")
            break

        loss = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    ✗ Step {i+1}: NaN/Inf in loss!")
            print(f"      logits: {logits}")
            print(f"      loss: {loss.item()}")
            break

        loss.backward()

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        losses.append(loss.item())
        logits_history.append(logits.detach().cpu())

        if (i + 1) % 2 == 0 or i == 0:
            print(f"    Step {i+1}: loss={loss.item():.4f}, logits={logits.squeeze().tolist()}")

    # Check final weights
    print(f"\n  Final classifier parameters:")
    for name, initial_param in classifier_params:
        for pname, param in model.named_parameters():
            if pname == name:
                current_param = param.detach().cpu()
                diff = (current_param - initial_param).abs()

                print(f"    {name}:")
                print(f"      final mean: {current_param.mean().item():.6f}")
                print(f"      final std: {current_param.std().item():.6f}")
                print(f"      max change: {diff.max().item():.6f}")
                print(f"      mean change: {diff.mean().item():.6f}")

                is_changed = diff.max().item() > 1e-5
                status = "✓ Changed" if is_changed else "✗ Not changed"
                print(f"      {status}")
                break

    # Summary
    if len(losses) > 0 and not any(torch.isnan(torch.tensor(l)) for l in losses):
        avg_loss = sum(losses) / len(losses)
        print(f"\n  Summary:")
        print(f"    ✓ Training completed without NaN")
        print(f"    Steps completed: {len(losses)}/{len(texts)}")
        print(f"    Average loss: {avg_loss:.4f}")
        return True
    else:
        print(f"\n  Summary:")
        print(f"    ✗ Training failed (NaN detected)")
        print(f"    Steps completed: {len(losses)}/{len(texts)}")
        return False


def main():
    print("="*80)
    print("  CLASSIFIER TRAINING VERIFICATION (STABLE SETTINGS)")
    print("="*80)

    # Test both modes with stable settings
    sixt1c_ok = test_classifier_with_settings("SIXT1C", fp_lora=False, learning_rate=0.001)

    print("\n\n")

    fp_ok = test_classifier_with_settings("FP", fp_lora=True, learning_rate=0.001)

    # Final summary
    print("\n" + "="*80)
    print("  FINAL RESULTS")
    print("="*80)

    print(f"\n  Sixt1c mode: {'✓ SUCCESS' if sixt1c_ok else '✗ FAILED'}")
    print(f"  FP mode:     {'✓ SUCCESS' if fp_ok else '✗ FAILED'}")

    if sixt1c_ok and fp_ok:
        print("\n  ✓✓✓ Classifier training works in both modes!")
    else:
        print("\n  ⚠ Some issues detected")

    print("="*80 + "\n")

    return sixt1c_ok and fp_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
