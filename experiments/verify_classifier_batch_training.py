"""
Verify classifier training with ACTUAL BATCH TRAINING (not single samples).
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model, load_glue_data
from transformers import AutoTokenizer, default_data_collator
from aihwkit.optim import AnalogSGD


def test_batch_training(mode_name, fp_lora, num_steps=20, batch_size=32):
    """Test classifier with real batch training."""
    print(f"\n{'='*80}")
    print(f"  {mode_name} MODE - BATCH TRAINING TEST")
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

    # Load ACTUAL training data
    print(f"\n  Loading SST-2 training data...")
    train_dataloader, _ = load_glue_data(task_name, tokenizer)  # Returns (train_loader, eval_loader)

    # Setup optimizer
    optimizer = AnalogSGD(model.parameters(), lr=0.001)
    optimizer.regroup_param_groups(model)

    # Store initial classifier weights
    initial_weights = {}
    for name, param in model.named_parameters():
        if 'classifier' in name:
            initial_weights[name] = param.clone().detach().cpu()

    # Training
    model.train()
    criterion = nn.CrossEntropyLoss()

    print(f"\n  Training for {num_steps} steps (batch_size={batch_size})...")

    losses = []
    grad_norms = []
    step = 0

    for batch in train_dataloader:
        if step >= num_steps:
            break

        # Move batch to device (remove non-input keys)
        labels = batch.pop('labels').to(device)
        if 'idx' in batch:
            batch.pop('idx')
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        # Forward
        outputs = model(**batch)
        logits = outputs.logits

        # Compute loss
        loss = criterion(logits, labels)

        # Check for numerical issues
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    Step {step+1}: NaN/Inf in loss!")
            break

        # Backward
        loss.backward()

        # Check classifier gradients
        classifier_grad_norm = 0.0
        for name, param in model.named_parameters():
            if 'classifier' in name and param.grad is not None:
                classifier_grad_norm += param.grad.norm().item()

        grad_norms.append(classifier_grad_norm)

        # Optimizer step
        optimizer.step()

        losses.append(loss.item())
        step += 1

        if (step) % 5 == 0 or step == 1:
            print(f"    Step {step}: loss={loss.item():.6f}, classifier_grad_norm={classifier_grad_norm:.6f}")

    # Check final weights
    print(f"\n  Checking weight changes...")

    weight_changes = {}
    for name, initial_weight in initial_weights.items():
        for pname, param in model.named_parameters():
            if pname == name:
                current_weight = param.detach().cpu()
                diff = (current_weight - initial_weight).abs()

                max_change = diff.max().item()
                mean_change = diff.mean().item()

                weight_changes[name] = {
                    'max': max_change,
                    'mean': mean_change
                }

                print(f"    {name}:")
                print(f"      max change: {max_change:.8f}")
                print(f"      mean change: {mean_change:.8f}")

                if max_change > 1e-5:
                    print(f"      ✓ Weights CHANGED!")
                else:
                    print(f"      ✗ Weights NOT changed!")
                break

    # Summary
    print(f"\n  Summary:")
    print(f"    Steps completed: {len(losses)}/{num_steps}")

    if len(losses) > 0:
        avg_loss = sum(losses) / len(losses)
        final_loss = losses[-1]
        print(f"    Average loss: {avg_loss:.6f}")
        print(f"    Final loss: {final_loss:.6f}")

        avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0
        print(f"    Avg classifier grad norm: {avg_grad:.6f}")

        all_changed = all(c['max'] > 1e-5 for c in weight_changes.values())

        if all_changed and avg_grad > 0:
            print(f"\n  ✓ Classifier IS training (weights changing, gradients flowing)")
            return True
        else:
            print(f"\n  ✗ Classifier NOT training properly")
            return False
    else:
        print(f"\n  ✗ Training failed (no steps completed)")
        return False


def main():
    print("="*80)
    print("  CLASSIFIER BATCH TRAINING VERIFICATION")
    print("="*80)

    # Test sixt1c mode with REAL batches
    sixt1c_ok = test_batch_training("SIXT1C", fp_lora=False, num_steps=20, batch_size=32)

    print("\n\n")

    # Test FP mode
    fp_ok = test_batch_training("FP", fp_lora=True, num_steps=20, batch_size=32)

    # Final summary
    print("\n" + "="*80)
    print("  FINAL RESULTS")
    print("="*80)

    print(f"\n  Sixt1c mode: {'✓ SUCCESS' if sixt1c_ok else '✗ FAILED'}")
    print(f"  FP mode:     {'✓ SUCCESS' if fp_ok else '✗ FAILED'}")

    if sixt1c_ok and fp_ok:
        print("\n  ✓✓✓ Classifier trains properly in BOTH modes with real batches!")
    elif sixt1c_ok:
        print("\n  ✓ Sixt1c works with batches (issue was single-sample testing)")
    else:
        print("\n  ⚠ Real problem exists in sixt1c mode")

    print("="*80 + "\n")

    return sixt1c_ok and fp_ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
