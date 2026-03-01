"""
Monitor LoRA weight updates during actual training with sweep script.
Modified version that tracks weights during training.
"""

import sys
import os
import json
import torch
import torch.nn as nn
from datetime import datetime
from collections import defaultdict
import numpy as np

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import (
    create_glue_model,
    load_glue_data,
    TASK_TO_METRIC,
    MODEL_NAME,
)
from transformers import AutoTokenizer, set_seed
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD


def collect_lora_weights(model):
    """Collect all LoRA A and B weights."""
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear) and ('lora_A' in name or 'lora_B' in name):
            try:
                w = module.get_weights()
                if isinstance(w, tuple):
                    w = w[0]
                weights[name] = w.clone().detach().cpu()
            except:
                pass
    return weights


def compare_weights(weights_before, weights_after):
    """Compare two sets of weights and return statistics."""
    stats = []
    for name in weights_before.keys():
        if name in weights_after:
            diff = (weights_after[name] - weights_before[name]).abs()
            stats.append({
                'name': name,
                'max_diff': diff.max().item(),
                'mean_diff': diff.mean().item(),
                'std_diff': diff.std().item(),
            })
    return stats


def main():
    print("\n" + "="*80)
    print("  MONITORING LORA UPDATES DURING TRAINING")
    print("="*80)

    # Configuration
    task_name = "sst2"
    target_modules = ["value"]  # Default from sweep script
    lora_alpha = 1.0
    num_epochs = 1  # Just 1 epoch to verify
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    print(f"\nConfiguration:")
    print(f"  Task: {task_name}")
    print(f"  Target modules: {target_modules}")
    print(f"  lora_alpha: {lora_alpha}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")
    print(f"  Mode: SIXT1C (analog)")

    # Load tokenizer and data
    print("\n[Step 1] Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_glue_data(task_name, tokenizer)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Eval batches: {len(eval_loader)}")

    # Create model (SIXT1C mode: fp_lora=False)
    print("\n[Step 2] Creating SIXT1C model...")
    model = create_glue_model(task_name, device, target_modules,
                             fp_lora=False, lora_alpha=lora_alpha)

    # Setup optimizer
    initial_lr = 1e-3  # Default from sweep script
    optimizer = AnalogSGD(model.parameters(), lr=initial_lr)
    optimizer.regroup_param_groups(model)

    # Count LoRA layers
    lora_a_count = sum(1 for n, m in model.named_modules()
                       if isinstance(m, AnalogLinear) and 'lora_A' in n)
    lora_b_count = sum(1 for n, m in model.named_modules()
                       if isinstance(m, AnalogLinear) and 'lora_B' in n)

    print(f"\n[Step 3] LoRA layer count:")
    print(f"  LoRA A layers: {lora_a_count}")
    print(f"  LoRA B layers: {lora_b_count}")
    print(f"  Total: {lora_a_count + lora_b_count}")

    # Collect initial weights
    print("\n[Step 4] Collecting initial weights...")
    weights_initial = collect_lora_weights(model)
    print(f"  Collected weights from {len(weights_initial)} layers")

    # Sample initial weights
    lora_b_layers = [n for n in weights_initial.keys() if 'lora_B' in n]
    if lora_b_layers:
        sample_layer = lora_b_layers[0]
        w = weights_initial[sample_layer]
        print(f"\n  Sample LoRA B (before training): {sample_layer.split('.')[-4]}")
        print(f"    Shape: {w.shape}")
        print(f"    Mean: {w.mean().item():.6f}")
        print(f"    Std: {w.std().item():.6f}")
        print(f"    Max abs: {w.abs().max().item():.6f}")

    # Training
    print(f"\n[Step 5] Training for {num_epochs} epoch(s)...")

    criterion = nn.CrossEntropyLoss()
    model.train()

    total_batches = 0
    checkpoint_batches = [10, 50, 100, len(train_loader)]  # Check at these points

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            total_batches += 1

            # Checkpoint: Check weight updates
            if (batch_idx + 1) in checkpoint_batches:
                current_weights = collect_lora_weights(model)
                stats = compare_weights(weights_initial, current_weights)

                # Separate A and B stats
                a_stats = [s for s in stats if 'lora_A' in s['name']]
                b_stats = [s for s in stats if 'lora_B' in s['name']]

                avg_a_diff = np.mean([s['max_diff'] for s in a_stats]) if a_stats else 0
                avg_b_diff = np.mean([s['max_diff'] for s in b_stats]) if b_stats else 0

                print(f"\n    Checkpoint at batch {batch_idx+1}/{len(train_loader)}:")
                print(f"      Loss: {loss.item():.4f}")
                print(f"      LoRA A avg max diff: {avg_a_diff:.6f}")
                print(f"      LoRA B avg max diff: {avg_b_diff:.6f}")

                # Show top 3 changes
                stats.sort(key=lambda x: x['max_diff'], reverse=True)
                print(f"      Top 3 weight changes:")
                for s in stats[:3]:
                    layer_type = 'A' if 'lora_A' in s['name'] else 'B'
                    layer_path = s['name'].split('.')[-4:-1]
                    print(f"        LoRA {layer_type} {'.'.join(layer_path)}: {s['max_diff']:.6f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n  Epoch {epoch+1} complete - Avg loss: {avg_loss:.4f}")

    # Final weight comparison
    print(f"\n[Step 6] Final weight analysis...")
    weights_final = collect_lora_weights(model)
    final_stats = compare_weights(weights_initial, weights_final)

    # Separate by layer type
    a_stats = [s for s in final_stats if 'lora_A' in s['name']]
    b_stats = [s for s in final_stats if 'lora_B' in s['name']]

    print(f"\n  LoRA A layers:")
    print(f"    Total: {len(a_stats)}")
    print(f"    Updated (diff > 1e-6): {sum(1 for s in a_stats if s['max_diff'] > 1e-6)}")
    print(f"    Avg max diff: {np.mean([s['max_diff'] for s in a_stats]):.6f}")
    print(f"    Max diff overall: {max([s['max_diff'] for s in a_stats]):.6f}")

    print(f"\n  LoRA B layers:")
    print(f"    Total: {len(b_stats)}")
    print(f"    Updated (diff > 1e-6): {sum(1 for s in b_stats if s['max_diff'] > 1e-6)}")
    print(f"    Avg max diff: {np.mean([s['max_diff'] for s in b_stats]):.6f}")
    print(f"    Max diff overall: {max([s['max_diff'] for s in b_stats]):.6f}")

    # Check sample layer again
    if sample_layer in weights_final:
        w = weights_final[sample_layer]
        print(f"\n  Sample LoRA B (after training): {sample_layer.split('.')[-4]}")
        print(f"    Mean: {w.mean().item():.6f}")
        print(f"    Std: {w.std().item():.6f}")
        print(f"    Max abs: {w.abs().max().item():.6f}")

        w_initial = weights_initial[sample_layer]
        diff = (w - w_initial).abs().max().item()
        print(f"    Change from initial: {diff:.6f}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    a_updated = sum(1 for s in a_stats if s['max_diff'] > 1e-6)
    b_updated = sum(1 for s in b_stats if s['max_diff'] > 1e-6)

    print(f"  LoRA A: {a_updated}/{len(a_stats)} layers updated ({100*a_updated/len(a_stats):.1f}%)")
    print(f"  LoRA B: {b_updated}/{len(b_stats)} layers updated ({100*b_updated/len(b_stats):.1f}%)")

    all_updated = a_updated == len(a_stats) and b_updated == len(b_stats)

    if all_updated:
        print(f"\n  ✓✓✓ SUCCESS!")
        print(f"  All LoRA layers are being updated during training!")
    else:
        print(f"\n  ⚠ WARNING!")
        print(f"  Not all LoRA layers are being updated!")

    print(f"{'='*80}\n")

    return all_updated


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
