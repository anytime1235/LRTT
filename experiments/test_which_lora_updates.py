"""
Check which LoRA layers are actually getting updated during training.
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
    print("  TEST: WHICH LORA LAYERS GET UPDATED?")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_name = "sst2"
    target_modules = ["query", "key", "value"]
    lora_alpha = 1.0

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Create model
    print("\n[Step 1] Creating model...")
    model = create_glue_model(task_name, device, target_modules,
                             fp_lora=False, lora_alpha=lora_alpha)

    # Collect initial weights for ALL LoRA layers
    print("\n[Step 2] Collecting initial weights...")
    initial_weights = {}

    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear) and ('lora_A' in name or 'lora_B' in name):
            try:
                weights = module.get_weights()
                if isinstance(weights, tuple):
                    w = weights[0]
                else:
                    w = weights
                initial_weights[name] = w.clone().detach().cpu()
            except:
                pass

    print(f"  Tracking {len(initial_weights)} LoRA layers")

    # Separate LoRA A and B
    lora_a_names = [n for n in initial_weights.keys() if 'lora_A' in n]
    lora_b_names = [n for n in initial_weights.keys() if 'lora_B' in n]
    print(f"  LoRA A: {len(lora_a_names)}")
    print(f"  LoRA B: {len(lora_b_names)}")

    # Training
    print("\n[Step 3] Training...")
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

    # Check which layers changed
    print("\n[Step 4] Analyzing weight changes...")

    lora_a_changes = []
    lora_b_changes = []

    for name, module in model.named_modules():
        if name in initial_weights:
            try:
                weights = module.get_weights()
                if isinstance(weights, tuple):
                    current_w = weights[0]
                else:
                    current_w = weights

                diff = (current_w.cpu() - initial_weights[name]).abs().max().item()

                if 'lora_A' in name:
                    lora_a_changes.append((name, diff))
                elif 'lora_B' in name:
                    lora_b_changes.append((name, diff))
            except:
                pass

    # Sort by change magnitude
    lora_a_changes.sort(key=lambda x: x[1], reverse=True)
    lora_b_changes.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*80)
    print("LORA A LAYERS (Top 10 changes):")
    print("="*80)
    for name, diff in lora_a_changes[:10]:
        layer_idx = name.split('.layer.')[1].split('.')[0] if '.layer.' in name else 'N/A'
        print(f"  Layer {layer_idx:2s}: {diff:.6f}  {name.split('.')[-4:]}")

    print(f"\n  LoRA A with NO change (diff < 1e-6): {sum(1 for _, d in lora_a_changes if d < 1e-6)}/{len(lora_a_changes)}")
    print(f"  LoRA A with changes (diff > 1e-6): {sum(1 for _, d in lora_a_changes if d > 1e-6)}/{len(lora_a_changes)}")

    print("\n" + "="*80)
    print("LORA B LAYERS (Top 10 changes):")
    print("="*80)
    for name, diff in lora_b_changes[:10]:
        layer_idx = name.split('.layer.')[1].split('.')[0] if '.layer.' in name else 'N/A'
        print(f"  Layer {layer_idx:2s}: {diff:.6f}  {name.split('.')[-4:]}")

    print(f"\n  LoRA B with NO change (diff < 1e-6): {sum(1 for _, d in lora_b_changes if d < 1e-6)}/{len(lora_b_changes)}")
    print(f"  LoRA B with changes (diff > 1e-6): {sum(1 for _, d in lora_b_changes if d > 1e-6)}/{len(lora_b_changes)}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)

    a_updated = sum(1 for _, d in lora_a_changes if d > 1e-6)
    b_updated = sum(1 for _, d in lora_b_changes if d > 1e-6)

    print(f"  Total LoRA A layers: {len(lora_a_changes)}")
    print(f"  LoRA A updated: {a_updated} ({100*a_updated/len(lora_a_changes):.1f}%)")
    print(f"  Total LoRA B layers: {len(lora_b_changes)}")
    print(f"  LoRA B updated: {b_updated} ({100*b_updated/len(lora_b_changes):.1f}%)")

    if a_updated > 0 and b_updated > 0:
        print("\n  ✓ Both LoRA A and B are being updated!")
    elif a_updated > 0:
        print("\n  ⚠ Only LoRA A is being updated (LoRA B frozen?)")
    elif b_updated > 0:
        print("\n  ⚠ Only LoRA B is being updated (LoRA A frozen?)")
    else:
        print("\n  ✗ NEITHER LoRA A nor B are being updated!")

    print("="*80 + "\n")

    return a_updated > 0 and b_updated > 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
