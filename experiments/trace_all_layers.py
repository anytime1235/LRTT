#!/usr/bin/env python
# coding=utf-8
"""
Trace All Layer Outputs

Track output magnitude through all 24 transformer layers to find
where the explosion starts.
"""

import sys
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

SEED = 42
MODEL_NAME = "google/mobilebert-uncased"

layer_stats = {}


def track_layer_output(name):
    """Create hook to track layer output statistics."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]  # Take first element if tuple

        if isinstance(output, torch.Tensor):
            output_flat = output.detach().cpu().float()
            layer_stats[name] = {
                'min': output_flat.min().item(),
                'max': output_flat.max().item(),
                'mean': output_flat.mean().item(),
                'std': output_flat.std().item() if output.numel() > 1 else 0.0,
                'abs_max': output_flat.abs().max().item(),
            }
    return hook


def main():
    print("=" * 80)
    print("TRACE: All Layer Outputs (Find Explosion Point)")
    print("=" * 80)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input
    print("\n[1/4] Creating input...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer(
        "This is a test.",
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = torch.tensor([1]).to(device)

    # Load model
    print("[2/4] Loading model...")
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )

    print("[3/4] Converting to Sixt1c-LoRA...")
    lrtt_config = create_lrtt_lora_config(
        rank=8,
        lora_alpha=1.22,
        output_noise_level=0.0,
        use_floating_point=False,
    )

    target_modules = ["query", "key", "value"]
    model = convert_model_to_lrtt_lora(model, lrtt_config, target_modules)
    model.to(device)
    model.train()

    print("[4/4] Registering hooks on all layers...")
    hook_count = 0

    # Hook embedding
    if hasattr(model, 'mobilebert') and hasattr(model.mobilebert, 'embeddings'):
        model.mobilebert.embeddings.register_forward_hook(
            track_layer_output("0_embeddings")
        )
        hook_count += 1

    # Hook each transformer layer
    for i in range(24):
        layer_name = f"mobilebert.encoder.layer.{i}"
        for name, module in model.named_modules():
            if name == layer_name:
                module.register_forward_hook(track_layer_output(f"{i+1:02d}_layer{i}"))
                hook_count += 1
                break

    # Hook classifier
    for name, module in model.named_modules():
        if name == "classifier":
            module.register_forward_hook(track_layer_output("25_classifier"))
            hook_count += 1
            break

    print(f"  ✓ Registered {hook_count} hooks\n")

    # Forward pass
    print("=" * 80)
    print("RUNNING FORWARD PASS")
    print("=" * 80)

    try:
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        loss = outputs.loss

        print("\n" + "=" * 80)
        print("LAYER-BY-LAYER OUTPUT MAGNITUDE")
        print("=" * 80)

        # Sort by order
        sorted_layers = sorted(layer_stats.items(), key=lambda x: x[0])

        print(f"\n{'Layer':<20} {'abs_max':>12} {'min':>12} {'max':>12} {'mean':>12}")
        print("-" * 80)

        explosion_layer = None
        prev_max = 0

        for name, stats in sorted_layers:
            abs_max = stats['abs_max']

            # Mark if explosion detected
            marker = ""
            if abs_max > 1000:
                marker = " ⚠️  LARGE"
            if abs_max > 100000:
                marker = " ❌ EXPLOSION"
                if explosion_layer is None:
                    explosion_layer = name

            # Check for sudden jump
            if prev_max > 0 and abs_max > prev_max * 10:
                marker += " [10x jump]"

            print(f"{name:<20} {abs_max:>12.2e} {stats['min']:>12.2e} "
                  f"{stats['max']:>12.2e} {stats['mean']:>12.2e}{marker}")

            prev_max = abs_max

        print("\n" + "=" * 80)
        print("DIAGNOSIS")
        print("=" * 80)

        print(f"\nFinal loss: {loss.item():.6e}")

        if explosion_layer:
            print(f"\n❌ Explosion first detected at: {explosion_layer}")
            print(f"   Magnitude: {layer_stats[explosion_layer]['abs_max']:.2e}")
        else:
            print("\n✓ No obvious explosion detected")

        # Check for gradual growth
        embeddings_max = layer_stats.get('0_embeddings', {}).get('abs_max', 0)
        final_max = sorted_layers[-1][1]['abs_max'] if sorted_layers else 0

        if embeddings_max > 0:
            growth_factor = final_max / embeddings_max
            print(f"\nGrowth from embeddings to final:")
            print(f"  Embeddings: {embeddings_max:.2e}")
            print(f"  Final:      {final_max:.2e}")
            print(f"  Factor:     {growth_factor:.2f}x")

            if growth_factor > 1000:
                print(f"  ⚠️  Excessive growth ({growth_factor:.0f}x)!")

    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("TRACE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
