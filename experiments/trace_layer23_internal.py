#!/usr/bin/env python
# coding=utf-8
"""
Trace Layer 23 Internal Components

Check which submodule causes explosion:
- Attention (Q/K/V - LRTT)
- Attention output (Digital)
- FFN intermediate (Digital)
- FFN output (Digital)
"""

import sys
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, set_seed

sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

SEED = 42
MODEL_NAME = "google/mobilebert-uncased"

layer23_outputs = {}


def track_output(name):
    """Track output of a specific module."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]

        if isinstance(output, torch.Tensor):
            output_flat = output.detach().cpu().float()
            layer23_outputs[name] = {
                'abs_max': output_flat.abs().max().item(),
                'min': output_flat.min().item(),
                'max': output_flat.max().item(),
            }
    return hook


def main():
    print("=" * 80)
    print("TRACE: Layer 23 Internal Components (LRTT vs Digital)")
    print("=" * 80)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer("Test", padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = torch.tensor([1]).to(device)

    # Load model
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

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

    print("\nRegistering hooks on Layer 23 submodules...")

    # Find Layer 23 and hook all submodules
    layer23_found = False
    for name, module in model.named_modules():
        if name == "mobilebert.encoder.layer.23":
            layer23_found = True
            print(f"\nFound Layer 23, hooking submodules:")

            # Hook all children
            for child_name, child_module in module.named_children():
                full_name = f"{name}.{child_name}"
                child_module.register_forward_hook(track_output(child_name))
                print(f"  ✓ {child_name}")

                # Hook sub-children too
                for subchild_name, subchild_module in child_module.named_children():
                    sub_full_name = f"{child_name}.{subchild_name}"
                    subchild_module.register_forward_hook(track_output(sub_full_name))
                    print(f"    ✓ {sub_full_name}")

            break

    if not layer23_found:
        print("❌ Layer 23 not found!")
        return

    # Also track Layer 22 output for comparison
    for name, module in model.named_modules():
        if name == "mobilebert.encoder.layer.22":
            module.register_forward_hook(track_output("layer22_output"))
            print(f"\n✓ Also tracking Layer 22 output")
            break

    # Forward pass
    print("\n" + "=" * 80)
    print("RUNNING FORWARD PASS")
    print("=" * 80)

    try:
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        print("\n" + "=" * 80)
        print("LAYER 23 INTERNAL OUTPUTS")
        print("=" * 80)

        print(f"\n{'Component':<30} {'abs_max':>15} {'min':>15} {'max':>15} {'Type':>10}")
        print("-" * 85)

        # Sort by abs_max to find explosion point
        sorted_outputs = sorted(layer23_outputs.items(), key=lambda x: x[1]['abs_max'])

        for name, stats in sorted_outputs:
            abs_max = stats['abs_max']

            # Determine if LRTT or Digital
            component_type = "LRTT" if any(x in name for x in ['query', 'key', 'value']) else "Digital"

            # Mark explosion
            marker = ""
            if abs_max > 1000:
                marker = " ⚠️  LARGE"
            if abs_max > 100000:
                marker = " ❌ EXPLOSION"

            print(f"{name:<30} {abs_max:>15.2e} {stats['min']:>15.2e} {stats['max']:>15.2e} "
                  f"{component_type:>10}{marker}")

        # Diagnosis
        print("\n" + "=" * 80)
        print("DIAGNOSIS")
        print("=" * 80)

        # Find first explosion point
        explosion_point = None
        for name, stats in sorted_outputs:
            if stats['abs_max'] > 100000:
                explosion_point = name
                break

        if explosion_point:
            component_type = "LRTT" if any(x in explosion_point for x in ['query', 'key', 'value']) else "Digital"
            print(f"\n❌ Explosion detected in: {explosion_point}")
            print(f"   Type: {component_type}")
            print(f"   Magnitude: {layer23_outputs[explosion_point]['abs_max']:.2e}")

            if component_type == "Digital":
                print(f"\n   → Digital layer (no quantization limits)")
                print(f"   → Can amplify large inputs without bounds")
            else:
                print(f"\n   → LRTT layer (should have quantization limits)")
                print(f"   → Unexpected explosion in analog layer!")

        print(f"\nFinal loss: {loss.item():.6e}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("TRACE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
