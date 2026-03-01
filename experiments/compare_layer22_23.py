#!/usr/bin/env python
# coding=utf-8
"""
Compare Layer 22 (Normal) vs Layer 23 (Explosion)

Check if both layers have similar amplification in FFN,
or if Layer 23 has a specific problem.
"""

import sys
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, set_seed

sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

SEED = 42
MODEL_NAME = "google/mobilebert-uncased"

layer_outputs = {}


def track_output(layer_id, component_name):
    """Track output with layer ID."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]

        if isinstance(output, torch.Tensor):
            output_flat = output.detach().cpu().float()

            # Also track input
            input_tensor = input[0] if isinstance(input, tuple) else input
            input_flat = input_tensor.detach().cpu().float() if isinstance(input_tensor, torch.Tensor) else None

            key = f"L{layer_id}_{component_name}"
            layer_outputs[key] = {
                'abs_max': output_flat.abs().max().item(),
                'min': output_flat.min().item(),
                'max': output_flat.max().item(),
                'mean': output_flat.mean().item(),
            }

            if input_flat is not None:
                layer_outputs[key]['input_abs_max'] = input_flat.abs().max().item()
                layer_outputs[key]['input_mean'] = input_flat.mean().item()
    return hook


def main():
    print("=" * 80)
    print("COMPARE: Layer 22 (Normal) vs Layer 23 (Explosion)")
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

    print("\nRegistering hooks on Layer 21, 22, 23...")

    # Hook layers 21, 22, 23
    for layer_id in [21, 22, 23]:
        for name, module in model.named_modules():
            if name == f"mobilebert.encoder.layer.{layer_id}":
                # Hook main components
                for child_name, child_module in module.named_children():
                    if child_name in ['attention', 'ffn', 'intermediate', 'output']:
                        child_module.register_forward_hook(track_output(layer_id, child_name))

                        # Hook FFN sub-components
                        if child_name == 'ffn':
                            for i, ffn_module in enumerate(child_module):
                                ffn_module.register_forward_hook(track_output(layer_id, f'ffn.{i}'))

                        # Hook intermediate sub-components
                        if child_name == 'intermediate':
                            for sub_name, sub_module in child_module.named_children():
                                sub_module.register_forward_hook(track_output(layer_id, f'intermediate.{sub_name}'))

                # Hook the whole layer output
                module.register_forward_hook(track_output(layer_id, 'LAYER_OUTPUT'))
                print(f"  ✓ Layer {layer_id}")
                break

    # Forward pass
    print("\n" + "=" * 80)
    print("RUNNING FORWARD PASS")
    print("=" * 80)

    try:
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        print("\n" + "=" * 80)
        print("LAYER-BY-LAYER COMPARISON")
        print("=" * 80)

        # Group by layer
        for layer_id in [21, 22, 23]:
            print(f"\n{'='*80}")
            print(f"LAYER {layer_id}")
            print('='*80)

            layer_data = {k: v for k, v in layer_outputs.items() if k.startswith(f"L{layer_id}_")}

            if not layer_data:
                print("  No data")
                continue

            # Sort by component order
            component_order = ['attention', 'ffn.0', 'ffn.1', 'ffn.2', 'ffn', 'intermediate.dense',
                             'intermediate.intermediate_act_fn', 'intermediate', 'output', 'LAYER_OUTPUT']

            print(f"\n{'Component':<25} {'Input':>12} {'Output':>12} {'Amplification':>15} {'Type':>10}")
            print("-" * 80)

            for comp in component_order:
                key = f"L{layer_id}_{comp}"
                if key not in layer_data:
                    continue

                data = layer_data[key]
                output_max = data['abs_max']
                input_max = data.get('input_abs_max', 0)

                amplification = output_max / input_max if input_max > 0 else 0

                # Determine type
                comp_type = "LRTT" if 'attention.self' in comp else "Digital"

                # Mark
                marker = ""
                if output_max > 1000:
                    marker = " ⚠️"
                if output_max > 100000:
                    marker = " ❌"

                print(f"{comp:<25} {input_max:>12.2e} {output_max:>12.2e} {amplification:>15.2f}x "
                      f"{comp_type:>10}{marker}")

        # Summary comparison
        print("\n" + "=" * 80)
        print("SUMMARY: FFN Amplification Comparison")
        print("=" * 80)

        print(f"\n{'Layer':<10} {'FFN Input':>15} {'FFN Output':>15} {'Amplification':>15}")
        print("-" * 60)

        for layer_id in [21, 22, 23]:
            ffn_key = f"L{layer_id}_ffn"
            if ffn_key in layer_outputs:
                data = layer_outputs[ffn_key]
                ffn_input = data.get('input_abs_max', 0)
                ffn_output = data['abs_max']
                amp = ffn_output / ffn_input if ffn_input > 0 else 0

                marker = " ❌ EXPLOSION!" if layer_id == 23 else ""

                print(f"Layer {layer_id:<4} {ffn_input:>15.2e} {ffn_output:>15.2e} {amp:>15.2f}x{marker}")

        # Attention output analysis
        print("\n" + "=" * 80)
        print("ATTENTION OUTPUT → FFN INPUT (Analog→Digital Boundary)")
        print("=" * 80)

        print(f"\n{'Layer':<10} {'Attn Out':>15} {'FFN In':>15} {'Match?':>10}")
        print("-" * 55)

        for layer_id in [21, 22, 23]:
            attn_key = f"L{layer_id}_attention"
            ffn_key = f"L{layer_id}_ffn"

            if attn_key in layer_outputs and ffn_key in layer_outputs:
                attn_out = layer_outputs[attn_key]['abs_max']
                ffn_in = layer_outputs[ffn_key].get('input_abs_max', 0)

                match = "✓" if abs(attn_out - ffn_in) < 10 else "✗"

                print(f"Layer {layer_id:<4} {attn_out:>15.2e} {ffn_in:>15.2e} {match:>10}")

        print(f"\nFinal loss: {loss.item():.6e}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
