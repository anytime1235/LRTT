#!/usr/bin/env python
# coding=utf-8
"""
Debug LRTT Tile Outputs

Monitor tile_a, tile_b, tile_c outputs during forward pass to identify
where numerical instability occurs.
"""

import sys
import torch
import numpy as np
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

# Global storage for intermediate outputs
tile_outputs = {}


def check_tensor(name, tensor):
    """Check tensor and return stats."""
    if tensor is None:
        return "None"

    is_nan = torch.isnan(tensor).any().item()
    is_inf = torch.isinf(tensor).any().item()

    tensor_np = tensor.detach().cpu().float()
    stats = {
        'shape': list(tensor.shape),
        'min': tensor_np.min().item(),
        'max': tensor_np.max().item(),
        'mean': tensor_np.mean().item(),
        'std': tensor_np.std().item() if tensor.numel() > 1 else 0.0,
        'nan': is_nan,
        'inf': is_inf,
    }
    return stats


def hook_lrtt_tile(tile_name):
    """Create a hook for LRTT tiles."""
    def hook(module, input, output):
        tile_outputs[tile_name] = {
            'input': check_tensor(f"{tile_name}_input", input[0] if isinstance(input, tuple) else input),
            'output': check_tensor(f"{tile_name}_output", output),
        }

        # Also check scaling parameters
        if hasattr(module, 'out_scaling_alpha'):
            alpha = module.out_scaling_alpha
            tile_outputs[tile_name]['out_scaling_alpha'] = check_tensor(f"{tile_name}_alpha", alpha)

        if hasattr(module, 'mapping_scales'):
            scales = module.mapping_scales
            tile_outputs[tile_name]['mapping_scales'] = check_tensor(f"{tile_name}_scales", scales)

    return hook


def main():
    print("=" * 80)
    print("DEBUG: LRTT Tile-by-Tile Output Analysis")
    print("=" * 80)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy input
    print("\n[1/5] Creating input...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dummy_text = "This is a test sentence."
    inputs = tokenizer(
        dummy_text,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = torch.tensor([1]).to(device)

    # Load and convert model
    print("[2/5] Loading model...")
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )

    print("[3/5] Converting to Sixt1c-LoRA...")
    lora_alpha = 1.22
    lrtt_config = create_lrtt_lora_config(
        rank=8,
        lora_alpha=lora_alpha,
        output_noise_level=0.0,
        use_floating_point=False,
    )

    target_modules = ["query", "key", "value"]
    model = convert_model_to_lrtt_lora(model, lrtt_config, target_modules)
    model.to(device)
    model.train()

    # Register hooks on first layer tiles
    print("[4/5] Registering tile hooks...")
    first_layer = None
    for name, module in model.named_modules():
        if "layer.0.attention.self.query" in name:
            if hasattr(module, 'analog_module'):
                analog_module = module.analog_module
                first_layer = name

                # Hook each sub-tile
                if hasattr(analog_module, 'tile_a'):
                    analog_module.tile_a.register_forward_hook(hook_lrtt_tile("tile_a"))
                    print("  ✓ Hooked tile_a")

                if hasattr(analog_module, 'tile_b'):
                    analog_module.tile_b.register_forward_hook(hook_lrtt_tile("tile_b"))
                    print("  ✓ Hooked tile_b")

                if hasattr(analog_module, 'tile_c'):
                    analog_module.tile_c.register_forward_hook(hook_lrtt_tile("tile_c"))
                    print("  ✓ Hooked tile_c")

                # Hook the full AnalogLinear too
                module.register_forward_hook(hook_lrtt_tile("full_layer"))
                print("  ✓ Hooked full AnalogLinear")

                break

    if not first_layer:
        print("❌ Could not find LRTT layer!")
        return

    print(f"  Monitoring: {first_layer}\n")

    # Forward pass
    print("[5/5] Running forward pass...\n")
    print("=" * 80)

    try:
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        loss = outputs.loss

        print("\n" + "=" * 80)
        print("TILE OUTPUT ANALYSIS")
        print("=" * 80)

        # Print results for each tile
        for tile_name in ['tile_c', 'tile_b', 'tile_a', 'full_layer']:
            if tile_name not in tile_outputs:
                continue

            print(f"\n{'='*80}")
            print(f"{tile_name.upper()}")
            print('='*80)

            data = tile_outputs[tile_name]

            # Input
            if 'input' in data and data['input'] != "None":
                inp = data['input']
                print(f"Input:")
                print(f"  shape: {inp['shape']}")
                print(f"  range: [{inp['min']:.6f}, {inp['max']:.6f}]")
                print(f"  mean: {inp['mean']:.6f}, std: {inp['std']:.6f}")
                if inp['nan'] or inp['inf']:
                    print(f"  ❌ NaN={inp['nan']}, Inf={inp['inf']}")

            # Output
            if 'output' in data and data['output'] != "None":
                out = data['output']
                print(f"Output:")
                print(f"  shape: {out['shape']}")
                print(f"  range: [{out['min']:.6f}, {out['max']:.6f}]")
                print(f"  mean: {out['mean']:.6f}, std: {out['std']:.6f}")
                if out['nan'] or out['inf']:
                    print(f"  ❌ NaN={out['nan']}, Inf={out['inf']}")
                elif abs(out['max']) > 1e6 or abs(out['min']) > 1e6:
                    print(f"  ⚠️  WARNING: Output magnitude > 1e6!")

            # Scaling parameters
            if 'out_scaling_alpha' in data:
                alpha = data['out_scaling_alpha']
                print(f"out_scaling_alpha:")
                print(f"  range: [{alpha['min']:.6f}, {alpha['max']:.6f}]")
                print(f"  mean: {alpha['mean']:.6f}")

            if 'mapping_scales' in data:
                scales = data['mapping_scales']
                print(f"mapping_scales:")
                print(f"  range: [{scales['min']:.6f}, {scales['max']:.6f}]")
                print(f"  mean: {scales['mean']:.6f}")

        # Final model output
        print(f"\n{'='*80}")
        print("FINAL MODEL OUTPUT")
        print('='*80)

        logits_stats = check_tensor("logits", logits)
        print(f"Logits:")
        print(f"  shape: {logits_stats['shape']}")
        print(f"  range: [{logits_stats['min']:.6f}, {logits_stats['max']:.6f}]")
        print(f"  mean: {logits_stats['mean']:.6f}")

        loss_stats = check_tensor("loss", loss)
        print(f"\nLoss: {loss.item():.6e}")
        if loss.item() > 1e6:
            print(f"  ❌ Loss explosion detected!")

        # Diagnosis
        print(f"\n{'='*80}")
        print("DIAGNOSIS")
        print('='*80)

        # Check each tile for problems
        for tile_name in ['tile_c', 'tile_b', 'tile_a']:
            if tile_name in tile_outputs:
                out = tile_outputs[tile_name]['output']
                if out['nan'] or out['inf']:
                    print(f"❌ {tile_name}: NaN/Inf detected!")
                elif abs(out['max']) > 1e6:
                    print(f"⚠️  {tile_name}: Output explosion (max={out['max']:.2e})")
                else:
                    print(f"✓ {tile_name}: OK (max={out['max']:.2e})")

    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
