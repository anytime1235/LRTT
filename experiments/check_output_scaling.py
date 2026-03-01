"""
Check output scaling factors in analog layers.
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from aihwkit.nn import AnalogLinear


def check_output_scaling(mode_name, fp_lora):
    """Check output scaling factors in analog layers."""
    print(f"\n{'='*80}")
    print(f"  {mode_name} MODE - OUTPUT SCALING CHECK")
    print(f"{'='*80}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    print(f"\n  Creating model (fp_lora={fp_lora})...")
    model = create_glue_model("sst2", device, ["value"],
                             fp_lora=fp_lora, lora_alpha=1.0)

    if fp_lora:
        print("\n  FP mode - no analog layers")
        return

    # Check analog layers
    print(f"\n  Checking analog layers...")

    analog_layers = []
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            analog_layers.append((name, module))

    print(f"\n  Found {len(analog_layers)} analog layers")

    # Check first few analog layers in detail
    print(f"\n  Detailed check (first 5 layers):")
    for i, (name, layer) in enumerate(analog_layers[:5]):
        print(f"\n    Layer {i+1}: {name}")
        print(f"      in_features: {layer.in_features}")
        print(f"      out_features: {layer.out_features}")

        # Check if layer has analog_tile
        if hasattr(layer, 'analog_tile'):
            tile = layer.analog_tile
            print(f"      Has analog_tile: True")

            # Check out_scaling_alpha
            if hasattr(tile, 'out_scaling_alpha'):
                alpha = tile.out_scaling_alpha
                if alpha is not None:
                    print(f"      out_scaling_alpha shape: {alpha.shape}")
                    print(f"      out_scaling_alpha (first 5): {alpha.flatten()[:5].tolist()}")
                    print(f"      out_scaling_alpha mean: {alpha.mean().item():.6f}")
                    print(f"      out_scaling_alpha max: {alpha.max().item():.6f}")
                    print(f"      out_scaling_alpha min: {alpha.min().item():.6f}")
                    print(f"      out_scaling_alpha requires_grad: {alpha.requires_grad}")
                else:
                    print(f"      out_scaling_alpha: None")
            else:
                print(f"      No out_scaling_alpha attribute")

            # Check weight
            try:
                weights = layer.get_weights()
                if isinstance(weights, tuple):
                    weight_tensor = weights[0]
                else:
                    weight_tensor = weights
                print(f"      Weight shape: {weight_tensor.shape}")
                print(f"      Weight mean: {weight_tensor.mean().item():.6f}")
                print(f"      Weight std: {weight_tensor.std().item():.6f}")
                print(f"      Weight max abs: {weight_tensor.abs().max().item():.6f}")
            except Exception as e:
                print(f"      Could not get weights: {e}")

        else:
            print(f"      Has analog_tile: False")

    # Test forward pass with simple input
    print(f"\n  Testing forward pass...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    print(f"\n  Logits: {logits.squeeze().tolist()}")
    print(f"  Logits magnitude: {logits.abs().max().item():.2f}")

    # Check intermediate activations
    print(f"\n  Checking hidden states...")

    # Hook to capture hidden states
    hidden_states = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            hidden_states.append({
                'output_mean': output.mean().item(),
                'output_std': output.std().item(),
                'output_max_abs': output.abs().max().item(),
            })

    # Register hooks on first few analog layers
    hooks = []
    for name, layer in analog_layers[:3]:
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print(f"\n  Hidden states from first 3 analog layers:")
    for i, stats in enumerate(hidden_states):
        print(f"    Layer {i+1}:")
        print(f"      mean: {stats['output_mean']:.6f}")
        print(f"      std: {stats['output_std']:.6f}")
        print(f"      max_abs: {stats['output_max_abs']:.2f}")


def main():
    print("="*80)
    print("  OUTPUT SCALING INVESTIGATION")
    print("="*80)

    check_output_scaling("SIXT1C", fp_lora=False)

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
