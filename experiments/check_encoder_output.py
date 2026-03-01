"""
Check encoder output directly before pooler.
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from transformers import AutoTokenizer


def check_encoder_output(mode_name, fp_lora):
    """Check encoder output."""
    print(f"\n{'='*80}")
    print(f"  {mode_name} MODE - ENCODER OUTPUT CHECK")
    print(f"{'='*80}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    # Create model
    print(f"\n  Creating model (fp_lora={fp_lora})...")
    model = create_glue_model("sst2", device, ["value"],
                             fp_lora=fp_lora, lora_alpha=1.0)

    # Prepare input
    text = "This movie is great!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()

    # Get encoder output manually
    print(f"\n  Running forward through encoder...")

    with torch.no_grad():
        # Use full forward pass with output_hidden_states
        outputs = model.base_model(**inputs, output_hidden_states=True, return_dict=True)

        # Get encoder last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # Last layer output

        print(f"\n  Encoder last_hidden_state:")
        print(f"    Shape: {last_hidden_state.shape}")
        print(f"    Mean: {last_hidden_state.mean().item():.6f}")
        print(f"    Std: {last_hidden_state.std().item():.6f}")
        print(f"    Max abs: {last_hidden_state.abs().max().item():.6f}")

        # Check if explosion is already in encoder output
        if last_hidden_state.abs().max().item() > 1000:
            print(f"    ⚠️  EXPLOSION ALREADY in encoder output!")

        # Check first token (used by pooler)
        first_token_tensor = last_hidden_state[:, 0]
        print(f"\n  First token representation (input to pooler):")
        print(f"    Shape: {first_token_tensor.shape}")
        print(f"    Mean: {first_token_tensor.mean().item():.6f}")
        print(f"    Std: {first_token_tensor.std().item():.6f}")
        print(f"    Max abs: {first_token_tensor.abs().max().item():.6f}")

        # Check if explosion is already here
        if first_token_tensor.abs().max().item() > 1000:
            print(f"    ⚠️  EXPLOSION in first token!")

        # Now run through pooler
        pooled_output = model.base_model.model.mobilebert.pooler(last_hidden_state)

        print(f"\n  Pooler output:")
        print(f"    Shape: {pooled_output.shape}")
        print(f"    Mean: {pooled_output.mean().item():.6f}")
        print(f"    Std: {pooled_output.std().item():.6f}")
        print(f"    Max abs: {pooled_output.abs().max().item():.6f}")

        if pooled_output.abs().max().item() > 1000:
            print(f"    ⚠️  EXPLOSION in pooler output!")

    # Check pooler weights
    print(f"\n  Pooler configuration:")
    pooler = model.base_model.model.mobilebert.pooler
    print(f"    Type: {type(pooler)}")

    if hasattr(pooler, 'dense'):
        dense = pooler.dense
        print(f"\n  Pooler.dense layer:")
        print(f"    Type: {type(dense)}")
        if hasattr(dense, 'weight'):
            print(f"    Weight shape: {dense.weight.shape}")
            print(f"    Weight mean: {dense.weight.mean().item():.6f}")
            print(f"    Weight std: {dense.weight.std().item():.6f}")
            print(f"    Weight max abs: {dense.weight.abs().max().item():.6f}")
        if hasattr(dense, 'bias') and dense.bias is not None:
            print(f"    Bias mean: {dense.bias.mean().item():.6f}")
            print(f"    Bias max abs: {dense.bias.abs().max().item():.6f}")

        # Check if pooler.dense is analog
        from aihwkit.nn import AnalogLinear
        if isinstance(dense, AnalogLinear):
            print(f"    ⚠️  Pooler.dense is ANALOG! (This might be the problem)")
        else:
            print(f"    ✓ Pooler.dense is DIGITAL (nn.Linear)")


def main():
    print("="*80)
    print("  ENCODER OUTPUT INVESTIGATION")
    print("="*80)

    check_encoder_output("SIXT1C", fp_lora=False)

    print("\n\n")

    check_encoder_output("FP", fp_lora=True)

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
