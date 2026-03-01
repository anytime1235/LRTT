"""
Test different lora_alpha values to find the explosion source.
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from transformers import AutoTokenizer


def test_alpha(alpha_value, fp_lora):
    """Test with specific alpha value."""
    mode_str = "FP" if fp_lora else "SIXT1C"
    print(f"\n  Testing {mode_str} mode with lora_alpha={alpha_value}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

    try:
        # Create model
        model = create_glue_model("sst2", device, ["value"],
                                 fp_lora=fp_lora, lora_alpha=alpha_value)

        # Prepare input
        text = "This movie is great!"
        inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                          max_length=128, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model.eval()

        # Forward pass
        with torch.no_grad():
            outputs = model.base_model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden_state = outputs.hidden_states[-1]
            logits = model(**inputs).logits

        max_hidden = last_hidden_state.abs().max().item()
        max_logit = logits.abs().max().item()

        status = "✓ OK" if max_hidden < 1000 else "✗ EXPLODED"
        print(f"    {status}: encoder_max={max_hidden:.2f}, logit_max={max_logit:.2f}")

        return max_hidden, max_logit

    except Exception as e:
        print(f"    ✗ ERROR: {str(e)[:100]}")
        return None, None


def main():
    print("="*80)
    print("  TESTING DIFFERENT LORA_ALPHA VALUES")
    print("="*80)

    # Test different alpha values
    alpha_values = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    print("\n" + "="*80)
    print("  FP MODE (Digital LoRA)")
    print("="*80)

    fp_results = []
    for alpha in alpha_values:
        max_h, max_l = test_alpha(alpha, fp_lora=True)
        fp_results.append((alpha, max_h, max_l))

    print("\n" + "="*80)
    print("  SIXT1C MODE (Analog LoRA)")
    print("="*80)

    sixt1c_results = []
    for alpha in alpha_values:
        max_h, max_l = test_alpha(alpha, fp_lora=False)
        sixt1c_results.append((alpha, max_h, max_l))

    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)

    print(f"\n  {'Alpha':<10} {'FP Hidden':<15} {'FP Logit':<15} {'Status':<10}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*10}")
    for alpha, max_h, max_l in fp_results:
        if max_h is not None:
            status = "OK" if max_h < 1000 else "EXPLODED"
            print(f"  {alpha:<10.1f} {max_h:<15.2f} {max_l:<15.2f} {status:<10}")

    print(f"\n  {'Alpha':<10} {'Sixt1c Hidden':<15} {'Sixt1c Logit':<15} {'Status':<10}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*10}")
    for alpha, max_h, max_l in sixt1c_results:
        if max_h is not None:
            status = "OK" if max_h < 1000 else "EXPLODED"
            print(f"  {alpha:<10.1f} {max_h:<15.2f} {max_l:<15.2f} {status:<10}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
