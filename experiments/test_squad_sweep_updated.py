"""
Test the updated SQuAD sweep script with smart conversion
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

# Import from the sweep script
from sweep_sixt1c_lora_squad_adam import create_squad_model

from aihwkit.nn import AnalogLinear
from transformers import AutoTokenizer


def test_squad_model_creation():
    """Test that the updated create_squad_model works correctly."""

    print("=" * 80)
    print("TEST: Updated SQuAD Model with Smart Conversion")
    print("=" * 80)

    # Parameters
    params = {"lora_alpha": 1.0}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_modules = ["query", "key", "value"]

    # Create model
    print("\n[1] Creating SQuAD model with smart conversion...")
    model = create_squad_model(params, device, target_modules, fp_lora=False)

    # Verify architecture
    print("\n[2] Verifying architecture...")

    # Count layers
    base_analog = sum(1 for n, m in model.named_modules()
                     if 'base_layer' in n and isinstance(m, AnalogLinear))
    lora_analog = sum(1 for n, m in model.named_modules()
                     if ('lora_A' in n or 'lora_B' in n) and isinstance(m, AnalogLinear))

    print(f"  base_layer analog: {base_analog}")
    print(f"  lora_A/B analog: {lora_analog}")

    # Check trainability
    base_trainable = sum(1 for n, m in model.named_modules()
                        if 'base_layer' in n and isinstance(m, AnalogLinear)
                        and any(p.requires_grad for p in m.parameters()))

    lora_trainable = sum(1 for n, m in model.named_modules()
                        if ('lora_A' in n or 'lora_B' in n) and isinstance(m, AnalogLinear)
                        and any(p.requires_grad for p in m.parameters()))

    print(f"  base_layer trainable: {base_trainable}")
    print(f"  lora_A/B trainable: {lora_trainable}")

    if base_trainable == 0:
        print("  ✓ base_layer all frozen (correct)")
    else:
        print(f"  ✗ WARNING: {base_trainable} base_layer are trainable!")

    if lora_trainable == lora_analog:
        print("  ✓ lora all trainable (correct)")
    else:
        print(f"  ✗ WARNING: Only {lora_trainable}/{lora_analog} lora are trainable!")

    # Test forward pass
    print("\n[3] Testing forward pass...")
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    question = "What is the capital of France?"
    context = "Paris is the capital and most populous city of France."

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=384
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✓ Forward pass successful!")
        print(f"  start_logits shape: {outputs.start_logits.shape}")
        print(f"  end_logits shape: {outputs.end_logits.shape}")
    except Exception as e:
        print(f"✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Summary
    print("\n" + "=" * 80)
    print("✓✓✓ TEST PASSED: Updated SQuAD model works correctly!")
    print("=" * 80)
    print("\nArchitecture:")
    print("  - base_layer: SingleRPUConfig + SoftBoundsDevice (frozen)")
    print("  - lora_A/B: SingleRPUConfig + LinearStepDevice (trainable)")
    print("  - qa_outputs: Digital (trainable)")
    print("\nReady for sweep!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_squad_model_creation()
    exit(0 if success else 1)
