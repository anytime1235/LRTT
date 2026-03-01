"""
Test: Convert only LoRA layers to analog (without converting base_layer)

Architecture:
- base_layer: Digital (no conversion)
- lora_A: Sixt1c Analog (6T1C)
- lora_B: Sixt1c Analog (6T1C)
"""

import sys
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
from sixt1c_config import gen_sixt1c_lora_config_standard
from related_functions import convert_lora_layers_only_to_analog

from aihwkit.nn import AnalogLinear


def test_lora_only_analog():
    """Test converting only LoRA layers to analog."""

    print("=" * 80)
    print("TEST: LoRA-only Analog Conversion")
    print("=" * 80)

    # Step 1: Load base model (digital)
    print("\n[Step 1] Loading base model...")
    model_name = "bert-base-uncased"
    model_config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
    print("✓ Base model loaded (all digital)")

    # Step 2: Apply PEFT LoRA
    print("\n[Step 2] Applying PEFT LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=1.0,
        lora_dropout=0.0,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    print("✓ PEFT LoRA applied")
    model.print_trainable_parameters()

    # Check layer structure BEFORE conversion
    print("\n[BEFORE Conversion] Layer structure:")
    lora_layers = []
    base_layers = []
    for name, module in model.named_modules():
        if 'lora_A' in name or 'lora_B' in name:
            if isinstance(module, nn.Linear):
                lora_layers.append(name)
                print(f"  LoRA layer (digital): {name}")
        elif 'base_layer' in name:
            if isinstance(module, nn.Linear):
                base_layers.append(name)

    print(f"\nFound {len(lora_layers)} LoRA layers (digital)")
    print(f"Found {len(base_layers)} base_layer layers (digital)")

    # Step 3: Convert ONLY lora_A/B to sixt1c analog
    print("\n[Step 3] Converting ONLY LoRA layers to sixt1c analog...")
    sixt1c_config = gen_sixt1c_lora_config_standard(output_noise_level=0.0)
    model = convert_lora_layers_only_to_analog(model, sixt1c_config)
    print("✓ LoRA layers converted to sixt1c analog")

    # Check layer structure AFTER conversion
    print("\n[AFTER Conversion] Layer structure:")
    analog_lora = []
    digital_base = []
    for name, module in model.named_modules():
        if 'lora_A' in name or 'lora_B' in name:
            if isinstance(module, AnalogLinear):
                analog_lora.append(name)
                print(f"  LoRA layer (ANALOG): {name}")
        elif 'base_layer' in name:
            if isinstance(module, nn.Linear):
                digital_base.append(name)
            elif isinstance(module, AnalogLinear):
                print(f"  WARNING: base_layer is ANALOG (unexpected): {name}")

    print(f"\nAnalog LoRA layers: {len(analog_lora)}")
    print(f"Digital base_layer layers: {len(digital_base)}")

    # Step 4: Test forward pass
    print("\n[Step 4] Testing forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_text = "This is a test sentence."
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✓ Forward pass successful!")
        print(f"  Output logits shape: {outputs.logits.shape}")
        print(f"  Output logits: {outputs.logits}")
    except Exception as e:
        print(f"✗ Forward pass FAILED: {e}")
        return False

    # Step 5: Verify architecture
    print("\n[Step 5] Architecture verification:")
    all_correct = True

    # Check: base_layer should be digital
    base_analog_count = sum(1 for n, m in model.named_modules()
                           if 'base_layer' in n and isinstance(m, AnalogLinear))
    if base_analog_count == 0:
        print("✓ base_layer: All DIGITAL (correct)")
    else:
        print(f"✗ base_layer: {base_analog_count} are ANALOG (should be digital)")
        all_correct = False

    # Check: lora_A/B should be analog
    lora_analog_count = sum(1 for n, m in model.named_modules()
                           if ('lora_A' in n or 'lora_B' in n) and isinstance(m, AnalogLinear))
    lora_digital_count = sum(1 for n, m in model.named_modules()
                            if ('lora_A' in n or 'lora_B' in n) and isinstance(m, nn.Linear))

    if lora_analog_count > 0 and lora_digital_count == 0:
        print(f"✓ lora_A/B: All {lora_analog_count} are ANALOG (correct)")
    else:
        print(f"✗ lora_A/B: {lora_analog_count} analog, {lora_digital_count} digital (should be all analog)")
        all_correct = False

    print("\n" + "=" * 80)
    if all_correct:
        print("✓✓✓ TEST PASSED: LoRA-only analog conversion works correctly!")
    else:
        print("✗✗✗ TEST FAILED: Architecture mismatch")
    print("=" * 80)

    return all_correct


if __name__ == "__main__":
    test_lora_only_analog()
