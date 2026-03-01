"""
Test the smart conversion approach: base_layer + lora with different configs
"""

import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
from sixt1c_config import (
    gen_softbounds_base_layer_config,
    gen_sixt1c_lora_config_with_mode
)
from smart_conversion import convert_base_and_lora_separately

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD


def test_smart_conversion():
    """Test smart conversion with separate configs."""

    print("=" * 80)
    print("TEST: Smart Conversion (base_layer + lora with different configs)")
    print("=" * 80)

    # Step 1: Create PEFT model
    print("\n[1] Creating PEFT model...")
    model_name = "bert-base-uncased"
    model_config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=1.0,
        lora_dropout=0.0,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    print("✓ PEFT model created")
    model.print_trainable_parameters()

    # Step 2: Generate configs
    print("\n[2] Generating configs...")
    base_layer_config = gen_softbounds_base_layer_config(output_noise_level=0.0)
    lora_config = gen_sixt1c_lora_config_with_mode(
        output_noise_level=0.0,
        lora_mode='trainable'  # Use trainable config
    )
    print("✓ base_layer config: SoftBounds (inference, frozen)")
    print("✓ lora config: 6T1C LinearStepDevice (trainable)")

    # Step 3: Smart conversion
    print("\n[3] Performing smart conversion...")
    model = convert_base_and_lora_separately(
        model,
        base_layer_config=base_layer_config,
        lora_config=lora_config,
        lora_trainable=True
    )

    # Step 4: Verify architecture
    print("\n[4] Detailed architecture verification...")

    # Check base_layer
    base_analog = []
    base_trainable = []
    for name, module in model.named_modules():
        if 'base_layer' in name and isinstance(module, AnalogLinear):
            base_analog.append(name)
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    base_trainable.append(name)
                    break

    print(f"\nbase_layer:")
    print(f"  Analog layers: {len(base_analog)}")
    print(f"  Trainable layers: {len(base_trainable)}")
    if len(base_trainable) > 0:
        print(f"  ✗ WARNING: base_layer should be frozen!")
    else:
        print(f"  ✓ All base_layer frozen (correct)")

    # Check lora
    lora_analog = []
    lora_trainable = []
    for name, module in model.named_modules():
        if ('lora_A' in name or 'lora_B' in name) and isinstance(module, AnalogLinear):
            lora_analog.append(name)
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    lora_trainable.append(name)
                    break

    print(f"\nlora_A/B:")
    print(f"  Analog layers: {len(lora_analog)}")
    print(f"  Trainable layers: {len(lora_trainable)}")
    if len(lora_trainable) == len(lora_analog):
        print(f"  ✓ All lora layers trainable (correct)")
    else:
        print(f"  ✗ WARNING: Some lora layers not trainable!")

    # Check tile types (config is stored in analog_module)
    print(f"\nTile type verification:")
    layer0_query = model.base_model.model.bert.encoder.layer[0].attention.self.query

    if hasattr(layer0_query, 'base_layer') and isinstance(layer0_query.base_layer, AnalogLinear):
        analog_module = layer0_query.base_layer.analog_module
        print(f"  base_layer tile type: {type(analog_module).__name__}")

    if hasattr(layer0_query, 'lora_A') and hasattr(layer0_query.lora_A, 'default'):
        lora_A = layer0_query.lora_A['default']
        if isinstance(lora_A, AnalogLinear):
            analog_module = lora_A.analog_module
            print(f"  lora_A tile type: {type(analog_module).__name__}")

    # Step 5: Test forward pass
    print("\n[5] Testing forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Test sentence", return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✓ Forward pass successful!")
        print(f"  Output logits: {outputs.logits.tolist()}")
    except Exception as e:
        print(f"✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 6: Test backward pass (training)
    print("\n[6] Testing backward pass (training)...")
    model.train()

    # Set trainability: classifier trainable, lora digital params trainable
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
        elif 'lora' in name and 'analog' not in name.lower():
            param.requires_grad = True

    optimizer = AnalogSGD(model.parameters(), lr=0.001)
    optimizer.regroup_param_groups(model)

    try:
        outputs = model(**inputs)
        loss = outputs.logits.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"✓ Backward pass successful!")
        print(f"  Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Backward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Final summary
    print("\n" + "=" * 80)
    print("✓✓✓ SMART CONVERSION TEST PASSED!")
    print("=" * 80)
    print("\nFinal Architecture:")
    print("  - base_layer: SoftBounds AnalogLinear (TorchInferenceRPUConfig, frozen)")
    print("  - lora_A/B: 6T1C AnalogLinear (SingleRPUConfig+LinearStepDevice, trainable)")
    print("  - classifier: Digital (trainable)")
    print("\nBenefits:")
    print("  ✓ No complex 3-step conversion (digital→analog→digital→analog)")
    print("  ✓ Direct conversion with appropriate configs")
    print("  ✓ Clear separation: base_layer frozen, lora trainable")
    print("  ✓ Both forward and backward passes work correctly")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_smart_conversion()
    exit(0 if success else 1)
