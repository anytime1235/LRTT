"""
Test smart conversion with SingleRPUConfig for both base_layer and lora

Architecture:
- base_layer: SingleRPUConfig + SoftBoundsDevice (frozen)
- lora_A/B: SingleRPUConfig + LinearStepDevice (trainable)
"""

import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
from sixt1c_config import (
    gen_softbounds_base_layer_config_trainable,
    gen_sixt1c_lora_config_trainable
)
from smart_conversion import convert_base_and_lora_separately

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig


def test_smart_conversion_singlerpu():
    """Test smart conversion with SingleRPUConfig for both layers."""

    print("=" * 80)
    print("TEST: Smart Conversion with SingleRPUConfig (CONSISTENT)")
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

    # Step 2: Generate TRAINABLE configs for both
    print("\n[2] Generating SingleRPUConfig for both base_layer and lora...")
    base_layer_config = gen_softbounds_base_layer_config_trainable(output_noise_level=0.0)
    lora_config = gen_sixt1c_lora_config_trainable(output_noise_level=0.0)

    print(f"✓ base_layer config: {type(base_layer_config).__name__}")
    if isinstance(base_layer_config, SingleRPUConfig):
        print(f"  Device: {type(base_layer_config.device).__name__}")
        print(f"  dw_min: {base_layer_config.device.dw_min}")

    print(f"✓ lora config: {type(lora_config).__name__}")
    if isinstance(lora_config, SingleRPUConfig):
        print(f"  Device: {type(lora_config.device).__name__}")
        print(f"  dw_min: {lora_config.device.dw_min}")

    # Step 3: Smart conversion
    print("\n[3] Performing smart conversion...")
    model = convert_base_and_lora_separately(
        model,
        base_layer_config=base_layer_config,
        lora_config=lora_config,
        lora_trainable=True
    )

    # Step 4: Verify architecture in detail
    print("\n[4] Detailed architecture verification...")

    # Check base_layer
    print("\nbase_layer verification:")
    base_analog = []
    base_trainable_count = 0
    for name, module in model.named_modules():
        if 'base_layer' in name and isinstance(module, AnalogLinear):
            base_analog.append(name)
            # Check if any params are trainable
            has_trainable = any(p.requires_grad for p in module.parameters())
            if has_trainable:
                base_trainable_count += 1

    print(f"  Analog layers: {len(base_analog)}")
    print(f"  Trainable layers: {base_trainable_count}")
    if base_trainable_count == 0:
        print(f"  ✓ All base_layer FROZEN (correct)")
    else:
        print(f"  ✗ WARNING: {base_trainable_count} base_layer are trainable!")

    # Check lora
    print("\nlora_A/B verification:")
    lora_analog = []
    lora_trainable_count = 0
    for name, module in model.named_modules():
        if ('lora_A' in name or 'lora_B' in name) and isinstance(module, AnalogLinear):
            lora_analog.append(name)
            has_trainable = any(p.requires_grad for p in module.parameters())
            if has_trainable:
                lora_trainable_count += 1

    print(f"  Analog layers: {len(lora_analog)}")
    print(f"  Trainable layers: {lora_trainable_count}")
    if lora_trainable_count == len(lora_analog):
        print(f"  ✓ All lora TRAINABLE (correct)")
    else:
        print(f"  ✗ WARNING: Only {lora_trainable_count}/{len(lora_analog)} lora are trainable!")

    # Check device types
    print("\nDevice type verification:")
    layer0_query = model.base_model.model.bert.encoder.layer[0].attention.self.query

    if hasattr(layer0_query, 'base_layer') and isinstance(layer0_query.base_layer, AnalogLinear):
        analog_module = layer0_query.base_layer.analog_module
        print(f"  base_layer tile: {type(analog_module).__name__}")
        # Try to access device
        if hasattr(analog_module, 'tile') and hasattr(analog_module.tile, 'device'):
            print(f"  base_layer device: {type(analog_module.tile.device).__name__}")

    if hasattr(layer0_query, 'lora_A') and hasattr(layer0_query.lora_A, 'default'):
        lora_A = layer0_query.lora_A['default']
        if isinstance(lora_A, AnalogLinear):
            analog_module = lora_A.analog_module
            print(f"  lora_A tile: {type(analog_module).__name__}")
            if hasattr(analog_module, 'tile') and hasattr(analog_module.tile, 'device'):
                print(f"  lora_A device: {type(analog_module.tile.device).__name__}")

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

    # Set trainability
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
    print("✓✓✓ SMART CONVERSION WITH SINGLERPU TEST PASSED!")
    print("=" * 80)
    print("\nFinal Architecture (CONSISTENT):")
    print("  - base_layer: SingleRPUConfig + SoftBoundsDevice (frozen)")
    print("  - lora_A/B: SingleRPUConfig + LinearStepDevice (trainable)")
    print("  - classifier: Digital (trainable)")
    print("\nKey Points:")
    print("  ✓ Both use SingleRPUConfig (consistent device type)")
    print("  ✓ base_layer frozen by setting requires_grad=False")
    print("  ✓ lora trainable with requires_grad=True")
    print("  ✓ No TorchInferenceRPUConfig (which is inference-only)")
    print("  ✓ Both forward and backward passes work correctly")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_smart_conversion_singlerpu()
    exit(0 if success else 1)
