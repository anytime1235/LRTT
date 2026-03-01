"""
Test both lora_mode options: 'inference' and 'trainable'
"""

import sys
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
from sixt1c_config import gen_sixt1c_lora_config_with_mode
from related_functions import get_parent_module

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import TorchInferenceRPUConfig, SingleRPUConfig
from aihwkit.optim import AnalogSGD


def make_peft_compatible(analog_layer):
    """Make analog layer PEFT-compatible."""
    try:
        # Try trainable device method (SingleRPUConfig)
        weights_tuple = analog_layer.get_weights()
        if isinstance(weights_tuple, tuple):
            weights = weights_tuple[0]
        else:
            weights = weights_tuple
    except:
        # Try inference device method (TorchInferenceRPUConfig)
        try:
            weights = analog_layer.analog_module.tile.get_weights()
        except:
            print(f"Warning: Could not get weights from analog layer")
            return analog_layer

    analog_layer._peft_weight = weights
    analog_layer.weight = weights
    return analog_layer


def convert_lora_to_analog(model, rpu_config, lora_mode):
    """Convert LoRA layers to analog using the given config."""
    # Find LoRA layers
    lora_layer_names = []
    for name, module in model.named_modules():
        if ('lora_A' in name or 'lora_B' in name) and isinstance(module, nn.Linear):
            lora_layer_names.append(name)

    print(f"Found {len(lora_layer_names)} LoRA layers to convert")

    # Convert each layer
    for layer_name in lora_layer_names:
        parent, attr = get_parent_module(model, layer_name)
        original_layer = getattr(parent, attr)

        # Convert to analog
        analog_layer = AnalogLinear.from_digital(original_layer, rpu_config)

        # Make PEFT-compatible
        analog_layer = make_peft_compatible(analog_layer)

        # Set trainability based on mode
        if lora_mode == 'trainable':
            for param in analog_layer.parameters():
                param.requires_grad = True
        else:
            for param in analog_layer.parameters():
                param.requires_grad = False

        # Replace
        setattr(parent, attr, analog_layer)

    print(f"✓ Converted {len(lora_layer_names)} layers to analog ({lora_mode} mode)")
    return model


def test_lora_mode(lora_mode):
    """Test a specific lora_mode."""
    print("\n" + "=" * 80)
    print(f"Testing lora_mode='{lora_mode}'")
    print("=" * 80)

    # Step 1: Create model with PEFT LoRA
    print("\n[1] Creating model with PEFT LoRA...")
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
    print("✓ PEFT LoRA applied")

    # Step 2: Generate config with specified mode
    print(f"\n[2] Generating config with lora_mode='{lora_mode}'...")
    rpu_config = gen_sixt1c_lora_config_with_mode(
        output_noise_level=0.0,
        lora_mode=lora_mode
    )
    print(f"✓ Config generated: {type(rpu_config).__name__}")
    if isinstance(rpu_config, SingleRPUConfig):
        print(f"  Device type: {type(rpu_config.device).__name__}")
        print(f"  dw_min: {rpu_config.device.dw_min}")
    elif isinstance(rpu_config, TorchInferenceRPUConfig):
        print(f"  Type: TorchInferenceRPUConfig (inference only)")

    # Step 3: Convert LoRA to analog
    print(f"\n[3] Converting LoRA to analog...")
    model = convert_lora_to_analog(model, rpu_config, lora_mode)

    # Step 4: Verify structure
    print(f"\n[4] Verifying structure...")
    analog_count = sum(1 for _, m in model.named_modules()
                      if isinstance(m, AnalogLinear))
    print(f"  Analog layers: {analog_count}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable_params:,}")

    # Step 5: Test forward pass
    print(f"\n[5] Testing forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Test sentence", return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✓ Forward pass successful! Logits: {outputs.logits.tolist()}")
    except Exception as e:
        print(f"✗ Forward pass FAILED: {e}")
        return False

    # Step 6: Test backward pass (only for trainable mode)
    if lora_mode == 'trainable':
        print(f"\n[6] Testing backward pass...")
        model.train()
        optimizer = AnalogSGD(model.parameters(), lr=0.001)
        optimizer.regroup_param_groups(model)

        try:
            outputs = model(**inputs)
            loss = outputs.logits.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"✓ Backward pass successful! Loss: {loss.item():.4f}")
        except Exception as e:
            print(f"✗ Backward pass FAILED: {e}")
            return False
    else:
        print(f"\n[6] Skipping backward pass (inference mode)")

    print("\n" + "=" * 80)
    print(f"✓✓✓ lora_mode='{lora_mode}' TEST PASSED!")
    print("=" * 80)
    return True


def main():
    """Test both modes."""
    print("=" * 80)
    print("LORA MODE TEST SUITE")
    print("=" * 80)

    results = {}

    # Test inference mode
    results['inference'] = test_lora_mode('inference')

    # Test trainable mode
    results['trainable'] = test_lora_mode('trainable')

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for mode, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  lora_mode='{mode}': {status}")

    if all(results.values()):
        print("\n✓✓✓ ALL TESTS PASSED!")
    else:
        print("\n✗✗✗ SOME TESTS FAILED!")

    print("=" * 80)


if __name__ == "__main__":
    main()
