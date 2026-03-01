"""
Test: Trainable Sixt1c (6T1C LinearStepDevice) LoRA conversion
"""

import sys
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Import aihwkit components
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.presets.utils import IOParameters
from aihwkit.simulator.configs.utils import (
    WeightModifierType,
    WeightClipType,
    WeightRemapType,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.optim import AnalogSGD

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
from related_functions import convert_lora_layers_only_to_analog, get_parent_module


def make_trainable_analog_peft_compatible(analog_layer):
    """
    Make a TRAINABLE AnalogLinear layer (SingleRPUConfig) compatible with PEFT.

    For trainable devices, use analog_layer.get_weights() which returns (weight, bias).
    """
    # Get weights from trainable analog layer
    weights_tuple = analog_layer.get_weights()
    if isinstance(weights_tuple, tuple):
        weights = weights_tuple[0]  # Extract weight tensor from tuple
    else:
        weights = weights_tuple

    # Store the weight tensor for PEFT compatibility
    analog_layer._peft_weight = weights
    analog_layer.weight = weights

    return analog_layer


def gen_trainable_sixt1c_config():
    """
    Create TRAINABLE 6T1C LinearStepDevice config for LoRA A/B.

    This is the CORRECT config for trainable analog LoRA layers.
    """
    # Create SingleRPUConfig with LinearStepDevice
    rpu_config = SingleRPUConfig(device=LinearStepDevice())

    # Configure the LinearStepDevice (6T1C parameters)
    rpu_config.device.dw_min = 0.001981  # 6T1C
    rpu_config.device.dw_min_dtod = 0.0
    rpu_config.device.dw_min_std = 0.0
    rpu_config.device.up_down = 0.0
    rpu_config.device.up_down_dtod = 0.0
    rpu_config.device.w_max = 1.0
    rpu_config.device.w_min = -1.0
    rpu_config.device.w_max_dtod = 0.0
    rpu_config.device.w_min_dtod = 0.0
    rpu_config.device.mult_noise = False  # Deterministic

    # Forward IO
    rpu_config.forward.inp_res = 1 / (2**8 - 2)  # 8-bit
    rpu_config.forward.out_res = 1 / (2**8 - 2)  # 8-bit
    rpu_config.forward.out_noise = 0.0
    rpu_config.forward.is_perfect = False
    rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX
    rpu_config.forward.bound_management = BoundManagementType.ITERATIVE

    return rpu_config


def convert_lora_to_trainable_analog(model, rpu_config):
    """
    Convert only LoRA adapter layers (lora_A, lora_B) to trainable analog.
    """
    # Collect all LoRA layer names
    lora_layer_names = []
    for name, module in model.named_modules():
        if ('lora_A' in name or 'lora_B' in name) and isinstance(module, nn.Linear):
            lora_layer_names.append(name)

    print(f"Found {len(lora_layer_names)} LoRA layers to convert to TRAINABLE analog:")
    for name in lora_layer_names[:6]:  # Show first 6
        print(f"  - {name}")
    if len(lora_layer_names) > 6:
        print(f"  ... and {len(lora_layer_names) - 6} more")

    # Convert each LoRA layer to analog
    for layer_name in lora_layer_names:
        parent, attr = get_parent_module(model, layer_name)
        original_layer = getattr(parent, attr)

        # Convert to TRAINABLE analog layer using SingleRPUConfig
        analog_layer = AnalogLinear.from_digital(
            original_layer,
            rpu_config
        )

        # Make PEFT-compatible by setting the weight attribute
        analog_layer = make_trainable_analog_peft_compatible(analog_layer)

        # Set requires_grad for analog weights
        for param in analog_layer.parameters():
            param.requires_grad = True

        # Replace the layer
        setattr(parent, attr, analog_layer)

    print(f"✓ Converted {len(lora_layer_names)} LoRA layers to TRAINABLE AnalogLinear")
    return model


def test_trainable_sixt1c():
    """Test trainable sixt1c LoRA conversion."""

    print("=" * 80)
    print("TEST: TRAINABLE Sixt1c (6T1C LinearStepDevice) LoRA")
    print("=" * 80)

    # Step 1: Load base model
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

    # Step 3: Convert lora_A/B to TRAINABLE sixt1c analog (6T1C LinearStepDevice)
    print("\n[Step 3] Converting LoRA to TRAINABLE sixt1c analog...")
    sixt1c_config = gen_trainable_sixt1c_config()
    print("\nTrainable Sixt1c RPU Config:")
    print(f"  Device type: {type(sixt1c_config.device).__name__}")
    print(f"  dw_min (6T1C): {sixt1c_config.device.dw_min}")
    print(f"  mult_noise: {sixt1c_config.device.mult_noise}")
    print(f"  noise_management: {sixt1c_config.forward.noise_management}")
    print(f"  bound_management: {sixt1c_config.forward.bound_management}")
    print(f"  weight_scaling_omega: {sixt1c_config.mapping.weight_scaling_omega}")

    model = convert_lora_to_trainable_analog(model, sixt1c_config)

    # Step 4: Verify structure
    print("\n[Step 4] Verifying layer structure...")
    analog_lora = []
    for name, module in model.named_modules():
        if ('lora_A' in name or 'lora_B' in name) and isinstance(module, AnalogLinear):
            analog_lora.append(name)

    print(f"  Analog LoRA layers: {len(analog_lora)}")

    # Check trainability
    trainable_analog_params = 0
    trainable_digital_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'analog' in name.lower() or any(lora in name for lora in analog_lora):
                trainable_analog_params += param.numel()
            else:
                trainable_digital_params += param.numel()

    print(f"  Trainable analog params: {trainable_analog_params:,}")
    print(f"  Trainable digital params: {trainable_digital_params:,}")

    # Step 5: Test forward pass
    print("\n[Step 5] Testing forward pass...")
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
        print(f"  Output logits: {outputs.logits}")
    except Exception as e:
        print(f"✗ Forward pass FAILED: {e}")
        return False

    # Step 6: Test backward pass (training step)
    print("\n[Step 6] Testing backward pass (training)...")
    model.train()

    # Create optimizer
    optimizer = AnalogSGD(model.parameters(), lr=0.001)
    optimizer.regroup_param_groups(model)

    try:
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.logits.sum()  # Dummy loss

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        print(f"✓ Backward pass successful!")
        print(f"  Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"✗ Backward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓✓✓ TEST PASSED: Trainable Sixt1c LoRA works correctly!")
    print("=" * 80)
    print("\nArchitecture:")
    print("  - base_layer: Digital (no conversion)")
    print("  - lora_A: TRAINABLE Analog (6T1C LinearStepDevice)")
    print("  - lora_B: TRAINABLE Analog (6T1C LinearStepDevice)")
    print("=" * 80)

    return True


if __name__ == "__main__":
    test_trainable_sixt1c()
