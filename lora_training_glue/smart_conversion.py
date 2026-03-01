"""
Smart conversion: Directly convert base_layer and lora layers with different configs

Both base_layer and lora use SingleRPUConfig (trainable devices):
- base_layer: SingleRPUConfig + SoftBoundsDevice (frozen after conversion)
- lora_A/B: SingleRPUConfig + LinearStepDevice (trainable)
"""

import torch.nn as nn
from aihwkit.nn import AnalogLinear


def get_parent_module(model, layer_name):
    """Get parent module and attribute name from layer path."""
    parts = layer_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def make_analog_peft_compatible(analog_layer, is_trainable=False):
    """
    Make analog layer PEFT-compatible.

    Args:
        analog_layer: AnalogLinear layer
        is_trainable: True for trainable (SingleRPUConfig), False for inference (TorchInferenceRPUConfig)
    """
    try:
        if is_trainable:
            # Trainable device (SingleRPUConfig)
            weights_tuple = analog_layer.get_weights()
            weights = weights_tuple[0] if isinstance(weights_tuple, tuple) else weights_tuple
        else:
            # Inference device (TorchInferenceRPUConfig)
            weights = analog_layer.analog_module.tile.get_weights()

        analog_layer._peft_weight = weights
        analog_layer.weight = weights
    except Exception as e:
        print(f"Warning: Could not make layer PEFT-compatible: {e}")

    return analog_layer


def convert_base_and_lora_separately(
    model,
    base_layer_config,
    lora_config,
    lora_trainable=True
):
    """
    Smart conversion: Convert base_layer and lora layers separately with different configs.

    This is the RECOMMENDED approach using SingleRPUConfig for both:
    - base_layer: SingleRPUConfig + SoftBoundsDevice (frozen after conversion)
    - lora_A/B: SingleRPUConfig + LinearStepDevice (trainable)

    Args:
        model: PEFT model with LoRA adapters
        base_layer_config: RPU config for base_layer (SingleRPUConfig + SoftBoundsDevice)
        lora_config: RPU config for lora_A/B (SingleRPUConfig + LinearStepDevice)
        lora_trainable: Whether lora layers should be trainable

    Returns:
        Model with base_layer and lora converted to analog
    """
    print("\n" + "=" * 80)
    print("SMART CONVERSION: Base_layer + LoRA separate conversion")
    print("=" * 80)

    # Step 1: Find all base_layer and lora layers
    base_layer_names = []
    lora_layer_names = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if 'base_layer' in name:
                base_layer_names.append(name)
            elif 'lora_A' in name or 'lora_B' in name:
                lora_layer_names.append(name)

    print(f"\n[Step 1] Found layers:")
    print(f"  base_layer: {len(base_layer_names)}")
    print(f"  lora_A/B: {len(lora_layer_names)}")

    # Step 2: Convert base_layer to SoftBounds analog (frozen)
    print(f"\n[Step 2] Converting base_layer to SoftBounds analog (frozen)...")
    for layer_name in base_layer_names:
        parent, attr = get_parent_module(model, layer_name)
        digital_layer = getattr(parent, attr)

        # Convert to analog with base_layer_config (SingleRPUConfig + SoftBoundsDevice)
        analog_layer = AnalogLinear.from_digital(
            digital_layer,
            base_layer_config
        )

        # Make PEFT-compatible (trainable=True for SingleRPUConfig)
        analog_layer = make_analog_peft_compatible(analog_layer, is_trainable=True)

        # Set as frozen (even though it's trainable device, we freeze it)
        for param in analog_layer.parameters():
            param.requires_grad = False

        # Replace
        setattr(parent, attr, analog_layer)

    print(f"✓ Converted {len(base_layer_names)} base_layer to SoftBounds analog (frozen)")

    # Step 3: Convert lora_A/B to trainable sixt1c analog
    print(f"\n[Step 3] Converting lora_A/B to sixt1c analog (trainable={lora_trainable})...")
    for layer_name in lora_layer_names:
        parent, attr = get_parent_module(model, layer_name)
        digital_layer = getattr(parent, attr)

        # Convert to analog with lora_config (SingleRPUConfig for trainable)
        analog_layer = AnalogLinear.from_digital(
            digital_layer,
            lora_config
        )

        # Make PEFT-compatible (trainable mode if lora_trainable=True)
        analog_layer = make_analog_peft_compatible(analog_layer, is_trainable=lora_trainable)

        # Set trainability
        for param in analog_layer.parameters():
            param.requires_grad = lora_trainable

        # Replace
        setattr(parent, attr, analog_layer)

    print(f"✓ Converted {len(lora_layer_names)} lora_A/B to sixt1c analog (trainable={lora_trainable})")

    # Step 4: Verify architecture
    print(f"\n[Step 4] Verifying architecture...")
    base_analog = sum(1 for n, m in model.named_modules()
                     if 'base_layer' in n and isinstance(m, AnalogLinear))
    lora_analog = sum(1 for n, m in model.named_modules()
                     if ('lora_A' in n or 'lora_B' in n) and isinstance(m, AnalogLinear))

    print(f"  base_layer analog: {base_analog} (should be {len(base_layer_names)})")
    print(f"  lora_A/B analog: {lora_analog} (should be {len(lora_layer_names)})")

    trainable_analog = sum(p.numel() for n, p in model.named_parameters()
                          if p.requires_grad and 'analog' in n.lower())
    print(f"  Trainable analog params: {trainable_analog}")

    print("\n" + "=" * 80)
    print("✓ SMART CONVERSION COMPLETE!")
    print("  Architecture:")
    print("    - base_layer: SoftBounds AnalogLinear (frozen)")
    print(f"    - lora_A/B: 6T1C AnalogLinear (trainable={lora_trainable})")
    print("=" * 80 + "\n")

    return model


def convert_base_and_lora_simple(
    model,
    base_layer_config,
    lora_config,
    lora_trainable=True,
    convert_base_layer=True
):
    """
    Simplified wrapper with option to skip base_layer conversion.

    Args:
        model: PEFT model
        base_layer_config: Config for base_layer (or None to skip)
        lora_config: Config for lora_A/B
        lora_trainable: Whether lora should be trainable
        convert_base_layer: If False, keep base_layer digital
    """
    if convert_base_layer:
        return convert_base_and_lora_separately(
            model, base_layer_config, lora_config, lora_trainable
        )
    else:
        print("\n[INFO] Skipping base_layer conversion (keeping digital)")
        print("[INFO] Only converting lora_A/B to analog\n")

        # Find and convert only lora layers
        lora_layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'lora_A' in name or 'lora_B' in name:
                    lora_layer_names.append(name)

        for layer_name in lora_layer_names:
            parent, attr = get_parent_module(model, layer_name)
            digital_layer = getattr(parent, attr)

            analog_layer = AnalogLinear.from_digital(digital_layer, lora_config)
            analog_layer = make_analog_peft_compatible(analog_layer, is_trainable=lora_trainable)

            for param in analog_layer.parameters():
                param.requires_grad = lora_trainable

            setattr(parent, attr, analog_layer)

        print(f"✓ Converted {len(lora_layer_names)} lora layers to analog (base_layer kept digital)")
        return model
