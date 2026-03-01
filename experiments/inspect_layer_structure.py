"""
Inspect the exact layer structure after LoRA-only analog conversion
"""

import sys
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, get_peft_model

sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
from sixt1c_config import gen_sixt1c_lora_config_standard
from related_functions import convert_lora_layers_only_to_analog

from aihwkit.nn import AnalogLinear


def inspect_structure():
    """Inspect layer structure in detail."""

    print("Creating model with LoRA-only analog conversion...")

    # Create model
    model_name = "bert-base-uncased"
    model_config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    # Apply PEFT LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=1.0,
        lora_dropout=0.0,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)

    # Convert only LoRA to analog
    sixt1c_config = gen_sixt1c_lora_config_standard(output_noise_level=0.0)
    model = convert_lora_layers_only_to_analog(model, sixt1c_config)

    print("\n" + "=" * 80)
    print("Detailed Layer Structure Inspection")
    print("=" * 80)

    # Check one specific layer in detail
    layer0_query = model.base_model.model.bert.encoder.layer[0].attention.self.query

    print("\nInspecting layer[0].attention.self.query:")
    print(f"  Type: {type(layer0_query)}")
    print(f"  Attributes: {dir(layer0_query)}")

    if hasattr(layer0_query, 'base_layer'):
        print(f"\n  base_layer type: {type(layer0_query.base_layer)}")
        print(f"  base_layer is nn.Linear: {isinstance(layer0_query.base_layer, nn.Linear)}")
        print(f"  base_layer is AnalogLinear: {isinstance(layer0_query.base_layer, AnalogLinear)}")

    if hasattr(layer0_query, 'lora_A'):
        print(f"\n  lora_A: {layer0_query.lora_A}")
        if hasattr(layer0_query.lora_A, 'default'):
            lora_A_default = layer0_query.lora_A['default']
            print(f"    lora_A.default type: {type(lora_A_default)}")
            print(f"    lora_A.default is nn.Linear: {isinstance(lora_A_default, nn.Linear)}")
            print(f"    lora_A.default is AnalogLinear: {isinstance(lora_A_default, AnalogLinear)}")

            # Check if it has the analog weight attribute
            if isinstance(lora_A_default, AnalogLinear):
                print(f"    ✓ lora_A is ANALOG (AnalogLinear)")
                print(f"    Has analog_tile: {hasattr(lora_A_default, 'analog_tile')}")
                if hasattr(lora_A_default, 'weight'):
                    print(f"    weight attribute: {type(lora_A_default.weight)}")

    if hasattr(layer0_query, 'lora_B'):
        print(f"\n  lora_B: {layer0_query.lora_B}")
        if hasattr(layer0_query.lora_B, 'default'):
            lora_B_default = layer0_query.lora_B['default']
            print(f"    lora_B.default type: {type(lora_B_default)}")
            print(f"    lora_B.default is nn.Linear: {isinstance(lora_B_default, nn.Linear)}")
            print(f"    lora_B.default is AnalogLinear: {isinstance(lora_B_default, AnalogLinear)}")

            if isinstance(lora_B_default, AnalogLinear):
                print(f"    ✓ lora_B is ANALOG (AnalogLinear)")
                print(f"    Has analog_tile: {hasattr(lora_B_default, 'analog_tile')}")
                if hasattr(lora_B_default, 'weight'):
                    print(f"    weight attribute: {type(lora_B_default.weight)}")

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)

    # Count actual structure
    base_layer_digital = 0
    base_layer_analog = 0
    lora_A_analog = 0
    lora_A_digital = 0
    lora_B_analog = 0
    lora_B_digital = 0

    for name, module in model.named_modules():
        # Check for base_layer
        if name.endswith('base_layer'):
            if isinstance(module, AnalogLinear):
                base_layer_analog += 1
            elif isinstance(module, nn.Linear):
                base_layer_digital += 1

        # Check for lora_A
        if 'lora_A' in name and name.endswith('default'):
            if isinstance(module, AnalogLinear):
                lora_A_analog += 1
                print(f"  ✓ ANALOG lora_A: {name}")
            elif isinstance(module, nn.Linear):
                lora_A_digital += 1

        # Check for lora_B
        if 'lora_B' in name and name.endswith('default'):
            if isinstance(module, AnalogLinear):
                lora_B_analog += 1
            elif isinstance(module, nn.Linear):
                lora_B_digital += 1

    print(f"\nbase_layer:")
    print(f"  Digital: {base_layer_digital}")
    print(f"  Analog: {base_layer_analog}")

    print(f"\nlora_A:")
    print(f"  Digital: {lora_A_digital}")
    print(f"  Analog: {lora_A_analog}")

    print(f"\nlora_B:")
    print(f"  Digital: {lora_B_digital}")
    print(f"  Analog: {lora_B_analog}")

    print("\n" + "=" * 80)
    if base_layer_digital > 0 and lora_A_analog > 0 and lora_B_analog > 0:
        print("✓ Architecture is CORRECT:")
        print("  - base_layer: DIGITAL (no conversion)")
        print("  - lora_A: ANALOG (sixt1c)")
        print("  - lora_B: ANALOG (sixt1c)")
    print("=" * 80)


if __name__ == "__main__":
    inspect_structure()
