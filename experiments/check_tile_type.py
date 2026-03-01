#!/usr/bin/env python3
"""Check the actual tile type to see if it's trainable."""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

from transformers import AutoConfig, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from aihwkit.nn import AnalogLinear

from sixt1c_config import (
    gen_sixt1c_lora_config_trainable,
    gen_softbounds_base_layer_config
)
from smart_conversion import convert_base_and_lora_separately

# Create model
model_config = AutoConfig.from_pretrained('google/mobilebert-uncased')
model_config.num_labels = 2
model_config.num_hidden_layers = 1

model = AutoModelForSequenceClassification.from_pretrained(
    'google/mobilebert-uncased',
    config=model_config,
    ignore_mismatched_sizes=True
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=1.0,
    target_modules=['query'],
    bias='none',
    lora_dropout=0.0,
)
model = get_peft_model(model, peft_config)

# Convert
base_config = gen_softbounds_base_layer_config()
lora_config = gen_sixt1c_lora_config_trainable()

print("=" * 80)
print("Config types:")
print(f"  base_config: {type(base_config).__name__}")
print(f"  lora_config: {type(lora_config).__name__}")
print("=" * 80)

model = convert_base_and_lora_separately(
    model,
    base_layer_config=base_config,
    lora_config=lora_config,
    lora_trainable=True
)

# Find modules
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        print(f"\n{'='*80}")
        print(f"Module: {name}")
        print(f"{'='*80}")

        # Check analog_module type
        analog_module = module.analog_module
        print(f"\nanalog_module type: {type(analog_module).__name__}")
        print(f"  Full type: {type(analog_module)}")

        # Check tile type
        tile = analog_module.tile
        print(f"\ntile type: {type(tile).__name__}")
        print(f"  Full type: {type(tile)}")

        # Check if tile has update method
        print(f"\nTile methods:")
        print(f"  has 'update': {hasattr(tile, 'update')}")
        print(f"  has 'forward': {hasattr(tile, 'forward')}")
        print(f"  has 'backward': {hasattr(tile, 'backward')}")

        # Check rpu_config
        if hasattr(analog_module, 'rpu_config'):
            rpu_config = analog_module.rpu_config
            print(f"\nRPU config type: {type(rpu_config).__name__}")

            # Is it trainable?
            is_inference = 'Inference' in type(analog_module).__name__
            is_single = 'SingleRPUConfig' in str(type(rpu_config))

            print(f"\nTrainability check:")
            print(f"  Is InferenceTile: {is_inference}")
            print(f"  Is SingleRPUConfig: {is_single}")

            if is_inference:
                print(f"  → ✗ This is an INFERENCE tile (cannot be trained)!")
            elif is_single:
                print(f"  → ✓ This is a TRAINABLE tile (SingleRPUConfig)")

        # Check if parameters are trainable
        trainable_params = [p for p in module.parameters() if p.requires_grad]
        print(f"\nTrainable parameters: {len(trainable_params)}")
        for p in trainable_params:
            print(f"  - {p.shape}: {type(p).__name__}")

        # Only check lora_A (skip base_layer)
        if 'lora_A' in name:
            break

print("\n" + "=" * 80)
