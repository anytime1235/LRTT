#!/usr/bin/env python3
"""Check analog_module for rpu_config."""

import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

import torch
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

model = convert_base_and_lora_separately(
    model,
    base_layer_config=base_config,
    lora_config=lora_config,
    lora_trainable=True
)

print("=" * 80)
print("CHECK ANALOG_MODULE FOR RPU_CONFIG")
print("=" * 80)

# Find lora_A module
for name, module in model.named_modules():
    if 'lora_A' in name and 'query' in name and isinstance(module, AnalogLinear):
        print(f"\n[Module: {name}]")

        analog_module = module.analog_module
        print(f"\nanalog_module type: {type(analog_module)}")
        print(f"analog_module has rpu_config: {hasattr(analog_module, 'rpu_config')}")

        if hasattr(analog_module, 'rpu_config'):
            rpu_config = analog_module.rpu_config
            print(f"\nrpu_config type: {type(rpu_config)}")
            print(f"rpu_config: {rpu_config}")

            # Check device
            if hasattr(rpu_config, 'device'):
                device = rpu_config.device
                print(f"\nDevice type: {type(device).__name__}")
                if hasattr(device, 'dw_min'):
                    print(f"Device dw_min: {device.dw_min}")
                if hasattr(device, 'w_max'):
                    print(f"Device w_max: {device.w_max}, w_min: {device.w_min}")
                if hasattr(device, 'mult_noise'):
                    print(f"Device mult_noise: {device.mult_noise}")

            # Check mapping
            if hasattr(rpu_config, 'mapping'):
                mapping = rpu_config.mapping
                print(f"\nMapping:")
                print(f"  weight_scaling_omega: {mapping.weight_scaling_omega}")
                print(f"  learn_out_scaling: {mapping.learn_out_scaling}")

            # Check forward
            if hasattr(rpu_config, 'forward'):
                fwd = rpu_config.forward
                print(f"\nForward:")
                print(f"  inp_res: {fwd.inp_res:.6f}")
                print(f"  out_res: {fwd.out_res:.6f}")
                print(f"  noise_management: {fwd.noise_management}")
                print(f"  bound_management: {fwd.bound_management}")

        break

# Check base_layer too
for name, module in model.named_modules():
    if 'base_layer' in name and 'query' in name and isinstance(module, AnalogLinear):
        print(f"\n" + "=" * 80)
        print(f"[Module: {name}]")

        analog_module = module.analog_module
        print(f"\nanalog_module type: {type(analog_module)}")
        print(f"analog_module has rpu_config: {hasattr(analog_module, 'rpu_config')}")

        if hasattr(analog_module, 'rpu_config'):
            rpu_config = analog_module.rpu_config
            print(f"\nrpu_config type: {type(rpu_config)}")

            # Check if it's TorchInferenceRPUConfig
            print(f"Is TorchInferenceRPUConfig: {'TorchInferenceRPUConfig' in str(type(rpu_config))}")

        break

print("\n" + "=" * 80)
