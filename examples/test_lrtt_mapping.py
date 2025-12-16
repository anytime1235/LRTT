#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test: PythonLRTTRPUConfig with MappingParameter
"""

import sys
sys.path.insert(0, '/root/LRTT/src')

import torch
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs import MappingParameter
from aihwkit.simulator.presets.devices import EcRamPresetDevice
from aihwkit.simulator.configs.devices import LinearStepDevice

print("="*80)
print("Test: PythonLRTTRPUConfig with MappingParameter")
print("="*80)

# 6T1C device
sixt1c_device = LinearStepDevice(
    dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
    gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True
)

# EcRAM device
c_device = EcRamPresetDevice()

# LRTT device
lrtt_device = PythonLRTTDevice(
    rank=4,
    transfer_every=100,
    lora_alpha=1.0,
    unit_cell_devices=[sixt1c_device, sixt1c_device, c_device]
)

print("\n1. WITHOUT MappingParameter:")
print("-" * 40)
rpu_cfg_without = PythonLRTTRPUConfig(device=lrtt_device)
print(f"   Has mapping attribute: {hasattr(rpu_cfg_without, 'mapping')}")
print(f"   Default weight_scaling_omega: {rpu_cfg_without.mapping.weight_scaling_omega}")
print(f"   Default max_input_size: {rpu_cfg_without.mapping.max_input_size}")
print(f"   Default max_output_size: {rpu_cfg_without.mapping.max_output_size}")

print("\n2. WITH MappingParameter:")
print("-" * 40)

# Create MappingParameter
mapping = MappingParameter(
    weight_scaling_omega=0.6,
    max_input_size=512,
    max_output_size=512
)

# Apply to PythonLRTTRPUConfig
rpu_cfg_with = PythonLRTTRPUConfig(device=lrtt_device, mapping=mapping)
print(f"   Has mapping attribute: {hasattr(rpu_cfg_with, 'mapping')}")
print(f"   Custom weight_scaling_omega: {rpu_cfg_with.mapping.weight_scaling_omega}")
print(f"   Custom max_input_size: {rpu_cfg_with.mapping.max_input_size}")
print(f"   Custom max_output_size: {rpu_cfg_with.mapping.max_output_size}")

print("\n3. Create AnalogLinear layer with mapping:")
print("-" * 40)
layer = AnalogLinear(784, 256, rpu_config=rpu_cfg_with, bias=False)
print(f"   ✓ Layer created successfully!")
print(f"   Layer shape: {layer.in_features} -> {layer.out_features}")

print("\n4. Test forward pass:")
print("-" * 40)
x = torch.randn(10, 784)
y = layer(x)
print(f"   ✓ Forward pass successful!")
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {y.shape}")

print("\n" + "="*80)
print("RESULT: ✓ PythonLRTTRPUConfig SUPPORTS MappingParameter!")
print("="*80)

print("\nUsage in test_mnist_single_pulse.py:")
print("-" * 80)
print("""
# Current code (line 168):
rpu_cfg_1 = PythonLRTTRPUConfig(device=device_config_layer1)

# With MappingParameter:
from aihwkit.simulator.configs import MappingParameter

mapping = MappingParameter(weight_scaling_omega=0.6)
rpu_cfg_1 = PythonLRTTRPUConfig(device=device_config_layer1, mapping=mapping)

# Then continue with existing code:
io_pars = IOParameters(out_noise=0.006, w_noise_type=WeightNoiseType.NONE)
rpu_cfg_1.forward = io_pars
rpu_cfg_1.backward = io_pars
""")

print("\n" + "="*80)
print("EXPLANATION:")
print("="*80)
print("""
PythonLRTTRPUConfig inherits from IOManagedRPUConfig
  -> IOManagedRPUConfig inherits from MappableRPU
     -> MappableRPU has a 'mapping' field

Therefore:
✓ PythonLRTTRPUConfig supports MappingParameter
✓ You can apply it exactly like TikiTakaPreset
✓ The mapping applies to the entire LRTT tile (not per device)
✓ weight_scaling_omega scales the full weight W = A⊗B + C
""")
