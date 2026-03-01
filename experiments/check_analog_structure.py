"""Check the structure of trainable analog layer"""

import torch
import torch.nn as nn
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice

# Create a simple digital linear layer
digital_layer = nn.Linear(10, 5, bias=True)

# Convert to trainable analog using SingleRPUConfig
rpu_config = SingleRPUConfig(device=LinearStepDevice())
analog_layer = AnalogLinear.from_digital(digital_layer, rpu_config)

print("Trainable Analog Layer Structure:")
print(f"  Type: {type(analog_layer)}")
print(f"  Attributes: {[attr for attr in dir(analog_layer) if not attr.startswith('_')]}")

print(f"\nanalog_module type: {type(analog_layer.analog_module)}")
print(f"analog_module attributes: {[attr for attr in dir(analog_layer.analog_module) if not attr.startswith('_')]}")

# Try to get weights
print("\nAttempting to get weights:")
try:
    # Method 1: get_weights()
    weights = analog_layer.get_weights()
    print(f"✓ Method 1 (get_weights): Success! Shape: {weights[0].shape}")
except Exception as e:
    print(f"✗ Method 1 (get_weights): {e}")

try:
    # Method 2: analog_module.get_weights()
    weights = analog_layer.analog_module.get_weights()
    print(f"✓ Method 2 (analog_module.get_weights): Success! Shape: {weights.shape}")
except Exception as e:
    print(f"✗ Method 2 (analog_module.get_weights): {e}")

try:
    # Method 3: Check if weight attribute exists
    if hasattr(analog_layer, 'weight') and analog_layer.weight is not None:
        print(f"✓ Method 3 (weight attribute): Success! Shape: {analog_layer.weight.shape}")
    else:
        print(f"✗ Method 3 (weight attribute): weight is None or doesn't exist")
except Exception as e:
    print(f"✗ Method 3 (weight attribute): {e}")

# Check parameters
print("\nParameters:")
for name, param in analog_layer.named_parameters():
    print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
