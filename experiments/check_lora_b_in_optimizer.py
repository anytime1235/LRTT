"""
Check if LoRA B tiles are in the optimizer parameter groups
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from aihwkit.optim import AnalogSGD
from aihwkit.nn import AnalogLinear

device = torch.device("cuda")
model = create_glue_model('sst2', device, ['query'], fp_lora=False, lora_alpha=0.01)

print("=" * 80)
print("CHECK: LoRA B Tiles in Optimizer")
print("=" * 80)

# Collect all LoRA B tiles
lora_b_tiles = {}
for name, module in model.named_modules():
    if 'lora_B' in name and isinstance(module, AnalogLinear):
        if hasattr(module, 'analog_module'):
            lora_b_tiles[name] = module.analog_module

print(f"\nFound {len(lora_b_tiles)} LoRA B tiles in model")
for name in list(lora_b_tiles.keys())[:3]:
    print(f"  {name}")

# Setup optimizer
optimizer = AnalogSGD(model.parameters(), lr=0.01)
print(f"\n[Before regroup] Optimizer has {len(optimizer.param_groups)} parameter groups")

optimizer.regroup_param_groups(model)
print(f"[After regroup] Optimizer has {len(optimizer.param_groups)} parameter groups")

# Check which tiles are in optimizer
tiles_in_optimizer = set()
for i, group in enumerate(optimizer.param_groups):
    if 'analog_tiles' in group:
        print(f"\nParameter group {i}: {len(group['analog_tiles'])} analog tiles")
        for tile in group['analog_tiles']:
            tiles_in_optimizer.add(id(tile))

print(f"\nTotal analog tiles in optimizer: {len(tiles_in_optimizer)}")

# Check if LoRA B tiles are in optimizer
lora_b_in_optimizer = 0
lora_b_not_in_optimizer = []

for name, tile in lora_b_tiles.items():
    if id(tile) in tiles_in_optimizer:
        lora_b_in_optimizer += 1
    else:
        lora_b_not_in_optimizer.append(name)

print(f"\nLoRA B tiles in optimizer: {lora_b_in_optimizer} / {len(lora_b_tiles)}")

if lora_b_not_in_optimizer:
    print("\n✗ LoRA B tiles NOT in optimizer:")
    for name in lora_b_not_in_optimizer[:5]:
        print(f"  {name}")
    print("\n⚠ THIS IS THE BUG! LoRA B tiles are not being optimized!")
else:
    print("\n✓ All LoRA B tiles are in optimizer")
    print("\n  Issue must be elsewhere (gradient flow, tile config, etc.)")

# Also check LoRA A for comparison
print("\n" + "-" * 80)
print("Checking LoRA A tiles for comparison:")
print("-" * 80)

lora_a_tiles = {}
for name, module in model.named_modules():
    if 'lora_A' in name and isinstance(module, AnalogLinear):
        if hasattr(module, 'analog_module'):
            lora_a_tiles[name] = module.analog_module

lora_a_in_optimizer = 0
lora_a_not_in_optimizer = []

for name, tile in lora_a_tiles.items():
    if id(tile) in tiles_in_optimizer:
        lora_a_in_optimizer += 1
    else:
        lora_a_not_in_optimizer.append(name)

print(f"\nLoRA A tiles in optimizer: {lora_a_in_optimizer} / {len(lora_a_tiles)}")

if lora_a_not_in_optimizer:
    print("\n✗ LoRA A tiles NOT in optimizer:")
    for name in lora_a_not_in_optimizer[:5]:
        print(f"  {name}")

# Check base_layer too
print("\n" + "-" * 80)
print("Checking base_layer tiles:")
print("-" * 80)

base_tiles = {}
for name, module in model.named_modules():
    if 'base_layer' in name and isinstance(module, AnalogLinear):
        if hasattr(module, 'analog_module'):
            base_tiles[name] = module.analog_module

base_in_optimizer = sum(1 for tile in base_tiles.values() if id(tile) in tiles_in_optimizer)

print(f"\nbase_layer tiles in optimizer: {base_in_optimizer} / {len(base_tiles)}")
print("(These should be 0 since base_layer is frozen)")

print("\n" + "=" * 80)
