"""Debug AnalogLinear parameter structure"""

import sys
import torch
sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model

device = torch.device("cuda")
model = create_glue_model("sst2", device, ["query", "key", "value"], fp_lora=False, lora_alpha=0.01)

# Find first LoRA layer
lora_layer = None
lora_name = None
for name, module in model.named_modules():
    if 'lora_A' in name and hasattr(module, 'analog_module'):
        lora_layer = module
        lora_name = name
        break

print(f"\nAnalyzing: {lora_name}")
print(f"Type: {type(lora_layer)}")
print(f"\nAttributes:")
for attr in dir(lora_layer):
    if not attr.startswith('_'):
        print(f"  {attr}")

print(f"\nParameters:")
for name, param in lora_layer.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}, dtype={param.dtype}")
    print(f"    is_leaf={param.is_leaf}, grad_fn={param.grad_fn}")

print(f"\nBuffers:")
for name, buf in lora_layer.named_buffers():
    print(f"  {name}: shape={buf.shape}, dtype={buf.dtype}")

print(f"\nAnalog module:")
print(f"  Type: {type(lora_layer.analog_module)}")
print(f"  Tile: {type(lora_layer.analog_module.tile)}")

# Try to get weights
try:
    weights = lora_layer.get_weights()
    print(f"\nget_weights() returned:")
    if isinstance(weights, tuple):
        for i, w in enumerate(weights):
            if w is not None:
                print(f"  [{i}]: shape={w.shape}, requires_grad={w.requires_grad}")
    else:
        print(f"  shape={weights.shape}, requires_grad={weights.requires_grad}")
except Exception as e:
    print(f"\nget_weights() failed: {e}")

# Check if weights are in parameters()
print(f"\nlist(parameters()):")
for i, p in enumerate(lora_layer.parameters()):
    print(f"  [{i}]: shape={p.shape}, requires_grad={p.requires_grad}")
