#!/usr/bin/env python
"""Check if LRTT tiles are on GPU or CPU"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from transformers import AutoModelForSequenceClassification
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.nn import AnalogLinear

MODEL_NAME = "google/mobilebert-uncased"

print("=" * 80)
print("CHECKING TILE DEVICE LOCATION")
print("=" * 80)

# Load and convert model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
lrtt_config = create_lrtt_lora_config(rank=8, lora_alpha=1.0, use_floating_point=False)
model = convert_model_to_lrtt_lora(model, lrtt_config, ["query"])

print("\nBefore .to('cuda'):")
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        tile = module.analog_module.array[0][0] if hasattr(module.analog_module, 'array') else module.analog_module

        weights_a, _ = tile.tile_a.get_weights()
        weights_b, _ = tile.tile_b.get_weights()
        weights_c, _ = tile.tile_c.get_weights()

        print(f"\n{name}:")
        print(f"  tile_a weights device: {weights_a.device}")
        print(f"  tile_b weights device: {weights_b.device}")
        print(f"  tile_c weights device: {weights_c.device}")
        print(f"  tile.device: {tile.device if hasattr(tile, 'device') else 'N/A'}")
        break

# Move to CUDA
model.to("cuda")

print("\n\nAfter .to('cuda'):")
for name, module in model.named_modules():
    if isinstance(module, AnalogLinear) and 'query' in name:
        tile = module.analog_module.array[0][0] if hasattr(module.analog_module, 'array') else module.analog_module

        weights_a, _ = tile.tile_a.get_weights()
        weights_b, _ = tile.tile_b.get_weights()
        weights_c, _ = tile.tile_c.get_weights()

        print(f"\n{name}:")
        print(f"  tile_a weights device: {weights_a.device}")
        print(f"  tile_b weights device: {weights_b.device}")
        print(f"  tile_c weights device: {weights_c.device}")
        print(f"  tile.device: {tile.device if hasattr(tile, 'device') else 'N/A'}")

        # Check tile operations device
        print(f"\n  Testing forward/backward/update on GPU:")
        x = torch.randn(32, 128, device='cuda')
        y = tile.forward(x, in_trans=False, out_trans=False)
        print(f"    Forward output device: {y.device}")

        d = torch.randn_like(y)
        x_grad = tile.backward(d, in_trans=False, out_trans=False)
        print(f"    Backward output device: {x_grad.device}")

        break

print("\n" + "=" * 80)
print("If all devices show 'cuda:0', tiles are on GPU")
print("If any show 'cpu', that's the bottleneck!")
print("=" * 80)
