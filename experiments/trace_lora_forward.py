"""
Trace PEFT LoRA forward pass with analog layers
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from transformers import AutoTokenizer

device = torch.device("cuda")
model = create_glue_model('sst2', device, ['query'], fp_lora=False, lora_alpha=0.01)
tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

# Find first LoRA layer
lora_layer = None
for name, module in model.named_modules():
    if 'layer.0.attention.self.query' in name and 'lora' not in name:
        if hasattr(module, 'lora_A'):
            lora_layer = module
            break

print(f"Testing LoRA layer: {type(lora_layer)}")
print(f"base_layer: {type(lora_layer.base_layer)}")
print(f"lora_A: {type(lora_layer.lora_A['default'])}")
print(f"lora_B: {type(lora_layer.lora_B['default'])}")
print(f"scaling: {lora_layer.scaling['default']}")

# Create input
text = "test"
inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
input_ids = inputs['input_ids'].to(device)

# Hook to capture actual input to query layer
captured = {}

def capture_hook(module, input, output):
    if isinstance(input, tuple):
        captured['input'] = input[0].clone()
    else:
        captured['input'] = input.clone()

# Register hook on query layer
handle = lora_layer.register_forward_hook(capture_hook)

# Run forward to capture input
with torch.no_grad():
    _ = model(input_ids=input_ids, attention_mask=inputs['attention_mask'].to(device))

handle.remove()

# Get captured input (will be [batch, seq_len, hidden_dim])
x = captured['input'][:, 0:1, :]  # Take first token: [1, 1, 128]

print(f"\nCaptured input shape: {x.shape}")
print(f"Input mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")

# Manual forward
# AnalogLinear needs 2D input [batch*seq, features]
x_2d = x.reshape(-1, x.shape[-1])  # [1, 128]
print(f"\nReshaped input for AnalogLinear: {x_2d.shape}")

with torch.no_grad():
    # Base layer
    base_out = lora_layer.base_layer(x_2d)
    print(f"\nBase output shape: {base_out.shape}")
    print(f"Base output mean: {base_out.mean().item():.6f}, std: {base_out.std().item():.6f}")
    print(f"Base output: {base_out[0, :5]}")

    # LoRA A
    lora_a_out = lora_layer.lora_A['default'](x_2d)
    print(f"\nLoRA A output shape: {lora_a_out.shape}")
    print(f"LoRA A output mean: {lora_a_out.mean().item():.6f}, std: {lora_a_out.std().item():.6f}")
    print(f"LoRA A output: {lora_a_out[0, :]}")

    # LoRA B
    lora_b_out = lora_layer.lora_B['default'](lora_a_out)
    print(f"\nLoRA B output shape: {lora_b_out.shape}")
    print(f"LoRA B output mean: {lora_b_out.mean().item():.6f}, std: {lora_b_out.std().item():.6f}")
    print(f"LoRA B output: {lora_b_out[0, :5]}")

    # LoRA contribution
    scaling = lora_layer.scaling['default']
    lora_contribution = scaling * lora_b_out
    print(f"\nLoRA contribution (scaled) mean: {lora_contribution.mean().item():.6f}")
    print(f"LoRA contribution: {lora_contribution[0, :5]}")

    # Final output (manual)
    manual_out = base_out + lora_contribution
    print(f"\nManual final output: {manual_out[0, :5]}")

    # PEFT forward (use 3D input - PEFT handles reshape internally)
    peft_out = lora_layer(x)
    # Reshape PEFT output to 2D for comparison
    peft_out_2d = peft_out.reshape(-1, peft_out.shape[-1])
    print(f"\nPEFT forward output shape: {peft_out.shape} -> {peft_out_2d.shape}")
    print(f"PEFT forward output: {peft_out_2d[0, :5]}")

    # Compare
    diff = (manual_out - peft_out_2d).abs().max().item()
    print(f"\nDifference (manual vs PEFT): {diff:.10f}")

    if diff < 1e-5:
        print("✓ Manual forward matches PEFT forward")
    else:
        print("✗ Manual forward DOES NOT match PEFT forward!")

    # Check LoRA contribution magnitude
    lora_mag = lora_contribution.abs().max().item()
    base_mag = base_out.abs().max().item()

    print(f"\nLoRA contribution magnitude: {lora_mag:.6f}")
    print(f"Base output magnitude: {base_mag:.6f}")
    print(f"LoRA / Base ratio: {lora_mag / base_mag if base_mag > 0 else 0:.6f}")

    if lora_mag > 1e-6:
        print("✓ LoRA IS contributing")
    else:
        print("✗ LoRA NOT contributing (too small)")
