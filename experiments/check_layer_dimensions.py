"""
Check actual layer dimensions in the Sixt1c-LoRA model
"""

import sys
import torch

sys.path.insert(0, '/data/LRTT_transformer/LRTT_glue')

from sweep_sixt1c_lora_glue_adam import create_glue_model
from aihwkit.nn import AnalogLinear

device = torch.device("cuda")
model = create_glue_model('sst2', device, ['query'], fp_lora=False, lora_alpha=0.01)

print("=" * 80)
print("LAYER DIMENSION CHECK")
print("=" * 80)

# Check embedding dimension
if hasattr(model.base_model.model.mobilebert, 'embeddings'):
    emb_config = model.base_model.model.mobilebert.config
    print(f"\nMobileBERT config:")
    print(f"  Embedding size: {emb_config.embedding_size}")
    print(f"  Hidden size: {emb_config.hidden_size}")
    print(f"  True hidden size: {emb_config.true_hidden_size}")

# Find first LoRA layer
print("\n" + "-" * 80)
print("First LoRA layer dimensions:")
print("-" * 80)

for name, module in model.named_modules():
    if 'layer.0.attention.self.query' in name:
        print(f"\nModule: {name}")
        print(f"  Type: {type(module)}")

        if hasattr(module, 'base_layer'):
            base = module.base_layer
            print(f"  base_layer type: {type(base)}")

            if isinstance(base, AnalogLinear):
                try:
                    weights = base.get_weights()
                    if isinstance(weights, tuple):
                        w = weights[0]
                    else:
                        w = weights
                    print(f"  base_layer shape: {w.shape} (out_features x in_features)")
                    print(f"  => in_features: {w.shape[1]}, out_features: {w.shape[0]}")
                except Exception as e:
                    print(f"  Error getting weights: {e}")

        if hasattr(module, 'lora_A'):
            lora_a = module.lora_A['default'] if isinstance(module.lora_A, dict) else module.lora_A
            if lora_a and isinstance(lora_a, AnalogLinear):
                try:
                    weights = lora_a.get_weights()
                    if isinstance(weights, tuple):
                        w = weights[0]
                    else:
                        w = weights
                    print(f"  lora_A shape: {w.shape}")
                    print(f"  => in_features: {w.shape[1]}, out_features (rank): {w.shape[0]}")
                except Exception as e:
                    print(f"  Error getting lora_A weights: {e}")

        if hasattr(module, 'lora_B'):
            lora_b = module.lora_B['default'] if isinstance(module.lora_B, dict) else module.lora_B
            if lora_b and isinstance(lora_b, AnalogLinear):
                try:
                    weights = lora_b.get_weights()
                    if isinstance(weights, tuple):
                        w = weights[0]
                    else:
                        w = weights
                    print(f"  lora_B shape: {w.shape}")
                    print(f"  => in_features (rank): {w.shape[1]}, out_features: {w.shape[0]}")
                except Exception as e:
                    print(f"  Error getting lora_B weights: {e}")

# Test actual forward pass to see what dimensions are expected
print("\n" + "-" * 80)
print("Testing actual forward pass:")
print("-" * 80)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
text = "test"
inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
input_ids = inputs['input_ids'].to(device)

# Get actual embedding output
with torch.no_grad():
    embeddings = model.base_model.model.mobilebert.embeddings(input_ids)

print(f"\nEmbedding output shape: {embeddings.shape}")
print(f"  (batch_size, seq_len, embedding_dim)")

# Try to extract what goes into the query layer by hooking
captured_input = {}

def capture_input_hook(name):
    def hook(module, input, output):
        if isinstance(input, tuple):
            captured_input[name] = input[0].shape
        else:
            captured_input[name] = input.shape
    return hook

# Register hook on the first query layer
for name, module in model.named_modules():
    if 'layer.0.attention.self.query' in name and 'lora' not in name:
        handle = module.register_forward_hook(capture_input_hook(name))
        break

# Run forward pass (move inputs to device)
inputs_on_device = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    try:
        _ = model(**inputs_on_device)
        print(f"\nActual input shape to query layer: {captured_input}")
    except Exception as e:
        print(f"\nError during forward pass: {e}")

handle.remove()

print("\n" + "=" * 80)
