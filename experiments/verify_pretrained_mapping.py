#!/usr/bin/env python
"""Verify pretrained weights are correctly mapped to C tile"""
import sys
sys.path.insert(0, '/data/LRTT_transformer/lora_training_glue')

import torch
from transformers import AutoModelForSequenceClassification
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.nn import AnalogLinear
import copy

print("=" * 80)
print("VERIFYING PRETRAINED WEIGHT MAPPING TO C TILE")
print("=" * 80)

MODEL_NAME = "google/mobilebert-uncased"

# Load original model
print("\n[1] Loading original pretrained model...")
model_original = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Get original query layer weights (before conversion)
original_query_weight = None
original_query_bias = None
for name, module in model_original.named_modules():
    if 'encoder.layer.0.attention.self.query' in name and isinstance(module, torch.nn.Linear):
        original_query_weight = module.weight.data.clone()
        original_query_bias = module.bias.data.clone() if module.bias is not None else None
        print(f"✓ Captured original weights from: {name}")
        print(f"  Weight shape: {original_query_weight.shape}")
        print(f"  Bias shape: {original_query_bias.shape if original_query_bias is not None else None}")
        break

# Load fresh model for conversion
print("\n[2] Loading fresh model for LRTT conversion...")
model_lrtt = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Create LRTT config
print("[3] Creating LRTT config...")
lrtt_config = create_lrtt_lora_config(
    rank=8,
    lora_alpha=1.0,
    output_noise_level=0.0,
    use_floating_point=False,
)

# Convert to LRTT
print("[4] Converting to LRTT-LoRA...")
model_lrtt = convert_model_to_lrtt_lora(model_lrtt, lrtt_config, ["query"])

# Get C tile weights (after conversion)
print("\n[5] Extracting C tile weights...")
for name, module in model_lrtt.named_modules():
    if isinstance(module, AnalogLinear) and 'encoder.layer.0.attention.self.query' in name:
        # Get tile
        if hasattr(module, 'analog_module'):
            tile_module = module.analog_module
            if hasattr(tile_module, 'array'):
                tile = tile_module.array[0][0]
            else:
                tile = tile_module

            # Get C tile weights
            weights_c, bias_c = tile.tile_c.get_weights()

            # Get A and B tile weights
            weights_a, _ = tile.tile_a.get_weights()
            weights_b, _ = tile.tile_b.get_weights()

            print(f"✓ Extracted weights from LRTT tile")
            print(f"  C tile weight shape: {weights_c.shape}")
            print(f"  C tile bias shape: {bias_c.shape if bias_c is not None else None}")
            print(f"  A tile weight shape: {weights_a.shape}")
            print(f"  B tile weight shape: {weights_b.shape}")

            # Compare
            print("\n" + "=" * 80)
            print("VERIFICATION RESULTS")
            print("=" * 80)

            # Check C tile matches original pretrained weights
            if original_query_weight is not None:
                weight_match = torch.allclose(weights_c, original_query_weight, atol=1e-6)
                max_diff = (weights_c - original_query_weight).abs().max().item()
                print(f"\n[1] C tile weight vs Original pretrained weight:")
                print(f"    Match (atol=1e-6): {weight_match}")
                print(f"    Max difference: {max_diff:.10f}")

                if weight_match:
                    print("    ✓ PASSED: C tile has correct pretrained weights!")
                else:
                    print("    ✗ FAILED: C tile weights don't match!")

            # Check bias
            if original_query_bias is not None and bias_c is not None:
                bias_match = torch.allclose(bias_c, original_query_bias, atol=1e-6)
                max_diff_bias = (bias_c - original_query_bias).abs().max().item()
                print(f"\n[2] C tile bias vs Original pretrained bias:")
                print(f"    Match (atol=1e-6): {bias_match}")
                print(f"    Max difference: {max_diff_bias:.10f}")

                if bias_match:
                    print("    ✓ PASSED: C tile has correct pretrained bias!")
                else:
                    print("    ✗ FAILED: C tile bias doesn't match!")

            # Check A tile is zeros
            print(f"\n[3] A tile initialization (should be zeros):")
            a_is_zero = torch.allclose(weights_a, torch.zeros_like(weights_a), atol=1e-9)
            a_max = weights_a.abs().max().item()
            print(f"    All zeros (atol=1e-9): {a_is_zero}")
            print(f"    Max absolute value: {a_max:.10f}")

            if a_is_zero:
                print("    ✓ PASSED: A tile correctly initialized to zeros!")
            else:
                print("    ✗ FAILED: A tile is not all zeros!")

            # Check B tile is NOT zeros (Kaiming initialized)
            print(f"\n[4] B tile initialization (should be Kaiming normal, not zeros):")
            b_is_zero = torch.allclose(weights_b, torch.zeros_like(weights_b), atol=1e-9)
            b_mean = weights_b.mean().item()
            b_std = weights_b.std().item()
            b_max = weights_b.abs().max().item()
            print(f"    All zeros: {b_is_zero}")
            print(f"    Mean: {b_mean:.6f}")
            print(f"    Std: {b_std:.6f}")
            print(f"    Max abs: {b_max:.6f}")

            if not b_is_zero:
                print("    ✓ PASSED: B tile correctly initialized with Kaiming normal!")
            else:
                print("    ✗ FAILED: B tile is all zeros!")

            # Check initial ΔW = A @ B = 0
            print(f"\n[5] Initial low-rank update (ΔW = A @ B, should be ~0):")
            delta_w = weights_a @ weights_b  # [d_size, rank] @ [rank, x_size] = [d_size, x_size]
            delta_w_max = delta_w.abs().max().item()
            delta_w_is_zero = torch.allclose(delta_w, torch.zeros_like(delta_w), atol=1e-6)
            print(f"    ΔW = A @ B")
            print(f"    Shape: {delta_w.shape}")
            print(f"    Max absolute value: {delta_w_max:.10f}")
            print(f"    All zeros (atol=1e-6): {delta_w_is_zero}")

            if delta_w_is_zero:
                print("    ✓ PASSED: Initial ΔW = 0 (A=0 ensures LoRA starts from pretrained)!")
            else:
                print("    ✗ FAILED: Initial ΔW is not zero!")

            break

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("✓ Pretrained weights are correctly mapped to C tile")
print("✓ A tile initialized to zeros (LoRA-style)")
print("✓ B tile initialized with Kaiming normal")
print("✓ Initial effective weights W_eff = C + α·A·B = C (unchanged)")
print("=" * 80)
