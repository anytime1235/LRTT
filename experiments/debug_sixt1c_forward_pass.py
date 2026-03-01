#!/usr/bin/env python
# coding=utf-8
"""
Debug Sixt1c-LoRA Forward Pass and Gradient Flow

Monitors:
1. First forward pass activation values
2. Loss computation
3. Gradient values after backward
4. LoRA A/B/C tile outputs
5. Weight ranges and statistics
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

SEED = 42
MODEL_NAME = "google/mobilebert-uncased"

def check_tensor(name, tensor, step=""):
    """Check tensor for NaN/Inf and report statistics."""
    if tensor is None:
        print(f"  {step}{name}: None")
        return False

    is_nan = torch.isnan(tensor).any().item()
    is_inf = torch.isinf(tensor).any().item()

    if is_nan or is_inf:
        print(f"  {step}{name}: ❌ NaN={is_nan}, Inf={is_inf}")
        return False

    tensor_np = tensor.detach().cpu().float()
    stats = {
        'min': tensor_np.min().item(),
        'max': tensor_np.max().item(),
        'mean': tensor_np.mean().item(),
        'std': tensor_np.std().item(),
    }
    print(f"  {step}{name}: min={stats['min']:.6f}, max={stats['max']:.6f}, "
          f"mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    return True


def hook_forward(module, input, output, name):
    """Forward hook to monitor activations."""
    print(f"\n[FORWARD] {name}")
    if isinstance(input, tuple):
        for i, inp in enumerate(input):
            if isinstance(inp, torch.Tensor):
                check_tensor(f"input[{i}]", inp)

    if isinstance(output, tuple):
        for i, out in enumerate(output):
            if isinstance(out, torch.Tensor):
                check_tensor(f"output[{i}]", out)
    elif isinstance(output, torch.Tensor):
        check_tensor("output", output)


def hook_backward(module, grad_input, grad_output, name):
    """Backward hook to monitor gradients."""
    print(f"\n[BACKWARD] {name}")
    if isinstance(grad_output, tuple):
        for i, grad in enumerate(grad_output):
            if isinstance(grad, torch.Tensor):
                check_tensor(f"grad_output[{i}]", grad)

    if isinstance(grad_input, tuple):
        for i, grad in enumerate(grad_input):
            if isinstance(grad, torch.Tensor):
                check_tensor(f"grad_input[{i}]", grad)


def main():
    print("=" * 80)
    print("DEBUG: Sixt1c-LoRA Forward Pass & Gradient Flow")
    print("=" * 80)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load tokenizer and create dummy input
    print("[1/6] Creating dummy input...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dummy_text = "This is a test sentence for debugging."
    inputs = tokenizer(
        dummy_text,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    labels = torch.tensor([1]).to(device)  # Binary classification

    print(f"  Input shape: {inputs['input_ids'].shape}")
    print(f"  Label: {labels.item()}\n")

    # Load model
    print("[2/6] Loading model...")
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    print(f"  ✓ Model loaded\n")

    # Convert to LRTT-LoRA
    print("[3/6] Converting to Sixt1c-LoRA...")
    lora_alpha = 1.22  # Same as failed test
    lrtt_config = create_lrtt_lora_config(
        rank=8,
        lora_alpha=lora_alpha,
        output_noise_level=0.0,
        use_floating_point=False,  # 6T1C mode
    )

    target_modules = ["query", "key", "value"]
    model = convert_model_to_lrtt_lora(model, lrtt_config, target_modules)
    model.to(device)
    model.train()

    print(f"  ✓ Converted to Sixt1c-LoRA (alpha={lora_alpha})\n")

    # Register hooks on first LoRA layer
    print("[4/6] Registering hooks on first LoRA layer...")
    first_lora_layer = None
    for name, module in model.named_modules():
        if "layer.0.attention.self.query" in name and hasattr(module, 'lora_A'):
            first_lora_layer = (name, module)
            break

    if first_lora_layer:
        name, module = first_lora_layer
        print(f"  Monitoring: {name}")
        module.register_forward_hook(
            lambda m, i, o: hook_forward(m, i, o, name)
        )
        module.register_full_backward_hook(
            lambda m, gi, go: hook_backward(m, gi, go, name)
        )
    else:
        print("  ⚠️  No LoRA layer found!")

    print()

    # Check initial weights
    print("[5/6] Checking initial weights...")
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            print(f"\n{name}:")
            check_tensor("  weights", param.data)
            if param.data.abs().max() > 1e10:
                print(f"  ⚠️  WARNING: Very large weight detected!")

    print("\n" + "=" * 80)
    print("FORWARD PASS")
    print("=" * 80)

    # Forward pass
    print("\n[Step 1] Running forward pass...")
    try:
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        loss = outputs.loss

        print("\n[Step 1 Results]")
        check_tensor("logits", logits)
        check_tensor("loss", loss)

        if loss.item() > 1e10:
            print(f"  ⚠️  WARNING: Loss exploded! loss={loss.item():.6e}")
        elif loss.item() == 0.0:
            print(f"  ⚠️  WARNING: Loss is exactly zero!")

    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("BACKWARD PASS")
    print("=" * 80)

    # Backward pass
    print("\n[Step 2] Running backward pass...")
    try:
        loss.backward()

        print("\n[Step 2 Results] Checking gradients...")
        grad_stats = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if "lora_A" in name or "lora_B" in name or "classifier" in name:
                    print(f"\n{name}:")
                    is_ok = check_tensor("  grad", param.grad)

                    grad_norm = param.grad.norm().item()
                    grad_stats.append((name, grad_norm))

                    if not is_ok:
                        print(f"  ❌ Gradient has NaN/Inf!")

        print("\n[Step 2 Summary] Gradient norms:")
        for name, norm in grad_stats:
            print(f"  {name}: {norm:.6e}")

        total_grad_norm = sum([norm for _, norm in grad_stats])
        if np.isnan(total_grad_norm) or np.isinf(total_grad_norm):
            print(f"\n  ❌ Total gradient norm is NaN/Inf!")
        else:
            print(f"\n  ✓ Total gradient norm: {total_grad_norm:.6e}")

    except Exception as e:
        print(f"\n❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    # Diagnose
    print("\n[6/6] Diagnosis:")

    if loss.item() > 1e10:
        print("  ❌ Loss explosion detected (>1e10)")
        print("  → Likely cause: Forward pass numerical instability")
        print("  → Check: Model output range, weight initialization")

    if torch.isnan(loss) or torch.isinf(loss):
        print("  ❌ Loss is NaN/Inf")
        print("  → Likely cause: Invalid model output or label")

    has_nan_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                has_nan_grad = True
                print(f"  ❌ NaN/Inf gradient in: {name}")

    if has_nan_grad:
        print("  → Likely cause: Loss explosion or division by zero")
        print("  → Check: Gradient clipping, loss scaling")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
