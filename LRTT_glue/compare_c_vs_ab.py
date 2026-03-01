#!/usr/bin/env python
# coding=utf-8
"""Compare C output vs A·B contribution by testing alpha=0 vs alpha=0.01.

alpha=0: y = C·x (pure base layer)
alpha=0.01: y = C·x + 0.01·A·(B·x)
Difference gives us the A·B contribution scale.
"""

import torch
import torch.nn as nn
import sys
import numpy as np

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, set_seed
from datasets import load_dataset

sys.path.insert(0, '/home/jovyan/work/LRTT/src')
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
SEED = 42


def create_sixt1c_lora_config(rank: int, lora_alpha: float):
    """Create noise-free sixt1c-LoRA config."""
    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.0, up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        gamma_up_dtod=0.0, gamma_down_dtod=0.0, dw_min_std=0.0,
        write_noise_std=0.0, mean_bound_reference=True,
        lifetime=0.0, lifetime_dtod=0.0, reset=0.0, reset_dtod=0.0,
    )

    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0, dw_min_dtod=0.0, dw_min_std=0.0,
        up_down=0.0, up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=True,
    )

    device_config = PythonLRTTDevice(
        rank=rank, transfer_every=1000000, lora_alpha=lora_alpha,
        reinit_gain=0.1, reinit_mode="hybrid", decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.001
    device_config.units_in_mbatch = True
    device_config.forward_inject = True
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


def list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def create_model(lora_alpha, device):
    """Create model with given lora_alpha."""
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    all_linear = list_linear_layers(model)
    target_modules = ["query", "key", "value"]
    exclude = [name for name in all_linear if not any(t in name for t in target_modules)]
    exclude.append("qa_outputs")

    rpu_config = create_sixt1c_lora_config(rank=RANK, lora_alpha=lora_alpha)
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)
    model.to(device)
    model.eval()
    return model


class LayerOutputCapture:
    def __init__(self):
        self.outputs = {}

    def hook(self, name):
        def fn(module, input, output):
            self.outputs[name] = output.detach().cpu()
        return fn


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*80}")
    print(f"C vs A·B Contribution Analysis")
    print(f"{'='*80}")
    print()

    # Load sample data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("squad", split="train[:10]")
    sample = dataset[0]
    inputs = tokenizer(
        sample["question"], sample["context"],
        max_length=384, truncation=True, padding="max_length",
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Test 1: alpha=0 (pure C output)
    print("Creating model with alpha=0 (pure C output)...")
    model_alpha0 = create_model(lora_alpha=0.0, device=device)

    capture0 = LayerOutputCapture()
    hooks0 = []
    target_modules = ["query", "key", "value"]
    layer_names = []

    for name, module in model_alpha0.named_modules():
        if any(t in name for t in target_modules) and hasattr(module, 'analog_module'):
            hook = module.register_forward_hook(capture0.hook(name))
            hooks0.append(hook)
            layer_names.append(name)

    with torch.no_grad():
        _ = model_alpha0(**inputs)

    for hook in hooks0:
        hook.remove()

    print(f"Captured {len(capture0.outputs)} layers with alpha=0")
    print()

    # Test 2: alpha=0.01 (C + 0.01·A·B)
    print("Creating model with alpha=0.01 (C + LoRA)...")
    model_alpha001 = create_model(lora_alpha=0.01, device=device)

    capture001 = LayerOutputCapture()
    hooks001 = []

    for name, module in model_alpha001.named_modules():
        if any(t in name for t in target_modules) and hasattr(module, 'analog_module'):
            hook = module.register_forward_hook(capture001.hook(name))
            hooks001.append(hook)

    with torch.no_grad():
        _ = model_alpha001(**inputs)

    for hook in hooks001:
        hook.remove()

    print(f"Captured {len(capture001.outputs)} layers with alpha=0.01")
    print()

    # Compare outputs
    print(f"{'='*80}")
    print(f"Output Comparison: C vs A·B Contribution")
    print(f"{'='*80}")
    print(f"{'Layer':<50} {'||C·x||':>12} {'||0.01AB||':>12} {'Ratio':>10}")
    print(f"{'-'*80}")

    c_norms = []
    ab_norms = []
    ratios = []

    for name in sorted(layer_names):
        if name in capture0.outputs and name in capture001.outputs:
            c_output = capture0.outputs[name]
            combined_output = capture001.outputs[name]

            # Difference: (C·x + 0.01·A·B·x) - (C·x) = 0.01·A·B·x
            ab_contribution = combined_output - c_output

            c_norm = torch.norm(c_output).item()
            ab_norm = torch.norm(ab_contribution).item()

            if ab_norm > 0:
                ratio = c_norm / ab_norm
            else:
                ratio = float('inf')

            c_norms.append(c_norm)
            ab_norms.append(ab_norm)
            ratios.append(ratio if ratio != float('inf') else 0)

            # Print short layer name
            short_name = name.split('.')[-3] + '.' + name.split('.')[-1]
            print(f"layer.{name.split('.')[3]}.{short_name:<40} {c_norm:>12.2f} {ab_norm:>12.2f} {ratio:>10.1f}x")

    print(f"{'='*80}")
    print()

    # Statistics
    print("Statistics:")
    print(f"  C output (||C·x||):")
    print(f"    Mean: {np.mean(c_norms):.2f}")
    print(f"    Std:  {np.std(c_norms):.2f}")
    print(f"    Range: [{np.min(c_norms):.2f}, {np.max(c_norms):.2f}]")
    print()
    print(f"  LoRA contribution (||0.01·A·B·x||):")
    print(f"    Mean: {np.mean(ab_norms):.2f}")
    print(f"    Std:  {np.std(ab_norms):.2f}")
    print(f"    Range: [{np.min(ab_norms):.2f}, {np.max(ab_norms):.2f}]")
    print()
    valid_ratios = [r for r in ratios if r > 0]
    print(f"  Ratio (||C·x|| / ||0.01·A·B·x||):")
    print(f"    Mean: {np.mean(valid_ratios):.1f}x")
    print(f"    Median: {np.median(valid_ratios):.1f}x")
    print(f"    Range: [{np.min(valid_ratios):.1f}x, {np.max(valid_ratios):.1f}x]")
    print()
    print(f"**With alpha=0.01, C output is ~{np.mean(valid_ratios):.0f}x larger than LoRA contribution**")


if __name__ == "__main__":
    main()
