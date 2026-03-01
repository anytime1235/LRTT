#!/usr/bin/env python
# coding=utf-8
"""Diagnose output norms: C vs A·B contribution in sixt1c-LoRA.

Forward inject mode: y = C·x + α·A·(B·x)
Check the scale ratio between C output and LoRA output.
"""

import torch
import torch.nn as nn
import sys

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, set_seed
from datasets import load_dataset

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

# LRTT config imports
sys.path.insert(0, '/home/jovyan/work/LRTT/src')
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

MODEL_NAME = "google/mobilebert-uncased"
RANK = 8
LORA_ALPHA = 0.01  # Test value
SEED = 42


def create_sixt1c_lora_config(rank: int, lora_alpha: float):
    """Create Sixt1c-LoRA config (noise-free for diagnosis)."""

    # A/B tiles: LinearStepDevice (noise-free)
    ab_device = LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        gamma_up_dtod=0.0,
        gamma_down_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=0.0,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )

    # C tile: SoftBoundsDevice (noise-free)
    c_device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )

    # Sixt1c-LoRA Device config
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=1000000,
        lora_alpha=lora_alpha,
        reinit_gain=0.1,
        reinit_mode="hybrid",
        decay_factor=1.0,
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


def list_linear_layers(model: nn.Module):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


class OutputCapture:
    """Hook to capture layer outputs."""
    def __init__(self):
        self.outputs = {}

    def hook(self, name):
        def fn(module, input, output):
            self.outputs[name] = output.detach()
        return fn


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*80}")
    print(f"Output Scale Diagnosis: Sixt1c-LoRA")
    print(f"{'='*80}")
    print(f"lora_alpha = {LORA_ALPHA}")
    print(f"rank = {RANK}")
    print(f"Device: {device}")
    print()

    # Create model
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Convert QKV to analog
    all_linear = list_linear_layers(model)
    target_modules = ["query", "key", "value"]
    exclude = [name for name in all_linear if not any(t in name for t in target_modules)]
    exclude.append("qa_outputs")

    rpu_config = create_sixt1c_lora_config(rank=RANK, lora_alpha=LORA_ALPHA)

    print("Converting QKV to analog sixt1c-LoRA...")
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)
    model.to(device)
    model.eval()

    # Load a sample
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset("squad", split="train[:10]")

    sample = dataset[0]
    inputs = tokenizer(
        sample["question"],
        sample["context"],
        max_length=384,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Register hooks on analog layers
    capture = OutputCapture()
    hooks = []

    analog_layers = []
    for name, module in model.named_modules():
        if any(t in name for t in target_modules):
            if hasattr(module, 'analog_module'):  # This is an analog layer
                hook = module.register_forward_hook(capture.hook(name))
                hooks.append(hook)
                analog_layers.append(name)

    print(f"Registered hooks on {len(analog_layers)} analog layers")
    print()

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Analyze outputs
    print(f"{'='*80}")
    print(f"Output Norms Analysis")
    print(f"{'='*80}")
    print(f"{'Layer':<60} {'Output Norm':>15}")
    print(f"{'-'*80}")

    norms = []
    for name in sorted(capture.outputs.keys()):
        output = capture.outputs[name]
        norm = torch.norm(output).item()
        norms.append((name, norm))
        print(f"{name:<60} {norm:>15.4f}")

    print(f"{'='*80}")
    print()

    # Statistics
    norm_values = [n for _, n in norms]
    print(f"Statistics:")
    print(f"  Mean norm: {sum(norm_values)/len(norm_values):.4f}")
    print(f"  Min norm:  {min(norm_values):.4f}")
    print(f"  Max norm:  {max(norm_values):.4f}")
    print()

    # NOTE: In LRTT forward_inject mode, the analog layer output is:
    # y = C·x + α·A·(B·x)
    # The output we capture is the COMBINED output.
    # To separate C and A·B contributions, we'd need to modify LRTT internals.

    print("NOTE: The measured outputs are COMBINED (C·x + α·A·(B·x))")
    print(f"With α={LORA_ALPHA}, the LoRA contribution is scaled by {LORA_ALPHA}x")
    print()
    print("To estimate C vs A·B contribution:")
    print("  - C output scale ~ pretrained weight norms (typically O(1))")
    print(f"  - A·B output scale ~ α × rank_scale ~ {LORA_ALPHA} × O(1)")
    print(f"  - Ratio: C / (α·A·B) ~ 1 / {LORA_ALPHA} = {1/LORA_ALPHA:.0f}x")
    print()
    print("So with α=0.01, C output is ~100x larger than LoRA contribution.")


if __name__ == "__main__":
    main()
