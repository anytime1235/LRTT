#!/usr/bin/env python
# coding=utf-8
"""Debug LRTT conversion for MobileBERT."""

import math
import os
import torch
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
)

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.configs import SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    return -TAU_SEC * math.log(1 - delta)


def create_lrtt_config(rank: int, te: int, tlr: float, lifetime: float) -> PythonLRTTRPUConfig:
    dt_batch_sec = lifetime_to_dt_batch_sec(lifetime)
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    ab_lifetime = 1.0 / delta if delta > 0 else 0.0

    SOFTBOUNDS_CONFIG = {
        'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
        'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
        'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
        'write_noise_std': 0.0, 'mult_noise': True,
    }

    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime, lifetime_dtod=0.1,
        reset=0.0, reset_dtod=0.0,
    )
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    return PythonLRTTRPUConfig(device=device_config)


def check_weights(model, name):
    """Check weight statistics."""
    print(f"\n=== {name} ===")
    all_weights = []
    for pname, param in model.named_parameters():
        if param.requires_grad:
            w = param.data.cpu().numpy().flatten()
            all_weights.extend(w)

    all_weights = np.array(all_weights)
    print(f"Total parameters: {len(all_weights):,}")
    print(f"Weight stats:")
    print(f"  Mean: {np.mean(all_weights):.6f}")
    print(f"  Std:  {np.std(all_weights):.6f}")
    print(f"  Min:  {np.min(all_weights):.6f}")
    print(f"  Max:  {np.max(all_weights):.6f}")
    print(f"  Abs mean: {np.mean(np.abs(all_weights)):.6f}")
    print(f"  NaN count: {np.sum(np.isnan(all_weights))}")
    print(f"  Inf count: {np.sum(np.isinf(all_weights))}")

    # Check for very large or very small values
    very_large = np.sum(np.abs(all_weights) > 10)
    very_small = np.sum(np.abs(all_weights) < 1e-6)
    print(f"  |w| > 10: {very_large}")
    print(f"  |w| < 1e-6: {very_small}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test both models
    for model_name in ["google/mobilebert-uncased", "bert-base-uncased"]:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        # Load original model
        model_config = AutoConfig.from_pretrained(model_name, num_labels=2)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=model_config, use_safetensors=True
        )

        check_weights(model, "BEFORE LRTT conversion")

        # Convert to LRTT
        rpu_config = create_lrtt_config(rank=4, te=1000, tlr=0.001, lifetime=100000)
        model = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
        model.to(device)

        check_weights(model, "AFTER LRTT conversion")

        # Do a simple forward pass with random input
        print("\n--- Forward pass test ---")
        dummy_input = {
            "input_ids": torch.randint(0, 1000, (1, 128)).to(device),
            "attention_mask": torch.ones(1, 128).to(device),
        }

        try:
            with torch.no_grad():
                output = model(**dummy_input)
                logits = output.logits
                print(f"Logits shape: {logits.shape}")
                print(f"Logits values: {logits.cpu().numpy()}")
                print(f"Logits mean: {logits.mean().item():.6f}")
                print(f"Logits std: {logits.std().item():.6f}")
                print(f"Logits has NaN: {torch.isnan(logits).any().item()}")
                print(f"Logits has Inf: {torch.isinf(logits).any().item()}")
        except Exception as e:
            print(f"Forward pass error: {e}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
