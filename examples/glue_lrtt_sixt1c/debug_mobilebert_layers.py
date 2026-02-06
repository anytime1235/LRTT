#!/usr/bin/env python
# coding=utf-8
"""Debug MobileBERT layers to find which ones have large weights."""

import torch
import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification


def main():
    model_name = "google/mobilebert-uncased"
    model_config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config, use_safetensors=True
    )

    print("="*80)
    print("MOBILEBERT LAYER WEIGHT ANALYSIS")
    print("="*80)
    print("\nLayers with |weight| > 5:\n")

    large_weight_layers = []

    for name, param in model.named_parameters():
        w = param.data.cpu().numpy().flatten()
        max_abs = np.max(np.abs(w))
        if max_abs > 5:
            large_weight_layers.append((name, np.min(w), np.max(w), max_abs, np.mean(np.abs(w)), param.shape))
            print(f"{name}")
            print(f"  Shape: {param.shape}")
            print(f"  Range: [{np.min(w):.4f}, {np.max(w):.4f}]")
            print(f"  Abs mean: {np.mean(np.abs(w)):.4f}")
            print()

    print("="*80)
    print("SUMMARY: Layers that need special handling for LRTT")
    print("="*80)
    print("\nThese layers should be EXCLUDED from LRTT conversion:")
    for name, min_v, max_v, max_abs, abs_mean, shape in sorted(large_weight_layers, key=lambda x: -x[3]):
        print(f"  - {name}: range=[{min_v:.2f}, {max_v:.2f}]")


if __name__ == "__main__":
    main()
