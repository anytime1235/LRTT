#!/usr/bin/env python3
"""Check that bias parameters are correctly frozen in TikiTaka models."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoConfig
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogLinear

# Import TikiTaka config from sweep script
from sweep_tikitaka_batch256 import create_tikitaka_v2_config, TARGET_MODULES, MODEL_NAME

def list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def check_model(model, model_type, head_name):
    """Check bias freeze status for a model."""
    print(f"\n{'='*70}")
    print(f"  {model_type} Model — Bias Freeze Check")
    print(f"{'='*70}")

    trainable_bias = []
    frozen_bias = []
    trainable_weight = []
    frozen_weight = []
    trainable_other = []
    frozen_other = []

    for name, param in model.named_parameters():
        is_bias = "bias" in name
        is_weight = "weight" in name and not is_bias

        if is_bias:
            if param.requires_grad:
                trainable_bias.append((name, param.shape))
            else:
                frozen_bias.append((name, param.shape))
        elif is_weight:
            if param.requires_grad:
                trainable_weight.append((name, param.shape))
            else:
                frozen_weight.append((name, param.shape))
        else:
            if param.requires_grad:
                trainable_other.append((name, param.shape))
            else:
                frozen_other.append((name, param.shape))

    # Summary
    print(f"\n--- BIAS Parameters ---")
    print(f"  Trainable bias:  {len(trainable_bias)}")
    print(f"  Frozen bias:     {len(frozen_bias)}")

    if trainable_bias:
        print(f"\n  [WARNING] Trainable biases found:")
        for name, shape in trainable_bias:
            print(f"    TRAINABLE: {name} {list(shape)}")
    else:
        print(f"\n  [OK] All biases are frozen.")

    if frozen_bias:
        print(f"\n  Frozen biases (sample):")
        for name, shape in frozen_bias[:5]:
            print(f"    FROZEN:    {name} {list(shape)}")
        if len(frozen_bias) > 5:
            print(f"    ... and {len(frozen_bias) - 5} more")

    # Weight summary
    print(f"\n--- WEIGHT Parameters ---")
    print(f"  Trainable weight: {len(trainable_weight)}")
    print(f"  Frozen weight:    {len(frozen_weight)}")

    if trainable_weight:
        print(f"\n  Trainable weights (sample):")
        for name, shape in trainable_weight[:5]:
            print(f"    TRAINABLE: {name} {list(shape)}")
        if len(trainable_weight) > 5:
            print(f"    ... and {len(trainable_weight) - 5} more")

    # Other (out_scaling, etc.)
    print(f"\n--- OTHER Parameters (out_scaling, etc.) ---")
    print(f"  Trainable other: {len(trainable_other)}")
    print(f"  Frozen other:    {len(frozen_other)}")
    if trainable_other:
        for name, shape in trainable_other[:3]:
            print(f"    TRAINABLE: {name} {list(shape)}")
        if len(trainable_other) > 3:
            print(f"    ... and {len(trainable_other) - 3} more")

    # Total
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bias_params = sum(p.numel() for _, p in model.named_parameters() if "bias" in _)
    trainable_bias_params = sum(p.numel() for n, p in model.named_parameters() if "bias" in n and p.requires_grad)

    print(f"\n--- TOTAL ---")
    print(f"  Total params:       {total_params:,}")
    print(f"  Trainable params:   {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Total bias params:  {bias_params:,}")
    print(f"  Trainable bias:     {trainable_bias_params:,}")
    print(f"  BIAS FROZEN: {'YES' if trainable_bias_params == 0 else 'NO !!!'}")

    return trainable_bias_params == 0


def main():
    device = torch.device("cpu")
    params = {
        "transfer_every": 20,
        "transfer_lr": 0.05,
        "fast_lr": 2.19,
        "auto_granularity": 169.0,
        "in_chop_prob": 0.061,
    }

    # --- SQuAD Model ---
    print("Loading SQuAD model...")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("qa_outputs")

    rpu_config = create_tikitaka_v2_config(**params)
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        if "bias" in name:
            param.requires_grad = False
        else:
            param.requires_grad = is_target or "qa_outputs" in name

    squad_ok = check_model(model, "SQuAD (QA)", "qa_outputs")
    del model

    # --- GLUE Model (SST-2 as example) ---
    print("\nLoading GLUE model (sst2)...")
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("classifier")

    rpu_config = create_tikitaka_v2_config(**params)
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        if "bias" in name:
            param.requires_grad = False
        else:
            param.requires_grad = is_target or "classifier" in name

    glue_ok = check_model(model, "GLUE (SST-2)", "classifier")

    # Final verdict
    print(f"\n{'='*70}")
    print(f"  FINAL VERDICT")
    print(f"{'='*70}")
    print(f"  SQuAD bias frozen: {'PASS' if squad_ok else 'FAIL'}")
    print(f"  GLUE bias frozen:  {'PASS' if glue_ok else 'FAIL'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
