#!/usr/bin/env python
# coding=utf-8
"""
Verify 6T1C-LoRA Implementation and Find lora_alpha Scaling Factor

This script:
1. Converts only Query layers in MobileBERT to LRTT-LoRA
2. Evaluates on GLUE SST-2 task
3. Compares forward output differences between FP-LoRA and 6T1C-LoRA
4. Finds the lora_alpha ratio needed to match outputs (for sweep reference)

Usage:
    python verify_6t1c_lora_alpha_scaling.py --mode compare_outputs
    python verify_6t1c_lora_alpha_scaling.py --mode find_scaling
"""

import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from datasets import load_dataset

from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora


def prepare_sst2_data(tokenizer, max_length=128, max_samples=100):
    """Prepare SST-2 validation data."""
    print(f"\n[1/3] Loading SST-2 validation data (max {max_samples} samples)...")

    dataset = load_dataset("glue", "sst2", split="validation")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print(f"  ✓ Loaded {len(dataset)} samples")
    return dataset


def create_model_with_query_lora(model_name, rank, lora_alpha, use_fp=False, classifier_seed=None):
    """Create MobileBERT with only Query layers converted to LRTT-LoRA."""
    print(f"\n[2/3] Creating model (FP={use_fp}, rank={rank}, alpha={lora_alpha})...")

    # Load pretrained model
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Set fixed classifier weights for fair comparison
    if classifier_seed is not None:
        torch.manual_seed(classifier_seed)
        torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(model.classifier.bias)

    # Create LRTT-LoRA config
    lrtt_config = create_lrtt_lora_config(
        rank=rank,
        lora_alpha=lora_alpha,
        output_noise_level=0.0,
        use_floating_point=use_fp,  # FP-LoRA or 6T1C-LoRA
    )

    # Convert only Query layers
    target_modules = ["query"]  # Only attention query layers
    model = convert_model_to_lrtt_lora(model, lrtt_config, target_modules)

    print(f"  ✓ Model created with Query-only LoRA")
    return model


def compare_forward_outputs(model_fp, model_6t1c, dataset, device="cuda"):
    """Compare forward outputs between FP-LoRA and 6T1C-LoRA."""
    print("\n[3/3] Comparing forward outputs...")

    model_fp.to(device).eval()
    model_6t1c.to(device).eval()

    differences = []
    relative_diffs = []

    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, desc="Processing")):
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            attention_mask = example["attention_mask"].unsqueeze(0).to(device)

            # FP-LoRA forward
            output_fp = model_fp(input_ids=input_ids, attention_mask=attention_mask)
            logits_fp = output_fp.logits

            # 6T1C-LoRA forward
            output_6t1c = model_6t1c(input_ids=input_ids, attention_mask=attention_mask)
            logits_6t1c = output_6t1c.logits

            # Compute differences
            abs_diff = (logits_fp - logits_6t1c).abs().max().item()
            rel_diff = (abs_diff / (logits_fp.abs().max().item() + 1e-8))

            differences.append(abs_diff)
            relative_diffs.append(rel_diff)

    # Statistics
    differences = np.array(differences)
    relative_diffs = np.array(relative_diffs)

    print("\n" + "="*80)
    print("FORWARD OUTPUT COMPARISON RESULTS")
    print("="*80)
    print(f"Number of samples: {len(differences)}")
    print(f"\nAbsolute Difference (logits):")
    print(f"  Mean:   {differences.mean():.6f}")
    print(f"  Median: {np.median(differences):.6f}")
    print(f"  Std:    {differences.std():.6f}")
    print(f"  Min:    {differences.min():.6f}")
    print(f"  Max:    {differences.max():.6f}")
    print(f"\nRelative Difference:")
    print(f"  Mean:   {relative_diffs.mean():.4%}")
    print(f"  Median: {np.median(relative_diffs):.4%}")
    print(f"  Max:    {relative_diffs.max():.4%}")
    print("="*80)

    return differences, relative_diffs


def find_alpha_scaling_factor(model_name, rank, dataset, device="cuda",
                              alpha_fp=1.0, alpha_search_range=(0.1, 10.0), num_trials=20):
    """Find the lora_alpha ratio that minimizes output difference."""
    print("\n" + "="*80)
    print("FINDING LORA_ALPHA SCALING FACTOR")
    print("="*80)
    print(f"FP-LoRA alpha: {alpha_fp}")
    print(f"6T1C alpha search range: {alpha_search_range}")
    print(f"Number of trials: {num_trials}")

    # Create FP-LoRA model (reference)
    classifier_seed = 12345
    model_fp = create_model_with_query_lora(model_name, rank, alpha_fp, use_fp=True,
                                           classifier_seed=classifier_seed)
    model_fp.to(device).eval()

    # Search for best alpha
    alpha_min, alpha_max = alpha_search_range
    alpha_values = np.linspace(alpha_min, alpha_max, num_trials)

    results = []

    for alpha_6t1c in tqdm(alpha_values, desc="Alpha search"):
        # Create 6T1C-LoRA model with this alpha
        model_6t1c = create_model_with_query_lora(model_name, rank, alpha_6t1c, use_fp=False,
                                                  classifier_seed=classifier_seed)
        model_6t1c.to(device).eval()

        # Compute average difference on subset
        diffs = []
        with torch.no_grad():
            for i, example in enumerate(dataset):
                if i >= 10:  # Use only 10 samples for speed
                    break

                input_ids = example["input_ids"].unsqueeze(0).to(device)
                attention_mask = example["attention_mask"].unsqueeze(0).to(device)

                output_fp = model_fp(input_ids=input_ids, attention_mask=attention_mask)
                output_6t1c = model_6t1c(input_ids=input_ids, attention_mask=attention_mask)

                diff = (output_fp.logits - output_6t1c.logits).abs().mean().item()
                diffs.append(diff)

        avg_diff = np.mean(diffs)
        results.append((alpha_6t1c, avg_diff))

        # Clean up
        del model_6t1c
        torch.cuda.empty_cache()

    # Find best alpha
    results.sort(key=lambda x: x[1])
    best_alpha, best_diff = results[0]

    print("\n" + "="*80)
    print("ALPHA SCALING SEARCH RESULTS")
    print("="*80)
    print(f"\nBest 5 alpha values:")
    for i, (alpha, diff) in enumerate(results[:5]):
        ratio = alpha / alpha_fp
        print(f"  {i+1}. alpha={alpha:.4f} (ratio={ratio:.4f}), diff={diff:.6f}")

    print(f"\n✓ Optimal 6T1C alpha: {best_alpha:.4f}")
    print(f"✓ Scaling ratio (6T1C/FP): {best_alpha/alpha_fp:.4f}")
    print(f"✓ Minimum difference: {best_diff:.6f}")
    print("="*80)

    # Clean up
    del model_fp
    torch.cuda.empty_cache()

    return best_alpha, best_alpha / alpha_fp, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="compare_outputs",
                       choices=["compare_outputs", "find_scaling"],
                       help="Mode: compare_outputs or find_scaling")
    parser.add_argument("--model_name", type=str, default="google/mobilebert-uncased")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Max samples for evaluation (100 for speed)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # For alpha scaling search
    parser.add_argument("--alpha_min", type=float, default=0.1)
    parser.add_argument("--alpha_max", type=float, default=10.0)
    parser.add_argument("--num_trials", type=int, default=20)

    args = parser.parse_args()

    set_seed(args.seed)

    print("="*80)
    print("6T1C-LORA VERIFICATION & ALPHA SCALING ANALYSIS")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model_name}")
    print(f"LoRA rank: {args.rank}")
    print(f"LoRA alpha (FP): {args.lora_alpha}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("="*80)

    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = prepare_sst2_data(tokenizer, max_samples=args.max_samples)

    if args.mode == "compare_outputs":
        # Mode 1: Compare FP-LoRA vs 6T1C-LoRA with same alpha
        print("\n" + "="*80)
        print("MODE: COMPARE FORWARD OUTPUTS (FP vs 6T1C)")
        print("="*80)

        # Use same classifier seed for fair comparison
        classifier_seed = 12345
        model_fp = create_model_with_query_lora(
            args.model_name, args.rank, args.lora_alpha, use_fp=True,
            classifier_seed=classifier_seed
        )
        model_6t1c = create_model_with_query_lora(
            args.model_name, args.rank, args.lora_alpha, use_fp=False,
            classifier_seed=classifier_seed
        )

        differences, relative_diffs = compare_forward_outputs(
            model_fp, model_6t1c, dataset, args.device
        )

        # Save results
        results_file = "/data/LRTT_transformer/experiments/6t1c_lora_comparison.npz"
        np.savez(results_file,
                 differences=differences,
                 relative_diffs=relative_diffs,
                 rank=args.rank,
                 lora_alpha=args.lora_alpha)
        print(f"\n✓ Results saved to {results_file}")

    elif args.mode == "find_scaling":
        # Mode 2: Find optimal alpha scaling factor
        print("\n" + "="*80)
        print("MODE: FIND LORA_ALPHA SCALING FACTOR")
        print("="*80)

        best_alpha, scaling_ratio, results = find_alpha_scaling_factor(
            args.model_name,
            args.rank,
            dataset,
            args.device,
            alpha_fp=args.lora_alpha,
            alpha_search_range=(args.alpha_min, args.alpha_max),
            num_trials=args.num_trials,
        )

        # Save results
        results_file = "/data/LRTT_transformer/experiments/6t1c_alpha_scaling.npz"
        alpha_vals = np.array([r[0] for r in results])
        diff_vals = np.array([r[1] for r in results])

        np.savez(results_file,
                 alpha_values=alpha_vals,
                 differences=diff_vals,
                 best_alpha=best_alpha,
                 scaling_ratio=scaling_ratio,
                 alpha_fp=args.lora_alpha,
                 rank=args.rank)
        print(f"\n✓ Results saved to {results_file}")

        # Print recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION FOR SWEEP")
        print("="*80)
        print(f"When using FP-LoRA with alpha={args.lora_alpha:.1f}:")
        print(f"  Use 6T1C-LoRA with alpha ≈ {best_alpha:.4f}")
        print(f"  Scaling ratio: {scaling_ratio:.4f}x")
        print(f"\nFor sweeps, consider alpha range:")
        print(f"  6T1C alpha ∈ [{best_alpha*0.5:.4f}, {best_alpha*2:.4f}]")
        print("="*80)


if __name__ == "__main__":
    main()
