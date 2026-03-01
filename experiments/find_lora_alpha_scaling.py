#!/usr/bin/env python
"""
Find lora_alpha scaling factor between FP-LoRA and 6T1C-LoRA.

This script sets A and B to fixed non-zero values and sweeps alpha_6t1c
to find what value matches FP-LoRA output.
"""

import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")

from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, set_seed
from datasets import load_dataset
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora
from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile


def set_fixed_ab_weights(model, a_scale=0.01, b_scale=0.01, seed=42):
    """Set fixed A/B weights in all LRTT query layers."""
    torch.manual_seed(seed)

    for name, module in model.named_modules():
        if "query" in name and hasattr(module, "analog_module"):
            # Get tile
            tile = module.analog_module
            if not isinstance(tile, LRTTSimulatorTile):
                from aihwkit.simulator.tiles.array import TileModuleArray
                if isinstance(tile, TileModuleArray):
                    tile = tile.array[0][0]

            # Set fixed A weights (small but non-zero)
            weights_a, _ = tile.tile_a.get_weights()
            torch.nn.init.normal_(weights_a, mean=0.0, std=a_scale)
            tile.tile_a.set_weights(weights_a)

            # Set fixed B weights
            weights_b, _ = tile.tile_b.get_weights()
            torch.nn.init.normal_(weights_b, mean=0.0, std=b_scale)
            tile.tile_b.set_weights(weights_b)


def compute_output_difference(model_fp, model_6t1c, dataset, device="cuda", num_samples=20):
    """Compute average output difference between FP and 6T1C models."""
    model_fp.to(device).eval()
    model_6t1c.to(device).eval()

    diffs = []

    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break

            input_ids = example["input_ids"].unsqueeze(0).to(device)
            attention_mask = example["attention_mask"].unsqueeze(0).to(device)

            output_fp = model_fp(input_ids=input_ids, attention_mask=attention_mask)
            output_6t1c = model_6t1c(input_ids=input_ids, attention_mask=attention_mask)

            diff = (output_fp.logits - output_6t1c.logits).abs().mean().item()
            diffs.append(diff)

    return np.mean(diffs), np.std(diffs)


def find_alpha_scaling_factor(
    model_name="google/mobilebert-uncased",
    rank=8,
    alpha_fp=1.0,
    alpha_range=(0.1, 10.0),
    num_trials=30,
    device="cuda",
    seed=42
):
    """Find optimal alpha_6t1c / alpha_fp ratio."""
    print("="*80)
    print("LORA_ALPHA SCALING FACTOR SEARCH")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Rank: {rank}")
    print(f"FP alpha (reference): {alpha_fp}")
    print(f"6T1C alpha search range: {alpha_range}")
    print(f"Number of trials: {num_trials}")
    print("="*80)

    set_seed(seed)

    # Load dataset
    print("\n[1/5] Loading SST-2 validation data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("glue", "sst2", split="validation")
    dataset = dataset.select(range(min(50, len(dataset))))

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=128,
            truncation=True,
        )

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    print(f"  ✓ Loaded {len(dataset)} samples")

    # Create FP-LoRA model (reference)
    print("\n[2/5] Creating FP-LoRA reference model...")
    config = AutoConfig.from_pretrained(model_name, num_labels=2)
    model_fp_base = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    # Set fixed classifier
    torch.manual_seed(12345)
    torch.nn.init.normal_(model_fp_base.classifier.weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(model_fp_base.classifier.bias)

    lrtt_config_fp = create_lrtt_lora_config(rank=rank, lora_alpha=alpha_fp, use_floating_point=True)
    model_fp = convert_model_to_lrtt_lora(model_fp_base, lrtt_config_fp, ["query"])

    # Set fixed A/B weights (NON-ZERO to test LoRA contribution)
    print("\n[3/5] Setting fixed A/B weights (non-zero for LoRA effect)...")
    set_fixed_ab_weights(model_fp, a_scale=0.01, b_scale=0.01, seed=seed)
    print("  ✓ A/B weights set (std=0.01)")

    # Search for best alpha_6t1c
    print(f"\n[4/5] Searching for optimal alpha_6t1c...")
    alpha_min, alpha_max = alpha_range
    alpha_values = np.linspace(alpha_min, alpha_max, num_trials)

    results = []

    for alpha_6t1c in tqdm(alpha_values, desc="Alpha sweep"):
        # Create 6T1C-LoRA model with this alpha
        config_6t1c = AutoConfig.from_pretrained(model_name, num_labels=2)
        model_6t1c_base = AutoModelForSequenceClassification.from_pretrained(model_name, config=config_6t1c)

        # Same fixed classifier
        torch.manual_seed(12345)
        torch.nn.init.normal_(model_6t1c_base.classifier.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(model_6t1c_base.classifier.bias)

        lrtt_config_6t1c = create_lrtt_lora_config(rank=rank, lora_alpha=alpha_6t1c, use_floating_point=False)
        model_6t1c = convert_model_to_lrtt_lora(model_6t1c_base, lrtt_config_6t1c, ["query"])

        # Set SAME fixed A/B weights
        set_fixed_ab_weights(model_6t1c, a_scale=0.01, b_scale=0.01, seed=seed)

        # Compute difference
        avg_diff, std_diff = compute_output_difference(model_fp, model_6t1c, dataset, device, num_samples=20)
        results.append((alpha_6t1c, avg_diff, std_diff))

        # Clean up
        del model_6t1c, model_6t1c_base
        torch.cuda.empty_cache()

    # Find best alpha
    results.sort(key=lambda x: x[1])
    best_alpha, best_diff, best_std = results[0]
    scaling_ratio = best_alpha / alpha_fp

    # Print results
    print("\n[5/5] Results:")
    print("="*80)
    print(f"\nTop 10 alpha values:")
    for i, (alpha, diff, std) in enumerate(results[:10]):
        ratio = alpha / alpha_fp
        print(f"  {i+1:2d}. alpha={alpha:6.3f} (ratio={ratio:6.3f}), diff={diff:.6f} ± {std:.6f}")

    print(f"\n{'='*80}")
    print("OPTIMAL SCALING FACTOR")
    print("="*80)
    print(f"  FP-LoRA alpha: {alpha_fp:.3f}")
    print(f"  Optimal 6T1C alpha: {best_alpha:.3f}")
    print(f"  Scaling ratio (6T1C/FP): {scaling_ratio:.3f}x")
    print(f"  Minimum difference: {best_diff:.6f} ± {best_std:.6f}")
    print("="*80)

    print(f"\n💡 RECOMMENDATION FOR SWEEPS:")
    print(f"  When using FP-LoRA with alpha={alpha_fp}:")
    print(f"    → Use 6T1C-LoRA with alpha ≈ {best_alpha:.3f}")
    print(f"  ")
    print(f"  For alpha sweeps:")
    print(f"    FP-LoRA:   alpha ∈ [0.1, 0.5, 1.0, 2.0, 5.0]")
    print(f"    6T1C-LoRA: alpha ∈ [{0.1*scaling_ratio:.2f}, {0.5*scaling_ratio:.2f}, {1.0*scaling_ratio:.2f}, {2.0*scaling_ratio:.2f}, {5.0*scaling_ratio:.2f}]")
    print(f"              (multiply by {scaling_ratio:.3f}x)")
    print("="*80)

    # Save results
    results_file = "/data/LRTT_transformer/experiments/lora_alpha_scaling_results.npz"
    alpha_vals = np.array([r[0] for r in results])
    diff_vals = np.array([r[1] for r in results])
    std_vals = np.array([r[2] for r in results])

    np.savez(results_file,
             alpha_values=alpha_vals,
             differences=diff_vals,
             stds=std_vals,
             best_alpha=best_alpha,
             scaling_ratio=scaling_ratio,
             alpha_fp=alpha_fp,
             rank=rank,
             best_diff=best_diff,
             best_std=best_std)

    print(f"\n✓ Results saved to {results_file}")

    # Clean up
    del model_fp, model_fp_base
    torch.cuda.empty_cache()

    return scaling_ratio, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha_fp", type=float, default=1.0, help="FP-LoRA alpha (reference)")
    parser.add_argument("--alpha_min", type=float, default=0.1, help="6T1C alpha search min")
    parser.add_argument("--alpha_max", type=float, default=10.0, help="6T1C alpha search max")
    parser.add_argument("--num_trials", type=int, default=30, help="Number of alpha values to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    scaling_ratio, results = find_alpha_scaling_factor(
        rank=args.rank,
        alpha_fp=args.alpha_fp,
        alpha_range=(args.alpha_min, args.alpha_max),
        num_trials=args.num_trials,
        device=args.device,
        seed=args.seed,
    )

    print(f"\n✅ DONE! Scaling ratio: {scaling_ratio:.3f}x")
