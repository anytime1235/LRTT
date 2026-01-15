# -*- coding: utf-8 -*-
"""Run all ViT-SPT-LSA experiments and plot combined results.

This script runs all 4 experiments (FP, TTv1, TTv2, LRTT) sequentially
and generates a combined plot of the results.
"""

import os
import sys
import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# Results directories
RESULTS_BASE = os.path.join(os.getcwd(), "results")
EXPERIMENTS = {
    "FP": {
        "script": "cifar10_vitsptlsa_fp_scratch.py",
        "results_dir": "VITSPTLSA_FP_SCRATCH",
        "color": "black",
        "linestyle": "-",
        "label": "FP (Baseline)"
    },
    "TTv1": {
        "script": "cifar10_vitsptlsa_ttv1_scratch.py",
        "results_dir": "VITSPTLSA_TTV1_SCRATCH",
        "color": "blue",
        "linestyle": "--",
        "label": "TTv1"
    },
    "TTv2": {
        "script": "cifar10_vitsptlsa_ttv2_scratch.py",
        "results_dir": "VITSPTLSA_TTV2_SCRATCH",
        "color": "green",
        "linestyle": "-.",
        "label": "TTv2"
    },
    "LRTT": {
        "script": "cifar10_vitsptlsa_lrtt_scratch.py",
        "results_dir": "VITSPTLSA_LRTT_SCRATCH",
        "color": "red",
        "linestyle": "-",
        "label": "LRTT (Ours)"
    }
}


def run_experiment(name, config):
    """Run a single experiment."""
    script_path = os.path.join(os.path.dirname(__file__), config["script"])
    print(f"\n{'='*60}")
    print(f"Running {name} experiment: {config['script']}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(script_path)
    )

    if result.returncode != 0:
        print(f"Warning: {name} experiment finished with return code {result.returncode}")

    return result.returncode == 0


def load_history(name, config):
    """Load epoch history from JSON file."""
    history_path = os.path.join(RESULTS_BASE, config["results_dir"], "epoch_history.json")

    if not os.path.exists(history_path):
        print(f"Warning: History file not found for {name}: {history_path}")
        return None

    with open(history_path, 'r') as f:
        data = json.load(f)

    return data


def plot_results(results):
    """Plot combined results from all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Validation Accuracy
    ax1 = axes[0, 0]
    for name, data in results.items():
        if data is None:
            continue
        config = EXPERIMENTS[name]
        epochs = [h["epoch"] for h in data["history"]]
        val_acc = [h["val_accuracy"] for h in data["history"]]
        ax1.plot(epochs, val_acc,
                color=config["color"],
                linestyle=config["linestyle"],
                linewidth=2,
                label=f"{config['label']} (Best: {data['best_accuracy']:.2f}%)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax1.set_title("Validation Accuracy vs Epoch", fontsize=14)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 40)

    # Plot 2: Validation Error
    ax2 = axes[0, 1]
    for name, data in results.items():
        if data is None:
            continue
        config = EXPERIMENTS[name]
        epochs = [h["epoch"] for h in data["history"]]
        val_err = [h["val_error"] for h in data["history"]]
        ax2.plot(epochs, val_err,
                color=config["color"],
                linestyle=config["linestyle"],
                linewidth=2,
                label=f"{config['label']} (Best: {100-data['best_accuracy']:.2f}%)")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation Error (%)", fontsize=12)
    ax2.set_title("Validation Error vs Epoch", fontsize=14)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 40)

    # Plot 3: Training Loss
    ax3 = axes[1, 0]
    for name, data in results.items():
        if data is None:
            continue
        config = EXPERIMENTS[name]
        epochs = [h["epoch"] for h in data["history"]]
        train_loss = [h["train_loss"] for h in data["history"]]
        ax3.plot(epochs, train_loss,
                color=config["color"],
                linestyle=config["linestyle"],
                linewidth=2,
                label=config["label"])
    ax3.set_xlabel("Epoch", fontsize=12)
    ax3.set_ylabel("Training Loss", fontsize=12)
    ax3.set_title("Training Loss vs Epoch", fontsize=14)
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, 40)

    # Plot 4: Training Accuracy
    ax4 = axes[1, 1]
    for name, data in results.items():
        if data is None:
            continue
        config = EXPERIMENTS[name]
        epochs = [h["epoch"] for h in data["history"]]
        train_acc = [h["train_accuracy"] for h in data["history"]]
        ax4.plot(epochs, train_acc,
                color=config["color"],
                linestyle=config["linestyle"],
                linewidth=2,
                label=config["label"])
    ax4.set_xlabel("Epoch", fontsize=12)
    ax4.set_ylabel("Training Accuracy (%)", fontsize=12)
    ax4.set_title("Training Accuracy vs Epoch", fontsize=14)
    ax4.legend(loc="lower right", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1, 40)

    plt.suptitle("ViT-SPT-LSA CIFAR-10 Training Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(RESULTS_BASE, "vitsptlsa_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF
    pdf_path = os.path.join(RESULTS_BASE, "vitsptlsa_comparison.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.show()

    return output_path


def print_summary(results):
    """Print summary of all experiments."""
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    for name, data in results.items():
        if data is None:
            print(f"{name}: No results available")
            continue
        config = EXPERIMENTS[name]
        print(f"\n{config['label']}:")
        print(f"  Best Accuracy: {data['best_accuracy']:.2f}%")
        print(f"  Best Error: {100 - data['best_accuracy']:.2f}%")
        print(f"  Best Epoch: {data['best_epoch']}")

    print(f"\n{'='*60}")
    print("Paper Reference Results:")
    print("  FP Baseline: 29.3% error")
    print("  TTv2 (no noise): 36.1% error")
    print("  c-TTv2 (no noise): 35.9% error")
    print(f"{'='*60}")


def main():
    """Main function to run all experiments and plot results."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ViT-SPT-LSA experiments")
    parser.add_argument("--skip-run", action="store_true",
                       help="Skip running experiments, only plot existing results")
    parser.add_argument("--experiments", nargs="+",
                       choices=list(EXPERIMENTS.keys()),
                       default=list(EXPERIMENTS.keys()),
                       help="Which experiments to run")
    args = parser.parse_args()

    os.makedirs(RESULTS_BASE, exist_ok=True)

    # Run experiments
    if not args.skip_run:
        for name in args.experiments:
            config = EXPERIMENTS[name]
            success = run_experiment(name, config)
            if not success:
                print(f"Warning: {name} experiment may have failed")

    # Load results
    results = {}
    for name in EXPERIMENTS.keys():
        config = EXPERIMENTS[name]
        results[name] = load_history(name, config)

    # Check if we have any results
    if all(v is None for v in results.values()):
        print("Error: No results found. Please run experiments first.")
        return

    # Print summary
    print_summary(results)

    # Plot results
    plot_results(results)


if __name__ == "__main__":
    main()
