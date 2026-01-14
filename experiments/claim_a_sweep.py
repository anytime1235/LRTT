#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Claim A Comprehensive Sweep: Matrix Size, Rank, Learning Rate Analysis

This script runs systematic experiments to verify Claim A across:
- Matrix sizes: 4x4, 16x16, 64x64
- Ranks: 1, 2, 4, 8 (capped by matrix size)
- Learning rates: 0.001, 0.01, 0.05, 0.1
- G modes: lowrank, decay

Usage:
    python experiments/claim_a_sweep.py --quick    # Fast test run
    python experiments/claim_a_sweep.py --full     # Full sweep
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from itertools import product
import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from lrtt_claims_benchmark import (
    BenchmarkConfig, LRTTClaimsBenchmark,
    generate_target_G, truncated_svd, compute_svd_optimal_error
)


def run_single_experiment(config: dict) -> dict:
    """Run a single Claim A experiment and return metrics."""

    bench_config = BenchmarkConfig(
        d_size=config['size'],
        x_size=config['size'],
        rank=config['rank'],
        steps=config['steps'],
        lr_update=config['lr'],
        G_mode=config['g_mode'],
        true_rank=config.get('true_rank', config['rank']),
        decay_rate=config.get('decay_rate', 0.5),
        use_onehot=True,
        use_sigma_delta=False,  # Default: False
        reinit_mode='standard',
        seed=config.get('seed', 42),
        log_every=config.get('log_every', 50),
    )

    try:
        benchmark = LRTTClaimsBenchmark(bench_config)
        df = benchmark.run_claim_a_experiment()

        # Get final metrics
        final = df.iloc[-1].to_dict()

        # Get intermediate metrics for convergence analysis
        convergence_data = df[['step', 'recon_rel_err_true', 'svd_gap_ratio_true', 'cosine_sim_true']].to_dict('records')

        return {
            'config': config,
            'final_metrics': final,
            'convergence': convergence_data,
            'svd_opt_err': benchmark.svd_opt_err,
            'G_norm': torch.norm(benchmark.G).item(),
            'success': True
        }
    except Exception as e:
        return {
            'config': config,
            'error': str(e),
            'success': False
        }


def run_sweep(experiment_configs: list, output_dir: Path) -> pd.DataFrame:
    """Run all experiments and collect results."""

    results = []
    total = len(experiment_configs)

    print(f"\n{'='*70}")
    print(f"Starting Claim A Sweep: {total} experiments")
    print(f"{'='*70}\n")

    for i, config in enumerate(experiment_configs):
        print(f"[{i+1}/{total}] size={config['size']}x{config['size']}, "
              f"rank={config['rank']}, lr={config['lr']}, g_mode={config['g_mode']}")

        start_time = time.time()
        result = run_single_experiment(config)
        elapsed = time.time() - start_time

        if result['success']:
            final = result['final_metrics']
            print(f"  -> recon_err={final['recon_rel_err_true']:.4f}, "
                  f"cos_sim={final['cosine_sim_true']:.4f}, "
                  f"time={elapsed:.1f}s")

            # Flatten for DataFrame
            row = {
                'size': config['size'],
                'rank': config['rank'],
                'lr': config['lr'],
                'g_mode': config['g_mode'],
                'steps': config['steps'],
                'recon_rel_err': final['recon_rel_err_true'],
                'svd_gap_ratio': final['svd_gap_ratio_true'],
                'cosine_sim': final['cosine_sim_true'],
                'norm_AB': final['norm_AB_true'],
                'svd_opt_err': result['svd_opt_err'],
                'G_norm': result['G_norm'],
                'elapsed_sec': elapsed
            }
            results.append(row)

            # Save convergence data
            conv_file = output_dir / f"conv_s{config['size']}_r{config['rank']}_lr{config['lr']}_{config['g_mode']}.json"
            with open(conv_file, 'w') as f:
                json.dump(result['convergence'], f)
        else:
            print(f"  -> FAILED: {result['error']}")

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, output_dir: Path):
    """Generate analysis plots."""
  │ use_sigma_delta
    # Plot 1: Reconstruction error vs rank for each size
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, size in enumerate([4, 16, 64]):
        ax = axes[idx]
        subset = df[(df['size'] == size) & (df['g_mode'] == 'decay')]

        if len(subset) == 0:
            continue

        for lr in subset['lr'].unique():
            data = subset[subset['lr'] == lr].sort_values('rank')
            ax.plot(data['rank'], data['recon_rel_err'], 'o-', label=f'lr={lr}')

        ax.set_xlabel('Rank')
        ax.set_ylabel('Reconstruction Relative Error')
        ax.set_title(f'{size}x{size} Matrix (decay G)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / 'recon_err_vs_rank.png', dpi=150)
    plt.close()

    # Plot 2: Cosine similarity vs rank
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, size in enumerate([4, 16, 64]):
        ax = axes[idx]
        subset = df[(df['size'] == size) & (df['g_mode'] == 'decay')]

        if len(subset) == 0:
            continue

        for lr in subset['lr'].unique():
            data = subset[subset['lr'] == lr].sort_values('rank')
            ax.plot(data['rank'], data['cosine_sim'], 'o-', label=f'lr={lr}')

        ax.set_xlabel('Rank')
        ax.set_ylabel('Cosine Similarity (AB, -G)')
        ax.set_title(f'{size}x{size} Matrix (decay G)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'cosine_sim_vs_rank.png', dpi=150)
    plt.close()

    # Plot 3: SVD gap ratio vs rank
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, size in enumerate([4, 16, 64]):
        ax = axes[idx]
        subset = df[(df['size'] == size) & (df['g_mode'] == 'decay')]

        if len(subset) == 0:
            continue

        for lr in subset['lr'].unique():
            data = subset[subset['lr'] == lr].sort_values('rank')
            # Cap very large values for visualization
            gap_ratio = data['svd_gap_ratio'].clip(upper=20)
            ax.plot(data['rank'], gap_ratio, 'o-', label=f'lr={lr}')

        ax.set_xlabel('Rank')
        ax.set_ylabel('SVD Gap Ratio')
        ax.set_title(f'{size}x{size} Matrix (decay G)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Optimal')

    plt.tight_layout()
    plt.savefig(output_dir / 'svd_gap_vs_rank.png', dpi=150)
    plt.close()

    # Plot 4: Learning rate effect (fixed size=16, rank=4)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    subset = df[(df['size'] == 16) & (df['rank'] == 4)]
    if len(subset) > 0:
        for g_mode in ['decay', 'lowrank']:
            data = subset[subset['g_mode'] == g_mode].sort_values('lr')
            if len(data) > 0:
                axes[0].plot(data['lr'], data['recon_rel_err'], 'o-', label=f'{g_mode}')
                axes[1].plot(data['lr'], data['cosine_sim'], 'o-', label=f'{g_mode}')

        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Reconstruction Error')
        axes[0].set_title('LR Effect on Reconstruction (16x16, rank=4)')
        axes[0].set_xscale('log')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].set_title('LR Effect on Direction Alignment (16x16, rank=4)')
        axes[1].set_xscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'lr_effect.png', dpi=150)
    plt.close()

    # Plot 5: Size scaling analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Fixed rank ratio (rank = size/4)
    size_data = []
    for size in [4, 16, 64]:
        rank = max(1, size // 4)
        subset = df[(df['size'] == size) & (df['rank'] == rank) & (df['g_mode'] == 'decay') & (df['lr'] == 0.01)]
        if len(subset) > 0:
            size_data.append({
                'size': size,
                'rank': rank,
                'recon_err': subset['recon_rel_err'].values[0],
                'cosine_sim': subset['cosine_sim'].values[0]
            })

    if size_data:
        size_df = pd.DataFrame(size_data)
        axes[0].plot(size_df['size'], size_df['recon_err'], 'o-', markersize=10)
        axes[0].set_xlabel('Matrix Size')
        axes[0].set_ylabel('Reconstruction Error')
        axes[0].set_title('Size Scaling (rank=size/4, lr=0.01)')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(size_df['size'], size_df['cosine_sim'], 'o-', markersize=10)
        axes[1].set_xlabel('Matrix Size')
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].set_title('Size Scaling (rank=size/4, lr=0.01)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'size_scaling.png', dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir}")


def print_summary_table(df: pd.DataFrame):
    """Print formatted summary table."""

    print("\n" + "="*90)
    print("CLAIM A SWEEP RESULTS SUMMARY")
    print("="*90)

    # Group by size and g_mode
    for g_mode in df['g_mode'].unique():
        print(f"\n📊 G Mode: {g_mode}")
        print("-"*80)
        print(f"{'Size':>6} | {'Rank':>4} | {'LR':>6} | {'Recon Err':>10} | {'Cos Sim':>8} | {'SVD Gap':>10} | {'||AB||':>8}")
        print("-"*80)

        subset = df[df['g_mode'] == g_mode].sort_values(['size', 'rank', 'lr'])
        for _, row in subset.iterrows():
            svd_gap = row['svd_gap_ratio']
            if svd_gap > 1000:
                svd_gap_str = f"{svd_gap:.0e}"
            else:
                svd_gap_str = f"{svd_gap:.2f}"

            print(f"{row['size']:>6} | {row['rank']:>4} | {row['lr']:>6.3f} | "
                  f"{row['recon_rel_err']:>10.4f} | {row['cosine_sim']:>8.4f} | "
                  f"{svd_gap_str:>10} | {row['norm_AB']:>8.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test run')
    parser.add_argument('--full', action='store_true', help='Full sweep')
    parser.add_argument('--output_dir', type=str, default='results/claim_a_sweep')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define experiment grid
    if args.quick:
        # Quick test
        sizes = [4, 16]
        ranks_by_size = {4: [1, 2], 16: [2, 4]}
        lrs = [0.01, 0.05]
        g_modes = ['decay']
        steps = 200
    else:
        # Full sweep - adjusted LRs to avoid divergence
        sizes = [4, 16, 64]
        ranks_by_size = {4: [1, 2], 16: [1, 2, 4, 8], 64: [2, 4, 8, 16]}
        lrs = [0.001, 0.005, 0.01, 0.02]  # Lower LRs to avoid divergence
        g_modes = ['decay', 'lowrank']
        steps = 500

    # Generate experiment configs
    experiments = []
    for size in sizes:
        for rank in ranks_by_size.get(size, [1]):
            for lr in lrs:
                for g_mode in g_modes:
                    # For lowrank mode, true_rank = min(rank, size//2)
                    true_rank = min(rank, size // 2) if g_mode == 'lowrank' else rank

                    experiments.append({
                        'size': size,
                        'rank': rank,
                        'lr': lr,
                        'g_mode': g_mode,
                        'true_rank': true_rank,
                        'steps': steps,
                        'decay_rate': 0.5,
                        'seed': 42,
                        'log_every': max(steps // 10, 10)
                    })

    print(f"Output directory: {output_dir}")
    print(f"Total experiments: {len(experiments)}")

    # Run sweep
    df = run_sweep(experiments, output_dir)

    # Save results
    df.to_csv(output_dir / 'sweep_results.csv', index=False)
    print(f"\nResults saved to {output_dir / 'sweep_results.csv'}")

    # Print summary
    print_summary_table(df)

    # Generate plots
    plot_results(df, output_dir)

    # Save config
    config = {
        'sizes': sizes,
        'ranks_by_size': {str(k): v for k, v in ranks_by_size.items()},
        'lrs': lrs,
        'g_modes': g_modes,
        'steps': steps,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_dir / 'sweep_config.json', 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
