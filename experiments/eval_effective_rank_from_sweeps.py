#!/usr/bin/env python3
"""Effective Rank Evaluation from LRTT Sweeps - Streaming Design.

Measures accumulating effective rank at best hyperparameter conditions for each LoRA rank,
comparing two reinit modes:
- softbound_lifetime: reinit_mode="decay" - Both A and B decay after transfer
- hardreset (hybrid): reinit_mode="hybrid" - A=0 hard reset, B unchanged (decay_factor=1.0)

Key features (streaming design):
- Weights are NOT stored in memory
- Metrics computed immediately at each checkpoint
- Results written to CSV in real-time (crash-safe)
- Plots generated from CSV after training

Usage:
    python eval_effective_rank_from_sweeps.py \\
        --config-json /data/results/mnist/lrtt_sweep_best_configs.json \\
        --output-dir ./outputs/erank_eval \\
        --checkpoint-every 100 \\
        --ranks 1 4 8 16 32 64 \\
        --epochs 30
"""

import os
os.environ["LRTT_SILENT"] = "1"

import argparse
import json
import math
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import StepLR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from lrtt_training_utils import (
    load_data, create_model, train_epoch, validate,
    get_lrtt_tiles, get_lrtt_component_weights,
    DEVICE, BATCH_SIZE
)
from aihwkit.optim import AnalogSGD


# =============================================================================
# Configuration Loading
# =============================================================================

def load_best_configs(json_path: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Load best configs for decay and hybrid sweeps from JSON file.

    Args:
        json_path: Path to lrtt_sweep_best_configs.json

    Returns:
        Tuple of (decay_configs, hybrid_configs)
        Each is {rank: {te, lifetime, lr, tlr, best_acc}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    decay_configs = {}
    hybrid_configs = {}

    # Parse decay_sweep results
    if 'decay_sweep' in data and 'best_per_rank' in data['decay_sweep']:
        for rank_str, cfg in data['decay_sweep']['best_per_rank'].items():
            rank = int(rank_str)
            decay_configs[rank] = {
                'te': cfg['te'],
                'lifetime': cfg['lifetime'],
                'lr': cfg['lr'],
                'tlr': cfg['tlr'],
                'best_acc': cfg.get('best_acc', 0.0)
            }

    # Parse hybrid_sweep results
    if 'hybrid_sweep' in data and 'best_per_rank' in data['hybrid_sweep']:
        for rank_str, cfg in data['hybrid_sweep']['best_per_rank'].items():
            rank = int(rank_str)
            hybrid_configs[rank] = {
                'te': cfg['te'],
                'lifetime': cfg['lifetime'],
                'lr': cfg['lr'],
                'tlr': cfg['tlr'],
                'best_acc': cfg.get('best_acc', 0.0)
            }

    return decay_configs, hybrid_configs


def load_hardreset_configs_legacy(json_path: str) -> Dict[int, Dict[str, Any]]:
    """Load best configs for hardreset (hybrid) mode from legacy JSON results.

    This is kept for backward compatibility with old results_partial.json format.

    Args:
        json_path: Path to results JSON file

    Returns:
        Dict mapping rank to best config
    """
    with open(json_path, 'r') as f:
        results = json.load(f)

    # Group by rank and find best by full_best_acc
    best_by_rank = {}
    for r in results:
        rank = r['rank']
        if rank not in best_by_rank or r['full_best_acc'] > best_by_rank[rank]['full_best_acc']:
            best_by_rank[rank] = r

    return best_by_rank


# =============================================================================
# Effective Rank Computation
# =============================================================================

def compute_erank(X: Tensor, eps: float = 1e-12) -> float:
    """Compute entropy-based effective rank (Roy & Vetterli, 2007).

    The effective rank is defined as:
        erank(X) = exp(H(p))

    where H(p) is the entropy of the normalized singular values:
        p_i = σ_i / Σσ_j
        H(p) = -Σ p_i log(p_i)

    For a matrix X of shape [n, m], we compute SVD and use the singular values.

    Args:
        X: Input matrix tensor
        eps: Small value for numerical stability

    Returns:
        Effective rank (float)
    """
    if X.numel() == 0:
        return 0.0

    # Handle 1D tensors
    if X.dim() == 1:
        X = X.unsqueeze(0)

    # Compute SVD
    try:
        # Use SVD to get singular values
        _, S, _ = torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        # Fallback for numerical issues
        return 0.0

    # Filter out very small singular values
    S = S[S > 1e-20]
    if len(S) == 0:
        return 0.0

    # Normalize to get probability distribution
    S_sum = S.sum()
    if S_sum < eps:
        return 0.0

    p = S / S_sum

    # Compute entropy
    log_p = torch.log(p + eps)
    H = -torch.sum(p * log_p)

    # Effective rank
    erank = torch.exp(H)
    return float(erank.item())


def compute_erank_via_gram(X: Tensor, eps: float = 1e-12) -> float:
    """Compute effective rank via Gram matrix eigenvalues.

    More efficient for tall/wide matrices:
        G = X @ X.T (if n < m) or X.T @ X (if m < n)
        λ_i = eigenvalues of G
        erank = exp(-Σ p_i log(p_i)) where p_i = λ_i / Σλ

    Args:
        X: Input matrix tensor
        eps: Small value for numerical stability

    Returns:
        Effective rank (float)
    """
    if X.numel() == 0:
        return 0.0

    if X.dim() == 1:
        X = X.unsqueeze(0)

    n, m = X.shape

    # Choose smaller dimension for Gram matrix
    if n < m:
        G = X @ X.T
    else:
        G = X.T @ X

    # Get eigenvalues (symmetric matrix)
    try:
        eigenvalues = torch.linalg.eigvalsh(G)
    except RuntimeError:
        return 0.0

    # Filter positive eigenvalues and take sqrt to get singular values
    eigenvalues = eigenvalues[eigenvalues > 1e-20]
    if len(eigenvalues) == 0:
        return 0.0

    S = torch.sqrt(eigenvalues)

    # Normalize singular values (same as SVD-based erank)
    S_sum = S.sum()
    if S_sum < eps:
        return 0.0

    p = S / S_sum

    # Entropy and effective rank
    H = -torch.sum(p * torch.log(p + eps))
    return float(torch.exp(H).item())


# =============================================================================
# Streaming Effective Rank Tracker
# =============================================================================

class StreamingErankTracker:
    """Streaming effective rank tracker.

    Key features:
    - Does NOT store weights in memory (only base_C and prev_C per layer)
    - Computes metrics immediately at each checkpoint
    - Writes results to CSV in real-time (crash-safe)
    - Maintains minimal cumulative state for running sums

    Metrics recorded per checkpoint:
    - erank_update: erank of cumulative ΔW = C_t - C_0
    - erank_dW: erank of this checkpoint's ΔW = C_t - C_{t-1}
    - sum_erank: running sum Σ erank(ΔW_k)
    - num_transfers: current transfer count from controller
    """

    def __init__(
        self,
        csv_path: str,
        checkpoint_every: int = 100,
        mode: str = "softbound_lifetime",
        rank: int = 4
    ):
        """Initialize streaming tracker.

        Args:
            csv_path: Path to output CSV file (created/overwritten)
            checkpoint_every: Checkpoint interval in steps
            mode: Training mode ("softbound_lifetime" or "hardreset")
            rank: LoRA rank
        """
        self.csv_path = csv_path
        self.checkpoint_every = checkpoint_every
        self.mode = mode
        self.rank = rank

        # Minimal state per layer (NOT storing full weight history)
        self.base_C: Dict[str, Tensor] = {}  # Initial C weights
        self.prev_C: Dict[str, Tensor] = {}  # Previous checkpoint's C
        self.sum_erank: Dict[str, float] = {}  # Running Σ erank(ΔW_k)

        self.checkpoint_idx = 0
        self.step_count = 0
        self.num_checkpoints = 0

        # Initialize CSV with header
        self._init_csv()

    def _init_csv(self) -> None:
        """Initialize CSV file with header."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'mode', 'rank', 'layer', 'checkpoint', 'step',
                'erank_update', 'erank_dW', 'sum_erank', 'num_transfers'
            ])

    def set_base_weights(self, model: nn.Module) -> None:
        """Record initial weights (called once before training).

        Args:
            model: LRTT model
        """
        tiles = get_lrtt_tiles(model)
        for name, tile in tiles.items():
            C, A, B = get_lrtt_component_weights(tile)
            if C is not None:
                self.base_C[name] = C.detach().cpu().clone()
                self.prev_C[name] = self.base_C[name].clone()
                self.sum_erank[name] = 0.0

    def on_step(self, step: int, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Called after each training step.

        At checkpoint intervals, immediately computes metrics and writes to CSV.

        Args:
            step: Step number within epoch (not used, we track global step)
            model: Current model
            optimizer: Current optimizer (not used)
        """
        self.step_count += 1

        # Only checkpoint at regular intervals
        if self.step_count % self.checkpoint_every != 0:
            return

        # Get current weights and compute metrics immediately
        tiles = get_lrtt_tiles(model)

        for name, tile in tiles.items():
            if name not in self.base_C:
                continue

            C, A, B = get_lrtt_component_weights(tile)
            if C is None:
                continue

            C = C.detach().cpu()

            # 1. Cumulative ΔW = C_t - C_0
            cumulative_dW = C - self.base_C[name]
            erank_update = compute_erank(cumulative_dW)

            # 2. This checkpoint's ΔW = C_t - C_{t-1}
            dW = C - self.prev_C[name]
            erank_dW = compute_erank(dW)
            self.sum_erank[name] += erank_dW

            # 3. Get num_transfers from controller
            num_transfers = 0
            if hasattr(tile, 'controller'):
                num_transfers = tile.controller.num_transfers

            # 4. Write to CSV immediately (crash-safe)
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.mode, self.rank, name, self.checkpoint_idx, self.step_count,
                    f'{erank_update:.6f}', f'{erank_dW:.6f}',
                    f'{self.sum_erank[name]:.6f}', num_transfers
                ])

            # 5. Update prev_C for next checkpoint (minimal memory)
            self.prev_C[name] = C.clone()

        self.checkpoint_idx += 1
        self.num_checkpoints += 1


# =============================================================================
# Training with Streaming Checkpoints
# =============================================================================

def run_evaluation_with_streaming(
    config: Dict[str, Any],
    mode: str,
    tracker: StreamingErankTracker,
    epochs: int = 30,
    device: torch.device = DEVICE,
    target_acc: Optional[float] = None
) -> Dict[str, Any]:
    """Run training with streaming checkpoint hooks.

    Args:
        config: Config dict with rank, te, lifetime, lr, tlr
        mode: "softbound_lifetime" or "hardreset"
        tracker: StreamingErankTracker instance
        epochs: Number of training epochs
        device: Target device
        target_acc: Optional target accuracy for early stopping

    Returns:
        Dict with training results
    """
    rank = config['rank']
    te = config['te']
    lifetime = config['lifetime']
    lr = config['lr']
    tlr = config['tlr']

    # Set reinit mode based on training mode
    if mode == "softbound_lifetime":
        reinit_mode = "decay"
        decay_factor = 1.0
    else:  # hardreset
        reinit_mode = "hybrid"
        decay_factor = 1.0

    # Create model
    model = create_model(
        rank=rank,
        te=te,
        lifetime=lifetime,
        lr=lr,
        tlr=tlr,
        reinit_mode=reinit_mode,
        decay_factor=decay_factor,
        device=device
    )

    # Setup training
    train_loader, val_loader = load_data()
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    # Record base weights (only stores base_C and prev_C per layer)
    tracker.set_base_weights(model)

    # Training loop with streaming callbacks
    best_acc = 0.0
    acc_history = []

    def step_callback(step: int, model: nn.Module, opt: torch.optim.Optimizer):
        tracker.on_step(step, model, opt)

    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, criterion, device, step_callback=step_callback)
        val_acc = validate(model, val_loader, device)
        scheduler.step()
        best_acc = max(best_acc, val_acc)
        acc_history.append(val_acc)
        print(f"  Epoch {epoch}/{epochs}: val_acc={val_acc:.2f}% (best={best_acc:.2f}%) [checkpoints: {tracker.num_checkpoints}]")

        # Early stopping if target accuracy reached
        if target_acc is not None and best_acc >= target_acc:
            print(f"  Target accuracy {target_acc}% reached, stopping early.")
            break

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {
        'best_acc': best_acc,
        'final_acc': acc_history[-1] if acc_history else 0.0,
        'acc_history': acc_history,
        'num_checkpoints': tracker.num_checkpoints
    }


# =============================================================================
# Plotting from CSV (Streaming-compatible)
# =============================================================================

def plot_from_csv(csv_path: str, output_dir: str) -> None:
    """Generate all plots from the streaming CSV file.

    Args:
        csv_path: Path to the CSV file with streaming results
        output_dir: Output directory for plots
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV file is empty")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Get unique modes and ranks
    modes = df['mode'].unique()
    ranks = sorted(df['rank'].unique())

    # 1. Plot erank_update vs checkpoint for each mode/rank
    for mode in modes:
        for rank in ranks:
            subset = df[(df['mode'] == mode) & (df['rank'] == rank)]
            if subset.empty:
                continue

            rank_dir = os.path.join(output_dir, mode, f'rank_{rank}')
            os.makedirs(rank_dir, exist_ok=True)

            _plot_erank_vs_checkpoint_from_df(subset, mode, rank, rank_dir)
            _plot_sum_erank_from_df(subset, mode, rank, rank_dir)

    # 2. Plot mode comparison
    _plot_mode_comparison_from_df(df, output_dir)

    # 3. Plot erank evolution over time (all ranks)
    _plot_erank_evolution_from_df(df, output_dir)


def _plot_erank_vs_checkpoint_from_df(df: pd.DataFrame, mode: str, rank: int, output_dir: str) -> None:
    """Plot erank_update vs checkpoint for a specific mode/rank."""
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = df['layer'].unique()
    for layer in layers:
        layer_data = df[df['layer'] == layer].sort_values('checkpoint')
        ax.plot(layer_data['checkpoint'], layer_data['erank_update'],
                label=layer, marker='o', markersize=2, linewidth=1)

    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('Effective Rank (cumulative ΔW)')
    ax.set_title(f'{mode} | rank={rank} | Effective Rank vs Checkpoint')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'erank_vs_checkpoint.png'), dpi=150)
    plt.close()


def _plot_sum_erank_from_df(df: pd.DataFrame, mode: str, rank: int, output_dir: str) -> None:
    """Plot sum_erank and erank_dW vs checkpoint."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = df['layer'].unique()

    # Left: erank_dW per checkpoint
    ax = axes[0]
    for layer in layers:
        layer_data = df[df['layer'] == layer].sort_values('checkpoint')
        ax.plot(layer_data['checkpoint'], layer_data['erank_dW'],
                label=layer, marker='o', markersize=2, linewidth=1)
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('erank(ΔW) per checkpoint')
    ax.set_title(f'{mode} | rank={rank} | Per-checkpoint erank')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Cumulative sum_erank
    ax = axes[1]
    for layer in layers:
        layer_data = df[df['layer'] == layer].sort_values('checkpoint')
        ax.plot(layer_data['checkpoint'], layer_data['sum_erank'],
                label=layer, marker='o', markersize=2, linewidth=1)
    ax.set_xlabel('Checkpoint')
    ax.set_ylabel('Σ erank(ΔW_k)')
    ax.set_title(f'{mode} | rank={rank} | Cumulative sum of erank')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sum_erank.png'), dpi=150)
    plt.close()


def _plot_mode_comparison_from_df(df: pd.DataFrame, output_dir: str) -> None:
    """Plot comparison between modes across ranks."""
    modes = df['mode'].unique()
    ranks = sorted(df['rank'].unique())

    if len(modes) < 2:
        return

    # Get final erank_update for each mode/rank (averaged across layers)
    final_erank = {}
    for mode in modes:
        final_erank[mode] = {}
        for rank in ranks:
            subset = df[(df['mode'] == mode) & (df['rank'] == rank)]
            if subset.empty:
                continue
            # Get last checkpoint for each layer and average
            last_values = subset.groupby('layer')['erank_update'].last()
            final_erank[mode][rank] = last_values.mean()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = list(range(len(ranks)))
    width = 0.35
    colors = ['#1f77b4', '#ff7f0e']

    for i, mode in enumerate(modes):
        values = [final_erank[mode].get(r, 0) for r in ranks]
        offset = (i - 0.5) * width
        ax.bar([xi + offset for xi in x], values, width, label=mode, color=colors[i % len(colors)])

    ax.set_xlabel('LoRA Rank')
    ax.set_ylabel('Final Effective Rank (averaged)')
    ax.set_title('Mode Comparison: Final Effective Rank vs LoRA Rank')
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mode_comparison.png'), dpi=150)
    plt.close()


def _plot_erank_evolution_from_df(df: pd.DataFrame, output_dir: str) -> None:
    """Plot erank evolution over training for all modes and ranks."""
    modes = df['mode'].unique()
    ranks = sorted(df['rank'].unique())

    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 6), squeeze=False)

    for idx, mode in enumerate(modes):
        ax = axes[0, idx]
        mode_data = df[df['mode'] == mode]

        for rank in ranks:
            rank_data = mode_data[mode_data['rank'] == rank]
            if rank_data.empty:
                continue
            # Average across layers
            avg_data = rank_data.groupby('checkpoint')['erank_update'].mean()
            ax.plot(avg_data.index, avg_data.values, label=f'rank={rank}', linewidth=1.5)

        ax.set_xlabel('Checkpoint')
        ax.set_ylabel('Effective Rank (avg across layers)')
        ax.set_title(f'{mode}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Effective Rank Evolution by Mode and Rank')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'erank_evolution.png'), dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate effective rank from LRTT sweeps (streaming)')
    parser.add_argument('--config-json', type=str,
                        default='/data/results/mnist/lrtt_sweep_best_configs.json',
                        help='Path to best configs JSON file')
    parser.add_argument('--output-dir', type=str, default='./outputs/erank_eval',
                        help='Output directory')
    parser.add_argument('--checkpoint-every', type=int, default=100,
                        help='Checkpoint every N steps')
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 4, 8, 16, 32, 64],
                        help='Ranks to evaluate')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs')
    parser.add_argument('--target-acc', type=float, default=None,
                        help='Target accuracy for early stopping (optional)')
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['softbound_lifetime', 'hardreset'],
                        help='Modes to evaluate')
    parser.add_argument('--te-override', type=int, default=None,
                        help='Override TE value for all modes (for fair comparison)')

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Single CSV file for all streaming results
    csv_path = output_dir / 'effective_rank_results.csv'

    # Load configs from JSON
    decay_configs = {}
    hybrid_configs = {}

    if os.path.exists(args.config_json):
        decay_configs, hybrid_configs = load_best_configs(args.config_json)
        print(f"Loaded configs from: {args.config_json}")
        print(f"  Decay (softbound) configs for ranks: {sorted(decay_configs.keys())}")
        print(f"  Hybrid (hardreset) configs for ranks: {sorted(hybrid_configs.keys())}")
    else:
        print(f"WARNING: Config file not found: {args.config_json}")
        print("Please provide a valid config JSON file.")
        return

    # Initialize combined CSV with header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'mode', 'rank', 'layer', 'checkpoint', 'step',
            'erank_update', 'erank_dW', 'sum_erank', 'num_transfers'
        ])

    all_results = []

    for rank in args.ranks:
        for mode in args.modes:
            print(f"\n{'='*60}")
            print(f"Evaluating: mode={mode}, rank={rank}")
            print(f"{'='*60}")

            # Get config for this rank/mode
            try:
                if mode == "softbound_lifetime":
                    if rank not in decay_configs:
                        print(f"  No decay config for rank={rank}, skipping")
                        continue
                    cfg = decay_configs[rank]
                    config = {
                        'rank': rank,
                        'te': cfg['te'],
                        'lifetime': cfg['lifetime'],
                        'lr': cfg['lr'],
                        'tlr': cfg['tlr']
                    }
                else:  # hardreset
                    if rank not in hybrid_configs:
                        print(f"  No hybrid config for rank={rank}, skipping")
                        continue
                    cfg = hybrid_configs[rank]
                    config = {
                        'rank': rank,
                        'te': cfg['te'],
                        'lifetime': cfg['lifetime'],
                        'lr': cfg['lr'],
                        'tlr': cfg['tlr']
                    }
            except (KeyError, ValueError) as e:
                print(f"  Error: {e}, skipping")
                continue

            # Apply TE override if specified
            if args.te_override is not None:
                config['te'] = args.te_override

            print(f"  Config: lr={config['lr']:.6f}, tlr={config['tlr']:.6f}, te={config['te']}, lifetime={config['lifetime']}")

            # Create streaming tracker (appends to combined CSV)
            # Use a temporary CSV for this run, then append to main
            temp_csv = output_dir / f'temp_{mode}_rank{rank}.csv'
            tracker = StreamingErankTracker(
                csv_path=str(temp_csv),
                checkpoint_every=args.checkpoint_every,
                mode=mode,
                rank=rank
            )

            # Run evaluation with streaming
            train_result = run_evaluation_with_streaming(
                config=config,
                mode=mode,
                tracker=tracker,
                epochs=args.epochs,
                device=DEVICE,
                target_acc=args.target_acc
            )

            # Append temp CSV to main CSV (skip header)
            with open(temp_csv, 'r') as temp_f:
                lines = temp_f.readlines()[1:]  # Skip header
            with open(csv_path, 'a') as main_f:
                main_f.writelines(lines)

            # Remove temp file
            os.remove(temp_csv)

            result = {
                'mode': mode,
                'rank': rank,
                'config': config,
                'best_acc': train_result['best_acc'],
                'num_checkpoints': train_result['num_checkpoints']
            }
            all_results.append(result)

            print(f"  Completed: {train_result['num_checkpoints']} checkpoints, "
                  f"best_acc={train_result['best_acc']:.2f}%")

    print(f"\nStreaming CSV saved: {csv_path}")

    # Generate plots from CSV
    print("\nGenerating plots from CSV...")
    plot_from_csv(str(csv_path), str(plots_dir))

    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'results': all_results
    }

    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Output directory: {output_dir}")
    print(f"CSV results: {csv_path}")
    print(f"Plots: {plots_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
