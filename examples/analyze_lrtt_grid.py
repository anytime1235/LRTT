# -*- coding: utf-8 -*-
"""
Analyze LRTT grid search results
Creates comprehensive visualizations and summary tables

Inputs: reports/lrtt_grid/all_epochs.csv, all_transfers.csv
Outputs: PDF plots and summary tables
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def load_data(datadir):
    """Load epoch and transfer data"""
    epoch_path = os.path.join(datadir, 'all_epochs.csv')
    transfer_path = os.path.join(datadir, 'all_transfers.csv')

    epochs_df = pd.read_csv(epoch_path)

    # Transfer data may not exist if no transfers occurred
    if os.path.exists(transfer_path):
        transfers_df = pd.read_csv(transfer_path)
    else:
        print(f"Warning: {transfer_path} not found. Transfer analysis will be skipped.")
        transfers_df = None

    return epochs_df, transfers_df


def plot_accuracy_vs_hyperparams(df, outdir):
    """Plot final test accuracy vs each hyperparameter"""

    # Get final epoch for each config
    final_df = df[df['epoch'] == df['epoch'].max()].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Rank
    ax = axes[0, 0]
    for lr in sorted(final_df['lr'].unique()):
        for transfer_lr in sorted(final_df['transfer_lr'].unique()):
            for transfer_every in sorted(final_df['transfer_every'].unique()):
                subset = final_df[(final_df['lr'] == lr) &
                                 (final_df['transfer_lr'] == transfer_lr) &
                                 (final_df['transfer_every'] == transfer_every)]
                if not subset.empty:
                    ax.plot(subset['rank'], subset['test_acc'], marker='o', alpha=0.5,
                           label=f'lr={lr}, tlr={transfer_lr}, te={transfer_every}')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy vs Rank')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
    ax.grid(True)

    # Learning Rate
    ax = axes[0, 1]
    grouped = final_df.groupby('lr')['test_acc'].agg(['mean', 'std', 'min', 'max'])
    lrs = grouped.index
    ax.errorbar(lrs, grouped['mean'], yerr=grouped['std'], marker='o', capsize=5)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy vs Learning Rate\n(mean ± std across all configs)')
    ax.set_xscale('log')
    ax.grid(True)

    # Transfer LR
    ax = axes[1, 0]
    grouped = final_df.groupby('transfer_lr')['test_acc'].agg(['mean', 'std', 'min', 'max'])
    tlrs = grouped.index
    ax.errorbar(tlrs, grouped['mean'], yerr=grouped['std'], marker='o', capsize=5)
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy vs Transfer LR\n(mean ± std across all configs)')
    ax.set_xscale('log')
    ax.grid(True)

    # Transfer Every
    ax = axes[1, 1]
    grouped = final_df.groupby('transfer_every')['test_acc'].agg(['mean', 'std', 'min', 'max'])
    tevery = grouped.index
    ax.errorbar(tevery, grouped['mean'], yerr=grouped['std'], marker='o', capsize=5)
    ax.set_xlabel('Transfer Every (steps)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy vs Transfer Frequency\n(mean ± std across all configs)')
    ax.set_xscale('log')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'accuracy_vs_hyperparams.pdf'))
    plt.close()


def plot_learning_curves(df, outdir, max_configs=20):
    """Plot learning curves for top configurations"""

    # Get final accuracy for each config
    final_df = df[df['epoch'] == df['epoch'].max()].copy()
    final_df['config_id'] = (final_df['rank'].astype(str) + '_' +
                             final_df['lr'].astype(str) + '_' +
                             final_df['transfer_lr'].astype(str) + '_' +
                             final_df['transfer_every'].astype(str))

    # Sort by final accuracy
    top_configs = final_df.nlargest(min(max_configs, len(final_df)), 'test_acc')['config_id'].values

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    for config_id in top_configs[:10]:
        parts = config_id.split('_')
        rank, lr, transfer_lr, transfer_every = int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])

        subset = df[(df['rank'] == rank) &
                   (df['lr'] == lr) &
                   (df['transfer_lr'] == transfer_lr) &
                   (df['transfer_every'] == transfer_every)]

        label = f'r={rank}, lr={lr}, tlr={transfer_lr}, te={transfer_every}'
        axes[0].plot(subset['epoch'], subset['test_acc'], marker='o', label=label, alpha=0.7)
        axes[1].plot(subset['epoch'], subset['test_loss'], marker='o', label=label, alpha=0.7)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Learning Curves: Top 10 Configurations (by final accuracy)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    axes[0].grid(True)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Loss')
    axes[1].set_title('Loss Curves: Top 10 Configurations')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'learning_curves_top10.pdf'))
    plt.close()


def plot_transfer_errors(df, outdir):
    """Plot transfer errors vs hyperparameters"""

    # Average across all transfers for each config
    grouped = df.groupby(['rank', 'lr', 'transfer_lr', 'transfer_every']).agg({
        'err_vs_true': 'mean',
        'err_vs_onehot': 'mean',
        'amp_vs_true': 'mean',
        'amp_vs_onehot': 'mean',
        'corr_true': 'mean',
        'corr_onehot': 'mean',
    }).reset_index()

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # Error vs True
    ax = axes[0, 0]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['err_vs_true'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('Relative Error vs True')
    ax.set_title('Transfer Error vs True (mean across transfers)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    # Error vs One-hot
    ax = axes[0, 1]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['err_vs_onehot'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('Relative Error vs One-hot')
    ax.set_title('Transfer Error vs One-hot (mean across transfers)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    # Amplification vs True
    ax = axes[1, 0]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['amp_vs_true'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.axhline(y=1.0, color='red', linestyle='--', label='ideal (1.0)')
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('Amplification vs True')
    ax.set_title('Magnitude Amplification vs True')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    # Amplification vs One-hot
    ax = axes[1, 1]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['amp_vs_onehot'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.axhline(y=1.0, color='red', linestyle='--', label='ideal (1.0)')
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('Amplification vs One-hot')
    ax.set_title('Magnitude Amplification vs One-hot')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    # Correlation vs True
    ax = axes[2, 0]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['corr_true'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.axhline(y=1.0, color='red', linestyle='--', label='ideal (1.0)')
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('Correlation with True')
    ax.set_title('Direction Correlation vs True')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    # Correlation vs One-hot
    ax = axes[2, 1]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['corr_onehot'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.axhline(y=1.0, color='red', linestyle='--', label='ideal (1.0)')
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('Correlation with One-hot')
    ax.set_title('Direction Correlation vs One-hot')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'transfer_errors.pdf'))
    plt.close()


def plot_read_write_errors(df, outdir):
    """Plot read-side and write-side errors"""

    # Average across all transfers
    grouped = df.groupby(['rank', 'lr', 'transfer_lr', 'transfer_every']).agg({
        'A_fro_err': 'mean',
        'B_fro_err': 'mean',
        'A_snr': 'mean',
        'B_snr': 'mean',
        'alpha_A': 'mean',
        'alpha_B': 'mean',
        'write_cv': 'mean',
    }).reset_index()

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    # Read error (A)
    ax = axes[0, 0]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['A_fro_err'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('A Read Error (rel. Frobenius)')
    ax.set_title('One-hot Read Error (A tile)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    # Read error (B)
    ax = axes[0, 1]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['B_fro_err'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('B Read Error (rel. Frobenius)')
    ax.set_title('One-hot Read Error (B tile)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    # SNR (A)
    ax = axes[1, 0]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['A_snr'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('A SNR')
    ax.set_title('One-hot Read SNR (A tile)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    # SNR (B)
    ax = axes[1, 1]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['B_snr'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('B SNR')
    ax.set_title('One-hot Read SNR (B tile)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    # Scale distortion (A)
    ax = axes[2, 0]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['alpha_A'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.axhline(y=1.0, color='red', linestyle='--', label='ideal (1.0)')
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('α (scale distortion)')
    ax.set_title('Read Scale Distortion (A tile)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    # Write CV
    ax = axes[2, 1]
    for rank in sorted(grouped['rank'].unique()):
        subset = grouped[grouped['rank'] == rank]
        ax.scatter(subset['transfer_lr'], subset['write_cv'],
                  label=f'rank={rank}', alpha=0.6, s=50)
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('CV (coefficient of variation)')
    ax.set_title('Write-side Noise (C tile)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'read_write_errors.pdf'))
    plt.close()


def plot_heatmaps(df, outdir):
    """Create heatmaps showing accuracy vs hyperparameter pairs"""

    final_df = df[df['epoch'] == df['epoch'].max()].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Rank vs LR
    ax = axes[0, 0]
    pivot = final_df.groupby(['rank', 'lr'])['test_acc'].mean().reset_index().pivot(
        index='lr', columns='rank', values='test_acc')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', ax=ax, cbar_kws={'label': 'Test Acc (%)'})
    ax.set_title('Test Accuracy: Rank vs Learning Rate')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Learning Rate')

    # Rank vs Transfer LR
    ax = axes[0, 1]
    pivot = final_df.groupby(['rank', 'transfer_lr'])['test_acc'].mean().reset_index().pivot(
        index='transfer_lr', columns='rank', values='test_acc')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', ax=ax, cbar_kws={'label': 'Test Acc (%)'})
    ax.set_title('Test Accuracy: Rank vs Transfer LR')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Transfer LR')

    # Transfer LR vs Transfer Every
    ax = axes[1, 0]
    pivot = final_df.groupby(['transfer_lr', 'transfer_every'])['test_acc'].mean().reset_index().pivot(
        index='transfer_every', columns='transfer_lr', values='test_acc')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', ax=ax, cbar_kws={'label': 'Test Acc (%)'})
    ax.set_title('Test Accuracy: Transfer LR vs Transfer Every')
    ax.set_xlabel('Transfer LR')
    ax.set_ylabel('Transfer Every')

    # LR vs Transfer Every
    ax = axes[1, 1]
    pivot = final_df.groupby(['lr', 'transfer_every'])['test_acc'].mean().reset_index().pivot(
        index='transfer_every', columns='lr', values='test_acc')
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', ax=ax, cbar_kws={'label': 'Test Acc (%)'})
    ax.set_title('Test Accuracy: LR vs Transfer Every')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Transfer Every')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'heatmaps_accuracy.pdf'))
    plt.close()


def create_summary_table(epoch_df, transfer_df, outdir):
    """Create summary table with best configurations"""

    # Get final epoch results
    final_df = epoch_df[epoch_df['epoch'] == epoch_df['epoch'].max()].copy()

    # Merge with average transfer stats
    transfer_grouped = transfer_df.groupby(['rank', 'lr', 'transfer_lr', 'transfer_every']).agg({
        'err_vs_true': 'mean',
        'err_vs_onehot': 'mean',
        'amp_vs_true': 'mean',
        'A_fro_err': 'mean',
        'write_cv': 'mean',
    }).reset_index()

    merged = final_df.merge(transfer_grouped,
                           on=['rank', 'lr', 'transfer_lr', 'transfer_every'],
                           how='left')

    # Sort by test accuracy
    merged_sorted = merged.sort_values('test_acc', ascending=False)

    # Save top 20
    top20 = merged_sorted.head(20)
    columns_to_save = ['rank', 'lr', 'transfer_lr', 'transfer_every', 'test_acc', 'test_loss',
                       'err_vs_true', 'err_vs_onehot', 'amp_vs_true', 'A_fro_err', 'write_cv']

    top20[columns_to_save].to_csv(os.path.join(outdir, 'summary_top20.csv'), index=False, float_format='%.4f')

    # Print to console
    print("\n" + "="*80)
    print("TOP 20 CONFIGURATIONS BY TEST ACCURACY")
    print("="*80 + "\n")
    print(top20[columns_to_save].to_string(index=False))
    print("\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze LRTT grid search results')
    parser.add_argument('--datadir', type=str, default='reports/lrtt_grid',
                       help='Directory containing all_epochs.csv and all_transfers.csv')
    parser.add_argument('--outdir', type=str, default=None,
                       help='Output directory for plots (default: same as datadir)')
    args = parser.parse_args()

    outdir = args.outdir if args.outdir else args.datadir
    os.makedirs(outdir, exist_ok=True)

    print(f"Loading data from {args.datadir}...")
    epochs_df, transfers_df = load_data(args.datadir)

    transfer_count = len(transfers_df) if transfers_df is not None else 0
    print(f"Loaded {len(epochs_df)} epoch records and {transfer_count} transfer records")
    print(f"Generating plots...")

    # Generate all plots
    plot_accuracy_vs_hyperparams(epochs_df, outdir)
    print("  ✓ accuracy_vs_hyperparams.pdf")

    plot_learning_curves(epochs_df, outdir)
    print("  ✓ learning_curves_top10.pdf")

    if transfers_df is not None and len(transfers_df) > 0:
        plot_transfer_errors(transfers_df, outdir)
        print("  ✓ transfer_errors.pdf")

        plot_read_write_errors(transfers_df, outdir)
        print("  ✓ read_write_errors.pdf")

        create_summary_table(epochs_df, transfers_df, outdir)
        print("  ✓ summary_top20.csv")
    else:
        print("  [Skipped] transfer_errors.pdf (no transfer data)")
        print("  [Skipped] read_write_errors.pdf (no transfer data)")
        print("  [Skipped] summary_top20.csv (no transfer data)")

    plot_heatmaps(epochs_df, outdir)
    print("  ✓ heatmaps_accuracy.pdf")

    print(f"\nAll plots saved to: {outdir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
