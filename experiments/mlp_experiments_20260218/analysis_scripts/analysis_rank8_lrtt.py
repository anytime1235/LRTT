#!/usr/bin/env python3
"""Analyze rank=8 lr/tlr relationship and generate 16 transfer_every configs."""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load best configs
with open('/root/LRTT/experiments/mnist_sweep_analysis/lrtt_sweep_best_configs.json', 'r') as f:
    configs = json.load(f)

# Extract rank=8 data
print("=" * 70)
print("RANK=8 ANALYSIS")
print("=" * 70)

# Hybrid rank=8
hybrid_8 = configs['hybrid_sweep']['by_rank_te']['8']
print("\nHYBRID (rank=8):")
print("TE  | LR      | TLR     | Acc")
print("-" * 40)
hybrid_tes = []
hybrid_lrs = []
hybrid_tlrs = []
hybrid_accs = []

for te_str, data in sorted(hybrid_8.items(), key=lambda x: int(x[0])):
    te = int(te_str)
    if isinstance(data, dict):
        lr = data.get('lr', 0)
        tlr = data.get('tlr', 0)
        acc = data.get('best_acc', 0)
    else:
        acc = data
        lr = tlr = 0

    print(f"{te:4d}| {lr:.4f} | {tlr:.4f} | {acc:.2f}%")
    if lr > 0:
        hybrid_tes.append(te)
        hybrid_lrs.append(lr)
        hybrid_tlrs.append(tlr)
        hybrid_accs.append(acc)

# Decay rank=8
decay_8 = configs['decay_sweep']['by_rank_te']['8']
print("\nDECAY (rank=8):")
print("TE  | LR      | TLR     | Acc")
print("-" * 40)
decay_tes = []
decay_lrs = []
decay_tlrs = []
decay_accs = []

for te_str, data in sorted(decay_8.items(), key=lambda x: int(x[0])):
    te = int(te_str)
    if isinstance(data, dict):
        lr = data.get('lr', 0)
        tlr = data.get('tlr', 0)
        acc = data.get('best_acc', 0)
    else:
        acc = data
        # Use decay sweep hyperparameters
        lr_map = {
            1: 0.089054,
            10: 0.001735,
            50: 0.092537,
            100: 0.493706,
            500: 0.013959,
            1000: 0.003435
        }
        tlr_map = {
            1: 0.001277,
            10: 0.008158,
            50: 3.202363,
            100: 0.011245,
            500: 0.048825,
            1000: 0.011312
        }
        lr = lr_map.get(te, 0)
        tlr = tlr_map.get(te, 0)

    print(f"{te:4d}| {lr:.4f} | {tlr:.4f} | {acc:.2f}%")
    if lr > 0:
        decay_tes.append(te)
        decay_lrs.append(lr)
        decay_tlrs.append(tlr)
        decay_accs.append(acc)

# Generate 16 transfer_every values (log scale from 1 to 10000)
print("\n" + "=" * 70)
print("PROPOSED 16 TRANSFER_EVERY VALUES (log-scale)")
print("=" * 70)

# Logarithmic spacing
te_16 = np.logspace(np.log10(1), np.log10(10000), 16, dtype=int)
te_16 = np.unique(te_16)  # Remove duplicates
if len(te_16) < 16:
    # Manual adjustment for better spacing
    te_16 = np.array([1, 2, 4, 7, 13, 23, 42, 75, 133, 237, 422, 750, 1333, 2371, 4217, 7500])

print("Transfer Every values:")
print(te_16)

# Interpolate lr and tlr for each mode
print("\n" + "=" * 70)
print("INTERPOLATED LR/TLR for 16 TE values")
print("=" * 70)

# Log-log interpolation for better fit
def log_interpolate(x_known, y_known, x_new):
    """Interpolate in log-log space with extrapolation."""
    # Remove zeros for log
    mask = np.array(y_known) > 0
    x_k = np.array(x_known)[mask]
    y_k = np.array(y_known)[mask]

    if len(x_k) < 2:
        return np.full(len(x_new), y_k[0] if len(y_k) > 0 else 0.1)

    log_x = np.log10(x_k)
    log_y = np.log10(y_k)

    # Linear interpolation in log space
    interp = interp1d(log_x, log_y, kind='linear', fill_value='extrapolate')
    log_y_new = interp(np.log10(x_new))

    return 10 ** log_y_new

# Hybrid interpolation
hybrid_lrs_interp = log_interpolate(hybrid_tes, hybrid_lrs, te_16)
hybrid_tlrs_interp = log_interpolate(hybrid_tes, hybrid_tlrs, te_16)

# Decay interpolation
decay_lrs_interp = log_interpolate(decay_tes, decay_lrs, te_16)
decay_tlrs_interp = log_interpolate(decay_tes, decay_tlrs, te_16)

print("\nHYBRID mode:")
print("TE    | LR     | TLR     | Variance for 3 trials")
print("-" * 60)
hybrid_configs = []
for i, te in enumerate(te_16):
    lr = hybrid_lrs_interp[i]
    tlr = hybrid_tlrs_interp[i]
    # Generate 3 variations (±20% and ±40%)
    lr_low = lr * 0.8
    lr_high = lr * 1.2
    tlr_low = tlr * 0.8
    tlr_high = tlr * 1.2

    hybrid_configs.append({
        'te': int(te),
        'lr_center': float(lr),
        'tlr_center': float(tlr),
        'trial_configs': [
            {'lr': float(lr), 'tlr': float(tlr)},
            {'lr': float(lr_low), 'tlr': float(tlr_high)},
            {'lr': float(lr_high), 'tlr': float(tlr_low)},
        ]
    })
    print(f"{te:5d} | {lr:.4f} | {tlr:.4f} | lr±20%, tlr±20%")

print("\nDECAY mode:")
print("TE    | LR     | TLR     | Variance for 3 trials")
print("-" * 60)
decay_configs = []
for i, te in enumerate(te_16):
    lr = decay_lrs_interp[i]
    tlr = decay_tlrs_interp[i]
    # Generate 3 variations
    lr_low = lr * 0.8
    lr_high = lr * 1.2
    tlr_low = tlr * 0.8
    tlr_high = tlr * 1.2

    decay_configs.append({
        'te': int(te),
        'lr_center': float(lr),
        'tlr_center': float(tlr),
        'trial_configs': [
            {'lr': float(lr), 'tlr': float(tlr)},
            {'lr': float(lr_low), 'tlr': float(tlr_high)},
            {'lr': float(lr_high), 'tlr': float(tlr_low)},
        ]
    })
    print(f"{te:5d} | {lr:.4f} | {tlr:.4f} | lr±20%, tlr±20%")

# Save to JSON
output = {
    'rank': 8,
    'num_transfer_every': 16,
    'transfer_every_values': te_16.tolist(),
    'trials_per_te': 3,
    'hybrid': {
        'configs': hybrid_configs,
        'known_data': {
            'tes': hybrid_tes,
            'lrs': hybrid_lrs,
            'tlrs': hybrid_tlrs,
            'accs': hybrid_accs,
        }
    },
    'decay': {
        'configs': decay_configs,
        'known_data': {
            'tes': decay_tes,
            'lrs': decay_lrs,
            'tlrs': decay_tlrs,
            'accs': decay_accs,
        }
    }
}

output_file = '/root/rank8_te16_configs.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n\nConfigs saved to: {output_file}")
print(f"Total experiments: 16 TE × 3 trials × 2 modes = {16*3*2} runs")

# Analysis summary
print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

print("\nHybrid mode lr/tlr relationship:")
hybrid_lr_tlr_corr = np.corrcoef(hybrid_lrs, hybrid_tlrs)[0, 1]
print(f"  Correlation: {hybrid_lr_tlr_corr:.3f}")
print(f"  Pattern: {'Negative' if hybrid_lr_tlr_corr < 0 else 'Positive'} correlation")
print(f"  LR range: {min(hybrid_lrs):.4f} - {max(hybrid_lrs):.4f}")
print(f"  TLR range: {min(hybrid_tlrs):.4f} - {max(hybrid_tlrs):.4f}")

print("\nDecay mode lr/tlr relationship:")
decay_lr_tlr_corr = np.corrcoef(decay_lrs, decay_tlrs)[0, 1]
print(f"  Correlation: {decay_lr_tlr_corr:.3f}")
print(f"  Pattern: {'Negative' if decay_lr_tlr_corr < 0 else 'Positive'} correlation")
print(f"  LR range: {min(decay_lrs):.4f} - {max(decay_lrs):.4f}")
print(f"  TLR range: {min(decay_tlrs):.4f} - {max(decay_tlrs):.4f}")
