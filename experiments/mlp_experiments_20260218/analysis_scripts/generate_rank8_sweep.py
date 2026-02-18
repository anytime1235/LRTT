#!/usr/bin/env python3
"""Generate rank=8 transfer_every sweep configs with realistic lr/tlr."""

import json
import numpy as np
from scipy.interpolate import interp1d

# Load best configs
with open('/root/LRTT/experiments/mnist_sweep_analysis/lrtt_sweep_best_configs.json', 'r') as f:
    configs = json.load(f)

# Extract rank=8 data
hybrid_8 = configs['hybrid_sweep']['by_rank_te']['8']
decay_8 = configs['decay_sweep']['by_rank_te']['8']

# Known data points for hybrid
hybrid_data = [
    (1, 0.1135, 0.0015, 90.07),
    (10, 0.0502, 0.0110, 90.96),
    (50, 0.7011, 0.0042, 96.20),
    (100, 0.1719, 0.0027, 96.20),
    (500, 0.2336, 0.0044, 95.17),
    (1000, 0.3198, 1.2764, 92.79),
]

# Known data points for decay (from sweep_softbounds_lifetime.py HYPERPARAMETERS)
decay_data = [
    (1, 0.089054, 0.001277, 96.58),
    (10, 0.001735, 0.008158, 97.17),
    (50, 0.092537, 3.202363, 96.02),
    (100, 0.493706, 0.011245, 96.92),
    (500, 0.013959, 0.048825, 94.08),
    (1000, 0.003435, 0.011312, 91.17),
]

print("=" * 70)
print("RANK=8 TRANSFER_EVERY SWEEP CONFIGURATION")
print("=" * 70)

# Generate 16 TE values in log-scale, focused on 1-1000 range
# Add a few beyond 1000 but not too far
te_values = np.array([1, 2, 4, 8, 15, 30, 50, 75, 100, 150, 250, 400, 600, 800, 1000, 1500], dtype=int)

print(f"\n16 Transfer Every values: {te_values.tolist()}")
print(f"Range: {te_values[0]} - {te_values[-1]}")

def smart_interpolate(known_data, te_new):
    """Interpolate with clamping for extrapolation."""
    tes = np.array([x[0] for x in known_data])
    lrs = np.array([x[1] for x in known_data])
    tlrs = np.array([x[2] for x in known_data])

    # Sort by TE
    sort_idx = np.argsort(tes)
    tes = tes[sort_idx]
    lrs = lrs[sort_idx]
    tlrs = tlrs[sort_idx]

    # For each new TE, find closest known TEs and interpolate
    result = []
    for te in te_new:
        if te <= tes[0]:
            # Clamp to first value
            lr = lrs[0]
            tlr = tlrs[0]
        elif te >= tes[-1]:
            # Clamp to last value
            lr = lrs[-1]
            tlr = tlrs[-1]
        else:
            # Find bracketing points
            idx_upper = np.searchsorted(tes, te)
            idx_lower = idx_upper - 1

            te_lower, te_upper = tes[idx_lower], tes[idx_upper]
            lr_lower, lr_upper = lrs[idx_lower], lrs[idx_upper]
            tlr_lower, tlr_upper = tlrs[idx_lower], tlrs[idx_upper]

            # Linear interpolation in log-space for TE
            if te_upper > te_lower:
                log_weight = (np.log10(te) - np.log10(te_lower)) / (np.log10(te_upper) - np.log10(te_lower))
            else:
                log_weight = 0.5

            # Geometric mean for lr and tlr (better for positive values)
            if lr_lower > 0 and lr_upper > 0:
                lr = np.exp(np.log(lr_lower) * (1 - log_weight) + np.log(lr_upper) * log_weight)
            else:
                lr = lr_lower * (1 - log_weight) + lr_upper * log_weight

            if tlr_lower > 0 and tlr_upper > 0:
                tlr = np.exp(np.log(tlr_lower) * (1 - log_weight) + np.log(tlr_upper) * log_weight)
            else:
                tlr = tlr_lower * (1 - log_weight) + tlr_upper * log_weight

        result.append((te, lr, tlr))

    return result

# Generate configs for hybrid
hybrid_configs = smart_interpolate(hybrid_data, te_values)

print("\n" + "=" * 70)
print("HYBRID MODE CONFIGS")
print("=" * 70)
print(f"{'TE':<6} | {'LR':<8} | {'TLR':<8} | Trial Variations")
print("-" * 70)

hybrid_full_configs = []
for te, lr, tlr in hybrid_configs:
    # Generate 3 trial variations:
    # 1. Center values
    # 2. lr -20%, tlr +20%
    # 3. lr +20%, tlr -20%
    trials = [
        {'lr': lr, 'tlr': tlr},
        {'lr': lr * 0.8, 'tlr': tlr * 1.2},
        {'lr': lr * 1.2, 'tlr': tlr * 0.8},
    ]

    hybrid_full_configs.append({
        'te': int(te),
        'lr_center': float(lr),
        'tlr_center': float(tlr),
        'trials': trials
    })

    print(f"{te:<6} | {lr:<8.5f} | {tlr:<8.5f} | 3 trials: center, lr±20%/tlr∓20%")

# Generate configs for decay
decay_configs = smart_interpolate(decay_data, te_values)

print("\n" + "=" * 70)
print("DECAY MODE CONFIGS")
print("=" * 70)
print(f"{'TE':<6} | {'LR':<8} | {'TLR':<8} | Trial Variations")
print("-" * 70)

decay_full_configs = []
for te, lr, tlr in decay_configs:
    # Generate 3 trial variations
    trials = [
        {'lr': lr, 'tlr': tlr},
        {'lr': lr * 0.8, 'tlr': tlr * 1.2},
        {'lr': lr * 1.2, 'tlr': tlr * 0.8},
    ]

    decay_full_configs.append({
        'te': int(te),
        'lr_center': float(lr),
        'tlr_center': float(tlr),
        'trials': trials
    })

    print(f"{te:<6} | {lr:<8.5f} | {tlr:<8.5f} | 3 trials: center, lr±20%/tlr∓20%")

# Save configuration
output = {
    'metadata': {
        'rank': 8,
        'lifetime': 100000,  # Use best lifetime for rank=8
        'num_transfer_every': 16,
        'trials_per_te': 3,
        'total_experiments': 16 * 3 * 2,
        'note': 'Each TE has 3 trials with lr/tlr variations. Best result saved per TE.'
    },
    'transfer_every_values': te_values.tolist(),
    'hybrid': {
        'mode': 'hybrid (A=0 hard reset, B unchanged)',
        'configs': hybrid_full_configs
    },
    'decay': {
        'mode': 'decay (A and B both decay)',
        'configs': decay_full_configs
    }
}

output_file = '/root/rank8_te16_sweep_configs.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Configuration saved to: {output_file}")
print(f"Total experiments: {16 * 3 * 2} runs")
print(f"  - Hybrid: 16 TE × 3 trials = 48 runs")
print(f"  - Decay:  16 TE × 3 trials = 48 runs")
print(f"\nEach TE will run 3 trials and save the best result.")
print(f"TE range: {te_values[0]} - {te_values[-1]}")
