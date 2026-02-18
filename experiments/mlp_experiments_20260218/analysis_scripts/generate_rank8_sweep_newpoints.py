#!/usr/bin/env python3
"""Generate rank=8 transfer_every sweep configs - NEW POINTS ONLY.

Excludes already tested TE values: [1, 10, 50, 100, 500, 1000]
- 1~1000: 20 points total (14 new)
- 1000~10000: 10 points (all new)
Total: 24 new TE values
"""

import json
import numpy as np

# Load best configs for reference
with open('/root/LRTT/experiments/mnist_sweep_analysis/lrtt_sweep_best_configs.json', 'r') as f:
    configs = json.load(f)

# Known data points for hybrid (rank=8)
hybrid_data = [
    (1, 0.1135, 0.0015, 90.07),
    (10, 0.0502, 0.0110, 90.96),
    (50, 0.7011, 0.0042, 96.20),
    (100, 0.1719, 0.0027, 96.20),
    (500, 0.2336, 0.0044, 95.17),
    (1000, 0.3198, 1.2764, 92.79),
]

# Known data points for decay (rank=8)
decay_data = [
    (1, 0.089054, 0.001277, 96.58),
    (10, 0.001735, 0.008158, 97.17),
    (50, 0.092537, 3.202363, 96.02),
    (100, 0.493706, 0.011245, 96.92),
    (500, 0.013959, 0.048825, 94.08),
    (1000, 0.003435, 0.011312, 91.17),
]

# Already tested TE values (to exclude)
EXISTING_TES = [1, 10, 50, 100, 500, 1000]

print("=" * 70)
print("RANK=8 TRANSFER_EVERY SWEEP - NEW POINTS ONLY")
print("=" * 70)
print(f"\nExcluding already tested TE values: {EXISTING_TES}")

# Generate 20 points for 1~1000 range (log scale)
te_1_1000_all = np.logspace(np.log10(1), np.log10(1000), 20, dtype=int)
te_1_1000_all = np.unique(te_1_1000_all)

# If not exactly 20, manually create better distribution
if len(te_1_1000_all) != 20:
    te_1_1000_all = np.array([
        1, 2, 4, 8, 12, 18, 27, 40, 60, 85,
        120, 170, 240, 340, 480, 680, 820, 900, 950, 1000
    ], dtype=int)

# Remove already tested values
te_1_1000_new = [te for te in te_1_1000_all if te not in EXISTING_TES]

# Generate 10 points for 1000~10000 range (approximately linear)
te_1k_10k = np.linspace(1000, 10000, 12, dtype=int)[1:]  # Exclude 1000, take 11 points
te_1k_10k = np.array([1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])

# Combine all new TE values
te_values = np.array(sorted(te_1_1000_new + te_1k_10k.tolist()), dtype=int)

print(f"\n1~1000 range: {len(te_1_1000_all)} total points, {len(te_1_1000_new)} new points")
print(f"1000~10000 range: {len(te_1k_10k)} new points")
print(f"Total new TE values: {len(te_values)}")
print(f"\nNew TE values to test:")
print(te_values.tolist())

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

    result = []
    for te in te_new:
        if te <= tes[0]:
            lr = lrs[0]
            tlr = tlrs[0]
        elif te >= tes[-1]:
            # For TE > 1000, use last known value
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

            # Geometric mean for lr and tlr
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
print("HYBRID MODE CONFIGS (NEW POINTS)")
print("=" * 70)
print(f"{'TE':<6} | {'LR':<8} | {'TLR':<8} | Trial Variations")
print("-" * 70)

hybrid_full_configs = []
for te, lr, tlr in hybrid_configs:
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
print("DECAY MODE CONFIGS (NEW POINTS)")
print("=" * 70)
print(f"{'TE':<6} | {'LR':<8} | {'TLR':<8} | Trial Variations")
print("-" * 70)

decay_full_configs = []
for te, lr, tlr in decay_configs:
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
        'lifetime': 46505,  # sixt1c value (fixed)
        'num_transfer_every': len(te_values),
        'trials_per_te': 3,
        'total_experiments': len(te_values) * 3 * 2,
        'note': 'NEW points only. Excludes already tested: [1, 10, 50, 100, 500, 1000]',
        'existing_tes_excluded': EXISTING_TES
    },
    'transfer_every_values': te_values.tolist(),
    'existing_tes': EXISTING_TES,
    'hybrid': {
        'mode': 'hybrid (A=0 hard reset, B unchanged)',
        'configs': hybrid_full_configs
    },
    'decay': {
        'mode': 'decay (A and B both decay)',
        'configs': decay_full_configs
    }
}

output_file = '/root/rank8_newte_sweep_configs.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Configuration saved to: {output_file}")
print(f"\nExcluded (already tested): {len(EXISTING_TES)} TE values")
print(f"New TE values to test: {len(te_values)}")
print(f"  - Range 1~1000: {len(te_1_1000_new)} new points")
print(f"  - Range 1000~10000: {len(te_1k_10k)} new points")
print(f"\nTotal NEW experiments: {len(te_values) * 3 * 2} runs")
print(f"  - Hybrid: {len(te_values)} TE × 3 trials = {len(te_values) * 3} runs")
print(f"  - Decay:  {len(te_values)} TE × 3 trials = {len(te_values) * 3} runs")
print(f"\nTE range: {te_values[0]} - {te_values[-1]}")
