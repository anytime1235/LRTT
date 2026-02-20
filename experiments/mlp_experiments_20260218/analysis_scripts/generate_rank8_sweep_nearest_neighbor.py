#!/usr/bin/env python3
"""Generate rank=8 sweep configs using Nearest Neighbor strategy.

For each new TE, use lr/tlr from the closest existing TE.
Add ±30% variations for 3 trials.
"""

import json
import numpy as np

# Load best configs
with open('/root/LRTT/experiments/mnist_sweep_analysis/lrtt_sweep_best_configs.json', 'r') as f:
    configs = json.load(f)

# Known TE values with lr/tlr for rank=8
EXISTING_TES = [1, 10, 50, 100, 500, 1000]

# Hybrid rank=8 existing data
hybrid_known = {
    1: {'lr': 0.1135, 'tlr': 0.0015},
    10: {'lr': 0.0502, 'tlr': 0.0110},
    50: {'lr': 0.7011, 'tlr': 0.0042},
    100: {'lr': 0.1719, 'tlr': 0.0027},
    500: {'lr': 0.2336, 'tlr': 0.0044},
    1000: {'lr': 0.3198, 'tlr': 1.2764},
}

# Decay rank=8 existing data (from sweep_softbounds_lifetime.py)
decay_known = {
    1: {'lr': 0.089054, 'tlr': 0.001277},
    10: {'lr': 0.001735, 'tlr': 0.008158},
    50: {'lr': 0.092537, 'tlr': 3.202363},
    100: {'lr': 0.493706, 'tlr': 0.011245},
    500: {'lr': 0.013959, 'tlr': 0.048825},
    1000: {'lr': 0.003435, 'tlr': 0.011312},
}

# New TE values (1~1000: 20 points total, 1000~10000: 10 points)
te_1_1000_all = np.array([
    1, 2, 4, 8, 12, 18, 27, 40, 60, 85,
    120, 170, 240, 340, 480, 680, 820, 900, 950, 1000
], dtype=int)

te_1k_10k = np.array([1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])

# Remove already tested
te_1_1000_new = [te for te in te_1_1000_all if te not in EXISTING_TES]
te_values = np.array(sorted(te_1_1000_new + te_1k_10k.tolist()), dtype=int)

print("=" * 70)
print("RANK=8 SWEEP - NEAREST NEIGHBOR STRATEGY")
print("=" * 70)
print(f"\nExisting TE values: {EXISTING_TES}")
print(f"New TE values: {len(te_values)}")
print(f"Total experiments: {len(te_values) * 3 * 2} = {len(te_values) * 3 * 2}")

def find_nearest_neighbor(te, known_tes):
    """Find the nearest TE in known_tes."""
    known_arr = np.array(known_tes)
    idx = np.argmin(np.abs(known_arr - te))
    return known_arr[idx]

def generate_configs(te_values, known_data, mode_name):
    """Generate configs using nearest neighbor."""
    known_tes = list(known_data.keys())
    configs = []

    print(f"\n{'=' * 70}")
    print(f"{mode_name.upper()} MODE - NEAREST NEIGHBOR MAPPING")
    print(f"{'=' * 70}")
    print(f"{'New TE':<8} {'Nearest':<8} {'LR':<10} {'TLR':<10} {'Trials'}")
    print("-" * 70)

    for te in te_values:
        # For TE > 1000, use TE=1000's values
        if te > 1000:
            nearest_te = 1000
        else:
            nearest_te = find_nearest_neighbor(te, known_tes)

        lr_base = known_data[nearest_te]['lr']
        tlr_base = known_data[nearest_te]['tlr']

        # Generate 3 trial variations with ±30%
        trials = [
            {'lr': lr_base, 'tlr': tlr_base},  # Center
            {'lr': lr_base * 0.7, 'tlr': tlr_base * 1.3},  # LR down, TLR up
            {'lr': lr_base * 1.3, 'tlr': tlr_base * 0.7},  # LR up, TLR down
        ]

        configs.append({
            'te': int(te),
            'nearest_te': int(nearest_te),
            'lr_base': float(lr_base),
            'tlr_base': float(tlr_base),
            'trials': trials
        })

        print(f"{te:<8} {nearest_te:<8} {lr_base:<10.5f} {tlr_base:<10.5f} ±30% (3 trials)")

    return configs

# Generate configs
hybrid_configs = generate_configs(te_values, hybrid_known, "Hybrid")
decay_configs = generate_configs(te_values, decay_known, "Decay")

# Save configuration
output = {
    'metadata': {
        'rank': 8,
        'lifetime': 46505,
        'strategy': 'nearest_neighbor',
        'num_transfer_every': len(te_values),
        'trials_per_te': 3,
        'variation': '±30%',
        'total_experiments': len(te_values) * 3 * 2,
        'note': 'Uses nearest existing TE for lr/tlr selection',
        'existing_tes_excluded': EXISTING_TES
    },
    'transfer_every_values': te_values.tolist(),
    'existing_tes': EXISTING_TES,
    'hybrid': {
        'mode': 'hybrid (A=0 hard reset, B unchanged)',
        'known_data': hybrid_known,
        'configs': hybrid_configs
    },
    'decay': {
        'mode': 'decay (A and B both decay)',
        'known_data': decay_known,
        'configs': decay_configs
    }
}

output_file = '/root/rank8_nn_sweep_configs.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Strategy: Nearest Neighbor")
print(f"Configuration saved to: {output_file}")
print(f"\nNew TE values: {len(te_values)}")
print(f"  Range 1~1000: {len([t for t in te_values if t <= 1000])} points")
print(f"  Range 1000~10000: {len([t for t in te_values if t > 1000])} points")
print(f"\nTotal experiments: {len(te_values) * 3 * 2}")
print(f"  - Hybrid: {len(te_values)} TE × 3 trials = {len(te_values) * 3}")
print(f"  - Decay:  {len(te_values)} TE × 3 trials = {len(te_values) * 3}")
