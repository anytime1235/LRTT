#!/usr/bin/env python
"""
Grid search for SST-2 with LRTT-LoRA
LR: [1.0, 0.1, 0.01, 0.001]
alpha_lrtt: [0.01, 0.1, 1.0, 10.0]
Total: 16 combinations, 3 epochs each
"""
import sys
import subprocess
from itertools import product

# Grid parameters
LRS = [1.0, 0.1, 0.01, 0.001]
ALPHAS = [0.01, 0.1, 1.0, 10.0]
EPOCHS = 3
RANK = 8

# Generate all combinations
combinations = list(product(LRS, ALPHAS))
total = len(combinations)

print("=" * 80)
print("SST-2 GRID SEARCH - LRTT-LoRA")
print("=" * 80)
print(f"Task: GLUE SST-2")
print(f"Target modules: query, key, value, classifier")
print(f"Mode: sixt1c_lora")
print(f"Epochs: {EPOCHS}")
print(f"Rank: {RANK}")
print(f"LR values: {LRS}")
print(f"Alpha (LRTT) values: {ALPHAS}")
print(f"Total combinations: {total}")
print("=" * 80)
print()

for idx, (lr, alpha) in enumerate(combinations, 1):
    print(f"[{idx}/{total}] Running: lr={lr}, alpha_lrtt={alpha}")
    
    # Convert alpha_lrtt back to alpha_standard for the sweep script
    # Since sweep script does: alpha_lrtt = alpha_standard / rank
    # We need: alpha_standard = alpha_lrtt * rank
    alpha_standard = alpha * RANK
    
    cmd = [
        sys.executable,
        "sweep_lrtt_lora_optuna.py",
        "--task", "glue",
        "--task_name", "sst2",
        "--mode", "sixt1c_lora",
        "--rank", str(RANK),
        "--target_modules", "query", "key", "value", "classifier",
        "--n_trials", "1",
        "--study_name", f"grid_sst2_{idx}",
        "--force_alpha", str(alpha_standard),
        "--force_lr", str(lr),
        "--epochs", str(EPOCHS)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"  ✓ Completed: lr={lr}, alpha_lrtt={alpha}")
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: lr={lr}, alpha_lrtt={alpha}")
        print(f"    Error: {e}")
    print()

print("=" * 80)
print("GRID SEARCH COMPLETED")
print("=" * 80)
