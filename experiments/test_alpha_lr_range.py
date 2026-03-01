#!/usr/bin/env python
# coding=utf-8
"""
Test alpha*lr combinations to find safe ranges for SST-2
Tests multiple combinations in sequence to find which ones avoid NaN
"""

import subprocess
import sys
import json
import os
from datetime import datetime

PYTHON = "/data/venvs/aihwkit_gpu/bin/python"
SCRIPT = "/data/LRTT_transformer/experiments/sweep_lrtt_lora_optuna.py"

# Test combinations: (alpha, lr)
# Start with safe ranges and expand
test_combinations = [
    # Conservative (should work)
    (1.0, 0.001),
    (1.0, 0.005),

    # Medium
    (0.5, 0.001),
    (2.0, 0.001),

    # Lower alpha (problematic?)
    (0.1, 0.001),
    (0.1, 0.005),

    # Higher lr
    (1.0, 0.008),
    (2.0, 0.005),

    # Edge cases
    (0.05, 0.001),  # Minimum safe alpha
    (10.0, 0.001),  # High alpha
]

results = []

print("=" * 80)
print("ALPHA * LR RANGE TESTING FOR SST-2")
print("=" * 80)
print(f"Testing {len(test_combinations)} combinations")
print(f"Task: SST-2, Mode: sixt1c_lora, Epochs: 1")
print("=" * 80)

for i, (alpha, lr) in enumerate(test_combinations):
    print(f"\n[Test {i+1}/{len(test_combinations)}] alpha={alpha}, lr={lr}, alpha*lr={alpha*lr:.6f}")

    # Create temporary modified script
    import tempfile
    import shutil

    # Read original script
    with open(SCRIPT, 'r') as f:
        script_content = f.read()

    # Replace hyperparameter sampling with fixed values
    modified_content = script_content.replace(
        'lora_alpha = trial.suggest_float("lora_alpha", 0.1, 100.0, log=True)',
        f'lora_alpha = {alpha}  # FIXED FOR TESTING'
    ).replace(
        'lr = trial.suggest_float("learning_rate", 5e-4, 1e-2, log=True)',
        f'lr = {lr}  # FIXED FOR TESTING'
    )

    # Write to temporary file
    temp_script = f"/tmp/sweep_test_{i}.py"
    with open(temp_script, 'w') as f:
        f.write(modified_content)

    # Run test
    study_name = f"test_a{alpha}_lr{lr}"
    log_file = f"/tmp/test_alpha_lr_{i}.log"

    try:
        result = subprocess.run(
            [PYTHON, temp_script,
             "--task", "glue",
             "--task_name", "sst2",
             "--mode", "sixt1c_lora",
             "--rank", "8",
             "--target_modules", "query", "key", "value",
             "--n_trials", "1",
             "--study_name", study_name],
            timeout=1200,  # 20 minutes max per trial
            capture_output=True,
            text=True
        )

        # Save log
        with open(log_file, 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)

        # Parse output for key metrics
        output = result.stdout + result.stderr

        # Check for NaN
        has_nan = "nan" in output.lower() or "NaN" in output

        # Extract eval loss if available
        eval_loss = None
        eval_accuracy = None
        for line in output.split('\n'):
            if 'eval_loss' in line.lower():
                try:
                    # Extract number after eval_loss
                    import re
                    match = re.search(r'eval_loss[:\s=]+(\d+\.\d+|nan)', line, re.IGNORECASE)
                    if match:
                        val = match.group(1)
                        eval_loss = float(val) if val.lower() != 'nan' else float('nan')
                except:
                    pass
            if 'eval_accuracy' in line.lower() or 'accuracy' in line.lower():
                try:
                    import re
                    match = re.search(r'accuracy[:\s=]+(\d+\.\d+)', line, re.IGNORECASE)
                    if match:
                        eval_accuracy = float(match.group(1))
                except:
                    pass

        status = "SUCCESS" if result.returncode == 0 and not has_nan else "FAILED"

        result_dict = {
            "alpha": alpha,
            "lr": lr,
            "alpha_lr": alpha * lr,
            "status": status,
            "has_nan": has_nan,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "log_file": log_file
        }

        results.append(result_dict)

        print(f"  Status: {status}")
        if eval_loss is not None:
            print(f"  Eval Loss: {eval_loss:.4f}")
        if eval_accuracy is not None:
            print(f"  Eval Accuracy: {eval_accuracy:.4f}")
        if has_nan:
            print(f"  ⚠️  NaN detected!")

    except subprocess.TimeoutExpired:
        print(f"  ⚠️  TIMEOUT (>20 min)")
        results.append({
            "alpha": alpha,
            "lr": lr,
            "alpha_lr": alpha * lr,
            "status": "TIMEOUT",
            "has_nan": False,
            "eval_loss": None,
            "eval_accuracy": None,
            "log_file": log_file
        })
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        results.append({
            "alpha": alpha,
            "lr": lr,
            "alpha_lr": alpha * lr,
            "status": "ERROR",
            "has_nan": False,
            "eval_loss": None,
            "eval_accuracy": None,
            "error": str(e)
        })

    # Clean up temp script
    if os.path.exists(temp_script):
        os.remove(temp_script)

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Save results
results_file = f"/tmp/alpha_lr_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {results_file}\n")

# Print table
print(f"{'Alpha':<8} {'LR':<10} {'Alpha*LR':<12} {'Status':<10} {'NaN?':<8} {'Eval Loss':<12} {'Accuracy':<10}")
print("-" * 80)
for r in results:
    eval_loss_str = f"{r['eval_loss']:.4f}" if r['eval_loss'] is not None else "N/A"
    acc_str = f"{r['eval_accuracy']:.4f}" if r['eval_accuracy'] is not None else "N/A"
    nan_str = "YES" if r['has_nan'] else "NO"
    print(f"{r['alpha']:<8.2f} {r['lr']:<10.4f} {r['alpha_lr']:<12.6f} {r['status']:<10} {nan_str:<8} {eval_loss_str:<12} {acc_str:<10}")

# Identify safe ranges
successful = [r for r in results if r['status'] == 'SUCCESS' and not r['has_nan']]
if successful:
    print(f"\n✓ {len(successful)}/{len(results)} combinations succeeded without NaN")

    alphas = [r['alpha'] for r in successful]
    lrs = [r['lr'] for r in successful]

    print(f"\nSafe ranges:")
    print(f"  Alpha: [{min(alphas)}, {max(alphas)}]")
    print(f"  LR: [{min(lrs)}, {max(lrs)}]")
    print(f"  Alpha*LR: [{min(r['alpha_lr'] for r in successful):.6f}, {max(r['alpha_lr'] for r in successful):.6f}]")
else:
    print(f"\n✗ No successful combinations found!")
    print(f"   All {len(results)} tests failed or had NaN")

print("=" * 80)
