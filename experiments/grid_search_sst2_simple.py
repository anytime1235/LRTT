#!/usr/bin/env python
"""Simple grid search for SST-2"""
import subprocess
import sys
from itertools import product

LRS = [1.0, 0.1, 0.01, 0.001]
ALPHAS_LRTT = [0.01, 0.1, 1.0, 10.0]
RANK = 8
EPOCHS = 3

combinations = list(product(LRS, ALPHAS_LRTT))
total = len(combinations)

print("="*80)
print(f"SST-2 GRID SEARCH - {total} combinations")
print("="*80)
print(f"LR: {LRS}")
print(f"Alpha_LRTT: {ALPHAS_LRTT}")
print(f"Epochs: {EPOCHS}, Rank: {RANK}")
print(f"Target: QKV + classifier")
print("="*80)

for idx, (lr, alpha_lrtt) in enumerate(combinations, 1):
    # Convert to standard alpha (script will convert back)
    alpha_std = alpha_lrtt * RANK
    
    print(f"\n[{idx}/{total}] lr={lr}, alpha_lrtt={alpha_lrtt}")
    
    # We'll use Optuna with GridSampler
    # Create a temporary script that uses GridSampler
    study_name = f"grid_sst2_{idx}_lr{lr}_a{alpha_lrtt}"
    
    cmd = [
        "/data/venvs/aihwkit_gpu/bin/python",
        "-c",
        f"""
import sys
sys.path.insert(0, "/data/LRTT_transformer/experiments")
sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")
import optuna
from sweep_lrtt_lora_optuna import objective, create_args_dict

# Grid sampler with single point
sampler = optuna.samplers.GridSampler({{"lora_alpha": [{alpha_std}], "learning_rate": [{lr}]}})
study = optuna.create_study(
    study_name="{study_name}",
    direction="maximize",
    sampler=sampler
)

args_dict = create_args_dict(
    task_type="glue",
    task_name="sst2",
    mode="sixt1c_lora",
    rank={RANK},
    target_modules=["query", "key", "value", "classifier"],
    num_epochs={EPOCHS}
)

study.optimize(lambda trial: objective(trial, args_dict), n_trials=1)
print(f"Best value: {{study.best_value}}")
"""
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"  ✓ Completed")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

print("\n" + "="*80)
print("GRID SEARCH COMPLETED")
print("="*80)
