#!/usr/bin/env python
"""Modify sweep script for grid search"""
with open("grid_sweep_sst2.py", "r") as f:
    content = f.read()

# Replace BoTorchSampler with GridSampler
content = content.replace(
    "from optuna_integration import BoTorchSampler",
    "# from optuna_integration import BoTorchSampler"
)

# Replace sampler creation
old_sampler = """    # Create study (same as TikiTaka v1: BoTorchSampler + NopPruner)
    sampler = BoTorchSampler()
    pruner = NopPruner()  # No pruning, use early stopping instead"""

new_sampler = """    # Grid search sampler
    # Define grid
    LRS = [1.0, 0.1, 0.01, 0.001]
    ALPHAS_STD = [0.08, 0.8, 8.0, 80.0]  # alpha_lrtt * rank for rank=8
    search_space = {"lora_alpha": ALPHAS_STD, "learning_rate": LRS}
    
    sampler = optuna.samplers.GridSampler(search_space)
    pruner = NopPruner()  # No pruning"""

content = content.replace(old_sampler, new_sampler)

# Change n_trials default to 16
content = content.replace(
    'parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")',
    'parser.add_argument("--n_trials", type=int, default=16, help="Number of trials (16 for 4x4 grid)")'
)

# Change epochs to 3
old_config = """TASK_CONFIGS = {
    "glue": {
        "max_seq_length": 128,
        "num_epochs": 3,"""

if "TASK_CONFIGS" in content:
    # Already correct
    pass

with open("grid_sweep_sst2.py", "w") as f:
    f.write(content)
    
print("Modified grid_sweep_sst2.py")
