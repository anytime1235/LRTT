#!/usr/bin/env python
"""
Check how scheduler applies to different parameter groups.
"""

import torch
from transformers import get_linear_schedule_with_warmup

# Simulate our parameter groups
dummy_params_qkv = [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(3)]
dummy_params_classifier = [torch.nn.Parameter(torch.randn(5, 5)) for _ in range(2)]

# Create optimizer with different learning rates
param_groups = [
    {'params': dummy_params_qkv, 'lr': 0.01, 'name': 'QKV'},
    {'params': dummy_params_classifier, 'lr': 1.0, 'name': 'Classifier'}
]

optimizer = torch.optim.SGD(param_groups, lr=1.0)  # base lr (used for classifier)

# Create scheduler (5% warmup, similar to our test)
total_steps = 3159  # Same as our test
warmup_steps = int(0.05 * total_steps)  # 158 steps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print("=" * 80)
print("SCHEDULER BEHAVIOR WITH MULTIPLE PARAMETER GROUPS")
print("=" * 80)
print(f"Total steps: {total_steps}")
print(f"Warmup steps: {warmup_steps} (5%)")
print(f"\nInitial LRs:")
print(f"  QKV group: {optimizer.param_groups[0]['lr']:.4f}")
print(f"  Classifier group: {optimizer.param_groups[1]['lr']:.4f}")
print("=" * 80)

# Simulate training and track LR changes
checkpoints = [0, 50, warmup_steps, 500, 1000, 2000, 3000, total_steps-1]

print("\nLR at different training steps:")
print(f"{'Step':<10} {'QKV LR':<15} {'Classifier LR':<15} {'Multiplier':<15}")
print("-" * 60)

for step in checkpoints:
    # Reset scheduler state
    optimizer = torch.optim.SGD(param_groups, lr=1.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Step to the checkpoint
    for _ in range(step + 1):
        scheduler.step()

    qkv_lr = optimizer.param_groups[0]['lr']
    classifier_lr = optimizer.param_groups[1]['lr']
    multiplier = classifier_lr / 1.0  # Since classifier base lr = 1.0

    print(f"{step:<10} {qkv_lr:<15.6f} {classifier_lr:<15.6f} {multiplier:<15.6f}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("The scheduler applies the SAME multiplier to all parameter groups.")
print(f"\nAt each step, the effective learning rates are:")
print(f"  QKV LR = 0.01 × multiplier")
print(f"  Classifier LR = 1.0 × multiplier")
print(f"\nWhere multiplier follows linear warmup + linear decay schedule:")
print(f"  - Steps 0-{warmup_steps}: warmup from 0 to 1.0")
print(f"  - Steps {warmup_steps}-{total_steps}: decay from 1.0 to 0.0")
print("=" * 80)
