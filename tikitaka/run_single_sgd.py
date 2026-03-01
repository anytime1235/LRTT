#!/usr/bin/env python
"""Single run: Adam-best params with AnalogSGD optimizer.

Reuses all infrastructure from sweep_tikitaka_batch256_te_extended.py
but runs a single fixed-param experiment instead of a sweep.
"""

import os
import sys
import json
import math
from datetime import datetime

import torch
from transformers import AutoTokenizer, set_seed
import wandb

sys.path.insert(0, '/data/LRTT_transformer/src')
from aihwkit.optim import AnalogSGD

# Import everything from the sweep script
sys.path.insert(0, os.path.dirname(__file__))
from sweep_tikitaka_batch256_te_extended import (
    MODEL_NAME, SEED, BATCH_SIZE, MIN_LR_RATE, WANDB_PROJECT,
    create_tikitaka_v2_config, create_squad_model, load_squad_data,
    evaluate_squad, train_epoch, get_linear_schedule_with_min_lr,
)

# Adam-best params (Trial 4 from previous sweep)
PARAMS = {
    "learning_rate": 0.002552951604697378,
    "transfer_lr": 0.21272133025080567,
    "fast_lr": 3.0,
    "transfer_every": 979,
    "auto_granularity": 169.0,
    "in_chop_prob": 0.061,
}

NUM_EPOCHS = 15
OPTIMIZER = "sgd"


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"/data/LRTT_transformer/tikitaka/results/single_sgd_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f"Single Run: Adam-best params with AnalogSGD")
    print("=" * 60)
    print(f"Optimizer: {OPTIMIZER.upper()}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Params: {json.dumps(PARAMS, indent=2)}")
    print(f"Results dir: {results_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(SEED)

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    # Create model
    model = create_squad_model(PARAMS, device)

    # AnalogSGD
    optimizer = AnalogSGD(model.parameters(), lr=PARAMS["learning_rate"])
    optimizer.regroup_param_groups(model)
    print(f"Optimizer: AnalogSGD (lr={PARAMS['learning_rate']:.6f})")

    # Scheduler
    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_min_lr(
        optimizer,
        num_training_steps=num_training_steps,
        min_lr_rate=MIN_LR_RATE,
    )

    # WandB
    os.environ["WANDB_MODE"] = "offline"
    wandb.init(
        project=WANDB_PROJECT,
        name=f"squad_sgd_adam_best_{timestamp}",
        config={"optimizer": OPTIMIZER, "epochs": NUM_EPOCHS, **PARAMS},
    )

    # Training
    best_f1 = -float('inf')
    best_em = -float('inf')
    epoch_results = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(
            model, optimizer, scheduler, train_loader, device,
            "squad", epoch, 0,
        )

        if math.isnan(train_loss):
            print(f"  Loss diverged at epoch {epoch}")
            break

        eval_f1, eval_em = evaluate_squad(model, eval_features, eval_examples, tokenizer, device)
        lr_now = scheduler.get_last_lr()[0]

        wandb.log({
            "epoch": epoch, "train/loss": train_loss,
            "eval/f1": eval_f1, "eval/em": eval_em, "lr": lr_now,
        })
        print(f"  [Epoch {epoch}/{NUM_EPOCHS}] Loss: {train_loss:.4f}, F1: {eval_f1:.2f}, EM: {eval_em:.2f}")

        epoch_results.append({
            "epoch": epoch, "loss": train_loss, "f1": eval_f1, "em": eval_em, "lr": lr_now,
        })

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_em = eval_em

    wandb.finish()

    # Save results
    result = {
        "optimizer": OPTIMIZER,
        "params": PARAMS,
        "num_epochs": NUM_EPOCHS,
        "best_f1": best_f1,
        "best_em": best_em,
        "epoch_results": epoch_results,
        "timestamp": timestamp,
    }

    result_file = os.path.join(results_dir, "result.json")
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: Best F1={best_f1:.2f}, EM={best_em:.2f}")
    print(f"Saved to: {result_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
