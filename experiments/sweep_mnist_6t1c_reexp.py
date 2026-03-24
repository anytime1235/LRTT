#!/usr/bin/env python3
"""sweep_mnist_6t1c_reexp.py — MNIST 6T1C re-experiment with formula-based HP.

HP is determined by mode/rank/te formula (no Optuna search).
Each (mode, rank, te) cell runs with 3 seeds.

Usage:
  # P1: Full grid
  python sweep_mnist_6t1c_reexp.py --priority 1

  # P2: Collapse analysis
  python sweep_mnist_6t1c_reexp.py --priority 2

  # Single cell test
  python sweep_mnist_6t1c_reexp.py --modes decay --ranks 8 --tes 100 --seeds 42
"""

import argparse
import csv
import json
import math
import os
from time import time

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTPreset
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.rpu_base import cuda

# ─── Device ───────────────────────────────────────────────────────────────────
USE_CUDA = 1 if cuda.is_compiled() else 0
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# ─── Constants ────────────────────────────────────────────────────────────────
INPUT_SIZE = 784
HIDDEN_SIZE = 256
OUTPUT_SIZE = 10
EPOCHS = 30
BATCH_SIZE = 64
PATH_DATASET = os.path.join("data", "DATASET")


# ═══════════════════════════════════════════════════════════════════════════════
# HP Formula (from trend analysis of prior experiments)
# ═══════════════════════════════════════════════════════════════════════════════

def get_hp(mode, rank, te, tlr_override=None):
    """HP assignment. Fixed lr=0.3, tlr=0.005 for all conditions.

    Args:
        tlr_override: If provided, use this tlr instead of default 0.005.
    """
    lr = 0.3
    tlr = tlr_override if tlr_override is not None else 0.005
    return lr, tlr


# ═══════════════════════════════════════════════════════════════════════════════
# Model / Data
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(PATH_DATASET, download=True, train=True, transform=transform)
    val_set = datasets.MNIST(PATH_DATASET, download=True, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def create_model(rank, te, mode):
    """Create MLP 784->256->10 with 6T1C LRTT."""
    reinit_mode = "standard" if mode == "reset" else "decay"

    device_config = PythonLRTTPreset.sixt1c_ab(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        dt_batch_sec=0.0,          # lifetime=0 (no retention)
        include_retention=False,
        c_device=None,             # IdealizedPresetDevice
        reinit_mode=reinit_mode,
        decay_factor=1.0,
    )
    device_config.correct_gradient_magnitudes = True
    device_config.forward_inject = False
    device_config.reinit_gain = 0.1
    rpu_config = PythonLRTTRPUConfig(device=device_config)

    std_config = SingleRPUConfig(device=IdealizedPresetDevice())

    model = AnalogSequential(
        AnalogLinear(INPUT_SIZE, HIDDEN_SIZE, rpu_config=rpu_config, bias=True),
        nn.Sigmoid(),
        AnalogLinear(HIDDEN_SIZE, OUTPUT_SIZE, rpu_config=std_config, bias=True),
        nn.LogSoftmax(dim=1),
    )
    return model.to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# Train / Eval
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.view(-1, INPUT_SIZE).to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        if torch.isnan(loss):
            return float('inf')
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate(model, val_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.view(-1, INPUT_SIZE).to(DEVICE)
            labels = labels.to(DEVICE)
            pred = model(images).argmax(dim=1)
            correct += pred.eq(labels).sum().item()
    return 100.0 * correct / len(val_loader.dataset)


def run_single(mode, rank, te, seed, train_loader, val_loader):
    """Run single training and return best accuracy."""
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)

    lr, tlr = get_hp(mode, rank, te)
    model = create_model(rank, te, mode)

    # Set transfer_lr on LRTT layers
    for layer in model:
        if hasattr(layer, 'analog_module') and hasattr(layer.analog_module, 'controller'):
            layer.analog_module.controller.transfer_lr = tlr

    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, criterion, optimizer)
        if loss == float('inf'):
            break
        acc = evaluate(model, val_loader)
        best_acc = max(best_acc, acc)
        scheduler.step()

    return best_acc, lr, tlr


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0,
                        help="1=fixed HP grid, 3=tlr sweep")
    parser.add_argument("--modes", type=str, default="reset,decay")
    parser.add_argument("--ranks", type=str, default="1,4,8,16,32,64")
    parser.add_argument("--tes", type=str, default="1,10,50,100,500,1000")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--tlr_values", type=str, default=None,
                        help="Comma-separated tlr values for Step 3 sweep")
    parser.add_argument("--output", type=str, default="reexp_results.json")
    args = parser.parse_args()

    # Step presets
    if args.step == 1:
        modes = ["reset", "decay"]
        ranks = [1, 4, 8, 16, 32, 64]
        tes = [1, 10, 50, 100, 500, 1000]
        seeds = [42, 43, 44]
        args.output = "reexp_step1_results.json"
    elif args.step == 3:
        modes = args.modes.split(",")
        ranks = [int(x) for x in args.ranks.split(",")]
        tes = [int(x) for x in args.tes.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]
        args.output = "reexp_step3_tlr_sweep.json"
    else:
        modes = args.modes.split(",")
        ranks = [int(x) for x in args.ranks.split(",")]
        tes = [int(x) for x in args.tes.split(",")]
        seeds = [int(x) for x in args.seeds.split(",")]

    print("=" * 70)
    print("MNIST 6T1C LRTT Re-Experiment (Formula-Based HP)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Modes:  {modes}")
    print(f"  Ranks:  {ranks}")
    print(f"  TEs:    {tes}")
    print(f"  Seeds:  {seeds}")
    total = len(modes) * len(ranks) * len(tes) * len(seeds)
    print(f"  Total runs: {total}")
    print(f"  Output: {args.output}")
    print("=" * 70)

    # Print HP table
    print("\n[HP Table]")
    print(f"{'Mode':<6s} {'Rank':>4s} {'TE':>5s} {'lr':>8s} {'tlr':>10s}")
    print("-" * 38)
    for mode in modes:
        for rank in ranks:
            for te in tes:
                lr, tlr = get_hp(mode, rank, te)
                print(f"{mode:<6s} {rank:>4d} {te:>5d} {lr:>8.4f} {tlr:>10.6f}")
    print()

    # Load data once
    train_loader, val_loader = load_data()

    # tlr values for sweep (Step 3) or single value (Step 1)
    if args.step == 3 and args.tlr_values:
        tlr_list = [float(x) for x in args.tlr_values.split(",")]
    else:
        tlr_list = [None]  # None = use default 0.005

    # Run
    results = []
    run_idx = 0
    total = len(modes) * len(ranks) * len(tes) * len(seeds) * len(tlr_list)
    t_start = time()

    for mode in modes:
        for rank in ranks:
            for te in tes:
                for tlr_val in tlr_list:
                    seed_accs = []
                    lr, tlr = get_hp(mode, rank, te, tlr_override=tlr_val)

                    for seed in seeds:
                        run_idx += 1
                        acc, _, _ = run_single(mode, rank, te, seed, train_loader, val_loader)
                        seed_accs.append(acc)
                        elapsed = time() - t_start
                        eta = elapsed / run_idx * (total - run_idx)
                        print(f"[{run_idx:>4d}/{total}] {mode:<6s} R={rank:>2d} TE={te:>4d} "
                              f"seed={seed} acc={acc:.2f}% "
                              f"({elapsed:.0f}s / ETA {eta:.0f}s)")

                    import numpy as np
                    mean_acc = np.mean(seed_accs)
                    std_acc = np.std(seed_accs)

                    result = {
                        "mode": mode,
                        "rank": rank,
                        "te": te,
                        "lr": lr,
                        "tlr": tlr,
                        "seed_accs": seed_accs,
                        "mean_acc": round(mean_acc, 2),
                        "std_acc": round(std_acc, 2),
                        "best_acc": round(max(seed_accs), 2),
                    }
                    results.append(result)

                    print(f"  -> mean={mean_acc:.2f}% +/- {std_acc:.2f}%")

    # Save JSON
    with open(args.output, "w") as f:
        json.dump({"config": {
            "modes": modes, "ranks": ranks, "tes": tes, "seeds": seeds,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "network": "MLP 784->256->10", "lifetime": 0,
            "hp_mode": "formula",
        }, "results": results}, f, indent=2)

    # Save CSV summary
    csv_path = args.output.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "rank", "te", "lr", "tlr",
                         "mean_acc", "std_acc", "best_acc"] +
                        [f"seed_{s}" for s in seeds])
        for r in results:
            writer.writerow([r["mode"], r["rank"], r["te"], r["lr"], r["tlr"],
                             r["mean_acc"], r["std_acc"], r["best_acc"]] +
                            r["seed_accs"])

    total_time = time() - t_start
    print(f"\nDone. {total} runs in {total_time/60:.1f} min")
    print(f"Results: {args.output}, {csv_path}")


if __name__ == "__main__":
    main()
