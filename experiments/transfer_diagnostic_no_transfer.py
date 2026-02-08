#!/usr/bin/env python3
"""Transfer diagnostic control experiment: TE=1000000 (no transfers)."""

import os

os.environ["LRTT_SILENT"] = "1"

import csv
import math
from time import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from aihwkit.optim import AnalogSGD

# Reuse everything from the main diagnostic
import experiments.transfer_diagnostic as td

DEVICE = td.DEVICE
BATCH_SIZE = td.BATCH_SIZE
EPOCHS = 30
LR = td.LR


def main() -> None:
    # Override TE to effectively disable transfers
    td.TE = 1000000

    print("=" * 70)
    print("LRTT Transfer Diagnostic — NO TRANSFER (TE=1000000)")
    print("=" * 70)
    print(f"Config: rank={td.RANK}, te={td.TE}, lr={LR}, tlr={td.TLR}")
    print(f"Epochs: {EPOCHS}, batch_size={BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print()

    train_loader, val_loader = td.load_data()
    print(f"Batches/epoch: {len(train_loader)}")
    print()

    model = td.create_model()

    optimizer = AnalogSGD(model.parameters(), lr=LR)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.NLLLoss()

    best_acc = 0.0
    rows: list[dict] = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time()
        model.train()
        for data, target in train_loader:
            data = data.to(DEVICE, non_blocking=True).view(data.shape[0], -1)
            target = target.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        val_acc = td.validate(model, val_loader)
        scheduler.step()
        best_acc = max(best_acc, val_acc)
        elapsed = time() - t0

        rows.append({"epoch": epoch, "val_acc": val_acc, "best_acc": best_acc})
        print(
            f"Epoch {epoch:2d}/{EPOCHS} | acc={val_acc:.2f}% (best={best_acc:.2f}%) | {elapsed:.1f}s"
        )

    # Save epoch CSV
    epoch_csv = "/root/LRTT/experiments/transfer_diagnostic_no_transfer_epochs.csv"
    with open(epoch_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_acc", "best_acc"])
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Results saved to: {epoch_csv}")
    print(f"Final accuracy: {val_acc:.2f}%, Best: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
