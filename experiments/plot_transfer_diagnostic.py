#!/usr/bin/env python3
"""Plot hybrid vs decay transfer diagnostic comparison."""

import csv

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def load_epoch_csv(path: str) -> dict[str, list[float]]:
    """Load epoch CSV into dict of lists."""
    data: dict[str, list[float]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except ValueError:
                    data[key].append(float("nan"))
    return data


# Load data
hybrid = load_epoch_csv("/root/LRTT/experiments/transfer_diagnostic_epochs.csv")
decay = load_epoch_csv("/root/LRTT/experiments/transfer_diagnostic_decay_epochs.csv")

epochs_h = np.array(hybrid["epoch"])
epochs_d = np.array(decay["epoch"])

fig, axes = plt.subplots(3, 2, figsize=(14, 13))
fig.suptitle("LRTT Transfer Diagnostic: Hybrid vs Decay Reinit", fontsize=15, fontweight="bold")

colors = {"hybrid": "#d62728", "decay": "#1f77b4"}
lw = 2.0

# --- (0,0) Validation Accuracy ---
ax = axes[0, 0]
ax.plot(epochs_h, hybrid["val_acc"], "-o", color=colors["hybrid"], lw=lw, ms=4, label="Hybrid")
ax.plot(epochs_d, decay["val_acc"], "-s", color=colors["decay"], lw=lw, ms=4, label="Decay")
ax.set_ylabel("Validation Accuracy (%)")
ax.set_xlabel("Epoch")
ax.set_title("Validation Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# --- (0,1) AB Magnitude (transfer signal size) ---
ax = axes[0, 1]
ax.plot(
    epochs_h, hybrid["ab_magnitude_mean"], "-o", color=colors["hybrid"], lw=lw, ms=4, label="Hybrid"
)
ax.plot(
    epochs_d, decay["ab_magnitude_mean"], "-s", color=colors["decay"], lw=lw, ms=4, label="Decay"
)
ax.set_ylabel("‖tlr × A@B‖_F")
ax.set_xlabel("Epoch")
ax.set_title("AB Transfer Signal Magnitude")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)

# --- (1,0) Cosine Similarity ---
ax = axes[1, 0]
ax.plot(
    epochs_h, hybrid["cosine_sim_mean"], "-o", color=colors["hybrid"], lw=lw, ms=4, label="Hybrid"
)
ax.plot(epochs_d, decay["cosine_sim_mean"], "-s", color=colors["decay"], lw=lw, ms=4, label="Decay")
ax.axhline(y=0.0, color="gray", ls="--", lw=1, alpha=0.5, label="Random (orthogonal)")
ax.set_ylabel("Cosine Similarity")
ax.set_xlabel("Epoch")
ax.set_title("Cosine Sim (AB signal vs actual ΔC)")
ax.set_ylim(-0.05, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)

# --- (1,1) Signal Ratio (changed elements) ---
ax = axes[1, 1]
ax.plot(
    epochs_h,
    hybrid["signal_ratio_changed_mean"],
    "-o",
    color=colors["hybrid"],
    lw=lw,
    ms=4,
    label="Hybrid",
)
ax.plot(
    epochs_d,
    decay["signal_ratio_changed_mean"],
    "-s",
    color=colors["decay"],
    lw=lw,
    ms=4,
    label="Decay",
)
ax.axhline(y=0.5, color="gray", ls="--", lw=1, alpha=0.5, label="50% (half noise)")
ax.set_ylabel("Signal Ratio")
ax.set_xlabel("Epoch")
ax.set_title("Signal Ratio (energy in AB direction / total ΔC)")
ax.set_ylim(0, 1.0)
ax.legend()
ax.grid(True, alpha=0.3)

# --- (2,0) Delta Ratio (noise amplification) ---
ax = axes[2, 0]
ax.plot(
    epochs_h, hybrid["delta_ratio_mean"], "-o", color=colors["hybrid"], lw=lw, ms=4, label="Hybrid"
)
ax.plot(
    epochs_d, decay["delta_ratio_mean"], "-s", color=colors["decay"], lw=lw, ms=4, label="Decay"
)
ax.axhline(y=1.0, color="gray", ls="--", lw=1, alpha=0.5, label="Ideal (no noise)")
ax.set_ylabel("‖ΔC‖ / ‖AB signal‖")
ax.set_xlabel("Epoch")
ax.set_title("Delta Ratio (noise amplification)")
ax.legend()
ax.grid(True, alpha=0.3)

# --- (2,1) C Tile Update Rate ---
ax = axes[2, 1]
update_h = [1.0 - v for v in hybrid["unchanged_elem_ratio_mean"]]
update_d = [1.0 - v for v in decay["unchanged_elem_ratio_mean"]]
ax.plot(
    epochs_h, np.array(update_h) * 100, "-o", color=colors["hybrid"], lw=lw, ms=4, label="Hybrid"
)
ax.plot(epochs_d, np.array(update_d) * 100, "-s", color=colors["decay"], lw=lw, ms=4, label="Decay")
ax.set_ylabel("Updated Elements (%)")
ax.set_xlabel("Epoch")
ax.set_title("C Tile Update Rate per Transfer")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = "/root/LRTT/experiments/transfer_diagnostic_comparison.png"
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
