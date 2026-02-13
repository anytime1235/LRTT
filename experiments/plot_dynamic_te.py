#!/usr/bin/env python3
"""Plot dynamic TE diagnostic results."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# === Load data ===
CSV_DIR = "/data/LRTT_transformer/experiments/dynamic_te_results"
df = pd.read_csv(f"{CSV_DIR}/epoch_summary.csv")
tf = pd.read_csv(f"{CSV_DIR}/transfer_diagnostics.csv")

# === Style ===
COLORS = {
    "BASELINE": "#555555",
    "POWER_0.5": "#2196F3",
    "POWER_1.0": "#FF9800",
    "POWER_2.0": "#F44336",
}
LABELS = {
    "BASELINE": "Static (TE=100)",
    "POWER_0.5": "Dynamic p=0.5",
    "POWER_1.0": "Dynamic p=1.0",
    "POWER_2.0": "Dynamic p=2.0",
}
CONDITIONS = ["BASELINE", "POWER_0.5", "POWER_1.0", "POWER_2.0"]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# =====================================================================
# Figure 1: Main overview (2x3)
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Dynamic Transfer Every (TE) — MNIST Diagnostic Overview", fontsize=16, fontweight="bold", y=0.98)

# --- (0,0) Validation Accuracy ---
ax = axes[0, 0]
for cond in CONDITIONS:
    sub = df[df.condition == cond]
    ax.plot(sub.epoch, sub.val_acc, color=COLORS[cond], label=LABELS[cond], linewidth=2, alpha=0.9)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("(a) Validation Accuracy")
ax.legend(loc="lower right")
ax.set_ylim(78, 98)
# LR step markers
for ep in [10, 20, 30]:
    ax.axvline(ep, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

# --- (0,1) Transfer Every ---
ax = axes[0, 1]
for cond in CONDITIONS:
    sub = df[df.condition == cond]
    ax.plot(sub.epoch, sub.te, color=COLORS[cond], label=LABELS[cond], linewidth=2, alpha=0.9)
ax.set_xlabel("Epoch")
ax.set_ylabel("Transfer Every (samples)")
ax.set_title("(b) Transfer Every Schedule")
ax.set_yscale("log")
ax.set_yticks([100, 200, 400, 1000])
ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
ax.legend(loc="upper left")
for ep in [10, 20, 30]:
    ax.axvline(ep, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

# --- (0,2) Number of Transfers per epoch ---
ax = axes[0, 2]
for cond in CONDITIONS:
    sub = df[df.condition == cond]
    ax.plot(sub.epoch, sub.n_transfers, color=COLORS[cond], label=LABELS[cond], linewidth=2, alpha=0.9)
ax.set_xlabel("Epoch")
ax.set_ylabel("Transfers / Epoch")
ax.set_title("(c) Transfer Count per Epoch")
ax.legend(loc="upper right")
for ep in [10, 20, 30]:
    ax.axvline(ep, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

# --- (1,0) AB Magnitude ---
ax = axes[1, 0]
for cond in CONDITIONS:
    sub = df[df.condition == cond]
    ax.plot(sub.epoch, sub.mean_ab_mag, color=COLORS[cond], label=LABELS[cond], linewidth=2, alpha=0.9)
ax.set_xlabel("Epoch")
ax.set_ylabel("AB Magnitude (Frobenius)")
ax.set_title("(d) Mean AB Magnitude per Epoch")
ax.legend(loc="upper right")
for ep in [10, 20, 30]:
    ax.axvline(ep, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

# --- (1,1) Cosine Similarity ---
ax = axes[1, 1]
for cond in CONDITIONS:
    sub = df[df.condition == cond]
    ax.plot(sub.epoch, sub.mean_cosine_sim, color=COLORS[cond], label=LABELS[cond], linewidth=2, alpha=0.9)
ax.set_xlabel("Epoch")
ax.set_ylabel("Cosine Similarity")
ax.set_title("(e) Transfer Direction Accuracy (cos sim)")
ax.legend(loc="lower left")
ax.set_ylim(0, 1.05)
for ep in [10, 20, 30]:
    ax.axvline(ep, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

# --- (1,2) Delta Ratio ---
ax = axes[1, 2]
for cond in CONDITIONS:
    sub = df[df.condition == cond]
    ax.plot(sub.epoch, sub.mean_delta_ratio, color=COLORS[cond], label=LABELS[cond], linewidth=2, alpha=0.9)
ax.set_xlabel("Epoch")
ax.set_ylabel("ΔC / AB_mag")
ax.set_title("(f) Delta Ratio (1.0 = perfect)")
ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, linewidth=1)
ax.legend(loc="upper left")
ax.set_ylim(0, 9)
for ep in [10, 20, 30]:
    ax.axvline(ep, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{CSV_DIR}/fig1_overview.png", dpi=150, bbox_inches="tight")
print(f"Saved fig1_overview.png")


# =====================================================================
# Figure 2: Phase-based bar chart comparison
# =====================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
fig2.suptitle("Dynamic TE — Phase Comparison (Early / Mid / Late)", fontsize=16, fontweight="bold", y=0.98)

def get_phase(epoch):
    if epoch <= 10: return "Early\n(1-10)"
    elif epoch <= 20: return "Mid\n(11-20)"
    else: return "Late\n(21-30)"

df["phase"] = df.epoch.apply(get_phase)
phase_order = ["Early\n(1-10)", "Mid\n(11-20)", "Late\n(21-30)"]

metrics = [
    ("val_acc", "Validation Accuracy (%)", "(a) Accuracy by Phase"),
    ("mean_ab_mag", "AB Magnitude", "(b) AB Magnitude by Phase"),
    ("mean_cosine_sim", "Cosine Similarity", "(c) Cosine Similarity by Phase"),
    ("mean_delta_ratio", "ΔC / AB_mag", "(d) Delta Ratio by Phase"),
    ("mean_unchanged_ratio", "Unchanged Ratio", "(e) Unchanged Element Ratio by Phase"),
    ("n_transfers", "Total Transfers", "(f) Total Transfers by Phase"),
]

bar_width = 0.18
for idx, (col, ylabel, title) in enumerate(metrics):
    ax = axes2[idx // 3, idx % 3]
    x = np.arange(len(phase_order))

    for i, cond in enumerate(CONDITIONS):
        sub = df[df.condition == cond]
        if col == "n_transfers":
            vals = [sub[sub.phase == p][col].sum() for p in phase_order]
        else:
            vals = [sub[sub.phase == p][col].mean() for p in phase_order]
        bars = ax.bar(x + i * bar_width, vals, bar_width, label=LABELS[cond],
                      color=COLORS[cond], alpha=0.85, edgecolor="white", linewidth=0.5)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            fmt = f"{v:.1f}" if v >= 10 else f"{v:.2f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(vals),
                    fmt, ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(phase_order)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if idx == 0:
        ax.legend(loc="lower left", fontsize=8.5)
    if col == "mean_delta_ratio":
        ax.axhline(1.0, color="black", linestyle=":", alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(f"{CSV_DIR}/fig2_phase_bars.png", dpi=150, bbox_inches="tight")
print(f"Saved fig2_phase_bars.png")


# =====================================================================
# Figure 3: Per-transfer scatter (AB_mag vs cosine_sim, colored by epoch)
# =====================================================================
fig3, axes3 = plt.subplots(1, 4, figsize=(20, 5))
fig3.suptitle("Per-Transfer: AB Magnitude vs Cosine Similarity (color = epoch)", fontsize=14, fontweight="bold", y=1.02)

for i, cond in enumerate(CONDITIONS):
    ax = axes3[i]
    sub = tf[tf.condition == cond].copy()
    sc = ax.scatter(sub.ab_magnitude, sub.cosine_sim, c=sub.epoch, cmap="viridis",
                    s=15, alpha=0.7, edgecolors="none", vmin=1, vmax=30)
    ax.set_xlabel("AB Magnitude")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(LABELS[cond])
    ax.set_xlim(-0.2, 7)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.3, linewidth=0.8)

cb = fig3.colorbar(sc, ax=axes3[-1], label="Epoch", shrink=0.9)
plt.tight_layout()
fig3.savefig(f"{CSV_DIR}/fig3_scatter.png", dpi=150, bbox_inches="tight")
print(f"Saved fig3_scatter.png")


# =====================================================================
# Figure 4: Accuracy + TE dual-axis for each condition
# =====================================================================
fig4, axes4 = plt.subplots(1, 4, figsize=(20, 4.5))
fig4.suptitle("Accuracy & TE Schedule per Condition", fontsize=14, fontweight="bold", y=1.02)

for i, cond in enumerate(CONDITIONS):
    ax = axes4[i]
    sub = df[df.condition == cond]

    ax.plot(sub.epoch, sub.val_acc, color=COLORS[cond], linewidth=2, label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)", color=COLORS[cond])
    ax.set_ylim(78, 98)
    ax.tick_params(axis="y", labelcolor=COLORS[cond])
    ax.set_title(LABELS[cond])

    ax2 = ax.twinx()
    ax2.fill_between(sub.epoch, sub.te, alpha=0.15, color=COLORS[cond])
    ax2.plot(sub.epoch, sub.te, color=COLORS[cond], linewidth=1, linestyle="--", alpha=0.6, label="TE")
    ax2.set_ylabel("Transfer Every", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    if cond == "POWER_2.0":
        ax2.set_ylim(0, 1200)
    else:
        ax2.set_ylim(0, max(sub.te) * 1.3)

    for ep in [10, 20, 30]:
        ax.axvline(ep, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

plt.tight_layout()
fig4.savefig(f"{CSV_DIR}/fig4_acc_te_dual.png", dpi=150, bbox_inches="tight")
print(f"Saved fig4_acc_te_dual.png")


# =====================================================================
# Figure 5: Cumulative accuracy advantage over baseline
# =====================================================================
fig5, ax5 = plt.subplots(figsize=(10, 5))
ax5.set_title("Accuracy Advantage over Static Baseline", fontsize=14, fontweight="bold")

baseline_acc = df[df.condition == "BASELINE"].val_acc.values
for cond in ["POWER_0.5", "POWER_1.0", "POWER_2.0"]:
    sub = df[df.condition == cond]
    diff = sub.val_acc.values - baseline_acc
    ax5.plot(sub.epoch.values, diff, color=COLORS[cond], label=LABELS[cond], linewidth=2)
    ax5.fill_between(sub.epoch.values, diff, alpha=0.1, color=COLORS[cond])

ax5.axhline(0, color="black", linewidth=1, alpha=0.5)
ax5.set_xlabel("Epoch")
ax5.set_ylabel("Δ Accuracy vs Baseline (%)")
ax5.legend()
for ep in [10, 20, 30]:
    ax5.axvline(ep, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
ax5.annotate("LR×0.5", xy=(10, ax5.get_ylim()[1]*0.9), fontsize=9, color="gray", ha="center")
ax5.annotate("LR×0.25", xy=(20, ax5.get_ylim()[1]*0.9), fontsize=9, color="gray", ha="center")

plt.tight_layout()
fig5.savefig(f"{CSV_DIR}/fig5_advantage.png", dpi=150, bbox_inches="tight")
print(f"Saved fig5_advantage.png")

print(f"\nAll figures saved to {CSV_DIR}/")
