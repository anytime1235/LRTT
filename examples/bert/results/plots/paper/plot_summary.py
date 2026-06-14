"""Summary plots for analog fine-tuning results.

Reads data from summary_data.json and generates plots.
Usage: python plot_summary.py
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "summary_data.json")

with open(DATA_PATH) as f:
    DATA = json.load(f)

# ── Plotting ──────────────────────────────────────────────────────────
def plot_bar(key, data, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(data["methods"], data["f1"],
                  color=data["colors"], width=1.0)

    ax.set_xlabel(data["xlabel"])
    ax.set_ylabel("Best F1 Score")
    ax.set_title(data["title"])

    lo = min(data["f1"]) - 5
    hi = max(data["f1"]) + 5
    ax.set_ylim(max(0, lo - lo % 5), min(100, hi + (5 - hi % 5)))

    for bar, score in zip(bars, data["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{score:.2f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, f"summary_{key}.svg")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_grouped_bar(key, data, output_dir):
    import numpy as np
    groups = data["groups"]
    bars = data["bars"]
    n_groups = len(groups)
    n_bars = len(bars)
    width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, bar in enumerate(bars):
        offset = (i - (n_bars - 1) / 2) * width
        rects = ax.bar(x + offset, bar["f1"], width, label=bar["label"],
                       color=bar["color"], alpha=0.9)
        for rect, score in zip(rects, bar["f1"]):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.3,
                    f"{score:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=8)

    ax.set_xlabel(data["xlabel"])
    ax.set_ylabel("Best F1 Score")
    ax.set_title(data["title"])
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()

    all_f1 = [v for bar in bars for v in bar["f1"]]
    lo = min(all_f1) - 5
    hi = max(all_f1) + 5
    ax.set_ylim(max(0, lo - lo % 5), min(100, hi + (5 - hi % 5)))

    plt.tight_layout()
    path = os.path.join(output_dir, f"summary_{key}.svg")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_line(key, data, output_dir):
    import numpy as np
    lines = data["lines"]

    # Dual x-axis if 2 lines with same number of points but different bit values
    if len(lines) == 2 and len(lines[0]["bits"]) == len(lines[1]["bits"]):
        n = len(lines[0]["bits"])
        x = np.arange(n)
        fig, ax1 = plt.subplots(figsize=(7, 4.5))

        for line in lines:
            ax1.plot(x, line["f1"], label=line["label"],
                     color=line["color"], marker=line["marker"], linewidth=2, markersize=6)
            for i, (b, f) in enumerate(zip(line["bits"], line["f1"])):
                offset_y = 8 if line is lines[0] else -14
                ax1.annotate(f"{f:.1f}", (x[i], f), textcoords="offset points",
                             xytext=(0, offset_y), ha="center", fontsize=7, color=line["color"])

        # Bottom x-axis: second line
        ax1.set_xticks(x)
        ax1.set_xticklabels(lines[1]["bits"])
        ax1.set_xlabel(f'{lines[1]["label"]} bits', color=lines[1]["color"])

        # Top x-axis: first line
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(x)
        ax2.set_xticklabels(lines[0]["bits"])
        ax2.set_xlabel(f'{lines[0]["label"]} bits', color=lines[0]["color"])

        ax1.set_ylabel("Best F1 Score")
        ax1.set_title(data["title"], pad=25)
        ax1.legend(loc="lower right")
    else:
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        for line in lines:
            ax1.plot(line["bits"], line["f1"], label=line["label"],
                     color=line["color"], marker=line["marker"], linewidth=2, markersize=6)
            for b, f in zip(line["bits"], line["f1"]):
                ax1.annotate(f"{f:.1f}", (b, f), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=7)
        ax1.set_xlabel(data["xlabel"])
        ax1.set_ylabel("Best F1 Score")
        ax1.set_title(data["title"])
        ax1.set_xticks(lines[0]["bits"])
        ax1.legend(loc="lower right")

    plt.tight_layout()
    path = os.path.join(output_dir, f"summary_{key}.svg")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_sweep(key, data, output_dir):
    """Plot x-y sweep data (noise_ratio, asymmetry_ratio, alpha_transfer_ratio)."""
    points = data["data"]
    if not points:
        return
    x_key = [k for k in points[0] if k != "f1"][0]
    xs = [p[x_key] for p in points]
    ys = [p["f1"] for p in points]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, color="#4C72B0", marker="o", linewidth=2, markersize=6)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)

    ax.set_xlabel(data["xlabel"])
    ax.set_ylabel(data["ylabel"])
    ax.set_title(data["title"])

    if key != "noise_ratio_sweep":
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        ax.set_xticks(xs)

    if key == "alpha_transfer_ratio_sweep":
        ax.text(0.98, 0.02, "transfer rate is fixed",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, fontstyle="italic", color="gray")

    plt.tight_layout()
    path = os.path.join(output_dir, f"summary_{key}.svg")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_erank(key, data, output_dir):
    """Plot effective rank dynamics for first and last tiles."""
    import numpy as np
    tiles = data["data"]
    if not tiles:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for ax, (tile_key, tile_label) in zip(axes, [("first_tile", "First Tile"), ("last_tile", "Last Tile")]):
        tile = tiles[tile_key]
        steps = tile["steps"]
        is_transfer = tile["is_transfer"]

        ax.plot(steps, tile["erank_C"], label="erank(C)",
                color="green", alpha=0.8, linewidth=1.0)
        ax.plot(steps, tile["erank_C_delta"], label="erank(C - C_init)",
                color="blue", alpha=0.8, linewidth=1.0)

        # Mark transfer steps
        t_steps = [s for s, t in zip(steps, is_transfer) if t]
        if t_steps:
            ax.axvline(t_steps[0], color="gray", alpha=0.3, linewidth=0.5, label="transfer")
            for ts in t_steps[1:]:
                ax.axvline(ts, color="gray", alpha=0.3, linewidth=0.5)

        # Show nominal rank as horizontal reference line
        rank = data.get("params", {}).get("rank")
        if rank is not None:
            ax.axhline(rank, color="red", linestyle=":", linewidth=1, alpha=0.4, label=f"rank={rank}")

        ax.set_xlim(steps[0], steps[-1])
        ax.set_xlabel("Step")
        ax.set_ylabel("Effective rank")
        ax.set_title(tile["name"])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(data["title"], fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f"summary_{key}.svg")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Plot type dispatch ───────────────────────────────────────────────
SWEEP_KEYS = {"noise_ratio_sweep", "asymmetry_ratio_sweep", "alpha_transfer_ratio_sweep"}

if __name__ == "__main__":
    for key, data in DATA.items():
        if "f1" in data and not data["f1"]:
            print(f"Skipped: {key} (empty data)")
            continue
        if key in SWEEP_KEYS:
            plot_sweep(key, data, SCRIPT_DIR)
        elif key == "erank_dynamics":
            plot_erank(key, data, SCRIPT_DIR)
        elif "lines" in data:
            plot_line(key, data, SCRIPT_DIR)
        elif "bars" in data:
            plot_grouped_bar(key, data, SCRIPT_DIR)
        elif "methods" in data and data["methods"]:
            plot_bar(key, data, SCRIPT_DIR)
        else:
            print(f"Skipped: {key} (no plot function yet)")
