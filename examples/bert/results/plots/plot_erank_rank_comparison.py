#!/usr/bin/env python3
"""Plot effective rank comparison across different LRTT ranks.

Compares erank(C) and erank(C - C_init) trajectories for rank=1,2,4,8,16,32,64
at abml=10, ab_dw_min=1.211e-4, c_dw_min=0.001953 (constantstepideal).
Each rank uses its own independently optimized hyperparameters.

Output: 2x2 plot (first_tile erank_C, first_tile erank_C_delta,
                   last_tile erank_C, last_tile erank_C_delta)
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "BERT_SQUAD_LRTT_FINE"
OUTPUT = Path(__file__).resolve().parent

RANK_FILES = {
    1:  RESULTS / "squad_diagnostic_log_20260421_144225_diag_rank1_T366.json",
    2:  RESULTS / "squad_diagnostic_log_20260421_144225_diag_rank2_T369.json",
    4:  RESULTS / "squad_diagnostic_log_20260421_144225_diag_rank4_T374.json",
    8:  RESULTS / "squad_diagnostic_log_20260421_144225_diag_rank8_T401.json",
    16: RESULTS / "squad_diagnostic_log_20260421_144225_diag_rank16_T426.json",
    32: RESULTS / "squad_diagnostic_log_te2_r32_onehot.json",
    64: RESULTS / "squad_diagnostic_log_20260421_144225_diag_rank64_T414.json",
}

RANK_F1 = {1: 82.46, 2: 82.85, 4: 83.31, 8: 83.98, 16: 84.31, 32: 84.79, 64: 84.60}

COLORS = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c", 8: "#d62728",
          16: "#9467bd", 32: "#e377c2", 64: "#7f7f7f"}


def load_erank(path, tile_key):
    with open(path) as f:
        d = json.load(f)
    steps_data = d[tile_key]["steps"]
    steps = [s["step"] for s in steps_data if s.get("erank_C") is not None]
    erank_C = [s["erank_C"] for s in steps_data if s.get("erank_C") is not None]
    erank_C_delta = [s["erank_C_delta"] for s in steps_data if s.get("erank_C") is not None]
    return steps, erank_C, erank_C_delta


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Effective Rank by LRTT Rank (abml=10, constantstepideal, per-rank optimized)", fontsize=14)

    for tile_idx, tile_key in enumerate(["first_tile", "last_tile"]):
        tile_label = "First Tile" if tile_key == "first_tile" else "Last Tile"

        ax_c = axes[tile_idx, 0]
        ax_d = axes[tile_idx, 1]

        for rank in sorted(RANK_FILES.keys()):
            path = RANK_FILES[rank]
            if not path.exists():
                print(f"SKIP: rank={rank}, {path} not found")
                continue

            steps, erank_C, erank_C_delta = load_erank(path, tile_key)
            label = f"rank={rank} (F1={RANK_F1[rank]:.1f})"
            color = COLORS[rank]

            # Subsample for cleaner plot
            n = len(steps)
            stride = max(1, n // 500)
            s_steps = steps[::stride]
            s_erank_C = erank_C[::stride]
            s_erank_delta = erank_C_delta[::stride]

            ax_c.plot(s_steps, s_erank_C, label=label, color=color, alpha=0.8, linewidth=1.0)
            ax_d.plot(s_steps, s_erank_delta, label=label, color=color, alpha=0.8, linewidth=1.0)

        ax_c.set_xlabel("Step")
        ax_c.set_ylabel("Effective Rank")
        ax_c.set_title(f"{tile_label}: erank(C)")
        ax_c.legend(fontsize=7, loc="best")
        ax_c.grid(True, alpha=0.3)

        ax_d.set_xlabel("Step")
        ax_d.set_ylabel("Effective Rank")
        ax_d.set_title(f"{tile_label}: erank(C - C_init)")
        ax_d.legend(fontsize=7, loc="best")
        ax_d.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = OUTPUT / "erank_rank_comparison.png"
    out_svg = OUTPUT / "erank_rank_comparison.svg"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")


if __name__ == "__main__":
    main()
