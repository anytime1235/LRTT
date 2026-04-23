"""Figure 6(d): Effective rank of ΔC (C - C_init) — last tile, by LRTT rank."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "BERT_SQUAD_LRTT_FINE"

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

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
})

fig, ax = plt.subplots(figsize=(5.5, 3.5))

for rank in sorted(RANK_FILES.keys()):
    path = RANK_FILES[rank]
    if not path.exists():
        print(f"SKIP: rank={rank}, {path} not found")
        continue

    with open(path) as f:
        d = json.load(f)
    steps_data = d["last_tile"]["steps"]
    steps = [s["step"] for s in steps_data if s.get("erank_C") is not None]
    erank_delta = [s["erank_C_delta"] for s in steps_data if s.get("erank_C") is not None]

    stride = max(1, len(steps) // 500)
    ax.plot(steps[::stride], erank_delta[::stride],
            color=COLORS[rank], linewidth=1.4, alpha=0.85,
            label=f'rank={rank}')

ax.axhline(y=599.26, color='gray', linestyle='--', linewidth=1.2, alpha=0.7,
           label='erank($C_{init}$)', zorder=2)
ax.set_xlabel('Training step')
ax.set_ylabel('Effective rank')
ax.set_title('Layer 11, attention output: erank($C - C_{init}$)')
ax.legend(fontsize=7, loc='lower right', ncol=4, framealpha=0.9, edgecolor='0.7',
          columnspacing=0.8, handlelength=1.5)
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

fig.tight_layout(pad=0.5)

OUT = '/root/LRTT/examples/bert/results/plots/fig6d_erank_delta.png'
fig.savefig(OUT, dpi=300, bbox_inches='tight')
fig.savefig(OUT.replace('.png', '.svg'), bbox_inches='tight')
print(f'Saved: {OUT}')
