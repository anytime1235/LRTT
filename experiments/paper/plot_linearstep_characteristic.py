#!/usr/bin/env python
"""Plot LinearStepDevice dG vs G (step size vs conductance) for different gamma levels.

Uses aihwkit's compute_pulse_response + compute_pulse_statistics to simulate
actual device behavior and extract dG-G curves.

Compares 4 gamma levels (G0 ~ G100) at 14-bit dw_min, noise-free.
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.parameters.enums import (
    BoundManagementType, NoiseManagementType, PulseType, WeightNoiseType,
)
from aihwkit.simulator.parameters.io import IOParameters
from aihwkit.utils.visualization import (
    compute_pulse_response,
    compute_pulse_statistics,
    get_tile_for_plotting,
    plot_device_compact,
)


# ============================================================================
# Parameters
# ============================================================================
DW_MIN_14BIT = 2.0 / (2 ** 14)  # ~1.22e-4
N_STEPS = 16384  # full sweep for 14-bit

GAMMA_LEVELS = [
    {"label": "G0 (ConstantStep)",  "gamma_up": 0.0,      "gamma_down": 0.0,
     "color": "#2196F3", "ls": "-"},
    {"label": "G50 (50% of 6T1C)", "gamma_up": -0.084,   "gamma_down": 0.071,
     "color": "#FF9800", "ls": "--"},
    {"label": "G100 (full 6T1C)",  "gamma_up": -0.1678,  "gamma_down": 0.1410,
     "color": "#F44336", "ls": "-."},
    {"label": "G300 (300% 6T1C)",  "gamma_up": -0.5034,  "gamma_down": 0.4230,
     "color": "#795548", "ls": "-"},
]

COLORS = [l["color"] for l in GAMMA_LEVELS]
LINESTYLES = [l["ls"] for l in GAMMA_LEVELS]


def make_device(gamma_up, gamma_down):
    """Create a noise-free LinearStepDevice at 14-bit resolution."""
    return LinearStepDevice(
        dw_min=DW_MIN_14BIT,
        w_max=1.0,
        w_min=-1.0,
        up_down=0.0,
        mult_noise=False,
        gamma_up=gamma_up,
        gamma_down=gamma_down,
        # All noise off
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        gamma_up_dtod=0.0,
        gamma_down_dtod=0.0,
        write_noise_std=0.0,
    )


def simulate_dg_g(device, n_steps, n_traces=1, num_nodes=200):
    """Simulate device and compute dG-G statistics using aihwkit internals.

    Returns:
        w_nodes: weight positions [num_nodes]
        dw_mean_up: mean up step [num_nodes, n_traces]
        dw_mean_down: mean down step [num_nodes, n_traces]
    """
    io_pars = IOParameters(
        out_noise=0.0, w_noise=0.0, inp_res=-1.0, out_bound=100.0, out_res=-1.0,
        bound_management=BoundManagementType.NONE,
        noise_management=NoiseManagementType.NONE,
        w_noise_type=WeightNoiseType.ADDITIVE_CONSTANT,
    )
    rpu_config = SingleRPUConfig(device=device, forward=io_pars)

    # 1 full loop: up then down
    total_iters = 2 * n_steps
    direction = np.sign(np.sin(np.pi * (np.arange(total_iters) + 1) / n_steps))

    analog_tile = get_tile_for_plotting(rpu_config, n_traces, use_cuda=False, noise_free=True)
    w_trace = compute_pulse_response(analog_tile, direction, use_forward=False)

    w_nodes = np.linspace(w_trace.min(), w_trace.max(), num_nodes)
    dw_mean_up, _ = compute_pulse_statistics(w_nodes, w_trace, direction, up_direction=True)
    dw_mean_down, _ = compute_pulse_statistics(w_nodes, w_trace, direction, up_direction=False)

    return w_nodes, dw_mean_up.reshape(-1, n_traces), dw_mean_down.reshape(-1, n_traces)


# ============================================================================
# Main
# ============================================================================
print("=" * 70)
print("Simulating dG vs G for 4 gamma levels using aihwkit engine...")
print(f"  dw_min (14-bit) = {DW_MIN_14BIT:.6e}")
print(f"  n_steps = {N_STEPS}")
print("=" * 70)

results = {}
for level in GAMMA_LEVELS:
    print(f"  Simulating {level['label']}...")
    dev = make_device(level["gamma_up"], level["gamma_down"])
    w_nodes, dw_up, dw_down = simulate_dg_g(dev, N_STEPS, n_traces=1, num_nodes=200)
    results[level["label"]] = {
        "w_nodes": w_nodes,
        "dw_up": dw_up[:, 0],
        "dw_down": dw_down[:, 0],
    }
    print(f"    done. w range: [{w_nodes.min():.3f}, {w_nodes.max():.3f}]")


# ============================================================================
# Figure 1: dG vs G overlay — UP and DOWN
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1, 1, 1]})

# --- (a) UP pulse dG vs G ---
ax = axes[0]
for i, level in enumerate(GAMMA_LEVELS):
    r = results[level["label"]]
    ax.plot(r["w_nodes"], r["dw_up"] / DW_MIN_14BIT,
            color=level["color"], ls=level["ls"], lw=2.2, label=level["label"])
ax.axhline(y=1.0, color="gray", ls=":", lw=1, alpha=0.5)
ax.axvline(x=0.0, color="gray", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Weight G (conductance)", fontsize=12)
ax.set_ylabel("dG / dw_min  (step multiplier)", fontsize=12)
ax.set_title("(a) UP pulse:  dG vs G", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(-1.05, 1.05)
ax.grid(True, alpha=0.3)

# --- (b) DOWN pulse |dG| vs G ---
ax = axes[1]
for i, level in enumerate(GAMMA_LEVELS):
    r = results[level["label"]]
    ax.plot(r["w_nodes"], np.abs(r["dw_down"]) / DW_MIN_14BIT,
            color=level["color"], ls=level["ls"], lw=2.2, label=level["label"])
ax.axhline(y=1.0, color="gray", ls=":", lw=1, alpha=0.5)
ax.axvline(x=0.0, color="gray", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Weight G (conductance)", fontsize=12)
ax.set_ylabel("|dG| / dw_min  (step multiplier)", fontsize=12)
ax.set_title("(b) DOWN pulse:  |dG| vs G", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(-1.05, 1.05)
ax.grid(True, alpha=0.3)

# --- (c) Asymmetry: |UP| / |DOWN| ---
ax = axes[2]
for i, level in enumerate(GAMMA_LEVELS):
    r = results[level["label"]]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(r["dw_up"]) / np.abs(r["dw_down"])
        ratio[~np.isfinite(ratio)] = np.nan
    ax.plot(r["w_nodes"], ratio,
            color=level["color"], ls=level["ls"], lw=2.2, label=level["label"])
ax.axhline(y=1.0, color="gray", ls=":", lw=1, alpha=0.5)
ax.axvline(x=0.0, color="gray", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Weight G (conductance)", fontsize=12)
ax.set_ylabel("|dG_up| / |dG_down|", fontsize=12)
ax.set_title("(c) UP/DOWN asymmetry", fontsize=14, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(-1.05, 1.05)
ax.grid(True, alpha=0.3)

fig.suptitle("LinearStepDevice (6T1C):  Step Size vs. Conductance\n"
             f"14-bit (dw_min={DW_MIN_14BIT:.2e}), noise-free, simulated with aihwkit",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
out1 = os.path.join(SCRIPT_DIR, "linearstep_dG_vs_G.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out1}")
plt.close()


# ============================================================================
# Figure 2: ACS double-column — (a) gamma sweep, (b) noise sweep
# ============================================================================
print("\nSimulating 2-panel figure (ACS format)...")

# --- ACS style setup ---
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "savefig.dpi": 600,
    "figure.dpi": 150,
})

io_pars_clean = IOParameters(
    out_noise=0.0, w_noise=0.0, inp_res=-1.0, out_bound=100.0, out_res=-1.0,
    bound_management=BoundManagementType.NONE,
    noise_management=NoiseManagementType.NONE,
    w_noise_type=WeightNoiseType.ADDITIVE_CONSTANT,
)

total_iters = 2 * N_STEPS * 1  # 1 full loop
direction = np.sign(np.sin(np.pi * (np.arange(total_iters) + 1) / N_STEPS))

colors = ["#0C5DA5", "#FF9500", "#00B945", "#FF2C00"]  # Nature palette

# --- 6T1C baseline noise parameters (r=1.0) ---
SIXT1C_NOISE = {
    "dw_min_std": 0.3,
    "dw_min_dtod": 0.1,
    "up_down_dtod": 0.01,
    "w_max_dtod": 0.05,
    "w_min_dtod": 0.05,
    "gamma_up_dtod": 0.05,
    "gamma_down_dtod": 0.05,
    "write_noise_std": 0.0,
}

# 6T1C gamma (fixed for noise sweep)
SIXT1C_GAMMA_UP = -0.1678
SIXT1C_GAMMA_DOWN = 0.1410

NOISE_RATIOS = [0, 0.33, 1.0, 2.0]   # r_noise levels
GAMMA_RATIOS = [0, 0.5, 1.0, 2.0]    # r_gamma levels


def make_device_noisy(gamma_up, gamma_down, noise_ratio):
    """Create LinearStepDevice with scaled noise at given gamma."""
    r = noise_ratio
    return LinearStepDevice(
        dw_min=DW_MIN_14BIT,
        w_max=1.0,
        w_min=-1.0,
        up_down=0.0,
        mult_noise=False,
        gamma_up=gamma_up,
        gamma_down=gamma_down,
        dw_min_std=SIXT1C_NOISE["dw_min_std"] * r,
        dw_min_dtod=SIXT1C_NOISE["dw_min_dtod"] * r,
        up_down_dtod=SIXT1C_NOISE["up_down_dtod"] * r,
        w_max_dtod=SIXT1C_NOISE["w_max_dtod"] * r,
        w_min_dtod=SIXT1C_NOISE["w_min_dtod"] * r,
        gamma_up_dtod=SIXT1C_NOISE["gamma_up_dtod"] * r,
        gamma_down_dtod=SIXT1C_NOISE["gamma_down_dtod"] * r,
        write_noise_std=SIXT1C_NOISE["write_noise_std"] * r,
    )


# ACS double-column = 7.0 in wide
fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.25))

# ---- Panel (a): Gamma sweep (noise = 0) ----
print("  Panel (a): gamma sweep...")
for i, r_g in enumerate(GAMMA_RATIOS):
    g_up = SIXT1C_GAMMA_UP * r_g
    g_dn = SIXT1C_GAMMA_DOWN * r_g
    device = make_device(g_up, g_dn)
    rpu_config = SingleRPUConfig(device=device, forward=io_pars_clean)
    tile = get_tile_for_plotting(rpu_config, n_traces=1, use_cuda=False, noise_free=True)
    w_trace = compute_pulse_response(tile, direction, use_forward=False).reshape(-1)

    w_min_a, w_max_a = w_trace.min(), w_trace.max()
    w_norm = (w_trace - w_min_a) / (w_max_a - w_min_a) * 2.0 - 1.0
    x_norm = np.arange(total_iters) / total_iters

    pct = int(r_g * 100)
    lbl = f"$r_\\gamma$ = {r_g:.1f}" if r_g > 0 else "$r_\\gamma$ = 0 (ideal)"
    ax_a.plot(x_norm, w_norm, color=colors[i], ls="-", lw=1.0, label=lbl)

ax_a.set_xlabel("Normalized pulse number")
ax_a.set_ylabel("Normalized weight")
ax_a.set_title("(a) Nonlinearity sweep (noise-free)", fontsize=8)
ax_a.set_xlim(0, 1)
ax_a.set_ylim(-1.08, 1.08)
ax_a.legend(loc="upper right", frameon=True, edgecolor="0.8",
            fancybox=False, handlelength=1.8, labelspacing=0.3)
ax_a.minorticks_on()
ax_a.grid(False)

# ---- Panel (b): Noise sweep (gamma = 6T1C 100%) ----
print("  Panel (b): noise sweep (gamma fixed at 6T1C)...")
N_NOISE_TRACES = 5  # show D-to-D spread

for i, r_n in enumerate(NOISE_RATIOS):
    device = make_device_noisy(SIXT1C_GAMMA_UP, SIXT1C_GAMMA_DOWN, r_n)
    rpu_config = SingleRPUConfig(device=device, forward=io_pars_clean)

    if r_n == 0:
        # Noise-free: single clean trace
        tile = get_tile_for_plotting(rpu_config, n_traces=1, use_cuda=False, noise_free=True)
        w_trace = compute_pulse_response(tile, direction, use_forward=False).reshape(-1)
        w_min_a, w_max_a = w_trace.min(), w_trace.max()
        w_norm = (w_trace - w_min_a) / (w_max_a - w_min_a) * 2.0 - 1.0
        x_norm = np.arange(total_iters) / total_iters
        ax_b.plot(x_norm, w_norm, color=colors[i], ls="-", lw=1.0,
                  label="$r_n$ = 0 (noise-free)")
    else:
        # Noisy: multiple traces to show D-to-D variation
        tile = get_tile_for_plotting(rpu_config, n_traces=N_NOISE_TRACES,
                                     use_cuda=False, noise_free=False)
        w_trace = compute_pulse_response(tile, direction, use_forward=False)
        # w_trace: [total_iters, 1, N_NOISE_TRACES] → reshape
        w_trace = w_trace.reshape(total_iters, N_NOISE_TRACES)
        x_norm = np.arange(total_iters) / total_iters

        for t in range(N_NOISE_TRACES):
            tr = w_trace[:, t]
            w_min_a, w_max_a = tr.min(), tr.max()
            if w_max_a - w_min_a < 1e-10:
                continue
            w_n = (tr - w_min_a) / (w_max_a - w_min_a) * 2.0 - 1.0
            lbl = f"$r_n$ = {r_n:.2f}" if t == 0 else None
            ax_b.plot(x_norm, w_n, color=colors[i], ls="-", lw=0.5,
                      alpha=0.7, label=lbl)

ax_b.set_xlabel("Normalized pulse number")
ax_b.set_ylabel("Normalized weight")
ax_b.set_title("(b) Noise sweep ($\\gamma$ = 6T1C)", fontsize=8)
ax_b.set_xlim(0, 1)
ax_b.set_ylim(-1.08, 1.08)
ax_b.legend(loc="upper right", frameon=True, edgecolor="0.8",
            fancybox=False, handlelength=1.8, labelspacing=0.3)
ax_b.minorticks_on()
ax_b.grid(False)

out2 = os.path.join(SCRIPT_DIR, "linearstep_characteristic.png")
fig2.tight_layout(pad=0.4)
fig2.savefig(out2, dpi=600, bbox_inches="tight")
out2_pdf = os.path.join(SCRIPT_DIR, "linearstep_characteristic.pdf")
fig2.savefig(out2_pdf, bbox_inches="tight")
print(f"Saved: {out2}")
print(f"Saved: {out2_pdf}")
plt.close()

# Reset rcParams
plt.rcdefaults()


# ============================================================================
# Figure 3: G100 full compact with dG-G side panels
# ============================================================================
g100 = GAMMA_LEVELS[3]
device_g100 = make_device(g100["gamma_up"], g100["gamma_down"])
out3 = os.path.join(SCRIPT_DIR, "linearstep_g100_dG_G.png")
plot_device_compact(device_g100, w_noise=0.0, n_steps=N_STEPS, n_traces=3, n_loops=2)
plt.suptitle(f"G100 (full 6T1C): γ_up={g100['gamma_up']}, γ_dn={g100['gamma_down']}\n"
             f"14-bit, dw_min={DW_MIN_14BIT:.2e}, noise-free",
             fontsize=13, fontweight="bold")
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close()


# ============================================================================
# Summary table
# ============================================================================
print("\n" + "=" * 70)
print("Step Multiplier Summary (from simulation)")
print("=" * 70)

for level in GAMMA_LEVELS:
    r = results[level["label"]]
    w = r["w_nodes"]
    # Find multiplier near w=-1, w=0, w=+1
    idx_m1 = np.argmin(np.abs(w - (-0.95)))
    idx_0 = np.argmin(np.abs(w - 0.0))
    idx_p1 = np.argmin(np.abs(w - 0.95))

    up_m1 = r["dw_up"][idx_m1] / DW_MIN_14BIT
    up_0 = r["dw_up"][idx_0] / DW_MIN_14BIT
    up_p1 = r["dw_up"][idx_p1] / DW_MIN_14BIT
    dn_m1 = abs(r["dw_down"][idx_m1]) / DW_MIN_14BIT
    dn_0 = abs(r["dw_down"][idx_0]) / DW_MIN_14BIT
    dn_p1 = abs(r["dw_down"][idx_p1]) / DW_MIN_14BIT

    print(f"\n  {level['label']}")
    print(f"    UP  step: @w=-0.95: {up_m1:.4f}x  @w=0: {up_0:.4f}x  @w=+0.95: {up_p1:.4f}x")
    print(f"    DN  step: @w=-0.95: {dn_m1:.4f}x  @w=0: {dn_0:.4f}x  @w=+0.95: {dn_p1:.4f}x")

print("\nDone.")
