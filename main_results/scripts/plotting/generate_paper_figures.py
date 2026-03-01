"""
generate_paper_figures.py — Analog BERT ADC Sweep Paper Figures

Generates publication-quality figures (PDF + PNG 300dpi + NPZ + JSON)
from ADC sweep diagnostic experiment results.

Usage:
    python generate_paper_figures.py [--data-dir DIR] [--out-dir DIR] [--only FIGNAME ...]

Output layout:
    plots/main/  — fig_main1_*, fig_main2_*, fig_main3_*
    plots/supp/  — fig_s1_*, fig_s2_*, fig_s3_*, fig_s4_*
"""

# ─────────────────────────────────────────────────────────────
# Section 1: imports, backend=Agg, rcParams
# ─────────────────────────────────────────────────────────────
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────
# Section 2: constants
# ─────────────────────────────────────────────────────────────
SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]
ADC_LIST = [4, 6, 8, 10, 12]

SUBLAYER_COLORS = {
    "Q": "#4C72B0", "K": "#DD8452", "V": "#55A868",
    "O": "#C44E52", "FFN1": "#9467BD", "FFN2": "#8C564B",
}
SUBLAYER_MARKERS = {
    "Q": "o", "K": "s", "V": "^", "O": "D", "FFN1": "p", "FFN2": "h",
}
ADC_COLORS = dict(zip(ADC_LIST, ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]))

# Global vmin/vmax for S1 consistent scales
S1_SCALES = {
    "mac_snr_db":        {"vmin": -0.60, "vmax": 42.11, "cmap": "RdYlGn", "log": False},
    "mac_nmse":          {"vmin": 1e-4,  "vmax": 0.590,  "cmap": "YlOrRd", "log": True},
    "cosine":            {"vmin": 0.64,  "vmax": 1.00,   "cmap": "RdYlGn", "log": False},
    "out_clip_ratio":    {"vmin": 0.0,   "vmax": 0.0018, "cmap": "Reds",   "log": False},
    "ref_deadzone_ratio":{"vmin": 2e-4,  "vmax": 0.990,  "cmap": "YlOrRd_r","log": True},
}

WARNINGS = []  # global warning list

# ─────────────────────────────────────────────────────────────
# Section 3: utilities
# ─────────────────────────────────────────────────────────────

def save_figure(fig, basename, outdir, npz_data: dict, meta: dict):
    """Save PDF + PNG(300dpi) + NPZ + JSON."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{basename}.png", dpi=300, bbox_inches="tight")
    np.savez_compressed(outdir / f"{basename}.npz", **npz_data)
    with open(outdir / f"{basename}.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved → {outdir}/{basename}.[pdf|png|npz|json]")


def make_meta(files_read: list, key_config: dict) -> dict:
    """Build JSON metadata dict."""
    return {
        "git_hash": "N/A",
        "timestamp": datetime.now().isoformat(),
        "files_read": [str(p) for p in files_read],
        "key_config": key_config,
    }


def load_module_summary_all(data_dir, adcs=None) -> pd.DataFrame:
    """Load adc*_module_mac_summary.csv for all ADC values, concat with adc_bits column."""
    if adcs is None:
        adcs = ADC_LIST
    frames = []
    for bits in adcs:
        path = Path(data_dir) / f"adc{bits}_module_mac_summary.csv"
        if not path.exists():
            WARNINGS.append(f"MISSING: {path}")
            continue
        df = pd.read_csv(path)
        df["adc_bits"] = bits
        frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined["sublayer"] = pd.Categorical(
        combined["sublayer"], categories=SUBLAYER_ORDER, ordered=True
    )
    return combined


def sanity_check(df):
    """Assert 12 layers × 6 sublayers per ADC."""
    for bits, grp in df.groupby("adc_bits"):
        n_layers = grp["layer_idx"].nunique()
        n_sl = grp["sublayer"].nunique()
        assert n_layers == 12, f"adc{bits}: expected 12 layers, got {n_layers}"
        assert n_sl == 6, f"adc{bits}: expected 6 sublayers, got {n_sl}"


def try_load(path):
    """Load CSV or return None (with warning) if missing."""
    p = Path(path)
    if not p.exists():
        WARNINGS.append(f"MISSING: {path} — figure skipped")
        return None
    return pd.read_csv(p)


def annotate_worst_cell(ax, pivot_df, fmt="{:.1f}", find_min=True):
    """Annotate worst cell (min or max) with red rectangle + text."""
    data = pivot_df.values.astype(float)
    if find_min:
        r, c = np.unravel_index(np.argmin(data), data.shape)
    else:
        r, c = np.unravel_index(np.argmax(data), data.shape)
    val = data[r, c]
    rect = mpatches.Rectangle(
        (c - 0.5, r - 0.5), 1, 1,
        linewidth=2, edgecolor="red", facecolor="none",
    )
    ax.add_patch(rect)
    # Label already drawn by cell text loop; no extra text needed here


def _cell_text_color(val, vmin, vmax, cmap_name):
    """Return 'white' or 'black' for contrast against cell color."""
    norm = (val - vmin) / (vmax - vmin + 1e-12)
    norm = np.clip(norm, 0, 1)
    cmap = plt.get_cmap(cmap_name)
    r, g, b, _ = cmap(norm)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 0.45 else "black"


def _draw_heatmap(ax, pivot, vmin, vmax, cmap, norm=None, title=""):
    """Draw imshow heatmap with cell annotations and return AxesImage."""
    if norm is not None:
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=7)
    ax.set_title(title, fontsize=8)
    # Cell text
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if np.isnan(val):
                continue
            if norm is not None:
                # log scale — estimate color by normalized log
                try:
                    norm_val = norm(val)
                except Exception:
                    norm_val = 0.5
                cmap_obj = plt.get_cmap(cmap)
                rgba = cmap_obj(norm_val)
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                tc = "white" if lum < 0.45 else "black"
                if val < 0.01:
                    txt = f"{val:.2e}"
                else:
                    txt = f"{val:.3f}"
            else:
                tc = _cell_text_color(val, vmin, vmax, cmap)
                txt = f"{val:.1f}"
            ax.text(c, r, txt, ha="center", va="center", fontsize=6, color=tc)
    return im


def _make_pivot(df_adc, metric):
    """Create 12×6 pivot: rows=layer_idx, cols=SUBLAYER_ORDER."""
    pivot = df_adc.pivot(index="layer_idx", columns="sublayer", values=metric)
    pivot = pivot.reindex(columns=SUBLAYER_ORDER)
    return pivot


# ─────────────────────────────────────────────────────────────
# Section 4: Main Fig 1 — ADC sweep SNR overview
# ─────────────────────────────────────────────────────────────

def gen_main1(data_dir, out_dir):
    """fig_main1_adc_sweep_snr: global SNR + per-sublayer SNR vs ADC bits."""
    basename = "fig_main1_adc_sweep_snr"
    src = Path(data_dir) / "summary_adc_sweep.csv"
    df = try_load(src)
    if df is None:
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("ADC Sweep: MAC SNR Overview", fontsize=10, fontweight="bold")

    # Panel A: global mean SNR
    ax_a.plot(df["adc_bits"], df["mac_snr_mean"], "o-", color="steelblue", lw=1.5, ms=5)
    ax_a.axvline(x=8, color="gray", ls="--", lw=1, label="knee (ADC-8)")
    for _, row in df.iterrows():
        ax_a.annotate(
            f"{row['mac_snr_mean']:.1f} dB",
            xy=(row["adc_bits"], row["mac_snr_mean"]),
            xytext=(4, 6), textcoords="offset points",
            fontsize=7, color="steelblue",
        )
    ax_a.set_xlabel("ADC bits")
    ax_a.set_ylabel("Mean MAC SNR (dB)")
    ax_a.set_title("(A) Global Mean SNR")
    ax_a.legend(fontsize=7)
    ax_a.set_xticks(ADC_LIST)
    ax_a.grid(True, alpha=0.25)

    # Panel B: per-sublayer SNR
    snr_mat = np.zeros((len(SUBLAYER_ORDER), len(ADC_LIST)))
    for i, sl in enumerate(SUBLAYER_ORDER):
        col = f"mac_snr_{sl}_mean"
        vals = df.set_index("adc_bits")[col].reindex(ADC_LIST).values
        snr_mat[i] = vals
        ax_b.plot(
            ADC_LIST, vals,
            marker=SUBLAYER_MARKERS[sl], color=SUBLAYER_COLORS[sl],
            lw=1.2, ms=5, label=sl,
        )
    ax_b.set_xlabel("ADC bits")
    ax_b.set_ylabel("MAC SNR (dB)")
    ax_b.set_title("(B) Per-Sublayer SNR")
    ax_b.legend(loc="upper left", ncol=2, fontsize=7)
    ax_b.set_xticks(ADC_LIST)
    ax_b.grid(True, alpha=0.25)

    plt.tight_layout()

    npz_data = {
        "adc_bits": np.array(ADC_LIST),
        "snr_mean": df.set_index("adc_bits")["mac_snr_mean"].reindex(ADC_LIST).values,
        "snr_per_sublayer": snr_mat,
        "sublayer_names": np.array(SUBLAYER_ORDER),
    }
    meta = make_meta(
        files_read=[src],
        key_config={"adc_bits": ADC_LIST, "source": "summary_adc_sweep.csv"},
    )
    save_figure(fig, basename, out_dir / "main", npz_data, meta)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Section 5: Main Fig 2 — Heatmap SNR adc6 vs adc8
# ─────────────────────────────────────────────────────────────

def gen_main2(data_dir, out_dir):
    """fig_main2_heatmap_snr_adc6_adc8: shared-colorbar SNR heatmaps."""
    basename = "fig_main2_heatmap_snr_adc6_adc8"
    src6 = Path(data_dir) / "adc6_module_mac_summary.csv"
    src8 = Path(data_dir) / "adc8_module_mac_summary.csv"
    df6 = try_load(src6)
    df8 = try_load(src8)
    if df6 is None or df8 is None:
        return

    pivot6 = _make_pivot(df6, "mac_snr_db")
    pivot8 = _make_pivot(df8, "mac_snr_db")

    vmin = 2.0
    vmax = 35.0

    fig, (ax6, ax8) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("MAC SNR (dB): ADC-6 vs ADC-8 bit", fontsize=10, fontweight="bold")

    im = None
    for ax, pivot, title in [
        (ax6, pivot6, "ADC-6 bit"),
        (ax8, pivot8, "ADC-8 bit"),
    ]:
        im = _draw_heatmap(ax, pivot, vmin, vmax, "RdYlGn", title=title)
        ax.set_xlabel("Sublayer")
        ax.set_ylabel("Layer index")
        annotate_worst_cell(ax, pivot, find_min=True)

    cbar = fig.colorbar(im, ax=[ax6, ax8], shrink=0.8, pad=0.02)
    cbar.set_label("MAC SNR (dB)", fontsize=8)

    plt.tight_layout()

    npz_data = {
        "pivot_adc6": pivot6.values,
        "pivot_adc8": pivot8.values,
        "layer_idx": np.arange(12),
        "sublayer_names": np.array(SUBLAYER_ORDER),
    }
    meta = make_meta(
        files_read=[src6, src8],
        key_config={"vmin": vmin, "vmax": vmax, "metric": "mac_snr_db"},
    )
    save_figure(fig, basename, out_dir / "main", npz_data, meta)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Section 6: Main Fig 3 — O sublayer deadzone vs ADC
# ─────────────────────────────────────────────────────────────

def gen_main3(data_dir, out_dir):
    """fig_main3_O_deadzone_vs_adc: ref_deadzone_ratio for O sublayer."""
    basename = "fig_main3_O_deadzone_vs_adc"
    df_all = load_module_summary_all(data_dir)
    if df_all is None:
        WARNINGS.append("MISSING: module_mac_summary files — fig_main3 skipped")
        return

    df_o = df_all[df_all["sublayer"] == "O"].copy()

    # per-layer per-ADC deadzone
    deadzone_per_layer = np.zeros((len(ADC_LIST), 12))
    deadzone_mean = np.zeros(len(ADC_LIST))
    for i, bits in enumerate(ADC_LIST):
        sub = df_o[df_o["adc_bits"] == bits].sort_values("layer_idx")
        vals = sub.set_index("layer_idx")["ref_deadzone_ratio"].reindex(range(12)).values
        deadzone_per_layer[i] = vals
        deadzone_mean[i] = np.nanmean(vals)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.suptitle("O Sublayer: Ref Deadzone Ratio vs ADC bits", fontsize=10, fontweight="bold")

    ax.semilogy(ADC_LIST, deadzone_mean, "o-", color="steelblue", lw=1.5, ms=6)
    ax.axhline(0.1, color="gray", ls=":", lw=1, label="10% threshold")
    for bits, val in zip(ADC_LIST, deadzone_mean):
        ax.annotate(
            f"{val*100:.0f}%",
            xy=(bits, val),
            xytext=(4, 6), textcoords="offset points",
            fontsize=7, color=ADC_COLORS[bits],
        )
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("Mean Ref Deadzone Ratio (log)")
    ax.set_title("O Sublayer — Deadzone Ratio")
    ax.legend(fontsize=7)
    ax.set_xticks(ADC_LIST)
    ax.grid(True, alpha=0.25, which="both")

    plt.tight_layout()

    npz_data = {
        "adc_bits": np.array(ADC_LIST),
        "deadzone_O_mean": deadzone_mean,
        "deadzone_O_per_layer": deadzone_per_layer,
    }
    meta = make_meta(
        files_read=[Path(data_dir) / f"adc{b}_module_mac_summary.csv" for b in ADC_LIST],
        key_config={"sublayer": "O", "metric": "ref_deadzone_ratio"},
    )
    save_figure(fig, basename, out_dir / "main", npz_data, meta)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Section 7: Supp S1 — 5 metric heatmaps, all ADC
# ─────────────────────────────────────────────────────────────

def gen_supp_s1(data_dir, out_dir):
    """fig_s1_heatmaps_*: one figure per metric, 5 ADC panels each."""
    df_all = load_module_summary_all(data_dir)
    if df_all is None:
        WARNINGS.append("MISSING: module_mac_summary files — fig_s1 skipped")
        return

    metric_info = {
        "mac_snr_db":         ("fig_s1_heatmaps_snr_all_adc",      "MAC SNR (dB)"),
        "mac_nmse":           ("fig_s1_heatmaps_nmse_all_adc",     "MAC NMSE"),
        "cosine":             ("fig_s1_heatmaps_cosine_all_adc",   "Cosine Similarity"),
        "out_clip_ratio":     ("fig_s1_heatmaps_clip_all_adc",     "Out Clip Ratio"),
        "ref_deadzone_ratio": ("fig_s1_heatmaps_deadzone_all_adc", "Ref Deadzone Ratio"),
    }

    for metric, (basename, label) in metric_info.items():
        sc = S1_SCALES[metric]
        vmin, vmax, cmap_name, use_log = sc["vmin"], sc["vmax"], sc["cmap"], sc["log"]

        if use_log:
            # clamp vmin > 0 for LogNorm
            _vmin = max(vmin, 1e-10)
            norm = mcolors.LogNorm(vmin=_vmin, vmax=vmax)
        else:
            norm = None

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        fig.suptitle(f"S1 — {label} Heatmaps (all ADC bits)", fontsize=10, fontweight="bold")

        pivots = {}
        im = None
        for ax, bits in zip(axes, ADC_LIST):
            sub = df_all[df_all["adc_bits"] == bits]
            pivot = _make_pivot(sub, metric)
            # For out_clip_ratio: some cells may be exactly 0 — keep as-is (no log)
            if use_log:
                # Replace 0s with a small value to avoid log issues in display
                vals = pivot.values.astype(float).copy()
                vals[vals <= 0] = _vmin
                import copy
                pivot_display = pivot.copy()
                pivot_display.values[:] = vals
            else:
                pivot_display = pivot

            im = _draw_heatmap(ax, pivot_display, vmin, vmax, cmap_name, norm=norm,
                               title=f"ADC-{bits} bit")
            ax.set_xlabel("Sublayer")
            if bits == ADC_LIST[0]:
                ax.set_ylabel("Layer index")
            else:
                ax.set_ylabel("")
            annotate_worst_cell(ax, pivot, find_min=(metric != "cosine" and "snr" not in metric))
            pivots[bits] = pivot.values

        if im is not None:
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
            cbar.set_label(label, fontsize=8)

        plt.tight_layout()

        npz_data = {f"pivot_adc{b}": pivots[b] for b in ADC_LIST}
        npz_data["layer_idx"] = np.arange(12)
        npz_data["sublayer_names"] = np.array(SUBLAYER_ORDER)
        npz_data["vmin"] = np.array([vmin])
        npz_data["vmax"] = np.array([vmax])

        meta = make_meta(
            files_read=[Path(data_dir) / f"adc{b}_module_mac_summary.csv" for b in ADC_LIST],
            key_config={"metric": metric, "vmin": vmin, "vmax": vmax, "log_scale": use_log},
        )
        save_figure(fig, basename, out_dir / "supp", npz_data, meta)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Section 8: Supp S2 — Boxplot SNR by sublayer adc6 vs adc8
# ─────────────────────────────────────────────────────────────

def gen_supp_s2(data_dir, out_dir):
    """fig_s2_boxplot_snr_by_sublayer_adc6_adc8."""
    basename = "fig_s2_boxplot_snr_by_sublayer_adc6_adc8"
    src6 = Path(data_dir) / "adc6_module_mac_summary.csv"
    src8 = Path(data_dir) / "adc8_module_mac_summary.csv"
    df6 = try_load(src6)
    df8 = try_load(src8)
    if df6 is None or df8 is None:
        return

    fig, (ax6, ax8) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("MAC SNR Distribution by Sublayer", fontsize=10, fontweight="bold")

    data_adc6 = np.zeros((len(SUBLAYER_ORDER), 12))
    data_adc8 = np.zeros((len(SUBLAYER_ORDER), 12))

    for ax, df, title, data_arr in [
        (ax6, df6, "ADC-6 bit", data_adc6),
        (ax8, df8, "ADC-8 bit", data_adc8),
    ]:
        data_per_sublayer = []
        for i, sl in enumerate(SUBLAYER_ORDER):
            vals = df[df["sublayer"] == sl].sort_values("layer_idx")["mac_snr_db"].values
            data_per_sublayer.append(vals)
            data_arr[i] = vals

        bp = ax.boxplot(
            data_per_sublayer,
            patch_artist=True,
            medianprops={"color": "black", "lw": 1.5},
            whiskerprops={"lw": 1},
            capprops={"lw": 1},
            flierprops={"marker": "x", "ms": 4, "alpha": 0.5},
        )
        for patch, sl in zip(bp["boxes"], SUBLAYER_ORDER):
            patch.set_facecolor(SUBLAYER_COLORS[sl])
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(SUBLAYER_ORDER) + 1))
        ax.set_xticklabels(SUBLAYER_ORDER)
        ax.set_xlabel("Sublayer")
        ax.set_ylabel("MAC SNR (dB)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()

    npz_data = {
        "data_adc6": data_adc6,
        "data_adc8": data_adc8,
        "sublayer_names": np.array(SUBLAYER_ORDER),
    }
    meta = make_meta(
        files_read=[src6, src8],
        key_config={"metric": "mac_snr_db", "n_layers": 12},
    )
    save_figure(fig, basename, out_dir / "supp", npz_data, meta)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Section 9: Supp S3 — Clip ratio summary
# ─────────────────────────────────────────────────────────────

def gen_supp_s3(data_dir, out_dir):
    """fig_s3_clipratio_summary: grouped bar + FFN per-layer traces."""
    basename = "fig_s3_clipratio_summary"
    df_all = load_module_summary_all(data_dir)
    if df_all is None:
        WARNINGS.append("MISSING: module_mac_summary files — fig_s3 skipped")
        return

    # mean clip ratio per (ADC, sublayer)
    mean_clip = np.zeros((len(ADC_LIST), len(SUBLAYER_ORDER)))
    ffn1_per_layer = np.zeros((len(ADC_LIST), 12))
    ffn2_per_layer = np.zeros((len(ADC_LIST), 12))

    for i, bits in enumerate(ADC_LIST):
        sub = df_all[df_all["adc_bits"] == bits]
        for j, sl in enumerate(SUBLAYER_ORDER):
            mean_clip[i, j] = sub[sub["sublayer"] == sl]["out_clip_ratio"].mean()
        for sl_name, arr in [("FFN1", ffn1_per_layer), ("FFN2", ffn2_per_layer)]:
            sl_sub = sub[sub["sublayer"] == sl_name].sort_values("layer_idx")
            arr[i] = sl_sub.set_index("layer_idx")["out_clip_ratio"].reindex(range(12)).values

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Out Clip Ratio Summary", fontsize=10, fontweight="bold")

    # Panel A: grouped bar chart
    x = np.arange(len(SUBLAYER_ORDER))
    n_adc = len(ADC_LIST)
    width = 0.15
    offsets = np.linspace(-(n_adc - 1) / 2, (n_adc - 1) / 2, n_adc) * width

    for i, (bits, offset) in enumerate(zip(ADC_LIST, offsets)):
        ax_a.bar(
            x + offset, mean_clip[i], width,
            color=ADC_COLORS[bits], label=f"ADC-{bits}",
            alpha=0.85,
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(SUBLAYER_ORDER)
    ax_a.set_xlabel("Sublayer")
    ax_a.set_ylabel("Mean Out Clip Ratio")
    ax_a.set_title("(A) Mean Clip Ratio by Sublayer & ADC")
    ax_a.set_yscale("symlog", linthresh=1e-6)
    ax_a.legend(fontsize=7, ncol=2)
    ax_a.grid(True, axis="y", alpha=0.25)

    # Panel B: FFN1/FFN2 per-layer traces
    for i, bits in enumerate(ADC_LIST):
        ax_b.plot(range(12), ffn1_per_layer[i], "-",
                  color=ADC_COLORS[bits], lw=1.2, label=f"FFN1 ADC-{bits}")
        ax_b.plot(range(12), ffn2_per_layer[i], "--",
                  color=ADC_COLORS[bits], lw=1.2, label=f"FFN2 ADC-{bits}")

    ax_b.set_xlabel("Layer index")
    ax_b.set_ylabel("Out Clip Ratio")
    ax_b.set_title("(B) FFN1 / FFN2 Per-Layer Clipping")
    ax_b.legend(fontsize=6, ncol=2)
    ax_b.grid(True, alpha=0.25)

    plt.tight_layout()

    npz_data = {
        "mean_clip_by_adc_sublayer": mean_clip,
        "ffn1_per_layer": ffn1_per_layer,
        "ffn2_per_layer": ffn2_per_layer,
        "adc_bits": np.array(ADC_LIST),
        "sublayer_names": np.array(SUBLAYER_ORDER),
    }
    meta = make_meta(
        files_read=[Path(data_dir) / f"adc{b}_module_mac_summary.csv" for b in ADC_LIST],
        key_config={"metric": "out_clip_ratio"},
    )
    save_figure(fig, basename, out_dir / "supp", npz_data, meta)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Section 10: Supp S4 — Logit single sample
# ─────────────────────────────────────────────────────────────

def gen_supp_s4(data_dir, out_dir):
    """fig_s4_logit_single_sample: single-row logit metrics visualization."""
    basename = "fig_s4_logit_single_sample"
    src = Path(data_dir) / "single_run_logit_metrics.csv"
    df = try_load(src)
    if df is None:
        return

    row = df.iloc[0]
    all_metric_cols = [c for c in df.columns if c != "step"]
    vals = row[all_metric_cols].values.astype(float)
    metrics_names = np.array(all_metric_cols)

    # Before/after pairs
    pair_defs = [("mse_start", "mse_end"), ("kl_start", "kl_end"), ("flip_start", "flip_end")]
    pair_labels, before_vals, after_vals = [], [], []
    for s, e in pair_defs:
        if s in df.columns and e in df.columns:
            pair_labels.append(s.replace("_start", ""))
            before_vals.append(float(row[s]))
            after_vals.append(float(row[e]))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Logit Metrics — Single Sample", fontsize=10, fontweight="bold")
    fig.text(
        0.5, 0.98,
        "⚠  SINGLE SAMPLE — NOT STATISTICALLY SIGNIFICANT",
        ha="center", va="top", fontsize=8, color="tomato", fontstyle="italic",
    )

    # Panel A: all metrics bar chart
    colors_bar = ["tomato" if v < 0 else "steelblue" for v in vals]
    bars = ax_a.bar(range(len(metrics_names)), vals, color=colors_bar)
    ax_a.set_xticks(range(len(metrics_names)))
    ax_a.set_xticklabels(metrics_names, rotation=35, ha="right", fontsize=7)
    ax_a.set_title("(A) All Logit Metrics")
    ax_a.axhline(0, color="black", lw=0.8)
    ax_a.grid(True, axis="y", alpha=0.25)
    for bar, v in zip(bars, vals):
        ypos = bar.get_height() + 0.0005 if v >= 0 else bar.get_height() - 0.003
        ax_a.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"{v:.4f}", ha="center", va="bottom", fontsize=6,
        )

    # Panel B: before/after pairs
    if pair_labels:
        x = np.arange(len(pair_labels))
        w = 0.35
        ax_b.bar(x - w / 2, before_vals, w, label="before (start)", color="steelblue", alpha=0.85)
        ax_b.bar(x + w / 2, after_vals,  w, label="after (end)",    color="tomato",    alpha=0.85)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(pair_labels, fontsize=9)
        ax_b.set_title("(B) Before vs After")
        ax_b.legend(fontsize=7)
        ax_b.grid(True, axis="y", alpha=0.25)
    else:
        ax_b.text(0.5, 0.5, "No before/after pairs found",
                  ha="center", va="center", transform=ax_b.transAxes)
        ax_b.set_title("(B) Before vs After")

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    npz_data = {
        "metrics_names": metrics_names,
        "metrics_values": vals,
        "before_values": np.array(before_vals) if before_vals else np.array([]),
        "after_values":  np.array(after_vals)  if after_vals  else np.array([]),
        "pair_labels":   np.array(pair_labels) if pair_labels else np.array([]),
    }
    meta = make_meta(
        files_read=[src],
        key_config={"n_rows": len(df), "note": "SINGLE SAMPLE — NOT STATISTICALLY SIGNIFICANT"},
    )
    save_figure(fig, basename, out_dir / "supp", npz_data, meta)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Section 11: main()
# ─────────────────────────────────────────────────────────────

ALL_FIGS = {
    "main1": gen_main1,
    "main2": gen_main2,
    "main3": gen_main3,
    "s1":    gen_supp_s1,
    "s2":    gen_supp_s2,
    "s3":    gen_supp_s3,
    "s4":    gen_supp_s4,
}


def write_readme(out_dir, generated_files, timestamp):
    """Write plots/README.md summarizing outputs."""
    lines = [
        "# Paper Figures — ADC Sweep Diagnostic",
        "",
        f"Generated: {timestamp}",
        "",
        "## Generated Files",
        "",
    ]

    main_files = sorted([f for f in generated_files if "main" in str(f)])
    supp_files = sorted([f for f in generated_files if "supp" in str(f)])

    if main_files:
        lines.append("### Main Figures (`plots/main/`)")
        lines.append("")
        for f in main_files:
            lines.append(f"- `{Path(f).name}`")
        lines.append("")

    if supp_files:
        lines.append("### Supplementary Figures (`plots/supp/`)")
        lines.append("")
        for f in supp_files:
            lines.append(f"- `{Path(f).name}`")
        lines.append("")

    lines.append("## Input Files")
    lines.append("")
    lines.append("| Figure | Source CSVs |")
    lines.append("|--------|-------------|")
    lines.append("| main1  | `summary_adc_sweep.csv` |")
    lines.append("| main2  | `adc6_module_mac_summary.csv`, `adc8_module_mac_summary.csv` |")
    lines.append("| main3  | `adc{4,6,8,10,12}_module_mac_summary.csv` (O sublayer) |")
    lines.append("| s1     | `adc{4,6,8,10,12}_module_mac_summary.csv` (5 metrics) |")
    lines.append("| s2     | `adc6_module_mac_summary.csv`, `adc8_module_mac_summary.csv` |")
    lines.append("| s3     | `adc{4,6,8,10,12}_module_mac_summary.csv` (FFN clip) |")
    lines.append("| s4     | `single_run_logit_metrics.csv` |")
    lines.append("")

    if WARNINGS:
        lines.append("## Skipped / Warnings")
        lines.append("")
        for w in WARNINGS:
            lines.append(f"- {w}")
        lines.append("")
    else:
        lines.append("## Warnings")
        lines.append("")
        lines.append("None.")
        lines.append("")

    readme_path = out_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nREADME → {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from ADC sweep CSV data."
    )
    parser.add_argument(
        "--data-dir", default="/data/main_results/results/csv/diag_fwd_io",
        help="Directory containing input CSV files",
    )
    parser.add_argument(
        "--out-dir", default="/data/main_results/results/figures/diag_fwd_io/plots",
        help="Root output directory (main/ and supp/ created inside)",
    )
    parser.add_argument(
        "--only", nargs="+", choices=list(ALL_FIGS.keys()), default=None,
        help="Generate only specific figures (e.g. --only main1 s1)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    (out_dir / "main").mkdir(parents=True, exist_ok=True)
    (out_dir / "supp").mkdir(parents=True, exist_ok=True)

    figs_to_run = args.only if args.only else list(ALL_FIGS.keys())

    print(f"=== Generating paper figures ===")
    print(f"  data_dir : {data_dir}")
    print(f"  out_dir  : {out_dir}")
    print(f"  figures  : {figs_to_run}")
    print()

    timestamp = datetime.now().isoformat()

    for name in figs_to_run:
        print(f"[{name}]")
        try:
            ALL_FIGS[name](data_dir, out_dir)
        except Exception as e:
            msg = f"ERROR in {name}: {e}"
            WARNINGS.append(msg)
            print(f"  !! {msg}")
        print()

    # Collect generated files
    generated = list((out_dir / "main").glob("*.pdf")) + \
                list((out_dir / "main").glob("*.png")) + \
                list((out_dir / "main").glob("*.npz")) + \
                list((out_dir / "main").glob("*.json")) + \
                list((out_dir / "supp").glob("*.pdf")) + \
                list((out_dir / "supp").glob("*.png")) + \
                list((out_dir / "supp").glob("*.npz")) + \
                list((out_dir / "supp").glob("*.json"))

    write_readme(out_dir, generated, timestamp)

    if WARNINGS:
        print("\nWarnings:")
        for w in WARNINGS:
            print(f"  ! {w}")
    else:
        print("No warnings.")

    n_pdf = len(list(out_dir.rglob("*.pdf")))
    n_png = len(list(out_dir.rglob("*.png")))
    print(f"\nDone: {n_pdf} PDFs, {n_png} PNGs")


if __name__ == "__main__":
    main()
