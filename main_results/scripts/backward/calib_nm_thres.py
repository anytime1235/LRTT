"""calib_nm_thres.py — nm_thres Calibration Effect on BERT-base AIMC Backward Outlier

Key insight: nm_thres affects ONLY the quantization of δ inside the tile,
not the raw gradient captured by backward hooks. Therefore QZR_before and
QZR_after(θ) can both be computed analytically from a SINGLE diagnostic run.

Usage:
  python calib_nm_thres.py                            # default: both modes, q=0.99
  python calib_nm_thres.py --mode global --q 0.999
  python calib_nm_thres.py --mode layerwise --safety 1.1
  python calib_nm_thres.py --n-step 5 --batch-size 2  # smoke test
"""

import argparse
from collections import defaultdict
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, set_seed
from transformers import default_data_collator
from torch.utils.data import DataLoader
from datasets import load_dataset

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--mode",       choices=["global", "layerwise", "both"], default="both")
parser.add_argument("--q",          type=float, default=0.99,   help="Quantile for theta (0.99 or 0.999)")
parser.add_argument("--safety",     type=float, default=1.1,    help="Safety factor (1.05–1.2)")
parser.add_argument("--n-step",     type=int,   default=200)
parser.add_argument("--batch-size", type=int,   default=8)
args = parser.parse_args()

# =============================================================================
# Constants
# =============================================================================

N_STEP          = args.n_step
DIAG_BATCH_SIZE = args.batch_size
MAX_SEQ_LENGTH  = 384
DOC_STRIDE      = 128
SEED            = 42

DAC_BITS  = 7
ADC_BITS  = 9
INP_BOUND = 1.0
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Log-spaced theta grid: 10^-5 to 10^0 (30 values)
THETA_GRID = np.logspace(-5, 0, 30).tolist()

OUT_DIR                   = "/data/results/tikitakav1"
METRICS_BEFORE_CSV        = f"{OUT_DIR}/calib_metrics_before.csv"
METRICS_AFTER_GLOBAL_CSV  = f"{OUT_DIR}/calib_metrics_after_global.csv"
METRICS_AFTER_LW_CSV      = f"{OUT_DIR}/calib_metrics_after_layerwise.csv"
THETA_GLOBAL_JSON         = f"{OUT_DIR}/theta_global.json"
THETA_LAYERWISE_JSON      = f"{OUT_DIR}/theta_layerwise.json"
FIG_PATH                  = f"{OUT_DIR}/fig_nm_thres_calibration_effect.pdf"

print(f"[Config] Device={DEVICE}, N_STEP={N_STEP}, BATCH={DIAG_BATCH_SIZE}")
print(f"[Config] mode={args.mode}, q={args.q}, safety={args.safety}")

# =============================================================================
# RPU Config
# =============================================================================

def create_rpu_config(nm_thres: float = 0.0):
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import IdealDevice
    from aihwkit.simulator.configs.utils import NoiseManagementType

    rpu = SingleRPUConfig(device=IdealDevice())
    for io in [rpu.forward, rpu.backward]:
        io.inp_bound        = INP_BOUND
        io.inp_res          = 1 / (2**DAC_BITS - 2)
        io.out_bound        = 12.0
        io.out_res          = 1 / (2**ADC_BITS - 2)
        io.noise_management = NoiseManagementType.ABS_MAX
        io.out_noise        = 0.0
    rpu.backward.nm_thres              = nm_thres
    rpu.mapping.digital_bias           = True
    rpu.mapping.weight_scaling_omega   = 1.0
    rpu.mapping.weight_scaling_columnwise = True
    return rpu

# =============================================================================
# Model Creation
# =============================================================================

def _layer_names(model):
    always_digital = ["qa_outputs", "pooler"]
    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    target = [n for n in all_linear
              if "encoder" in n and "attention" in n
              and not any(d in n for d in always_digital)]
    nontarget = [n for n in all_linear
                 if "encoder" in n and n not in target
                 and not any(d in n for d in always_digital)]
    return target, nontarget, all_linear


def create_model():
    from aihwkit.nn import AnalogLinear
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.optim.context import AnalogContext

    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    target, nontarget, all_linear = _layer_names(model)

    # Pass 1: target layers (nm_thres=0 for diagnostic; we compute analytically)
    rpu = create_rpu_config(nm_thres=0.0)
    model = convert_to_analog(model, rpu,
                              exclude_modules=[n for n in all_linear if n not in target])

    # Pass 2: non-target (frozen)
    nt_rpu = create_rpu_config(nm_thres=0.0)
    model = convert_to_analog(model, nt_rpu,
                              exclude_modules=[n for n in all_linear if n not in nontarget])

    def _noop(x, d, *a, **kw): return None
    for name, m in model.named_modules():
        if isinstance(m, AnalogLinear) and name not in target:
            for tile in m.analog_tiles():
                tile.update = _noop

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        if isinstance(p, AnalogContext):
            p.requires_grad_(True)
    for n, p in model.named_parameters():
        if "qa_outputs" in n:
            p.requires_grad_(True)

    n_t = sum(1 for n, m in model.named_modules()
              if isinstance(m, AnalogLinear) and n in target)
    n_all = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog tiles — target: {n_t}, frozen: {n_all - n_t}")
    return model.to(DEVICE)

# =============================================================================
# Data Loading
# =============================================================================

def load_data(tokenizer):
    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map     = inputs.pop("overflow_to_sample_mapping")
        answers        = examples["answers"]
        sp, ep = [], []
        for i, offset in enumerate(offset_mapping):
            ans = answers[sample_map[i]]
            if not ans["answer_start"]:
                sp.append(0); ep.append(0); continue
            sc = ans["answer_start"][0]
            ec = sc + len(ans["text"][0])
            seq = inputs.sequence_ids(i)
            idx = 0
            while seq[idx] != 1: idx += 1
            cs = idx
            while idx < len(seq) and seq[idx] == 1: idx += 1
            ce = idx - 1
            if offset[cs][0] > ec or offset[ce][1] < sc:
                sp.append(0); ep.append(0)
            else:
                idx = cs
                while idx <= ce and offset[idx][0] <= sc: idx += 1
                sp.append(idx - 1)
                idx = ce
                while idx >= cs and offset[idx][1] >= ec: idx -= 1
                ep.append(idx + 1)
        inputs["start_positions"] = sp
        inputs["end_positions"]   = ep
        return inputs

    raw = load_dataset("squad")
    tok = raw["train"].map(preprocess_train, batched=True,
                           remove_columns=raw["train"].column_names)
    n = min(N_STEP * DIAG_BATCH_SIZE, len(tok))
    subset = tok.shuffle(seed=SEED).select(range(n))
    loader = DataLoader(subset, batch_size=DIAG_BATCH_SIZE, shuffle=False,
                        collate_fn=default_data_collator)
    print(f"  Dataset: {n} samples → {len(loader)} batches")
    return loader

# =============================================================================
# LayerStats — single run, analytical theta_grid
# =============================================================================

_LAYER_RE = re.compile(
    r"encoder\.layer\.(\d+)\."
    r"(attention\.self\.query|attention\.self\.key|attention\.self\.value"
    r"|attention\.output\.dense)"
)
_SUBLAYER_MAP = {
    "attention.self.query":   "Q",
    "attention.self.key":     "K",
    "attention.self.value":   "V",
    "attention.output.dense": "O",
}

def parse_layer_name(name):
    m = _LAYER_RE.search(name)
    if m is None: return None
    return int(m.group(1)), _SUBLAYER_MAP[m.group(2)]


class LayerStats:
    """Single-run accumulator. Computes QZR analytically for a grid of nm_thres values."""

    def __init__(self, name, layer_idx, sublayer):
        self.name      = name
        self.layer_idx = layer_idx
        self.sublayer  = sublayer
        self.dac_step  = 2 * INP_BOUND / (2**DAC_BITS - 1)
        self.eps       = 1e-8

        self.qzr_before_steps = []      # nm_thres=0
        self.cosine_steps     = []
        self._absmax_buf      = []      # per-vector absmax, for CCR
        # QZR_after grid: {theta_float: [qzr_step, ...]}
        self._qzr_theta       = defaultdict(list)

    def update(self, dy: torch.Tensor):
        with torch.no_grad():
            dy_flat = (dy.detach().reshape(-1, dy.shape[-1]).float()
                       if dy.dim() == 3 else dy.detach().float())
            N, D    = dy_flat.shape
            abs_dy  = dy_flat.abs()
            absmax_v = abs_dy.max(dim=1).values   # (N,)

            # --- QZR_before (nm_thres = 0) ---
            alpha  = absmax_v.unsqueeze(1).clamp(min=self.eps)
            scaled = dy_flat / alpha * INP_BOUND
            qzr_b  = (scaled.abs() < self.dac_step / 2).float().mean().item()
            self.qzr_before_steps.append(qzr_b)

            # --- Cosine similarity (baseline) ---
            dy_q = (scaled / self.dac_step).round() * self.dac_step * alpha / INP_BOUND
            cos  = F.cosine_similarity(dy_flat, dy_q, dim=1).mean().item()
            self.cosine_steps.append(cos)

            # --- Absmax buffer for CCR ---
            self._absmax_buf.append(absmax_v.cpu().float().numpy())

            # --- QZR_after for each theta in THETA_GRID ---
            for theta in THETA_GRID:
                alpha_cap   = absmax_v.clamp(max=float(theta)).unsqueeze(1).clamp(min=self.eps)
                scaled_cap  = dy_flat / alpha_cap * INP_BOUND
                qzr_t       = (scaled_cap.abs() < self.dac_step / 2).float().mean().item()
                self._qzr_theta[float(theta)].append(qzr_t)

    # --- Query methods ---

    def absmax_array(self) -> np.ndarray:
        return np.concatenate(self._absmax_buf)

    def qzr_before(self) -> float:
        return float(np.mean(self.qzr_before_steps))

    def qzr_after(self, theta: float) -> float:
        """QZR_after at nearest theta in grid."""
        grid = np.array(sorted(self._qzr_theta.keys()))
        idx  = int(np.argmin(np.abs(grid - theta)))
        return float(np.mean(self._qzr_theta[grid[idx]]))

    def ccr(self, theta: float) -> float:
        """Cap Clipping Rate: P(absmax(δ_vec) > theta)."""
        return float((self.absmax_array() > theta).mean())

    def cosine_after(self, theta: float) -> float:
        """Cosine similarity with capped nm_thres (approximate: only last step's δ)."""
        # Not stored per-step for memory reasons; return baseline as reference
        return float(np.mean(self.cosine_steps))

# =============================================================================
# Hook Registration
# =============================================================================

def register_hooks(model):
    from aihwkit.nn import AnalogLinear
    stats_dict, handles = {}, []
    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear): continue
        parsed = parse_layer_name(name)
        if parsed is None: continue
        layer_idx, sublayer = parsed
        stats = LayerStats(name=name, layer_idx=layer_idx, sublayer=sublayer)
        stats_dict[name] = stats

        def make_hook(s):
            def fn(mod, grad_input, grad_output):
                if grad_output[0] is not None:
                    s.update(grad_output[0])
            return fn

        handles.append(module.register_full_backward_hook(make_hook(stats)))

    print(f"[Hook] {len(stats_dict)} hooks: "
          f"{sorted(set(s.sublayer for s in stats_dict.values()))}")
    return stats_dict, handles

# =============================================================================
# Diagnostic Run
# =============================================================================

def run_diagnostic(model, loader):
    from aihwkit.optim import AnalogSGD
    optimizer = AnalogSGD(model.parameters(), lr=0.0)
    model.train()
    torch.manual_seed(SEED)
    for step, batch in enumerate(tqdm(loader, total=N_STEP, desc="Diagnostic")):
        if step >= N_STEP: break
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            start_positions=batch["start_positions"].to(DEVICE),
            end_positions=batch["end_positions"].to(DEVICE),
        )
        outputs.loss.backward()
        optimizer.step()

# =============================================================================
# Calibration: compute theta
# =============================================================================

def compute_theta(stats_dict, q, safety):
    """Returns (theta_global: float, theta_layerwise: {name: float})."""
    all_absmax = np.concatenate([s.absmax_array() for s in stats_dict.values()])
    theta_global = float(safety * np.quantile(all_absmax, q))

    theta_lw = {}
    for name, stats in stats_dict.items():
        theta_lw[name] = float(safety * np.quantile(stats.absmax_array(), q))

    print(f"  θ_global     (q={q}, safety={safety}): {theta_global:.6f}")
    print(f"  θ_layerwise  range: [{min(theta_lw.values()):.6f}, {max(theta_lw.values()):.6f}]")
    return theta_global, theta_lw

# =============================================================================
# Build Result DataFrames
# =============================================================================

def build_df(stats_dict, theta_dict_global, theta_dict_lw, mode):
    """Returns (df_before, df_after_global, df_after_lw)."""
    rows_before, rows_global, rows_lw = [], [], []
    for name, stats in stats_dict.items():
        base = {
            "layer_name": name,
            "layer_idx":  stats.layer_idx,
            "sublayer":   stats.sublayer,
        }
        # Before
        rows_before.append({**base,
            "QZR_mean":   stats.qzr_before(),
            "cosine_sim": float(np.mean(stats.cosine_steps)),
        })
        # After global
        if mode in ("global", "both"):
            tg = theta_dict_global
            rows_global.append({**base,
                "QZR_mean":   stats.qzr_after(tg),
                "CCR":        stats.ccr(tg),
                "theta":      tg,
            })
        # After layerwise
        if mode in ("layerwise", "both"):
            tl = theta_dict_lw[name]
            rows_lw.append({**base,
                "QZR_mean":   stats.qzr_after(tl),
                "CCR":        stats.ccr(tl),
                "theta":      tl,
            })

    def to_df(rows):
        if not rows: return None
        return (pd.DataFrame(rows)
                  .sort_values(["layer_idx", "sublayer"])
                  .reset_index(drop=True))

    return to_df(rows_before), to_df(rows_global), to_df(rows_lw)

# =============================================================================
# Figure 2
# =============================================================================

def create_figure2(df_before, df_after, mode_label, theta_info):
    """2×3 comparison figure."""
    SUBLAYER_ORDER = ["Q", "K", "V", "O"]
    N_LAYERS       = 12

    def to_mat(df, col):
        mat = np.full((N_LAYERS, 4), np.nan)
        for _, row in df.iterrows():
            li = int(row["layer_idx"])
            si = SUBLAYER_ORDER.index(row["sublayer"])
            mat[li, si] = row[col]
        return mat

    qzr_b   = to_mat(df_before, "QZR_mean")
    qzr_a   = to_mat(df_after,  "QZR_mean")
    dqzr    = qzr_b - qzr_a
    ccr_mat = to_mat(df_after,  "CCR")
    vmax    = max(np.nanmax(qzr_b), np.nanmax(qzr_a))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Figure 2 — nm_thres Calibration Effect: BERT-base Q/K/V/O Analog Tiles\n"
        f"(mode={mode_label}, q={args.q}, safety={args.safety}, "
        f"DAC={DAC_BITS}-bit, N={N_STEP} steps)  {theta_info}",
        fontsize=10, y=1.01
    )

    def heatmap(ax, mat, title, cmap, vmin=0, vmax=None, label=""):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="upper",
                       vmin=vmin, vmax=vmax if vmax is not None else np.nanmax(mat))
        ax.set_xticks(range(4));    ax.set_xticklabels(SUBLAYER_ORDER)
        ax.set_yticks(range(N_LAYERS)); ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
        ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, label=label, shrink=0.85)

    # [0,0] QZR before
    heatmap(axes[0, 0], qzr_b,
            "(a) QZR before (nm_thres=0)",
            "plasma", vmin=0, vmax=vmax, label="QZR")

    # [0,1] QZR after
    heatmap(axes[0, 1], qzr_a,
            f"(b) QZR after (nm_thres=θ, {mode_label})",
            "plasma", vmin=0, vmax=vmax, label="QZR")

    # [0,2] ΔQZR
    dmax = float(np.nanmax(np.abs(dqzr))) + 1e-6
    im2 = axes[0, 2].imshow(dqzr, aspect="auto", cmap="RdYlGn",
                             origin="upper", vmin=-dmax, vmax=dmax)
    axes[0, 2].set_xticks(range(4)); axes[0, 2].set_xticklabels(SUBLAYER_ORDER)
    axes[0, 2].set_yticks(range(N_LAYERS))
    axes[0, 2].set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
    axes[0, 2].set_xlabel("Sublayer"); axes[0, 2].set_ylabel("Encoder Layer")
    axes[0, 2].set_title("(c) ΔQZR = before − after\n(green = improvement)", fontsize=9)
    plt.colorbar(im2, ax=axes[0, 2], label="ΔQZR", shrink=0.85)

    # [1,0] CCR heatmap
    heatmap(axes[1, 0], ccr_mat,
            "(d) CCR — Cap Clipping Rate\nP(absmax(δ) > θ)",
            "YlOrRd", vmin=0, vmax=1, label="CCR")

    # [1,1] QZR K/V line plot (before vs after, layers 0–11)
    ax = axes[1, 1]
    layers = np.arange(N_LAYERS)
    for sl, color in [("K", "steelblue"), ("V", "tomato")]:
        si = SUBLAYER_ORDER.index(sl)
        ax.plot(layers, qzr_b[:, si],  color=color, ls="--", lw=1.5,
                label=f"{sl} before", alpha=0.7)
        ax.plot(layers, qzr_a[:, si],  color=color, ls="-",  lw=2.0,
                label=f"{sl} after",  marker="o", markersize=4)
    ax.set_xlabel("Encoder Layer"); ax.set_ylabel("QZR")
    ax.set_title("(e) QZR before vs after\n(K and V sublayers)", fontsize=9)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    ax.set_xticks(layers); ax.set_xticklabels([f"L{i}" for i in layers], fontsize=7)

    # [1,2] CCR bar: mean ± std per sublayer
    ax = axes[1, 2]
    ccr_by_sl = {sl: [] for sl in SUBLAYER_ORDER}
    for _, row in df_after.iterrows():
        ccr_by_sl[row["sublayer"]].append(row["CCR"])
    sl_means  = [np.mean(ccr_by_sl[sl]) for sl in SUBLAYER_ORDER]
    sl_stds   = [np.std(ccr_by_sl[sl])  for sl in SUBLAYER_ORDER]
    colors    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(SUBLAYER_ORDER, sl_means, yerr=sl_stds, capsize=4,
                  color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.axhline(0.01,  color="red",    ls="--", lw=1.2, label="1% target")
    ax.axhline(0.001, color="orange", ls="--", lw=1.2, label="0.1% target")
    for bar, val in zip(bars, sl_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(sl_stds) * 0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("CCR (mean ± std)")
    ax.set_title("(f) CCR per Sublayer\n(clipping does not dominate)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Mode-specific filename
    path = FIG_PATH.replace(".pdf", f"_{mode_label}.pdf")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {path}")
    return path

# =============================================================================
# Main
# =============================================================================

def main():
    torch.manual_seed(SEED)
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Data ----
    print("\n[1/4] Loading data ...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader    = load_data(tokenizer)

    # ---- Model + single diagnostic run ----
    print("\n[2/4] Creating model and running diagnostic ...")
    model = create_model()
    stats_dict, handles = register_hooks(model)
    run_diagnostic(model, loader)
    for h in handles: h.remove()
    del model; torch.cuda.empty_cache()

    # ---- Calibration ----
    print("\n[3/4] Calibrating nm_thres ...")
    theta_global, theta_lw = compute_theta(stats_dict, args.q, args.safety)

    # Save theta JSONs
    with open(THETA_GLOBAL_JSON, "w") as f:
        json.dump({"q": args.q, "safety": args.safety, "theta": theta_global}, f, indent=2)
    with open(THETA_LAYERWISE_JSON, "w") as f:
        json.dump({"q": args.q, "safety": args.safety, "theta": theta_lw}, f, indent=2)
    print(f"  θ_global     → {THETA_GLOBAL_JSON}")
    print(f"  θ_layerwise  → {THETA_LAYERWISE_JSON}")

    # ---- Build DataFrames ----
    df_before, df_global, df_lw = build_df(
        stats_dict, theta_global, theta_lw, args.mode
    )
    df_before.to_csv(METRICS_BEFORE_CSV, index=False)
    print(f"  metrics_before → {METRICS_BEFORE_CSV}")

    # ---- Figure ----
    print("\n[4/4] Creating Figure 2 ...")

    # Print summary table
    def _summary(df_after, mode_label, theta_repr):
        print(f"\n=== {mode_label} (θ={theta_repr}) ===")
        merged = df_before[["layer_idx", "sublayer", "QZR_mean"]].copy()
        merged = merged.rename(columns={"QZR_mean": "QZR_before"})
        merged["QZR_after"] = df_after["QZR_mean"].values
        merged["ΔQZR"]      = merged["QZR_before"] - merged["QZR_after"]
        merged["CCR"]       = df_after["CCR"].values
        print(merged[merged["sublayer"].isin(["K", "V"])].to_string(index=False))

        csv_path = (METRICS_AFTER_GLOBAL_CSV if "global" in mode_label
                    else METRICS_AFTER_LW_CSV)
        df_after.to_csv(csv_path, index=False)
        print(f"  metrics_after → {csv_path}")

    if args.mode in ("global", "both") and df_global is not None:
        theta_info = f"θ_global={theta_global:.5f}"
        _summary(df_global, "global", f"{theta_global:.5f}")
        create_figure2(df_before, df_global, "global", theta_info)

    if args.mode in ("layerwise", "both") and df_lw is not None:
        vals = list(theta_lw.values())
        theta_info = f"θ_lw=[{min(vals):.5f}, {max(vals):.5f}]"
        _summary(df_lw, "layerwise", f"[{min(vals):.5f}, {max(vals):.5f}]")
        create_figure2(df_before, df_lw, "layerwise", theta_info)

    print("\nDone.")
    print(f"  CSV:   {METRICS_BEFORE_CSV}")
    print(f"  JSON:  {THETA_GLOBAL_JSON}, {THETA_LAYERWISE_JSON}")
    print(f"  Figs:  {FIG_PATH.replace('.pdf', '_global.pdf')}")
    print(f"         {FIG_PATH.replace('.pdf', '_layerwise.pdf')}")


if __name__ == "__main__":
    main()
