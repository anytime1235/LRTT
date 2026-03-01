"""calib_nm_thres_tworun.py — Two-Run nm_thres Calibration Experiment

Key insight: A single-run analytical approach (calib_nm_thres.py) cannot capture the
cascade effect: when nm_thres clips δ in upper layers, the δ arriving at lower layers
changes. This two-run approach measures the real cascade-affected QZR.

- Run 1 (nm_thres=0): collect baseline δ distribution → compute θ
- Run 2 (nm_thres=θ): same seed/data → measure actual cascade-affected QZR

Expected result: Run 2 QZR_K/V ≈ Run 1 QZR_K/V (ΔQZR < 0.01), confirming nm_thres
cannot resolve the structural K/V sparsity problem.

Usage:
  python calib_nm_thres_tworun.py                            # default
  python calib_nm_thres_tworun.py --n-step 5 --batch-size 2  # smoke test
  python calib_nm_thres_tworun.py --q 0.99 --safety 1.1
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
parser.add_argument("--q",          type=float, default=0.99,  help="Quantile for theta")
parser.add_argument("--safety",     type=float, default=1.1,   help="Safety factor")
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

OUT_DIR         = "/data/results/tikitakav1"
METRICS_R1_CSV  = f"{OUT_DIR}/tworun_metrics_run1.csv"
METRICS_R2_CSV  = f"{OUT_DIR}/tworun_metrics_run2.csv"
THETA_JSON      = f"{OUT_DIR}/tworun_theta.json"
FIG_PATH        = f"{OUT_DIR}/fig_nm_thres_tworun.pdf"

print(f"[Config] Device={DEVICE}, N_STEP={N_STEP}, BATCH={DIAG_BATCH_SIZE}")
print(f"[Config] q={args.q}, safety={args.safety}")

# =============================================================================
# RPU Config (parameterized by nm_thres)
# =============================================================================

def create_rpu_config(nm_thres: float = 0.0):
    """SingleRPUConfig + IdealDevice with explicit backward IOParams.
    nm_thres=0: baseline (no cap). nm_thres=θ: Run 2 with calibrated cap.
    """
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
    rpu.backward.nm_thres               = nm_thres
    rpu.mapping.digital_bias            = True
    rpu.mapping.weight_scaling_omega    = 1.0
    rpu.mapping.weight_scaling_columnwise = True
    return rpu

# =============================================================================
# Layer Name Utilities
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
    if m is None:
        return None
    return int(m.group(1)), _SUBLAYER_MAP[m.group(2)]


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

# =============================================================================
# Model Creation (parameterized by nm_thres)
# =============================================================================

def create_model(nm_thres: float = 0.0):
    """2-pass conversion:
    - Pass 1: Q/K/V/O attention → IdealDevice (nm_thres=given)
    - Pass 2: non-target encoder → IdealDevice frozen (noop update)
    - qa_outputs, pooler: digital
    """
    from aihwkit.nn import AnalogLinear
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.optim.context import AnalogContext

    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    target, nontarget, all_linear = _layer_names(model)

    # Pass 1: target layers
    rpu = create_rpu_config(nm_thres=nm_thres)
    model = convert_to_analog(model, rpu,
                              exclude_modules=[n for n in all_linear if n not in target])

    # Pass 2: non-target (frozen)
    nt_rpu = create_rpu_config(nm_thres=nm_thres)
    model = convert_to_analog(model, nt_rpu,
                              exclude_modules=[n for n in all_linear if n not in nontarget])

    # Freeze non-target tile updates (noop)
    def _noop(x, d, *a, **kw):
        return None

    for name, m in model.named_modules():
        if isinstance(m, AnalogLinear) and name not in target:
            for tile in m.analog_tiles():
                tile.update = _noop

    # requires_grad: only AnalogContext + qa_outputs
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        if isinstance(p, AnalogContext):
            p.requires_grad_(True)
    for n, p in model.named_parameters():
        if "qa_outputs" in n:
            p.requires_grad_(True)

    n_t   = sum(1 for n, m in model.named_modules()
                if isinstance(m, AnalogLinear) and n in target)
    n_all = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog tiles — target: {n_t}, frozen: {n_all - n_t}, nm_thres={nm_thres}")
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
# LayerStats (simplified — no theta grid, single nm_thres)
# =============================================================================

class LayerStats:
    """Per-layer backward stats for one run (no theta grid needed)."""

    def __init__(self, name, layer_idx, sublayer, nm_thres=0.0):
        self.name      = name
        self.layer_idx = layer_idx
        self.sublayer  = sublayer
        self.nm_thres  = nm_thres
        self.dac_step  = 2 * INP_BOUND / (2**DAC_BITS - 1)
        self.eps       = 1e-8

        self.odr_steps = []
        self.qzr_steps = []
        self.ccr_steps = []
        self._absmax_buf = []

    def update(self, dy: torch.Tensor):
        with torch.no_grad():
            dy_flat  = (dy.detach().reshape(-1, dy.shape[-1]).float()
                        if dy.dim() == 3 else dy.detach().float())
            abs_dy   = dy_flat.abs()
            absmax_v = abs_dy.max(dim=1).values   # (N,)

            # ODR: absmax / median per vector
            absmed_v = abs_dy.median(dim=1).values
            odr = (absmax_v / (absmed_v + self.eps)).mean().item()
            self.odr_steps.append(odr)

            # CCR: P(absmax > nm_thres); 0 if nm_thres==0
            if self.nm_thres > 0:
                ccr = (absmax_v > self.nm_thres).float().mean().item()
            else:
                ccr = 0.0
            self.ccr_steps.append(ccr)

            # QZR with nm_thres-capped alpha
            if self.nm_thres > 0:
                alpha = absmax_v.clamp(max=self.nm_thres).unsqueeze(1).clamp(min=self.eps)
            else:
                alpha = absmax_v.unsqueeze(1).clamp(min=self.eps)
            scaled = dy_flat / alpha * INP_BOUND
            qzr = (scaled.abs() < self.dac_step / 2).float().mean().item()
            self.qzr_steps.append(qzr)

            self._absmax_buf.append(absmax_v.cpu().float().numpy())

    def absmax_array(self):
        return np.concatenate(self._absmax_buf)

    def summary(self):
        return {
            "layer_name":  self.name,
            "layer_idx":   self.layer_idx,
            "sublayer":    self.sublayer,
            "nm_thres":    self.nm_thres,
            "ODR_mean":    float(np.mean(self.odr_steps)),
            "QZR_mean":    float(np.mean(self.qzr_steps)),
            "CCR_mean":    float(np.mean(self.ccr_steps)),
            "absmax_q50":  float(np.quantile(self.absmax_array(), 0.50)),
            "absmax_q99":  float(np.quantile(self.absmax_array(), 0.99)),
            "absmax_q999": float(np.quantile(self.absmax_array(), 0.999)),
        }

# =============================================================================
# Hook Registration (parameterized by nm_thres)
# =============================================================================

def register_hooks(model, nm_thres=0.0):
    from aihwkit.nn import AnalogLinear

    stats_dict, handles = {}, []
    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear):
            continue
        parsed = parse_layer_name(name)
        if parsed is None:
            continue
        layer_idx, sublayer = parsed
        stats = LayerStats(name=name, layer_idx=layer_idx,
                           sublayer=sublayer, nm_thres=nm_thres)
        stats_dict[name] = stats

        def make_hook(s):
            def fn(mod, gin, gout):
                if gout[0] is not None:
                    s.update(gout[0])
            return fn

        handles.append(module.register_full_backward_hook(make_hook(stats)))

    print(f"[Hook] {len(stats_dict)} hooks, nm_thres={nm_thres}")
    return stats_dict, handles

# =============================================================================
# Diagnostic Run
# =============================================================================

def run_diagnostic(model, loader):
    """N_STEP forward+backward passes with AnalogSGD(lr=0) to flush tile buffers."""
    from aihwkit.optim import AnalogSGD

    optimizer = AnalogSGD(model.parameters(), lr=0.0)
    model.train()
    torch.manual_seed(SEED)

    for step, batch in enumerate(tqdm(loader, total=N_STEP, desc="Diagnostic")):
        if step >= N_STEP:
            break
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
# Compute Theta
# =============================================================================

def compute_theta(stats_dict, q=0.99, safety=1.1):
    """Global θ: q-quantile of pooled absmax × safety factor."""
    all_absmax = np.concatenate([s.absmax_array() for s in stats_dict.values()])
    theta = float(safety * np.quantile(all_absmax, q))
    print(f"  θ_global (q={q}, safety={safety}): {theta:.6f}")
    return theta

# =============================================================================
# Build DataFrame
# =============================================================================

def to_df(stats_dict):
    rows = [s.summary() for s in stats_dict.values()]
    return (pd.DataFrame(rows)
              .sort_values(["layer_idx", "sublayer"])
              .reset_index(drop=True))

# =============================================================================
# Print Comparison
# =============================================================================

def print_comparison(df_r1, df_r2, theta):
    print(f"\n=== Two-Run Comparison (θ={theta:.6f}) ===")
    merged = df_r1[["layer_idx", "sublayer", "QZR_mean"]].copy()
    merged = merged.rename(columns={"QZR_mean": "QZR_R1"})
    merged["QZR_R2"] = df_r2["QZR_mean"].values
    merged["ΔQZR"]   = merged["QZR_R1"] - merged["QZR_R2"]
    merged["CCR_R2"] = df_r2["CCR_mean"].values

    print("\n--- K and V sublayers ---")
    kv = merged[merged["sublayer"].isin(["K", "V"])]
    print(kv.to_string(index=False))

    print("\n--- All sublayers (mean per type) ---")
    by_sl = merged.groupby("sublayer")[["QZR_R1", "QZR_R2", "ΔQZR", "CCR_R2"]].mean()
    print(by_sl.to_string())

# =============================================================================
# Figure (2×3 layout)
# =============================================================================

def create_figure(df_r1, df_r2, theta):
    """
    Figure: Two-Run nm_thres Calibration Comparison
    Layout (2 rows × 3 cols):
      [0,0] QZR heatmap — Run 1 (nm_thres=0)
      [0,1] QZR heatmap — Run 2 (nm_thres=θ)
      [0,2] ΔQZR heatmap (R1 − R2, green = improvement)
      [1,0] CCR heatmap — Run 2
      [1,1] QZR K/V line: before vs after (layers 0–11)
      [1,2] CCR bar: mean ± std per sublayer (Run 2)
    """
    SUBLAYER_ORDER = ["Q", "K", "V", "O"]
    N_LAYERS       = 12

    def to_mat(df, col):
        mat = np.full((N_LAYERS, 4), np.nan)
        for _, row in df.iterrows():
            li = int(row["layer_idx"])
            si = SUBLAYER_ORDER.index(row["sublayer"])
            mat[li, si] = row[col]
        return mat

    qzr_r1  = to_mat(df_r1, "QZR_mean")
    qzr_r2  = to_mat(df_r2, "QZR_mean")
    dqzr    = qzr_r1 - qzr_r2
    ccr_mat = to_mat(df_r2, "CCR_mean")
    vmax    = max(float(np.nanmax(qzr_r1)), float(np.nanmax(qzr_r2)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Figure 2 — Two-Run nm_thres Calibration: BERT-base Q/K/V/O Analog Tiles\n"
        f"(q={args.q}, safety={args.safety}, θ={theta:.5f}, "
        f"DAC={DAC_BITS}-bit, N={N_STEP} steps, batch={DIAG_BATCH_SIZE})",
        fontsize=10, y=1.01
    )

    def heatmap(ax, mat, title, cmap, vmin=0, vmax_val=None, label=""):
        vmax_use = vmax_val if vmax_val is not None else float(np.nanmax(mat))
        im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="upper",
                       vmin=vmin, vmax=vmax_use)
        ax.set_xticks(range(4));        ax.set_xticklabels(SUBLAYER_ORDER)
        ax.set_yticks(range(N_LAYERS)); ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
        ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, label=label, shrink=0.85)

    # [0,0] QZR Run 1
    heatmap(axes[0, 0], qzr_r1,
            "(a) QZR — Run 1 (nm_thres=0)",
            "plasma", vmin=0, vmax_val=vmax, label="QZR")

    # [0,1] QZR Run 2
    heatmap(axes[0, 1], qzr_r2,
            f"(b) QZR — Run 2 (nm_thres=θ={theta:.4f})",
            "plasma", vmin=0, vmax_val=vmax, label="QZR")

    # [0,2] ΔQZR
    dmax = float(np.nanmax(np.abs(dqzr))) + 1e-6
    im2 = axes[0, 2].imshow(dqzr, aspect="auto", cmap="RdYlGn",
                             origin="upper", vmin=-dmax, vmax=dmax)
    axes[0, 2].set_xticks(range(4));        axes[0, 2].set_xticklabels(SUBLAYER_ORDER)
    axes[0, 2].set_yticks(range(N_LAYERS)); axes[0, 2].set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
    axes[0, 2].set_xlabel("Sublayer");      axes[0, 2].set_ylabel("Encoder Layer")
    axes[0, 2].set_title("(c) ΔQZR = R1 − R2\n(green = improvement)", fontsize=9)
    plt.colorbar(im2, ax=axes[0, 2], label="ΔQZR", shrink=0.85)

    # Annotate K and V ΔQZR means
    kv_rows = df_r1[df_r1["sublayer"].isin(["K", "V"])]
    for _, row in kv_rows.iterrows():
        li = int(row["layer_idx"])
        si = SUBLAYER_ORDER.index(row["sublayer"])
        delta = dqzr[li, si]
        if not np.isnan(delta):
            axes[0, 2].text(si, li, f"{delta:.3f}", ha="center", va="center",
                            fontsize=5, color="black")

    # [1,0] CCR heatmap Run 2
    heatmap(axes[1, 0], ccr_mat,
            "(d) CCR — Run 2 Cap Clipping Rate\nP(absmax(δ) > θ)",
            "YlOrRd", vmin=0, vmax_val=1.0, label="CCR")

    # [1,1] QZR K/V line plot
    ax = axes[1, 1]
    layers = np.arange(N_LAYERS)
    for sl, color in [("K", "steelblue"), ("V", "tomato")]:
        si = SUBLAYER_ORDER.index(sl)
        ax.plot(layers, qzr_r1[:, si], color=color, ls="--", lw=1.5,
                label=f"{sl} Run 1", alpha=0.7)
        ax.plot(layers, qzr_r2[:, si], color=color, ls="-",  lw=2.0,
                label=f"{sl} Run 2", marker="o", markersize=4)
    ax.set_xlabel("Encoder Layer"); ax.set_ylabel("QZR")
    ax.set_title("(e) QZR K/V: Run 1 vs Run 2\n(lines should overlap → nm_thres insufficient)", fontsize=9)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    ax.set_xticks(layers); ax.set_xticklabels([f"L{i}" for i in layers], fontsize=7)

    # [1,2] CCR bar per sublayer (Run 2)
    ax = axes[1, 2]
    ccr_by_sl = {sl: [] for sl in SUBLAYER_ORDER}
    for _, row in df_r2.iterrows():
        ccr_by_sl[row["sublayer"]].append(row["CCR_mean"])
    sl_means = [np.mean(ccr_by_sl[sl]) for sl in SUBLAYER_ORDER]
    sl_stds  = [np.std(ccr_by_sl[sl])  for sl in SUBLAYER_ORDER]
    colors   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(SUBLAYER_ORDER, sl_means, yerr=sl_stds, capsize=4,
                  color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.axhline(0.01,  color="red",    ls="--", lw=1.2, label="1% target")
    ax.axhline(0.001, color="orange", ls="--", lw=1.2, label="0.1% target")
    for bar, val in zip(bars, sl_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(sl_stds) * 0.05 + 1e-4,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("CCR (mean ± std)")
    ax.set_title("(f) CCR per Sublayer — Run 2\n(K/V clipping rate → cascade effect?)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {FIG_PATH}")

# =============================================================================
# Main
# =============================================================================

def main():
    torch.manual_seed(SEED)
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Data ----
    print("\n[1/5] Loading data ...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader    = load_data(tokenizer)

    # ---- Run 1: nm_thres = 0 (baseline) ----
    print("\n[2/5] Run 1: nm_thres=0 (baseline) ...")
    model = create_model(nm_thres=0.0)
    stats_r1, handles = register_hooks(model, nm_thres=0.0)
    run_diagnostic(model, loader)
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    # Save Run 1 CSV
    df_r1 = to_df(stats_r1)
    df_r1.to_csv(METRICS_R1_CSV, index=False)
    print(f"  Run 1 metrics → {METRICS_R1_CSV}")

    # ---- Compute theta ----
    print("\n[3/5] Computing theta ...")
    theta = compute_theta(stats_r1, q=args.q, safety=args.safety)
    with open(THETA_JSON, "w") as f:
        json.dump({"q": args.q, "safety": args.safety, "theta": theta}, f, indent=2)
    print(f"  θ → {THETA_JSON}")

    # ---- Run 2: nm_thres = theta (calibrated) ----
    print(f"\n[4/5] Run 2: nm_thres={theta:.6f} (calibrated) ...")
    model2 = create_model(nm_thres=theta)
    stats_r2, handles2 = register_hooks(model2, nm_thres=theta)
    run_diagnostic(model2, loader)    # same loader → same batches
    for h in handles2:
        h.remove()
    del model2
    torch.cuda.empty_cache()

    # Save Run 2 CSV
    df_r2 = to_df(stats_r2)
    df_r2.to_csv(METRICS_R2_CSV, index=False)
    print(f"  Run 2 metrics → {METRICS_R2_CSV}")

    # ---- Summary + Figure ----
    print("\n[5/5] Summary and figure ...")
    print_comparison(df_r1, df_r2, theta)
    create_figure(df_r1, df_r2, theta)

    print("\nDone.")
    print(f"  Run 1 CSV:  {METRICS_R1_CSV}")
    print(f"  Run 2 CSV:  {METRICS_R2_CSV}")
    print(f"  Theta JSON: {THETA_JSON}")
    print(f"  Figure:     {FIG_PATH}")


if __name__ == "__main__":
    main()
