"""diag_backward_outlier.py — Backward Outlier Diagnosis for AIMC BERT-base

Goal: Prove that backward outliers in analog tile (Q/K/V/O) layers damage training
by measuring ODR (Outlier Dominance Ratio) and QZR (Quantization Zero Rate).

Key mechanism:
  - AbsMax noise_management scales per-vector (δ vectors)
  - Outlier-dominant vectors cause small gradients to round to 0 in DAC (QZR)
  - backward bound_management is IGNORED by AIHWKit (by design)
  → outlier tile updates are silenced with no automatic recovery

Usage:
  python diag_backward_outlier.py                    # full run (N_STEP=200)
  python diag_backward_outlier.py --n-step 5 --batch-size 2  # smoke test
"""

import argparse
import os
import re
import sys

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
parser.add_argument("--n-step",    type=int, default=200)
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

# =============================================================================
# Config Constants
# =============================================================================

N_STEP          = args.n_step
DIAG_BATCH_SIZE = args.batch_size
MAX_SEQ_LENGTH  = 384
DOC_STRIDE      = 128
SEED            = 42

DAC_BITS   = 7        # inp_res = 1/(2**7-2)
ADC_BITS   = 9        # out_res = 1/(2**9-2)
INP_BOUND  = 1.0      # backward.inp_bound default
TARGET_DEVICE = "ideal"  # see create_diag_rpu_config

OUT_DIR  = "/data/results/tikitakav1"
CSV_PATH = f"{OUT_DIR}/metrics_backward_outlier.csv"
FIG_PATH = f"{OUT_DIR}/fig_backward_outlier_diagnosis.pdf"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Device={DEVICE}, N_STEP={N_STEP}, BATCH={DIAG_BATCH_SIZE}")

# =============================================================================
# Step 2: RPU Config — Baseline (AbsMax, nm_thres=0)
# =============================================================================

def create_diag_rpu_config():
    """SingleRPU + IdealDevice with EXPLICIT baseline backward IOParameters.

    Note: backward bound_management is IGNORED by AIHWKit (by design).
    Outliers in δ cannot be recovered by iterative re-scaling.
    This is the diagnostic baseline: AbsMax, nm_thres=0 (no cap).
    """
    from aihwkit.simulator.configs import SingleRPUConfig, IOParameters
    from aihwkit.simulator.configs.devices import IdealDevice
    from aihwkit.simulator.configs.utils import NoiseManagementType, BoundManagementType

    rpu_config = SingleRPUConfig(device=IdealDevice())

    # Forward: standard settings (diagnostic target is backward)
    rpu_config.forward.inp_bound = INP_BOUND
    rpu_config.forward.inp_res   = 1 / (2**DAC_BITS - 2)
    rpu_config.forward.out_bound = 12.0
    rpu_config.forward.out_res   = 1 / (2**ADC_BITS - 2)
    rpu_config.forward.noise_management = NoiseManagementType.ABS_MAX
    rpu_config.forward.out_noise = 0.0

    # Backward: EXPLICIT baseline — AbsMax, no threshold cap
    # bound_management is set but will be IGNORED by AIHWKit (documented behavior)
    rpu_config.backward.inp_bound = INP_BOUND
    rpu_config.backward.inp_res   = 1 / (2**DAC_BITS - 2)   # 7-bit DAC
    rpu_config.backward.out_bound = 12.0
    rpu_config.backward.out_res   = 1 / (2**ADC_BITS - 2)   # 9-bit ADC
    rpu_config.backward.noise_management = NoiseManagementType.ABS_MAX
    rpu_config.backward.nm_thres  = 0.0   # no cap → baseline
    rpu_config.backward.out_noise = 0.0

    rpu_config.mapping.digital_bias              = True
    rpu_config.mapping.weight_scaling_omega      = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True

    return rpu_config


# =============================================================================
# Step 3: Model Creation
# =============================================================================

def create_diag_model():
    """BERT-base 2-pass analog conversion:
    - Target (Q/K/V/O): IdealDevice (trainable analog, backward stats collection)
    - Non-target encoder: IdealDevice frozen (analog forward, weight update blocked)
    - qa_outputs, pooler: digital (unchanged)

    Non-target layers are converted to IdealDevice so the backward pass
    traverses analog tiles → realistic gradient flow.
    """
    from aihwkit.nn import AnalogLinear
    from aihwkit.nn.conversion import convert_to_analog

    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")

    always_digital = ["qa_outputs", "pooler"]
    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    # Target: Q/K/V/O attention layers
    target_names = [
        n for n in all_linear
        if "encoder" in n and "attention" in n
        and not any(d in n for d in always_digital)
    ]

    # Non-target: remaining encoder linear layers (FFN etc.)
    nontarget_names = [
        n for n in all_linear
        if "encoder" in n and n not in target_names
        and not any(d in n for d in always_digital)
    ]

    # --- Pass 1: Target → IdealDevice (diag RPU config) ---
    rpu_config = create_diag_rpu_config()
    exclude_p1 = [n for n in all_linear if n not in target_names]
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude_p1)

    # --- Pass 2: Non-target → IdealDevice frozen ---
    nt_rpu_config = create_diag_rpu_config()
    exclude_p2 = [n for n in all_linear if n not in nontarget_names]
    model = convert_to_analog(model, nt_rpu_config, exclude_modules=exclude_p2)

    # Freeze non-target tile updates (noop)
    def _noop_update(x, d, *a, **kw):
        return None

    for name, m in model.named_modules():
        if isinstance(m, AnalogLinear) and name not in target_names:
            for tile in m.analog_tiles():
                tile.update = _noop_update

    # requires_grad: only qa_outputs + AnalogContext (for gradient flow)
    from aihwkit.optim.context import AnalogContext
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        if isinstance(p, AnalogContext):
            p.requires_grad_(True)
    for n, p in model.named_parameters():
        if "qa_outputs" in n:
            p.requires_grad_(True)

    n_target = sum(
        1 for n, m in model.named_modules()
        if isinstance(m, AnalogLinear) and n in target_names
    )
    n_total = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog tiles — target(Q/K/V/O): {n_target}, "
          f"non-target(frozen): {n_total - n_target}")

    return model.to(DEVICE)


# =============================================================================
# Step 4: Data Loading
# =============================================================================

def load_diag_data(tokenizer):
    """SQuAD v1.1 — first N_STEP batches. Seed-fixed for reproducibility."""

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

        start_positions, end_positions = [], []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer     = answers[sample_idx]

            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char   = start_char + len(answer["text"][0])

            sequence_ids  = inputs.sequence_ids(i)
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"]   = end_positions
        return inputs

    raw_datasets   = load_dataset("squad")
    tokenized_train = raw_datasets["train"].map(
        preprocess_train, batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    n_samples = min(N_STEP * DIAG_BATCH_SIZE, len(tokenized_train))
    subset    = tokenized_train.shuffle(seed=SEED).select(range(n_samples))
    loader    = DataLoader(subset, batch_size=DIAG_BATCH_SIZE, shuffle=False,
                           collate_fn=default_data_collator)
    print(f"  Dataset: {n_samples} samples → {len(loader)} batches")
    return loader


# =============================================================================
# Step 5: LayerStats — Streaming Accumulation
# =============================================================================

class LayerStats:
    """Per-layer backward gradient statistics accumulator."""

    def __init__(self, name: str, layer_idx: int, sublayer: str):
        self.name       = name
        self.layer_idx  = layer_idx   # int 0..11
        self.sublayer   = sublayer    # "Q" | "K" | "V" | "O"

        # DAC quantization step size
        self.dac_step = 2 * INP_BOUND / (2**DAC_BITS - 1)
        self.eps      = 1e-8

        # Per-step scalar lists (length = N_STEP)
        self.odr_steps    = []
        self.qzr_steps    = []
        self.p_clip_steps = []
        self.cosine_steps = []
        self.q50_steps    = []
        self.q90_steps    = []
        self.q99_steps    = []
        self.q999_steps   = []

        # Buffer for ECDF: list of np arrays
        self._absmax_buf  = []

    def update(self, dy: torch.Tensor):
        """Called from backward hook. dy = grad_output[0] = δ."""
        with torch.no_grad():
            # Reshape to (N_vec, D): handle 3D (B, S, D) and 2D (B*S, D)
            if dy.dim() == 3:
                dy_flat = dy.detach().reshape(-1, dy.shape[-1]).float()
            else:
                dy_flat = dy.detach().float()

            N, D     = dy_flat.shape
            abs_dy   = dy_flat.abs()                           # (N, D)
            absmax_v = abs_dy.max(dim=1).values                # (N,)
            absmed_v = abs_dy.median(dim=1).values             # (N,)

            # A) ODR: absmax / median per vector → mean across N
            odr = (absmax_v / (absmed_v + self.eps)).mean().item()
            self.odr_steps.append(odr)

            # B) Quantiles of absmax across N vectors
            absmax_sorted = absmax_v.sort().values
            n = len(absmax_sorted)
            self.q50_steps.append(absmax_sorted[max(0, int(0.50 * n) - 1)].item())
            self.q90_steps.append(absmax_sorted[max(0, int(0.90 * n) - 1)].item())
            self.q99_steps.append(absmax_sorted[max(0, int(0.99 * n) - 1)].item())
            self.q999_steps.append(absmax_sorted[min(int(0.999 * n), n - 1)].item())

            # C) p_clip_in: P(|δ| > inp_bound) element-wise
            p_clip = (abs_dy > INP_BOUND).float().mean().item()
            self.p_clip_steps.append(p_clip)

            # D) QZR: fraction that rounds to 0 after AbsMax DAC quantization
            #    α = absmax per vector; scale into [-inp_bound, inp_bound]
            alpha  = absmax_v.unsqueeze(1).clamp(min=self.eps)    # (N, 1)
            scaled = dy_flat / alpha * INP_BOUND                   # (N, D)
            qzr    = (scaled.abs() < self.dac_step / 2).float().mean().item()
            self.qzr_steps.append(qzr)

            # E) Cosine similarity: FP32 vs DAC-quantized
            dy_q_scaled = (scaled / self.dac_step).round() * self.dac_step
            dy_q        = dy_q_scaled * alpha / INP_BOUND
            cos_sim     = torch.nn.functional.cosine_similarity(
                dy_flat, dy_q, dim=1
            ).mean().item()
            self.cosine_steps.append(cos_sim)

            # Store absmax for ECDF (CPU numpy)
            self._absmax_buf.append(absmax_v.cpu().float().numpy())

    def summary(self) -> dict:
        return {
            "layer_name": self.name,
            "layer_idx":  self.layer_idx,
            "sublayer":   self.sublayer,
            "ODR_mean":   float(np.mean(self.odr_steps)),
            "QZR_mean":   float(np.mean(self.qzr_steps)),
            "p_clip_in":  float(np.mean(self.p_clip_steps)),
            "cosine_sim": float(np.mean(self.cosine_steps)),
            "absmax_q50": float(np.mean(self.q50_steps)),
            "absmax_q90": float(np.mean(self.q90_steps)),
            "absmax_q99": float(np.mean(self.q99_steps)),
            "absmax_q999":float(np.mean(self.q999_steps)),
        }

    def absmax_array(self) -> np.ndarray:
        """Full absmax distribution for ECDF plots."""
        return np.concatenate(self._absmax_buf)


# =============================================================================
# Step 6: Layer Name Parsing
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


def parse_layer_name(name: str):
    """Returns (layer_idx: int, sublayer: str) or None if not a target layer."""
    m = _LAYER_RE.search(name)
    if m is None:
        return None
    return int(m.group(1)), _SUBLAYER_MAP[m.group(2)]


# =============================================================================
# Step 7: Hook Registration
# =============================================================================

def register_hooks(model) -> tuple:
    """Register full backward hooks on all AnalogLinear Q/K/V/O modules.
    Returns (stats_dict: {name: LayerStats}, hook_handles: list).
    """
    from aihwkit.nn import AnalogLinear

    stats_dict = {}
    handles    = []

    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear):
            continue
        parsed = parse_layer_name(name)
        if parsed is None:
            continue
        layer_idx, sublayer = parsed
        stats = LayerStats(name=name, layer_idx=layer_idx, sublayer=sublayer)
        stats_dict[name] = stats

        def make_hook(s):
            def hook_fn(mod, grad_input, grad_output):
                if grad_output[0] is not None:
                    s.update(grad_output[0])
            return hook_fn

        handle = module.register_full_backward_hook(make_hook(stats))
        handles.append(handle)

    print(f"[Hook] Registered {len(stats_dict)} backward hooks: "
          f"{sorted(set(s.sublayer for s in stats_dict.values()))}")
    return stats_dict, handles


# =============================================================================
# Step 8: Diagnostic Run Loop
# =============================================================================

def run_diagnostic(model, loader, stats_dict):
    """N_STEP forward+backward passes. AnalogSGD(lr=0) used to properly flush
    analog tile gradient state each step — without lr=0 the tile update is a
    no-op but the internal analog_grad_output buffer is correctly cleared."""
    from aihwkit.optim import AnalogSGD

    # lr=0 → weight update = 0, but tile.update() is called → buffers flushed
    optimizer = AnalogSGD(model.parameters(), lr=0.0)

    model.train()
    torch.manual_seed(SEED)

    for step, batch in enumerate(tqdm(loader, desc="Diagnostic", total=N_STEP)):
        if step >= N_STEP:
            break

        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        start_pos      = batch["start_positions"].to(DEVICE)
        end_pos        = batch["end_positions"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_pos,
            end_positions=end_pos,
        )
        outputs.loss.backward()
        optimizer.step()  # flushes analog tile grad buffers; lr=0 → no weight change


# =============================================================================
# Step 9: Figure Creation (2×3 layout)
# =============================================================================

def create_figure(df: pd.DataFrame, stats_dict: dict):
    """
    Figure 1 — Backward Gradient Outlier Diagnosis in AIMC BERT-base

    Layout (2 rows × 3 cols):
      [0,0] Heatmap: ODR (log10)  per layer × sublayer
      [0,1] Heatmap: QZR          per layer × sublayer
      [0,2] Scatter:  ODR vs QZR  (color = encoder depth)
      [1,0] ECDF: absmax(δ), Top-1 worst layer (by QZR) + Δ/2 threshold
      [1,1] ECDF: absmax(δ), Top-2 worst layer
      [1,2] ECDF: absmax(δ), Top-3 worst layer
    """
    SUBLAYER_ORDER = ["Q", "K", "V", "O"]
    N_LAYERS       = 12

    # Build 12×4 matrices
    odr_mat = np.full((N_LAYERS, 4), np.nan)
    qzr_mat = np.full((N_LAYERS, 4), np.nan)
    for _, row in df.iterrows():
        li = int(row["layer_idx"])
        si = SUBLAYER_ORDER.index(row["sublayer"])
        odr_mat[li, si] = row["ODR_mean"]
        qzr_mat[li, si] = row["QZR_mean"]

    # Top-3 layers by QZR_mean
    top3 = df.nlargest(3, "QZR_mean")[
        ["layer_name", "layer_idx", "sublayer", "QZR_mean", "ODR_mean"]
    ].reset_index(drop=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Figure 1 — Backward Gradient Outlier Diagnosis: BERT-base Q/K/V/O Analog Tiles\n"
        f"(AbsMax noise management, nm_thres=0, DAC={DAC_BITS}-bit, ADC={ADC_BITS}-bit, "
        f"N={N_STEP} steps, batch={DIAG_BATCH_SIZE})",
        fontsize=11, y=1.01,
    )

    # [0,0] Heatmap ODR (log10)
    ax      = axes[0, 0]
    log_odr = np.log10(np.clip(odr_mat, 1e-3, None))
    im      = ax.imshow(log_odr, aspect="auto", cmap="hot_r", origin="upper")
    ax.set_xticks(range(4));    ax.set_xticklabels(SUBLAYER_ORDER)
    ax.set_yticks(range(N_LAYERS)); ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
    ax.set_xlabel("Attention Sublayer"); ax.set_ylabel("Encoder Layer")
    ax.set_title("(a) ODR$_\\ell$ — Outlier Dominance Ratio (log$_{10}$)")
    plt.colorbar(im, ax=ax, label="log$_{10}$(ODR)")

    # [0,1] Heatmap QZR
    ax = axes[0, 1]
    im = ax.imshow(qzr_mat, aspect="auto", cmap="plasma",
                   origin="upper", vmin=0, vmax=1)
    ax.set_xticks(range(4));    ax.set_xticklabels(SUBLAYER_ORDER)
    ax.set_yticks(range(N_LAYERS)); ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
    ax.set_xlabel("Attention Sublayer"); ax.set_ylabel("Encoder Layer")
    ax.set_title(f"(b) QZR$_\\ell$ — Quant. Zero Rate (DAC {DAC_BITS}-bit, AbsMax)")
    plt.colorbar(im, ax=ax, label="QZR (fraction rounded to 0)")

    # [0,2] Scatter: ODR vs QZR (color = layer depth)
    ax = axes[0, 2]
    sc = ax.scatter(
        df["ODR_mean"], df["QZR_mean"],
        c=df["layer_idx"], cmap="viridis",
        alpha=0.8, s=60, edgecolors="k", linewidths=0.4,
    )
    ax.set_xlabel("ODR$_\\ell$ (Outlier Dominance Ratio)")
    ax.set_ylabel("QZR$_\\ell$ (Quantization Zero Rate)")
    ax.set_title("(c) ODR vs QZR — Outlier→Collapse Correlation")
    ax.set_xscale("log")
    for _, row in top3.iterrows():
        ax.annotate(
            f"L{int(row['layer_idx'])}{row['sublayer']}",
            (row["ODR_mean"], row["QZR_mean"]),
            fontsize=7,
        )
    plt.colorbar(sc, ax=ax, label="Encoder depth")

    # DAC zero-threshold: Δ/2 = inp_bound / (2^DAC_BITS - 1)
    dac_thresh = INP_BOUND / (2**DAC_BITS - 1)

    # [1,0..2] ECDF for top-3 worst layers
    for k, (panel_ax, (_, row)) in enumerate(zip(axes[1, :3], top3.iterrows())):
        name  = row["layer_name"]
        label = f"L{int(row['layer_idx'])}{row['sublayer']} (QZR={row['QZR_mean']:.3f})"
        vals  = stats_dict[name].absmax_array()
        vals_sorted = np.sort(vals)
        ecdf_y      = np.arange(1, len(vals_sorted) + 1) / len(vals_sorted)

        panel_ax.plot(vals_sorted, ecdf_y, lw=1.5, label=label)
        panel_ax.axvline(
            dac_thresh, color="red", ls="--", lw=1.2,
            label=f"Δ/2 = {dac_thresh:.4f}\n(DAC zero threshold)",
        )
        panel_ax.set_xlabel(r"$\|\delta_{vec}\|_\infty$ (per-token absmax of $\delta$)")
        panel_ax.set_ylabel("ECDF")
        panel_ax.set_title(f"(d{k+1}) {label}")
        panel_ax.legend(fontsize=8)
        panel_ax.set_xscale("log")
        panel_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {FIG_PATH}")


# =============================================================================
# Step 10: Main
# =============================================================================

def main():
    torch.manual_seed(SEED)
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/5] Loading tokenizer + data ...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader    = load_diag_data(tokenizer)

    print("[2/5] Creating analog model ...")
    model = create_diag_model()

    print("[3/5] Registering backward hooks ...")
    stats_dict, handles = register_hooks(model)

    print(f"[4/5] Running diagnostic ({N_STEP} steps, no weight update) ...")
    run_diagnostic(model, loader, stats_dict)

    # Remove hooks
    for h in handles:
        h.remove()

    # Metrics CSV
    rows = [s.summary() for s in stats_dict.values()]
    df   = (pd.DataFrame(rows)
              .sort_values(["layer_idx", "sublayer"])
              .reset_index(drop=True))
    df.to_csv(CSV_PATH, index=False)
    print(f"Metrics saved → {CSV_PATH}")
    print(df[["layer_idx", "sublayer", "ODR_mean", "QZR_mean",
              "p_clip_in", "cosine_sim"]].to_string())

    print("[5/5] Creating figure ...")
    create_figure(df, stats_dict)

    print("\n=== Summary ===")
    worst = df.nlargest(5, "QZR_mean")[
        ["layer_idx", "sublayer", "ODR_mean", "QZR_mean", "cosine_sim"]
    ]
    print("Top-5 layers by QZR_mean:")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
