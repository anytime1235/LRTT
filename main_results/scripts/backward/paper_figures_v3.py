"""paper_figures.py — 논문용 Figure A/B/C 생성 (v4: per-layer precision + nm_thres/p_clip 통합)

Three publication-quality figures based on diag_kv_rootcause.py patterns,
extended with FFN1/FFN2 layers and parameterized adc_bits.

- Figure A: QKVO+FFN Root Cause (2×3, 6 sublayers)
- Figure B: IO Resolution Sweep (bits=[4,6,8,10,12], 2×2)
- Figure C: Solutions Comparison (8 variants, 4 categories), 4×2 + CDF
    baseline           — uniform 8-bit, ABS_MAX
    Cat1 lp_q20/q10/q05 — per-layer bits (QZR target sweep, cost-neutral)
    Cat2 nm_thres_p50  — per-layer nm_thres (clip=50%, absmax p50, aggressive)
    Cat2 nm_thres_p80  — per-layer nm_thres (clip=20%, absmax p80)
    Cat2 nm_thres_p90  — per-layer nm_thres (clip=10%, absmax p90)
    Cat2 nm_thres_p95  — per-layer nm_thres (clip=5%,  absmax p95, conservative)
    Cat3 nmthres_mixed — nm_thres_p95 + layer_prec combined
    Cat4 avg_absmax    — AVERAGE_ABS_MAX backward noise management
    Cat4 constant_nm   — CONSTANT backward noise mgmt (calibrated theta)
    Cat5 all_combined  — nm_thres(p95) + AVERAGE_ABS_MAX + mixed precision

Key differences from v3:
  - v4: DAC_BITS = 8 baseline (8-bit 기준 문제 분석)
  - v4: sto_round variant 제거
  - v4: nm_thres + p_clip 통합 — per-layer absmax percentile 기반 nm_thres
  - v4: per-layer precision allocation (severity 기반 bit 분배)
  - v4: calibrate_layer_severity(), allocate_precision(), set_per_layer_config()
  - v3: step_size = 2 * inp_bound * res_ratio (AIHWKit UniformQuantize consistent)
  - v3: l2_retention, rel_l2_error, clip_rate_scaled metrics
  - v3: ratio reservoir sampling (200k cap) to prevent OOM
  - v3: CSV split into summary/steps/CDF

Usage:
  python paper_figures.py                                         # full run
  python paper_figures.py --n-step 5 --n-step-sweep 3 --batch-size 2 --run-tag smoke
"""

import argparse
import gc
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

parser = argparse.ArgumentParser(description="Paper Figures A/B/C — AIMC BERT-base")
parser.add_argument("--n-step",       type=int, default=200,
                    help="steps for Figure A/C runs (default: 200)")
parser.add_argument("--batch-size",   type=int, default=8)
parser.add_argument("--n-step-sweep", type=int, default=100,
                    help="steps per bits configuration for Figure B (default: 100)")
parser.add_argument("--out-dir",      type=str, default="./results/tikitakav1")
parser.add_argument("--figures",      type=str, default="ABC",
                    help="Which figures to generate: A, B, C or any combo (default: ABC)")
parser.add_argument("--run-tag",      type=str, default="v4")
args = parser.parse_args()

# =============================================================================
# Constants & Paths
# =============================================================================

N_STEP       = args.n_step
N_STEP_SWEEP = args.n_step_sweep
BATCH_SIZE   = args.batch_size
RUN_A        = "A" in args.figures.upper()
RUN_B        = "B" in args.figures.upper()
RUN_C        = "C" in args.figures.upper()
RUN_TAG      = args.run_tag

# OUT_DIR: prefer /data/results/tikitakav1 if it exists, else use args
_default_out = "/data/results/tikitakav1"
OUT_DIR = _default_out if os.path.isdir(os.path.dirname(_default_out)) else args.out_dir

MAX_SEQ_LENGTH = 384
DOC_STRIDE     = 128
SEED           = 42
INP_BOUND      = 1.0
DAC_BITS       = 8
ADC_BITS       = 8
N_LAYERS       = 12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output files
CSV_A_SUMMARY = os.path.join(OUT_DIR, "metrics_paper_A_rootcause_summary.csv")
CSV_A_STEPS   = os.path.join(OUT_DIR, "metrics_paper_A_rootcause_steps.csv")
CSV_A_CDF     = os.path.join(OUT_DIR, "metrics_paper_A_rootcause_cdf.csv")
CSV_B_SUMMARY = os.path.join(OUT_DIR, "metrics_paper_B_bitsweep_summary.csv")
CSV_B_STEPS   = os.path.join(OUT_DIR, "metrics_paper_B_bitsweep_steps.csv")
CSV_C_SUMMARY = os.path.join(OUT_DIR, "metrics_paper_C_solutions_summary.csv")
CSV_C_STEPS   = os.path.join(OUT_DIR, "metrics_paper_C_solutions_steps.csv")
FIG_A = os.path.join(OUT_DIR, "fig_paper_A_rootcause_qkvo_ffn.png")
FIG_B = os.path.join(OUT_DIR, "fig_paper_B_bitsweep.png")
FIG_C = os.path.join(OUT_DIR, "fig_paper_C_solutions.png")

print(f"[Config] Device={DEVICE}, N_STEP={N_STEP}, N_STEP_SWEEP={N_STEP_SWEEP}, "
      f"BATCH={BATCH_SIZE}, RUN_TAG={RUN_TAG}")
print(f"[Config] OUT_DIR={OUT_DIR}")
print(f"[Config] Figures={args.figures.upper()} (A={RUN_A}, B={RUN_B}, C={RUN_C})")

# =============================================================================
# Layer Name Utilities
# =============================================================================

# attention.output.dense must appear BEFORE output.dense to prevent collision
_LAYER_RE = re.compile(
    r"encoder\.layer\.(\d+)\."
    r"(attention\.self\.query|attention\.self\.key|attention\.self\.value"
    r"|attention\.output\.dense|intermediate\.dense|output\.dense)"
)
_SUBLAYER_MAP = {
    "attention.self.query":   "Q",
    "attention.self.key":     "K",
    "attention.self.value":   "V",
    "attention.output.dense": "O",
    "intermediate.dense":     "FFN1",
    "output.dense":           "FFN2",
}
SUBLAYER_ORDER = ["Q", "K", "V", "O", "FFN1", "FFN2"]


def parse_layer_name(name: str):
    """Returns (layer_idx: int, sublayer: str) or None."""
    m = _LAYER_RE.search(name)
    if m is None:
        return None
    return int(m.group(1)), _SUBLAYER_MAP[m.group(2)]


def _layer_names(model):
    """Split encoder linear layers into target (QKVO) and nontarget (FFN)."""
    always_digital = ["qa_outputs", "pooler"]
    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    target = [
        n for n in all_linear
        if "encoder" in n and "attention" in n
        and not any(d in n for d in always_digital)
    ]
    nontarget = [
        n for n in all_linear
        if "encoder" in n and n not in target
        and not any(d in n for d in always_digital)
    ]
    return target, nontarget, all_linear


# =============================================================================
# RPU Config (adc_bits parameterized)
# =============================================================================

def create_rpu_config(nm_thres=0.0, dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                      bwd_noise_mgmt="ABS_MAX", bwd_nm_decay=0.001):
    """SingleRPU with noise-free SoftBoundsDevice. inp_res = 1/(2^dac_bits - 2).

    bwd_noise_mgmt: backward noise management type.
        "ABS_MAX" | "AVERAGE_ABS_MAX" | "CONSTANT" | "NONE"
    bwd_nm_decay: decay for AVERAGE_ABS_MAX (default 0.001).
    """
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import SoftBoundsDevice
    from aihwkit.simulator.configs.utils import NoiseManagementType

    _NM_MAP = {
        "ABS_MAX":         NoiseManagementType.ABS_MAX,
        "AVERAGE_ABS_MAX": NoiseManagementType.AVERAGE_ABS_MAX,
        "CONSTANT":        NoiseManagementType.CONSTANT,
        "NONE":            NoiseManagementType.NONE,
        "MAX":             NoiseManagementType.MAX,
    }

    device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=False,
    )
    rpu = SingleRPUConfig(device=device)
    for io in [rpu.forward, rpu.backward]:
        io.inp_bound        = INP_BOUND
        io.inp_res          = 1.0 / (2**dac_bits - 2)
        io.out_bound        = 12.0
        io.out_res          = 1.0 / (2**adc_bits - 2)
        io.noise_management = NoiseManagementType.ABS_MAX
        io.out_noise        = 0.0
        io.inp_sto_round    = False
    # backward-only overrides
    bwd_nm = _NM_MAP.get(bwd_noise_mgmt, NoiseManagementType.ABS_MAX)
    rpu.backward.noise_management       = bwd_nm
    rpu.backward.nm_thres               = nm_thres
    if bwd_nm == NoiseManagementType.AVERAGE_ABS_MAX:
        rpu.backward.nm_decay           = bwd_nm_decay
    rpu.mapping.digital_bias            = True
    rpu.mapping.weight_scaling_omega    = 1.0
    rpu.mapping.weight_scaling_columnwise = True
    return rpu


# =============================================================================
# Model Creation
# =============================================================================

def create_model(nm_thres=0.0, dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                 bwd_noise_mgmt="ABS_MAX"):
    """BERT-base 2-pass analog conversion.

    Pass 1: Q/K/V/O (target) — analog, weight updates enabled (lr=0 so noop)
    Pass 2: FFN (nontarget)  — analog, tile.update = _noop (frozen)

    All AnalogContext.requires_grad = True → backward flows through FFN tiles
    → register_full_backward_hook fires on FFN layers too.
    """
    from aihwkit.nn import AnalogLinear
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.optim.context import AnalogContext

    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    target, nontarget, all_linear = _layer_names(model)

    # Pass 1: target (QKVO)
    rpu = create_rpu_config(nm_thres=nm_thres,
                            dac_bits=dac_bits, adc_bits=adc_bits,
                            bwd_noise_mgmt=bwd_noise_mgmt)
    model = convert_to_analog(
        model, rpu,
        exclude_modules=[n for n in all_linear if n not in target]
    )

    # Pass 2: nontarget (FFN) — same config, update will be nooped
    nt_rpu = create_rpu_config(nm_thres=nm_thres,
                               dac_bits=dac_bits, adc_bits=adc_bits,
                               bwd_noise_mgmt=bwd_noise_mgmt)
    model = convert_to_analog(
        model, nt_rpu,
        exclude_modules=[n for n in all_linear if n not in nontarget]
    )

    # Freeze nontarget tile updates
    def _noop(x, d, *a, **kw):
        return None

    for name, m in model.named_modules():
        if isinstance(m, AnalogLinear) and name not in target:
            for tile in m.analog_tiles():
                tile.update = _noop

    # Gradient flow: disable all, then re-enable AnalogContext + qa_outputs
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
    print(f"  Analog tiles — target(QKVO):{n_t}, frozen(FFN):{n_all - n_t}, "
          f"nm_thres={nm_thres:.4g}, "
          f"dac={dac_bits}b, adc={adc_bits}b")
    return model.to(DEVICE)


# =============================================================================
# Data Loading
# =============================================================================

def load_data(tokenizer, n_step, batch_size):
    """SQuAD v1.1 — first n_step batches. Seed-fixed for reproducibility."""

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
            while seq[idx] != 1:
                idx += 1
            cs = idx
            while idx < len(seq) and seq[idx] == 1:
                idx += 1
            ce = idx - 1
            if offset[cs][0] > ec or offset[ce][1] < sc:
                sp.append(0); ep.append(0)
            else:
                idx = cs
                while idx <= ce and offset[idx][0] <= sc:
                    idx += 1
                sp.append(idx - 1)
                idx = ce
                while idx >= cs and offset[idx][1] >= ec:
                    idx -= 1
                ep.append(idx + 1)
        inputs["start_positions"] = sp
        inputs["end_positions"]   = ep
        return inputs

    raw = load_dataset("squad")
    tok = raw["train"].map(
        preprocess_train, batched=True,
        remove_columns=raw["train"].column_names
    )
    n = min(n_step * batch_size, len(tok))
    subset = tok.shuffle(seed=SEED).select(range(n))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        collate_fn=default_data_collator)
    print(f"  Dataset: {n} samples → {len(loader)} batches")
    return loader


# =============================================================================
# MaskBuffer
# =============================================================================

class MaskBuffer:
    """Captures attention_mask from each forward pass for padding exclusion."""

    def __init__(self):
        self.val = None

    def register(self, model):
        def _hook(mod, args, kwargs):
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                self.val = kwargs["attention_mask"].bool().cpu()
        model.register_forward_pre_hook(_hook, with_kwargs=True)


# =============================================================================
# LayerStats — AIHWKit-consistent quantization + extended metrics
# =============================================================================

class LayerStats:
    """Per-layer backward gradient statistics accumulator.

    zero_thresh = inp_res = 1/(2^b - 2)
    step_size = 2 * INP_BOUND * res_ratio  (AIHWKit UniformQuantize consistent)
    nm_thres > 0 → alpha = min(absmax, nm_thres) simulating tile nm_thres cap.
    """

    def __init__(self, name: str, layer_idx: int, sublayer: str,
                 inp_res: float, nm_thres: float = 0.0,
                 store_sweep: bool = True):
        self.name        = name
        self.layer_idx   = layer_idx
        self.sublayer    = sublayer
        self.inp_res     = inp_res
        self.nm_thres    = nm_thres
        self.zero_thresh = inp_res   # threshold: ratio < zero_thresh → quant 0
        self.eps         = 1e-8

        # AIHWKit-consistent quantization parameters
        self.res_ratio  = inp_res if inp_res <= 1.0 else 1.0 / inp_res
        self.step_size  = 2.0 * INP_BOUND * self.res_ratio

        # per-step scalar lists (length grows with steps)
        self.ezr_steps       = []
        self.qzr_all_steps   = []
        self.qzr_nz_steps    = []
        self.odr_steps       = []
        self.cosine_steps    = []
        self.ratio_q50_steps = []
        self.ratio_q90_steps = []
        self.ratio_q99_steps = []

        # new metrics (from diag_backward_outlier.py)
        self.p_clip_steps      = []   # P(|δ| > INP_BOUND) element-wise
        self.absmax_q50_steps  = []   # per-vector absmax의 50th percentile
        self.absmax_q90_steps  = []   # 90th
        self.absmax_q99_steps  = []   # 99th
        self.absmax_q999_steps = []   # 99.9th

        # v3 new metrics
        self.l2_retention_steps      = []
        self.rel_l2_error_steps      = []
        self.clip_rate_scaled_steps  = []
        self.n_vec_steps             = []

        # raw per-vector absmax buffer for ECDF (always collected)
        self._absmax_buf = []

        # ratio reservoir sampling (replaces _sweep_steps)
        self._store_sweep = store_sweep
        self._ratio_reservoir_max = 200_000
        self._ratio_reservoir = []
        self._ratio_reservoir_count = 0

        # padding-included versions
        self.ezr_pad_steps     = []
        self.qzr_all_pad_steps = []

    def update(self, dy: torch.Tensor, mask=None):
        """Called from backward hook. dy = grad_output[0]."""
        with torch.no_grad():
            dy = dy.detach().float()
            # padding-included
            dy_flat_pad = dy.reshape(-1, dy.shape[-1])
            self._compute_pad(dy_flat_pad)
            # mask-excluded
            if mask is not None and dy.dim() == 3:
                B, S, D = dy.shape
                if mask.shape == (B, S):
                    mask_dev = mask.to(dy.device)
                    dy_real  = dy[mask_dev]   # (N_real, D)
                else:
                    dy_real = dy_flat_pad
            else:
                dy_real = dy_flat_pad
            self._compute_main(dy_real)

    def _compute_pad(self, dy_flat):
        abs_dy  = dy_flat.abs()
        self.ezr_pad_steps.append((abs_dy == 0).float().mean().item())
        absmax_v = abs_dy.max(dim=1).values.clamp(min=self.eps).unsqueeze(1)
        scaled   = dy_flat / absmax_v
        self.qzr_all_pad_steps.append(
            (scaled.abs() < self.zero_thresh).float().mean().item()
        )

    def _compute_main(self, dy_real: torch.Tensor):
        """Core computation on real-token (mask-excluded) vectors."""
        abs_dy = dy_real.abs()
        N, D   = abs_dy.shape
        if N == 0:
            for lst in [self.ezr_steps, self.qzr_all_steps, self.qzr_nz_steps]:
                lst.append(0.0)
            self.odr_steps.append(1.0)
            self.cosine_steps.append(1.0)
            for lst in [self.ratio_q50_steps, self.ratio_q90_steps, self.ratio_q99_steps]:
                lst.append(0.0)
            self.p_clip_steps.append(0.0)
            for lst in [self.absmax_q50_steps, self.absmax_q90_steps,
                        self.absmax_q99_steps, self.absmax_q999_steps]:
                lst.append(0.0)
            self.clip_rate_scaled_steps.append(0.0)
            self.l2_retention_steps.append(1.0)
            self.rel_l2_error_steps.append(0.0)
            self.n_vec_steps.append(0)
            return

        # EZR
        self.ezr_steps.append((abs_dy == 0).float().mean().item())

        # per-vector absmax (uncapped, for ECDF + reservoir)
        absmax_v = abs_dy.max(dim=1).values   # (N,)
        self._absmax_buf.append(absmax_v.cpu().float().numpy())

        # alpha: cap by nm_thres if >0
        if self.nm_thres > 0:
            alpha = absmax_v.clamp(max=self.nm_thres).clamp(min=self.eps).unsqueeze(1)
        else:
            alpha = absmax_v.clamp(min=self.eps).unsqueeze(1)

        # ratio = |dy| / alpha
        ratio = abs_dy / alpha   # (N, D)

        # QZR_all and QZR_nonzero
        zero_mask = (abs_dy == 0)
        nz_mask   = ~zero_mask
        self.qzr_all_steps.append((ratio < self.zero_thresh).float().mean().item())
        if nz_mask.any():
            self.qzr_nz_steps.append(
                (ratio[nz_mask] < self.zero_thresh).float().mean().item()
            )
        else:
            self.qzr_nz_steps.append(0.0)

        # ODR: absmax / median per vector
        absmed_v = abs_dy.median(dim=1).values.clamp(min=self.eps)
        self.odr_steps.append((absmax_v / absmed_v).mean().item())

        # p_clip: P(|δ| > INP_BOUND) element-wise
        self.p_clip_steps.append((abs_dy > INP_BOUND).float().mean().item())

        # absmax quantiles across N vectors
        absmax_sorted = absmax_v.sort().values
        n = len(absmax_sorted)
        self.absmax_q50_steps.append(absmax_sorted[max(0, int(0.50 * n) - 1)].item())
        self.absmax_q90_steps.append(absmax_sorted[max(0, int(0.90 * n) - 1)].item())
        self.absmax_q99_steps.append(absmax_sorted[max(0, int(0.99 * n) - 1)].item())
        self.absmax_q999_steps.append(absmax_sorted[min(int(0.999 * n), n - 1)].item())

        # Cosine sim: FP32 dy vs DAC-quantized dy (AIHWKit-consistent step)
        scaled = dy_real / alpha * INP_BOUND                        # [-INP_BOUND, +INP_BOUND]
        scaled_q = (scaled / self.step_size).round() * self.step_size
        self.clip_rate_scaled_steps.append(
            (scaled.abs() > INP_BOUND).float().mean().item()
        )
        scaled_q = scaled_q.clamp(-INP_BOUND, INP_BOUND)
        dy_q = scaled_q * alpha / INP_BOUND
        self.cosine_steps.append(
            F.cosine_similarity(dy_real, dy_q, dim=1).mean().item()
        )

        # l2_retention and rel_l2_error
        dy_norm   = dy_real.norm(dim=1).clamp(min=self.eps)
        dy_q_norm = dy_q.norm(dim=1)
        self.l2_retention_steps.append((dy_q_norm / dy_norm).mean().item())
        err_norm = (dy_real - dy_q).norm(dim=1)
        self.rel_l2_error_steps.append((err_norm / dy_norm).mean().item())

        # n_vec
        self.n_vec_steps.append(N)

        # ratio quantiles (always using uncapped absmax)
        ratio_orig = abs_dy / absmax_v.clamp(min=self.eps).unsqueeze(1)
        ratio_np   = ratio_orig.reshape(-1).cpu().float().numpy()
        self.ratio_q50_steps.append(float(np.quantile(ratio_np, 0.50)))
        self.ratio_q90_steps.append(float(np.quantile(ratio_np, 0.90)))
        self.ratio_q99_steps.append(float(np.quantile(ratio_np, 0.99)))

        # ratio reservoir sampling (replaces _sweep_steps)
        if self._store_sweep:
            ratio_flat = ratio_orig.reshape(-1).cpu().float().numpy()
            n_elem = len(ratio_flat)
            sample_size = min(4096, n_elem)
            if n_elem <= sample_size:
                sampled = ratio_flat
            else:
                rng = np.random.default_rng(seed=self._ratio_reservoir_count)
                sampled = ratio_flat[rng.choice(n_elem, sample_size, replace=False)]
            self._ratio_reservoir.append(sampled)
            self._ratio_reservoir_count += n_elem
            # compact if over max
            total = sum(len(a) for a in self._ratio_reservoir)
            if total > self._ratio_reservoir_max:
                all_s = np.concatenate(self._ratio_reservoir)
                keep = np.random.default_rng(42).choice(
                    len(all_s), self._ratio_reservoir_max, replace=False
                )
                self._ratio_reservoir = [all_s[keep]]

    def absmax_array(self) -> np.ndarray:
        """Full absmax distribution for ECDF plots."""
        return np.concatenate(self._absmax_buf) if self._absmax_buf else np.array([])

    def ratio_reservoir_array(self) -> np.ndarray:
        """Sampled ratio distribution for CDF plots."""
        return np.concatenate(self._ratio_reservoir) if self._ratio_reservoir else np.array([])

    def summary(self, label: str = "baseline", dac_bits: int = DAC_BITS,
                adc_bits: int = ADC_BITS, figure_id: str = "",
                run_tag: str = "") -> dict:
        def _m(lst): return float(np.mean(lst)) if lst else float("nan")
        return {
            "figure_id":        figure_id,
            "run_tag":          run_tag,
            "layer_name":       self.name,
            "layer_idx":        self.layer_idx,
            "sublayer":         self.sublayer,
            "variant":          label,
            "dac_bits":         dac_bits,
            "adc_bits":         adc_bits,
            "inp_bound":        INP_BOUND,
            "inp_res":          self.inp_res,
            "res_ratio":        self.res_ratio,
            "step_size":        self.step_size,
            "nm_thres":         self.nm_thres,
            "EZR":              _m(self.ezr_steps),
            "QZR_all":          _m(self.qzr_all_steps),
            "QZR_nonzero":      _m(self.qzr_nz_steps),
            "ODR":              _m(self.odr_steps),
            "cosine_sim":       _m(self.cosine_steps),
            "l2_retention":     _m(self.l2_retention_steps),
            "rel_l2_error":     _m(self.rel_l2_error_steps),
            "clip_rate_scaled": _m(self.clip_rate_scaled_steps),
            "ratio_q50":        _m(self.ratio_q50_steps),
            "ratio_q90":        _m(self.ratio_q90_steps),
            "ratio_q99":        _m(self.ratio_q99_steps),
            "p_clip":           _m(self.p_clip_steps),
            "absmax_q50":       _m(self.absmax_q50_steps),
            "absmax_q90":       _m(self.absmax_q90_steps),
            "absmax_q99":       _m(self.absmax_q99_steps),
            "absmax_q999":      _m(self.absmax_q999_steps),
            "EZR_pad":          _m(self.ezr_pad_steps),
            "QZR_all_pad":      _m(self.qzr_all_pad_steps),
        }

    def step_records(self, label: str = "baseline", dac_bits: int = DAC_BITS,
                     adc_bits: int = ADC_BITS, figure_id: str = "",
                     run_tag: str = "") -> list:
        """Per-step records for steps CSV."""
        n = len(self.cosine_steps)
        records = []
        for i in range(n):
            def _g(lst, idx, default=float("nan")):
                return lst[idx] if idx < len(lst) else default
            rec = {
                "figure_id":        figure_id,
                "run_tag":          run_tag,
                "layer_name":       self.name,
                "layer_idx":        self.layer_idx,
                "sublayer":         self.sublayer,
                "variant":          label,
                "dac_bits":         dac_bits,
                "adc_bits":         adc_bits,
                "inp_bound":        INP_BOUND,
                "inp_res":          self.inp_res,
                "res_ratio":        self.res_ratio,
                "step_size":        self.step_size,
                "nm_thres":         self.nm_thres,
                "step_idx":         i,
                "n_vec":            _g(self.n_vec_steps, i, 0),
                "EZR":              _g(self.ezr_steps, i),
                "QZR_all":          _g(self.qzr_all_steps, i),
                "QZR_nonzero":      _g(self.qzr_nz_steps, i),
                "ODR":              _g(self.odr_steps, i),
                "cosine_sim":       _g(self.cosine_steps, i),
                "l2_retention":     _g(self.l2_retention_steps, i),
                "rel_l2_error":     _g(self.rel_l2_error_steps, i),
                "clip_rate_scaled": _g(self.clip_rate_scaled_steps, i),
                "ratio_q50":        _g(self.ratio_q50_steps, i),
                "ratio_q90":        _g(self.ratio_q90_steps, i),
                "ratio_q99":        _g(self.ratio_q99_steps, i),
                "p_clip":           _g(self.p_clip_steps, i),
                "absmax_q50":       _g(self.absmax_q50_steps, i),
                "absmax_q90":       _g(self.absmax_q90_steps, i),
                "absmax_q99":       _g(self.absmax_q99_steps, i),
                "absmax_q999":      _g(self.absmax_q999_steps, i),
                "EZR_pad":          _g(self.ezr_pad_steps, i),
                "QZR_all_pad":      _g(self.qzr_all_pad_steps, i),
            }
            records.append(rec)
        return records


# =============================================================================
# Hook Registration
# =============================================================================

def register_hooks(model, mask_buf: MaskBuffer, inp_res: float,
                   nm_thres: float = 0.0,
                   inp_res_map: dict = None,
                   nm_thres_map: dict = None,
                   store_sweep: bool = True) -> tuple:
    """Register full backward hooks on all AnalogLinear matching the regex.

    Per-layer overrides via inp_res_map / nm_thres_map:
        {(layer_idx, sublayer): value}
    These override the global inp_res / nm_thres for matching layers.
    """
    from aihwkit.nn import AnalogLinear

    stats_dict, handles = {}, []
    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear):
            continue
        parsed = parse_layer_name(name)
        if parsed is None:
            continue
        layer_idx, sublayer = parsed
        key = (layer_idx, sublayer)

        # per-layer overrides
        lr_inp_res  = inp_res_map[key]  if (inp_res_map  and key in inp_res_map)  else inp_res
        lr_nm_thres = nm_thres_map[key] if (nm_thres_map and key in nm_thres_map) else nm_thres

        stats = LayerStats(name=name, layer_idx=layer_idx, sublayer=sublayer,
                           inp_res=lr_inp_res, nm_thres=lr_nm_thres,
                           store_sweep=store_sweep)
        stats_dict[name] = stats

        def make_hook(s, mb):
            def fn(mod, gin, gout):
                if gout[0] is not None:
                    s.update(gout[0], mask=mb.val)
            return fn

        handles.append(module.register_full_backward_hook(make_hook(stats, mask_buf)))

    sublayers_found = sorted(set(s.sublayer for s in stats_dict.values()))
    n_custom = sum(1 for s in stats_dict.values()
                   if (inp_res_map and (s.layer_idx, s.sublayer) in inp_res_map)
                   or (nm_thres_map and (s.layer_idx, s.sublayer) in nm_thres_map))
    print(f"[Hook] {len(stats_dict)} hooks, sublayers={sublayers_found}, "
          f"inp_res={inp_res:.6f}, nm_thres={nm_thres:.4g}, "
          f"per-layer overrides={n_custom}")
    return stats_dict, handles


# =============================================================================
# Per-Layer Calibration & Precision Allocation
# =============================================================================

def calibrate_layer_severity(stats_dict: dict) -> dict:
    """Compute per-layer severity score from baseline run.

    Severity = QZR_nonzero (직접 측정: 현재 bit에서 비zero gradient 중
    양자화로 0이 되는 비율).

    이 값만이 "bit를 올리면 개선되는 정도"를 직접 나타냄.
    p_clip은 nm_thres가 해결하는 문제이므로 bit 할당에 사용하지 않음.
    cosine_sim은 QZR_nz의 종속 변수이므로 중복.

    Returns: {(layer_idx, sublayer): {
        'qzr_nz', 'p_clip', 'cosine', 'absmax_q99', 'severity'
    }}
    """
    severity = {}
    for name, s in stats_dict.items():
        key = (s.layer_idx, s.sublayer)
        qzr_nz     = float(np.mean(s.qzr_nz_steps))    if s.qzr_nz_steps    else 0.0
        p_clip     = float(np.mean(s.p_clip_steps))     if s.p_clip_steps     else 0.0
        cosine     = float(np.mean(s.cosine_steps))      if s.cosine_steps     else 1.0
        absmax_q99 = float(np.mean(s.absmax_q99_steps)) if s.absmax_q99_steps else 0.0
        # Severity = QZR_nonzero only
        # "비zero gradient 중 양자화로 0이 되는 비율"
        # 이것이 bit를 올려서 직접 개선 가능한 유일한 지표
        score = qzr_nz
        severity[key] = {
            'qzr_nz':     qzr_nz,
            'p_clip':     p_clip,
            'cosine':     cosine,
            'absmax_q99': absmax_q99,
            'severity':   score,
        }
    return severity


def allocate_precision(stats_dict: dict, base_bits: int = 8,
                       min_bits: int = 4, max_bits: int = 12,
                       qzr_target: float = 0.10,
                       candidate_bits: tuple = (4, 6, 8, 10, 12)) -> dict:
    """Cost-neutral per-layer bit allocation from ratio distribution.

    For each layer, compute QZR_nz at every candidate bit level using the
    stored ratio distribution (no additional model run needed):
        QZR_nz(b) = P(ratio < 1/(2^b - 2))  among non-zero ratios

    Then find the minimum bits where QZR_nz < qzr_target.
    Finally, enforce cost-neutrality (total = N_layers × base_bits).

    Returns: {(layer_idx, sublayer): allocated_bits}
    """
    # Phase 1: per-layer minimum bits from ratio distribution
    bits_map = {}
    layer_qzr_table = {}   # for logging

    for name, s in stats_dict.items():
        key = (s.layer_idx, s.sublayer)
        ratio_arr = s.ratio_reservoir_array()

        if len(ratio_arr) == 0:
            bits_map[key] = base_bits
            continue

        # Remove exact zeros from ratio (EZR is structural, not resolution issue)
        nz_ratios = ratio_arr[ratio_arr > 0]
        if len(nz_ratios) == 0:
            bits_map[key] = min_bits
            continue

        # Compute QZR_nz at each candidate bit level
        qzr_at_bits = {}
        for b in candidate_bits:
            thresh = 1.0 / (2**b - 2)
            qzr_at_bits[b] = float(np.mean(nz_ratios < thresh))

        layer_qzr_table[key] = qzr_at_bits

        # Find minimum bits where QZR_nz < target
        assigned = max_bits  # fallback
        for b in candidate_bits:
            if qzr_at_bits[b] < qzr_target:
                assigned = b
                break
        bits_map[key] = assigned

    # Phase 2: enforce cost neutrality (total = N × base_bits)
    n_layers = len(bits_map)
    total_budget = n_layers * base_bits
    current_total = sum(bits_map.values())

    if current_total != total_budget:
        delta = total_budget - current_total
        # Over budget → reduce layers with most headroom (lowest QZR_nz at lower bits)
        # Under budget → boost layers with worst QZR_nz
        severity = {k: v.get(bits_map[k], 0.0)
                    for k, v in layer_qzr_table.items()}
        sorted_keys = sorted(bits_map.keys(),
                             key=lambda k: severity.get(k, 0.0),
                             reverse=(delta < 0))
        for key in sorted_keys:
            if delta == 0:
                break
            step = 2 if abs(delta) >= 2 else abs(delta)
            if delta > 0 and bits_map[key] < max_bits:
                add = min(step, max_bits - bits_map[key], delta)
                bits_map[key] += add
                delta -= add
            elif delta < 0 and bits_map[key] > min_bits:
                sub = min(step, bits_map[key] - min_bits, -delta)
                bits_map[key] -= sub
                delta += sub

    # Print per-layer virtual bit sweep
    print(f"\n  [Bit allocation] target QZR_nz < {qzr_target}, "
          f"budget={total_budget} ({n_layers}×{base_bits})")
    print(f"  {'Layer':<10} " +
          " ".join(f"{b:>6}b" for b in candidate_bits) + "  → alloc")
    for key in sorted(layer_qzr_table.keys()):
        li, sl = key
        qzr = layer_qzr_table[key]
        vals = " ".join(f"{qzr[b]:6.3f}" for b in candidate_bits)
        print(f"  L{li:<2} {sl:<5}  {vals}  → {bits_map[key]:>3}b")
    print(f"  Total: {sum(bits_map.values())} bits "
          f"(avg {sum(bits_map.values())/n_layers:.1f}b)")

    return bits_map


def calibrate_per_layer_thetas(stats_dict: dict,
                               clip_targets: tuple = (0.10, 0.05)) -> dict:
    """p_clip-driven nm_thres calibration.

    For each layer, find the nm_thres (theta) that would result in a target
    fraction of vectors being clipped (= having absmax > theta).

    clip_target = 0.05 means "accept 5% of vectors being clipped"
      → theta = absmax_p95 (95th percentile of per-vector absmax)
      → nm_thres = theta → α = min(absmax, theta)
      → top 5% outlier vectors get capped, bottom 95% preserved

    The mapping:
      clip_target  →  absmax percentile  →  nm_thres
      0.10 (10%)   →  p90                →  aggressive cap
      0.05 (5%)    →  p95                →  moderate cap

    Returns: {clip_target: {(layer_idx, sublayer): theta_value}}
    Also prints per-layer p_clip → theta mapping.
    """
    thetas = {ct: {} for ct in clip_targets}
    print(f"\n  [p_clip → nm_thres calibration]")
    print(f"  clip_targets = {clip_targets}")

    for name, s in stats_dict.items():
        arr = s.absmax_array()   # all per-vector absmax values from baseline
        if len(arr) == 0:
            continue
        key = (s.layer_idx, s.sublayer)
        baseline_pclip = float(np.mean(s.p_clip_steps)) if s.p_clip_steps else 0.0

        for ct in clip_targets:
            # clip_target=0.05 → we want 5% of vectors clipped
            # → theta = absmax at (1 - clip_target) = 95th percentile
            pct = 1.0 - ct
            theta = float(np.quantile(arr, pct))
            thetas[ct][key] = theta

        # Log for key layers
        if s.sublayer in ["K", "V"]:
            theta_strs = ", ".join(
                f"clip={ct:.0%}→θ={thetas[ct][key]:.6f}" for ct in clip_targets)
            print(f"    L{s.layer_idx} {s.sublayer}: "
                  f"baseline p_clip={baseline_pclip:.4f}, {theta_strs}")

    return thetas


def set_per_layer_config(model, bits_map: dict = None, thres_map: dict = None,
                         noise_mgmt_str: str = None, nm_decay: float = 0.001):
    """Set per-layer precision, nm_thres, and/or noise management on tiles.

    bits_map:       {(layer_idx, sublayer): dac_bits}   → tile backward.inp_res
    thres_map:      {(layer_idx, sublayer): nm_thres}   → tile backward.nm_thres
    noise_mgmt_str: if set, override backward noise_management on ALL matched tiles
    nm_decay:       decay for AVERAGE_ABS_MAX
    """
    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.configs.utils import NoiseManagementType

    _NM_MAP = {
        "ABS_MAX":         NoiseManagementType.ABS_MAX,
        "AVERAGE_ABS_MAX": NoiseManagementType.AVERAGE_ABS_MAX,
        "CONSTANT":        NoiseManagementType.CONSTANT,
        "NONE":            NoiseManagementType.NONE,
    }

    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear):
            continue
        parsed = parse_layer_name(name)
        if parsed is None:
            continue
        key = parsed   # (layer_idx, sublayer)
        modified = False
        for tile in module.analog_tiles():
            if bits_map and key in bits_map:
                bits = bits_map[key]
                tile.rpu_config.backward.inp_res = 1.0 / (2**bits - 2)
                modified = True
            if thres_map and key in thres_map:
                tile.rpu_config.backward.nm_thres = thres_map[key]
                modified = True
            if noise_mgmt_str and noise_mgmt_str in _NM_MAP:
                tile.rpu_config.backward.noise_management = _NM_MAP[noise_mgmt_str]
                if noise_mgmt_str == "AVERAGE_ABS_MAX":
                    tile.rpu_config.backward.nm_decay = nm_decay
                modified = True
        if modified:
            count += 1
    desc_parts = []
    if bits_map:   desc_parts.append(f"bits={len(bits_map)}")
    if thres_map:  desc_parts.append(f"thres={len(thres_map)}")
    if noise_mgmt_str: desc_parts.append(f"nm={noise_mgmt_str}")
    print(f"  [set_per_layer_config] Updated {count} modules ({', '.join(desc_parts)})")


def print_severity_report(severity: dict, bits_map: dict = None):
    """Print per-layer severity and allocated bits."""
    print("\n  ┌──────────────────────────────────────────────────────"
          "────────────────────────────┐")
    print("  │ Layer  Sub   QZR_nz   p_clip  cosine  absmax_q99"
          "  severity  bits(alloc) │")
    print("  ├──────────────────────────────────────────────────────"
          "────────────────────────────┤")
    for key in sorted(severity.keys()):
        v = severity[key]
        li, sl = key
        bits_str = f"{bits_map[key]:>4}b" if (bits_map and key in bits_map) else "   —"
        print(f"  │ L{li:<3}  {sl:<5} {v['qzr_nz']:7.4f}  {v['p_clip']:6.4f}"
              f"  {v['cosine']:6.4f}  {v['absmax_q99']:10.6f}"
              f"  {v['severity']:8.4f}  {bits_str:>10} │")
    print("  └──────────────────────────────────────────────────────"
          "────────────────────────────┘")
    if bits_map:
        from collections import Counter
        dist = Counter(bits_map.values())
        avg = np.mean(list(bits_map.values()))
        print(f"  Bit distribution: {dict(sorted(dist.items()))}, avg={avg:.1f}b")


# =============================================================================
# P99ClipHook — per-vector 99th percentile gradient clip
# =============================================================================

def _p99_clip_grad(g: torch.Tensor, mask) -> torch.Tensor:
    """Clip gradient per-vector to p99(abs(dy)).

    For 3D (B, S, D) with mask (B, S): only clips real tokens, zeros padding.
    For 2D or no mask: clips all rows.
    """
    with torch.no_grad():
        gf = g.detach().float()
        if mask is not None and gf.dim() == 3:
            B, S, D = gf.shape
            m = mask.to(gf.device)
            if m.shape == (B, S) and D > 0:
                g_real = gf[m]   # (N_real, D)
                if g_real.shape[0] > 0:
                    k    = max(1, int(D * 0.01))
                    p99  = torch.topk(g_real.abs(), k, dim=1)[0][:, -1:]  # (N_real, 1)
                    clipped = torch.where(g_real.abs() > p99, p99 * g_real.sign(), g_real)
                    gout = gf.clone()
                    gout[m]  = clipped
                    gout[~m] = 0.0
                    return gout.to(g.dtype)
        # fallback: no mask or 2D
        gf2 = gf.reshape(-1, gf.shape[-1])
        if gf2.shape[0] > 0 and gf2.shape[1] > 0:
            k    = max(1, int(gf2.shape[1] * 0.01))
            p99  = torch.topk(gf2.abs(), k, dim=1)[0][:, -1:]
            gout = torch.where(gf2.abs() > p99, p99 * gf2.sign(), gf2)
            return gout.reshape(gf.shape).to(g.dtype)
    return g


class P99ClipHook:
    """Forward hook that clips incoming backward gradient to per-vector p99.

    Registers output tensor hooks during forward pass.
    Backward order:
      1. Downstream gradient arrives at output tensor
      2. output.register_hook(clip_grad) fires → returns clipped gradient
      3. Module backward uses clipped gradient (tile quantization sees clipped δ)
      4. register_full_backward_hook fires with grad_output[0] = clipped gradient

    LayerStats therefore sees the already-clipped gradient.
    """

    def __init__(self, mask_buf: MaskBuffer):
        self.mask_buf = mask_buf
        self._handles = []

    def register(self, model):
        from aihwkit.nn import AnalogLinear

        for name, module in model.named_modules():
            if not isinstance(module, AnalogLinear):
                continue
            if parse_layer_name(name) is None:
                continue

            def make_fwd_hook(mb):
                def fwd_hook(mod, inp, out):
                    if isinstance(out, torch.Tensor) and out.requires_grad:
                        def clip_grad(g):
                            return _p99_clip_grad(g, mb.val)
                        out.register_hook(clip_grad)
                return fwd_hook

            h = module.register_forward_hook(make_fwd_hook(self.mask_buf))
            self._handles.append(h)

        print(f"[P99ClipHook] Registered on {len(self._handles)} AnalogLinear modules")

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# =============================================================================
# NPZ Save — raw absmax for ECDF
# =============================================================================

def save_absmax_npz(stats_dict: dict, filepath: str):
    """Save raw per-vector absmax arrays for all layers to .npz.
    Keys: 'L{idx}_{sublayer}' e.g. 'L0_K', 'L5_FFN1'.
    """
    data = {}
    for name, s in stats_dict.items():
        key = f"L{s.layer_idx}_{s.sublayer}"
        arr = s.absmax_array()
        if len(arr) > 0:
            data[key] = arr
    np.savez_compressed(filepath, **data)
    n_keys = len(data)
    total  = sum(len(v) for v in data.values())
    print(f"  absmax npz → {filepath}  ({n_keys} layers, {total:,} values)")


# =============================================================================
# CDF CSV — worst-N layer ratio CDF
# =============================================================================

def save_cdf_csv(stats_dict, df_summary, filepath, n_points=2000, n_worst=3):
    """Save CDF of |dy|/absmax ratio for worst-N K layers by QZR_nonzero."""
    k_df = df_summary[df_summary["sublayer"] == "K"].sort_values(
        "QZR_nonzero", ascending=False
    )
    worst = k_df.head(n_worst)
    rows = []
    for _, row in worst.iterrows():
        lname = row["layer_name"]
        li    = int(row["layer_idx"])
        stats = stats_dict.get(lname)
        if stats is None:
            continue
        all_ratios = stats.ratio_reservoir_array()
        if len(all_ratios) == 0:
            continue
        sorted_r = np.sort(all_ratios)
        cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        idx = np.linspace(0, len(sorted_r) - 1, min(n_points, len(sorted_r)), dtype=int)
        for j in idx:
            rows.append({
                "layer_idx": li,
                "sublayer": "K",
                "layer_name": lname,
                "ratio": float(sorted_r[j]),
                "cdf": float(cdf[j]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"  CDF CSV → {filepath}  ({len(df)} rows)")


# =============================================================================
# CSV Validation
# =============================================================================

def validate_csv(filepath, required_columns, min_rows, critical_columns=None):
    """Validate CSV: existence, columns, row count, no NaN in critical cols."""
    assert os.path.exists(filepath), f"[CSV FAIL] Missing: {filepath}"
    df = pd.read_csv(filepath)
    for col in required_columns:
        assert col in df.columns, f"[CSV FAIL] Missing column '{col}' in {filepath}"
    assert len(df) >= min_rows, (
        f"[CSV FAIL] {filepath}: {len(df)} rows < min {min_rows}"
    )
    if critical_columns:
        for col in critical_columns:
            if col in df.columns:
                n_nan = df[col].isna().sum()
                assert n_nan == 0, (
                    f"[CSV FAIL] {filepath}: {n_nan} NaN in critical column '{col}'"
                )
    print(f"  [CSV OK] {filepath} ({len(df)} rows, {len(df.columns)} cols)")


# =============================================================================
# Diagnostic Run
# =============================================================================

def run_diagnostic(model, loader, n_step: int, desc: str = "Diag"):
    """n_step forward+backward passes. lr=0 → no weight change."""
    from aihwkit.optim import AnalogSGD

    optimizer = AnalogSGD(model.parameters(), lr=0.0)
    model.train()
    torch.manual_seed(SEED)

    for step, batch in enumerate(tqdm(loader, total=n_step, desc=desc)):
        if step >= n_step:
            break
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            start_positions=batch["start_positions"].to(DEVICE),
            end_positions=batch["end_positions"].to(DEVICE),
        )
        outputs.loss.backward()
        optimizer.step()   # lr=0 → no weight change; flushes tile grad buffers


# =============================================================================
# Root Cause Analytics (for Figure A)
# =============================================================================

def compute_rootcause(stats_dict: dict, dac_bits: int = DAC_BITS,
                      adc_bits: int = ADC_BITS, figure_id: str = "A",
                      run_tag: str = ""):
    """Run consistency checks and compute auto-diagnosis verdict."""
    zero_thresh = 1.0 / (2**dac_bits - 2)

    # Consistency check
    for s in stats_dict.values():
        qzr_nz = np.mean(s.qzr_nz_steps) if s.qzr_nz_steps else 0.0
        odr    = np.array(s.odr_steps) if s.odr_steps else np.array([1.0])
        thresh = (2**dac_bits - 1) / 2
        if qzr_nz > 0.5 and np.median(odr) < thresh:
            print(f"  [WARN] {s.sublayer} L{s.layer_idx}: "
                  f"QZR_nz={qzr_nz:.3f}>0.5 but ODR_p50={np.median(odr):.1f}<{thresh:.1f}")

    rows = [s.summary("baseline", dac_bits=dac_bits, adc_bits=adc_bits,
                       figure_id=figure_id, run_tag=run_tag)
            for s in stats_dict.values()]
    df   = (pd.DataFrame(rows)
              .sort_values(["layer_idx", "sublayer"])
              .reset_index(drop=True))

    kv  = df[df["sublayer"].isin(["K", "V"])]
    qo  = df[df["sublayer"].isin(["Q", "O"])]
    ffn = df[df["sublayer"].isin(["FFN1", "FFN2"])]

    ezr_kv    = kv["EZR"].mean()
    qzr_nz_kv = kv["QZR_nonzero"].mean()
    cos_kv    = kv["cosine_sim"].mean()
    r50_kv    = kv["ratio_q50"].mean()

    print(f"\n[Root Cause] zero_thresh={zero_thresh:.6f} (1/{2**dac_bits-2})")
    print(f"  K/V  EZR={ezr_kv:.4f}, QZR_nz={qzr_nz_kv:.4f}, cosine={cos_kv:.4f}, "
          f"ratio_q50={r50_kv:.6f}")
    print(f"  Q/O  EZR={qo['EZR'].mean():.4f}, QZR_nz={qo['QZR_nonzero'].mean():.4f}")
    if len(ffn) > 0:
        print(f"  FFN  EZR={ffn['EZR'].mean():.4f}, "
              f"QZR_nz={ffn['QZR_nonzero'].mean():.4f}")

    structural = (ezr_kv > 0.3 and (kv["QZR_all"].mean() - ezr_kv) < 0.1)
    bulk_tiny  = (ezr_kv < 0.1 and qzr_nz_kv > 0.5 and r50_kv < zero_thresh)
    if structural and bulk_tiny:
        verdict = f"혼합: EZR={ezr_kv:.2%}, QZR_nz={qzr_nz_kv:.2%} (양쪽 기여)"
    elif structural:
        verdict = "구조적 exact-zero 지배 (마스크/attention sparsity)"
    elif bulk_tiny:
        verdict = "bulk tiny / outlier-dominant (scale 이슈)"
    else:
        verdict = (f"비정형: EZR={ezr_kv:.2%}, QZR_nz={qzr_nz_kv:.2%} "
                   f"ratio_q50={r50_kv:.6f}")
    print(f"  [판정] K/V: {verdict}")
    return df, verdict


# =============================================================================
# Figure A — QKVO+FFN Root Cause (2×3, 6 sublayers)
# =============================================================================

def figure_A(df: pd.DataFrame, verdict: str, stats_dict: dict,
             dac_bits: int = DAC_BITS, n_step: int = N_STEP):
    zero_thresh = 1.0 / (2**dac_bits - 2)

    def to_mat(col):
        mat = np.full((N_LAYERS, len(SUBLAYER_ORDER)), np.nan)
        for _, row in df.iterrows():
            li = int(row["layer_idx"])
            sl = row["sublayer"]
            if sl in SUBLAYER_ORDER and li < N_LAYERS:
                mat[li, SUBLAYER_ORDER.index(sl)] = row[col]
        return mat

    qzr_nz_mat = to_mat("QZR_nonzero")
    ezr_mat    = to_mat("EZR")
    l2r_mat    = to_mat("l2_retention")

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    fig.suptitle(
        f"Figure A — QKVO+FFN Backward Root Cause Diagnosis "
        f"(BERT-base, {n_step} steps × batch {BATCH_SIZE})\n"
        f"DAC={dac_bits}-bit, zero_thresh=1/{2**dac_bits-2}={zero_thresh:.5f}, "
        f"6 sublayers (QKVO+FFN1+FFN2)",
        fontsize=10, y=1.01
    )

    def heatmap(ax, mat, title, cmap, vmin=0.0, vmax=1.0, label=""):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="upper",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(SUBLAYER_ORDER)))
        ax.set_xticklabels(SUBLAYER_ORDER, fontsize=8)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, label=label, shrink=0.85)

    # [0,0] QZR_nonzero heatmap
    heatmap(axes[0, 0], qzr_nz_mat,
            "(a) QZR_nonzero (mask-excluded)\n∈[0,1]: high = bulk-tiny/outlier-dom",
            "plasma", vmin=0.0, vmax=1.0, label="QZR_nonzero")

    # [0,1] EZR heatmap
    heatmap(axes[0, 1], ezr_mat,
            "(b) EZR — Exact Zero Ratio (mask-excluded)\nhigh = structural sparsity",
            "YlOrRd", vmin=0.0, vmax=1.0, label="EZR")

    # [0,2] l2_retention heatmap (v3: replaces cosine_sim)
    heatmap(axes[0, 2], l2r_mat,
            "(c) L2 Retention ||dy_q|| / ||dy||\n(mask-excluded, higher = better)",
            "RdYlGn", vmin=0.0, vmax=1.0, label="l2_retention")

    # [1,0] Bar: ratio_q50 per sublayer + zero_thresh line
    ax = axes[1, 0]
    sublayer_means = {
        sl: df[df["sublayer"] == sl]["ratio_q50"].mean() if len(df[df["sublayer"] == sl]) > 0
        else 0.0
        for sl in SUBLAYER_ORDER
    }
    colors6 = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#9467BD", "#8C564B"]
    bars = ax.bar(SUBLAYER_ORDER,
                  [sublayer_means[sl] for sl in SUBLAYER_ORDER],
                  color=colors6, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.axhline(zero_thresh, color="red", ls="--", lw=1.5,
               label=f"zero_thresh=1/{2**dac_bits-2}={zero_thresh:.5f}")
    for bar, sl in zip(bars, SUBLAYER_ORDER):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + zero_thresh * 0.05,
                f"{sublayer_means[sl]:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("ratio_q50 = q50(|dy|/absmax)")
    ax.set_title("(d) Median ratio per sublayer (mask-excluded)\n"
                 "ratio < zero_thresh → quantized to zero", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # [1,1] CDF: worst 3 K layers by QZR_nonzero (v3: uses ratio_reservoir)
    ax = axes[1, 1]
    k_df   = df[df["sublayer"] == "K"].sort_values("QZR_nonzero", ascending=False)
    worst3 = k_df.head(3)
    colors3 = ["#e41a1c", "#377eb8", "#4daf4a"]
    for (_, row), col in zip(worst3.iterrows(), colors3):
        lname  = row["layer_name"]
        li     = int(row["layer_idx"])
        stats  = stats_dict.get(lname)
        if stats is None:
            continue
        all_ratios = stats.ratio_reservoir_array()
        if len(all_ratios) == 0:
            continue
        sorted_r = np.sort(all_ratios)
        cdf      = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        idx = np.linspace(0, len(sorted_r) - 1, min(2000, len(sorted_r)), dtype=int)
        ax.plot(sorted_r[idx], cdf[idx], color=col, lw=1.5,
                label=f"K L{li}  QZR_nz={row['QZR_nonzero']:.3f}")
    ax.axvline(zero_thresh, color="red", ls="--", lw=1.5,
               label=f"zero_thresh={zero_thresh:.5f}")
    ax.set_xlabel("|dy|/absmax ratio"); ax.set_ylabel("CDF")
    ax.set_title("(e) CDF of |dy|/absmax — worst 3 K layers\n"
                 "left of threshold → quantized to zero", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # [1,2] Text panel: auto-diagnosis summary
    ax = axes[1, 2]
    ax.axis("off")
    kv  = df[df["sublayer"].isin(["K", "V"])]
    ffn = df[df["sublayer"].isin(["FFN1", "FFN2"])]
    lines = [
        "Auto-Diagnosis Summary",
        "=" * 34,
        f"DAC bits:     {dac_bits}",
        f"inp_res:      1/{2**dac_bits-2} = {zero_thresh:.6f}",
        f"step_size:    {2.0 * INP_BOUND / (2**dac_bits - 2):.6f}",
        f"zero_thresh:  {zero_thresh:.6f}",
        "",
        f"K/V EZR:      {kv['EZR'].mean():.4f}",
        f"K/V QZR_all:  {kv['QZR_all'].mean():.4f}",
        f"K/V QZR_nz:   {kv['QZR_nonzero'].mean():.4f}",
        f"K/V cosine:   {kv['cosine_sim'].mean():.4f}",
        f"K/V l2_ret:   {kv['l2_retention'].mean():.4f}",
        f"K/V r_q50:    {kv['ratio_q50'].mean():.6f}",
        "",
    ]
    if len(ffn) > 0:
        lines += [
            f"FFN EZR:      {ffn['EZR'].mean():.4f}",
            f"FFN QZR_nz:   {ffn['QZR_nonzero'].mean():.4f}",
            f"FFN cosine:   {ffn['cosine_sim'].mean():.4f}",
            f"FFN l2_ret:   {ffn['l2_retention'].mean():.4f}",
            "",
        ]
    lines.append("판정 (K/V):")
    # word-wrap verdict
    words, line, wrapped = verdict.split(" "), "", []
    for w in words:
        if len(line) + len(w) + 1 > 28:
            wrapped.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        wrapped.append(line)
    lines += wrapped
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    ax.set_title("(f) Diagnosis Panel", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_A, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure A → {FIG_A}")


# =============================================================================
# Figure B — IO Resolution Sweep (2×2)
# =============================================================================

def figure_B(df_b: pd.DataFrame):
    bits_list_all = sorted(df_b["dac_bits"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Figure B — IO Resolution Sweep (BERT-base, "
        f"bits={bits_list_all})\n"
        f"sweep: dac_bits=adc_bits=bits, {N_STEP_SWEEP} steps; "
        f"baseline: dac=7b, adc=9b, {N_STEP} steps",
        fontsize=10, y=1.01
    )

    sublayer_colors = {
        "Q": "#4C72B0", "K": "#DD8452", "V": "#55A868",
        "O": "#C44E52", "FFN1": "#9467BD", "FFN2": "#8C564B"
    }
    sublayer_markers = {
        "Q": "o", "K": "s", "V": "^", "O": "D", "FFN1": "p", "FFN2": "h"
    }

    # [0,0] Line: bits × sublayer → QZR_nonzero
    ax = axes[0, 0]
    for sl in SUBLAYER_ORDER:
        sub = df_b[df_b["sublayer"] == sl].groupby("dac_bits")["QZR_nonzero"].mean()
        if len(sub) == 0:
            continue
        ax.plot(sub.index, sub.values,
                color=sublayer_colors[sl], marker=sublayer_markers[sl],
                label=sl, lw=1.5, markersize=6)
    if 7 in bits_list_all:
        ax.axvline(7, color="gray", ls=":", lw=1.2, label="baseline 7b")
    ax.set_xlabel("bits (dac_bits)"); ax.set_ylabel("QZR_nonzero")
    ax.set_title("(a) QZR_nonzero vs bits per sublayer\n(lower = better)", fontsize=9)
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    ax.set_xticks(bits_list_all)

    # [0,1] Line: bits × sublayer → cosine_sim
    ax = axes[0, 1]
    for sl in SUBLAYER_ORDER:
        sub = df_b[df_b["sublayer"] == sl].groupby("dac_bits")["cosine_sim"].mean()
        if len(sub) == 0:
            continue
        ax.plot(sub.index, sub.values,
                color=sublayer_colors[sl], marker=sublayer_markers[sl],
                label=sl, lw=1.5, markersize=6)
    if 7 in bits_list_all:
        ax.axvline(7, color="gray", ls=":", lw=1.2, label="baseline 7b")
    ax.set_xlabel("bits (dac_bits)"); ax.set_ylabel("cosine_sim")
    ax.set_title("(b) cosine_sim vs bits per sublayer\n(higher = better)", fontsize=9)
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    ax.set_xticks(bits_list_all)

    # [1,0] Heatmap: bits(y) × layer(x) for K QZR_nonzero
    ax = axes[1, 0]
    k_df  = df_b[df_b["sublayer"] == "K"]
    mat_k = np.full((len(bits_list_all), N_LAYERS), np.nan)
    for bi, b in enumerate(bits_list_all):
        for li in range(N_LAYERS):
            sub = k_df[(k_df["dac_bits"] == b) & (k_df["layer_idx"] == li)]
            if len(sub) > 0:
                mat_k[bi, li] = sub["QZR_nonzero"].mean()
    im = ax.imshow(mat_k, aspect="auto", cmap="plasma", origin="upper", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
    ax.set_yticks(range(len(bits_list_all)))
    ax.set_yticklabels([f"{b}b" for b in bits_list_all])
    ax.set_xlabel("Encoder Layer"); ax.set_ylabel("bits")
    ax.set_title("(c) K QZR_nonzero: bits × layer", fontsize=9)
    plt.colorbar(im, ax=ax, label="QZR_nonzero", shrink=0.85)

    # [1,1] Heatmap: bits(y) × layer(x) for V QZR_nonzero
    ax = axes[1, 1]
    v_df  = df_b[df_b["sublayer"] == "V"]
    mat_v = np.full((len(bits_list_all), N_LAYERS), np.nan)
    for bi, b in enumerate(bits_list_all):
        for li in range(N_LAYERS):
            sub = v_df[(v_df["dac_bits"] == b) & (v_df["layer_idx"] == li)]
            if len(sub) > 0:
                mat_v[bi, li] = sub["QZR_nonzero"].mean()
    im = ax.imshow(mat_v, aspect="auto", cmap="plasma", origin="upper", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
    ax.set_yticks(range(len(bits_list_all)))
    ax.set_yticklabels([f"{b}b" for b in bits_list_all])
    ax.set_xlabel("Encoder Layer"); ax.set_ylabel("bits")
    ax.set_title("(d) V QZR_nonzero: bits × layer", fontsize=9)
    plt.colorbar(im, ax=ax, label="QZR_nonzero", shrink=0.85)

    plt.tight_layout()
    fig.savefig(FIG_B, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure B → {FIG_B}")


# =============================================================================
# Figure C — Solutions Comparison (4×2) + CDF
# =============================================================================

# 10 variants mapping to 5 solution categories:
#   Cat 1 — Layer-wise precision:     "layer_prec"
#   Cat 2 — nm_thres (p_clip 통합):   "nm_thres_p50/p80/p90/p95"
#           clip_target → absmax percentile → per-layer nm_thres
#   Cat 3 — nm_thres + mixed prec:    "nmthres_mixed"
#   Cat 4 — Alt bound method:         "avg_absmax", "constant_nm"
#   Cat 5 — All combined:             "all_combined"
#           nm_thres(clip=5%) + AVERAGE_ABS_MAX + mixed precision
# QZR targets for mixed precision sweep
QZR_TARGETS = (0.20, 0.10, 0.05)

C_VARIANTS = [
    "baseline",       # uniform 8-bit, ABS_MAX, no mitigation
    "lp_q20",         # Cat 1: layer_prec, QZR target < 0.20 (loose)
    "lp_q10",         # Cat 1: layer_prec, QZR target < 0.10
    "lp_q05",         # Cat 1: layer_prec, QZR target < 0.05 (strict)
    "nm_thres_p50",   # Cat 2: clip=50% → absmax p50 (aggressive)
    "nm_thres_p80",   # Cat 2: clip=20% → absmax p80
    "nm_thres_p90",   # Cat 2: clip=10% → absmax p90
    "nm_thres_p95",   # Cat 2: clip=5%  → absmax p95 (conservative)
    "nmthres_mixed",  # Cat 3: nm_thres(clip=5%) + lp_q10
    "avg_absmax",     # Cat 4: AVERAGE_ABS_MAX backward noise mgmt
    "constant_nm",    # Cat 4: CONSTANT backward noise mgmt (calibrated)
    "all_combined",   # Cat 5: nm_thres + AVERAGE_ABS_MAX + lp_q10
]
C_COLORS = [
    "#4C72B0",  # baseline — blue
    "#FFBB78",  # lp_q20 — light orange
    "#DD8452",  # lp_q10 — orange
    "#D62728",  # lp_q05 — dark red
    "#FFD700",  # nm_thres_p50 — gold
    "#17BECF",  # nm_thres_p80 — cyan
    "#55A868",  # nm_thres_p90 — green
    "#8172B2",  # nm_thres_p95 — purple
    "#C44E52",  # nmthres_mixed — red
    "#937860",  # avg_absmax — brown
    "#DA8BC3",  # constant_nm — pink
    "#2CA02C",  # all_combined — dark green
]


def figure_C(df_c: pd.DataFrame, cdf_data: dict = None):
    """Generate Figure C: 4×2 grid (bars + heatmaps + CDF).

    cdf_data: {variant_name: {layer_key: np.ndarray(absmax)}} for CDF subplot.
    """
    VARIANTS = [v for v in C_VARIANTS if v in df_c["variant"].unique()]
    COLORS   = [C_COLORS[C_VARIANTS.index(v)] for v in VARIANTS]

    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle(
        f"Figure C — Solutions ({len(VARIANTS)} variants, BERT-base, "
        f"{DAC_BITS}-bit baseline, {N_STEP} steps × batch {BATCH_SIZE})\n"
        f"Cat1: layer_prec | Cat2: nm_thres_pXX | "
        f"Cat3: nmthres+mixed | Cat4: avg_absmax/constant_nm",
        fontsize=10, y=1.01
    )

    kv_df = df_c[df_c["sublayer"].isin(["K", "V"])]
    x     = np.arange(2)
    n_v   = len(VARIANTS)
    width = max(0.08, 0.72 / n_v)
    offsets = np.linspace(-(n_v - 1) * width / 2, (n_v - 1) * width / 2, n_v)

    # ---- helper for grouped bar ----
    def _grouped_bar(ax, metric, ylabel, title, higher_better=True):
        for vi, (variant, color) in enumerate(zip(VARIANTS, COLORS)):
            var_df = kv_df[kv_df["variant"] == variant]
            vals = []
            for sl in ["K", "V"]:
                sub = var_df[var_df["sublayer"] == sl][metric]
                vals.append(float(sub.mean()) if len(sub) > 0 else float("nan"))
            bars = ax.bar(x + offsets[vi], vals, width=width, label=variant,
                          color=color, alpha=0.8, edgecolor="k", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=5, rotation=60)
        ax.set_xticks(x); ax.set_xticklabels(["K", "V"])
        ax.set_xlabel("Sublayer"); ax.set_ylabel(ylabel)
        direction = "higher = better" if higher_better else "lower = better"
        ax.set_title(f"{title}\n({direction})", fontsize=9)
        ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3, axis="y")

    # Row 0: grouped bars
    _grouped_bar(axes[0, 0], "QZR_nonzero", "QZR_nonzero",
                 "(a) K/V Mean QZR_nonzero", higher_better=False)
    _grouped_bar(axes[0, 1], "cosine_sim", "cosine_sim",
                 "(b) K/V Mean cosine_sim", higher_better=True)

    # Row 1: more bars
    _grouped_bar(axes[1, 0], "l2_retention", "l2_retention",
                 "(c) K/V Mean l2_retention", higher_better=True)
    _grouped_bar(axes[1, 1], "p_clip", "p_clip",
                 "(d) K/V Mean p_clip", higher_better=False)

    # Row 2: heatmaps
    def build_variant_layer_mat(sublayer: str):
        mat = np.full((len(VARIANTS), N_LAYERS), np.nan)
        for vi, variant in enumerate(VARIANTS):
            sub_df = df_c[(df_c["variant"] == variant) & (df_c["sublayer"] == sublayer)]
            for _, row in sub_df.iterrows():
                li = int(row["layer_idx"])
                if li < N_LAYERS:
                    mat[vi, li] = row["QZR_nonzero"]
        return mat

    for ci, (sl, title) in enumerate([("K", "(e) K QZR_nonzero"),
                                       ("V", "(f) V QZR_nonzero")]):
        ax = axes[2, ci]
        mat = build_variant_layer_mat(sl)
        im = ax.imshow(mat, aspect="auto", cmap="plasma", origin="upper",
                       vmin=0.0, vmax=1.0)
        ax.set_xticks(range(N_LAYERS))
        ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_yticks(range(len(VARIANTS)))
        ax.set_yticklabels(VARIANTS, fontsize=7)
        ax.set_xlabel("Encoder Layer"); ax.set_ylabel("Variant")
        ax.set_title(f"{title}: variant × layer", fontsize=9)
        plt.colorbar(im, ax=ax, label="QZR_nonzero", shrink=0.85)

    # Row 3: CDF of per-vector absmax for worst K/V layer (all variants)
    ax_cdf_k = axes[3, 0]
    ax_cdf_v = axes[3, 1]

    if cdf_data:
        base_df = df_c[df_c["variant"] == "baseline"]
        for ax_cdf, sl, panel in [(ax_cdf_k, "K", "(g)"), (ax_cdf_v, "V", "(h)")]:
            sl_df = base_df[base_df["sublayer"] == sl].sort_values(
                "QZR_nonzero", ascending=False)
            worst_li = int(sl_df.iloc[0]["layer_idx"]) if len(sl_df) > 0 else 0
            lkey = f"L{worst_li}_{sl}"

            for vi, (variant, color) in enumerate(zip(VARIANTS, COLORS)):
                vdata = cdf_data.get(variant, {})
                arr = vdata.get(lkey, np.array([]))
                if len(arr) == 0:
                    continue
                sorted_a = np.sort(arr)
                cdf_vals = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
                n_pts = min(2000, len(sorted_a))
                idx = np.linspace(0, len(sorted_a) - 1, n_pts, dtype=int)
                ax_cdf.plot(sorted_a[idx], cdf_vals[idx], label=variant,
                            color=color, linewidth=1.2, alpha=0.8)

            ax_cdf.set_xlabel("per-vector absmax")
            ax_cdf.set_ylabel("CDF")
            ax_cdf.set_title(
                f"{panel} CDF of per-vector absmax — worst {sl} (L{worst_li})",
                fontsize=9)
            ax_cdf.legend(fontsize=6, ncol=2)
            ax_cdf.grid(True, alpha=0.3)
            ax_cdf.set_xscale("log")
    else:
        for ax_cdf in [ax_cdf_k, ax_cdf_v]:
            ax_cdf.text(0.5, 0.5, "No CDF data", ha="center", va="center",
                        transform=ax_cdf.transAxes, fontsize=12, color="gray")

    plt.tight_layout()
    fig.savefig(FIG_C, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure C → {FIG_C}")


# =============================================================================
# Main
# =============================================================================

def main():
    torch.manual_seed(SEED)
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # [1] Data — load once with enough samples for all runs               #
    # ------------------------------------------------------------------ #
    print("\n[1/6] Loading data ...")
    tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
    n_step_max = max(N_STEP, N_STEP_SWEEP)
    loader     = load_data(tokenizer, n_step=n_step_max, batch_size=BATCH_SIZE)

    # ------------------------------------------------------------------ #
    # [2] Baseline run — needed for A, B, C                              #
    # ------------------------------------------------------------------ #
    print(f"\n[2/6] Baseline {DAC_BITS}-bit (DAC=ADC={DAC_BITS}), {N_STEP} steps ...")
    inp_res_base = 1.0 / (2**DAC_BITS - 2)   # 1/(2^8-2) = 1/254
    mask_buf_a = MaskBuffer()
    model_a    = create_model(nm_thres=0.0,
                              dac_bits=DAC_BITS, adc_bits=ADC_BITS)
    mask_buf_a.register(model_a)
    stats_baseline, handles_a = register_hooks(model_a, mask_buf_a,
                                               inp_res=inp_res_base)
    run_diagnostic(model_a, loader, n_step=N_STEP, desc="A-baseline")
    for h in handles_a:
        h.remove()
    del model_a
    torch.cuda.empty_cache()
    gc.collect()

    # Save raw absmax for baseline (used by ECDF, theta calibration)
    save_absmax_npz(stats_baseline,
                    os.path.join(OUT_DIR, f"absmax_raw_A_baseline_{DAC_BITS}b.npz"))

    # ------------------------------------------------------------------ #
    # [3] Figure A — root cause analysis + figure                        #
    # ------------------------------------------------------------------ #
    if RUN_A:
        print("\n[3/6] Computing root cause → Figure A ...")
        df_a, verdict_a = compute_rootcause(stats_baseline, dac_bits=DAC_BITS,
                                            adc_bits=ADC_BITS, figure_id="A",
                                            run_tag=RUN_TAG)
        df_a.to_csv(CSV_A_SUMMARY, index=False)
        print(f"  → {CSV_A_SUMMARY}")

        # Steps CSV
        all_a_step_records = []
        for s in stats_baseline.values():
            all_a_step_records.extend(s.step_records(
                "baseline", dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                figure_id="A", run_tag=RUN_TAG))
        pd.DataFrame(all_a_step_records).to_csv(CSV_A_STEPS, index=False)
        print(f"  → {CSV_A_STEPS} ({len(all_a_step_records)} rows)")

        # CDF CSV
        save_cdf_csv(stats_baseline, df_a, CSV_A_CDF)

        figure_A(df_a, verdict_a, stats_baseline, dac_bits=DAC_BITS, n_step=N_STEP)

    # ------------------------------------------------------------------ #
    # [4] Figure B — bits sweep [4,6,8,10,12] + baseline 8-bit           #
    # ------------------------------------------------------------------ #
    if RUN_B:
        print(f"\n[4/6] Figure B: bits sweep {[4,6,8,10,12]}, {N_STEP_SWEEP} steps each ...")
        all_b_rows = []
        all_b_step_records = []

        # baseline contribution (8-bit, N_STEP runs)
        for s in stats_baseline.values():
            all_b_rows.append(s.summary("baseline", dac_bits=DAC_BITS,
                                        adc_bits=ADC_BITS, figure_id="B",
                                        run_tag=RUN_TAG))
            all_b_step_records.extend(s.step_records(
                "baseline", dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                figure_id="B", run_tag=RUN_TAG))

        for bits in [4, 6, 8, 10, 12]:
            print(f"\n  B-sweep bits={bits} (dac_bits=adc_bits={bits}) ...")
            inp_res_b  = 1.0 / (2**bits - 2)
            mask_buf_b = MaskBuffer()
            model_b    = create_model(nm_thres=0.0,
                                      dac_bits=bits, adc_bits=bits)
            mask_buf_b.register(model_b)
            stats_b, handles_b = register_hooks(model_b, mask_buf_b, inp_res=inp_res_b,
                                                 store_sweep=False)
            run_diagnostic(model_b, loader, n_step=N_STEP_SWEEP, desc=f"B-{bits}b")
            for h in handles_b:
                h.remove()
            del model_b
            torch.cuda.empty_cache()
            gc.collect()

            save_absmax_npz(stats_b,
                            os.path.join(OUT_DIR, f"absmax_raw_B_sweep_{bits}b.npz"))

            for s in stats_b.values():
                all_b_rows.append(s.summary("sweep", dac_bits=bits, adc_bits=bits,
                                            figure_id="B", run_tag=RUN_TAG))
                all_b_step_records.extend(s.step_records(
                    "sweep", dac_bits=bits, adc_bits=bits,
                    figure_id="B", run_tag=RUN_TAG))

        df_b = (pd.DataFrame(all_b_rows)
                  .sort_values(["dac_bits", "layer_idx", "sublayer"])
                  .reset_index(drop=True))
        df_b.to_csv(CSV_B_SUMMARY, index=False)
        print(f"  → {CSV_B_SUMMARY}")

        pd.DataFrame(all_b_step_records).to_csv(CSV_B_STEPS, index=False)
        print(f"  → {CSV_B_STEPS} ({len(all_b_step_records)} rows)")

        figure_B(df_b)

    # ------------------------------------------------------------------ #
    # [5] Figure C — 8 variants (4 solution categories)                  #
    #     Cat1: layer_prec | Cat2: nm_thres_p90/p95                      #
    #     Cat3: nmthres_mixed | Cat4: avg_absmax, constant_nm            #
    #     Cat5: all_combined (nm_thres+avg_absmax+mixed)                 #
    # ------------------------------------------------------------------ #
    if RUN_C:
        print(f"\n[5/6] Figure C: {len(C_VARIANTS)} variants, {N_STEP} steps each ...")
        all_c_rows = []
        all_c_step_records = []
        cdf_data = {}   # {variant: {layer_key: absmax_array}} for CDF plots

        def _collect(variant_name, stats_dict):
            """Collect summary, steps, and CDF data from a stats_dict."""
            cdf_data[variant_name] = {}
            for s in stats_dict.values():
                all_c_rows.append(s.summary(variant_name, dac_bits=DAC_BITS,
                                            adc_bits=ADC_BITS, figure_id="C",
                                            run_tag=RUN_TAG))
                all_c_step_records.extend(s.step_records(
                    variant_name, dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                    figure_id="C", run_tag=RUN_TAG))
                lkey = f"L{s.layer_idx}_{s.sublayer}"
                arr = s.absmax_array()
                if len(arr) > 0:
                    cdf_data[variant_name][lkey] = arr

        def _run_variant(variant_name, model, mask_buf, inp_res=inp_res_base,
                         inp_res_map=None, nm_thres_map=None,
                         p99_hook_obj=None):
            """Run diagnostic, collect, cleanup. Returns stats_dict."""
            stats, handles = register_hooks(
                model, mask_buf, inp_res=inp_res,
                inp_res_map=inp_res_map, nm_thres_map=nm_thres_map,
                store_sweep=True)   # store_sweep=True for CDF data
            run_diagnostic(model, loader, n_step=N_STEP, desc=f"C-{variant_name}")
            for h in handles:
                h.remove()
            if p99_hook_obj:
                p99_hook_obj.remove()
            save_absmax_npz(stats,
                            os.path.join(OUT_DIR, f"absmax_raw_C_{variant_name}.npz"))
            _collect(variant_name, stats)
            del model
            torch.cuda.empty_cache()
            gc.collect()
            return stats

        # ---- Calibration from baseline ----
        print("  Calibrating per-layer severity & thetas from baseline ...")
        severity  = calibrate_layer_severity(stats_baseline)

        # p_clip → nm_thres: clip_target=0.50 means "cap top 50% outlier vectors"
        #   clip=50% → absmax_p50, clip=20% → p80, clip=10% → p90, clip=5% → p95
        CLIP_TARGETS = (0.50, 0.20, 0.10, 0.05)
        thetas = calibrate_per_layer_thetas(stats_baseline,
                                            clip_targets=CLIP_TARGETS)

        # Compute bit allocations for each QZR target
        bits_maps = {}
        for qt in QZR_TARGETS:
            print(f"\n  --- Bit allocation for QZR target < {qt} ---")
            bits_maps[qt] = allocate_precision(
                stats_baseline, base_bits=DAC_BITS,
                min_bits=4, max_bits=12, qzr_target=qt)
        # Default bits_map for Cat3/Cat5 (use q10)
        bits_map = bits_maps.get(0.10, bits_maps[QZR_TARGETS[0]])
        print_severity_report(severity, bits_map)

        # Global constant theta for Cat4 CONSTANT: median of per-layer clip=5% thetas
        tmap_5pct = thetas.get(0.05, {})
        constant_theta = float(np.median(list(tmap_5pct.values()))) if tmap_5pct else 0.1
        print(f"  constant_nm theta (median of clip=5% thetas): {constant_theta:.6f}")

        # ================================================================
        # variant 1: baseline (reuse stats_baseline)
        # ================================================================
        _collect("baseline", stats_baseline)

        # ================================================================
        # Cat 1 — variants 2-4: layer_prec at QZR targets 0.20, 0.10, 0.05
        # ================================================================
        qzr_to_name = {0.20: "lp_q20", 0.10: "lp_q10", 0.05: "lp_q05"}
        for qt in QZR_TARGETS:
            vname = qzr_to_name[qt]
            bm = bits_maps[qt]
            irm = {k: 1.0 / (2**b - 2) for k, b in bm.items()}
            print(f"  C-variant: {vname} (QZR<{qt}, avg={np.mean(list(bm.values())):.1f}b) ...")
            mb = MaskBuffer()
            mdl = create_model(nm_thres=0.0, dac_bits=DAC_BITS, adc_bits=ADC_BITS)
            mb.register(mdl)
            set_per_layer_config(mdl, bits_map=bm)
            _run_variant(vname, mdl, mb, inp_res_map=irm)

        # ================================================================
        # Cat 2 — variants 3-4: nm_thres_p90, nm_thres_p95
        #   clip_target=0.10 → nm_thres_p90 (cap top 10% vectors)
        #   clip_target=0.05 → nm_thres_p95 (cap top 5% vectors)
        # ================================================================
        clip_to_name = {0.50: "nm_thres_p50", 0.20: "nm_thres_p80",
                        0.10: "nm_thres_p90", 0.05: "nm_thres_p95"}
        for ct in CLIP_TARGETS:
            vname = clip_to_name[ct]
            tmap = thetas[ct]
            if not tmap:
                print(f"  [WARN] No absmax data for {vname} — skipping")
                continue
            print(f"  C-variant: {vname} (clip={ct:.0%}, per-layer, "
                  f"{len(tmap)} layers) ...")
            mb = MaskBuffer()
            mdl = create_model(nm_thres=0.0, dac_bits=DAC_BITS, adc_bits=ADC_BITS)
            mb.register(mdl)
            set_per_layer_config(mdl, thres_map=tmap)
            _run_variant(vname, mdl, mb, nm_thres_map=tmap)

        # ================================================================
        # Cat 3 — nmthres_mixed (clip=5% + lp_q10)
        # ================================================================
        tmap95 = thetas.get(0.05, {})
        inp_res_map_q10 = {k: 1.0 / (2**b - 2) for k, b in bits_map.items()}
        if tmap95:
            print(f"  C-variant: nmthres_mixed (clip=5% thres + lp_q10 bits) ...")
            mb = MaskBuffer()
            mdl = create_model(nm_thres=0.0, dac_bits=DAC_BITS, adc_bits=ADC_BITS)
            mb.register(mdl)
            set_per_layer_config(mdl, bits_map=bits_map, thres_map=tmap95)
            _run_variant("nmthres_mixed", mdl, mb,
                         inp_res_map=inp_res_map_q10, nm_thres_map=tmap95)
        else:
            print("  [WARN] No p95 theta data — skipping nmthres_mixed")

        # ================================================================
        # Cat 4 — variant 6: avg_absmax (AVERAGE_ABS_MAX backward)
        # ================================================================
        print("  C-variant: avg_absmax (AVERAGE_ABS_MAX backward) ...")
        mb = MaskBuffer()
        mdl = create_model(nm_thres=0.0, dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                           bwd_noise_mgmt="AVERAGE_ABS_MAX")
        mb.register(mdl)
        _run_variant("avg_absmax", mdl, mb)

        # ================================================================
        # Cat 4 — variant 7: constant_nm (CONSTANT backward, calibrated)
        # ================================================================
        print(f"  C-variant: constant_nm (CONSTANT bwd, theta={constant_theta:.6f}) ...")
        mb = MaskBuffer()
        mdl = create_model(nm_thres=constant_theta, dac_bits=DAC_BITS,
                           adc_bits=ADC_BITS, bwd_noise_mgmt="CONSTANT")
        mb.register(mdl)
        _run_variant("constant_nm", mdl, mb)

        # ================================================================
        # Cat 5 — variant 8: all_combined
        #   nm_thres(p95) + AVERAGE_ABS_MAX + mixed precision
        # ================================================================
        if tmap95:
            print("  C-variant: all_combined (nm_thres_p95 + AVG_ABSMAX + layer bits) ...")
            mb = MaskBuffer()
            mdl = create_model(nm_thres=0.0, dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                               bwd_noise_mgmt="AVERAGE_ABS_MAX")
            mb.register(mdl)
            set_per_layer_config(mdl, bits_map=bits_map, thres_map=tmap95)
            _run_variant("all_combined", mdl, mb,
                         inp_res_map=inp_res_map_q10, nm_thres_map=tmap95)
        else:
            print("  [WARN] No p95 theta data — skipping all_combined")

        # ---- Save CSVs & Figure ----
        df_c = (pd.DataFrame(all_c_rows)
                  .sort_values(["variant", "layer_idx", "sublayer"])
                  .reset_index(drop=True))
        df_c.to_csv(CSV_C_SUMMARY, index=False)
        print(f"  → {CSV_C_SUMMARY}")

        pd.DataFrame(all_c_step_records).to_csv(CSV_C_STEPS, index=False)
        print(f"  → {CSV_C_STEPS} ({len(all_c_step_records)} rows)")

        figure_C(df_c, cdf_data=cdf_data)

    # Free stats_baseline after all figures are done
    del stats_baseline
    gc.collect()

    # ------------------------------------------------------------------ #
    # [6] CSV Validation + Summary                                       #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("[6/6] Validation + Summary")
    print("=" * 60)

    COMMON_COLS = ["figure_id", "run_tag", "variant", "dac_bits", "adc_bits",
                   "inp_bound", "inp_res", "res_ratio", "step_size", "nm_thres",
                   "layer_idx", "sublayer"]
    METRIC_COLS = ["EZR", "QZR_all", "QZR_nonzero", "ODR", "cosine_sim",
                   "l2_retention", "rel_l2_error", "clip_rate_scaled",
                   "ratio_q50", "ratio_q90", "ratio_q99",
                   "absmax_q50", "absmax_q90", "absmax_q99", "absmax_q999"]
    STEP_EXTRA  = ["step_idx", "n_vec"]

    if RUN_A:
        validate_csv(CSV_A_SUMMARY, COMMON_COLS + METRIC_COLS, min_rows=72,
                     critical_columns=["QZR_nonzero", "l2_retention"])
        validate_csv(CSV_A_STEPS, COMMON_COLS + STEP_EXTRA + METRIC_COLS,
                     min_rows=72 * N_STEP,
                     critical_columns=["QZR_nonzero"])
        validate_csv(CSV_A_CDF, ["layer_idx", "ratio", "cdf"], min_rows=100)
    if RUN_B:
        validate_csv(CSV_B_SUMMARY, COMMON_COLS + METRIC_COLS, min_rows=432,
                     critical_columns=["QZR_nonzero", "l2_retention"])
        validate_csv(CSV_B_STEPS, COMMON_COLS + STEP_EXTRA + METRIC_COLS,
                     min_rows=72 * N_STEP)
    if RUN_C:
        # 12 variants × 72 layers = 864 min rows
        validate_csv(CSV_C_SUMMARY, COMMON_COLS + METRIC_COLS, min_rows=864,
                     critical_columns=["QZR_nonzero", "l2_retention"])
        validate_csv(CSV_C_STEPS, COMMON_COLS + STEP_EXTRA + METRIC_COLS,
                     min_rows=72 * N_STEP)

    # File listing
    files_to_check = []
    if RUN_A:
        files_to_check.extend([CSV_A_SUMMARY, CSV_A_STEPS, CSV_A_CDF, FIG_A])
    if RUN_B:
        files_to_check.extend([CSV_B_SUMMARY, CSV_B_STEPS, FIG_B])
    if RUN_C:
        files_to_check.extend([CSV_C_SUMMARY, CSV_C_STEPS, FIG_C])
    for f in files_to_check:
        status = "OK" if os.path.exists(f) else "MISSING"
        print(f"  [{status}] {f}")

    # K/V delta table
    if RUN_C:
        print("\nK/V QZR_nonzero per variant (Δ vs baseline):")
        kv_c    = df_c[df_c["sublayer"].isin(["K", "V"])]
        base_k  = kv_c[(kv_c["variant"] == "baseline") & (kv_c["sublayer"] == "K")]["QZR_nonzero"].mean()
        base_v  = kv_c[(kv_c["variant"] == "baseline") & (kv_c["sublayer"] == "V")]["QZR_nonzero"].mean()
        for variant in C_VARIANTS:
            sub = kv_c[kv_c["variant"] == variant]
            if len(sub) == 0:
                continue
            k_val = sub[sub["sublayer"] == "K"]["QZR_nonzero"].mean()
            v_val = sub[sub["sublayer"] == "V"]["QZR_nonzero"].mean()
            print(f"  {variant:<15}: K={k_val:.4f} (Δ={k_val-base_k:+.4f}), "
                  f"V={v_val:.4f} (Δ={v_val-base_v:+.4f})")

    # bits crossover (K QZR_nz < 0.2)
    if RUN_B:
        print("\nBits crossover (K QZR_nonzero < 0.2):")
        k_b = df_b[df_b["sublayer"] == "K"].groupby("dac_bits")["QZR_nonzero"].mean()
        crossover = k_b[k_b < 0.2]
        if len(crossover) > 0:
            print(f"  Minimum bits for K QZR_nz < 0.2: {int(crossover.index.min())}-bit")
        else:
            print(f"  No crossover found — K QZR_nz >= 0.2 for all bits tested")
            print(f"  Closest: {int(k_b.idxmin())}-bit with QZR_nz={k_b.min():.4f}")


if __name__ == "__main__":
    main()
