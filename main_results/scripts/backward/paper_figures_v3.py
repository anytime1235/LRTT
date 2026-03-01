"""paper_figures.py — 논문용 Figure A/B/C 생성 (v3: AIHWKit-consistent metrics)

Three publication-quality figures based on diag_kv_rootcause.py patterns,
extended with FFN1/FFN2 layers and parameterized adc_bits.

- Figure A: QKVO+FFN Root Cause (2×3, 6 sublayers)
- Figure B: IO Resolution Sweep (bits=[4,6,8,10,12], 2×2)
- Figure C: Solutions Comparison (4 variants: baseline, sto_round,
            nm_thres_cal, p99_clip), 2×2

Key differences from diag_kv_rootcause.py:
  - FFN1/FFN2 sublayers added (6 total instead of 4)
  - adc_bits fully parameterized
  - zero_thresh = inp_res = 1/(2^b-2)  (not 1/(2^b-1))
  - nm_thres_cal: actual live run with calibrated theta
  - P99ClipHook: per-vector p99 gradient clip via output tensor hook
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
parser.add_argument("--run-tag",      type=str, default="v3")
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
DAC_BITS       = 7
ADC_BITS       = 9
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

def create_rpu_config(nm_thres=0.0, sto_round=False, dac_bits=DAC_BITS, adc_bits=ADC_BITS):
    """SingleRPU with noise-free SoftBoundsDevice. inp_res = 1/(2^dac_bits - 2).

    Matches optuna_bert_squad_tiki.py create_single_rpu_config() device config.
    """
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import SoftBoundsDevice
    from aihwkit.simulator.configs.utils import NoiseManagementType

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
        io.inp_sto_round    = sto_round
    rpu.backward.nm_thres               = nm_thres
    rpu.mapping.digital_bias            = True
    rpu.mapping.weight_scaling_omega    = 1.0
    rpu.mapping.weight_scaling_columnwise = True
    return rpu


# =============================================================================
# Model Creation
# =============================================================================

def create_model(nm_thres=0.0, sto_round=False, dac_bits=DAC_BITS, adc_bits=ADC_BITS):
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
    rpu = create_rpu_config(nm_thres=nm_thres, sto_round=sto_round,
                            dac_bits=dac_bits, adc_bits=adc_bits)
    model = convert_to_analog(
        model, rpu,
        exclude_modules=[n for n in all_linear if n not in target]
    )

    # Pass 2: nontarget (FFN) — same config, update will be nooped
    nt_rpu = create_rpu_config(nm_thres=nm_thres, sto_round=sto_round,
                               dac_bits=dac_bits, adc_bits=adc_bits)
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
          f"nm_thres={nm_thres:.4g}, sto_round={sto_round}, "
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
                run_tag: str = "", sto_round: bool = False) -> dict:
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
            "sto_round":        sto_round,
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
                     run_tag: str = "", sto_round: bool = False) -> list:
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
                "sto_round":        sto_round,
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
                   store_sweep: bool = True) -> tuple:
    """Register full backward hooks on all AnalogLinear matching the regex.

    Includes both QKVO (target) and FFN1/FFN2 (nontarget frozen) layers.
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
        stats = LayerStats(name=name, layer_idx=layer_idx, sublayer=sublayer,
                           inp_res=inp_res, nm_thres=nm_thres,
                           store_sweep=store_sweep)
        stats_dict[name] = stats

        def make_hook(s, mb):
            def fn(mod, gin, gout):
                if gout[0] is not None:
                    s.update(gout[0], mask=mb.val)
            return fn

        handles.append(module.register_full_backward_hook(make_hook(stats, mask_buf)))

    sublayers_found = sorted(set(s.sublayer for s in stats_dict.values()))
    print(f"[Hook] {len(stats_dict)} hooks, sublayers={sublayers_found}, "
          f"inp_res={inp_res:.6f}, nm_thres={nm_thres:.4g}")
    return stats_dict, handles


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
# Figure C — Solutions Comparison (2×2)
# =============================================================================

def figure_C(df_c: pd.DataFrame):
    VARIANTS = ["baseline", "sto_round", "nm_thres_cal", "p99_clip"]
    COLORS   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Figure C — Solutions Comparison (4 variants, BERT-base, "
        f"{N_STEP} steps × batch {BATCH_SIZE})\n"
        f"baseline(no nm_thres, 7b) | sto_round | "
        f"nm_thres_cal(p95 K/V absmax) | p99_clip(output hook)",
        fontsize=10, y=1.01
    )

    kv_df = df_c[df_c["sublayer"].isin(["K", "V"])]
    x     = np.arange(2)
    n_v   = len(VARIANTS)
    width = 0.18
    offsets = np.linspace(-(n_v - 1) * width / 2, (n_v - 1) * width / 2, n_v)

    # [0,0] Grouped bar: K/V mean QZR_nonzero
    ax = axes[0, 0]
    for vi, (variant, color) in enumerate(zip(VARIANTS, COLORS)):
        var_df = kv_df[kv_df["variant"] == variant]
        vals = []
        for sl in ["K", "V"]:
            sub = var_df[var_df["sublayer"] == sl]["QZR_nonzero"]
            vals.append(float(sub.mean()) if len(sub) > 0 else float("nan"))
        bars = ax.bar(x + offsets[vi], vals, width=width, label=variant,
                      color=color, alpha=0.8, edgecolor="k", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(["K", "V"])
    ax.set_xlabel("Sublayer"); ax.set_ylabel("QZR_nonzero")
    ax.set_title("(a) K/V Mean QZR_nonzero per variant\n(lower = better)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # [0,1] Grouped bar: K/V mean cosine_sim
    ax = axes[0, 1]
    for vi, (variant, color) in enumerate(zip(VARIANTS, COLORS)):
        var_df = kv_df[kv_df["variant"] == variant]
        vals = []
        for sl in ["K", "V"]:
            sub = var_df[var_df["sublayer"] == sl]["cosine_sim"]
            vals.append(float(sub.mean()) if len(sub) > 0 else float("nan"))
        bars = ax.bar(x + offsets[vi], vals, width=width, label=variant,
                      color=color, alpha=0.8, edgecolor="k", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=7, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(["K", "V"])
    ax.set_xlabel("Sublayer"); ax.set_ylabel("cosine_sim")
    ax.set_title("(b) K/V Mean cosine_sim per variant\n(higher = better)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    def build_variant_layer_mat(sublayer: str):
        """Returns (n_variants, N_LAYERS) matrix of QZR_nonzero."""
        mat = np.full((len(VARIANTS), N_LAYERS), np.nan)
        for vi, variant in enumerate(VARIANTS):
            sub_df = df_c[(df_c["variant"] == variant) & (df_c["sublayer"] == sublayer)]
            for _, row in sub_df.iterrows():
                li = int(row["layer_idx"])
                if li < N_LAYERS:
                    mat[vi, li] = row["QZR_nonzero"]
        return mat

    # [1,0] Heatmap: K QZR_nonzero (variant × layer)
    ax = axes[1, 0]
    mat_k = build_variant_layer_mat("K")
    im = ax.imshow(mat_k, aspect="auto", cmap="plasma", origin="upper", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels(VARIANTS, fontsize=9)
    ax.set_xlabel("Encoder Layer"); ax.set_ylabel("Variant")
    ax.set_title("(c) K QZR_nonzero: variant × layer", fontsize=9)
    plt.colorbar(im, ax=ax, label="QZR_nonzero", shrink=0.85)

    # [1,1] Heatmap: V QZR_nonzero (variant × layer)
    ax = axes[1, 1]
    mat_v = build_variant_layer_mat("V")
    im = ax.imshow(mat_v, aspect="auto", cmap="plasma", origin="upper", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(N_LAYERS))
    ax.set_xticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
    ax.set_yticks(range(len(VARIANTS)))
    ax.set_yticklabels(VARIANTS, fontsize=9)
    ax.set_xlabel("Encoder Layer"); ax.set_ylabel("Variant")
    ax.set_title("(d) V QZR_nonzero: variant × layer", fontsize=9)
    plt.colorbar(im, ax=ax, label="QZR_nonzero", shrink=0.85)

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
    print(f"\n[2/6] Baseline 7-bit, {N_STEP} steps ...")
    inp_res_7  = 1.0 / (2**DAC_BITS - 2)   # 1/126
    mask_buf_a = MaskBuffer()
    model_a    = create_model(nm_thres=0.0, sto_round=False,
                              dac_bits=DAC_BITS, adc_bits=ADC_BITS)
    mask_buf_a.register(model_a)
    stats_baseline, handles_a = register_hooks(model_a, mask_buf_a,
                                               inp_res=inp_res_7)
    run_diagnostic(model_a, loader, n_step=N_STEP, desc="A-baseline")
    for h in handles_a:
        h.remove()
    del model_a
    torch.cuda.empty_cache()
    gc.collect()

    # Save raw absmax for baseline (used by ECDF, theta calibration)
    save_absmax_npz(stats_baseline,
                    os.path.join(OUT_DIR, "absmax_raw_A_baseline_7b.npz"))

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
    # [4] Figure B — bits sweep [4,6,8,10,12] + baseline 7-bit           #
    # ------------------------------------------------------------------ #
    if RUN_B:
        print(f"\n[4/6] Figure B: bits sweep {[4,6,8,10,12]}, {N_STEP_SWEEP} steps each ...")
        all_b_rows = []
        all_b_step_records = []

        # baseline contribution (7-bit, N_STEP runs)
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
            model_b    = create_model(nm_thres=0.0, sto_round=False,
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
    # [5] Figure C — 4 variants (baseline, p99_clip,                     #
    #                             nm_thres_cal, sto_round)                #
    # ------------------------------------------------------------------ #
    if RUN_C:
        print(f"\n[5/6] Figure C: 4 variants, {N_STEP} steps each ...")
        all_c_rows = []
        all_c_step_records = []

        # variant 1: baseline (reuse stats_baseline)
        for s in stats_baseline.values():
            all_c_rows.append(s.summary("baseline", dac_bits=DAC_BITS,
                                        adc_bits=ADC_BITS, figure_id="C",
                                        run_tag=RUN_TAG))
            all_c_step_records.extend(s.step_records(
                "baseline", dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                figure_id="C", run_tag=RUN_TAG))

        # variant 2: p99_clip — P99ClipHook clips gradient before tile backward
        print("  C-variant: p99_clip ...")
        mask_buf_p99 = MaskBuffer()
        model_p99    = create_model(nm_thres=0.0, sto_round=False,
                                    dac_bits=DAC_BITS, adc_bits=ADC_BITS)
        mask_buf_p99.register(model_p99)
        p99_hook = P99ClipHook(mask_buf_p99)
        p99_hook.register(model_p99)
        stats_p99, handles_p99 = register_hooks(model_p99, mask_buf_p99, inp_res=inp_res_7,
                                                 store_sweep=False)
        run_diagnostic(model_p99, loader, n_step=N_STEP, desc="C-p99_clip")
        for h in handles_p99:
            h.remove()
        p99_hook.remove()
        del model_p99
        torch.cuda.empty_cache()
        gc.collect()
        save_absmax_npz(stats_p99,
                        os.path.join(OUT_DIR, "absmax_raw_C_p99_clip.npz"))
        for s in stats_p99.values():
            all_c_rows.append(s.summary("p99_clip", dac_bits=DAC_BITS,
                                        adc_bits=ADC_BITS, figure_id="C",
                                        run_tag=RUN_TAG))
            all_c_step_records.extend(s.step_records(
                "p99_clip", dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                figure_id="C", run_tag=RUN_TAG))

        # variant 3: nm_thres_cal — calibrate theta from baseline K/V absmax
        print("  C-variant: nm_thres_cal (calibrating theta) ...")
        kv_absmax_arrs = [
            arr
            for s in stats_baseline.values()
            if s.sublayer in ["K", "V"]
            for arr in s._absmax_buf
            if len(arr) > 0
        ]
        if kv_absmax_arrs:
            kv_absmax_all = np.concatenate(kv_absmax_arrs)
            theta = float(np.quantile(kv_absmax_all, 0.95))
            print(f"  theta = {theta:.6f}  (p95 of K/V per-vector absmax)")

            mask_buf_nt = MaskBuffer()
            model_nt    = create_model(nm_thres=theta, sto_round=False,
                                       dac_bits=DAC_BITS, adc_bits=ADC_BITS)
            mask_buf_nt.register(model_nt)
            stats_nt, handles_nt = register_hooks(model_nt, mask_buf_nt,
                                                  inp_res=inp_res_7, nm_thres=theta,
                                                  store_sweep=False)
            run_diagnostic(model_nt, loader, n_step=N_STEP, desc="C-nm_thres_cal")
            for h in handles_nt:
                h.remove()
            del model_nt
            torch.cuda.empty_cache()
            gc.collect()
            save_absmax_npz(stats_nt,
                            os.path.join(OUT_DIR, "absmax_raw_C_nm_thres_cal.npz"))
            for s in stats_nt.values():
                all_c_rows.append(s.summary("nm_thres_cal", dac_bits=DAC_BITS,
                                            adc_bits=ADC_BITS, figure_id="C",
                                            run_tag=RUN_TAG))
                all_c_step_records.extend(s.step_records(
                    "nm_thres_cal", dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                    figure_id="C", run_tag=RUN_TAG))
        else:
            print("  [WARN] No K/V absmax data — skipping nm_thres_cal variant")

        # variant 4: sto_round
        print("  C-variant: sto_round ...")
        mask_buf_sr = MaskBuffer()
        model_sr    = create_model(nm_thres=0.0, sto_round=True,
                                   dac_bits=DAC_BITS, adc_bits=ADC_BITS)
        mask_buf_sr.register(model_sr)
        stats_sr, handles_sr = register_hooks(model_sr, mask_buf_sr, inp_res=inp_res_7,
                                               store_sweep=False)
        run_diagnostic(model_sr, loader, n_step=N_STEP, desc="C-sto_round")
        for h in handles_sr:
            h.remove()
        del model_sr
        torch.cuda.empty_cache()
        gc.collect()
        save_absmax_npz(stats_sr,
                        os.path.join(OUT_DIR, "absmax_raw_C_sto_round.npz"))
        for s in stats_sr.values():
            all_c_rows.append(s.summary("sto_round", dac_bits=DAC_BITS,
                                        adc_bits=ADC_BITS, figure_id="C",
                                        run_tag=RUN_TAG, sto_round=True))
            all_c_step_records.extend(s.step_records(
                "sto_round", dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                figure_id="C", run_tag=RUN_TAG, sto_round=True))

        df_c = (pd.DataFrame(all_c_rows)
                  .sort_values(["variant", "layer_idx", "sublayer"])
                  .reset_index(drop=True))
        df_c.to_csv(CSV_C_SUMMARY, index=False)
        print(f"  → {CSV_C_SUMMARY}")

        pd.DataFrame(all_c_step_records).to_csv(CSV_C_STEPS, index=False)
        print(f"  → {CSV_C_STEPS} ({len(all_c_step_records)} rows)")

        figure_C(df_c)

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
        validate_csv(CSV_C_SUMMARY, COMMON_COLS + METRIC_COLS, min_rows=288,
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
        for variant in ["baseline", "p99_clip", "nm_thres_cal", "sto_round"]:
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
