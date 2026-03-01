"""diag_kv_rootcause.py — K/V Backward Root Cause Diagnosis + Alternative Comparison

Three-part deep diagnosis following calib_nm_thres_tworun.py result (ΔQZR_K/V < 0.001):

- Part A: "outlier-dominant α" vs "structural bulk-tiny" root cause determination
          (EZR, QZR_nonzero, ODR, ratio-quantile, consistency check +
           attention_mask-based padding exclusion)
- Part B: nm_thres θ sweep CCR–QZR_nonzero–cosine Pareto analysis (fully offline)
- Part C: 3 alternative comparisons (sto_round, dac8bit, p99_scale emulation)

Usage:
  python diag_kv_rootcause.py                            # full run
  python diag_kv_rootcause.py --n-step 5 --batch-size 2  # smoke test
"""

import argparse
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
DAC_BITS        = 7
ADC_BITS        = 9
INP_BOUND       = 1.0
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR         = "/data/results/tikitakav1"
CSV_ROOTCAUSE   = f"{OUT_DIR}/metrics_rootcause.csv"
CSV_SWEEP       = f"{OUT_DIR}/metrics_nmthres_sweep.csv"
CSV_SOLUTIONS   = f"{OUT_DIR}/metrics_solutions.csv"
FIG_ROOTCAUSE   = f"{OUT_DIR}/fig_rootcause_diagnosis.pdf"
FIG_FEASIBILITY = f"{OUT_DIR}/fig_nmthres_feasibility.pdf"
FIG_SOLUTIONS   = f"{OUT_DIR}/fig_solution_comparison.pdf"

# theta sweep candidates
THETA_QUANTILES = [0.90, 0.95, 0.99, 0.995, 0.999]
SAFETY_FACTORS  = [1.0, 1.1, 1.2]

print(f"[Config] Device={DEVICE}, N_STEP={N_STEP}, BATCH={DIAG_BATCH_SIZE}")
print(f"[Config] DAC={DAC_BITS}-bit, ADC={ADC_BITS}-bit, INP_BOUND={INP_BOUND}")

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
# RPU Config (parameterized)
# =============================================================================


def create_rpu_config(nm_thres=0.0, sto_round=False, dac_bits=DAC_BITS):
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.configs.devices import IdealDevice
    from aihwkit.simulator.configs.utils import NoiseManagementType

    rpu = SingleRPUConfig(device=IdealDevice())
    for io in [rpu.forward, rpu.backward]:
        io.inp_bound        = INP_BOUND
        io.inp_res          = 1 / (2**dac_bits - 2)
        io.out_bound        = 12.0
        io.out_res          = 1 / (2**ADC_BITS - 2)
        io.noise_management = NoiseManagementType.ABS_MAX
        io.out_noise        = 0.0
        io.inp_sto_round    = sto_round
    rpu.backward.nm_thres               = nm_thres
    rpu.mapping.digital_bias            = True
    rpu.mapping.weight_scaling_omega    = 1.0
    rpu.mapping.weight_scaling_columnwise = True
    return rpu

# =============================================================================
# Model Creation (parameterized)
# =============================================================================


def create_model(nm_thres=0.0, sto_round=False, dac_bits=DAC_BITS):
    from aihwkit.nn import AnalogLinear
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.optim.context import AnalogContext

    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    target, nontarget, all_linear = _layer_names(model)

    rpu = create_rpu_config(nm_thres=nm_thres, sto_round=sto_round, dac_bits=dac_bits)
    model = convert_to_analog(model, rpu,
                              exclude_modules=[n for n in all_linear if n not in target])

    nt_rpu = create_rpu_config(nm_thres=nm_thres, sto_round=sto_round, dac_bits=dac_bits)
    model = convert_to_analog(model, nt_rpu,
                              exclude_modules=[n for n in all_linear if n not in nontarget])

    def _noop(x, d, *a, **kw):
        return None

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

    n_t   = sum(1 for n, m in model.named_modules()
                if isinstance(m, AnalogLinear) and n in target)
    n_all = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog tiles — target: {n_t}, frozen: {n_all - n_t}, "
          f"nm_thres={nm_thres}, sto_round={sto_round}, dac_bits={dac_bits}")
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
# MaskBuffer — captures attention_mask from each forward pass
# =============================================================================


class MaskBuffer:
    """Captures attention_mask from each forward pass for padding exclusion."""
    def __init__(self):
        self.val = None

    def register(self, model):
        def _hook(mod, args, kwargs):
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                self.val = kwargs['attention_mask'].bool().cpu()
        model.register_forward_pre_hook(_hook, with_kwargs=True)

# =============================================================================
# LayerStats — comprehensive per-layer accumulator
# =============================================================================


class LayerStats:
    """Per-layer backward stats with mask-aware padding exclusion."""

    def __init__(self, name, layer_idx, sublayer, dac_bits=DAC_BITS):
        self.name      = name
        self.layer_idx = layer_idx
        self.sublayer  = sublayer
        self.dac_bits  = dac_bits
        # dac_step = 2*INP_BOUND / (2^b - 1)
        # zero threshold = dac_step/2 = INP_BOUND / (2^b - 1)
        self.zero_thresh = INP_BOUND / (2**dac_bits - 1)   # 1/127 for 7-bit
        self.eps         = 1e-8

        # mask-excluded stats (per step)
        self.ezr_steps       = []   # exact zero ratio
        self.qzr_all_steps   = []   # QZR including exact zeros
        self.qzr_nz_steps    = []   # QZR conditioned on |dy| > 0
        self.odr_steps       = []   # absmax / median per vector → mean
        self.cosine_steps    = []
        self.ratio_q50_steps = []
        self.ratio_q90_steps = []
        self.ratio_q99_steps = []

        # buffers for offline Part B theta sweep (mask-excluded)
        # store list of (absmax_arr_step: np.ndarray (N,), ratio_arr_step: np.ndarray (N*D,))
        self._sweep_steps = []   # list of (absmax_np, ratio_np)

        # padding-included versions for comparison
        self.qzr_all_pad_steps = []
        self.ezr_pad_steps     = []

    def update(self, dy: torch.Tensor, mask=None):
        """
        dy: (B, S, D_out) or (B*S, D_out)
        mask: (B, S) bool, True=real token
        """
        with torch.no_grad():
            dy = dy.detach().float()

            # --- padding-included version ---
            dy_flat_pad = dy.reshape(-1, dy.shape[-1])   # (N_all, D)
            self._compute_pad(dy_flat_pad)

            # --- mask-excluded version ---
            if mask is not None and dy.dim() == 3:
                B, S, D = dy.shape
                if mask.shape == (B, S):
                    # move mask to same device as dy for indexing
                    mask_dev = mask.to(dy.device)
                    dy_real = dy[mask_dev]   # (N_real, D)
                else:
                    dy_real = dy_flat_pad
            else:
                dy_real = dy_flat_pad   # fallback: no mask

            self._compute_main(dy_real)

    def _compute_pad(self, dy_flat):
        abs_dy   = dy_flat.abs()
        ezr      = (abs_dy == 0).float().mean().item()
        self.ezr_pad_steps.append(ezr)
        absmax_v = abs_dy.max(dim=1).values.clamp(min=self.eps).unsqueeze(1)
        scaled   = dy_flat / absmax_v * INP_BOUND
        qzr_all  = (scaled.abs() < self.zero_thresh).float().mean().item()
        self.qzr_all_pad_steps.append(qzr_all)

    def _compute_main(self, dy_real):
        """Core computation on real-token (mask-excluded) vectors."""
        abs_dy = dy_real.abs()   # (N, D)
        N, D   = abs_dy.shape
        if N == 0:
            # edge case: all padding
            self.ezr_steps.append(0.0)
            self.qzr_all_steps.append(0.0)
            self.qzr_nz_steps.append(0.0)
            self.odr_steps.append(1.0)
            self.cosine_steps.append(1.0)
            self.ratio_q50_steps.append(0.0)
            self.ratio_q90_steps.append(0.0)
            self.ratio_q99_steps.append(0.0)
            self._sweep_steps.append((np.array([]), np.array([])))
            return

        # EZR
        ezr = (abs_dy == 0).float().mean().item()
        self.ezr_steps.append(ezr)

        # per-vector absmax
        absmax_v = abs_dy.max(dim=1).values          # (N,)
        alpha    = absmax_v.clamp(min=self.eps).unsqueeze(1)   # (N, 1)

        # element-wise ratio |dy| / absmax  ∈ [0, 1]
        ratio = abs_dy / alpha                        # (N, D)

        # QZR_all and QZR_nonzero
        zero_mask = (abs_dy == 0)
        qzr_all   = (ratio < self.zero_thresh).float().mean().item()
        nz_mask   = ~zero_mask
        if nz_mask.any():
            qzr_nz = (ratio[nz_mask] < self.zero_thresh).float().mean().item()
        else:
            qzr_nz = 0.0
        self.qzr_all_steps.append(qzr_all)
        self.qzr_nz_steps.append(qzr_nz)

        # ODR: absmax / median per vector
        absmed_v = abs_dy.median(dim=1).values.clamp(min=self.eps)
        odr      = (absmax_v / absmed_v).mean().item()
        self.odr_steps.append(odr)

        # Cosine sim: FP32 dy vs DAC-quantized dy
        dac_step = 2 * INP_BOUND / (2**self.dac_bits - 1)
        scaled   = dy_real / alpha * INP_BOUND
        dy_q     = (scaled / dac_step).round() * dac_step * alpha / INP_BOUND
        cos      = F.cosine_similarity(dy_real, dy_q, dim=1).mean().item()
        self.cosine_steps.append(cos)

        # ratio quantiles
        ratio_np = ratio.reshape(-1).cpu().float().numpy()
        self.ratio_q50_steps.append(float(np.quantile(ratio_np, 0.50)))
        self.ratio_q90_steps.append(float(np.quantile(ratio_np, 0.90)))
        self.ratio_q99_steps.append(float(np.quantile(ratio_np, 0.99)))

        # buffer for offline theta sweep
        absmax_np = absmax_v.cpu().float().numpy()   # (N,)
        self._sweep_steps.append((absmax_np, ratio_np))

    def consistency_check(self):
        """QZR_nonzero > 0.5 → ODR p50 should be ≥ (2^b - 1)/2; warn if not."""
        qzr_nz_mean = np.mean(self.qzr_nz_steps) if self.qzr_nz_steps else 0.0
        odr_vals     = np.array(self.odr_steps) if self.odr_steps else np.array([1.0])
        threshold    = (2**self.dac_bits - 1) / 2   # 63.5 for 7-bit
        if qzr_nz_mean > 0.5 and np.median(odr_vals) < threshold:
            print(f"  [WARN] {self.sublayer} L{self.layer_idx}: "
                  f"QZR_nz={qzr_nz_mean:.3f}>0.5 but "
                  f"ODR_p50={np.median(odr_vals):.1f}<{threshold:.1f} — 계산 축 오류 의심")

    def summary(self, label="baseline"):
        def _m(lst): return float(np.mean(lst)) if lst else float("nan")
        return {
            "layer_name":       self.name,
            "layer_idx":        self.layer_idx,
            "sublayer":         self.sublayer,
            "variant":          label,
            "dac_bits":         self.dac_bits,
            "EZR_mean":         _m(self.ezr_steps),
            "QZR_all_mean":     _m(self.qzr_all_steps),
            "QZR_nonzero_mean": _m(self.qzr_nz_steps),
            "ODR_mean":         _m(self.odr_steps),
            "cosine_sim":       _m(self.cosine_steps),
            "ratio_q50":        _m(self.ratio_q50_steps),
            "ratio_q90":        _m(self.ratio_q90_steps),
            "ratio_q99":        _m(self.ratio_q99_steps),
            "EZR_pad_mean":     _m(self.ezr_pad_steps),
            "QZR_all_pad_mean": _m(self.qzr_all_pad_steps),
        }

# =============================================================================
# Hook Registration (mask-aware)
# =============================================================================


def register_hooks(model, mask_buf, dac_bits=DAC_BITS):
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
                           sublayer=sublayer, dac_bits=dac_bits)
        stats_dict[name] = stats

        def make_hook(s, mb):
            def fn(mod, gin, gout):
                if gout[0] is not None:
                    s.update(gout[0], mask=mb.val)
            return fn

        handles.append(module.register_full_backward_hook(make_hook(stats, mask_buf)))

    print(f"[Hook] {len(stats_dict)} hooks, dac_bits={dac_bits}")
    return stats_dict, handles

# =============================================================================
# Diagnostic Run (no weight change: lr=0)
# =============================================================================


def run_diagnostic(model, loader, desc="Diag"):
    from aihwkit.optim import AnalogSGD

    optimizer = AnalogSGD(model.parameters(), lr=0.0)
    model.train()
    torch.manual_seed(SEED)

    for step, batch in enumerate(tqdm(loader, total=N_STEP, desc=desc)):
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
        optimizer.step()   # lr=0 → no weight change, but flushes tile grad buffers

# =============================================================================
# Part A — Root Cause Analytics
# =============================================================================


def compute_rootcause(stats_dict):
    """Run consistency checks and emit auto-diagnosis. Returns DataFrame."""
    print("\n[Part A] Consistency checks ...")
    for s in stats_dict.values():
        s.consistency_check()

    rows = [s.summary("baseline") for s in stats_dict.values()]
    df   = (pd.DataFrame(rows)
              .sort_values(["layer_idx", "sublayer"])
              .reset_index(drop=True))
    df.to_csv(CSV_ROOTCAUSE, index=False)
    print(f"  metrics_rootcause → {CSV_ROOTCAUSE}")

    # --- Auto diagnosis ---
    kv = df[df["sublayer"].isin(["K", "V"])]
    qo = df[df["sublayer"].isin(["Q", "O"])]
    zero_thresh = INP_BOUND / (2**DAC_BITS - 1)

    ezr_kv   = kv["EZR_mean"].mean()
    qzr_all_kv = kv["QZR_all_mean"].mean()
    qzr_nz_kv  = kv["QZR_nonzero_mean"].mean()
    ezr_qo   = qo["EZR_mean"].mean()
    qzr_nz_qo  = qo["QZR_nonzero_mean"].mean()

    print("\n" + "="*60)
    print("Part A — Root Cause Diagnosis")
    print("="*60)
    print(f"  zero_thresh = {zero_thresh:.6f} (1/{2**DAC_BITS - 1})")
    print(f"  K/V EZR_mean      = {ezr_kv:.4f}  (Q/O: {ezr_qo:.4f})")
    print(f"  K/V QZR_all_mean  = {qzr_all_kv:.4f}")
    print(f"  K/V QZR_nz_mean   = {qzr_nz_kv:.4f}  (Q/O: {qzr_nz_qo:.4f})")
    print(f"  K/V ratio_q50     = {kv['ratio_q50'].mean():.6f}")

    diagnoses = []
    structural = (ezr_kv > 0.3 and (qzr_all_kv - ezr_kv) < 0.1)
    bulk_tiny  = (ezr_kv < 0.1 and qzr_nz_kv > 0.5 and
                  kv["ratio_q50"].mean() < zero_thresh)

    if structural and bulk_tiny:
        verdict = (f"혼합: EZR={ezr_kv:.2%}, QZR_nz={qzr_nz_kv:.2%} (양쪽 기여)")
    elif structural:
        verdict = "구조적 exact-zero 지배 (마스크/attention sparsity)"
    elif bulk_tiny:
        verdict = "bulk tiny / outlier-dominant (scale 이슈)"
    else:
        verdict = (f"비정형: EZR={ezr_kv:.2%}, QZR_nz={qzr_nz_kv:.2%} "
                   f"(ratio_q50={kv['ratio_q50'].mean():.6f})")
    print(f"\n  [판정] K/V: {verdict}")

    # pad vs no-pad comparison
    ezr_pad_kv   = kv["EZR_pad_mean"].mean()
    qzr_pad_kv   = kv["QZR_all_pad_mean"].mean()
    print(f"\n  [Padding Effect] EZR: no-pad={ezr_kv:.4f}, pad={ezr_pad_kv:.4f}")
    print(f"                   QZR: no-pad={qzr_all_kv:.4f}, pad={qzr_pad_kv:.4f}")

    return df, verdict

# =============================================================================
# Part B — nm_thres θ Sweep (fully offline)
# =============================================================================


def _compute_qzr_nz_capped(sweep_steps, absmax_global, theta, zero_thresh, dac_bits):
    """
    Compute QZR_nonzero after capping alpha = min(absmax, theta).
    Uses per-step (absmax_np, ratio_np) pairs stored in _sweep_steps.
    ratio_np = |dy| / absmax_v (original uncapped ratio).
    After capping: ratio_capped = |dy| / min(absmax, theta) = ratio * absmax / min(absmax, theta).
    """
    numer = 0
    denom = 0
    for absmax_np, ratio_np in sweep_steps:
        if len(absmax_np) == 0:
            continue
        N = len(absmax_np)
        D = len(ratio_np) // N if N > 0 else 1
        if D == 0 or len(ratio_np) != N * D:
            continue
        # per-vector cap factor: absmax / min(absmax, theta)
        cap_factor = absmax_np / np.minimum(absmax_np, theta + 1e-12)   # (N,)
        # tile cap_factor to (N*D,)
        cap_factor_elem = np.repeat(cap_factor, D)                        # (N*D,)
        ratio_capped = ratio_np * cap_factor_elem                         # (N*D,)
        # nonzero mask: original ratio > 0 (exact zero stays exact zero)
        nz_mask = ratio_np > 0
        if nz_mask.sum() > 0:
            numer += (ratio_capped[nz_mask] < zero_thresh).sum()
            denom += nz_mask.sum()
    return float(numer / denom) if denom > 0 else 0.0


def _approx_cosine_after_cap(sweep_steps, theta, dac_bits):
    """
    Approximate cosine similarity after alpha capping using per-step stored data.
    ratio_np stores |dy|/absmax ∈ [0,1]. After cap:
      scaled_capped = dy / min(absmax, theta) * INP_BOUND
    We approximate cosine from ratio distribution.
    cos ≈ mean over vectors of: (sum ratio_capped * round(ratio_capped / dac_step) * dac_step) /
                                  (||ratio_capped|| * ||round(ratio_capped / dac_step) * dac_step||)
    Since we only have magnitudes (ratio = |dy|/absmax) we compute on ratio vectors.
    """
    dac_step = 2 * INP_BOUND / (2**dac_bits - 1)
    total_cos = 0.0
    count     = 0
    for absmax_np, ratio_np in sweep_steps:
        if len(absmax_np) == 0:
            continue
        N = len(absmax_np)
        D = len(ratio_np) // N if N > 0 else 1
        if D == 0 or len(ratio_np) != N * D:
            continue
        cap_factor = absmax_np / np.minimum(absmax_np, theta + 1e-12)   # (N,)
        # per-element scaled ratio after cap, scaled to [-1,1] range via INP_BOUND
        ratio_mat      = ratio_np.reshape(N, D)                  # (N, D)
        cap_mat        = cap_factor[:, None]                     # (N, 1)
        scaled_capped  = ratio_mat * cap_mat * INP_BOUND        # (N, D) in [0, INP_BOUND]
        # DAC quantize
        scaled_q       = np.round(scaled_capped / dac_step) * dac_step
        scaled_q       = np.clip(scaled_q, -INP_BOUND, INP_BOUND)
        # cosine sim (vectors are non-negative since we use |dy|/absmax)
        dot   = (scaled_capped * scaled_q).sum(axis=1)          # (N,)
        norm1 = np.linalg.norm(scaled_capped, axis=1)
        norm2 = np.linalg.norm(scaled_q,      axis=1)
        denom = norm1 * norm2
        valid = denom > 1e-8
        if valid.sum() > 0:
            total_cos += (dot[valid] / denom[valid]).sum()
            count     += valid.sum()
    return float(total_cos / count) if count > 0 else 0.0


def theta_sweep(stats_dict):
    """Fully offline Pareto analysis of θ sweep. Returns DataFrame."""
    print("\n[Part B] nm_thres θ sweep (offline) ...")
    zero_thresh = INP_BOUND / (2**DAC_BITS - 1)
    rows = []
    for name, stats in stats_dict.items():
        # pool all absmax values for quantile computation
        all_absmax = np.concatenate([step[0] for step in stats._sweep_steps
                                     if len(step[0]) > 0]) if stats._sweep_steps else np.array([])
        if len(all_absmax) == 0:
            continue
        for q in THETA_QUANTILES:
            for sf in SAFETY_FACTORS:
                theta = sf * float(np.quantile(all_absmax, q))
                ccr   = float((all_absmax > theta).mean())
                qzr_nz_capped = _compute_qzr_nz_capped(
                    stats._sweep_steps, all_absmax, theta, zero_thresh, stats.dac_bits)
                cos_capped = _approx_cosine_after_cap(
                    stats._sweep_steps, theta, stats.dac_bits)
                rows.append({
                    "layer_name": name,
                    "layer_idx":  stats.layer_idx,
                    "sublayer":   stats.sublayer,
                    "q":          q,
                    "safety":     sf,
                    "theta":      theta,
                    "CCR":        ccr,
                    "QZR_nonzero": qzr_nz_capped,
                    "cosine_sim": cos_capped,
                })

    df_sweep = (pd.DataFrame(rows)
                  .sort_values(["layer_idx", "sublayer", "q", "safety"])
                  .reset_index(drop=True))
    df_sweep.to_csv(CSV_SWEEP, index=False)
    print(f"  metrics_nmthres_sweep → {CSV_SWEEP}")

    # --- Feasibility assessment ---
    kv_sweep = df_sweep[df_sweep["sublayer"].isin(["K", "V"])]
    if len(kv_sweep) > 0:
        feasible = kv_sweep[
            (kv_sweep["QZR_nonzero"] <= 0.20) &
            (kv_sweep["cosine_sim"]  >= 0.90) &
            (kv_sweep["CCR"]         <= 0.05)
        ]
        print(f"\n  [Part B 판정]")
        if len(feasible) > 0:
            print(f"  nm_thres 충분: {len(feasible)} configs meet CCR<5%, "
                  f"QZR_nz<20%, cos>0.90")
        else:
            best = kv_sweep.loc[kv_sweep["QZR_nonzero"].idxmin()]
            print(f"  nm_thres 불충분 (structural):")
            print(f"    best QZR_nz={best['QZR_nonzero']:.4f} at "
                  f"q={best['q']}, sf={best['safety']}, theta={best['theta']:.6f}")
            print(f"    CCR={best['CCR']:.4f}, cos={best['cosine_sim']:.4f}")

    return df_sweep

# =============================================================================
# Part C — p99 Scaling Emulation (offline, from baseline stats)
# =============================================================================


def p99_scale_emulation(stats_dict):
    """
    Emulate p99-based scaling: alpha_p99 = p99(absmax_arr) per layer.
    QZR_nz and cosine recomputed offline.
    """
    zero_thresh = INP_BOUND / (2**DAC_BITS - 1)
    rows = []
    for name, stats in stats_dict.items():
        all_absmax = np.concatenate([step[0] for step in stats._sweep_steps
                                     if len(step[0]) > 0]) if stats._sweep_steps else np.array([])
        if len(all_absmax) == 0:
            continue
        alpha_p99 = float(np.quantile(all_absmax, 0.99))
        if alpha_p99 < 1e-8:
            alpha_p99 = 1e-8

        # use theta sweep with theta = alpha_p99 (fixed)
        qzr_nz = _compute_qzr_nz_capped(
            stats._sweep_steps, all_absmax, alpha_p99, zero_thresh, stats.dac_bits)
        cos_sim = _approx_cosine_after_cap(
            stats._sweep_steps, alpha_p99, stats.dac_bits)
        ccr     = float((all_absmax > alpha_p99).mean())

        rows.append({
            "layer_name":       name,
            "layer_idx":        stats.layer_idx,
            "sublayer":         stats.sublayer,
            "variant":          "p99_scale",
            "dac_bits":         stats.dac_bits,
            "EZR_mean":         float(np.mean(stats.ezr_steps)) if stats.ezr_steps else float("nan"),
            "QZR_all_mean":     float("nan"),   # not computed for p99
            "QZR_nonzero_mean": qzr_nz,
            "ODR_mean":         float(np.mean(stats.odr_steps)) if stats.odr_steps else float("nan"),
            "cosine_sim":       cos_sim,
            "ratio_q50":        float("nan"),
            "ratio_q90":        float("nan"),
            "ratio_q99":        float("nan"),
            "EZR_pad_mean":     float("nan"),
            "QZR_all_pad_mean": float("nan"),
            "CCR":              ccr,
        })
    return pd.DataFrame(rows).sort_values(["layer_idx", "sublayer"]).reset_index(drop=True)

# =============================================================================
# Figure 1 — Root Cause Diagnosis (2×3)
# =============================================================================


def figure_rootcause(df_baseline, verdict, stats_dict):
    SUBLAYER_ORDER = ["Q", "K", "V", "O"]
    N_LAYERS       = 12

    def to_mat(df, col):
        mat = np.full((N_LAYERS, 4), np.nan)
        for _, row in df.iterrows():
            li = int(row["layer_idx"])
            si = SUBLAYER_ORDER.index(row["sublayer"])
            if li < N_LAYERS:
                mat[li, si] = row[col]
        return mat

    qzr_nz_mat  = to_mat(df_baseline, "QZR_nonzero_mean")
    ezr_mat      = to_mat(df_baseline, "EZR_mean")
    cos_mat      = to_mat(df_baseline, "cosine_sim")
    ratio50_mat  = to_mat(df_baseline, "ratio_q50")
    zero_thresh  = INP_BOUND / (2**DAC_BITS - 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Figure 1 — K/V Backward Root Cause Diagnosis (BERT-base, {N_STEP} steps × batch {DIAG_BATCH_SIZE})\n"
        f"DAC={DAC_BITS}-bit, zero_thresh=1/{2**DAC_BITS-1}={zero_thresh:.5f}",
        fontsize=10, y=1.01
    )

    def heatmap(ax, mat, title, cmap, vmin=0.0, vmax=None, label=""):
        vmax_use = vmax if vmax is not None else float(np.nanmax(mat)) + 1e-8
        im = ax.imshow(mat, aspect="auto", cmap=cmap, origin="upper",
                       vmin=vmin, vmax=vmax_use)
        ax.set_xticks(range(4)); ax.set_xticklabels(SUBLAYER_ORDER)
        ax.set_yticks(range(N_LAYERS)); ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
        ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, label=label, shrink=0.85)

    # [0,0] QZR_nonzero heatmap
    heatmap(axes[0, 0], qzr_nz_mat,
            "(a) QZR_nonzero (mask-excluded)\n∈[0,1]: high=bulk tiny",
            "plasma", vmin=0.0, vmax=1.0, label="QZR_nonzero")

    # [0,1] EZR heatmap
    heatmap(axes[0, 1], ezr_mat,
            "(b) EZR — Exact Zero Ratio (mask-excluded)\nhigh=structural sparsity",
            "YlOrRd", vmin=0.0, vmax=1.0, label="EZR")

    # [0,2] cosine_sim heatmap
    heatmap(axes[0, 2], cos_mat,
            "(c) Cosine Similarity FP32 vs DAC-quantized\n(mask-excluded)",
            "RdYlGn", vmin=0.0, vmax=1.0, label="cosine_sim")

    # [1,0] Bar: ratio_q50 K/V vs Q/O + zero_thresh line
    ax = axes[1, 0]
    sublayer_means = {}
    for sl in SUBLAYER_ORDER:
        sub_df = df_baseline[df_baseline["sublayer"] == sl]
        sublayer_means[sl] = sub_df["ratio_q50"].mean() if len(sub_df) > 0 else 0.0
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(SUBLAYER_ORDER, [sublayer_means[sl] for sl in SUBLAYER_ORDER],
                  color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.axhline(zero_thresh, color="red", ls="--", lw=1.5, label=f"zero_thresh=1/{2**DAC_BITS-1}")
    for bar, sl in zip(bars, SUBLAYER_ORDER):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + zero_thresh * 0.05,
                f"{sublayer_means[sl]:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("ratio_q50 = q50(|dy|/absmax)")
    ax.set_title("(d) Median ratio per sublayer (mask-excluded)\n"
                 "ratio < zero_thresh → quantized to 0", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # [1,1] CDF: worst 3 K layers (highest QZR_nonzero)
    ax = axes[1, 1]
    kv_df  = df_baseline[df_baseline["sublayer"] == "K"].sort_values(
        "QZR_nonzero_mean", ascending=False)
    worst3 = kv_df.head(3)
    colors3 = ["#e41a1c", "#377eb8", "#4daf4a"]
    for (_, row), col in zip(worst3.iterrows(), colors3):
        lname  = row["layer_name"]
        li     = int(row["layer_idx"])
        stats  = stats_dict.get(lname)
        if stats is None or not stats._sweep_steps:
            continue
        all_ratios = np.concatenate([step[1] for step in stats._sweep_steps
                                     if len(step[1]) > 0])
        if len(all_ratios) == 0:
            continue
        sorted_r = np.sort(all_ratios)
        cdf      = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        # subsample for plot speed
        idx = np.linspace(0, len(sorted_r) - 1, min(2000, len(sorted_r)), dtype=int)
        ax.plot(sorted_r[idx], cdf[idx], color=col, lw=1.5,
                label=f"K L{li} QZR_nz={row['QZR_nonzero_mean']:.3f}")
    ax.axvline(zero_thresh, color="red", ls="--", lw=1.5, label=f"zero_thresh={zero_thresh:.5f}")
    ax.set_xlabel("|dy|/absmax ratio"); ax.set_ylabel("CDF")
    ax.set_title("(e) CDF of |dy|/absmax — worst 3 K layers\n"
                 "left of threshold = quantized to zero", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # [1,2] Text panel: auto-diagnosis result
    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        "Auto-Diagnosis Summary",
        "=" * 36,
        f"DAC bits:       {DAC_BITS}",
        f"zero_thresh:    {zero_thresh:.6f}",
        "",
        f"K/V EZR:        {df_baseline[df_baseline['sublayer'].isin(['K','V'])]['EZR_mean'].mean():.4f}",
        f"K/V QZR_all:    {df_baseline[df_baseline['sublayer'].isin(['K','V'])]['QZR_all_mean'].mean():.4f}",
        f"K/V QZR_nz:     {df_baseline[df_baseline['sublayer'].isin(['K','V'])]['QZR_nonzero_mean'].mean():.4f}",
        f"K/V cosine:     {df_baseline[df_baseline['sublayer'].isin(['K','V'])]['cosine_sim'].mean():.4f}",
        f"K/V ratio_q50:  {df_baseline[df_baseline['sublayer'].isin(['K','V'])]['ratio_q50'].mean():.6f}",
        "",
        "판정:",
    ]
    # word-wrap verdict
    words   = verdict.split(" ")
    line    = ""
    wrapped = []
    for w in words:
        if len(line) + len(w) + 1 > 30:
            wrapped.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        wrapped.append(line)
    lines += wrapped
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    ax.set_title("(f) Diagnosis Panel", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_ROOTCAUSE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 1 → {FIG_ROOTCAUSE}")

# =============================================================================
# Figure 2 — nm_thres Feasibility (2×2)
# =============================================================================


def figure_feasibility(df_sweep, df_baseline):
    SUBLAYER_ORDER = ["Q", "K", "V", "O"]
    N_LAYERS       = 12

    kv_sweep = df_sweep[df_sweep["sublayer"].isin(["K", "V"])].copy()
    # best theta per layer (lowest QZR_nonzero)
    best_per_layer = (kv_sweep.sort_values("QZR_nonzero")
                              .groupby(["layer_name", "sublayer"])
                              .first()
                              .reset_index())

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f"Figure 2 — nm_thres Feasibility Pareto Analysis (offline θ sweep)\n"
        f"DAC={DAC_BITS}-bit, {N_STEP} steps × batch {DIAG_BATCH_SIZE}",
        fontsize=10, y=1.01
    )

    q_colors = {0.90: "#1b9e77", 0.95: "#d95f02", 0.99: "#7570b3",
                0.995: "#e7298a", 0.999: "#66a61e"}
    sf_markers = {1.0: "o", 1.1: "s", 1.2: "^"}

    # [0,0] Pareto: CCR vs QZR_nonzero
    ax = axes[0, 0]
    for (li, sl), grp in kv_sweep.groupby(["layer_idx", "sublayer"]):
        for q in THETA_QUANTILES:
            for sf in SAFETY_FACTORS:
                pts = grp[(grp["q"] == q) & (grp["safety"] == sf)]
                if len(pts) == 0: continue
                ax.scatter(pts["CCR"].values, pts["QZR_nonzero"].values,
                           c=q_colors.get(q, "gray"),
                           marker=sf_markers.get(sf, "o"),
                           s=30, alpha=0.5)
    ax.axvline(0.05, color="red",    ls="--", lw=1.2, label="CCR=5%")
    ax.axhline(0.20, color="orange", ls="--", lw=1.2, label="QZR_nz=20%")
    # legend for q colors
    for q, c in q_colors.items():
        ax.scatter([], [], c=c, label=f"q={q}", s=30)
    ax.set_xlabel("CCR (Cap Clipping Rate)"); ax.set_ylabel("QZR_nonzero (capped)")
    ax.set_title("(a) Pareto: CCR vs QZR_nonzero\nK/V layers, all θ configs", fontsize=9)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # [0,1] Pareto: CCR vs cosine_sim
    ax = axes[0, 1]
    for (li, sl), grp in kv_sweep.groupby(["layer_idx", "sublayer"]):
        for q in THETA_QUANTILES:
            for sf in SAFETY_FACTORS:
                pts = grp[(grp["q"] == q) & (grp["safety"] == sf)]
                if len(pts) == 0: continue
                ax.scatter(pts["CCR"].values, pts["cosine_sim"].values,
                           c=q_colors.get(q, "gray"),
                           marker=sf_markers.get(sf, "o"),
                           s=30, alpha=0.5)
    ax.axvline(0.05, color="red",    ls="--", lw=1.2, label="CCR=5%")
    ax.axhline(0.90, color="green",  ls="--", lw=1.2, label="cos=0.90")
    for q, c in q_colors.items():
        ax.scatter([], [], c=c, label=f"q={q}", s=30)
    ax.set_xlabel("CCR (Cap Clipping Rate)"); ax.set_ylabel("cosine_sim (capped)")
    ax.set_title("(b) Pareto: CCR vs cosine_sim\nK/V layers, all θ configs", fontsize=9)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # [1,0] Heatmap: ΔQZR_nonzero (best θ* - baseline)
    ax = axes[1, 0]
    base_kv = df_baseline[df_baseline["sublayer"].isin(["K", "V"])][
        ["layer_idx", "sublayer", "QZR_nonzero_mean"]].copy()

    delta_mat = np.full((N_LAYERS, 4), np.nan)
    for _, row in best_per_layer.iterrows():
        li = int(row["layer_idx"])
        sl = row["sublayer"]
        si = SUBLAYER_ORDER.index(sl)
        base_val = base_kv[(base_kv["layer_idx"] == li) &
                            (base_kv["sublayer"]  == sl)]["QZR_nonzero_mean"]
        if len(base_val) > 0 and li < N_LAYERS:
            delta_mat[li, si] = row["QZR_nonzero"] - float(base_val.iloc[0])

    dmax = float(np.nanmax(np.abs(delta_mat))) + 1e-6
    im   = ax.imshow(delta_mat, aspect="auto", cmap="RdYlGn_r",
                     origin="upper", vmin=-dmax, vmax=dmax)
    ax.set_xticks(range(4)); ax.set_xticklabels(SUBLAYER_ORDER)
    ax.set_yticks(range(N_LAYERS)); ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)])
    ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer")
    ax.set_title("(c) ΔQZR_nonzero (best θ* − baseline)\ngreen = improvement", fontsize=9)
    plt.colorbar(im, ax=ax, label="ΔQZR_nonzero", shrink=0.85)

    # [1,1] Bar: CCR(best θ*) per sublayer + 1%/5% lines
    ax = axes[1, 1]
    ccr_by_sl = {sl: [] for sl in SUBLAYER_ORDER}
    for _, row in best_per_layer.iterrows():
        ccr_by_sl[row["sublayer"]].append(row["CCR"])
    sl_means = [np.mean(ccr_by_sl[sl]) if ccr_by_sl[sl] else 0.0 for sl in SUBLAYER_ORDER]
    sl_stds  = [np.std(ccr_by_sl[sl])  if ccr_by_sl[sl] else 0.0 for sl in SUBLAYER_ORDER]
    colors_b = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(SUBLAYER_ORDER, sl_means, yerr=sl_stds, capsize=4,
                  color=colors_b, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax.axhline(0.01,  color="red",    ls="--", lw=1.2, label="1%")
    ax.axhline(0.05,  color="orange", ls="--", lw=1.2, label="5%")
    for bar, val in zip(bars, sl_means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("CCR at best θ*")
    ax.set_title("(d) CCR at best θ* per sublayer\n(K/V only for best configs)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_FEASIBILITY, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 2 → {FIG_FEASIBILITY}")

# =============================================================================
# Figure 3 — Solution Comparison (1×3)
# =============================================================================


def figure_solutions(df_solutions):
    SUBLAYER_ORDER = ["Q", "K", "V", "O"]
    VARIANTS       = ["baseline", "sto_round", "dac8bit", "p99_scale"]
    COLORS         = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Figure 3 — Alternative Solution Comparison (K/V sublayers)\n"
        f"BERT-base, {N_STEP} steps × batch {DIAG_BATCH_SIZE}",
        fontsize=10, y=1.01
    )

    metrics = [
        ("QZR_nonzero_mean", "(a) K/V Mean QZR_nonzero\n(lower = better)", "QZR_nonzero"),
        ("cosine_sim",       "(b) K/V Mean cosine_sim\n(higher = better)",   "cosine_sim"),
        ("ODR_mean",         "(c) K/V Mean ODR (absmax/median)\n(lower = better)", "ODR"),
    ]

    for ax, (col, title, ylabel) in zip(axes, metrics):
        kv_df = df_solutions[df_solutions["sublayer"].isin(["K", "V"])]
        # per sublayer per variant mean
        n_variants = len(VARIANTS)
        width      = 0.35
        x          = np.arange(2)   # K and V
        offsets    = np.linspace(-(n_variants - 1) * width / 2,
                                   (n_variants - 1) * width / 2,
                                   n_variants)
        for vi, (variant, color) in enumerate(zip(VARIANTS, COLORS)):
            var_df = kv_df[kv_df["variant"] == variant]
            vals   = []
            for sl in ["K", "V"]:
                sub = var_df[var_df["sublayer"] == sl][col]
                vals.append(float(sub.mean()) if len(sub) > 0 else float("nan"))
            bars = ax.bar(x + offsets[vi], vals, width=width,
                          label=variant, color=color, alpha=0.8,
                          edgecolor="k", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom",
                            fontsize=7, rotation=45)
        ax.set_xticks(x); ax.set_xticklabels(["K", "V"])
        ax.set_xlabel("Sublayer"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_SOLUTIONS, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 3 → {FIG_SOLUTIONS}")

# =============================================================================
# Main
# =============================================================================


def main():
    torch.manual_seed(SEED)
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # [1] Data                                                             #
    # ------------------------------------------------------------------ #
    print("\n[1/7] Loading data ...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader    = load_data(tokenizer)

    # ------------------------------------------------------------------ #
    # [2] Baseline run (nm_thres=0, sto_round=False, dac_bits=7)          #
    # ------------------------------------------------------------------ #
    print("\n[2/7] Baseline run (nm_thres=0, 7-bit DAC) ...")
    mask_buf  = MaskBuffer()
    model     = create_model(nm_thres=0.0, sto_round=False, dac_bits=DAC_BITS)
    mask_buf.register(model)
    stats_baseline, handles = register_hooks(model, mask_buf, dac_bits=DAC_BITS)
    run_diagnostic(model, loader, desc="Baseline")
    for h in handles: h.remove()
    del model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # [3] Part A — Root Cause                                             #
    # ------------------------------------------------------------------ #
    print("\n[3/7] Part A — Root Cause Analytics ...")
    df_baseline, verdict = compute_rootcause(stats_baseline)

    # ------------------------------------------------------------------ #
    # [4] Part B — nm_thres Sweep (offline)                               #
    # ------------------------------------------------------------------ #
    print("\n[4/7] Part B — nm_thres Sweep ...")
    df_sweep = theta_sweep(stats_baseline)

    # ------------------------------------------------------------------ #
    # [5] Alternative Runs: sto_round + dac8bit                           #
    # ------------------------------------------------------------------ #
    all_variant_dfs = [df_baseline]

    for label, sto_round, dac_bits_v in [
        ("sto_round", True,  DAC_BITS),
        ("dac8bit",   False, 8),
    ]:
        print(f"\n[5/7] Variant: {label} ...")
        mask_buf_v = MaskBuffer()
        model_v    = create_model(nm_thres=0.0, sto_round=sto_round, dac_bits=dac_bits_v)
        mask_buf_v.register(model_v)
        stats_v, handles_v = register_hooks(model_v, mask_buf_v, dac_bits=dac_bits_v)
        run_diagnostic(model_v, loader, desc=label)
        for h in handles_v: h.remove()
        del model_v
        torch.cuda.empty_cache()

        rows_v = [s.summary(label=label) for s in stats_v.values()]
        df_v   = (pd.DataFrame(rows_v)
                    .sort_values(["layer_idx", "sublayer"])
                    .reset_index(drop=True))
        all_variant_dfs.append(df_v)

    # ------------------------------------------------------------------ #
    # [6] Part C — p99 scaling emulation (offline from baseline)          #
    # ------------------------------------------------------------------ #
    print("\n[6/7] Part C — p99_scale emulation (offline) ...")
    df_p99   = p99_scale_emulation(stats_baseline)
    all_variant_dfs.append(df_p99)

    df_solutions = pd.concat(all_variant_dfs, ignore_index=True)
    df_solutions.to_csv(CSV_SOLUTIONS, index=False)
    print(f"  metrics_solutions → {CSV_SOLUTIONS}")

    # ------------------------------------------------------------------ #
    # [7] Figures                                                         #
    # ------------------------------------------------------------------ #
    print("\n[7/7] Creating figures ...")
    figure_rootcause(df_baseline, verdict, stats_baseline)
    figure_feasibility(df_sweep, df_baseline)
    figure_solutions(df_solutions)

    # ------------------------------------------------------------------ #
    # Final summary                                                       #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    kv_bl = df_baseline[df_baseline["sublayer"].isin(["K", "V"])]
    print(f"  Baseline K/V QZR_nonzero : {kv_bl['QZR_nonzero_mean'].mean():.4f}")
    print(f"  Baseline K/V cosine_sim  : {kv_bl['cosine_sim'].mean():.4f}")
    print(f"  Baseline K/V ODR_mean    : {kv_bl['ODR_mean'].mean():.2f}")

    for variant in ["sto_round", "dac8bit", "p99_scale"]:
        kv_v = df_solutions[
            (df_solutions["variant"] == variant) &
            (df_solutions["sublayer"].isin(["K", "V"]))
        ]
        if len(kv_v) == 0:
            continue
        dqzr  = kv_bl["QZR_nonzero_mean"].mean() - kv_v["QZR_nonzero_mean"].mean()
        dcos  = kv_v["cosine_sim"].mean() - kv_bl["cosine_sim"].mean()
        print(f"  [{variant}] ΔQZR_nz={dqzr:+.4f}, Δcosine={dcos:+.4f}")

    # nm_thres feasibility
    kv_sweep_df = df_sweep[df_sweep["sublayer"].isin(["K", "V"])]
    if len(kv_sweep_df) > 0:
        feasible = kv_sweep_df[
            (kv_sweep_df["QZR_nonzero"] <= 0.20) &
            (kv_sweep_df["cosine_sim"]  >= 0.90) &
            (kv_sweep_df["CCR"]         <= 0.05)
        ]
        if len(feasible) == 0:
            print("  [nm_thres] 불충분: CCR<5% + QZR_nz<20% + cos>0.90 동시 불가")
        else:
            print(f"  [nm_thres] 충분: {len(feasible)} configs satisfy all 3 criteria")

    print(f"\n  Root cause: {verdict}")
    print(f"\nOutputs:")
    print(f"  {CSV_ROOTCAUSE}")
    print(f"  {CSV_SWEEP}")
    print(f"  {CSV_SOLUTIONS}")
    print(f"  {FIG_ROOTCAUSE}")
    print(f"  {FIG_FEASIBILITY}")
    print(f"  {FIG_SOLUTIONS}")
    print("\nDone.")


if __name__ == "__main__":
    main()
