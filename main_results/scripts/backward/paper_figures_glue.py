"""paper_figures_glue.py — GLUE backward-underflow diagnostic (Figures A/B/D/E/F)

Replicates the SQuAD backward-underflow diagnostic from paper_figures.py on
GLUE tasks with BERT-base, adds multi-seed robustness, and produces cross-task
aggregate figures (E, F).

- Figure A: QKVO+FFN Root Cause (2x3, 6 sublayers)          [per task/seed]
- Figure B: IO Resolution Sweep (bits=[4,6,8,10,12], 2x2)   [per task/seed]
- Figure D: Layerwise Mixed-Precision Validation (2x2)       [per task/seed]
- Figure E: Required bits heatmap (task x sublayer)           [aggregate]
- Figure F: Seed-variance error bars                          [aggregate]

Figure C (solutions: sto_round, nm_thres_cal, p99_clip) is deprecated —
Figure D (layerwise selective bit upgrade) is the final solution.

Key differences from paper_figures.py:
  - AutoModelForSequenceClassification instead of AutoModelForQuestionAnswering
  - GLUE data loading with DataCollatorWithPadding (dynamic padding)
  - Multi-seed support (seed passed to data shuffle AND torch.manual_seed)
  - always_digital = ["classifier", "pooler"] instead of ["qa_outputs", "pooler"]
  - Gradient flow re-enables "classifier" instead of "qa_outputs"
  - Per-(task,seed) output directories + aggregate cross-task figures

Usage:
  # Smoke test (fast)
  python paper_figures_glue.py --tasks rte --seeds 42 \\
    --figures ABD --n-step 5 --n-step-sweep 3 --batch-size 2 --run-tag smoke

  # Multi-seed smoke
  python paper_figures_glue.py --tasks rte,mrpc --seeds 42,43 \\
    --figures ABDEF --n-step 5 --n-step-sweep 3 --batch-size 2 --run-tag smoke2

  # Full single-task
  python paper_figures_glue.py --tasks rte --seeds 42,43,44 \\
    --figures ABDEF --run-tag glue_v1
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from torch.utils.data import DataLoader
from datasets import load_dataset

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Paper Figures A/B/D/E/F — AIMC BERT-base GLUE Diagnostic"
)
parser.add_argument("--tasks", type=str, default="cola,rte,mrpc,mnli,sst2,stsb,qqp,qnli",
                    help="Comma-separated GLUE tasks")
parser.add_argument("--seeds", type=str, default="42,43,44",
                    help="Comma-separated random seeds")
parser.add_argument("--bits", type=str, default="4,6,8,10,12",
                    help="Comma-separated bits for sweep")
parser.add_argument("--n-step", type=int, default=200,
                    help="Steps for Figure A runs (default: 200)")
parser.add_argument("--n-step-sweep", type=int, default=100,
                    help="Steps per bits config for Figure B (default: 100)")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--max-length", type=int, default=128,
                    help="Max sequence length for GLUE tokenization")
parser.add_argument("--out-dir", type=str, default="/data/main_results/results")
parser.add_argument("--run-tag", type=str, default="glue_v1")
parser.add_argument("--figures", type=str, default="ABDEF",
                    help="Which figures to generate: A,B,D,E,F or combo (default: ABDEF)")
parser.add_argument("--baseline-dac", type=int, default=7,
                    help="Baseline DAC bits (default: 7)")
parser.add_argument("--baseline-adc", type=int, default=9,
                    help="Baseline ADC bits (default: 9)")
# Figure D (layerwise) params
parser.add_argument("--layerwise-policy", type=str, default="hotspot",
                    choices=["hotspot", "sublayer", "layer"])
parser.add_argument("--layerwise-high-bits", type=int, default=10)
parser.add_argument("--layerwise-topk", type=int, default=12)
parser.add_argument("--layerwise-qzr-threshold", type=float, default=None)
parser.add_argument("--layerwise-sublayers", type=str, default="K,V,FFN1")
parser.add_argument("--layerwise-layers", type=str, default="")
args = parser.parse_args()

# =============================================================================
# Constants & Config
# =============================================================================

TASKS_LIST  = [t.strip() for t in args.tasks.split(",") if t.strip()]
SEEDS_LIST  = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
BITS_LIST   = [int(b.strip()) for b in args.bits.split(",") if b.strip()]
N_STEP      = args.n_step
N_STEP_SWEEP = args.n_step_sweep
BATCH_SIZE  = args.batch_size
MAX_LENGTH  = args.max_length
RUN_TAG     = args.run_tag
DAC_BITS    = args.baseline_dac
ADC_BITS    = args.baseline_adc

RUN_A = "A" in args.figures.upper()
RUN_B = "B" in args.figures.upper()
RUN_D = "D" in args.figures.upper()
RUN_E = "E" in args.figures.upper()
RUN_F = "F" in args.figures.upper()

OUT_DIR    = args.out_dir
INP_BOUND  = 1.0
N_LAYERS   = 12

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# GLUE Task Configuration
# =============================================================================

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp":  ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte":  ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}

TASK_TO_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
    "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1,
}

print(f"[Config] Device={DEVICE}, N_STEP={N_STEP}, N_STEP_SWEEP={N_STEP_SWEEP}, "
      f"BATCH={BATCH_SIZE}, MAX_LEN={MAX_LENGTH}, RUN_TAG={RUN_TAG}")
print(f"[Config] OUT_DIR={OUT_DIR}")
print(f"[Config] Tasks={TASKS_LIST}, Seeds={SEEDS_LIST}")
print(f"[Config] Figures={args.figures.upper()} "
      f"(A={RUN_A}, B={RUN_B}, D={RUN_D}, E={RUN_E}, F={RUN_F})")
print(f"[Config] Baseline DAC={DAC_BITS}b, ADC={ADC_BITS}b")

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
    always_digital = ["classifier", "pooler"]
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
# Hotspot Selection for Layerwise Mixed-Precision (Figure D)
# =============================================================================

def select_high_modules(model, stats_baseline, policy, cli_args):
    """Return set of module-name strings to upgrade to high bits."""
    target, nontarget, all_linear = _layer_names(model)
    modules_all = set(target + nontarget)
    name_to_pair = {}   # module_name -> (layer_idx, sublayer)
    for n in modules_all:
        p = parse_layer_name(n)
        if p:
            name_to_pair[n] = p
    pair_to_names = {}   # (layer_idx, sublayer) -> [module_names]
    for n, p in name_to_pair.items():
        pair_to_names.setdefault(p, []).append(n)

    if policy == "hotspot":
        rows = []
        for s in stats_baseline.values():
            rows.append({"layer_idx": s.layer_idx, "sublayer": s.sublayer,
                         "QZR_nonzero": np.mean(s.qzr_nz_steps) if s.qzr_nz_steps else 0})
        df = pd.DataFrame(rows).sort_values("QZR_nonzero", ascending=False)
        if cli_args.layerwise_qzr_threshold is not None:
            selected = df[df["QZR_nonzero"] > cli_args.layerwise_qzr_threshold]
        else:
            selected = df.head(cli_args.layerwise_topk)
        high_pairs = set(zip(selected["layer_idx"], selected["sublayer"]))

    elif policy == "sublayer":
        sub_list = [s.strip() for s in cli_args.layerwise_sublayers.split(",")]
        high_pairs = {(l, s) for (l, s) in pair_to_names.keys() if s in sub_list}

    elif policy == "layer":
        layer_list = [int(x) for x in cli_args.layerwise_layers.split(",") if x.strip()]
        high_pairs = {(l, s) for (l, s) in pair_to_names.keys() if l in layer_list}

    else:
        high_pairs = set()

    high_names = set()
    for pair in high_pairs:
        high_names.update(pair_to_names.get(pair, []))
    return high_names


# =============================================================================
# RPU Config (adc_bits parameterized)
# =============================================================================

def create_rpu_config(nm_thres=0.0, sto_round=False, dac_bits=None, adc_bits=None):
    """SingleRPU with noise-free SoftBoundsDevice. inp_res = 1/(2^dac_bits - 2).

    Matches optuna_bert_squad_tiki.py create_single_rpu_config() device config.
    """
    if dac_bits is None:
        dac_bits = DAC_BITS
    if adc_bits is None:
        adc_bits = ADC_BITS

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

def create_model(num_labels=2, nm_thres=0.0, sto_round=False,
                 dac_bits=None, adc_bits=None):
    """BERT-base SequenceClassification 2-pass analog conversion.

    Pass 1: Q/K/V/O (target) -- analog, weight updates enabled (lr=0 so noop)
    Pass 2: FFN (nontarget)  -- analog, tile.update = _noop (frozen)

    All AnalogContext.requires_grad = True -> backward flows through FFN tiles
    -> register_full_backward_hook fires on FFN layers too.
    """
    if dac_bits is None:
        dac_bits = DAC_BITS
    if adc_bits is None:
        adc_bits = ADC_BITS

    from aihwkit.nn import AnalogLinear
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.optim.context import AnalogContext

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    )
    target, nontarget, all_linear = _layer_names(model)

    # Pass 1: target (QKVO)
    rpu = create_rpu_config(nm_thres=nm_thres, sto_round=sto_round,
                            dac_bits=dac_bits, adc_bits=adc_bits)
    model = convert_to_analog(
        model, rpu,
        exclude_modules=[n for n in all_linear if n not in target]
    )

    # Pass 2: nontarget (FFN) -- same config, update will be nooped
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

    # Gradient flow: disable all, then re-enable AnalogContext + classifier
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        if isinstance(p, AnalogContext):
            p.requires_grad_(True)
    for n, p in model.named_parameters():
        if "classifier" in n:
            p.requires_grad_(True)

    n_t   = sum(1 for n, m in model.named_modules()
                if isinstance(m, AnalogLinear) and n in target)
    n_all = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog tiles -- target(QKVO):{n_t}, frozen(FFN):{n_all - n_t}, "
          f"nm_thres={nm_thres:.4g}, sto_round={sto_round}, "
          f"dac={dac_bits}b, adc={adc_bits}b, num_labels={num_labels}")
    return model.to(DEVICE)


# =============================================================================
# Layerwise Mixed-Precision Model Creation (Figure D)
# =============================================================================

def create_model_layerwise(high_names, base_dac_bits, base_adc_bits,
                           high_dac_bits, high_adc_bits,
                           num_labels=2,
                           nm_thres=0.0, sto_round=False):
    """Mixed-precision: high_names get high bits, rest get base bits.

    Returns: (model, dac_bits_map, adc_bits_map, inp_res_map)
    """
    from aihwkit.nn import AnalogLinear
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.optim.context import AnalogContext

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    )
    target, nontarget, all_linear = _layer_names(model)
    modules_all = set(target + nontarget)
    high_names = high_names & modules_all
    base_names = modules_all - high_names

    # Pass 1: high-precision modules
    rpu_high = create_rpu_config(nm_thres=nm_thres, sto_round=sto_round,
                                  dac_bits=high_dac_bits, adc_bits=high_adc_bits)
    if high_names:
        model = convert_to_analog(model, rpu_high,
            exclude_modules=[n for n in all_linear if n not in high_names])

    # Pass 2: base-precision modules (AnalogLinear from pass 1 won't re-convert)
    rpu_base = create_rpu_config(nm_thres=nm_thres, sto_round=sto_round,
                                  dac_bits=base_dac_bits, adc_bits=base_adc_bits)
    if base_names:
        model = convert_to_analog(model, rpu_base,
            exclude_modules=[n for n in all_linear if n not in base_names])

    # Freeze nontarget tile updates (same as create_model)
    def _noop(x, d, *a, **kw):
        return None

    for name, m in model.named_modules():
        if isinstance(m, AnalogLinear) and name not in target:
            for tile in m.analog_tiles():
                tile.update = _noop

    # Gradient flow (same as create_model)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        if isinstance(p, AnalogContext):
            p.requires_grad_(True)
    for n, p in model.named_parameters():
        if "classifier" in n:
            p.requires_grad_(True)

    # Build per-module maps
    dac_bits_map, adc_bits_map, inp_res_map = {}, {}, {}
    for name in modules_all:
        if name in high_names:
            dac_bits_map[name] = high_dac_bits
            adc_bits_map[name] = high_adc_bits
            inp_res_map[name] = 1.0 / (2**high_dac_bits - 2)
        else:
            dac_bits_map[name] = base_dac_bits
            adc_bits_map[name] = base_adc_bits
            inp_res_map[name] = 1.0 / (2**base_dac_bits - 2)

    n_high = sum(1 for n, m in model.named_modules()
                 if isinstance(m, AnalogLinear) and n in high_names)
    n_all = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Layerwise analog -- high:{n_high}({high_dac_bits}b), "
          f"base:{n_all-n_high}({base_dac_bits}b), num_labels={num_labels}")

    return model.to(DEVICE), dac_bits_map, adc_bits_map, inp_res_map


# =============================================================================
# GLUE Data Loading
# =============================================================================

def load_glue_data(task, tokenizer, n_step, batch_size, seed, max_length=128):
    """Load GLUE task data with dynamic padding.

    Returns a DataLoader with n_step batches.
    """
    assert task in TASK_TO_KEYS, f"Unknown GLUE task: {task}"
    key1, key2 = TASK_TO_KEYS[task]

    raw = load_dataset("nyu-mll/glue", task)
    # Use train split
    split = raw["train"]

    def preprocess(examples):
        if key2 is None:
            texts = tokenizer(
                examples[key1],
                max_length=max_length,
                truncation=True,
            )
        else:
            texts = tokenizer(
                examples[key1], examples[key2],
                max_length=max_length,
                truncation=True,
            )
        return texts

    tok = split.map(preprocess, batched=True,
                    remove_columns=[c for c in split.column_names if c != "label"])
    # Rename "label" -> "labels" for HuggingFace model forward
    tok = tok.rename_column("label", "labels")

    n_samples = min(n_step * batch_size, len(tok))
    subset = tok.shuffle(seed=seed).select(range(n_samples))

    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        collate_fn=collator)
    print(f"  Dataset({task}): {n_samples} samples -> {len(loader)} batches "
          f"(seed={seed}, max_len={max_length})")
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
# LayerStats -- AIHWKit-consistent quantization + extended metrics
# =============================================================================

class LayerStats:
    """Per-layer backward gradient statistics accumulator.

    zero_thresh = inp_res = 1/(2^b - 2)
    step_size = 2 * INP_BOUND * res_ratio  (AIHWKit UniformQuantize consistent)
    nm_thres > 0 -> alpha = min(absmax, nm_thres) simulating tile nm_thres cap.
    """

    def __init__(self, name: str, layer_idx: int, sublayer: str,
                 inp_res: float, nm_thres: float = 0.0,
                 store_sweep: bool = True):
        self.name        = name
        self.layer_idx   = layer_idx
        self.sublayer    = sublayer
        self.inp_res     = inp_res
        self.nm_thres    = nm_thres
        self.zero_thresh = inp_res   # threshold: ratio < zero_thresh -> quant 0
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
        self.p_clip_steps      = []
        self.absmax_q50_steps  = []
        self.absmax_q90_steps  = []
        self.absmax_q99_steps  = []
        self.absmax_q999_steps = []

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

        # p_clip: P(|d| > INP_BOUND) element-wise
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

    def summary(self, label: str = "baseline", dac_bits: int = None,
                adc_bits: int = None, figure_id: str = "",
                run_tag: str = "", sto_round: bool = False) -> dict:
        if dac_bits is None:
            dac_bits = DAC_BITS
        if adc_bits is None:
            adc_bits = ADC_BITS
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

    def step_records(self, label: str = "baseline", dac_bits: int = None,
                     adc_bits: int = None, figure_id: str = "",
                     run_tag: str = "", sto_round: bool = False) -> list:
        """Per-step records for steps CSV."""
        if dac_bits is None:
            dac_bits = DAC_BITS
        if adc_bits is None:
            adc_bits = ADC_BITS
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

def register_hooks(model, mask_buf: MaskBuffer, inp_res,
                   nm_thres: float = 0.0,
                   store_sweep: bool = True) -> tuple:
    """Register full backward hooks on all AnalogLinear matching the regex.

    Includes both QKVO (target) and FFN1/FFN2 (nontarget frozen) layers.
    inp_res: float (uniform) or dict[str, float] (per-module).
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
        module_inp_res = inp_res[name] if isinstance(inp_res, dict) else inp_res
        stats = LayerStats(name=name, layer_idx=layer_idx, sublayer=sublayer,
                           inp_res=module_inp_res, nm_thres=nm_thres,
                           store_sweep=store_sweep)
        stats_dict[name] = stats

        def make_hook(s, mb):
            def fn(mod, gin, gout):
                if gout[0] is not None:
                    s.update(gout[0], mask=mb.val)
            return fn

        handles.append(module.register_full_backward_hook(make_hook(stats, mask_buf)))

    sublayers_found = sorted(set(s.sublayer for s in stats_dict.values()))
    inp_res_str = f"dict({len(inp_res)} entries)" if isinstance(inp_res, dict) else f"{inp_res:.6f}"
    print(f"[Hook] {len(stats_dict)} hooks, sublayers={sublayers_found}, "
          f"inp_res={inp_res_str}, nm_thres={nm_thres:.4g}")
    return stats_dict, handles


# =============================================================================
# NPZ Save -- raw absmax for ECDF
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
    print(f"  absmax npz -> {filepath}  ({n_keys} layers, {total:,} values)")


# =============================================================================
# CDF CSV -- worst-N layer ratio CDF
# =============================================================================

def save_cdf_csv(stats_dict, df_summary, filepath, n_points=2000,
                 n_worst=3, worst_sublayer="K"):
    """Save CDF of |dy|/absmax ratio for worst-N layers by QZR_nonzero."""
    sl_df = df_summary[df_summary["sublayer"] == worst_sublayer].sort_values(
        "QZR_nonzero", ascending=False
    )
    worst = sl_df.head(n_worst)
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
                "sublayer": worst_sublayer,
                "layer_name": lname,
                "ratio": float(sorted_r[j]),
                "cdf": float(cdf[j]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"  CDF CSV -> {filepath}  ({len(df)} rows)")


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

def run_diagnostic(model, loader, n_step: int, seed: int, desc: str = "Diag"):
    """n_step forward+backward passes. lr=0 -> no weight change."""
    from aihwkit.optim import AnalogSGD

    optimizer = AnalogSGD(model.parameters(), lr=0.0)
    model.train()
    torch.manual_seed(seed)

    for step, batch in enumerate(tqdm(loader, total=n_step, desc=desc)):
        if step >= n_step:
            break
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            labels=batch["labels"].to(DEVICE),
        )
        outputs.loss.backward()
        optimizer.step()   # lr=0 -> no weight change; flushes tile grad buffers


# =============================================================================
# Root Cause Analytics (for Figure A)
# =============================================================================

def compute_rootcause(stats_dict: dict, dac_bits: int = None,
                      adc_bits: int = None, figure_id: str = "A",
                      run_tag: str = ""):
    """Run consistency checks and compute auto-diagnosis verdict."""
    if dac_bits is None:
        dac_bits = DAC_BITS
    if adc_bits is None:
        adc_bits = ADC_BITS
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
        verdict = f"mixed: EZR={ezr_kv:.2%}, QZR_nz={qzr_nz_kv:.2%} (both contribute)"
    elif structural:
        verdict = "structural exact-zero dominant (mask/attention sparsity)"
    elif bulk_tiny:
        verdict = "bulk tiny / outlier-dominant (scale issue)"
    else:
        verdict = (f"atypical: EZR={ezr_kv:.2%}, QZR_nz={qzr_nz_kv:.2%} "
                   f"ratio_q50={r50_kv:.6f}")
    print(f"  [Verdict] K/V: {verdict}")
    return df, verdict


# =============================================================================
# Figure A -- QKVO+FFN Root Cause (2x3, 6 sublayers)
# =============================================================================

def figure_A(df: pd.DataFrame, verdict: str, stats_dict: dict,
             dac_bits: int = None, n_step: int = None,
             task_name: str = "", seed: int = 0, output_path: str = ""):
    if dac_bits is None:
        dac_bits = DAC_BITS
    if n_step is None:
        n_step = N_STEP
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
        f"Figure A -- QKVO+FFN Backward Root Cause Diagnosis "
        f"(BERT-base, {task_name.upper()}, seed={seed}, "
        f"{n_step} steps x batch {BATCH_SIZE})\n"
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
            "(a) QZR_nonzero (mask-excluded)\n[0,1]: high = bulk-tiny/outlier-dom",
            "plasma", vmin=0.0, vmax=1.0, label="QZR_nonzero")

    # [0,1] EZR heatmap
    heatmap(axes[0, 1], ezr_mat,
            "(b) EZR -- Exact Zero Ratio (mask-excluded)\nhigh = structural sparsity",
            "YlOrRd", vmin=0.0, vmax=1.0, label="EZR")

    # [0,2] l2_retention heatmap
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
                 "ratio < zero_thresh -> quantized to zero", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # [1,1] CDF: worst 3 K layers by QZR_nonzero
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
    ax.set_title("(e) CDF of |dy|/absmax -- worst 3 K layers\n"
                 "left of threshold -> quantized to zero", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # [1,2] Text panel: auto-diagnosis summary
    ax = axes[1, 2]
    ax.axis("off")
    kv  = df[df["sublayer"].isin(["K", "V"])]
    ffn = df[df["sublayer"].isin(["FFN1", "FFN2"])]
    lines = [
        f"Auto-Diagnosis: {task_name.upper()} seed={seed}",
        "=" * 38,
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
    lines.append("Verdict (K/V):")
    # word-wrap verdict
    words, line, wrapped = verdict.split(" "), "", []
    for w in words:
        if len(line) + len(w) + 1 > 32:
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
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure A -> {output_path}")


# =============================================================================
# Figure B -- IO Resolution Sweep (2x2)
# =============================================================================

def figure_B(df_b: pd.DataFrame, task_name: str = "", seed: int = 0,
             output_path: str = ""):
    bits_list_all = sorted(df_b["dac_bits"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Figure B -- IO Resolution Sweep (BERT-base, {task_name.upper()}, "
        f"seed={seed}, bits={bits_list_all})\n"
        f"sweep: dac_bits=adc_bits=bits, {N_STEP_SWEEP} steps; "
        f"baseline: dac={DAC_BITS}b, adc={ADC_BITS}b, {N_STEP} steps",
        fontsize=10, y=1.01
    )

    sublayer_colors = {
        "Q": "#4C72B0", "K": "#DD8452", "V": "#55A868",
        "O": "#C44E52", "FFN1": "#9467BD", "FFN2": "#8C564B"
    }
    sublayer_markers = {
        "Q": "o", "K": "s", "V": "^", "O": "D", "FFN1": "p", "FFN2": "h"
    }

    # [0,0] Line: bits x sublayer -> QZR_nonzero
    ax = axes[0, 0]
    for sl in SUBLAYER_ORDER:
        sub = df_b[df_b["sublayer"] == sl].groupby("dac_bits")["QZR_nonzero"].mean()
        if len(sub) == 0:
            continue
        ax.plot(sub.index, sub.values,
                color=sublayer_colors[sl], marker=sublayer_markers[sl],
                label=sl, lw=1.5, markersize=6)
    if DAC_BITS in bits_list_all:
        ax.axvline(DAC_BITS, color="gray", ls=":", lw=1.2, label=f"baseline {DAC_BITS}b")
    ax.set_xlabel("bits (dac_bits)"); ax.set_ylabel("QZR_nonzero")
    ax.set_title("(a) QZR_nonzero vs bits per sublayer\n(lower = better)", fontsize=9)
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    ax.set_xticks(bits_list_all)

    # [0,1] Line: bits x sublayer -> cosine_sim
    ax = axes[0, 1]
    for sl in SUBLAYER_ORDER:
        sub = df_b[df_b["sublayer"] == sl].groupby("dac_bits")["cosine_sim"].mean()
        if len(sub) == 0:
            continue
        ax.plot(sub.index, sub.values,
                color=sublayer_colors[sl], marker=sublayer_markers[sl],
                label=sl, lw=1.5, markersize=6)
    if DAC_BITS in bits_list_all:
        ax.axvline(DAC_BITS, color="gray", ls=":", lw=1.2, label=f"baseline {DAC_BITS}b")
    ax.set_xlabel("bits (dac_bits)"); ax.set_ylabel("cosine_sim")
    ax.set_title("(b) cosine_sim vs bits per sublayer\n(higher = better)", fontsize=9)
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    ax.set_xticks(bits_list_all)

    # [1,0] Heatmap: bits(y) x layer(x) for K QZR_nonzero
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
    ax.set_title("(c) K QZR_nonzero: bits x layer", fontsize=9)
    plt.colorbar(im, ax=ax, label="QZR_nonzero", shrink=0.85)

    # [1,1] Heatmap: bits(y) x layer(x) for V QZR_nonzero
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
    ax.set_title("(d) V QZR_nonzero: bits x layer", fontsize=9)
    plt.colorbar(im, ax=ax, label="QZR_nonzero", shrink=0.85)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure B -> {output_path}")


# =============================================================================
# Figure D -- Layerwise Mixed-Precision Validation (2x2)
# =============================================================================

def figure_D(df_d: pd.DataFrame, high_names: set,
             task_name: str = "", seed: int = 0, output_path: str = "",
             high_bits: int = 10):
    """2x2 figure: (a,b) delta-QZR heatmaps, (c,d) grouped bars by sublayer."""
    VARIANTS_D = ["baseline", "full_high", "layerwise"]
    COLORS_D   = ["#4C72B0", "#DD8452", "#55A868"]

    # Build per-variant heatmap matrices (12 layers x 6 sublayers)
    def to_mat(variant, col):
        sub = df_d[df_d["variant"] == variant]
        mat = np.full((N_LAYERS, len(SUBLAYER_ORDER)), np.nan)
        for _, row in sub.iterrows():
            li = int(row["layer_idx"])
            sl = row["sublayer"]
            if sl in SUBLAYER_ORDER and li < N_LAYERS:
                mat[li, SUBLAYER_ORDER.index(sl)] = row[col]
        return mat

    baseline_qzr = to_mat("baseline", "QZR_nonzero")
    full_high_qzr = to_mat("full_high", "QZR_nonzero")
    layerwise_qzr = to_mat("layerwise", "QZR_nonzero")

    delta_lw = layerwise_qzr - baseline_qzr
    delta_fh = full_high_qzr - baseline_qzr

    # Build high_names mask for border markers
    high_mask = np.zeros((N_LAYERS, len(SUBLAYER_ORDER)), dtype=bool)
    for name in high_names:
        p = parse_layer_name(name)
        if p:
            li, sl = p
            if sl in SUBLAYER_ORDER and li < N_LAYERS:
                high_mask[li, SUBLAYER_ORDER.index(sl)] = True

    total_tracked = N_LAYERS * len(SUBLAYER_ORDER)
    n_upgraded = int(high_mask.sum())

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f"Figure D -- Layerwise Mixed-Precision Validation "
        f"(BERT-base, {task_name.upper()}, seed={seed}, "
        f"{N_STEP} steps x batch {BATCH_SIZE})\n"
        f"Budget: {n_upgraded}/{total_tracked} modules upgraded  |  "
        f"baseline={DAC_BITS}b, high={high_bits}b",
        fontsize=10, y=1.01
    )

    vabs = max(np.nanmax(np.abs(delta_lw)), np.nanmax(np.abs(delta_fh)), 0.01)

    # (a) Heatmap: delta-QZR (layerwise - baseline)
    ax = axes[0, 0]
    im = ax.imshow(delta_lw, aspect="auto", cmap="RdBu_r", origin="upper",
                   vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER, fontsize=8)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer")
    ax.set_title("(a) ΔQZR_nonzero (layerwise - baseline)\n"
                 "blue = improved, red = degraded", fontsize=9)
    plt.colorbar(im, ax=ax, label="ΔQZR_nonzero", shrink=0.85)
    # Mark upgraded modules with rectangles
    for li in range(N_LAYERS):
        for si in range(len(SUBLAYER_ORDER)):
            if high_mask[li, si]:
                rect = plt.Rectangle((si - 0.5, li - 0.5), 1, 1,
                                     linewidth=2, edgecolor="lime",
                                     facecolor="none")
                ax.add_patch(rect)

    # (b) Heatmap: delta-QZR (full_high - baseline)
    ax = axes[0, 1]
    im = ax.imshow(delta_fh, aspect="auto", cmap="RdBu_r", origin="upper",
                   vmin=-vabs, vmax=vabs)
    ax.set_xticks(range(len(SUBLAYER_ORDER)))
    ax.set_xticklabels(SUBLAYER_ORDER, fontsize=8)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer")
    ax.set_title("(b) ΔQZR_nonzero (full_high - baseline)\n"
                 "blue = improved, red = degraded", fontsize=9)
    plt.colorbar(im, ax=ax, label="ΔQZR_nonzero", shrink=0.85)

    # (c) Bar: mean QZR_nonzero per sublayer for 3 variants
    ax = axes[1, 0]
    x = np.arange(len(SUBLAYER_ORDER))
    n_v = len(VARIANTS_D)
    width = 0.22
    offsets = np.linspace(-(n_v - 1) * width / 2, (n_v - 1) * width / 2, n_v)
    for vi, (variant, color) in enumerate(zip(VARIANTS_D, COLORS_D)):
        var_df = df_d[df_d["variant"] == variant]
        vals = []
        for sl in SUBLAYER_ORDER:
            sub = var_df[var_df["sublayer"] == sl]["QZR_nonzero"]
            vals.append(float(sub.mean()) if len(sub) > 0 else float("nan"))
        bars = ax.bar(x + offsets[vi], vals, width=width, label=variant,
                      color=color, alpha=0.8, edgecolor="k", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=6, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(SUBLAYER_ORDER)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("QZR_nonzero")
    ax.set_title("(c) Mean QZR_nonzero per sublayer (3 variants)\n"
                 "(lower = better)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # (d) Bar: mean rel_l2_error per sublayer for 3 variants
    ax = axes[1, 1]
    for vi, (variant, color) in enumerate(zip(VARIANTS_D, COLORS_D)):
        var_df = df_d[df_d["variant"] == variant]
        vals = []
        for sl in SUBLAYER_ORDER:
            sub = var_df[var_df["sublayer"] == sl]["rel_l2_error"]
            vals.append(float(sub.mean()) if len(sub) > 0 else float("nan"))
        bars = ax.bar(x + offsets[vi], vals, width=width, label=variant,
                      color=color, alpha=0.8, edgecolor="k", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom",
                        fontsize=6, rotation=45)
    ax.set_xticks(x); ax.set_xticklabels(SUBLAYER_ORDER)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("rel_l2_error")
    ax.set_title("(d) Mean rel_l2_error per sublayer (3 variants)\n"
                 "(lower = better)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure D -> {output_path}")


# =============================================================================
# Figure E -- Required Bits Heatmap (task x sublayer) [aggregate]
# =============================================================================

def figure_E(agg_df: pd.DataFrame, output_path: str, qzr_threshold: float = 0.20):
    """Heatmap: tasks (y) x sublayers (x).
    Cell value = minimum dac_bits where mean QZR_nonzero < threshold (avg across seeds).
    Annotate cells with bit values; ">12" sentinel for never-reached.
    """
    # Filter to sweep data (figure_id == "B")
    sweep_df = agg_df[agg_df["figure_id"] == "B"].copy()
    if len(sweep_df) == 0:
        print(f"  [WARN] No Figure B data for Figure E, skipping")
        return

    tasks = sorted(sweep_df["task"].unique())
    sublayers = SUBLAYER_ORDER

    # For each (task, sublayer): find min bits where mean QZR_nz < threshold
    mat = np.full((len(tasks), len(sublayers)), np.nan)
    annot = [[" " for _ in sublayers] for _ in tasks]

    for ti, task in enumerate(tasks):
        for si, sl in enumerate(sublayers):
            sub = sweep_df[(sweep_df["task"] == task) & (sweep_df["sublayer"] == sl)]
            if len(sub) == 0:
                annot[ti][si] = "?"
                continue
            # Average across seeds, then find min bits
            mean_by_bits = sub.groupby("dac_bits")["QZR_nonzero"].mean()
            found = mean_by_bits[mean_by_bits < qzr_threshold]
            if len(found) > 0:
                min_bits = int(found.index.min())
                mat[ti, si] = min_bits
                annot[ti][si] = str(min_bits)
            else:
                mat[ti, si] = 14  # sentinel for "never reached"
                annot[ti][si] = ">12"

    fig, ax = plt.subplots(figsize=(10, max(4, len(tasks) * 0.8 + 2)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", origin="upper",
                   vmin=4, vmax=14)
    ax.set_xticks(range(len(sublayers)))
    ax.set_xticklabels(sublayers, fontsize=9)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t.upper() for t in tasks], fontsize=9)
    ax.set_xlabel("Sublayer"); ax.set_ylabel("GLUE Task")

    # Annotate cells
    for ti in range(len(tasks)):
        for si in range(len(sublayers)):
            ax.text(si, ti, annot[ti][si], ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if (mat[ti, si] > 10 or np.isnan(mat[ti, si])) else "black")

    ax.set_title(
        f"Figure E -- Minimum DAC bits for QZR_nonzero < {qzr_threshold:.0%}\n"
        f"(task x sublayer, averaged across seeds)",
        fontsize=11
    )
    plt.colorbar(im, ax=ax, label="Min bits", shrink=0.85)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure E -> {output_path}")


# =============================================================================
# Figure F -- Seed Variance Error Bars [aggregate]
# =============================================================================

def figure_F(agg_df: pd.DataFrame, output_path: str):
    """1x2 subplot:
    (a) Worst-layer QZR per task (max over sublayers/layers), error bars = seed std
    (b) Per-sublayer grouped bar x task, error bars = seed std
    """
    # Filter to baseline data (figure_id == "A")
    base_df = agg_df[agg_df["figure_id"] == "A"].copy()
    if len(base_df) == 0:
        print(f"  [WARN] No Figure A data for Figure F, skipping")
        return

    tasks = sorted(base_df["task"].unique())
    sublayers = SUBLAYER_ORDER

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"Figure F -- Seed Variance Analysis (GLUE, baseline {DAC_BITS}b)",
        fontsize=11, y=1.02
    )

    # (a) Worst-layer QZR per task
    ax = axes[0]
    task_means, task_stds = [], []
    for task in tasks:
        # For each seed: find max QZR_nonzero across all sublayers/layers
        seed_maxes = []
        for seed in base_df[base_df["task"] == task]["seed"].unique():
            sub = base_df[(base_df["task"] == task) & (base_df["seed"] == seed)]
            seed_maxes.append(sub["QZR_nonzero"].max())
        task_means.append(np.mean(seed_maxes))
        task_stds.append(np.std(seed_maxes) if len(seed_maxes) > 1 else 0.0)

    x = np.arange(len(tasks))
    bars = ax.bar(x, task_means, yerr=task_stds, capsize=5,
                  color="#4C72B0", alpha=0.8, edgecolor="k", linewidth=0.5,
                  error_kw=dict(lw=1.5))
    for bar, mean, std in zip(bars, task_means, task_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks], fontsize=9)
    ax.set_xlabel("GLUE Task"); ax.set_ylabel("Worst-layer QZR_nonzero")
    ax.set_title("(a) Worst-layer QZR_nonzero per task\n"
                 "(max over sublayers/layers, error = seed std)", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # (b) Per-sublayer grouped bar x task
    ax = axes[1]
    sublayer_colors = {
        "Q": "#4C72B0", "K": "#DD8452", "V": "#55A868",
        "O": "#C44E52", "FFN1": "#9467BD", "FFN2": "#8C564B"
    }
    n_sl = len(sublayers)
    width = 0.12
    offsets = np.linspace(-(n_sl - 1) * width / 2, (n_sl - 1) * width / 2, n_sl)
    for si, sl in enumerate(sublayers):
        means, stds = [], []
        for task in tasks:
            seed_vals = []
            for seed_val in base_df[base_df["task"] == task]["seed"].unique():
                sub = base_df[(base_df["task"] == task) &
                              (base_df["seed"] == seed_val) &
                              (base_df["sublayer"] == sl)]
                if len(sub) > 0:
                    seed_vals.append(sub["QZR_nonzero"].mean())
            means.append(np.mean(seed_vals) if seed_vals else 0.0)
            stds.append(np.std(seed_vals) if len(seed_vals) > 1 else 0.0)
        ax.bar(x + offsets[si], means, width=width, yerr=stds, capsize=2,
               label=sl, color=sublayer_colors[sl], alpha=0.8,
               edgecolor="k", linewidth=0.3,
               error_kw=dict(lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in tasks], fontsize=9)
    ax.set_xlabel("GLUE Task"); ax.set_ylabel("Mean QZR_nonzero")
    ax.set_title("(b) Mean QZR_nonzero per sublayer x task\n"
                 "(error = seed std)", fontsize=9)
    ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure F -> {output_path}")


# =============================================================================
# Per-(task, seed) Runner
# =============================================================================

def run_single_task_seed(task, seed, num_labels, tokenizer, seed_dir,
                         cli_args, run_figures):
    """Run full A->B->D pipeline for one (task, seed).

    Returns list of summary rows (dicts) for aggregate CSV.
    """
    print(f"\n{'='*60}")
    print(f"  Task={task.upper()}, Seed={seed}, num_labels={num_labels}")
    print(f"  Output: {seed_dir}")
    print(f"{'='*60}")

    os.makedirs(seed_dir, exist_ok=True)
    agg_rows = []

    # CSV/figure paths
    csv_a_summary = os.path.join(seed_dir, "metrics_A_rootcause_summary.csv")
    csv_a_steps   = os.path.join(seed_dir, "metrics_A_rootcause_steps.csv")
    csv_a_cdf     = os.path.join(seed_dir, "metrics_A_rootcause_cdf.csv")
    csv_b_summary = os.path.join(seed_dir, "metrics_B_bitsweep_summary.csv")
    csv_b_steps   = os.path.join(seed_dir, "metrics_B_bitsweep_steps.csv")
    csv_d_summary = os.path.join(seed_dir, "metrics_D_layerwise_summary.csv")
    csv_d_steps   = os.path.join(seed_dir, "metrics_D_layerwise_steps.csv")
    fig_a_path    = os.path.join(seed_dir, "fig_A_rootcause.png")
    fig_b_path    = os.path.join(seed_dir, "fig_B_bitsweep.png")
    fig_d_path    = os.path.join(seed_dir, "fig_D_layerwise.png")

    # ------------------------------------------------------------------ #
    # [1] Data loading                                                    #
    # ------------------------------------------------------------------ #
    n_step_max = max(N_STEP, N_STEP_SWEEP)
    loader = load_glue_data(task, tokenizer, n_step=n_step_max,
                            batch_size=BATCH_SIZE, seed=seed,
                            max_length=MAX_LENGTH)

    # ------------------------------------------------------------------ #
    # [2] Baseline run                                                    #
    # ------------------------------------------------------------------ #
    print(f"\n  [{task}/{seed}] Baseline {DAC_BITS}-bit, {N_STEP} steps ...")
    inp_res_base = 1.0 / (2**DAC_BITS - 2)
    mask_buf_a = MaskBuffer()
    model_a = create_model(num_labels=num_labels, nm_thres=0.0, sto_round=False,
                           dac_bits=DAC_BITS, adc_bits=ADC_BITS)
    mask_buf_a.register(model_a)
    stats_baseline, handles_a = register_hooks(model_a, mask_buf_a,
                                               inp_res=inp_res_base)
    run_diagnostic(model_a, loader, n_step=N_STEP, seed=seed,
                   desc=f"A-{task}-s{seed}")
    for h in handles_a:
        h.remove()
    del model_a
    torch.cuda.empty_cache()
    gc.collect()

    # Save raw absmax for baseline
    save_absmax_npz(stats_baseline,
                    os.path.join(seed_dir, "absmax_raw_A_baseline.npz"))

    # ------------------------------------------------------------------ #
    # [3] Figure A: root cause                                           #
    # ------------------------------------------------------------------ #
    if "A" in run_figures:
        print(f"\n  [{task}/{seed}] Computing root cause -> Figure A ...")
        df_a, verdict_a = compute_rootcause(stats_baseline, dac_bits=DAC_BITS,
                                            adc_bits=ADC_BITS, figure_id="A",
                                            run_tag=RUN_TAG)
        df_a["task"] = task
        df_a["seed"] = seed
        df_a.to_csv(csv_a_summary, index=False)
        print(f"  -> {csv_a_summary}")

        # Steps CSV
        all_a_step_records = []
        for s in stats_baseline.values():
            all_a_step_records.extend(s.step_records(
                "baseline", dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                figure_id="A", run_tag=RUN_TAG))
        df_a_steps = pd.DataFrame(all_a_step_records)
        df_a_steps["task"] = task
        df_a_steps["seed"] = seed
        df_a_steps.to_csv(csv_a_steps, index=False)
        print(f"  -> {csv_a_steps} ({len(all_a_step_records)} rows)")

        # CDF CSV
        save_cdf_csv(stats_baseline, df_a, csv_a_cdf, worst_sublayer="K")

        figure_A(df_a, verdict_a, stats_baseline, dac_bits=DAC_BITS, n_step=N_STEP,
                 task_name=task, seed=seed, output_path=fig_a_path)

        # Collect for aggregate
        for _, row in df_a.iterrows():
            agg_rows.append(row.to_dict())

    # ------------------------------------------------------------------ #
    # [4] Figure B: bits sweep                                           #
    # ------------------------------------------------------------------ #
    if "B" in run_figures:
        print(f"\n  [{task}/{seed}] Figure B: bits sweep {BITS_LIST}, "
              f"{N_STEP_SWEEP} steps each ...")
        all_b_rows = []
        all_b_step_records = []

        # baseline contribution
        for s in stats_baseline.values():
            all_b_rows.append(s.summary("baseline", dac_bits=DAC_BITS,
                                        adc_bits=ADC_BITS, figure_id="B",
                                        run_tag=RUN_TAG))
            all_b_step_records.extend(s.step_records(
                "baseline", dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                figure_id="B", run_tag=RUN_TAG))

        for bits in BITS_LIST:
            print(f"\n    B-sweep bits={bits} (dac_bits=adc_bits={bits}) ...")
            inp_res_b  = 1.0 / (2**bits - 2)
            mask_buf_b = MaskBuffer()
            model_b    = create_model(num_labels=num_labels, nm_thres=0.0,
                                      sto_round=False,
                                      dac_bits=bits, adc_bits=bits)
            mask_buf_b.register(model_b)
            stats_b, handles_b = register_hooks(model_b, mask_buf_b,
                                                 inp_res=inp_res_b,
                                                 store_sweep=False)
            run_diagnostic(model_b, loader, n_step=N_STEP_SWEEP, seed=seed,
                           desc=f"B-{bits}b-{task}-s{seed}")
            for h in handles_b:
                h.remove()
            del model_b
            torch.cuda.empty_cache()
            gc.collect()

            save_absmax_npz(stats_b,
                            os.path.join(seed_dir, f"absmax_raw_B_sweep_{bits}b.npz"))

            for s in stats_b.values():
                all_b_rows.append(s.summary("sweep", dac_bits=bits, adc_bits=bits,
                                            figure_id="B", run_tag=RUN_TAG))
                all_b_step_records.extend(s.step_records(
                    "sweep", dac_bits=bits, adc_bits=bits,
                    figure_id="B", run_tag=RUN_TAG))

        df_b = (pd.DataFrame(all_b_rows)
                  .sort_values(["dac_bits", "layer_idx", "sublayer"])
                  .reset_index(drop=True))
        df_b["task"] = task
        df_b["seed"] = seed
        df_b.to_csv(csv_b_summary, index=False)
        print(f"  -> {csv_b_summary}")

        df_b_steps = pd.DataFrame(all_b_step_records)
        df_b_steps["task"] = task
        df_b_steps["seed"] = seed
        df_b_steps.to_csv(csv_b_steps, index=False)
        print(f"  -> {csv_b_steps} ({len(all_b_step_records)} rows)")

        figure_B(df_b, task_name=task, seed=seed, output_path=fig_b_path)

        # Collect for aggregate
        for _, row in df_b.iterrows():
            agg_rows.append(row.to_dict())

    # ------------------------------------------------------------------ #
    # [5] Figure D: layerwise mixed-precision                            #
    # ------------------------------------------------------------------ #
    if "D" in run_figures:
        high_dac = cli_args.layerwise_high_bits
        high_adc = cli_args.layerwise_high_bits  # same as dac for simplicity

        print(f"\n  [{task}/{seed}] Figure D: layerwise mixed-precision "
              f"(policy={cli_args.layerwise_policy}, high={high_dac}b) ...")

        # Select hotspot modules from baseline stats
        tmp_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels)
        high_names = select_high_modules(tmp_model, stats_baseline,
                                          cli_args.layerwise_policy, cli_args)
        del tmp_model
        print(f"  Selected {len(high_names)} modules for high-precision upgrade")

        all_d_rows, all_d_step_records = [], []

        # Variant 1: baseline (reuse stats_baseline)
        for s in stats_baseline.values():
            all_d_rows.append(s.summary("baseline", dac_bits=DAC_BITS,
                                        adc_bits=ADC_BITS, figure_id="D",
                                        run_tag=RUN_TAG))
            all_d_step_records.extend(s.step_records(
                "baseline", dac_bits=DAC_BITS, adc_bits=ADC_BITS,
                figure_id="D", run_tag=RUN_TAG))

        # Variant 2: full_high -- all modules at high bits
        print(f"    D-variant: full_high (dac={high_dac}b, adc={high_adc}b) ...")
        inp_res_fh = 1.0 / (2**high_dac - 2)
        mask_buf_fh = MaskBuffer()
        model_fh = create_model(num_labels=num_labels,
                                dac_bits=high_dac, adc_bits=high_adc)
        mask_buf_fh.register(model_fh)
        stats_fh, handles_fh = register_hooks(model_fh, mask_buf_fh,
                                               inp_res=inp_res_fh,
                                               store_sweep=False)
        run_diagnostic(model_fh, loader, n_step=N_STEP, seed=seed,
                       desc=f"D-full_high-{task}-s{seed}")
        for h in handles_fh:
            h.remove()
        del model_fh
        torch.cuda.empty_cache()
        gc.collect()
        save_absmax_npz(stats_fh,
                        os.path.join(seed_dir, "absmax_raw_D_full_high.npz"))
        for s in stats_fh.values():
            all_d_rows.append(s.summary("full_high", dac_bits=high_dac,
                                        adc_bits=high_adc, figure_id="D",
                                        run_tag=RUN_TAG))
            all_d_step_records.extend(s.step_records(
                "full_high", dac_bits=high_dac, adc_bits=high_adc,
                figure_id="D", run_tag=RUN_TAG))

        # Variant 3: layerwise -- selective modules at high bits
        print(f"    D-variant: layerwise ({len(high_names)} modules upgraded) ...")
        model_lw, dac_map, adc_map, inp_res_map = create_model_layerwise(
            high_names, DAC_BITS, ADC_BITS, high_dac, high_adc,
            num_labels=num_labels)
        mask_buf_lw = MaskBuffer()
        mask_buf_lw.register(model_lw)
        stats_lw, handles_lw = register_hooks(model_lw, mask_buf_lw,
                                               inp_res=inp_res_map,
                                               store_sweep=False)
        run_diagnostic(model_lw, loader, n_step=N_STEP, seed=seed,
                       desc=f"D-layerwise-{task}-s{seed}")
        for h in handles_lw:
            h.remove()
        del model_lw
        torch.cuda.empty_cache()
        gc.collect()
        save_absmax_npz(stats_lw,
                        os.path.join(seed_dir, "absmax_raw_D_layerwise.npz"))
        for s in stats_lw.values():
            all_d_rows.append(s.summary("layerwise",
                dac_bits=dac_map.get(s.name, DAC_BITS),
                adc_bits=adc_map.get(s.name, ADC_BITS),
                figure_id="D", run_tag=RUN_TAG))
            all_d_step_records.extend(s.step_records(
                "layerwise",
                dac_bits=dac_map.get(s.name, DAC_BITS),
                adc_bits=adc_map.get(s.name, ADC_BITS),
                figure_id="D", run_tag=RUN_TAG))

        # Save D CSVs
        df_d = (pd.DataFrame(all_d_rows)
                  .sort_values(["variant", "layer_idx", "sublayer"])
                  .reset_index(drop=True))
        df_d["task"] = task
        df_d["seed"] = seed
        df_d.to_csv(csv_d_summary, index=False)
        print(f"  -> {csv_d_summary} ({len(df_d)} rows)")

        df_d_steps = pd.DataFrame(all_d_step_records)
        df_d_steps["task"] = task
        df_d_steps["seed"] = seed
        df_d_steps.to_csv(csv_d_steps, index=False)
        print(f"  -> {csv_d_steps} ({len(df_d_steps)} rows)")

        # Figure D
        figure_D(df_d, high_names, task_name=task, seed=seed,
                 output_path=fig_d_path, high_bits=high_dac)

        # Collect for aggregate
        for _, row in df_d.iterrows():
            agg_rows.append(row.to_dict())

        del stats_fh, stats_lw

    # ------------------------------------------------------------------ #
    # Validation                                                         #
    # ------------------------------------------------------------------ #
    COMMON_COLS = ["figure_id", "run_tag", "variant", "dac_bits", "adc_bits",
                   "inp_bound", "inp_res", "res_ratio", "step_size", "nm_thres",
                   "layer_idx", "sublayer"]
    METRIC_COLS = ["EZR", "QZR_all", "QZR_nonzero", "ODR", "cosine_sim",
                   "l2_retention", "rel_l2_error", "clip_rate_scaled",
                   "ratio_q50", "ratio_q90", "ratio_q99",
                   "absmax_q50", "absmax_q90", "absmax_q99", "absmax_q999"]
    STEP_EXTRA  = ["step_idx", "n_vec"]

    if "A" in run_figures:
        validate_csv(csv_a_summary, COMMON_COLS + METRIC_COLS, min_rows=72,
                     critical_columns=["QZR_nonzero", "l2_retention"])
        validate_csv(csv_a_steps, COMMON_COLS + STEP_EXTRA + METRIC_COLS,
                     min_rows=72 * min(N_STEP, len(loader)),
                     critical_columns=["QZR_nonzero"])
        validate_csv(csv_a_cdf, ["layer_idx", "ratio", "cdf"], min_rows=10)
    if "B" in run_figures:
        validate_csv(csv_b_summary, COMMON_COLS + METRIC_COLS, min_rows=72,
                     critical_columns=["QZR_nonzero", "l2_retention"])
    if "D" in run_figures:
        validate_csv(csv_d_summary, COMMON_COLS + METRIC_COLS, min_rows=72 * 3,
                     critical_columns=["QZR_nonzero", "l2_retention"])
        # Sanity: layerwise variant must have both base and high dac_bits
        df_d_check = pd.read_csv(csv_d_summary)
        lw = df_d_check[df_d_check["variant"] == "layerwise"]
        assert lw["dac_bits"].nunique() >= 2, \
            "[CSV FAIL] layerwise variant should have mixed dac_bits"
        print(f"  [CSV OK] layerwise dac_bits values: {sorted(lw['dac_bits'].unique())}")

    # Cleanup
    del stats_baseline
    gc.collect()
    torch.cuda.empty_cache()

    return agg_rows


# =============================================================================
# Aggregate Cross-Task
# =============================================================================

def aggregate_cross_task(tasks, seeds, out_dir, run_figures):
    """Read per-(task,seed) summary CSVs, concatenate, write aggregate CSV,
    call figure_E/F.
    """
    agg_dir = os.path.join(out_dir, "glue", "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    all_dfs = []
    for task in tasks:
        for seed in seeds:
            seed_dir = os.path.join(out_dir, "glue", task, f"seed_{seed}")
            for csv_name in ["metrics_A_rootcause_summary.csv",
                             "metrics_B_bitsweep_summary.csv",
                             "metrics_D_layerwise_summary.csv"]:
                csv_path = os.path.join(seed_dir, csv_name)
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    if "task" not in df.columns:
                        df["task"] = task
                    if "seed" not in df.columns:
                        df["seed"] = seed
                    all_dfs.append(df)

    if not all_dfs:
        print("[WARN] No CSVs found for aggregation")
        return

    agg_df = pd.concat(all_dfs, ignore_index=True)
    agg_csv = os.path.join(agg_dir, "metrics_glue_task_summary.csv")
    agg_df.to_csv(agg_csv, index=False)
    print(f"\n[Aggregate] -> {agg_csv} ({len(agg_df)} rows)")

    # Figure E
    if "E" in run_figures:
        fig_e_path = os.path.join(agg_dir, "fig_E_required_bits_heatmap.png")
        figure_E(agg_df, fig_e_path)

    # Figure F
    if "F" in run_figures:
        fig_f_path = os.path.join(agg_dir, "fig_F_seed_variance.png")
        figure_F(agg_df, fig_f_path)

    return agg_df


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Determine which per-task figures to run
    run_figures = args.figures.upper()

    # Load tokenizer once
    print("\n[0] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    all_agg_rows = []

    for task in TASKS_LIST:
        num_labels = TASK_TO_NUM_LABELS.get(task, 2)
        for seed in SEEDS_LIST:
            set_seed(seed)
            seed_dir = os.path.join(OUT_DIR, "glue", task, f"seed_{seed}")

            summary_rows = run_single_task_seed(
                task=task,
                seed=seed,
                num_labels=num_labels,
                tokenizer=tokenizer,
                seed_dir=seed_dir,
                cli_args=args,
                run_figures=run_figures,
            )
            all_agg_rows.extend(summary_rows)

            # Aggressive cleanup between (task, seed)
            gc.collect()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Aggregate cross-task figures (E, F)                                #
    # ------------------------------------------------------------------ #
    if "E" in run_figures or "F" in run_figures:
        print(f"\n{'='*60}")
        print(f"  Aggregate Cross-Task Figures")
        print(f"{'='*60}")
        aggregate_cross_task(TASKS_LIST, SEEDS_LIST, OUT_DIR, run_figures)

    # ------------------------------------------------------------------ #
    # Final summary                                                      #
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  DONE — GLUE Backward-Underflow Diagnostic")
    print(f"{'='*60}")
    print(f"  Tasks:   {TASKS_LIST}")
    print(f"  Seeds:   {SEEDS_LIST}")
    print(f"  Figures: {run_figures}")
    print(f"  Output:  {OUT_DIR}/glue/")

    # List output files
    for task in TASKS_LIST:
        for seed in SEEDS_LIST:
            seed_dir = os.path.join(OUT_DIR, "glue", task, f"seed_{seed}")
            if os.path.isdir(seed_dir):
                files = sorted(os.listdir(seed_dir))
                n_csv = sum(1 for f in files if f.endswith(".csv"))
                n_png = sum(1 for f in files if f.endswith(".png"))
                n_npz = sum(1 for f in files if f.endswith(".npz"))
                print(f"  {task}/seed_{seed}: {n_csv} CSVs, {n_png} PNGs, {n_npz} NPZs")

    agg_dir = os.path.join(OUT_DIR, "glue", "aggregate")
    if os.path.isdir(agg_dir):
        files = sorted(os.listdir(agg_dir))
        print(f"  aggregate/: {', '.join(files)}")


if __name__ == "__main__":
    main()
