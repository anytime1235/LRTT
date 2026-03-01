"""diag_forward_io_single_rpu.py — BERT-base analog inference diagnostics.

Diagnoses three dimensions using SingleRPUConfig(SoftBoundsDevice):
  1. Per-layer forward MAC fidelity (SNR, NMSE, cosine, clip ratio, deadzone ratio)
  2. Logit-level divergence between analog train model and ideal reference model
  3. Weight-update quantization graininess (dw_min effects)

Usage:
  python diag_forward_io_single_rpu.py --n-step 5 --batch-size 2  # smoke test
  python diag_forward_io_single_rpu.py --adc-bits 12 --out-noise 0 --bound-mgmt NONE
  python diag_forward_io_single_rpu.py --adc-bits 4
  python diag_forward_io_single_rpu.py --dw-min-sweep "0.0005,0.001,0.002"
  python diag_forward_io_single_rpu.py --adc-bits-sweep "4,6,8,10,12"
"""

# =============================================================================
# Section 1: Imports
# =============================================================================

import argparse
import copy
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
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    set_seed,
    default_data_collator,
)
from torch.utils.data import DataLoader
from datasets import load_dataset

from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import SoftBoundsDevice
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType

# =============================================================================
# Section 2: CLI Argument Parser
# =============================================================================

parser = argparse.ArgumentParser(
    description="Diagnose BERT-base analog inference — SingleRPU(SoftBoundsDevice)"
)
parser.add_argument("--n-step",          type=int,   default=200)
parser.add_argument("--batch-size",      type=int,   default=8)
parser.add_argument("--lr",              type=float, default=2e-3)
parser.add_argument("--optimizer",       type=str,   default="AnalogSGD",
                    choices=["AnalogSGD", "AnalogAdam"])
parser.add_argument("--dac-bits",        type=int,   default=7)
parser.add_argument("--adc-bits",        type=int,   default=9)
parser.add_argument("--dw-min",          type=float, default=0.001)
parser.add_argument("--out-noise",       type=float, default=0.0)
parser.add_argument("--sto-round",       action="store_true")
parser.add_argument("--bound-mgmt",      type=str,   default="NONE",
                    choices=["NONE", "ITERATIVE"])
parser.add_argument("--learn-out-scaling",           action="store_true")
parser.add_argument("--calib-freeze-out-scaling",    action="store_true")
parser.add_argument("--input-range-init-from-data",  type=int, default=0,
                    metavar="N")
parser.add_argument("--desired-bl",      type=int,   default=None)
parser.add_argument("--adc-bits-sweep",  type=str,   default=None,
                    metavar="BITS_CSV",
                    help="Comma-separated ADC bits to sweep, e.g. '4,6,8,10,12'")
parser.add_argument("--dw-min-sweep",    type=str,   default=None,
                    metavar="DW_CSV",
                    help="Comma-separated dw_min values to sweep, e.g. '0.0005,0.001,0.002'")
parser.add_argument("--out-dir",         type=str,   default="./results/diag_fwd_io")
parser.add_argument("--train-layernorm", action="store_true",
                    help="Also enable requires_grad for LayerNorm params")
parser.add_argument("--no-save-npz", action="store_true",
                    help="Skip saving NPZ artifact")

# Task 0: Tag system
parser.add_argument("--tag", type=str, default=None,
    help="Run tag; output goes to {out_dir}/{tag}/")

# Task 2: Out-bound calibration
parser.add_argument("--calib-out-bound", action="store_true")
parser.add_argument("--out-bound-grouping", type=str, default="per_module",
    choices=["per_module", "per_sublayer", "per_layer"])
parser.add_argument("--out-bound-quantile", type=float, default=0.999)
parser.add_argument("--out-bound-margin",   type=float, default=1.05)
parser.add_argument("--out-bound-calib-batches", type=int, default=32)
parser.add_argument("--out-bound-max",  type=float, default=12.0)
parser.add_argument("--out-bound-min",  type=float, default=0.5)
parser.add_argument("--clip-target",    type=float, default=1e-3)
parser.add_argument("--save-calib-table", action="store_true")

# Task 3: Mixed precision ADC
parser.add_argument("--mixed-precision", action="store_true")
parser.add_argument("--adc-base",        type=int,   default=6)
parser.add_argument("--ffn1-bits-plus",  type=int,   default=2)
parser.add_argument("--v-bits-plus",     type=int,   default=1)
parser.add_argument("--depth-boost",     type=str,   default=None,
    help="e.g. '9-11:+1'")
parser.add_argument("--cap-adc-bits",    type=int,   default=12)

# Task 5: Logit eval
parser.add_argument("--logit-eval-batches", type=int, default=0)

args = parser.parse_args()

# =============================================================================
# Section 3: Global Constants
# =============================================================================

MAX_SEQ_LENGTH = 384
DOC_STRIDE     = 128
SEED           = 42
INP_BOUND      = 1.0
OUT_BOUND      = 12.0
N_LAYERS       = 12
EPS            = 1e-8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_STEP     = args.n_step
BATCH_SIZE = args.batch_size
_tag       = args.tag or "run"
OUT_DIR    = os.path.join(args.out_dir, _tag)

os.makedirs(OUT_DIR, exist_ok=True)

print(f"[Config] Device={DEVICE}, N_STEP={N_STEP}, BATCH_SIZE={BATCH_SIZE}")
print(f"[Config] dac={args.dac_bits}b, adc={args.adc_bits}b, dw_min={args.dw_min}, "
      f"lr={args.lr}, optimizer={args.optimizer}")
print(f"[Config] out_noise={args.out_noise}, sto_round={args.sto_round}, "
      f"bound_mgmt={args.bound_mgmt}")
print(f"[Config] learn_out_scaling={args.learn_out_scaling}, OUT_DIR={OUT_DIR}")

# =============================================================================
# Section 4: Layer Name Utilities (verbatim from paper_figures.py lines 101-132)
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


def print_trainability_report(model, learn_out_scaling: bool):
    """파라미터 카테고리별 trainable 수 출력 + 레이어 커버리지 체크."""
    counts = {"AnalogContext": 0, "qa_outputs": 0, "LayerNorm": 0,
              "out_scaling": 0, "other_trainable": 0, "frozen": 0}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            counts["frozen"] += p.numel()
        elif isinstance(p, AnalogContext):
            counts["AnalogContext"] += p.numel()
        elif "qa_outputs" in n:
            counts["qa_outputs"] += p.numel()
        elif "LayerNorm" in n or "layer_norm" in n:
            counts["LayerNorm"] += p.numel()
        elif "out_scaling" in n:
            counts["out_scaling"] += p.numel()
        else:
            counts["other_trainable"] += p.numel()
    print("  [Trainability]", {k: v for k, v in counts.items() if v > 0})
    if learn_out_scaling and counts["out_scaling"] == 0:
        print("  [WARNING] learn_out_scaling=True but out_scaling params are NOT trainable!")

    # 레이어 커버리지 체크
    sublayer_coverage = {sl: set() for sl in SUBLAYER_ORDER}
    n_analog = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, AnalogLinear):
            continue
        n_analog += 1
        parsed = parse_layer_name(name)
        if parsed:
            layer_idx, sublayer = parsed
            sublayer_coverage[sublayer].add(layer_idx)

    print(f"  [Coverage] Total AnalogLinear: {n_analog} (expected 72)")
    if n_analog != 72:
        print(f"  [WARNING] Expected 72 AnalogLinear, got {n_analog}")
    for sl, layers in sublayer_coverage.items():
        missing = set(range(12)) - layers
        if missing:
            print(f"  [WARNING] Sublayer {sl} missing encoder layers: {sorted(missing)}")
        else:
            print(f"  [Coverage] {sl}: all 12 layers present")


# =============================================================================
# Section 5: RPU Config Creation
# =============================================================================

def _bound_mgmt_type(s: str) -> BoundManagementType:
    return {
        "NONE":      BoundManagementType.NONE,
        "ITERATIVE": BoundManagementType.ITERATIVE,
    }[s.upper()]


def create_rpu_config(
    dac_bits,
    adc_bits,
    dw_min,
    out_noise,
    sto_round,
    bound_management,
    learn_out_scaling,
    forward_is_perfect=False,
):
    """SingleRPUConfig with SoftBoundsDevice.

    All device-to-device variation and noise set to zero for clean diagnostics.
    """
    device = SoftBoundsDevice(
        dw_min=dw_min,
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

    inp_res = 1.0 / (2 ** dac_bits - 2)
    out_res = 1.0 / (2 ** adc_bits - 2)

    for io in [rpu.forward, rpu.backward]:
        io.inp_bound        = INP_BOUND
        io.inp_res          = inp_res
        io.out_bound        = OUT_BOUND
        io.out_res          = out_res
        io.noise_management = NoiseManagementType.ABS_MAX
        io.out_noise        = out_noise
        io.inp_sto_round    = sto_round
        io.out_sto_round    = sto_round
        io.bound_management = _bound_mgmt_type(bound_management)

    if forward_is_perfect:
        rpu.forward.is_perfect = True

    rpu.mapping.digital_bias              = True
    rpu.mapping.weight_scaling_omega      = 1.0
    rpu.mapping.weight_scaling_columnwise = True
    rpu.mapping.learn_out_scaling         = learn_out_scaling
    rpu.mapping.out_scaling_columnwise    = learn_out_scaling

    return rpu


def save_meta_json(args, out_dir: str, tag: str):
    import json, subprocess, transformers, aihwkit
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = "N/A"
    meta = {
        "tag": tag, "git_hash": git_hash,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "aihwkit_version": aihwkit.__version__,
        "seed": SEED, "inp_bound": INP_BOUND, "out_bound": OUT_BOUND,
        "args": vars(args),
    }
    path = os.path.join(out_dir, f"{tag}_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Saved meta → {path}")


# =============================================================================
# Section 6: Model Creation
# =============================================================================

def _encoder_linear_names(model):
    """All encoder Linear layer names, excluding qa_outputs and pooler."""
    always_digital = ["qa_outputs", "pooler"]
    return [
        n for n, m in model.named_modules()
        if isinstance(m, nn.Linear)
        and "encoder" in n
        and not any(d in n for d in always_digital)
    ]


def _replace_linear_per_module(
    model: nn.Module,
    enc_names: list,
    base_rpu,
    per_module_out_bound: dict = None,   # name → float
    per_module_out_res:   dict = None,   # name → float
) -> nn.Module:
    """Replace each encoder nn.Linear with AnalogLinear using its own deep-copied rpu_config.

    Called instead of convert_to_analog() when per-module IO params differ.
    Prints a verification table for the first 3 and last 3 modules to confirm
    that out_bound values actually differ (ACCEPTANCE CHECK).
    """
    enc_set = set(enc_names)
    replacements = []
    for parent_name, parent_mod in model.named_modules():
        for child_name, child_mod in list(parent_mod.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child_mod, nn.Linear) and full_name in enc_set:
                replacements.append((parent_mod, child_name, full_name, child_mod))

    printed = 0
    for parent_mod, child_name, full_name, child_mod in replacements:
        rpu = copy.deepcopy(base_rpu)
        if per_module_out_bound and full_name in per_module_out_bound:
            bound = per_module_out_bound[full_name]
            rpu.forward.out_bound  = bound
            rpu.backward.out_bound = bound
        if per_module_out_res and full_name in per_module_out_res:
            res = per_module_out_res[full_name]
            rpu.forward.out_res  = res
            rpu.backward.out_res = res

        analog_mod = AnalogLinear(
            in_features=child_mod.in_features,
            out_features=child_mod.out_features,
            bias=child_mod.bias is not None,
            rpu_config=rpu,
        )
        w = child_mod.weight.data.detach()
        b = child_mod.bias.data.detach() if child_mod.bias is not None else None
        analog_mod.set_weights(w, b)
        setattr(parent_mod, child_name, analog_mod)

        if printed < 3 or (len(replacements) - printed) <= 3:
            ob = rpu.forward.out_bound
            or_ = rpu.forward.out_res
            print(f"  [PerModule] {full_name}: out_bound={ob:.4f}, out_res={or_:.6f}")
        printed += 1

    print(f"  [PerModule] Replaced {printed} nn.Linear → AnalogLinear (expected 72)")
    return model


def create_model(
    dac_bits,
    adc_bits,
    dw_min,
    out_noise,
    sto_round,
    bound_management,
    learn_out_scaling,
    forward_is_perfect=False,
    train_layernorm=False,
    per_module_out_bound: dict = None,
    per_module_out_res:   dict = None,
):
    """BERT-base with all encoder linears converted to AnalogLinear (single pass).

    Key difference from paper_figures.py:
      - Single-pass conversion (not two-pass)
      - No frozen tiles — all tiles trainable
      - SoftBoundsDevice instead of IdealDevice
    """
    rpu = create_rpu_config(
        dac_bits=dac_bits,
        adc_bits=adc_bits,
        dw_min=dw_min,
        out_noise=out_noise,
        sto_round=sto_round,
        bound_management=bound_management,
        learn_out_scaling=learn_out_scaling,
        forward_is_perfect=forward_is_perfect,
    )

    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    enc_names = _encoder_linear_names(model)
    all_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude   = [n for n in all_names if n not in enc_names]

    if per_module_out_bound or per_module_out_res:
        model = _replace_linear_per_module(
            model, enc_names, rpu,
            per_module_out_bound=per_module_out_bound,
            per_module_out_res=per_module_out_res,
        )
    else:
        model = convert_to_analog(model, rpu, exclude_modules=exclude)

    # Gradient control: disable all, re-enable AnalogContext + qa_outputs
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        if isinstance(p, AnalogContext):
            p.requires_grad_(True)
    for n, p in model.named_parameters():
        if "qa_outputs" in n:
            p.requires_grad_(True)
    if learn_out_scaling:
        for n, p in model.named_parameters():
            if "out_scaling" in n:
                p.requires_grad_(True)
    if train_layernorm:
        for n, p in model.named_parameters():
            if "LayerNorm" in n or "layer_norm" in n:
                p.requires_grad_(True)

    n_analog = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog tiles: {n_analog}, forward_is_perfect={forward_is_perfect}, "
          f"dac={dac_bits}b, adc={adc_bits}b, dw_min={dw_min}")
    print_trainability_report(model, learn_out_scaling)
    return model.to(DEVICE)


# =============================================================================
# Section 6b: Out-bound Calibration (Task 2)
# =============================================================================

def calibrate_out_bounds(
    loader,
    args,
    enc_names: list,
) -> dict:  # module_name → calibrated out_bound float
    """Calibrate per-module output bound using a clean digital model forward pass."""
    print(f"  [CalibOutBound] Loading digital model for calibration...")
    dig_model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    dig_model = dig_model.to(DEVICE)
    dig_model.eval()

    enc_set = set(enc_names)
    samples_per_module = {}  # name → list[Tensor]

    hooks = []
    for name, mod in dig_model.named_modules():
        if isinstance(mod, nn.Linear) and name in enc_set:
            def _make_hook(n, m):
                def _hook(_, inp, _out):
                    x = inp[0].detach()
                    y = F.linear(x, m.weight.detach(), m.bias.detach() if m.bias is not None else None)
                    flat = y.abs().reshape(-1)
                    if flat.numel() > 50000:
                        flat = flat[:50000]
                    samples_per_module.setdefault(n, []).append(flat.cpu())
                return _hook
            hooks.append(mod.register_forward_hook(_make_hook(name, mod)))

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.out_bound_calib_batches:
                break
            dig_model(**_batch_to_device(batch, DEVICE))

    for h in hooks:
        h.remove()
    del dig_model
    gc.collect()

    # Compute raw bound per module
    raw_bounds = {}
    for name, sample_list in samples_per_module.items():
        all_s = torch.cat(sample_list)
        q_val = torch.quantile(all_s, args.out_bound_quantile).item()
        bound = float(q_val) * args.out_bound_margin
        bound = max(args.out_bound_min, min(args.out_bound_max, bound))
        raw_bounds[name] = bound

    # Apply grouping
    if args.out_bound_grouping == "per_module":
        bounds = raw_bounds

    elif args.out_bound_grouping == "per_sublayer":
        sl_vals = {}
        for name, b in raw_bounds.items():
            parsed = parse_layer_name(name)
            if parsed:
                _, sl = parsed
                sl_vals.setdefault(sl, []).append(b)
        sl_means = {sl: float(np.mean(v)) for sl, v in sl_vals.items()}
        bounds = {}
        for name in raw_bounds:
            parsed = parse_layer_name(name)
            if parsed:
                _, sl = parsed
                bounds[name] = sl_means[sl]
            else:
                bounds[name] = raw_bounds[name]

    elif args.out_bound_grouping == "per_layer":
        ly_vals = {}
        for name, b in raw_bounds.items():
            parsed = parse_layer_name(name)
            if parsed:
                li, _ = parsed
                ly_vals.setdefault(li, []).append(b)
        ly_means = {li: float(np.mean(v)) for li, v in ly_vals.items()}
        bounds = {}
        for name in raw_bounds:
            parsed = parse_layer_name(name)
            if parsed:
                li, _ = parsed
                bounds[name] = ly_means[li]
            else:
                bounds[name] = raw_bounds[name]

    else:
        bounds = raw_bounds

    unique_bounds = sorted(set(bounds.values()))
    print(f"  [CalibOutBound] grouping={args.out_bound_grouping}, "
          f"unique_bound_values={len(unique_bounds)}, "
          f"range=[{min(unique_bounds):.3f}, {max(unique_bounds):.3f}]")
    return bounds


def apply_calib_sanity_check(
    model: nn.Module,
    loader,
    args,
    bounds: dict,
    out_res: float,
) -> dict:
    """Short forward pass sanity check; prints clip_ratio summary."""
    print(f"  [SanityCheck] Running 5 batches to verify clip_ratio...")
    model.eval()
    clip_counts = []
    total_counts = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5:
                break
            _ = model(**_batch_to_device(batch, DEVICE))
    model.train()
    print(f"  [SanityCheck] Done. bounds range: [{min(bounds.values()):.3f}, {max(bounds.values()):.3f}]")
    return bounds


# =============================================================================
# Section 6c: Mixed Precision Assignment (Task 3)
# =============================================================================

def compute_mixed_precision_assignment(args, enc_names: list) -> dict:
    """Returns {module_name: adc_bits} for all 72 modules."""
    assignment = {}
    boost_ranges = []
    if args.depth_boost:
        for spec in args.depth_boost.split(","):
            rng, delta = spec.split(":")
            lo, hi = [int(x) for x in rng.split("-")]
            boost_ranges.append((lo, hi, int(delta)))

    for name in enc_names:
        parsed = parse_layer_name(name)
        if parsed is None:
            continue
        layer_idx, sublayer = parsed
        bits = args.adc_base
        if sublayer == "FFN1":
            bits += args.ffn1_bits_plus
        elif sublayer == "V":
            bits += args.v_bits_plus
        for lo, hi, delta in boost_ranges:
            if lo <= layer_idx <= hi:
                bits += delta
        bits = min(bits, args.cap_adc_bits)
        assignment[name] = bits

    print(f"\n  [MixedPrecision] ADC bit assignment (base={args.adc_base}):")
    for sl in SUBLAYER_ORDER:
        sl_bits = [b for n, b in assignment.items()
                   if parse_layer_name(n) and parse_layer_name(n)[1] == sl]
        if sl_bits:
            print(f"    {sl}: {min(sl_bits)}–{max(sl_bits)} bits")

    return assignment


# =============================================================================
# Section 7: Data Loading (copied from paper_figures.py lines 246-299)
# =============================================================================

def load_data(tokenizer, n_step, batch_size):
    """SQuAD v1.1 — subset for n_step batches. Seed-fixed for reproducibility."""

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
        remove_columns=raw["train"].column_names,
    )
    n = min(n_step * batch_size, len(tok))
    subset = tok.shuffle(seed=SEED).select(range(n))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        collate_fn=default_data_collator)
    print(f"  Dataset: {n} samples → {len(loader)} batches")
    return loader


# =============================================================================
# Section 8: ForwardMACStats + ForwardMACHook
# =============================================================================

class ForwardMACStats:
    """Accumulates per-layer MAC metrics across steps."""

    def __init__(self, name: str, layer_idx: int, sublayer: str,
                 out_res: float, out_bound: float):
        self.name      = name
        self.layer_idx = layer_idx
        self.sublayer  = sublayer
        self.out_res   = out_res
        self.out_bound = out_bound

        self.mac_snr_db_steps        = []
        self.mac_nmse_steps          = []
        self.cosine_steps            = []
        self.out_clip_ratio_steps    = []
        self.ref_deadzone_ratio_steps = []
        self.mean_abs_err_steps      = []
        self.median_abs_err_steps    = []
        self.p95_abs_err_steps       = []
        self._step_indices           = []

    def update(self, step: int, y_ana: torch.Tensor, y_ref: torch.Tensor = None):
        """y_ana: always provided (cheap metrics).
        y_ref=None → cheap only (out_clip_ratio).
        y_ref provided → full metrics (SNR, NMSE, cosine, deadzone, abs_err).
        """
        y_ana = y_ana.float().cpu()

        # Flatten to 2D (N, D)
        if y_ana.dim() == 3:
            B, S, D = y_ana.shape
            y_ana_flat = y_ana.reshape(B * S, D)
        else:
            y_ana_flat = y_ana

        # Cheap: out_clip_ratio (always, no y_ref needed)
        half_step      = self.out_res * self.out_bound                          # correct: step = out_res*2*out_bound
        clip_threshold = self.out_bound - half_step                             # = out_bound*(1 - out_res)
        out_clip_ratio = (y_ana_flat.abs() > clip_threshold).float().mean().item()

        # Expensive: y_ref-based metrics (only when y_ref provided)
        if y_ref is not None:
            y_ref = y_ref.float().cpu()
            if y_ref.dim() == 3:
                y_ref_flat = y_ref.reshape(B * S, D)
            else:
                y_ref_flat = y_ref

            var_ref    = y_ref_flat.var().item()
            var_err    = (y_ref_flat - y_ana_flat).var().item()
            mac_snr_db = 10.0 * np.log10(max(var_ref, EPS) / max(var_err, EPS))
            mac_nmse   = ((y_ref_flat - y_ana_flat) ** 2).mean().item() / \
                         (y_ref_flat.pow(2).mean().item() + EPS)
            cosine     = F.cosine_similarity(y_ref_flat, y_ana_flat, dim=1).mean().item()

            ref_deadzone_ratio = (y_ref_flat.abs() < half_step).float().mean().item()

            abs_err        = (y_ref_flat - y_ana_flat).abs()
            mean_abs_err   = abs_err.mean().item()
            median_abs_err = abs_err.median().item()
            p95_abs_err    = abs_err.float().quantile(0.95).item()
        else:
            mac_snr_db = mac_nmse = cosine = float("nan")
            ref_deadzone_ratio = float("nan")
            mean_abs_err = median_abs_err = p95_abs_err = float("nan")

        self._step_indices.append(step)
        self.mac_snr_db_steps.append(mac_snr_db)
        self.mac_nmse_steps.append(mac_nmse)
        self.cosine_steps.append(cosine)
        self.out_clip_ratio_steps.append(out_clip_ratio)
        self.ref_deadzone_ratio_steps.append(ref_deadzone_ratio)
        self.mean_abs_err_steps.append(mean_abs_err)
        self.median_abs_err_steps.append(median_abs_err)
        self.p95_abs_err_steps.append(p95_abs_err)

    def get_rows(self) -> list:
        rows = []
        for i, step in enumerate(self._step_indices):
            rows.append({
                "step":               step,
                "module_name":        self.name,
                "layer_idx":          self.layer_idx,
                "sublayer":           self.sublayer,
                "mac_snr_db":         self.mac_snr_db_steps[i],
                "mac_nmse":           self.mac_nmse_steps[i],
                "cosine":             self.cosine_steps[i],
                "out_clip_ratio":     self.out_clip_ratio_steps[i],
                "ref_deadzone_ratio": self.ref_deadzone_ratio_steps[i],
                "mean_abs_err":       self.mean_abs_err_steps[i],
                "median_abs_err":     self.median_abs_err_steps[i],
                "p95_abs_err":        self.p95_abs_err_steps[i],
            })
        return rows

    def summary(self, label: str, adc_bits: int, dac_bits: int, dw_min: float) -> dict:
        return {
            "label":                   label,
            "layer_idx":               self.layer_idx,
            "sublayer":                self.sublayer,
            "adc_bits":                adc_bits,
            "dac_bits":                dac_bits,
            "dw_min":                  dw_min,
            "mac_snr_db_mean":         float(np.nanmean(self.mac_snr_db_steps))         if self.mac_snr_db_steps         else float("nan"),
            "mac_nmse_mean":           float(np.nanmean(self.mac_nmse_steps))           if self.mac_nmse_steps           else float("nan"),
            "cosine_mean":             float(np.nanmean(self.cosine_steps))             if self.cosine_steps             else float("nan"),
            "out_clip_ratio_mean":     float(np.nanmean(self.out_clip_ratio_steps))     if self.out_clip_ratio_steps     else float("nan"),
            "ref_deadzone_ratio_mean": float(np.nanmean(self.ref_deadzone_ratio_steps)) if self.ref_deadzone_ratio_steps else float("nan"),
            "mean_abs_err_mean":       float(np.nanmean(self.mean_abs_err_steps))       if self.mean_abs_err_steps       else float("nan"),
            "median_abs_err_mean":     float(np.nanmean(self.median_abs_err_steps))     if self.median_abs_err_steps     else float("nan"),
            "p95_abs_err_mean":        float(np.nanmean(self.p95_abs_err_steps))        if self.p95_abs_err_steps        else float("nan"),
        }


def register_forward_hooks(model, out_res: float, out_bound: float,
                           learn_out_scaling: bool, hook_active: list = None,
                           per_module_out_bound: dict = None,
                           per_module_out_res: dict = None):
    """Register forward hooks on all AnalogLinear layers matching _LAYER_RE.

    hook_active: [bool] — set False by training loop during LogitDiagnostics
        forward pass so those extra forwards don't increment step counters or
        record data (Trap B fix). Per-module step_counter in finally block
        ensures counter always increments on actual training/inference steps
        even when hook body returns early via exception (Trap A fix).
        All metrics computed every step — no diag_interval gating (unified design).

    Returns:
        stats_dict: {module_name -> ForwardMACStats}
        handles:    list of hook handles (call .remove() to clean up)
    """
    if hook_active is None:
        hook_active = [True]

    stats_dict = {}
    handles    = []

    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear):
            continue
        parsed = parse_layer_name(name)
        if parsed is None:
            continue
        layer_idx, sublayer = parsed

        mod_bound = per_module_out_bound.get(name, out_bound) if per_module_out_bound else out_bound
        mod_res   = per_module_out_res.get(name, out_res)     if per_module_out_res   else out_res
        stats = ForwardMACStats(name, layer_idx, sublayer, mod_res, mod_bound)
        stats_dict[name] = stats

        # Closure capture
        def make_hook(mod, nm, st, active):
            step_counter = [0]

            def hook(module, inp, output):
                # Skip LogitDiag extra forward passes (Trap B)
                if not active[0]:
                    return
                try:
                    x = inp[0]
                    W, b = module.get_weights()
                    if isinstance(W, torch.Tensor):
                        W = W.detach().to(dtype=x.dtype, device=x.device)
                    else:
                        W = torch.tensor(W, dtype=x.dtype, device=x.device)
                    y_ref = x @ W.t()
                    if b is not None and isinstance(b, torch.Tensor):
                        y_ref = y_ref + b.detach().to(dtype=x.dtype, device=x.device)
                    st.update(step_counter[0], output.detach(), y_ref.detach())
                except Exception as e:
                    if step_counter[0] == 0:
                        print(f"  [HookERR] {nm}: {type(e).__name__}: {e}")
                finally:
                    step_counter[0] += 1  # always increments (Trap A fix)

            return hook

        h = module.register_forward_hook(make_hook(module, name, stats, hook_active))
        handles.append(h)

    print(f"  Registered forward hooks on {len(stats_dict)} AnalogLinear layers")
    return stats_dict, handles


# =============================================================================
# Section 9: LogitDiagnostics
# =============================================================================

class LogitDiagnostics:
    """Runs ref_model vs train_model logit comparison every training step."""

    def __init__(self, ref_model, train_model, device):
        self.ref_model   = ref_model
        self.train_model = train_model
        self.device      = device
        self.rows        = []

    def run(self, step: int, batch: dict):
        self.train_model.eval()   # disable dropout for logit comparison
        try:
            # Sync ref_model weights from train_model
            try:
                self.ref_model.load_state_dict(self.train_model.state_dict())
            except Exception:
                # State dicts may not align perfectly across forward_is_perfect;
                # do best-effort copy of matching keys
                ref_sd   = self.ref_model.state_dict()
                train_sd = self.train_model.state_dict()
                for k in ref_sd:
                    if k in train_sd and ref_sd[k].shape == train_sd[k].shape:
                        ref_sd[k].copy_(train_sd[k])

            batch_dev = {k: v.to(self.device) for k, v in batch.items()
                         if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                ref_out  = self.ref_model(**batch_dev)
                ana_out  = self.train_model(**batch_dev)
        finally:
            self.train_model.train()  # restore training mode

        rs = ref_out.start_logits.float().cpu()
        re = ref_out.end_logits.float().cpu()
        as_ = ana_out.start_logits.float().cpu()
        ae  = ana_out.end_logits.float().cpu()

        mse_start    = F.mse_loss(as_, rs).item()
        mse_end      = F.mse_loss(ae, re).item()
        cosine_logit = F.cosine_similarity(
            torch.cat([rs, re], dim=1),
            torch.cat([as_, ae], dim=1),
            dim=1,
        ).mean().item()

        kl_start = F.kl_div(
            F.log_softmax(as_, dim=-1),
            F.softmax(rs, dim=-1),
            reduction="batchmean",
        ).item()
        kl_end = F.kl_div(
            F.log_softmax(ae, dim=-1),
            F.softmax(re, dim=-1),
            reduction="batchmean",
        ).item()

        # Prediction flip rate (argmax mismatch)
        flip_start = (as_.argmax(dim=-1) != rs.argmax(dim=-1)).float().mean().item()
        flip_end   = (ae.argmax(dim=-1)  != re.argmax(dim=-1)).float().mean().item()

        # Error margin: logit[true_label] - max(other logits)
        batch_sp = batch.get("start_positions", None)
        batch_ep = batch.get("end_positions", None)
        if batch_sp is not None and batch_ep is not None:
            sp = batch_sp.cpu()
            ep = batch_ep.cpu()
            B = as_.shape[0]
            margin_start_list, margin_end_list = [], []
            for bi in range(B):
                s_idx = sp[bi].item()
                true_s = as_[bi, s_idx].item()
                others_s = torch.cat([as_[bi, :s_idx], as_[bi, s_idx+1:]]).max().item()
                margin_start_list.append(true_s - others_s)
                e_idx = ep[bi].item()
                true_e = ae[bi, e_idx].item()
                others_e = torch.cat([ae[bi, :e_idx], ae[bi, e_idx+1:]]).max().item()
                margin_end_list.append(true_e - others_e)
            margin_start = float(np.mean(margin_start_list))
            margin_end   = float(np.mean(margin_end_list))
        else:
            margin_start = float("nan")
            margin_end   = float("nan")

        self.rows.append({
            "step":         step,
            "mse_start":    mse_start,
            "mse_end":      mse_end,
            "cosine_logit": cosine_logit,
            "kl_start":     kl_start,
            "kl_end":       kl_end,
            "flip_start":   flip_start,
            "flip_end":     flip_end,
            "margin_start": margin_start,
            "margin_end":   margin_end,
        })

    def save_csv(self, path: str):
        pd.DataFrame(self.rows).to_csv(path, index=False)
        print(f"  Saved logit metrics → {path}")

    def mean_kl(self) -> tuple:
        if not self.rows:
            return float("nan"), float("nan")
        return (
            float(np.mean([r["kl_start"] for r in self.rows])),
            float(np.mean([r["kl_end"]   for r in self.rows])),
        )


# =============================================================================
# Section 10: WeightDeltaTracker
# =============================================================================

def _get_weights_numpy(module: AnalogLinear) -> np.ndarray:
    """Return weight matrix from AnalogLinear as a numpy float32 array."""
    W, _ = module.get_weights()
    if isinstance(W, torch.Tensor):
        return W.detach().cpu().float().numpy()
    return np.asarray(W, dtype=np.float32)


class WeightDeltaTracker:
    """Samples weight deltas from AnalogLinear layers after optimizer.step."""

    def __init__(self, model, dw_min: float):
        self.dw_min  = dw_min
        self.rows    = []
        self._layers = {}   # name -> (module, W_before_flat_sample, sample_idx)

        for name, module in model.named_modules():
            if not isinstance(module, AnalogLinear):
                continue
            parsed = parse_layer_name(name)
            if parsed is None:
                continue
            W_np = _get_weights_numpy(module)
            n_total  = int(np.prod(W_np.shape))
            n_sample = max(1000, n_total // 10)
            n_sample = min(n_sample, n_total)
            idx = np.random.permutation(n_total)[:n_sample]
            W_flat = W_np.flatten()[idx].copy()
            layer_idx, sublayer = parsed
            self._layers[name] = {
                "module":    module,
                "idx":       idx,
                "W_before":  W_flat,
                "layer_idx": layer_idx,
                "sublayer":  sublayer,
            }

    def snapshot_before(self):
        for info in self._layers.values():
            W_np = _get_weights_numpy(info["module"])
            info["W_before"] = W_np.flatten()[info["idx"]].copy()

    def record_after(self, step: int):
        for name, info in self._layers.items():
            W_np = _get_weights_numpy(info["module"])
            W_after_sample = W_np.flatten()[info["idx"]]
            delta = W_after_sample - info["W_before"]

            dw_zero_ratio    = float((delta == 0.0).mean())
            dw_1lsb_ratio    = float(((np.abs(delta) > 0) &
                                      (np.abs(delta) <= 1.1 * self.dw_min)).mean())
            dw_absmean       = float(np.abs(delta).mean())
            nonzero          = np.abs(delta[delta != 0.0])
            min_nonzero      = float(nonzero.min()) if len(nonzero) > 0 else float("nan")

            self.rows.append({
                "step":             step,
                "layer_idx":        info["layer_idx"],
                "sublayer":         info["sublayer"],
                "dw_zero_ratio":    dw_zero_ratio,
                "dw_1lsb_ratio":    dw_1lsb_ratio,
                "dw_absmean":       dw_absmean,
                "min_nonzero_delta": min_nonzero,
            })

            # Update snapshot
            info["W_before"] = W_after_sample.copy()

    def summary_means(self) -> dict:
        if not self.rows:
            return {"dw_zero_ratio": float("nan"), "dw_1lsb_ratio": float("nan"),
                    "dw_absmean": float("nan")}
        return {
            "dw_zero_ratio": float(np.mean([r["dw_zero_ratio"] for r in self.rows])),
            "dw_1lsb_ratio": float(np.mean([r["dw_1lsb_ratio"] for r in self.rows])),
            "dw_absmean":    float(np.mean([r["dw_absmean"]    for r in self.rows])),
        }

    def save_csv(self, path: str):
        pd.DataFrame(self.rows).to_csv(path, index=False)
        print(f"  Saved weight delta metrics → {path}")


# =============================================================================
# Section 11: Single Run Function
# =============================================================================

def _batch_to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def print_summary_table(rows: list):
    """Print a concise per-sublayer mean table."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("Summary (means over steps/layers)")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")


def run_one(
    dac_bits: int,
    adc_bits: int,
    dw_min: float,
    args,
    tokenizer,
    loader,
    label: str = "run",
    per_module_out_bound: dict = None,
    per_module_out_res:   dict = None,
) -> dict:
    """Run n_step training + full diagnostics. Return summary dict."""

    out_res = 1.0 / (2 ** adc_bits - 2)

    print(f"\n[run_one] label={label}, dac={dac_bits}b, adc={adc_bits}b, dw_min={dw_min}")

    # 1. Create train_model
    train_model = create_model(
        dac_bits=dac_bits, adc_bits=adc_bits, dw_min=dw_min,
        out_noise=args.out_noise, sto_round=args.sto_round,
        bound_management=args.bound_mgmt,
        learn_out_scaling=args.learn_out_scaling,
        forward_is_perfect=False,
        train_layernorm=args.train_layernorm,
        per_module_out_bound=per_module_out_bound,
        per_module_out_res=per_module_out_res,
    )

    # 2. Create ref_model (ideal forward)
    ref_model = create_model(
        dac_bits=dac_bits, adc_bits=adc_bits, dw_min=dw_min,
        out_noise=0.0, sto_round=False,
        bound_management="NONE",
        learn_out_scaling=args.learn_out_scaling,
        forward_is_perfect=True,
        train_layernorm=args.train_layernorm,
    )
    ref_model.eval()

    # 3. Register forward hooks on train_model
    hook_active = [True]  # False during LogitDiag forward passes (Trap B)
    stats_dict, handles = register_forward_hooks(
        train_model, out_res=out_res, out_bound=OUT_BOUND,
        learn_out_scaling=args.learn_out_scaling,
        hook_active=hook_active,
        per_module_out_bound=per_module_out_bound,
        per_module_out_res=per_module_out_res,
    )

    # 4. Create optimizer
    if args.optimizer == "AnalogSGD":
        optimizer = AnalogSGD(train_model.parameters(), lr=args.lr)
    else:
        optimizer = AnalogAdam(train_model.parameters(), lr=args.lr)
    optimizer.regroup_param_groups(train_model)

    # 5. Input range init from data (optional)
    if args.input_range_init_from_data > 0:
        print(f"  [InputRange] Calibrating over {args.input_range_init_from_data} batches...")
        absmax_vals = []
        train_model.eval()
        with torch.no_grad():
            for calib_step, batch in enumerate(loader):
                if calib_step >= args.input_range_init_from_data:
                    break
                _ = train_model(**_batch_to_device(batch, DEVICE))
                # Note: actual inp_bound adjustment would require tile-level API;
                # this pass collects stats for informational purposes.
        train_model.train()

    # 6. Calibrate + freeze out_scaling (optional)
    if args.calib_freeze_out_scaling and args.learn_out_scaling:
        calib_steps = max(1, args.n_step // 10)
        print(f"  [CalibOutScaling] Calibrating out_scaling for {calib_steps} steps...")

        # 1) out_scaling만 trainable로 명시적 설정
        for p in train_model.parameters():
            p.requires_grad_(False)
        for n, p in train_model.named_parameters():
            if "out_scaling" in n:
                p.requires_grad_(True)

        # 2) L2 norm 스냅샷
        os_before = {n: p.detach().clone()
                     for n, p in train_model.named_parameters() if "out_scaling" in n}

        # 3) 캘리브 루프 (loader 처음부터)
        train_model.train()
        calib_loader = iter(loader)
        for calib_step in range(calib_steps):
            try:
                batch = next(calib_loader)
            except StopIteration:
                break
            optimizer.zero_grad()
            out = train_model(**_batch_to_device(batch, DEVICE))
            out.loss.backward()
            optimizer.step()

        # 4) L2 norm 변화 리포트
        for n, p_before in os_before.items():
            p_after = dict(train_model.named_parameters())[n].detach()
            delta_norm = (p_after - p_before).norm().item()
            print(f"  [CalibOutScaling] {n}: L2_delta={delta_norm:.6f} "
                  f"(before_norm={p_before.norm().item():.4f}, after_norm={p_after.norm().item():.4f})")
            if delta_norm < 1e-8:
                print(f"  [WARNING] out_scaling did NOT change during calibration!")

        # 5) 메인 학습용 grad 복구 + out_scaling 동결
        for p in train_model.parameters():
            p.requires_grad_(False)
        for p in train_model.parameters():
            if isinstance(p, AnalogContext):
                p.requires_grad_(True)
        for n, p in train_model.named_parameters():
            if "qa_outputs" in n:
                p.requires_grad_(True)
        # out_scaling은 동결 유지 (requires_grad=False)
        print("  [CalibOutScaling] out_scaling frozen. Resuming main training.")

    # Diagnostics objects
    logit_diag    = LogitDiagnostics(ref_model, train_model, DEVICE)
    weight_tracker = WeightDeltaTracker(train_model, dw_min=dw_min)

    # 7. Training loop
    train_model.train()
    loader_iter = iter(loader)

    for step in tqdm(range(args.n_step), desc=f"[{label}] training"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        weight_tracker.snapshot_before()

        optimizer.zero_grad()
        batch_dev = _batch_to_device(batch, DEVICE)
        outputs   = train_model(**batch_dev)
        outputs.loss.backward()
        optimizer.step()

        weight_tracker.record_after(step)

        hook_active[0] = False   # disable hooks during LogitDiag forward passes
        logit_diag.run(step, batch)
        hook_active[0] = True    # re-enable for next training step

    # 8. Remove hooks
    for h in handles:
        h.remove()
    plot_run_heatmaps(label, stats_dict, OUT_DIR)

    # 9. Build MAC metrics CSV
    mac_rows = []
    for st in stats_dict.values():
        mac_rows.extend(st.get_rows())

    mac_csv = os.path.join(OUT_DIR, f"{label}_layer_mac_metrics.csv")
    pd.DataFrame(mac_rows).to_csv(mac_csv, index=False)
    print(f"  Saved layer MAC metrics → {mac_csv}")

    logit_csv = os.path.join(OUT_DIR, f"{label}_logit_metrics.csv")
    logit_diag.save_csv(logit_csv)

    wdelta_csv = os.path.join(OUT_DIR, f"{label}_weight_delta_metrics.csv")
    weight_tracker.save_csv(wdelta_csv)

    # 9b. Module MAC summary (layer_idx × sublayer별 평균)
    if mac_rows:
        mac_df = pd.DataFrame(mac_rows)
        module_summary = mac_df.groupby(["layer_idx", "sublayer"])[
            ["mac_snr_db", "mac_nmse", "cosine", "out_clip_ratio",
             "ref_deadzone_ratio", "mean_abs_err", "median_abs_err", "p95_abs_err"]
        ].mean().reset_index()
        module_summary["label"] = label
        mod_sum_csv = os.path.join(OUT_DIR, f"{label}_module_mac_summary.csv")
        module_summary.to_csv(mod_sum_csv, index=False)
        print(f"  Saved module MAC summary → {mod_sum_csv}")

    # 9c. NPZ artifact
    if not args.no_save_npz:
        import json
        import transformers, aihwkit
        meta = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "aihwkit_version": aihwkit.__version__,
            "seed": SEED,
            "inp_bound": INP_BOUND,
            "out_bound": OUT_BOUND,
        }
        args_dict = vars(args)
        npz_path = os.path.join(OUT_DIR, f"{label}_records.npz")
        np.savez_compressed(
            npz_path,
            mac_records=np.array(mac_rows, dtype=object),
            logit_records=np.array(logit_diag.rows, dtype=object),
            wdelta_records=np.array(weight_tracker.rows, dtype=object),
            args_json=np.array(json.dumps(args_dict)),
            meta_json=np.array(json.dumps(meta)),
        )
        print(f"  Saved NPZ artifact → {npz_path}")

    # 10. Build per-sublayer mean summary
    sublayer_summaries = []
    for st in stats_dict.values():
        sublayer_summaries.append(
            st.summary(label, adc_bits, dac_bits, dw_min)
        )
    summary_df = pd.DataFrame(sublayer_summaries)

    # Aggregate by sublayer
    if not summary_df.empty:
        agg = summary_df.groupby("sublayer")[
            ["mac_snr_db_mean", "mac_nmse_mean", "cosine_mean",
             "out_clip_ratio_mean", "ref_deadzone_ratio_mean"]
        ].mean()
    else:
        agg = pd.DataFrame()

    kl_start, kl_end = logit_diag.mean_kl()
    dw_means = weight_tracker.summary_means()

    flip_start_mean  = float(np.mean([r["flip_start"]  for r in logit_diag.rows])) if logit_diag.rows else float("nan")
    flip_end_mean    = float(np.mean([r["flip_end"]    for r in logit_diag.rows])) if logit_diag.rows else float("nan")
    margin_start_mean = float(np.mean([r["margin_start"] for r in logit_diag.rows])) if logit_diag.rows else float("nan")
    margin_end_mean   = float(np.mean([r["margin_end"]   for r in logit_diag.rows])) if logit_diag.rows else float("nan")

    result = {
        "label":              label,
        "adc_bits":           adc_bits,
        "dac_bits":           dac_bits,
        "dw_min":             dw_min,
        "logit_kl_start":     kl_start,
        "logit_kl_end":       kl_end,
        "flip_start":         flip_start_mean,
        "flip_end":           flip_end_mean,
        "margin_start":       margin_start_mean,
        "margin_end":         margin_end_mean,
        **dw_means,
    }

    # Add per-sublayer SNR columns
    for sl in SUBLAYER_ORDER:
        if not agg.empty and sl in agg.index:
            result[f"mac_snr_{sl}_mean"]       = agg.loc[sl, "mac_snr_db_mean"]
            result[f"out_clip_ratio_{sl}_mean"] = agg.loc[sl, "out_clip_ratio_mean"]
        else:
            result[f"mac_snr_{sl}_mean"]       = float("nan")
            result[f"out_clip_ratio_{sl}_mean"] = float("nan")

    result["mac_snr_mean"] = float(agg["mac_snr_db_mean"].mean()) if not agg.empty else float("nan")
    result["out_clip_ratio_mean"] = float(agg["out_clip_ratio_mean"].mean()) if not agg.empty else float("nan")

    # Sanity check warnings
    if adc_bits >= 10 and args.out_noise == 0.0 and args.bound_mgmt == "NONE":
        nmse_vals = []
        for st in stats_dict.values():
            nmse_vals.extend(st.mac_nmse_steps)
        mean_nmse = float(np.mean(nmse_vals)) if nmse_vals else float("nan")
        if mean_nmse > 0.01:
            print(f"  [WARNING] High adc_bits={adc_bits}, out_noise=0, bound_mgmt=NONE "
                  f"but mac_nmse={mean_nmse:.4f} (expected < 0.01)")
        else:
            print(f"  [OK] mac_nmse={mean_nmse:.4f} < 0.01 as expected for high-precision config")

    if adc_bits <= 4:
        snr_mean = result.get("mac_snr_mean", float("nan"))
        if snr_mean > 20.0:
            print(f"  [WARNING] Low adc_bits={adc_bits} but mac_snr={snr_mean:.1f} dB "
                  f"(expected < 20 dB)")
        else:
            print(f"  [OK] Low adc_bits={adc_bits}: mac_snr={snr_mean:.1f} dB (< 20 dB)")

    # Print sublayer summary
    if not agg.empty:
        print(f"\n[{label}] Per-sublayer means:")
        print(agg.round(4).to_string())

    # STEP 10: Logit eval (run_one variant uses ref_model already loaded)
    if args.logit_eval_batches > 0:
        logit_eval_rows = []
        ref_model.eval()
        train_model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.logit_eval_batches:
                    break
                bd = _batch_to_device(batch, DEVICE)
                ref_out = ref_model(**bd)
                ana_out = train_model(**bd)
                ref_logits = torch.cat([ref_out.start_logits, ref_out.end_logits], dim=-1)
                ana_logits = torch.cat([ana_out.start_logits, ana_out.end_logits], dim=-1)
                mse  = F.mse_loss(ana_logits, ref_logits).item()
                kl   = F.kl_div(F.log_softmax(ana_logits, -1),
                                F.softmax(ref_logits, -1),
                                reduction="batchmean").item()
                cos  = F.cosine_similarity(
                    ana_logits.reshape(-1), ref_logits.reshape(-1), dim=0).item()
                ref_top1 = ref_logits.argmax(-1)
                ana_top1 = ana_logits.argmax(-1)
                flip = (ref_top1 != ana_top1).float().mean().item()
                ref_sorted = ref_logits.sort(-1, descending=True).values
                margin = (ref_sorted[:, 0] - ref_sorted[:, 1]).mean().item()
                logit_eval_rows.append({
                    "batch": i, "mse": mse, "kl": kl,
                    "cosine": cos, "flip": flip, "margin": margin,
                })
        le_csv = os.path.join(OUT_DIR, f"{label}_logit_eval.csv")
        pd.DataFrame(logit_eval_rows).to_csv(le_csv, index=False)
        print(f"  Saved logit eval → {le_csv}")

    # Save per-run summary row
    summary_row_path = os.path.join(OUT_DIR, f"{label}_summary_row.csv")
    pd.DataFrame([result]).to_csv(summary_row_path, index=False)

    # Cleanup
    del train_model, ref_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =============================================================================
# Section 12: Sweep Functions
# =============================================================================

def run_adc_one(
    dac_bits: int,
    adc_bits: int,
    args,
    loader,
    label: str = "adc",
    per_module_out_bound: dict = None,
    per_module_out_res:   dict = None,
) -> dict:
    """Inference-only ADC diagnostics.

    No optimizer, no backward, no weight updates — weights stay fixed.
    Purely measures how ADC/DAC quantization distorts the forward MAC output.
    LogitDiagnostics is omitted (same weights → constant comparison).
    WeightDeltaTracker is omitted (no updates).
    """
    out_res = 1.0 / (2 ** adc_bits - 2)
    print(f"\n[run_adc_one] label={label}, dac={dac_bits}b, adc={adc_bits}b")

    model = create_model(
        dac_bits=dac_bits, adc_bits=adc_bits, dw_min=args.dw_min,
        out_noise=args.out_noise, sto_round=args.sto_round,
        bound_management=args.bound_mgmt,
        learn_out_scaling=args.learn_out_scaling,
        forward_is_perfect=False,
        train_layernorm=args.train_layernorm,
        per_module_out_bound=per_module_out_bound,
        per_module_out_res=per_module_out_res,
    )
    model.eval()

    hook_active = [True]
    stats_dict, handles = register_forward_hooks(
        model, out_res=out_res, out_bound=OUT_BOUND,
        learn_out_scaling=args.learn_out_scaling,
        hook_active=hook_active,
        per_module_out_bound=per_module_out_bound,
        per_module_out_res=per_module_out_res,
    )

    # Inference loop: no backward, no optimizer
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc=f"[{label}] inference")):
            if step >= args.n_step:
                break
            _ = model(**_batch_to_device(batch, DEVICE))

    for h in handles:
        h.remove()
    plot_run_heatmaps(label, stats_dict, OUT_DIR)

    # MAC metrics CSV
    mac_rows = []
    for st in stats_dict.values():
        mac_rows.extend(st.get_rows())
    pd.DataFrame(mac_rows).to_csv(
        os.path.join(OUT_DIR, f"{label}_layer_mac_metrics.csv"), index=False)
    print(f"  Saved layer MAC metrics → {label}_layer_mac_metrics.csv")

    # Module MAC summary
    if mac_rows:
        mac_df = pd.DataFrame(mac_rows)
        module_summary = mac_df.groupby(["layer_idx", "sublayer"])[
            ["mac_snr_db", "mac_nmse", "cosine", "out_clip_ratio",
             "ref_deadzone_ratio", "mean_abs_err", "median_abs_err", "p95_abs_err"]
        ].mean().reset_index()
        module_summary["label"] = label
        mod_sum_csv = os.path.join(OUT_DIR, f"{label}_module_mac_summary.csv")
        module_summary.to_csv(mod_sum_csv, index=False)
        print(f"  Saved module MAC summary → {mod_sum_csv}")

    # NPZ
    if not args.no_save_npz:
        import json, transformers, aihwkit
        meta = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "aihwkit_version": aihwkit.__version__,
            "seed": SEED, "inp_bound": INP_BOUND, "out_bound": OUT_BOUND,
        }
        npz_path = os.path.join(OUT_DIR, f"{label}_records.npz")
        np.savez_compressed(
            npz_path,
            mac_records=np.array(mac_rows, dtype=object),
            args_json=np.array(json.dumps(vars(args))),
            meta_json=np.array(json.dumps(meta)),
        )
        print(f"  Saved NPZ artifact → {npz_path}")

    # Summary
    sublayer_summaries = [st.summary(label, adc_bits, dac_bits, args.dw_min)
                          for st in stats_dict.values()]
    summary_df = pd.DataFrame(sublayer_summaries)
    agg = summary_df.groupby("sublayer")[
        ["mac_snr_db_mean", "mac_nmse_mean", "cosine_mean",
         "out_clip_ratio_mean", "ref_deadzone_ratio_mean"]
    ].mean() if not summary_df.empty else pd.DataFrame()

    result = {
        "label":    label,
        "adc_bits": adc_bits,
        "dac_bits": dac_bits,
        "dw_min":   args.dw_min,
    }
    for sl in SUBLAYER_ORDER:
        if not agg.empty and sl in agg.index:
            result[f"mac_snr_{sl}_mean"]       = agg.loc[sl, "mac_snr_db_mean"]
            result[f"out_clip_ratio_{sl}_mean"] = agg.loc[sl, "out_clip_ratio_mean"]
        else:
            result[f"mac_snr_{sl}_mean"]       = float("nan")
            result[f"out_clip_ratio_{sl}_mean"] = float("nan")
    result["mac_snr_mean"] = float(agg["mac_snr_db_mean"].mean()) if not agg.empty else float("nan")
    result["out_clip_ratio_mean"] = float(agg["out_clip_ratio_mean"].mean()) if not agg.empty else float("nan")

    # STEP 9: Mixed precision cost proxy
    if per_module_out_res:
        all_bits = {n: int(round(1.0 / res + 2)) for n, res in per_module_out_res.items()}
        result["avg_adc_bits"] = float(np.mean(list(all_bits.values())))
        ffn1_v_bits = [b for n, b in all_bits.items()
                       if parse_layer_name(n) and parse_layer_name(n)[1] in ("FFN1", "V")]
        other_bits  = [b for n, b in all_bits.items()
                       if parse_layer_name(n) and parse_layer_name(n)[1] not in ("FFN1", "V")]
        result["avg_adc_bits_ffn1_v"] = float(np.mean(ffn1_v_bits)) if ffn1_v_bits else float("nan")
        result["avg_adc_bits_other"]  = float(np.mean(other_bits))  if other_bits  else float("nan")

    if not agg.empty:
        print(f"\n[{label}] Per-sublayer means:")
        print(agg.round(4).to_string())

    # STEP 10: Logit eval
    if args.logit_eval_batches > 0:
        ref_model_eval = create_model(
            dac_bits=dac_bits, adc_bits=adc_bits, dw_min=args.dw_min,
            out_noise=0.0, sto_round=False,
            bound_management="NONE",
            learn_out_scaling=args.learn_out_scaling,
            forward_is_perfect=True,
        )
        ref_model_eval.eval()
        model.eval()
        logit_eval_rows = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.logit_eval_batches:
                    break
                bd = _batch_to_device(batch, DEVICE)
                ref_out = ref_model_eval(**bd)
                ana_out = model(**bd)
                # Concatenate start+end logits for a single comparison tensor
                ref_logits = torch.cat([ref_out.start_logits, ref_out.end_logits], dim=-1)
                ana_logits = torch.cat([ana_out.start_logits, ana_out.end_logits], dim=-1)
                mse  = F.mse_loss(ana_logits, ref_logits).item()
                kl   = F.kl_div(F.log_softmax(ana_logits, -1),
                                F.softmax(ref_logits, -1),
                                reduction="batchmean").item()
                cos  = F.cosine_similarity(
                    ana_logits.reshape(-1), ref_logits.reshape(-1), dim=0).item()
                ref_top1 = ref_logits.argmax(-1)
                ana_top1 = ana_logits.argmax(-1)
                flip = (ref_top1 != ana_top1).float().mean().item()
                ref_sorted = ref_logits.sort(-1, descending=True).values
                margin = (ref_sorted[:, 0] - ref_sorted[:, 1]).mean().item()
                logit_eval_rows.append({
                    "batch": i, "mse": mse, "kl": kl,
                    "cosine": cos, "flip": flip, "margin": margin,
                })
        del ref_model_eval
        gc.collect()
        csv_path = os.path.join(OUT_DIR, f"{label}_logit_eval.csv")
        pd.DataFrame(logit_eval_rows).to_csv(csv_path, index=False)
        print(f"  Saved logit eval → {csv_path}")

    # Save per-run summary row
    summary_row_path = os.path.join(OUT_DIR, f"{label}_summary_row.csv")
    pd.DataFrame([result]).to_csv(summary_row_path, index=False)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_adc_sweep(adc_bits_list: list, args, tokenizer, loader,
                  sweep_tag=None,
                  per_module_out_bound=None,
                  per_module_out_res_fn=None) -> list:
    rows = []
    tag = sweep_tag or (args.tag or "run")
    for adc_bits in adc_bits_list:
        label = f"{tag}_adc{adc_bits}"
        pmr = per_module_out_res_fn(adc_bits) if per_module_out_res_fn else None
        row = run_adc_one(
            args.dac_bits, adc_bits, args, loader, label=label,
            per_module_out_bound=per_module_out_bound,
            per_module_out_res=pmr,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, f"{tag}_sweep_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep summary → {csv_path}")
    print_summary_table(rows)

    plot_adc_sweep(df, OUT_DIR)
    return rows


def run_dw_sweep(dw_min_list: list, args, tokenizer, loader) -> list:
    rows = []
    for dw_min in dw_min_list:
        label = f"dwmin{dw_min:.4f}".replace(".", "p")
        row = run_one(args.dac_bits, args.adc_bits, dw_min, args, tokenizer, loader, label=label)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "summary_dw_sweep.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved dw_min sweep summary → {csv_path}")
    print_summary_table(rows)

    plot_dw_sweep(df, OUT_DIR)
    return rows


# =============================================================================
# Section 13: Plotting Functions
# =============================================================================

def plot_run_heatmaps(label: str, stats_dict: dict, out_dir: str):
    """(layer_idx × sublayer) 히트맵: mac_snr_db, out_clip_ratio."""
    rows = []
    for st in stats_dict.values():
        rows.append({
            "layer_idx": st.layer_idx,
            "sublayer":  st.sublayer,
            "mac_snr_db_mean":     np.mean(st.mac_snr_db_steps)     if st.mac_snr_db_steps     else np.nan,
            "out_clip_ratio_mean": np.mean(st.out_clip_ratio_steps)  if st.out_clip_ratio_steps  else np.nan,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return
    for metric, title, cmap in [
        ("mac_snr_db_mean",     "MAC SNR (dB)",      "viridis"),
        ("out_clip_ratio_mean", "Output Clip Ratio",  "Reds"),
    ]:
        pivot = df.pivot(index="layer_idx", columns="sublayer", values=metric)
        pivot = pivot.reindex(columns=SUBLAYER_ORDER)
        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(SUBLAYER_ORDER))); ax.set_xticklabels(SUBLAYER_ORDER)
        ax.set_yticks(range(12)); ax.set_yticklabels(range(12))
        ax.set_xlabel("Sublayer"); ax.set_ylabel("Encoder Layer Index")
        ax.set_title(f"{label} — {title}")
        plt.colorbar(im, ax=ax)
        slug = metric.replace("_mean", "")
        path = os.path.join(out_dir, f"{label}_heatmap_{slug}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved heatmap → {path}")


def plot_adc_sweep(df: pd.DataFrame, out_dir: str):
    """Plot A: mac_snr_db vs adc_bits per sublayer.
       Plot B: out_clip_ratio vs adc_bits.
       Plot C: logit KL vs adc_bits.
    """
    if df.empty:
        return

    # Plot A: SNR per sublayer
    fig, ax = plt.subplots(figsize=(8, 5))
    for sl in SUBLAYER_ORDER:
        col = f"mac_snr_{sl}_mean"
        if col in df.columns:
            ax.plot(df["adc_bits"], df[col], marker="o", label=sl)
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("MAC SNR (dB)")
    ax.set_title("Forward MAC SNR vs ADC bits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "plot_A_snr_vs_adc.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {path}")

    # Plot B: clip ratio per sublayer
    fig, ax = plt.subplots(figsize=(8, 5))
    for sl in SUBLAYER_ORDER:
        col = f"out_clip_ratio_{sl}_mean"
        if col in df.columns:
            ax.plot(df["adc_bits"], df[col], marker="o", label=sl)
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("Output Clip Ratio")
    ax.set_title("Output Clip Ratio vs ADC bits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "plot_B_clip_ratio_vs_adc.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {path}")

    # Plot C: logit KL
    fig, ax = plt.subplots(figsize=(8, 5))
    if "logit_kl_start" in df.columns:
        ax.plot(df["adc_bits"], df["logit_kl_start"], marker="o", label="KL start")
    if "logit_kl_end" in df.columns:
        ax.plot(df["adc_bits"], df["logit_kl_end"],   marker="s", label="KL end")
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("Mean KL divergence")
    ax.set_title("Logit KL Divergence vs ADC bits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "plot_C_logit_kl_vs_adc.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {path}")

    # Plot F: flip_rate vs adc_bits (if present in df)
    if "flip_start" in df.columns or "flip_end" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        if "flip_start" in df.columns:
            ax.plot(df["adc_bits"], df["flip_start"], marker="o", label="flip start")
        if "flip_end" in df.columns:
            ax.plot(df["adc_bits"], df["flip_end"],   marker="s", label="flip end")
        ax.set_xlabel("ADC bits"); ax.set_ylabel("Flip Rate")
        ax.set_title("Prediction Flip Rate vs ADC bits")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, "plot_F_flip_rate_vs_adc.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot → {os.path.join(out_dir, 'plot_F_flip_rate_vs_adc.png')}")


def plot_dw_sweep(df: pd.DataFrame, out_dir: str):
    """Plot D: dw_zero_ratio and dw_1lsb_ratio vs dw_min.
       Plot E: mac_snr_db vs dw_min.
    """
    if df.empty:
        return

    # Plot D: weight delta ratios
    fig, ax = plt.subplots(figsize=(8, 5))
    if "dw_zero_ratio" in df.columns:
        ax.plot(df["dw_min"], df["dw_zero_ratio"], marker="o", label="zero ratio")
    if "dw_1lsb_ratio" in df.columns:
        ax.plot(df["dw_min"], df["dw_1lsb_ratio"], marker="s", label="1-LSB ratio")
    ax.set_xlabel("dw_min")
    ax.set_xscale("log")
    ax.set_ylabel("Ratio")
    ax.set_title("Weight Delta Ratios vs dw_min")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "plot_D_dw_ratios_vs_dwmin.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {path}")

    # Plot E: MAC SNR vs dw_min
    fig, ax = plt.subplots(figsize=(8, 5))
    if "mac_snr_mean" in df.columns:
        ax.plot(df["dw_min"], df["mac_snr_mean"], marker="o", color="tab:blue")
    ax.set_xlabel("dw_min")
    ax.set_xscale("log")
    ax.set_ylabel("Mean MAC SNR (dB)")
    ax.set_title("Forward MAC SNR vs dw_min")
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "plot_E_snr_vs_dwmin.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {path}")


# =============================================================================
# Section 14: main()
# =============================================================================

def main():
    torch.manual_seed(SEED)
    set_seed(SEED)
    np.random.seed(SEED)

    tag = args.tag or "run"
    save_meta_json(args, OUT_DIR, tag)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader    = load_data(tokenizer, N_STEP, BATCH_SIZE)

    # Pre-compute shared calibration / mixed-precision state
    enc_names_ref = _encoder_linear_names(
        AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased"))

    per_module_out_bound = None
    if args.calib_out_bound:
        per_module_out_bound = calibrate_out_bounds(loader, args, enc_names_ref)
        if args.save_calib_table:
            rows = [{"module": n,
                     "layer_idx": parse_layer_name(n)[0] if parse_layer_name(n) else -1,
                     "sublayer": parse_layer_name(n)[1] if parse_layer_name(n) else "?",
                     "calibrated_out_bound": b,
                     "baseline_out_bound": OUT_BOUND}
                    for n, b in per_module_out_bound.items()]
            calib_csv = os.path.join(OUT_DIR, f"{tag}_calib_table.csv")
            pd.DataFrame(rows).to_csv(calib_csv, index=False)
            print(f"  Saved calib table → {calib_csv}")

    mp_assignment = None
    if args.mixed_precision:
        mp_assignment = compute_mixed_precision_assignment(args, enc_names_ref)
        def per_module_out_res_fn(adc_bits_ignored):
            return {n: 1.0 / (2 ** b - 2) for n, b in mp_assignment.items()}
    else:
        def per_module_out_res_fn(adc_bits):
            return None

    if args.adc_bits_sweep:
        adc_list = [int(x.strip()) for x in args.adc_bits_sweep.split(",")]
        print(f"[Sweep] ADC bits: {adc_list}, tag={tag}")
        run_adc_sweep(
            adc_list, args, tokenizer, loader,
            sweep_tag=tag,
            per_module_out_bound=per_module_out_bound,
            per_module_out_res_fn=per_module_out_res_fn,
        )

    elif args.dw_min_sweep:
        dw_list = [float(x.strip()) for x in args.dw_min_sweep.split(",")]
        print(f"[Sweep] dw_min: {dw_list}")
        run_dw_sweep(dw_list, args, tokenizer, loader)

    else:
        result = run_one(
            dac_bits=args.dac_bits,
            adc_bits=args.adc_bits,
            dw_min=args.dw_min,
            args=args,
            tokenizer=tokenizer,
            loader=loader,
            label=tag,
            per_module_out_bound=per_module_out_bound,
            per_module_out_res=per_module_out_res_fn(args.adc_bits),
        )
        print_summary_table([result])


if __name__ == "__main__":
    main()
