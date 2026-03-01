"""diag_fwdio_utils.py — Shared forward I/O diagnostic utilities (task-agnostic).

Extracted from diag_forward_io_single_rpu.py with minimal task-agnostic changes:
  - calibrate_out_bounds: takes device + model_fn callable instead of global DEVICE
  - print_trainability_report: learn_out_scaling optional kwarg
  - save_meta_json: seed/inp_bound/out_bound as params
  - _batch_to_device: takes device as param
  - plot_adc_sweep: handles GLUE/SQuAD logit columns generically

Correctness fixes (E1-E3) replicated from diag_forward_io_single_rpu.py:
  E1: half_step = out_res * out_bound          (not * 0.5)
  E2: per-module bound/res lookup from dicts
  E3: bits proxy = round(log2(1/out_res + 2))  in summary computations
"""

# =============================================================================
# Imports
# =============================================================================

import copy
import gc
import json
import os
import re
import subprocess
from math import log2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim.context import AnalogContext
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import SoftBoundsDevice
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType

# =============================================================================
# Constants
# =============================================================================

EPS       = 1e-8
OUT_BOUND = 12.0
INP_BOUND = 1.0
N_LAYERS  = 12

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


# =============================================================================
# Layer Name Utilities
# =============================================================================

def parse_layer_name(name: str):
    """Returns (layer_idx: int, sublayer: str) or None."""
    m = _LAYER_RE.search(name)
    if m is None:
        return None
    return int(m.group(1)), _SUBLAYER_MAP[m.group(2)]


# =============================================================================
# ForwardMACStats
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

        self.mac_snr_db_steps         = []
        self.mac_nmse_steps           = []
        self.cosine_steps             = []
        self.out_clip_ratio_steps     = []
        self.ref_deadzone_ratio_steps = []
        self.mean_abs_err_steps       = []
        self.median_abs_err_steps     = []
        self.p95_abs_err_steps        = []
        self._step_indices            = []

    def update(self, step: int, y_ana: torch.Tensor, y_ref: torch.Tensor = None):
        """y_ana: always provided (cheap metrics).
        y_ref=None → clip-only (out_clip_ratio).
        y_ref provided → full metrics (SNR, NMSE, cosine, deadzone, abs_err).
        """
        y_ana = y_ana.float().cpu()

        if y_ana.dim() == 3:
            B, S, D = y_ana.shape
            y_ana_flat = y_ana.reshape(B * S, D)
        else:
            y_ana_flat = y_ana

        # E1 fix: half_step = out_res * out_bound  (correct step = 2 * half_step)
        half_step      = self.out_res * self.out_bound
        clip_threshold = self.out_bound - half_step   # = out_bound * (1 - out_res)
        out_clip_ratio = (y_ana_flat.abs() > clip_threshold).float().mean().item()

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

            # E1 fix: deadzone uses same half_step
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
                "out_bound":          self.out_bound,
                "out_res":            self.out_res,
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
            "out_bound":               self.out_bound,
            "out_res":                 self.out_res,
            "mac_snr_db_mean":         float(np.nanmean(self.mac_snr_db_steps))         if self.mac_snr_db_steps         else float("nan"),
            "mac_nmse_mean":           float(np.nanmean(self.mac_nmse_steps))           if self.mac_nmse_steps           else float("nan"),
            "cosine_mean":             float(np.nanmean(self.cosine_steps))             if self.cosine_steps             else float("nan"),
            "out_clip_ratio_mean":     float(np.nanmean(self.out_clip_ratio_steps))     if self.out_clip_ratio_steps     else float("nan"),
            "ref_deadzone_ratio_mean": float(np.nanmean(self.ref_deadzone_ratio_steps)) if self.ref_deadzone_ratio_steps else float("nan"),
            "mean_abs_err_mean":       float(np.nanmean(self.mean_abs_err_steps))       if self.mean_abs_err_steps       else float("nan"),
            "median_abs_err_mean":     float(np.nanmean(self.median_abs_err_steps))     if self.median_abs_err_steps     else float("nan"),
            "p95_abs_err_mean":        float(np.nanmean(self.p95_abs_err_steps))        if self.p95_abs_err_steps        else float("nan"),
        }


# =============================================================================
# register_forward_hooks
# =============================================================================

def register_forward_hooks(
    model,
    out_res: float,
    out_bound: float,
    learn_out_scaling: bool = False,
    hook_active: list = None,
    per_module_out_bound: dict = None,   # name → float  (E2)
    per_module_out_res:   dict = None,   # name → float  (E2)
):
    """Register forward hooks on all AnalogLinear encoder layers.

    hook_active: [bool] — set False by logit eval loop to avoid contaminating
        MAC metric accumulation with those extra forward passes (E4).

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

        # E2: per-module lookup
        mod_bound = per_module_out_bound.get(name, out_bound) if per_module_out_bound else out_bound
        mod_res   = per_module_out_res.get(name, out_res)     if per_module_out_res   else out_res

        stats = ForwardMACStats(name, layer_idx, sublayer, mod_res, mod_bound)
        stats_dict[name] = stats

        def make_hook(nm, st, active):
            step_counter = [0]

            def hook(module, inp, output):
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

        h = module.register_forward_hook(make_hook(name, stats, hook_active))
        handles.append(h)

    print(f"  Registered forward hooks on {len(stats_dict)} AnalogLinear layers")
    return stats_dict, handles


# =============================================================================
# RPU Config Creation
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
    out_noise=0.0,
    sto_round=False,
    bound_management="NONE",
    learn_out_scaling=False,
    forward_is_perfect=False,
    out_bound=OUT_BOUND,
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
        io.out_bound        = out_bound
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


# =============================================================================
# Model Utilities
# =============================================================================

def _encoder_linear_names(model, always_digital=None):
    """All encoder Linear layer names, excluding always_digital layers."""
    if always_digital is None:
        always_digital = ["pooler"]
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

    Per-module IO params differ (E2). Prints verification table for first/last 3 modules.
    """
    enc_set = set(enc_names)
    replacements = []
    for parent_name, parent_mod in model.named_modules():
        for child_name, child_mod in list(parent_mod.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child_mod, nn.Linear) and full_name in enc_set:
                replacements.append((parent_mod, child_name, full_name, child_mod))

    printed = 0
    n_total = len(replacements)
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

        if printed < 3 or (n_total - printed) <= 3:
            ob  = rpu.forward.out_bound
            or_ = rpu.forward.out_res
            print(f"  [PerModule] {full_name}: out_bound={ob:.4f}, out_res={or_:.6f}")
        printed += 1

    print(f"  [PerModule] Replaced {printed} nn.Linear → AnalogLinear (expected 72)")
    return model


# =============================================================================
# Out-bound Calibration
# =============================================================================

def calibrate_out_bounds(
    loader,
    args,
    enc_names: list,
    device,
    model_fn,           # callable() → digital nn.Module (for clean forward pass)
) -> dict:             # module_name → calibrated out_bound float
    """Calibrate per-module output bound using a clean digital model forward pass.

    model_fn is task-agnostic: it should return the pretrained digital model
    without any analog conversion.
    """
    print("  [CalibOutBound] Loading digital model for calibration...")
    dig_model = model_fn()
    dig_model = dig_model.to(device)
    dig_model.eval()

    enc_set = set(enc_names)
    samples_per_module = {}  # name → list[Tensor]

    hooks = []
    for name, mod in dig_model.named_modules():
        if isinstance(mod, nn.Linear) and name in enc_set:
            def _make_hook(n, m):
                def _hook(_, inp, _out):
                    x = inp[0].detach()
                    y = F.linear(x, m.weight.detach(),
                                 m.bias.detach() if m.bias is not None else None)
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
            dig_model(**_batch_to_device(batch, device))

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


# =============================================================================
# Mixed Precision Assignment
# =============================================================================

def compute_mixed_precision_assignment(args, enc_names: list) -> dict:
    """Returns {module_name: adc_bits} for all 72 modules.

    E3: bits stored as int; inverse from out_res uses round(log2(1/out_res + 2)).
    """
    assignment = {}
    boost_ranges = []
    cap_adc_bits = getattr(args, "cap_adc_bits", 12)

    if getattr(args, "depth_boost", None):
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
        bits = min(bits, cap_adc_bits)
        assignment[name] = bits

    print(f"\n  [MixedPrecision] ADC bit assignment (base={args.adc_base}):")
    for sl in SUBLAYER_ORDER:
        sl_bits = [b for n, b in assignment.items()
                   if parse_layer_name(n) and parse_layer_name(n)[1] == sl]
        if sl_bits:
            print(f"    {sl}: {min(sl_bits)}–{max(sl_bits)} bits")

    return assignment


# =============================================================================
# Trainability Report
# =============================================================================

def print_trainability_report(model, learn_out_scaling=False):
    """Parameter category trainable count + layer coverage check."""
    counts = {"AnalogContext": 0, "head": 0, "LayerNorm": 0,
              "out_scaling": 0, "other_trainable": 0, "frozen": 0}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            counts["frozen"] += p.numel()
        elif isinstance(p, AnalogContext):
            counts["AnalogContext"] += p.numel()
        elif any(h in n for h in ["qa_outputs", "classifier"]):
            counts["head"] += p.numel()
        elif "LayerNorm" in n or "layer_norm" in n:
            counts["LayerNorm"] += p.numel()
        elif "out_scaling" in n:
            counts["out_scaling"] += p.numel()
        else:
            counts["other_trainable"] += p.numel()
    print("  [Trainability]", {k: v for k, v in counts.items() if v > 0})
    if learn_out_scaling and counts["out_scaling"] == 0:
        print("  [WARNING] learn_out_scaling=True but out_scaling params are NOT trainable!")

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


# =============================================================================
# Meta JSON
# =============================================================================

def save_meta_json(args, out_dir: str, tag: str, seed=42, inp_bound=1.0, out_bound=12.0):
    import transformers
    import aihwkit
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
        "seed": seed, "inp_bound": inp_bound, "out_bound": out_bound,
        "args": vars(args),
    }
    path = os.path.join(out_dir, f"{tag}_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  Saved meta → {path}")


# =============================================================================
# Plotting
# =============================================================================

def plot_run_heatmaps(label: str, stats_dict: dict, out_dir: str):
    """(layer_idx × sublayer) heatmaps: mac_snr_db, out_clip_ratio, ref_deadzone_ratio."""
    rows = []
    for st in stats_dict.values():
        rows.append({
            "layer_idx":              st.layer_idx,
            "sublayer":               st.sublayer,
            "mac_snr_db_mean":        np.nanmean(st.mac_snr_db_steps)        if st.mac_snr_db_steps        else np.nan,
            "out_clip_ratio_mean":    np.nanmean(st.out_clip_ratio_steps)    if st.out_clip_ratio_steps    else np.nan,
            "ref_deadzone_ratio_mean": np.nanmean(st.ref_deadzone_ratio_steps) if st.ref_deadzone_ratio_steps else np.nan,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    metrics = [
        ("mac_snr_db_mean",        "MAC SNR (dB)",          "viridis"),
        ("out_clip_ratio_mean",    "Output Clip Ratio",      "Reds"),
        ("ref_deadzone_ratio_mean", "Ref Deadzone Ratio",    "Blues"),
    ]
    for metric, title, cmap in metrics:
        if metric not in df.columns:
            continue
        pivot = df.pivot(index="layer_idx", columns="sublayer", values=metric)
        pivot = pivot.reindex(columns=[c for c in SUBLAYER_ORDER if c in pivot.columns])
        fig, ax = plt.subplots(figsize=(9, 6))
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        ax.set_xlabel("Sublayer")
        ax.set_ylabel("Encoder Layer Index")
        ax.set_title(f"{label} — {title}")
        plt.colorbar(im, ax=ax)
        slug = metric.replace("_mean", "")
        path = os.path.join(out_dir, f"{label}_heatmap_{slug}.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved heatmap → {path}")


def plot_adc_sweep(df: pd.DataFrame, out_dir: str):
    """Plot MAC SNR, clip ratio, and available logit metrics vs ADC bits."""
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

    # Plot C: logit KL (generic — handles both SQuAD and GLUE column names)
    kl_cols = [c for c in df.columns if "kl" in c.lower() and "bits" not in c.lower()]
    if kl_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in kl_cols:
            ax.plot(df["adc_bits"], df[col], marker="o", label=col)
        ax.set_xlabel("ADC bits")
        ax.set_ylabel("Mean KL divergence")
        ax.set_title("Logit KL Divergence vs ADC bits")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(out_dir, "plot_C_logit_kl_vs_adc.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot → {path}")

    # Plot F: flip rate (generic)
    flip_cols = [c for c in df.columns if "flip" in c.lower() and "bits" not in c.lower()]
    if flip_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in flip_cols:
            ax.plot(df["adc_bits"], df[col], marker="o", label=col)
        ax.set_xlabel("ADC bits")
        ax.set_ylabel("Flip Rate")
        ax.set_title("Prediction Flip Rate vs ADC bits")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, "plot_F_flip_rate_vs_adc.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot → {os.path.join(out_dir, 'plot_F_flip_rate_vs_adc.png')}")

    # Plot G: deadzone ratio per sublayer
    dz_cols = [f"ref_deadzone_ratio_{sl}_mean" for sl in SUBLAYER_ORDER]
    dz_present = [c for c in dz_cols if c in df.columns]
    if dz_present:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in dz_present:
            sl = col.split("_")[3]
            ax.plot(df["adc_bits"], df[col], marker="o", label=sl)
        ax.set_xlabel("ADC bits")
        ax.set_ylabel("Ref Deadzone Ratio")
        ax.set_title("Deadzone Ratio vs ADC bits")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, "plot_G_deadzone_vs_adc.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved plot → {os.path.join(out_dir, 'plot_G_deadzone_vs_adc.png')}")


# =============================================================================
# Batch Utility
# =============================================================================

def _batch_to_device(batch: dict, device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}
