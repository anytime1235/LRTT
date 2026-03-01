"""diag_forward_io_glue.py — BERT-base GLUE analog inference diagnostics.

Extends the SQuAD forward I/O diagnostic (diag_forward_io_single_rpu.py) to
GLUE classification / regression tasks using shared utilities from diag_fwdio_utils.

Diagnoses three dimensions using SingleRPU(SoftBoundsDevice):
  1. Per-layer forward MAC fidelity (SNR, NMSE, cosine, clip ratio, deadzone ratio)
  2. Logit-level divergence between analog model and ideal reference model
  3. ADC sweep / seed sweep / mixed-precision / out-bound calibration

Usage:
  # Smoke test
  python diag_forward_io_glue.py --glue-task sst2 --n-step 2 --batch-size 2 \\
      --adc-bits-sweep "4,8" --logit-eval-batches 2 --tag smoke_glue \\
      --out-dir /tmp/smoke_glue

  # ADC sweep
  python diag_forward_io_glue.py --glue-task sst2 --adc-bits-sweep "4,6,8,10,12" \\
      --tag baseline_sst2 --out-dir ./results/diag_fwd_io_glue

  # Out-bound calibration
  python diag_forward_io_glue.py --glue-task sst2 --adc-bits-sweep "4,6,8,10,12" \\
      --calib-out-bound --out-bound-grouping per_module --save-calib-table \\
      --tag obcal_sst2 --out-dir ./results/diag_fwd_io_glue

  # Seed sweep
  python diag_forward_io_glue.py --glue-task sst2 --adc-bits-sweep "4,6,8,10,12" \\
      --seed-sweep "42,43,44" --tag seed_sweep_sst2 --out-dir ./results/diag_fwd_io_glue
"""

# =============================================================================
# Imports
# =============================================================================

import argparse
import gc
import json
import os
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

from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim.context import AnalogContext

# Shared utilities
sys.path.insert(0, os.path.dirname(__file__))
from diag_fwdio_utils import (
    EPS, OUT_BOUND, INP_BOUND, N_LAYERS,
    SUBLAYER_ORDER,
    parse_layer_name,
    ForwardMACStats,
    register_forward_hooks,
    create_rpu_config,
    _encoder_linear_names,
    _replace_linear_per_module,
    calibrate_out_bounds,
    compute_mixed_precision_assignment,
    print_trainability_report,
    save_meta_json,
    plot_run_heatmaps,
    plot_adc_sweep,
    _batch_to_device,
)

# =============================================================================
# GLUE Task Config (verbatim from paper_figures_glue.py)
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
    "wnli": ("sentence1", "sentence2"),
}

TASK_TO_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
    "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1, "wnli": 2,
}

IS_REGRESSION = {"stsb"}

# =============================================================================
# CLI
# =============================================================================

parser = argparse.ArgumentParser(
    description="Diagnose BERT-base analog inference on GLUE — SingleRPU(SoftBoundsDevice)"
)
parser.add_argument("--glue-task", type=str, default="sst2",
                    choices=list(TASK_TO_KEYS.keys()),
                    help="GLUE task name (default: sst2)")
parser.add_argument("--max-seq-length", type=int, default=128,
                    help="Max sequence length for tokenization (default: 128)")
parser.add_argument("--n-step",          type=int,   default=200)
parser.add_argument("--batch-size",      type=int,   default=8)
parser.add_argument("--dac-bits",        type=int,   default=7)
parser.add_argument("--adc-bits",        type=int,   default=9)
parser.add_argument("--dw-min",          type=float, default=0.001)
parser.add_argument("--sto-round",       action="store_true")
parser.add_argument("--learn-out-scaling", action="store_true")
parser.add_argument("--bound-mgmt",      type=str,   default="NONE",
                    choices=["NONE", "ITERATIVE"])

# ADC / seed sweep
parser.add_argument("--adc-bits-sweep",  type=str,   default=None,
                    metavar="BITS_CSV",
                    help="Comma-separated ADC bits to sweep, e.g. '4,6,8,10,12'")
parser.add_argument("--seed-sweep",      type=str,   default=None,
                    metavar="SEEDS_CSV",
                    help="Comma-separated seeds for seed sweep, e.g. '42,43,44'")

# Out-bound calibration
parser.add_argument("--calib-out-bound",      action="store_true")
parser.add_argument("--out-bound-grouping",   type=str, default="per_module",
                    choices=["per_module", "per_sublayer", "per_layer"])
parser.add_argument("--out-bound-quantile",   type=float, default=0.999)
parser.add_argument("--out-bound-margin",     type=float, default=1.05)
parser.add_argument("--out-bound-calib-batches", type=int, default=32)
parser.add_argument("--out-bound-max",        type=float, default=12.0)
parser.add_argument("--out-bound-min",        type=float, default=0.5)
parser.add_argument("--save-calib-table",     action="store_true")

# Mixed precision
parser.add_argument("--mixed-precision", action="store_true")
parser.add_argument("--adc-base",        type=int,   default=6)
parser.add_argument("--ffn1-bits-plus",  type=int,   default=2)
parser.add_argument("--v-bits-plus",     type=int,   default=1)
parser.add_argument("--depth-boost",     type=str,   default=None,
                    help="e.g. '9-11:+1'")
parser.add_argument("--cap-adc-bits",    type=int,   default=12)

# Logit eval
parser.add_argument("--logit-eval-batches", type=int, default=0)

# Output
parser.add_argument("--out-dir",    type=str, default="./results/diag_fwd_io_glue")
parser.add_argument("--tag",        type=str, default=None)
parser.add_argument("--no-save-npz", action="store_true")

# Data capping
parser.add_argument("--train-subset-size", type=int, default=0,
                    help="Cap training samples (0 = use n_step * batch_size)")
parser.add_argument("--eval-subset-size",  type=int, default=0)

args = parser.parse_args()

# =============================================================================
# Global Setup
# =============================================================================

SEED       = 42
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_STEP     = args.n_step
BATCH_SIZE = args.batch_size
_tag       = args.tag or "run"
OUT_DIR    = os.path.join(args.out_dir, _tag)

os.makedirs(OUT_DIR, exist_ok=True)

print(f"[Config] Device={DEVICE}, GLUE={args.glue_task}, "
      f"N_STEP={N_STEP}, BATCH={BATCH_SIZE}")
print(f"[Config] dac={args.dac_bits}b, adc={args.adc_bits}b, dw_min={args.dw_min}, "
      f"sto_round={args.sto_round}")
print(f"[Config] OUT_DIR={OUT_DIR}")

# =============================================================================
# Deadzone Sanity (Acceptance I3)
# =============================================================================

print("\n[Sanity] Deadzone half_step = out_res * out_bound (should decrease with bits):")
for _bits in [4, 6, 8, 10, 12]:
    _out_res  = 1.0 / (2 ** _bits - 2)
    _half_step = _out_res * OUT_BOUND   # E1 correct formula
    print(f"  adc={_bits:2d}: half_step={_half_step:.5f}  (step={2*_half_step:.5f})")
print()

# =============================================================================
# GLUE Data Loading (verbatim from paper_figures_glue.py)
# =============================================================================

def load_glue_data(task, tokenizer, n_step, batch_size, seed, max_length=128):
    """Load GLUE task data with dynamic padding.

    Returns a DataLoader with n_step batches.
    Verbatim from paper_figures_glue.py.
    """
    assert task in TASK_TO_KEYS, f"Unknown GLUE task: {task}"
    key1, key2 = TASK_TO_KEYS[task]

    raw = load_dataset("nyu-mll/glue", task)
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
# GLUE Model Creation
# =============================================================================

def create_model_glue(
    num_labels,
    adc_bits,
    dac_bits,
    dw_min,
    out_noise=0.0,
    sto_round=False,
    bound_management="NONE",
    learn_out_scaling=False,
    forward_is_perfect=False,
    per_module_out_bound: dict = None,
    per_module_out_res:   dict = None,
):
    """BERT-base SequenceClassification with all 72 encoder linears as AnalogLinear.

    Excludes ["classifier", "pooler"] from analog conversion (always digital).
    Asserts exactly 72 AnalogLinear modules after conversion.
    """
    always_digital = ["classifier", "pooler"]

    rpu = create_rpu_config(
        dac_bits=dac_bits,
        adc_bits=adc_bits,
        dw_min=dw_min,
        out_noise=out_noise,
        sto_round=sto_round,
        bound_management=bound_management,
        learn_out_scaling=learn_out_scaling,
        forward_is_perfect=forward_is_perfect,
        out_bound=OUT_BOUND,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    )
    enc_names = _encoder_linear_names(model, always_digital=always_digital)
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

    # Gradient control
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.parameters():
        if isinstance(p, AnalogContext):
            p.requires_grad_(True)
    for n, p in model.named_parameters():
        if "classifier" in n:
            p.requires_grad_(True)
    if learn_out_scaling:
        for n, p in model.named_parameters():
            if "out_scaling" in n:
                p.requires_grad_(True)

    # Acceptance check I1: exactly 72 AnalogLinear
    n_analog = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    assert n_analog == 72, (
        f"Expected 72 AnalogLinear modules, got {n_analog}. "
        f"Check always_digital exclusion list: {always_digital}"
    )
    print(f"  Analog tiles: {n_analog} ✓, forward_is_perfect={forward_is_perfect}, "
          f"dac={dac_bits}b, adc={adc_bits}b, num_labels={num_labels}")
    print_trainability_report(model, learn_out_scaling)
    return model.to(DEVICE)


# =============================================================================
# GLUE Logit Diagnostics
# =============================================================================

class GlueLogitDiagnostics:
    """Compares analog model vs ideal reference model logits."""

    def __init__(self, ref_model, ana_model, task_name: str, device):
        self.ref_model  = ref_model
        self.ana_model  = ana_model
        self.task_name  = task_name
        self.device     = device
        self.rows       = []
        self._is_regression = task_name in IS_REGRESSION

    def run(self, step: int, batch: dict, hook_active: list = None):
        """Run comparison. Sets hook_active[0]=False during eval (E4)."""
        if hook_active is not None:
            hook_active[0] = False

        self.ana_model.eval()
        try:
            # Inference batch: exclude labels to avoid loss computation
            batch_inf = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items() if k != "labels"}

            with torch.no_grad():
                ref_out = self.ref_model(**batch_inf)
                ana_out = self.ana_model(**batch_inf)

            ref_logits = ref_out.logits.float().cpu()
            ana_logits = ana_out.logits.float().cpu()

            row = {"step": step}

            if self._is_regression:
                # STS-B: regression metrics
                ref_vals = ref_logits.squeeze(-1)
                ana_vals = ana_logits.squeeze(-1)
                mse_logit   = F.mse_loss(ana_vals, ref_vals).item()
                mae_logit   = (ana_vals - ref_vals).abs().mean().item()
                row["mse_logit"] = mse_logit
                row["mae_logit"] = mae_logit

                # Pearson and Spearman drift
                try:
                    from scipy.stats import pearsonr, spearmanr
                    r_val = ref_vals.numpy()
                    a_val = ana_vals.numpy()
                    if len(r_val) > 2:
                        row["pearson_drift"]  = float(pearsonr(r_val, a_val)[0])
                        row["spearman_drift"] = float(spearmanr(r_val, a_val)[0])
                    else:
                        row["pearson_drift"]  = float("nan")
                        row["spearman_drift"] = float("nan")
                except ImportError:
                    row["pearson_drift"]  = float("nan")
                    row["spearman_drift"] = float("nan")

                row["kl"] = float("nan")
                row["cosine_logit"] = float("nan")
                row["flip_rate"]    = float("nan")
                row["margin"]       = float("nan")

            else:
                # Classification metrics
                kl = F.kl_div(
                    F.log_softmax(ana_logits, dim=-1),
                    F.softmax(ref_logits, dim=-1),
                    reduction="batchmean",
                ).item()
                mse_logit   = F.mse_loss(ana_logits, ref_logits).item()
                cosine_logit = F.cosine_similarity(
                    ref_logits, ana_logits, dim=1
                ).mean().item()

                ref_top1 = ref_logits.argmax(dim=-1)
                ana_top1 = ana_logits.argmax(dim=-1)
                flip_rate = (ref_top1 != ana_top1).float().mean().item()

                ref_sorted = ref_logits.sort(dim=-1, descending=True).values
                if ref_sorted.shape[-1] >= 2:
                    margin = (ref_sorted[:, 0] - ref_sorted[:, 1]).mean().item()
                else:
                    margin = float("nan")

                row["kl"]            = kl
                row["mse_logit"]     = mse_logit
                row["cosine_logit"]  = cosine_logit
                row["flip_rate"]     = flip_rate
                row["margin"]        = margin
                row["mae_logit"]     = float("nan")
                row["pearson_drift"] = float("nan")
                row["spearman_drift"] = float("nan")

            self.rows.append(row)

        finally:
            self.ana_model.train()
            if hook_active is not None:
                hook_active[0] = True  # restore (E4)

    def save_csv(self, path: str):
        pd.DataFrame(self.rows).to_csv(path, index=False)
        print(f"  Saved logit eval → {path}")

    def mean_kl(self) -> float:
        if not self.rows:
            return float("nan")
        return float(np.nanmean([r.get("kl", float("nan")) for r in self.rows]))


# =============================================================================
# Single ADC Run (inference-only)
# =============================================================================

def run_adc_one_glue(
    dac_bits: int,
    adc_bits: int,
    args,
    loader,
    label: str = "adc",
    per_module_out_bound: dict = None,
    per_module_out_res:   dict = None,
) -> dict:
    """Inference-only ADC diagnostics for a GLUE task.

    No optimizer, no backward, no weight updates — weights stay fixed.
    """
    task     = args.glue_task
    num_labels = TASK_TO_NUM_LABELS[task]
    out_res  = 1.0 / (2 ** adc_bits - 2)

    print(f"\n[run_adc_one_glue] label={label}, task={task}, "
          f"dac={dac_bits}b, adc={adc_bits}b")

    # Step 1: Create analog model
    model = create_model_glue(
        num_labels=num_labels,
        adc_bits=adc_bits,
        dac_bits=dac_bits,
        dw_min=args.dw_min,
        sto_round=args.sto_round,
        bound_management=args.bound_mgmt,
        learn_out_scaling=args.learn_out_scaling,
        forward_is_perfect=False,
        per_module_out_bound=per_module_out_bound,
        per_module_out_res=per_module_out_res,
    )
    model.eval()

    # Step 2: 72-module assert (already in create_model_glue)

    # Step 3: Register forward hooks
    hook_active = [True]
    stats_dict, handles = register_forward_hooks(
        model,
        out_res=out_res,
        out_bound=OUT_BOUND,
        hook_active=hook_active,
        per_module_out_bound=per_module_out_bound,
        per_module_out_res=per_module_out_res,
    )

    # Step 4: Inference loop (no backward)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc=f"[{label}] inference")):
            if step >= args.n_step:
                break
            # Exclude labels for pure inference (avoid HF loss computation)
            batch_inf = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items() if k != "labels"}
            model(**batch_inf)

    # Step 5: Remove hooks and plot heatmaps
    for h in handles:
        h.remove()
    plot_run_heatmaps(label, stats_dict, OUT_DIR)

    # Step 6: Save MAC metrics CSV
    mac_rows = []
    for st in stats_dict.values():
        mac_rows.extend(st.get_rows())

    mac_csv = os.path.join(OUT_DIR, f"{label}_layer_mac_metrics.csv")
    pd.DataFrame(mac_rows).to_csv(mac_csv, index=False)
    print(f"  Saved layer MAC metrics → {mac_csv}")

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

    # NPZ artifact
    if not args.no_save_npz:
        import transformers
        import aihwkit
        meta = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "aihwkit_version": aihwkit.__version__,
            "seed": SEED,
            "inp_bound": INP_BOUND,
            "out_bound": OUT_BOUND,
        }
        npz_path = os.path.join(OUT_DIR, f"{label}_records.npz")
        np.savez_compressed(
            npz_path,
            mac_records=np.array(mac_rows, dtype=object),
            args_json=np.array(json.dumps(vars(args))),
            meta_json=np.array(json.dumps(meta)),
        )
        print(f"  Saved NPZ artifact → {npz_path}")

    # Step 7: Build per-sublayer mean summary
    sublayer_summaries = [st.summary(label, adc_bits, dac_bits, args.dw_min)
                          for st in stats_dict.values()]
    summary_df = pd.DataFrame(sublayer_summaries)
    agg = summary_df.groupby("sublayer")[
        ["mac_snr_db_mean", "mac_nmse_mean", "cosine_mean",
         "out_clip_ratio_mean", "ref_deadzone_ratio_mean"]
    ].mean() if not summary_df.empty else pd.DataFrame()

    result = {
        "label":    label,
        "task":     task,
        "adc_bits": adc_bits,
        "dac_bits": dac_bits,
        "dw_min":   args.dw_min,
    }
    from diag_fwdio_utils import SUBLAYER_ORDER as _SLO
    for sl in _SLO:
        if not agg.empty and sl in agg.index:
            result[f"mac_snr_{sl}_mean"]              = agg.loc[sl, "mac_snr_db_mean"]
            result[f"out_clip_ratio_{sl}_mean"]        = agg.loc[sl, "out_clip_ratio_mean"]
            result[f"ref_deadzone_ratio_{sl}_mean"]    = agg.loc[sl, "ref_deadzone_ratio_mean"]
        else:
            result[f"mac_snr_{sl}_mean"]           = float("nan")
            result[f"out_clip_ratio_{sl}_mean"]    = float("nan")
            result[f"ref_deadzone_ratio_{sl}_mean"] = float("nan")
    result["mac_snr_mean"] = float(agg["mac_snr_db_mean"].mean()) if not agg.empty else float("nan")
    result["out_clip_ratio_mean"] = float(agg["out_clip_ratio_mean"].mean()) if not agg.empty else float("nan")

    # Mixed precision cost proxy (E3)
    if per_module_out_res:
        from math import log2
        all_bits = {}
        for n, res in per_module_out_res.items():
            try:
                all_bits[n] = int(round(log2(1.0 / res + 2)))
            except Exception:
                all_bits[n] = adc_bits
        result["avg_adc_bits"] = float(np.mean(list(all_bits.values())))

    if not agg.empty:
        print(f"\n[{label}] Per-sublayer means:")
        print(agg.round(4).to_string())

    # Step 8: Logit eval
    if args.logit_eval_batches > 0:
        ref_model = create_model_glue(
            num_labels=num_labels,
            adc_bits=adc_bits,
            dac_bits=dac_bits,
            dw_min=args.dw_min,
            sto_round=False,
            forward_is_perfect=True,
        )
        ref_model.eval()
        model.eval()

        logit_diag = GlueLogitDiagnostics(ref_model, model, task, DEVICE)
        # E4: hook_active[0] = False during logit eval (already handled inside run())
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.logit_eval_batches:
                    break
                logit_diag.run(i, batch, hook_active=hook_active)

        le_csv = os.path.join(OUT_DIR, f"{label}_logit_eval.csv")
        logit_diag.save_csv(le_csv)
        result["logit_kl_mean"]       = logit_diag.mean_kl()
        result["logit_flip_rate_mean"] = float(np.nanmean(
            [r.get("flip_rate", float("nan")) for r in logit_diag.rows]))

        del ref_model
        gc.collect()

    # Step 9: Save per-run summary row
    summary_row_path = os.path.join(OUT_DIR, f"{label}_summary_row.csv")
    pd.DataFrame([result]).to_csv(summary_row_path, index=False)

    # Step 10: Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =============================================================================
# ADC Sweep
# =============================================================================

def run_adc_sweep_glue(
    adc_bits_list: list,
    args,
    loader,
    tag: str,
    per_module_out_bound: dict = None,
    per_module_out_res_fn=None,   # callable(adc_bits) → {name: res} | None
) -> list:
    """Run run_adc_one_glue for each adc_bits. Aggregate sweep summary CSV."""
    rows = []
    for adc_bits in adc_bits_list:
        label = f"{tag}_adc{adc_bits}"
        pmr = per_module_out_res_fn(adc_bits) if per_module_out_res_fn else None
        row = run_adc_one_glue(
            dac_bits=args.dac_bits,
            adc_bits=adc_bits,
            args=args,
            loader=loader,
            label=label,
            per_module_out_bound=per_module_out_bound,
            per_module_out_res=pmr,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, f"{tag}_sweep_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved sweep summary → {csv_path}")
    plot_adc_sweep(df, OUT_DIR)
    return rows


# =============================================================================
# Seed Sweep
# =============================================================================

def run_seed_sweep(
    adc_bits_list: list,
    args,
    tokenizer,
    tag: str,
    per_module_out_bound_fn=None,  # callable(seed) → {name: bound} | None
    per_module_out_res_fn=None,    # callable(adc_bits) → {name: res} | None
):
    """Run full ADC sweep for each seed. Compute mean±std per metric per adc_bits."""
    seeds = [int(s.strip()) for s in args.seed_sweep.split(",") if s.strip()]
    print(f"\n[SeedSweep] seeds={seeds}, adc_bits={adc_bits_list}")

    all_rows = []  # list of dicts, one per (seed, adc_bits)

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"[SeedSweep] seed={seed}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        set_seed(seed)

        # Reload data with new seed
        loader = load_glue_data(
            task=args.glue_task,
            tokenizer=tokenizer,
            n_step=args.n_step,
            batch_size=args.batch_size,
            seed=seed,
            max_length=args.max_seq_length,
        )

        pmob = per_module_out_bound_fn(seed) if per_module_out_bound_fn else None

        for adc_bits in adc_bits_list:
            label   = f"{tag}_seed{seed}_adc{adc_bits}"
            pmr     = per_module_out_res_fn(adc_bits) if per_module_out_res_fn else None
            row     = run_adc_one_glue(
                dac_bits=args.dac_bits,
                adc_bits=adc_bits,
                args=args,
                loader=loader,
                label=label,
                per_module_out_bound=pmob,
                per_module_out_res=pmr,
            )
            row["seed"] = seed
            all_rows.append(row)

    # Compute mean±std per adc_bits
    df_all = pd.DataFrame(all_rows)
    numeric_cols = [c for c in df_all.columns
                    if c not in ("label", "task", "seed") and df_all[c].dtype != object]

    agg_rows = []
    for adc_bits, grp in df_all.groupby("adc_bits"):
        row_mean = {"adc_bits": adc_bits}
        for col in numeric_cols:
            if col == "adc_bits":
                continue
            row_mean[f"{col}_mean"] = grp[col].mean()
            row_mean[f"{col}_std"]  = grp[col].std()
        agg_rows.append(row_mean)

    agg_df = pd.DataFrame(agg_rows)
    seed_csv = os.path.join(OUT_DIR, f"{tag}_seed_sweep_summary.csv")
    agg_df.to_csv(seed_csv, index=False)
    print(f"\nSaved seed sweep summary → {seed_csv}")

    # Plot key metrics with error bars
    _plot_seed_sweep(agg_df, tag, OUT_DIR)

    return df_all


def _plot_seed_sweep(agg_df: pd.DataFrame, tag: str, out_dir: str):
    """Plot seed mean±std for SNR per sublayer and logit KL."""
    if agg_df.empty:
        return

    from diag_fwdio_utils import SUBLAYER_ORDER as _SLO

    # SNR per sublayer with std bands
    fig, ax = plt.subplots(figsize=(9, 5))
    for sl in _SLO:
        mean_col = f"mac_snr_{sl}_mean_mean"
        std_col  = f"mac_snr_{sl}_mean_std"
        if mean_col in agg_df.columns:
            ax.plot(agg_df["adc_bits"], agg_df[mean_col], marker="o", label=sl)
            if std_col in agg_df.columns:
                ax.fill_between(
                    agg_df["adc_bits"],
                    agg_df[mean_col] - agg_df[std_col],
                    agg_df[mean_col] + agg_df[std_col],
                    alpha=0.2,
                )
    ax.set_xlabel("ADC bits")
    ax.set_ylabel("MAC SNR (dB) mean±std")
    ax.set_title(f"{tag} — Seed Sweep SNR")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, f"{tag}_seed_sweep_snr.png"),
                dpi=100, bbox_inches="tight")
    plt.close(fig)

    # KL with error bars
    kl_mean_col = "logit_kl_mean_mean"
    kl_std_col  = "logit_kl_mean_std"
    if kl_mean_col in agg_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            agg_df["adc_bits"], agg_df[kl_mean_col],
            yerr=agg_df[kl_std_col] if kl_std_col in agg_df.columns else None,
            marker="o", capsize=4,
        )
        ax.set_xlabel("ADC bits")
        ax.set_ylabel("Mean KL divergence")
        ax.set_title(f"{tag} — Seed Sweep Logit KL")
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, f"{tag}_seed_sweep_kl.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved seed sweep plots → {out_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    torch.manual_seed(SEED)
    set_seed(SEED)
    np.random.seed(SEED)

    tag = args.tag or "run"
    save_meta_json(args, OUT_DIR, tag, seed=SEED, inp_bound=INP_BOUND, out_bound=OUT_BOUND)

    # Step 1: Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Step 2: Load GLUE data
    loader = load_glue_data(
        task=args.glue_task,
        tokenizer=tokenizer,
        n_step=N_STEP,
        batch_size=BATCH_SIZE,
        seed=SEED,
        max_length=args.max_seq_length,
    )

    # Step 3: Get encoder names (from a dummy digital model)
    task     = args.glue_task
    num_labels = TASK_TO_NUM_LABELS[task]
    always_digital = ["classifier", "pooler"]
    _dummy = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels)
    enc_names = _encoder_linear_names(_dummy, always_digital=always_digital)
    del _dummy
    gc.collect()

    # Step 4: Calibrate out bounds (optional)
    per_module_out_bound = None
    if args.calib_out_bound:
        def _digital_model_fn():
            return AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=num_labels)

        per_module_out_bound = calibrate_out_bounds(
            loader, args, enc_names, DEVICE, _digital_model_fn
        )

        # Acceptance check I2: per_module must have > 1 unique value in per_module mode
        if args.out_bound_grouping == "per_module":
            n_unique = len(set(per_module_out_bound.values()))
            assert n_unique > 1, (
                f"[FAIL I2] per_module calibration produced only {n_unique} unique bound value; "
                f"expected > 1 unique values"
            )
            print(f"  [OK I2] per_module calibration: {n_unique} unique bound values ✓")

        if args.save_calib_table:
            calib_rows = [
                {
                    "module": n,
                    "layer_idx": parse_layer_name(n)[0] if parse_layer_name(n) else -1,
                    "sublayer": parse_layer_name(n)[1] if parse_layer_name(n) else "?",
                    "calibrated_out_bound": b,
                    "baseline_out_bound": OUT_BOUND,
                }
                for n, b in per_module_out_bound.items()
            ]
            calib_csv = os.path.join(OUT_DIR, f"{tag}_calib_table.csv")
            pd.DataFrame(calib_rows).to_csv(calib_csv, index=False)
            print(f"  Saved calib table → {calib_csv}")

    # Step 5: Mixed precision assignment (optional)
    mp_assignment = None
    if args.mixed_precision:
        mp_assignment = compute_mixed_precision_assignment(args, enc_names)

        def per_module_out_res_fn(adc_bits_ignored):
            return {n: 1.0 / (2 ** b - 2) for n, b in mp_assignment.items()}
    else:
        def per_module_out_res_fn(adc_bits):
            return None

    # Step 6: Choose run mode
    if args.seed_sweep:
        adc_list = ([int(x.strip()) for x in args.adc_bits_sweep.split(",")]
                    if args.adc_bits_sweep else [args.adc_bits])
        run_seed_sweep(
            adc_bits_list=adc_list,
            args=args,
            tokenizer=tokenizer,
            tag=tag,
            per_module_out_res_fn=per_module_out_res_fn,
        )

    elif args.adc_bits_sweep:
        adc_list = [int(x.strip()) for x in args.adc_bits_sweep.split(",")]
        print(f"[Sweep] ADC bits: {adc_list}, tag={tag}")
        run_adc_sweep_glue(
            adc_bits_list=adc_list,
            args=args,
            loader=loader,
            tag=tag,
            per_module_out_bound=per_module_out_bound,
            per_module_out_res_fn=per_module_out_res_fn,
        )

    else:
        result = run_adc_one_glue(
            dac_bits=args.dac_bits,
            adc_bits=args.adc_bits,
            args=args,
            loader=loader,
            label=tag,
            per_module_out_bound=per_module_out_bound,
            per_module_out_res=per_module_out_res_fn(args.adc_bits),
        )
        print(f"\n[Result] {result}")


if __name__ == "__main__":
    main()
