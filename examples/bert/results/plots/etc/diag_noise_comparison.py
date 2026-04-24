#!/usr/bin/env python3
"""
LRTT Noise Comparison Experiments
===================================
4 targeted experiments to answer: can LRTT + stochastic exceed 81% F1?

Experiments:
  A  nwd         nwd-best params (trial 1)    -- reference
  B  stochastic  nwd-best params (trial 1)    -- same params, noise on
  C  stoch-nn    nwd-best params (trial 1)    -- same params, std=0/dtd=0 (quant only)
  D  stochastic  stoch-best params (trial 64) -- optuna best for stochastic

Each run: TRAIN_SUBSET_SIZE=8000, 5 epochs (~30 min).

Metrics per epoch:
  - F1 / EM
  - train loss
  - A/B mean weight norm
  - A/B saturation fraction (|w| > 0.95 * w_bound)
  - mean |A@B|_F (transfer signal size)

Usage:
  HF_HUB_DISABLE_XET=1 python diag_noise_comparison.py [--exp A B C D] [--gpu 3]
"""

import os, sys, json, math, argparse, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

# ── point to repo root ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# ── import training infrastructure from optuna script ─────────────────────────
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "optuna_lrtt",
    Path(__file__).parent / "optuna_bert_squad_lrtt.py"
)
_mod = importlib.util.module_from_spec(_spec)
# Pre-set sys.argv so argparse in the module doesn't fire
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(_mod)
sys.argv = _saved_argv

# ── Experiment configurations ──────────────────────────────────────────────────

# nwd best params (from trial 1 on external machine)
NWD_BEST = {
    "learning_rate":       0.009901,
    "transfer_lr":         5587.220,
    "transfer_every":      4,
    "rank_exp":            5,          # rank = 2^5 = 32
    "fast_lr":             0.04823,
    "tau_sec":             0.0,
    "ab_dw_min":           0.001981,
    "ab_desired_bl":       31,
    "out_noise":           0.0,
    "ab_weight_scaling_omega": 0.0,
    "min_lr_rate":         0.0,
}

# stochastic best params (trial 64)
STOCH_BEST = {
    "learning_rate":       0.005293052575077726,
    "transfer_lr":         1.417412811464136,
    "transfer_every":      1,
    "rank_exp":            4,          # rank = 16
    "fast_lr":             11.428507925525787,
    "tau_sec":             0.0,
    "ab_dw_min":           0.013679234045096049,
    "ab_desired_bl":       31,
    "out_noise":           0.0,
    "ab_weight_scaling_omega": 0.0,
    "min_lr_rate":         0.0,
}

EXPERIMENTS = {
    "A": dict(label="A: nwd @ nwd-best",        params=NWD_BEST,   pulse_type="none_with_device", noise=True),
    "B": dict(label="B: stoch @ nwd-best",       params=NWD_BEST,   pulse_type="default",          noise=True),
    "C": dict(label="C: stoch-nn @ nwd-best",    params=NWD_BEST,   pulse_type="default",          noise=False),
    "D": dict(label="D: stoch @ stoch-best",     params=STOCH_BEST, pulse_type="default",          noise=True),
}

RESULT_DIR = Path(__file__).parent / "results" / "optuna_bert_squad_lrtt"
TRAIN_SUBSET = 8000
N_EPOCHS     = 5

# ── Helpers ────────────────────────────────────────────────────────────────────

def set_globals(pulse_type, noise):
    """Configure module-level globals in the imported training module.
    Settings match the original bs48_adam_hybrid_nowd_nomom_nonest_set_linearstep_fwinj_perfect_noos_fica_qkvo study."""
    # Match run_enqueued.py / original experiment settings exactly
    _mod.AB_DEVICE          = "linearstep"
    _mod.BATCH_SIZE         = 48
    _mod.GRAD_ACCUM_STEPS   = 1
    _mod.TRAIN_SUBSET_SIZE  = TRAIN_SUBSET
    _mod.N_EPOCHS           = N_EPOCHS
    _mod.WARMUP_STEPS       = 365
    _mod.TRANSFER_METHOD    = "set"
    _mod.IO_NOISE           = False
    _mod.FORWARD_INJECT     = True
    _mod.IS_PERFECT         = True
    _mod.NO_QUANT           = False
    _mod.LORA_TARGET        = "qkvo"
    _mod.HEAD_LAYER         = False
    _mod.ENCODER_ANALOG     = True
    _mod.HEAD_ANALOG        = True
    _mod.BACKWARD_OUT_BOUND = 12.0
    _mod.REINIT_GAIN        = 0.01
    _mod.SEED               = 42
    _mod.DYNAMIC_TE         = False
    _mod.TE_WARMUP_STEPS    = 0
    _mod.TE_WARMUP_SCHEDULE = []

    _mod.OPT_CONFIG['ab_pulse_type']               = pulse_type
    _mod.OPT_CONFIG['scale_transfer_lr']           = True
    _mod.OPT_CONFIG['auto_scale_mode']             = 'none'
    _mod.OPT_CONFIG['correct_gradient_magnitudes'] = False
    _mod.OPT_CONFIG['no_adc_ab_proj']              = False
    _mod.OPT_CONFIG['learn_out_scaling']           = False
    _mod.OPT_CONFIG['optimizer']                   = 'AnalogAdam'
    _mod.OPT_CONFIG['reinit_mode']                 = 'hybrid'
    _mod.OPT_CONFIG['tune_wd']                     = False
    _mod.OPT_CONFIG['tune_momentum']               = False
    _mod.OPT_CONFIG['tune_nesterov']               = False
    _mod.OPT_CONFIG['no_transfer']                 = False
    _mod.OPT_CONFIG['fi_continuous_alpha']         = False

    if not noise:
        # Monkey-patch _create_ab_device to return noise-free LinearStepDevice
        from aihwkit.simulator.configs.devices import LinearStepDevice
        orig_dw_min = [None]
        def _create_ab_nonoise(tau_sec=0.0, dw_min=0.001981):
            return LinearStepDevice(
                dw_min=dw_min,
                dw_min_std=0.0,
                dw_min_dtod=0.0,
            )
        _mod._create_ab_device = _create_ab_nonoise
    else:
        # Restore original (LinearStepDevice with default dw_min_std/dtod=0.3)
        def _create_ab_default(tau_sec=0.0, dw_min=0.001981):
            from aihwkit.simulator.configs.devices import LinearStepDevice
            return LinearStepDevice(dw_min=dw_min)
        _mod._create_ab_device = _create_ab_default


def build_params(cfg):
    """Build full params dict expected by training code."""
    p = dict(cfg["params"])
    # Keep rank_exp — objective() calls trial.suggest_int("rank_exp", ...)
    p["reinit_mode"] = "hybrid"
    p["weight_decay"]= 0.0
    p["momentum"]    = 0.0
    p["nesterov"]    = False
    p["optimizer"]   = "AnalogAdam"
    p["c_dw_min"]    = 0.001
    p["c_desired_bl"]= 31
    p["lora_alpha"]  = 1.0
    return p


class FixedTrial:
    """Mock optuna trial that returns fixed params and records intermediate F1."""
    def __init__(self, params, number=0):
        self._p = params
        self.number = number
        self._intermediate = {}
        self._user_attrs   = {}

    def suggest_float(self, name, *a, **kw):   return float(self._p[name])
    def suggest_int(self,   name, *a, **kw):   return int(self._p[name])
    def suggest_categorical(self, name, choices, **kw): return self._p[name]

    def report(self, value, step):
        self._intermediate[step] = value

    def set_user_attr(self, key, value):
        self._user_attrs[key] = value

    def should_prune(self):
        return False


def collect_ab_stats(model):
    """
    Walk LRTT analog tiles and collect per-epoch weight statistics.

    Returns:
      ab_norm        : mean Frobenius norm of A and B tiles
      ab_sat         : fraction of A/B elements at |w| > 0.95 * w_bound
      transfer_size  : mean ||A@B||_F  (signal being transferred to C)
      c_norm         : mean Frobenius norm of C tile
    """
    from aihwkit.nn import AnalogLinear

    ab_norms, ab_sats, transfer_sizes, c_norms = [], [], [], []
    W_BOUND = 0.6

    def to_np(x):
        if hasattr(x, 'detach'):
            return x.detach().cpu().float().numpy()
        return np.array(x, dtype=np.float32)

    for module in model.modules():
        if not isinstance(module, AnalogLinear):
            continue
        # TileModuleArray -> ModuleList -> LRTTSimulatorTile (has tile_a/tile_b/tile_c)
        for _, sub in module.analog_module.named_modules():
            if not (hasattr(sub, 'tile_a') and hasattr(sub, 'tile_b') and hasattr(sub, 'tile_c')):
                continue
            try:
                A_np = to_np(sub.tile_a.get_weights()[0])
                B_np = to_np(sub.tile_b.get_weights()[0])
                C_np = to_np(sub.tile_c.get_weights()[0])
                for w in (A_np, B_np):
                    ab_norms.append(np.linalg.norm(w, 'fro'))
                    ab_sats.append(float(np.mean(np.abs(w) > 0.95 * W_BOUND)))
                transfer_sizes.append(np.linalg.norm(A_np @ B_np, 'fro'))
                c_norms.append(np.linalg.norm(C_np, 'fro'))
            except Exception:
                pass

    return {
        "ab_norm":       float(np.mean(ab_norms))       if ab_norms       else float('nan'),
        "ab_sat":        float(np.mean(ab_sats))         if ab_sats        else float('nan'),
        "transfer_size": float(np.mean(transfer_sizes))  if transfer_sizes  else float('nan'),
        "c_norm":        float(np.mean(c_norms))         if c_norms        else float('nan'),
    }


def collect_grad_stats(model, ab_dw_min, fast_lr):
    """
    Collect gradient norm statistics for A/B tiles immediately after backward.
    Also computes expected n_pulses = round(||g||_elem * fast_lr / dw_min).

    Must be called inside the training loop after loss.backward() and before
    optimizer.step().  Returns None if no gradients found.
    """
    from aihwkit.nn import AnalogLinear

    elem_norms = []
    for module in model.modules():
        if not isinstance(module, AnalogLinear):
            continue
        for p in module.parameters():
            if p.grad is not None:
                # per-element RMS of gradient
                elem_norms.append(float(p.grad.detach().abs().mean().cpu()))

    if not elem_norms:
        return None

    mean_elem = float(np.mean(elem_norms))
    n_pulses_est = min(round(mean_elem * fast_lr / max(ab_dw_min, 1e-9)), 31)
    return {
        "grad_elem_mean": mean_elem,
        "n_pulses_est":   n_pulses_est,
    }


def run_experiment(exp_id, cfg, device_str, train_loader, eval_features, eval_examples, tokenizer):
    """Run one experiment, return per-epoch metrics dict."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {exp_id}: {cfg['label']}")
    print(f"  pulse_type={cfg['pulse_type']}  noise={cfg['noise']}")
    print(f"  params={cfg['params']}")
    print(f"{'='*70}")

    set_globals(cfg["pulse_type"], cfg["noise"])
    _mod.DEVICE = torch.device(device_str)

    params = build_params(cfg)
    trial  = FixedTrial(params, number=ord(exp_id) - ord('A'))

    # Patch objective to also collect AB stats per epoch
    per_epoch = {"f1": [], "loss": [],
                 "ab_norm": [], "ab_sat": [], "transfer_size": [],
                 "c_norm": [], "grad_elem_mean": [], "n_pulses_est": []}

    # We call the objective function but intercept per-epoch logging
    # by monkey-patching trial.report to also collect stats
    _model_ref = [None]
    _orig_train = _mod.train_epoch if hasattr(_mod, 'train_epoch') else None

    ab_dw_min = cfg["params"]["ab_dw_min"]
    fast_lr   = cfg["params"]["fast_lr"]

    orig_report = trial.report
    def patched_report(value, step):
        orig_report(value, step)
        per_epoch["f1"].append(value)
        if _model_ref[0] is not None:
            stats = collect_ab_stats(_model_ref[0])
            per_epoch["ab_norm"].append(stats["ab_norm"])
            per_epoch["ab_sat"].append(stats["ab_sat"])
            per_epoch["transfer_size"].append(stats["transfer_size"])
            per_epoch["c_norm"].append(stats["c_norm"])
            # grad stats stored via user_attr from inside training loop
            if _grad_accum:
                mean_g = float(np.mean(_grad_accum))
                per_epoch["grad_elem_mean"].append(mean_g)
                np_est = min(round(mean_g * fast_lr / max(ab_dw_min, 1e-9)), 31)
                per_epoch["n_pulses_est"].append(np_est)
                _grad_accum.clear()
        loss_key = f"train_loss_epoch_{step}"
        if loss_key in trial._user_attrs:
            per_epoch["loss"].append(trial._user_attrs[loss_key])
    trial.report = patched_report

    # Accumulate gradient element norms across batches within each epoch
    _grad_accum = []

    def _make_grad_hook(name):
        def hook(grad):
            if grad is not None:
                _grad_accum.append(float(grad.detach().abs().mean().cpu()))
        return hook

    # Patch create_model to capture model reference and register grad hooks
    _orig_create_model = _mod.create_model
    def patched_create_model(p):
        from aihwkit.nn import AnalogLinear
        m = _orig_create_model(p)
        _model_ref[0] = m
        # Register gradient hooks on AnalogLinear parameters that require grad
        for mod in m.modules():
            if isinstance(mod, AnalogLinear):
                for param in mod.parameters():
                    if param.requires_grad:
                        param.register_hook(_make_grad_hook(None))
        return m
    _mod.create_model = patched_create_model

    try:
        best_f1 = _mod.objective(trial, train_loader, eval_features, eval_examples, tokenizer)
    except Exception as e:
        import traceback
        print(f"[ERROR] Experiment {exp_id} failed: {e}")
        traceback.print_exc()
        best_f1 = 0.0
    finally:
        _mod.create_model = _orig_create_model
        gc.collect()
        torch.cuda.empty_cache()

    # Fill loss from user_attrs if not captured via report
    if not per_epoch["loss"]:
        for ep in range(1, N_EPOCHS + 1):
            k = f"train_loss_epoch_{ep}"
            if k in trial._user_attrs:
                per_epoch["loss"].append(trial._user_attrs[k])

    return {
        "exp_id":        exp_id,
        "label":         cfg["label"],
        "pulse_type":    cfg["pulse_type"],
        "noise":         cfg["noise"],
        "params":        cfg["params"],
        "best_f1":       best_f1,
        "per_epoch":     per_epoch,
        "intermediate":  dict(trial._intermediate),
    }


# ── Visualization ──────────────────────────────────────────────────────────────

COLORS = {"A": "green", "B": "crimson", "C": "purple", "D": "darkorange"}

def make_plots(results, save_path):
    ids = list(results.keys())
    n   = len(ids)
    epochs = list(range(1, N_EPOCHS + 1))

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(
        "LRTT Noise Comparison: nwd vs stochastic vs stoch-no-noise\n"
        f"[LinearStepDevice, BERT SQuAD, bs=48, {N_EPOCHS} epochs, "
        f"TRAIN_SUBSET={TRAIN_SUBSET}]",
        fontsize=11
    )

    def _epochs_x(data_list):
        return list(range(1, len(data_list) + 1))

    # ── 1. F1 per epoch ────────────────────────────────────────────────────────
    ax = axes[0, 0]
    for eid, res in results.items():
        f1s = res["per_epoch"]["f1"]
        if f1s:
            ax.plot(_epochs_x(f1s), f1s, 'o-', color=COLORS[eid],
                    label=res["label"], linewidth=1.8, markersize=5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 (%)")
    ax.set_title("F1 per epoch")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── 2. Train loss per epoch ────────────────────────────────────────────────
    ax = axes[0, 1]
    for eid, res in results.items():
        ls = res["per_epoch"]["loss"]
        if ls:
            ax.plot(_epochs_x(ls), ls, 'o-', color=COLORS[eid],
                    label=res["label"], linewidth=1.8, markersize=5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train loss")
    ax.set_title("Training loss per epoch")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── 3. Final F1 bar chart ──────────────────────────────────────────────────
    ax = axes[0, 2]
    bar_ids  = [eid for eid in ids if results[eid]["best_f1"] > 0]
    bar_vals = [results[eid]["best_f1"] for eid in bar_ids]
    bars = ax.bar(bar_ids, bar_vals,
                  color=[COLORS[e] for e in bar_ids], alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val:.2f}%", ha='center', va='bottom', fontsize=9)
    ax.set_ylabel("Best F1 (%)"); ax.set_title("Best F1 comparison")
    ax.set_ylim(bottom=max(0, min(bar_vals) - 5) if bar_vals else 0)
    ax.grid(True, alpha=0.3, axis='y')

    # ── 4. A/B weight norm ────────────────────────────────────────────────────
    ax = axes[1, 0]
    has_norm = False
    for eid, res in results.items():
        ns = res["per_epoch"]["ab_norm"]
        if ns and not all(math.isnan(v) for v in ns):
            ax.plot(_epochs_x(ns), ns, 'o-', color=COLORS[eid],
                    label=res["label"], linewidth=1.8, markersize=5)
            has_norm = True
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean A/B weight norm (Frobenius)")
    ax.set_title("A/B tile weight norm\n(indicates saturation trend)")
    if has_norm: ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 5. Saturation fraction ────────────────────────────────────────────────
    ax = axes[1, 1]
    has_sat = False
    for eid, res in results.items():
        ss = res["per_epoch"]["ab_sat"]
        if ss and not all(math.isnan(v) for v in ss):
            ax.plot(_epochs_x(ss), [v * 100 for v in ss], 'o-',
                    color=COLORS[eid], label=res["label"], linewidth=1.8, markersize=5)
            has_sat = True
    ax.set_xlabel("Epoch"); ax.set_ylabel("Saturation fraction (%)")
    ax.set_title("A/B saturation: % weights at |w| > 0.95 * w_bound\n"
                 "(high saturation = transfer signal clipped)")
    ax.set_ylim(0, 100)
    if has_sat: ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 6. Transfer size ──────────────────────────────────────────────────────
    ax = axes[1, 2]
    has_tr = False
    for eid, res in results.items():
        ts = res["per_epoch"]["transfer_size"]
        if ts and not all(math.isnan(v) for v in ts):
            ax.plot(_epochs_x(ts), ts, 'o-', color=COLORS[eid],
                    label=res["label"], linewidth=1.8, markersize=5)
            has_tr = True
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean |A@B|_F (transfer signal)")
    ax.set_title("Transfer signal per layer\n"
                 "(how much info each transfer moves into C)")
    if has_tr: ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 7. C tile norm ────────────────────────────────────────────────────────
    ax = axes[2, 0]
    has_c = False
    for eid, res in results.items():
        cs = res["per_epoch"]["c_norm"]
        if cs and not all(math.isnan(v) for v in cs):
            ax.plot(_epochs_x(cs), cs, 'o-', color=COLORS[eid],
                    label=res["label"], linewidth=1.8, markersize=5)
            has_c = True
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean C tile weight norm (Frobenius)")
    ax.set_title("C tile norm\n"
                 "(should grow steadily — erratic = noisy transfer)")
    if has_c: ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 8. Gradient element mean ──────────────────────────────────────────────
    ax = axes[2, 1]
    has_g = False
    for eid, res in results.items():
        gs = res["per_epoch"]["grad_elem_mean"]
        if gs and not all(math.isnan(v) for v in gs):
            ax.plot(_epochs_x(gs), gs, 'o-', color=COLORS[eid],
                    label=res["label"], linewidth=1.8, markersize=5)
            has_g = True
    # Draw dw_min / fast_lr threshold lines
    for eid, res in results.items():
        if res["per_epoch"]["grad_elem_mean"]:
            dw  = res["params"]["ab_dw_min"]
            flr = res["params"]["fast_lr"]
            thresh = dw / max(flr, 1e-9)
            ax.axhline(thresh, color=COLORS[eid], linestyle='--', linewidth=1,
                       alpha=0.6, label=f'{eid} 1-pulse threshold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean |grad| per element")
    ax.set_title("Gradient magnitude vs A/B tiles\n"
                 "(dashed = dw_min/fast_lr: 1-pulse threshold;\n"
                 " below this line = most elements get 0 pulses)")
    if has_g: ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── 9. Estimated n_pulses ─────────────────────────────────────────────────
    ax = axes[2, 2]
    has_np = False
    for eid, res in results.items():
        nps = res["per_epoch"]["n_pulses_est"]
        if nps:
            ax.plot(_epochs_x(nps), nps, 'o-', color=COLORS[eid],
                    label=res["label"], linewidth=1.8, markersize=5)
            has_np = True
    ax.axhline(31, color='gray', linestyle='--', linewidth=1, label='BL=31 (saturated)')
    ax.axhline(1,  color='gray', linestyle=':',  linewidth=1, label='1 pulse (minimum)')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Estimated n_pulses per step")
    ax.set_title("Estimated pulses per weight update\n"
                 "= round(|grad|_elem * fast_lr / dw_min)\n"
                 "optimal: 5-20 pulses")
    ax.set_ylim(0, 35)
    if has_np: ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved -> {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global TRAIN_SUBSET, N_EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',  nargs='+', default=list(EXPERIMENTS.keys()),
                        choices=list(EXPERIMENTS.keys()),
                        help='Which experiments to run (default: all)')
    parser.add_argument('--gpu',  default='0', help='GPU index (default: 0)')
    parser.add_argument('--subset', type=int, default=TRAIN_SUBSET,
                        help=f'Training subset size (default: {TRAIN_SUBSET})')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS,
                        help=f'Number of epochs (default: {N_EPOCHS})')
    parser.add_argument('--single', metavar='EXP_ID',
                        help='Run a single experiment in-process (used internally by subprocess mode)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip experiments that already have valid results in JSON')
    args = parser.parse_args()

    TRAIN_SUBSET = args.subset
    N_EPOCHS     = args.epochs

    results_path = RESULT_DIR / "diag_noise_comparison_results.json"
    plot_path    = RESULT_DIR / "diag_noise_comparison.png"

    # ── Single-experiment mode (called by subprocess) ────────────────────────
    if args.single:
        exp_id    = args.single
        device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        print(f"[subprocess] Device: {device_str}, Exp: {exp_id}")

        _mod.DEVICE = torch.device(device_str)
        set_globals("default", True)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(_mod.MODEL_NAME)
        train_loader, eval_features, eval_examples = _mod.load_data(tokenizer)

        cfg = EXPERIMENTS[exp_id]
        res = run_experiment(exp_id, cfg, device_str,
                             train_loader, eval_features, eval_examples, tokenizer)

        # Load existing results, merge, save
        all_results = {}
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)
        all_results[exp_id] = res
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved results to {results_path}")

        # Print mini-summary for this experiment
        f1s = res["per_epoch"]["f1"]
        print(f"  {exp_id}: best_f1={res['best_f1']:.4f}%  per_epoch_f1={[f'{v:.1f}' for v in f1s]}")
        return

    # ── Orchestrator mode: spawn one subprocess per experiment ──────────────
    import subprocess as sp

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_str}")
    print(f"Experiments: {args.exp}")
    print(f"TRAIN_SUBSET={TRAIN_SUBSET}, N_EPOCHS={N_EPOCHS}")
    print("(Each experiment runs in its own subprocess for clean GPU memory)\n")

    # Load existing results if any (for --skip-existing)
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"Loaded existing results: {list(all_results.keys())}")

    script_path = str(Path(__file__).resolve())
    env = os.environ.copy()

    for exp_id in args.exp:
        if args.skip_existing and exp_id in all_results and all_results[exp_id].get("best_f1", 0) > 0:
            print(f"Skipping {exp_id} (already done, F1={all_results[exp_id]['best_f1']:.2f}%)")
            continue

        print(f"\n{'='*70}")
        print(f"Launching subprocess for experiment {exp_id}...")
        print(f"{'='*70}")

        cmd = [
            sys.executable, script_path,
            '--single', exp_id,
            '--gpu', args.gpu,
            '--subset', str(TRAIN_SUBSET),
            '--epochs', str(N_EPOCHS),
        ]
        ret = sp.run(cmd, env=env)
        if ret.returncode != 0:
            print(f"[WARN] Subprocess for {exp_id} exited with code {ret.returncode}")

        # Reload results after each subprocess
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)

        # Regenerate plot after each finished experiment
        if all_results:
            make_plots(all_results, plot_path)

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    for eid in sorted(all_results.keys()):
        res = all_results[eid]
        f1s = res["per_epoch"]["f1"]
        print(f"  {eid}: {res['label']}")
        print(f"      best_f1={res['best_f1']:.4f}%  "
              f"per_epoch_f1={[f'{v:.1f}' for v in f1s]}")


if __name__ == "__main__":
    main()
