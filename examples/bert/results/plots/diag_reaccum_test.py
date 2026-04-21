#!/usr/bin/env python3
"""
Per-step A/B dynamics test: re-accumulation after reset.

Runs 1 epoch with a small subset and logs A/B norm, n_pulses, transfer_size
every LOG_EVERY steps to visualise how nwd vs stochastic behave between transfers.

Usage:
  CUDA_VISIBLE_DEVICES=0 python diag_reaccum_test.py [--gpu 0] [--exp A B D]
  # Single experiment (called internally by subprocess mode):
  CUDA_VISIBLE_DEVICES=0 python diag_reaccum_test.py --single A --gpu 0
"""

import os, sys, json, argparse, gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
import subprocess as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "optuna_lrtt", Path(__file__).parent / "optuna_bert_squad_lrtt.py"
)
_mod = importlib.util.module_from_spec(_spec)
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(_mod)
sys.argv = _saved_argv

# ── Experiment configs (reuse from diag_noise_comparison.py) ──────────────────
NWD_BEST = {
    "learning_rate":           0.009901,
    "transfer_lr":             5587.220,
    "transfer_every":          4,
    "rank_exp":                5,
    "fast_lr":                 0.04823,
    "tau_sec":                 0.0,
    "ab_dw_min":               0.001981,
    "ab_desired_bl":           31,
    "out_noise":               0.0,
    "ab_weight_scaling_omega": 0.0,
    "min_lr_rate":             0.0,
}
STOCH_BEST = {
    "learning_rate":           0.005293052575077726,
    "transfer_lr":             1.417412811464136,
    "transfer_every":          1,
    "rank_exp":                4,
    "fast_lr":                 11.428507925525787,
    "tau_sec":                 0.0,
    "ab_dw_min":               0.013679234045096049,
    "ab_desired_bl":           31,
    "out_noise":               0.0,
    "ab_weight_scaling_omega": 0.0,
    "min_lr_rate":             0.0,
}

EXPERIMENTS = {
    "A": dict(label="A: nwd @ nwd-best",      params=NWD_BEST,   pulse_type="none_with_device", noise=True),
    "B": dict(label="B: stoch @ nwd-best",     params=NWD_BEST,   pulse_type="default",          noise=True),
    "D": dict(label="D: stoch @ stoch-best",   params=STOCH_BEST, pulse_type="default",          noise=True),
}

RESULT_DIR   = Path(__file__).parent / "results" / "optuna_bert_squad_lrtt"
TRAIN_SUBSET = 1000
N_EPOCHS     = 1
LOG_EVERY    = 5      # collect stats every N steps

COLORS = {"A": "#2196F3", "B": "#F44336", "D": "#FF9800"}


# ── Helpers (identical to diag_noise_comparison.py) ───────────────────────────

def set_globals(pulse_type, noise):
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
        def _create_ab_nonoise(tau_sec=0.0, dw_min=0.001981):
            from aihwkit.simulator.configs.devices import LinearStepDevice
            return LinearStepDevice(dw_min=dw_min, dw_min_std=0.0, dw_min_dtod=0.0)
        _mod._create_ab_device = _create_ab_nonoise
    else:
        def _create_ab_default(tau_sec=0.0, dw_min=0.001981):
            from aihwkit.simulator.configs.devices import LinearStepDevice
            return LinearStepDevice(dw_min=dw_min)
        _mod._create_ab_device = _create_ab_default


def build_params(cfg):
    p = dict(cfg["params"])
    p["reinit_mode"] = "hybrid"
    p["weight_decay"] = 0.0
    p["momentum"]     = 0.0
    p["nesterov"]     = False
    p["optimizer"]    = "AnalogAdam"
    p["c_dw_min"]     = 0.001
    p["c_desired_bl"] = 31
    p["lora_alpha"]   = 1.0
    return p


class FixedTrial:
    def __init__(self, params, number=0):
        self._p = params
        self.number = number
        self._intermediate = {}
        self._user_attrs   = {}
    def suggest_float(self, name, *a, **kw):         return float(self._p[name])
    def suggest_int(self,   name, *a, **kw):         return int(self._p[name])
    def suggest_categorical(self, name, choices, **kw): return self._p[name]
    def report(self, value, step):                   self._intermediate[step] = value
    def set_user_attr(self, key, value):             self._user_attrs[key] = value
    def should_prune(self):                          return False


def collect_ab_stats(model):
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
        "ab_norm":       float(np.mean(ab_norms))      if ab_norms      else float('nan'),
        "ab_sat":        float(np.mean(ab_sats))        if ab_sats       else float('nan'),
        "transfer_size": float(np.mean(transfer_sizes)) if transfer_sizes else float('nan'),
        "c_norm":        float(np.mean(c_norms))        if c_norms       else float('nan'),
    }


# ── Per-step callback ─────────────────────────────────────────────────────────

class StepStatsCallback:
    """HuggingFace TrainerCallback that records per-step AB metrics."""

    def __init__(self, model_ref_list, ab_dw_min, fast_lr, log_every=LOG_EVERY):
        self._model_ref  = model_ref_list   # [model] set after model creation
        self._dw_min     = ab_dw_min
        self._fast_lr    = fast_lr
        self._log_every  = log_every
        self.steps       = []
        self.ab_norms    = []
        self.ab_sats     = []
        self.transfer_sizes = []
        self.c_norms     = []
        self.n_pulses_est = []
        self._grad_buf   = []

    # Called after each optimizer step
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step % self._log_every != 0:
            return
        model = self._model_ref[0]
        if model is None:
            return
        stats = collect_ab_stats(model)
        # With AnalogAdam the effective update is ±fast_lr (gradient magnitude
        # is normalised away), so n_pulses = round(fast_lr / dw_min), constant.
        np_est = min(31, round(self._fast_lr / max(self._dw_min, 1e-9)))
        self.steps.append(step)
        self.ab_norms.append(stats["ab_norm"])
        self.ab_sats.append(stats["ab_sat"])
        self.transfer_sizes.append(stats["transfer_size"])
        self.c_norms.append(stats["c_norm"])
        self.n_pulses_est.append(np_est)

    def to_dict(self):
        return {
            "steps":          self.steps,
            "ab_norm":        self.ab_norms,
            "ab_sat":         self.ab_sats,
            "transfer_size":  self.transfer_sizes,
            "c_norm":         self.c_norms,
            "n_pulses_est":   self.n_pulses_est,
        }


def run_experiment(exp_id, cfg, device_str, train_loader, eval_features, eval_examples, tokenizer):
    print(f"\n{'='*60}\nEXP {exp_id}: {cfg['label']}\n{'='*60}")
    set_globals(cfg["pulse_type"], cfg["noise"])
    _mod.DEVICE = torch.device(device_str)

    params = build_params(cfg)
    trial  = FixedTrial(params, number=ord(exp_id) - ord('A'))

    model_ref = [None]
    cb = StepStatsCallback(model_ref, cfg["params"]["ab_dw_min"], cfg["params"]["fast_lr"])

    # Patch create_model to capture reference
    _orig_create_model = _mod.create_model
    def patched_create_model(p):
        m = _orig_create_model(p)
        model_ref[0] = m
        return m
    _mod.create_model = patched_create_model

    # Inject callback into Trainer via monkey-patch
    try:
        from transformers import TrainerCallback
    except ImportError:
        TrainerCallback = object

    class _CB(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            cb.on_step_end(args, state, control, **kwargs)

    _orig_trainer_init = None
    try:
        import transformers
        _OrigTrainer = transformers.Trainer
        class _PatchedTrainer(_OrigTrainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.add_callback(_CB())
        transformers.Trainer = _PatchedTrainer
    except Exception as e:
        print(f"[WARN] Could not patch Trainer: {e}")

    best_f1 = 0.0
    try:
        best_f1 = _mod.objective(trial, train_loader, eval_features, eval_examples, tokenizer)
    except Exception as e:
        import traceback
        print(f"[ERROR] Exp {exp_id}: {e}")
        traceback.print_exc()
    finally:
        _mod.create_model = _orig_create_model
        try:
            transformers.Trainer = _OrigTrainer
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    print(f"  best_f1={best_f1:.2f}%  steps_logged={len(cb.steps)}")
    return {
        "exp_id":    exp_id,
        "label":     cfg["label"],
        "best_f1":   best_f1,
        "params":    cfg["params"],
        "per_step":  cb.to_dict(),
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def make_plots(results, save_path, transfer_every_map):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "A/B Re-accumulation Dynamics (per step)\n"
        f"subset={TRAIN_SUBSET}, 1 epoch, logged every {LOG_EVERY} steps",
        fontsize=12
    )

    def _plot(ax, key, ylabel, title, yscale='linear', hlines=None):
        for eid, res in results.items():
            ps = res["per_step"]
            if ps["steps"] and ps[key]:
                ax.plot(ps["steps"], ps[key], color=COLORS[eid],
                        lw=1.8, label=res["label"], marker='o', ms=3)
            # mark transfer events
            te = transfer_every_map.get(eid, 4)
            max_step = max(ps["steps"]) if ps["steps"] else 0
            for t in range(te, max_step + 1, te):
                ax.axvline(t, color=COLORS[eid], ls=':', lw=0.7, alpha=0.5)
        if hlines:
            for y, ls, lbl in hlines:
                ax.axhline(y, color='gray', ls=ls, lw=1.2, label=lbl)
        ax.set_yscale(yscale)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], "ab_norm", "mean ||A||_F / ||B||_F", "A/B weight norm per step")
    _plot(axes[0, 1], "transfer_size", "mean ||A@B||_F", "Transfer signal magnitude\n(dotted lines = transfer events)")
    _plot(axes[0, 2], "c_norm", "mean ||C||_F", "C tile norm (slow accumulation)")
    _plot(axes[1, 0], "ab_sat", "saturation fraction", "A/B saturation (|w|>0.95*w_bound)",
          hlines=[(0.0, '--', '0%'), (1.0, '--', '100%')])
    _plot(axes[1, 1], "n_pulses_est", "est. n_pulses",
          "Estimated n_pulses\n= round(ab_norm * fast_lr / dw_min)",
          hlines=[(31, '--', 'BL=31 saturated'), (1, ':', 'n=1 min')])
    axes[1, 1].set_ylim(0, 36)

    # Final F1 bar
    ax = axes[1, 2]
    bar_ids  = [eid for eid in results if results[eid]["best_f1"] > 0]
    bar_vals = [results[eid]["best_f1"] for eid in bar_ids]
    if bar_vals:
        bars = ax.bar(bar_ids, bar_vals, color=[COLORS[e] for e in bar_ids],
                      alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{val:.1f}%", ha='center', va='bottom', fontsize=9)
        ax.set_ylim(bottom=max(0, min(bar_vals) - 5))
    ax.set_ylabel("F1 (%)"); ax.set_title("Final F1"); ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved -> {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global TRAIN_SUBSET, N_EPOCHS

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',    nargs='+', default=list(EXPERIMENTS.keys()),
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--gpu',    default='0')
    parser.add_argument('--subset', type=int, default=TRAIN_SUBSET)
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--single', metavar='EXP_ID')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    TRAIN_SUBSET = args.subset
    N_EPOCHS     = args.epochs

    results_path = RESULT_DIR / "diag_reaccum_test_results.json"
    plot_path    = RESULT_DIR / "diag_reaccum_test.png"

    # ── Single mode ──────────────────────────────────────────────────────────
    if args.single:
        exp_id     = args.single
        device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        _mod.DEVICE = torch.device(device_str)
        set_globals("default", True)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(_mod.MODEL_NAME)
        train_loader, eval_features, eval_examples = _mod.load_data(tokenizer)

        cfg = EXPERIMENTS[exp_id]
        res = run_experiment(exp_id, cfg, device_str,
                             train_loader, eval_features, eval_examples, tokenizer)

        all_results = {}
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)
        all_results[exp_id] = res
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved -> {results_path}")
        return

    # ── Orchestrator mode ─────────────────────────────────────────────────────
    all_results = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)

    script_path = str(Path(__file__).resolve())
    env = os.environ.copy()

    for exp_id in args.exp:
        if args.skip_existing and exp_id in all_results and all_results[exp_id].get("best_f1", 0) > 0:
            print(f"Skipping {exp_id}")
            continue
        print(f"\nLaunching subprocess for {exp_id}...")
        cmd = [sys.executable, script_path,
               '--single', exp_id, '--gpu', args.gpu,
               '--subset', str(TRAIN_SUBSET), '--epochs', str(N_EPOCHS)]
        ret = sp.run(cmd, env=env)
        if ret.returncode != 0:
            print(f"[WARN] Subprocess {exp_id} exited {ret.returncode}")
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)

    if all_results:
        te_map = {eid: EXPERIMENTS[eid]["params"]["transfer_every"]
                  for eid in all_results if eid in EXPERIMENTS}
        make_plots(all_results, plot_path, te_map)

    print("\n=== SUMMARY ===")
    for eid in sorted(all_results):
        r = all_results[eid]
        steps = r["per_step"]["steps"]
        print(f"  {eid}: {r['label']}  best_f1={r['best_f1']:.2f}%  "
              f"steps_logged={len(steps)}")


if __name__ == "__main__":
    main()
