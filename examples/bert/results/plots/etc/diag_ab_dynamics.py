#!/usr/bin/env python3
"""
Diagnostic: compare nwd vs stochastic A/B update dynamics.

Runs through warmup silently, then observes N_OBS steps after warmup.
Only monitors first (L0) and last (L11) transformer blocks, grouped by
target (q/k/v/o). Records per-group (averages over tile splits within group).

Logs per step per group: loss, ||A||, ||B||, ||A@B||, ||C||, delta_A/B.

Usage:
    CUDA_VISIBLE_DEVICES=1 python diag_ab_dynamics.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import optuna_bert_squad_lrtt as O

# ── Config ────────────────────────────────────────────────────────────────────
WARMUP_STEPS = 365   # run silently through warmup
N_OBS        = 504   # steps to observe after warmup (stop after this)
LOG_EVERY    = 10    # log every N steps in observation window
LOG_START    = 4     # first log at this obs_step (4, 14, 24, ..., 504)

# nwd-best params from constantstepideal study (trial 100)
FIXED_PARAMS = {
    'learning_rate':            0.009901,
    'transfer_lr':              5587.0,
    'transfer_every':           99999,    # no transfer during diagnostic
    'rank_exp':                 5,        # rank = 32
    'fast_lr':                  0.04823,
    'tau_sec':                  0.0,
    'ab_dw_min':                2 / 2**14,  # 14b constantstepideal
    'ab_desired_bl':            31,
    'out_noise':                0.0,
    'ab_weight_scaling_omega':  0.0,
    'min_lr_rate':              0.0,
}

# ── Experiment globals ────────────────────────────────────────────────────────
O.BATCH_SIZE        = 48
O.GRAD_ACCUM_STEPS  = 1
O.N_EPOCHS          = 1
O.WARMUP_STEPS      = WARMUP_STEPS
O.TRANSFER_METHOD   = "set"
O.AB_DEVICE         = "constantstepideal"
O.C_DEVICE          = "constantstepideal"
O.IO_NOISE          = False
O.FORWARD_INJECT    = True
O.IS_PERFECT        = True
O.NO_QUANT          = False
O.LORA_TARGET       = "qkvo"
O.HEAD_LAYER        = "freeze"
O.ENCODER_ANALOG    = False
O.HEAD_ANALOG       = False
O.BACKWARD_OUT_BOUND = 12.0
O.REINIT_GAIN       = 0.01
O.SEED              = 42
O.TRAIN_SUBSET_SIZE = 0
O.DYNAMIC_TE        = False
O.DYNAMIC_TE_POWER  = 1.0
O.TE_WARMUP_SCHEDULE = []
O.TE_WARMUP_STEPS   = 0

BASE_OPT_CONFIG = {
    'optimizer':                  'AnalogAdam',
    'reinit_mode':                'hybrid',
    'tune_wd':                    False,
    'tune_momentum':              False,
    'tune_nesterov':              False,
    'no_transfer':                False,
    'learn_out_scaling':          False,
    'correct_gradient_magnitudes': False,
    'no_adc_ab_proj':             False,
    'auto_scale_mode':            'none',
    'scale_transfer_lr':          True,
    'fi_continuous_alpha':        True,
    'transfer_rank_schedule':     'all',
    'transfer_ranks_per_step':    1,
}

# Groups to monitor: (layer_idx, target_substring)
# Layer 0 = first block, layer 11 = last block
MONITOR_GROUPS = {
    'L0_q':  (0,  'query'),
    'L0_k':  (0,  'key'),
    'L0_v':  (0,  'value'),
    'L0_o':  (0,  'output'),
    'L11_q': (11, 'query'),
    'L11_k': (11, 'key'),
    'L11_v': (11, 'value'),
    'L11_o': (11, 'output'),
}
GROUP_NAMES = list(MONITOR_GROUPS.keys())

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_selected_controllers(model):
    """Return dict {group_name: [ctrl, ...]} for first and last blocks only.

    Each group may contain multiple controllers (one per tile split).
    """
    from aihwkit.nn import AnalogLinear

    groups = {g: [] for g in MONITOR_GROUPS}
    seen = set()

    for name, module in model.named_modules():
        if not isinstance(module, AnalogLinear):
            continue

        # Match layer index and target from module name
        matched_group = None
        for gname, (layer_idx, target_sub) in MONITOR_GROUPS.items():
            layer_str = f'layer.{layer_idx}.'
            if layer_str in name and target_sub in name:
                matched_group = gname
                break

        if matched_group is None:
            continue

        for tile in module.analog_tiles():
            ctrl = getattr(tile, '_lrtt_controller', None)
            if ctrl is not None and id(ctrl) not in seen:
                seen.add(id(ctrl))
                groups[matched_group].append(ctrl)

    return groups


def snapshot_groups(groups):
    """Return dict {group_name: [(A, B, C), ...]} of float-CPU tensors."""
    snaps = {}
    for gname, ctrls in groups.items():
        snaps[gname] = []
        for ctrl in ctrls:
            A = ctrl.tile_a.get_weights()[0].float().cpu()
            B = ctrl.tile_b.get_weights()[0].float().cpu()
            C = ctrl.tile_c.get_weights()[0].float().cpu()
            snaps[gname].append((A, B, C))
    return snaps


def compute_group_stats(snaps_group):
    """Compute mean norms for one group (averaging over splits)."""
    a_norms, b_norms, ab_norms, c_norms = [], [], [], []
    for A, B, C in snaps_group:
        a_norms.append(A.norm().item())
        b_norms.append(B.norm().item())
        ab_norms.append((A @ B).norm().item())
        c_norms.append(C.norm().item())
    return {
        'a_norm':  float(np.mean(a_norms)),
        'b_norm':  float(np.mean(b_norms)),
        'ab_norm': float(np.mean(ab_norms)),
        'c_norm':  float(np.mean(c_norms)),
    }


def compute_group_delta(snaps_before, snaps_after):
    """Compute mean |ΔA|, |ΔB| for one group (averaging over splits)."""
    da, db = [], []
    for (A0, B0, _), (A1, B1, _) in zip(snaps_before, snaps_after):
        da.append((A1 - A0).abs().mean().item())
        db.append((B1 - B0).abs().mean().item())
    return float(np.mean(da)), float(np.mean(db))


def make_log():
    """Create empty per-group log dict."""
    scalar_keys = ('step', 'loss', 'lr')
    group_keys  = ('a_norm', 'b_norm', 'ab_norm', 'c_norm', 'delta_a', 'delta_b')
    log = {k: [] for k in scalar_keys}
    for g in GROUP_NAMES:
        for k in group_keys:
            log[f'{g}/{k}'] = []
    return log


# ── Main run ─────────────────────────────────────────────────────────────────

def run(pulse_type, train_loader, eval_features, eval_examples, tokenizer):
    from aihwkit.optim import AnalogAdam
    from transformers import set_seed

    print(f"\n{'='*60}")
    print(f"  ab_pulse_type = {pulse_type}")
    print(f"  warmup {WARMUP_STEPS} steps (silent) -> observe {N_OBS} steps")
    print(f"{'='*60}")

    O.OPT_CONFIG = dict(BASE_OPT_CONFIG)
    O.OPT_CONFIG['ab_pulse_type'] = pulse_type

    set_seed(O.SEED)

    rank = 2 ** FIXED_PARAMS['rank_exp']
    params = {
        "rank":                     rank,
        "transfer_every":           FIXED_PARAMS['transfer_every'],
        "transfer_lr":              FIXED_PARAMS['transfer_lr'],
        "fast_lr":                  FIXED_PARAMS['fast_lr'],
        "reinit_mode":              'hybrid',
        "tau_sec":                  FIXED_PARAMS['tau_sec'],
        "ab_dw_min":                FIXED_PARAMS['ab_dw_min'],
        "ab_desired_bl":            FIXED_PARAMS['ab_desired_bl'],
        "c_dw_min":                 0.001,
        "c_desired_bl":             None,
        "out_noise":                FIXED_PARAMS['out_noise'],
        "ab_weight_scaling_omega":  FIXED_PARAMS['ab_weight_scaling_omega'],
        "lora_alpha":               1.0,
    }

    model = O.create_model(params)

    optimizer = AnalogAdam(model.parameters(), lr=FIXED_PARAMS['learning_rate'])
    optimizer.regroup_param_groups()
    optimizer._grad_accum_steps = O.GRAD_ACCUM_STEPS

    full_training_steps = 4580
    scheduler = O.get_linear_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=full_training_steps,
        min_lr_rate=0.0,
    )

    groups = get_selected_controllers(model)
    for gname, ctrls in groups.items():
        print(f"  {gname}: {len(ctrls)} split(s)")

    log = make_log()

    model.train()
    optimizer.zero_grad()
    step = 0
    micro = 0
    observing = False
    snaps_before = None

    for batch in train_loader:
        input_ids       = batch['input_ids'].to(O.DEVICE)
        attention_mask  = batch['attention_mask'].to(O.DEVICE)
        start_positions = batch['start_positions'].to(O.DEVICE)
        end_positions   = batch['end_positions'].to(O.DEVICE)

        # Snapshot at start of accumulation cycle, only on logging steps
        obs_step_preview = (step - WARMUP_STEPS) if observing else -1
        next_obs = obs_step_preview + 1
        is_log_step = (next_obs >= LOG_START and (next_obs - LOG_START) % LOG_EVERY == 0)
        if micro % O.GRAD_ACCUM_STEPS == 0 and observing and is_log_step:
            snaps_before = snapshot_groups(groups)
        elif micro % O.GRAD_ACCUM_STEPS == 0:
            snaps_before = None

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            start_positions=start_positions, end_positions=end_positions,
        )
        loss = outputs.loss / O.GRAD_ACCUM_STEPS

        if torch.isnan(loss):
            print(f"  NaN at step {step}. Stopping.")
            break

        loss.backward()
        micro += 1

        if micro % O.GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step == WARMUP_STEPS:
                observing = True
                print(f"  Warmup done at step {step}. Starting observation...")

            obs_step = step - WARMUP_STEPS
            is_log = (obs_step >= LOG_START and (obs_step - LOG_START) % LOG_EVERY == 0)
            if observing and is_log and snaps_before is not None:
                snaps_after = snapshot_groups(groups)

                log['step'].append(obs_step)
                log['loss'].append(loss.item() * O.GRAD_ACCUM_STEPS)
                log['lr'].append(optimizer.param_groups[0]['lr'])

                stats_l0 = []   # collect for print summary
                for gname in GROUP_NAMES:
                    stats = compute_group_stats(snaps_after[gname])
                    da, db = compute_group_delta(snaps_before[gname], snaps_after[gname])
                    for k, v in stats.items():
                        log[f'{gname}/{k}'].append(v)
                    log[f'{gname}/delta_a'].append(da)
                    log[f'{gname}/delta_b'].append(db)
                    if gname.startswith('L0_'):
                        stats_l0.append((gname, stats, da, db))

                # Print summary line (L0 groups only to keep it readable)
                parts = []
                for gname, stats, da, db in stats_l0:
                    tgt = gname.split('_')[1]
                    parts.append(f"{tgt}:||A||={stats['a_norm']:.4f} dA={da:.2e}")
                print(f"  obs+{obs_step:3d} | loss={log['loss'][-1]:.4f} | lr={log['lr'][-1]:.2e} | L0: {' | '.join(parts)}")

                if obs_step >= N_OBS:
                    break

    del model, optimizer
    torch.cuda.empty_cache()
    return log


# ── Plot ──────────────────────────────────────────────────────────────────────

# Colors for q/k/v/o targets
TARGET_COLORS = {'q': '#1f77b4', 'k': '#ff7f0e', 'v': '#2ca02c', 'o': '#d62728'}

def plot(log_nwd, log_stoch, out_path):
    # Layout: 4 rows × 4 cols
    # Rows: a_norm, ab_norm, delta_a, delta_b
    # Cols: L0 / L11 for nwd, L0 / L11 for stoch
    # → Actually: rows=metric, cols=(L0_nwd, L0_stoch, L11_nwd, L11_stoch)
    metrics = [
        ('a_norm',  '||A|| (Frob)'),
        ('b_norm',  '||B|| (Frob)'),
        ('ab_norm', '||A@B|| (Frob)'),
        ('c_norm',  '||C|| (Frob)'),
        ('delta_a', 'mean |ΔA| per step'),
        ('delta_b', 'mean |ΔB| per step'),
    ]
    fig, axes = plt.subplots(6, 4, figsize=(20, 20))
    fig.suptitle(
        f"A/B Dynamics: nwd vs StochasticCompressed — First (L0) & Last (L11) block\n"
        f"constantstepideal (C+AB), rank=32, fast_lr={FIXED_PARAMS['fast_lr']}, "
        f"ab_dw_min={FIXED_PARAMS['ab_dw_min']:.4e} (14b), no transfer\n"
        f"x-axis: steps after warmup ({WARMUP_STEPS} steps)",
        fontsize=9, fontweight='bold',
    )

    col_specs = [
        ('L0',  log_nwd,   'nwd',   '-'),
        ('L0',  log_stoch, 'stoch', '--'),
        ('L11', log_nwd,   'nwd',   '-'),
        ('L11', log_stoch, 'stoch', '--'),
    ]
    col_titles = ['L0  nwd', 'L0  stoch', 'L11  nwd', 'L11  stoch']

    steps = log_nwd['step']

    for row, (mkey, mlabel) in enumerate(metrics):
        for col, (block, lg, mode, ls) in enumerate(col_specs):
            ax = axes[row][col]
            for tgt in ('q', 'k', 'v', 'o'):
                gname = f'{block}_{tgt}'
                vals = lg.get(f'{gname}/{mkey}', [])
                if vals:
                    ax.plot(steps[:len(vals)], vals,
                            color=TARGET_COLORS[tgt], lw=1.5, ls=ls,
                            label=tgt)
            ax.set_title(f"{col_titles[col]} | {mlabel}", fontsize=7.5)
            ax.set_xlabel('steps after warmup', fontsize=7)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            if mkey in ('delta_a', 'delta_b') and any(
                v > 0 for v in lg.get(f'{block}_q/{mkey}', []) if v == v
            ):
                try:
                    ax.set_yscale('log')
                except Exception:
                    pass

    # Add loss / lr as a small extra row at bottom (spanning all cols)
    # Actually just add text summary
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"\nSaved -> {out_path}")


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    from transformers import BertTokenizerFast

    print(f"Loading tokenizer & data...")
    tokenizer = BertTokenizerFast.from_pretrained(O.MODEL_NAME)
    print(f"  Data loaded. Running nwd then stochastic (same seed={O.SEED}).")

    train_loader_nwd,   eval_features, eval_examples = O.load_data(tokenizer)
    train_loader_stoch, _,             _             = O.load_data(tokenizer)

    log_nwd   = run('none_with_device', train_loader_nwd,   eval_features, eval_examples, tokenizer)
    log_stoch = run('default',          train_loader_stoch, eval_features, eval_examples, tokenizer)

    # Summary
    print(f"\n── Summary (obs window mean) ───────────────────────────────")
    for gname in GROUP_NAMES:
        for mkey in ('a_norm', 'ab_norm', 'delta_a', 'delta_b'):
            n_vals = log_nwd.get(f'{gname}/{mkey}', [])
            s_vals = log_stoch.get(f'{gname}/{mkey}', [])
            n = float(np.mean(n_vals)) if n_vals else float('nan')
            s = float(np.mean(s_vals)) if s_vals else float('nan')
            print(f"  {gname}/{mkey:8s}  nwd={n:.4e}  stoch={s:.4e}  ratio={s/(n+1e-12):.3f}")

    out = Path("results/optuna_bert_squad_lrtt/diag_ab_dynamics.png")
    plot(log_nwd, log_stoch, out)


if __name__ == "__main__":
    main()
