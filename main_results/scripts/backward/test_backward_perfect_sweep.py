#!/usr/bin/env python3
"""Sweep: lora_alpha × target_ab_lr × backward_perfect (1 epoch each).

Grid:
  lora_alpha:    [0.001, 0.01, 0.1, 1.0]
  target_ab_lr:  [0.001, 0.01, 0.1]
  backward:      [default, perfect]

= 24 experiments, each 1 epoch on STS-B full.

For each experiment, records:
  - Per-step train loss
  - End-of-epoch spearmanr
  - ΔA, ΔB (Frobenius norm)
  - LoRA contribution ratio + cosine similarity
  - Gradient nonzero ratio

Saves results to JSON + generates comparison plots.

Usage:
    /data/venvs/lrtt/bin/python test_backward_perfect_sweep.py
"""

import os, sys, json, math, time
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

os.environ["WANDB_MODE"] = "offline"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_EPOCHS = 1
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
BASE_LR = 1.45e-3  # digital params LR (fixed)
RANK = 16

# Sweep grid
LORA_ALPHAS = [0.001, 0.01, 0.1, 1.0]
TARGET_AB_LRS = [0.001, 0.01, 0.1]

OUTPUT_DIR = "/data/probe/lora"

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.tiles.analog import AnalogTile

from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, set_seed,
)
from datasets import load_dataset
from scipy.stats import spearmanr as scipy_spearmanr


# ============================================================
# Gradient capture hook
# ============================================================
_orig_backward = AnalogTile.backward
_grad_cap = {"on": False, "recs": []}


def _hooked_bwd(self, d_input, ctx=None):
    if _grad_cap["on"]:
        with torch.no_grad():
            nonzero_ratio = (d_input.abs() > 1e-8).float().mean().item()
            norm = d_input.norm().item()
            _grad_cap["recs"].append({
                "norm": norm,
                "nonzero_ratio": nonzero_ratio,
            })
    return _orig_backward(self, d_input, ctx)


AnalogTile.backward = _hooked_bwd


# ============================================================
# Model & Data
# ============================================================

def create_lrtt_config(lora_alpha, backward_perfect=False):
    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
        write_noise_std=0.0, mean_bound_reference=True, lifetime=0.0,
    )
    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=False,
    )
    device_config = PythonLRTTDevice(
        rank=RANK, transfer_every=10000000, lora_alpha=lora_alpha,
        reinit_gain=1.0, reinit_mode="decay", decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.1
    device_config.units_in_mbatch = True
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = True
    device_config.combined_out_scaling = True

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    if backward_perfect:
        rpu_config.backward.is_perfect = True

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


def create_model(lora_alpha, backward_perfect=False):
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        "albert/albert-base-v2", num_labels=1)
    torch.manual_seed(SEED)
    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
    nn.init.zeros_(model.classifier.bias)

    lrtt_patterns = ["attention"]
    always_digital = ["classifier", "albert.encoder.embedding_hidden_mapping_in"]
    all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude = [n for n in all_linear
               if any(d in n for d in always_digital)
               or "encoder" not in n
               or not any(p in n for p in lrtt_patterns)]
    exclude += always_digital
    exclude = list(set(exclude))

    lrtt_config = create_lrtt_config(lora_alpha, backward_perfect=backward_perfect)
    model = convert_to_analog(model, lrtt_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            param.requires_grad = True
        elif "tile_c" in name and "analog_ctx" in name:
            param.requires_grad = True
        elif "out_scaling" in name:
            param.requires_grad = True
        elif "classifier" in name:
            param.requires_grad = True
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model.to(DEVICE)


def load_stsb_data():
    tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
    raw = load_dataset("nyu-mll/glue", "stsb")

    def preprocess(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"],
                         max_length=128, truncation=True)

    remove_cols = [c for c in raw["train"].column_names if c != "label"]
    tokenized = raw.map(preprocess, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer)
    print(f"  STS-B: Train={len(tokenized['train'])}, Eval={len(tokenized['validation'])}")
    return tokenized, collator


def make_loaders(tokenized, collator):
    train_loader = torch.utils.data.DataLoader(
        tokenized["train"], batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(SEED),
    )
    eval_loader = torch.utils.data.DataLoader(
        tokenized["validation"], batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=collator,
    )
    return train_loader, eval_loader


# ============================================================
# Diagnostic helpers
# ============================================================

def get_ab_weight_snapshots(model):
    snapshots = {}
    for name, m in model.named_modules():
        if hasattr(m, 'tile_a') and hasattr(m, 'tile_b'):
            wa = m.tile_a.get_weights()[0].detach().cpu().clone()
            wb = m.tile_b.get_weights()[0].detach().cpu().clone()
            snapshots[name] = (wa, wb)
    return snapshots


def compute_weight_changes(prev_snaps, curr_snaps):
    da_norms, db_norms = [], []
    for key in prev_snaps:
        if key in curr_snaps:
            da_norms.append((curr_snaps[key][0] - prev_snaps[key][0]).norm().item())
            db_norms.append((curr_snaps[key][1] - prev_snaps[key][1]).norm().item())
    return np.mean(da_norms) if da_norms else 0.0, \
           np.mean(db_norms) if db_norms else 0.0


def compute_lora_contribution(model, eval_batch):
    model.eval()
    inputs = {k: v.to(DEVICE) for k, v in eval_batch.items()
              if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}

    with torch.no_grad():
        y_full = model(**inputs).logits.detach().flatten()

        controllers = []
        orig_alphas = []
        for m in model.modules():
            if hasattr(m, 'controller') and hasattr(m.controller, 'lora_alpha'):
                controllers.append(m.controller)
                orig_alphas.append(m.controller.lora_alpha)
                m.controller.lora_alpha = 0.0

        y_c_only = model(**inputs).logits.detach().flatten()

        for ctrl, alpha in zip(controllers, orig_alphas):
            ctrl.lora_alpha = alpha

    model.train()

    y_lora = y_full - y_c_only
    lora_norm = y_lora.norm().item()
    c_norm = y_c_only.norm().item()
    ratio = lora_norm / max(c_norm, 1e-8)
    cos = torch.nn.functional.cosine_similarity(
        y_lora.unsqueeze(0), y_c_only.unsqueeze(0)
    ).item()
    return {"ratio": ratio, "cosine": cos}


def evaluate_stsb(model, eval_loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in eval_loader:
            inputs = {k: v.to(DEVICE) for k, v in batch.items()
                      if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
            outputs = model(**inputs)
            total_loss += outputs.loss.item() * len(batch["labels"])
            preds = outputs.logits.squeeze().cpu().numpy()
            all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else [preds])
            all_labels.extend(batch["labels"].cpu().tolist())

    model.train()
    n = len(all_labels)
    avg_loss = total_loss / n if n > 0 else 0.0
    sr = scipy_spearmanr(all_preds, all_labels)[0]
    return sr, avg_loss


# ============================================================
# Single experiment
# ============================================================

def run_single(lora_alpha, target_ab_lr, backward_perfect, tokenized, collator):
    """Run 1-epoch experiment. Returns result dict."""
    tag = f"α={lora_alpha}, ab_lr={target_ab_lr}, {'perf' if backward_perfect else 'def'}"

    train_loader, eval_loader = make_loaders(tokenized, collator)

    set_seed(SEED)
    model = create_model(lora_alpha=lora_alpha, backward_perfect=backward_perfect)

    # Optimizer: digital params get BASE_LR, LRTT tiles get lrtt_lr
    optimizer = AnalogAdam(model.parameters(), lr=BASE_LR, weight_decay=0.0)
    optimizer.regroup_param_groups()

    # Set LRTT tile LR: lrtt_lr = target_ab_lr / lora_alpha
    lrtt_lr = target_ab_lr / lora_alpha
    lrtt_tile_ids = set()
    for m in model.modules():
        if hasattr(m, 'tile_a'):
            lrtt_tile_ids.update([id(m.tile_a), id(m.tile_b), id(m.tile_c)])
    for group in optimizer.param_groups:
        for p in group["params"]:
            if hasattr(p, 'analog_tile') and id(p.analog_tile) in lrtt_tile_ids:
                group["lr"] = lrtt_lr
                p.analog_tile.set_learning_rate(lrtt_lr)

    eval_batch = next(iter(eval_loader))
    prev_snaps = get_ab_weight_snapshots(model)

    model.train()
    step_losses = []
    grad_recs = []

    t0 = time.time()
    for step, batch in enumerate(train_loader):
        inputs = {k: v.to(DEVICE) for k, v in batch.items()
                  if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss

        _grad_cap["on"] = True
        _grad_cap["recs"] = []
        loss.backward()
        _grad_cap["on"] = False
        grad_recs.extend(_grad_cap["recs"])

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step_losses.append(loss.item())

        if step % 100 == 0:
            print(f"    [{tag}] step {step:4d}/{len(train_loader)} loss={loss.item():.4f}")

    elapsed = time.time() - t0

    # End-of-epoch diagnostics
    curr_snaps = get_ab_weight_snapshots(model)
    da, db = compute_weight_changes(prev_snaps, curr_snaps)
    lora_info = compute_lora_contribution(model, eval_batch)
    sr, eval_loss = evaluate_stsb(model, eval_loader)

    avg_gnz = np.mean([r["nonzero_ratio"] for r in grad_recs]) if grad_recs else 0.0
    avg_gnorm = np.mean([r["norm"] for r in grad_recs]) if grad_recs else 0.0

    result = {
        "lora_alpha": lora_alpha,
        "target_ab_lr": target_ab_lr,
        "lrtt_lr": lrtt_lr,
        "backward_perfect": backward_perfect,
        "spearmanr": sr,
        "eval_loss": eval_loss,
        "train_loss_start": step_losses[0] if step_losses else 0,
        "train_loss_end": np.mean(step_losses[-20:]) if len(step_losses) >= 20 else step_losses[-1] if step_losses else 0,
        "train_loss_mean": np.mean(step_losses),
        "delta_a": da,
        "delta_b": db,
        "lora_ratio": lora_info["ratio"],
        "lora_cosine": lora_info["cosine"],
        "grad_nonzero": avg_gnz,
        "grad_norm": avg_gnorm,
        "n_steps": len(step_losses),
        "elapsed_sec": elapsed,
        "step_losses": step_losses,  # full curve for plotting
    }

    print(f"  [{tag}] spr={sr:.4f}, ΔA={da:.4f}, ΔB={db:.4f}, "
          f"LoRA%={lora_info['ratio']*100:.1f}%, cos={lora_info['cosine']:+.3f}, "
          f"gnz={avg_gnz*100:.1f}%, {elapsed:.0f}s")

    # Free GPU memory
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# Plot
# ============================================================

def generate_plots(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    alphas = sorted(set(r["lora_alpha"] for r in all_results))
    ab_lrs = sorted(set(r["target_ab_lr"] for r in all_results))

    # --- Figure 1: Spearmanr heatmap (default vs perfect) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("STS-B Spearmanr after 1 Epoch: Default vs Perfect Backward", fontsize=13)

    for idx, bwd in enumerate([False, True]):
        ax = axes[idx]
        tag = "Perfect" if bwd else "Default"
        data = np.full((len(ab_lrs), len(alphas)), np.nan)
        for r in all_results:
            if r["backward_perfect"] == bwd:
                ai = alphas.index(r["lora_alpha"])
                bi = ab_lrs.index(r["target_ab_lr"])
                data[bi, ai] = r["spearmanr"]

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.0, vmax=0.9)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a}" for a in alphas])
        ax.set_yticks(range(len(ab_lrs)))
        ax.set_yticklabels([f"{l}" for l in ab_lrs])
        ax.set_xlabel("lora_alpha")
        ax.set_ylabel("target_ab_lr")
        ax.set_title(f"{tag} Backward")
        for bi in range(len(ab_lrs)):
            for ai in range(len(alphas)):
                if not np.isnan(data[bi, ai]):
                    ax.text(ai, bi, f"{data[bi,ai]:.3f}", ha='center', va='center',
                            fontsize=10, fontweight='bold',
                            color='white' if data[bi,ai] < 0.3 else 'black')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sweep_spearmanr_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Figure 2: ΔA/ΔB + LoRA ratio per combo ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Sweep: 1-Epoch Diagnostics (lora_alpha × target_ab_lr × backward_mode)",
                 fontsize=13, fontweight='bold')

    colors_bwd = {False: '#2196F3', True: '#E91E63'}
    n_combos = len(alphas) * len(ab_lrs)
    x = np.arange(n_combos)
    labels = []
    vals = {metric: {False: [], True: []} for metric in
            ['delta_a', 'lora_ratio', 'spearmanr', 'grad_nonzero']}

    for ab_lr in ab_lrs:
        for alpha in alphas:
            labels.append(f"α={alpha}\nlr={ab_lr}")
            for bwd in [False, True]:
                match = [r for r in all_results
                         if r["lora_alpha"] == alpha
                         and r["target_ab_lr"] == ab_lr
                         and r["backward_perfect"] == bwd]
                if match:
                    r = match[0]
                    vals['delta_a'][bwd].append(r['delta_a'])
                    vals['lora_ratio'][bwd].append(r['lora_ratio'] * 100)
                    vals['spearmanr'][bwd].append(r['spearmanr'])
                    vals['grad_nonzero'][bwd].append(r['grad_nonzero'] * 100)
                else:
                    for k in vals:
                        vals[k][bwd].append(0)

    w = 0.35
    titles = ['ΔA (Weight Change)', 'LoRA Contribution (%)',
              'Spearmanr', 'Gradient Nonzero (%)']
    keys = ['delta_a', 'lora_ratio', 'spearmanr', 'grad_nonzero']

    for i, (ax, key, title) in enumerate(zip(axes.flat, keys, titles)):
        ax.bar(x - w/2, vals[key][False], w, color=colors_bwd[False], label='Default', alpha=0.8)
        ax.bar(x + w/2, vals[key][True], w, color=colors_bwd[True], label='Perfect', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "sweep_diagnostics_bar.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Figure 3: LoRA cosine direction ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LoRA Direction: cosine(y_lora, y_c) — +1=constructive, -1=destructive",
                 fontsize=12)

    for idx, bwd in enumerate([False, True]):
        ax = axes[idx]
        tag = "Perfect" if bwd else "Default"
        data = np.full((len(ab_lrs), len(alphas)), np.nan)
        for r in all_results:
            if r["backward_perfect"] == bwd:
                ai = alphas.index(r["lora_alpha"])
                bi = ab_lrs.index(r["target_ab_lr"])
                data[bi, ai] = r["lora_cosine"]

        im = ax.imshow(data, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a}" for a in alphas])
        ax.set_yticks(range(len(ab_lrs)))
        ax.set_yticklabels([f"{l}" for l in ab_lrs])
        ax.set_xlabel("lora_alpha")
        ax.set_ylabel("target_ab_lr")
        ax.set_title(f"{tag} Backward")
        for bi in range(len(ab_lrs)):
            for ai in range(len(alphas)):
                if not np.isnan(data[bi, ai]):
                    ax.text(ai, bi, f"{data[bi,ai]:+.2f}", ha='center', va='center',
                            fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sweep_lora_cosine_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Figure 4: Train loss curves (selected combos) ---
    fig, axes = plt.subplots(len(ab_lrs), len(alphas), figsize=(20, 12), sharex=True)
    fig.suptitle("Per-step Train Loss (1 epoch): Default vs Perfect\n"
                 "Row=target_ab_lr, Col=lora_alpha", fontsize=13)

    for bi, ab_lr in enumerate(ab_lrs):
        for ai, alpha in enumerate(alphas):
            ax = axes[bi, ai] if len(ab_lrs) > 1 else axes[ai]
            for bwd in [False, True]:
                match = [r for r in all_results
                         if r["lora_alpha"] == alpha
                         and r["target_ab_lr"] == ab_lr
                         and r["backward_perfect"] == bwd]
                if match:
                    losses = match[0]["step_losses"]
                    # Moving average
                    w = 10
                    if len(losses) > w:
                        ma = [np.mean(losses[max(0,i-w+1):i+1]) for i in range(len(losses))]
                    else:
                        ma = losses
                    label = "Perfect" if bwd else "Default"
                    ax.plot(ma, color=colors_bwd[bwd], alpha=0.8, linewidth=1, label=label)

            ax.set_title(f"α={alpha}, lr={ab_lr}", fontsize=9)
            if bi == len(ab_lrs) - 1:
                ax.set_xlabel("Step", fontsize=8)
            if ai == 0:
                ax.set_ylabel(f"Loss\n(ab_lr={ab_lr})", fontsize=8)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.2)
            ax.set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "sweep_loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Figure 5: Perfect - Default difference heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Perfect − Default: Improvement from backward_perfect=True", fontsize=13)

    diff_metrics = [
        ('spearmanr', 'Δ Spearmanr', 'RdYlGn'),
        ('delta_a', 'Δ ΔA (weight change)', 'RdYlGn'),
        ('lora_ratio', 'Δ LoRA Ratio', 'RdYlGn'),
    ]

    for ax_idx, (metric, title, cmap) in enumerate(diff_metrics):
        ax = axes[ax_idx]
        data = np.full((len(ab_lrs), len(alphas)), np.nan)

        for ab_lr_i, ab_lr in enumerate(ab_lrs):
            for alpha_i, alpha in enumerate(alphas):
                def_match = [r for r in all_results
                             if r["lora_alpha"] == alpha
                             and r["target_ab_lr"] == ab_lr
                             and not r["backward_perfect"]]
                perf_match = [r for r in all_results
                              if r["lora_alpha"] == alpha
                              and r["target_ab_lr"] == ab_lr
                              and r["backward_perfect"]]
                if def_match and perf_match:
                    d_val = def_match[0][metric]
                    p_val = perf_match[0][metric]
                    if metric == 'lora_ratio':
                        data[ab_lr_i, alpha_i] = (p_val - d_val) * 100
                    else:
                        data[ab_lr_i, alpha_i] = p_val - d_val

        vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
        im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a}" for a in alphas])
        ax.set_yticks(range(len(ab_lrs)))
        ax.set_yticklabels([f"{l}" for l in ab_lrs])
        ax.set_xlabel("lora_alpha")
        ax.set_ylabel("target_ab_lr")
        ax.set_title(title)
        for bi in range(len(ab_lrs)):
            for ai in range(len(alphas)):
                if not np.isnan(data[bi, ai]):
                    fmt = f"{data[bi,ai]:+.3f}" if metric == 'spearmanr' else f"{data[bi,ai]:+.2f}"
                    ax.text(ai, bi, fmt, ha='center', va='center',
                            fontsize=9, fontweight='bold')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sweep_perfect_minus_default.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading STS-B data...")
    tokenized, collator = load_stsb_data()

    total = len(LORA_ALPHAS) * len(TARGET_AB_LRS) * 2
    print(f"\nSweep: {len(LORA_ALPHAS)} alphas × {len(TARGET_AB_LRS)} ab_lrs × 2 backward = {total} experiments")
    print(f"Each: 1 epoch, {len(tokenized['train'])} samples, batch={BATCH_SIZE}")
    print(f"Device: {DEVICE}")

    all_results = []
    exp_idx = 0

    for alpha in LORA_ALPHAS:
        for ab_lr in TARGET_AB_LRS:
            for bwd_perfect in [False, True]:
                exp_idx += 1
                tag = "PERFECT" if bwd_perfect else "DEFAULT"
                print(f"\n[{exp_idx}/{total}] alpha={alpha}, ab_lr={ab_lr}, "
                      f"lrtt_lr={ab_lr/alpha:.4f}, {tag}")

                result = run_single(alpha, ab_lr, bwd_perfect, tokenized, collator)
                all_results.append(result)

    # Save raw results (without step_losses for compact JSON)
    results_compact = []
    for r in all_results:
        rc = {k: v for k, v in r.items() if k != 'step_losses'}
        results_compact.append(rc)

    json_path = os.path.join(OUTPUT_DIR, "sweep_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_compact, f, indent=2)
    print(f"\nSaved results: {json_path}")

    # Print summary table
    print(f"\n{'='*110}")
    print(f"  SWEEP SUMMARY: 1 Epoch, Default vs Perfect")
    print(f"{'='*110}")
    print(f"  {'alpha':>7} {'ab_lr':>7} {'Mode':>7} {'spr':>8} {'ΔA':>8} {'ΔB':>8} "
          f"{'L%':>7} {'Lcos':>7} {'gnz%':>6} {'loss_end':>9}")
    print(f"  {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} "
          f"{'-'*7} {'-'*7} {'-'*6} {'-'*9}")

    for r in sorted(all_results, key=lambda x: (x['lora_alpha'], x['target_ab_lr'], x['backward_perfect'])):
        mode = "perf" if r['backward_perfect'] else "def"
        print(f"  {r['lora_alpha']:7.3f} {r['target_ab_lr']:7.3f} {mode:>7} "
              f"{r['spearmanr']:8.4f} {r['delta_a']:8.4f} {r['delta_b']:8.4f} "
              f"{r['lora_ratio']*100:6.1f}% {r['lora_cosine']:+6.3f} "
              f"{r['grad_nonzero']*100:5.1f}% {r['train_loss_end']:9.4f}")

    # Generate plots
    print(f"\nGenerating plots...")
    generate_plots(all_results)
    print(f"\nDone! All plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
