#!/usr/bin/env python3
"""Diagnostic: Does backward_perfect=True enable meaningful LRTT LoRA learning?

Runs 5 diagnostic checks comparing default vs perfect backward:
  1. A/B Tile Weight Change Tracking (||ΔA||, ||ΔB|| Frobenius norm)
  2. LoRA Forward Contribution Ratio (||y_full - y_c_only|| / ||y_full||)
  3. Eval Metric (STS-B spearmanr) convergence comparison
  4. Gradient Signal Quality (DA norm, nonzero ratio in update path)
  5. Per-step Train Loss convergence speed

Usage:
    /data/venvs/lrtt/bin/python test_backward_perfect_effectiveness.py
"""

import os, sys, math, copy
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

os.environ["WANDB_MODE"] = "offline"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_EPOCHS = 5
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
LR = 1.45e-3
RANK = 16
LORA_ALPHA = 1.0

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
# Diagnostic 4: Hook AnalogTile.backward to capture DA quality
# ============================================================
_orig_backward = AnalogTile.backward
_grad_cap = {"on": False, "recs": []}


def _hooked_bwd(self, d_input, ctx=None):
    if _grad_cap["on"]:
        with torch.no_grad():
            inp_abs = d_input.abs()
            nonzero_ratio = (inp_abs > 1e-8).float().mean().item()
            norm = d_input.norm().item()
            _grad_cap["recs"].append({
                "norm": norm,
                "nonzero_ratio": nonzero_ratio,
                "numel": d_input.numel(),
            })
    return _orig_backward(self, d_input, ctx)


AnalogTile.backward = _hooked_bwd


# ============================================================
# Model & Data (from test_backward_outlier_diag.py baseline)
# ============================================================

def create_lrtt_config(backward_perfect=False):
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
        rank=RANK, transfer_every=10000000, lora_alpha=LORA_ALPHA,
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


def create_model(backward_perfect=False):
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

    lrtt_config = create_lrtt_config(backward_perfect=backward_perfect)
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
    """Load and tokenize STS-B dataset. Returns (tokenized_dataset, collator).

    DataLoaders are created per-experiment via make_loaders() to ensure
    identical data ordering across runs (same generator seed each time).
    """
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
    """Create fresh DataLoaders with a new generator seeded identically.

    This ensures both default and perfect experiments see the exact same
    data ordering (same generator seed → same shuffle sequence).
    """
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
    """Get current A/B tile weight norms as {layer_name: (A_weights, B_weights)}."""
    snapshots = {}
    for name, m in model.named_modules():
        if hasattr(m, 'tile_a') and hasattr(m, 'tile_b'):
            # tile_a: [d_size, rank], tile_b: [rank, x_size]
            wa = m.tile_a.get_weights()[0].detach().cpu()
            wb = m.tile_b.get_weights()[0].detach().cpu()
            snapshots[name] = (wa.clone(), wb.clone())
    return snapshots


def compute_weight_changes(prev_snaps, curr_snaps):
    """Compute Frobenius norm of weight changes for A and B tiles."""
    delta_a_norms = []
    delta_b_norms = []
    for key in prev_snaps:
        if key in curr_snaps:
            da = (curr_snaps[key][0] - prev_snaps[key][0]).norm().item()
            db = (curr_snaps[key][1] - prev_snaps[key][1]).norm().item()
            delta_a_norms.append(da)
            delta_b_norms.append(db)
    return np.mean(delta_a_norms) if delta_a_norms else 0.0, \
           np.mean(delta_b_norms) if delta_b_norms else 0.0


def compute_lora_contribution(model, eval_batch):
    """Measure LoRA contribution on a single eval batch.

    Returns dict with:
      - ratio: ||y_lora|| / ||y_c_only||  (relative magnitude of LoRA output)
      - cosine: cosine_similarity(y_lora, y_c_only)  (direction alignment)
      - lora_norm: ||y_lora||
      - c_norm: ||y_c_only||

    Temporarily disables LoRA by setting lora_alpha=0 on each controller,
    then restores it.
    """
    model.eval()
    inputs = {k: v.to(DEVICE) for k, v in eval_batch.items()
              if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}

    with torch.no_grad():
        # Full forward (C + LoRA)
        y_full = model(**inputs).logits.detach().flatten()

        # Temporarily disable LoRA contribution
        controllers = []
        orig_alphas = []
        for m in model.modules():
            if hasattr(m, 'controller') and hasattr(m.controller, 'lora_alpha'):
                controllers.append(m.controller)
                orig_alphas.append(m.controller.lora_alpha)
                m.controller.lora_alpha = 0.0

        # C-only forward
        y_c_only = model(**inputs).logits.detach().flatten()

        # Restore
        for ctrl, alpha in zip(controllers, orig_alphas):
            ctrl.lora_alpha = alpha

    model.train()

    y_lora = y_full - y_c_only  # pure LoRA contribution
    lora_norm = y_lora.norm().item()
    c_norm = y_c_only.norm().item()
    ratio = lora_norm / max(c_norm, 1e-8)

    # Cosine similarity: +1 = same direction, -1 = opposing
    cos = torch.nn.functional.cosine_similarity(
        y_lora.unsqueeze(0), y_c_only.unsqueeze(0)
    ).item()

    return {"ratio": ratio, "cosine": cos, "lora_norm": lora_norm, "c_norm": c_norm}


def evaluate_stsb(model, eval_loader):
    """Evaluate model on STS-B validation. Returns (spearmanr, avg_loss)."""
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
# Main experiment runner
# ============================================================

def run_experiment(backward_perfect, tokenized, collator):
    """Run full training experiment and collect all 5 diagnostics.

    Creates fresh DataLoaders each time to guarantee identical data ordering.
    """
    tag = "PERFECT" if backward_perfect else "DEFAULT"
    print(f"\n{'='*70}")
    print(f"  [{tag}] backward.is_perfect = {backward_perfect}")
    print(f"{'='*70}")

    # Fresh loaders — same seed → same shuffle order for both experiments
    train_loader, eval_loader = make_loaders(tokenized, collator)

    set_seed(SEED)
    model = create_model(backward_perfect=backward_perfect)

    # Verify config
    for n, m in model.named_modules():
        if hasattr(m, 'rpu_config'):
            bwd = m.rpu_config.backward
            print(f"  Config: bwd.is_perfect={bwd.is_perfect}, "
                  f"inp_res={bwd.inp_res:.6f}, noise_mgmt={bwd.noise_management}")
            break

    optimizer = AnalogAdam(model.parameters(), lr=LR, weight_decay=0.0)
    optimizer.regroup_param_groups()

    # Grab first eval batch for LoRA contribution check
    eval_batch_for_lora = next(iter(eval_loader))

    # Results storage
    results = {
        "epoch_spearmanr": [],
        "epoch_eval_loss": [],
        "epoch_delta_a": [],
        "epoch_delta_b": [],
        "epoch_lora_ratio": [],     # ||y_lora|| / ||y_c||
        "epoch_lora_cosine": [],    # cosine(y_lora, y_c)
        "step_losses": [],
        "epoch_grad_norm": [],
        "epoch_grad_nonzero": [],
    }

    # Initial weight snapshot (Diagnostic 1)
    prev_snaps = get_ab_weight_snapshots(model)

    model.train()
    global_step = 0

    for epoch in range(N_EPOCHS):
        epoch_losses = []

        # --- Diagnostic 4: gradient capture for this epoch ---
        epoch_grad_recs = []

        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()
                      if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}

            optimizer.zero_grad()

            # Forward
            outputs = model(**inputs)
            loss = outputs.loss

            # Enable gradient capture during backward
            _grad_cap["on"] = True
            _grad_cap["recs"] = []
            loss.backward()
            _grad_cap["on"] = False

            epoch_grad_recs.extend(_grad_cap["recs"])

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

            if step % 50 == 0:
                print(f"  [{tag}] Epoch {epoch+1}/{N_EPOCHS} Step {step:4d} | "
                      f"Loss: {loss.item():.4f}")

        results["step_losses"].extend(epoch_losses)

        # --- End of epoch diagnostics ---

        # Diagnostic 1: A/B weight changes
        curr_snaps = get_ab_weight_snapshots(model)
        da_mean, db_mean = compute_weight_changes(prev_snaps, curr_snaps)
        results["epoch_delta_a"].append(da_mean)
        results["epoch_delta_b"].append(db_mean)
        prev_snaps = curr_snaps

        # Diagnostic 2: LoRA contribution ratio + direction
        lora_info = compute_lora_contribution(model, eval_batch_for_lora)
        results["epoch_lora_ratio"].append(lora_info["ratio"])
        results["epoch_lora_cosine"].append(lora_info["cosine"])

        # Diagnostic 3: Eval metric
        sr, eval_loss = evaluate_stsb(model, eval_loader)
        results["epoch_spearmanr"].append(sr)
        results["epoch_eval_loss"].append(eval_loss)

        # Diagnostic 4: Gradient signal quality (epoch average)
        if epoch_grad_recs:
            avg_norm = np.mean([r["norm"] for r in epoch_grad_recs])
            avg_nonzero = np.mean([r["nonzero_ratio"] for r in epoch_grad_recs])
        else:
            avg_norm = 0.0
            avg_nonzero = 0.0
        results["epoch_grad_norm"].append(avg_norm)
        results["epoch_grad_nonzero"].append(avg_nonzero)

        # Epoch summary
        train_loss = np.mean(epoch_losses)
        print(f"  [{tag}] Epoch {epoch+1}: train_loss={train_loss:.4f}, "
              f"spearmanr={sr:.4f}, eval_loss={eval_loss:.4f}, "
              f"ΔA={da_mean:.6f}, ΔB={db_mean:.6f}, "
              f"LoRA_ratio={lora_info['ratio']*100:.2f}%, "
              f"LoRA_cos={lora_info['cosine']:+.3f}, "
              f"grad_norm={avg_norm:.4f}, grad_nz={avg_nonzero*100:.1f}%")

    return results


# ============================================================
# Comparison and verdict
# ============================================================

def print_comparison(res_def, res_perf):
    """Print side-by-side comparison table and verdict."""
    print(f"\n{'='*110}")
    print(f"  COMPARISON: Default vs Perfect Backward ({N_EPOCHS} epochs, STS-B)")
    print(f"  NOTE: Both experiments use identical data ordering (fresh DataLoader per run)")
    print(f"{'='*110}")

    # Header
    print(f"\n  {'Ep':>2}  {'Def_spr':>8}  {'Prf_spr':>8}  "
          f"{'ΔA_d':>8}  {'ΔA_p':>8}  {'ΔB_d':>8}  {'ΔB_p':>8}  "
          f"{'L%_d':>6}  {'L%_p':>6}  "
          f"{'Lcos_d':>7}  {'Lcos_p':>7}  "
          f"{'Gnz_d':>6}  {'Gnz_p':>6}")
    print(f"  {'--':>2}  {'--------':>8}  {'--------':>8}  "
          f"{'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}  "
          f"{'------':>6}  {'------':>6}  "
          f"{'-------':>7}  {'-------':>7}  "
          f"{'------':>6}  {'------':>6}")

    for e in range(N_EPOCHS):
        print(f"  {e+1:2d}  "
              f"{res_def['epoch_spearmanr'][e]:8.4f}  "
              f"{res_perf['epoch_spearmanr'][e]:8.4f}  "
              f"{res_def['epoch_delta_a'][e]:8.4f}  "
              f"{res_perf['epoch_delta_a'][e]:8.4f}  "
              f"{res_def['epoch_delta_b'][e]:8.4f}  "
              f"{res_perf['epoch_delta_b'][e]:8.4f}  "
              f"{res_def['epoch_lora_ratio'][e]*100:5.1f}%  "
              f"{res_perf['epoch_lora_ratio'][e]*100:5.1f}%  "
              f"{res_def['epoch_lora_cosine'][e]:+6.3f}  "
              f"{res_perf['epoch_lora_cosine'][e]:+6.3f}  "
              f"{res_def['epoch_grad_nonzero'][e]*100:5.1f}%  "
              f"{res_perf['epoch_grad_nonzero'][e]*100:5.1f}%")

    # --- LoRA direction analysis ---
    print(f"\n  --- LoRA Direction Analysis ---")
    print(f"  cosine(y_lora, y_c): +1=complementary, -1=destructive, 0=orthogonal")
    for e in range(N_EPOCHS):
        d_cos = res_def['epoch_lora_cosine'][e]
        p_cos = res_perf['epoch_lora_cosine'][e]
        d_tag = "complementary" if d_cos > 0.3 else "destructive" if d_cos < -0.3 else "orthogonal"
        p_tag = "complementary" if p_cos > 0.3 else "destructive" if p_cos < -0.3 else "orthogonal"
        print(f"    Epoch {e+1}: default={d_cos:+.3f} ({d_tag}), "
              f"perfect={p_cos:+.3f} ({p_tag})")

    # --- Diagnostic 5: Loss convergence ---
    print(f"\n  --- Per-step Train Loss (moving avg, window=20) ---")
    window = 20
    def moving_avg(arr, w):
        if len(arr) < w:
            return arr
        return [np.mean(arr[max(0, i-w+1):i+1]) for i in range(len(arr))]

    ma_def = moving_avg(res_def["step_losses"], window)
    ma_perf = moving_avg(res_perf["step_losses"], window)
    n_steps = min(len(ma_def), len(ma_perf))
    checkpoints = [0, n_steps//4, n_steps//2, 3*n_steps//4, n_steps-1]
    print(f"  {'Step':>6}  {'Default':>10}  {'Perfect':>10}  {'Diff':>10}")
    for idx in checkpoints:
        if idx < n_steps:
            d = ma_perf[idx] - ma_def[idx]
            print(f"  {idx:6d}  {ma_def[idx]:10.4f}  {ma_perf[idx]:10.4f}  {d:+10.4f}")

    # --- Gradient signal quality summary ---
    print(f"\n  --- Gradient Signal Quality (avg across epochs) ---")
    avg_norm_def = np.mean(res_def["epoch_grad_norm"])
    avg_norm_perf = np.mean(res_perf["epoch_grad_norm"])
    avg_nz_def = np.mean(res_def["epoch_grad_nonzero"])
    avg_nz_perf = np.mean(res_perf["epoch_grad_nonzero"])
    print(f"  Default:  avg_grad_norm={avg_norm_def:.4f}, avg_nonzero={avg_nz_def*100:.1f}%")
    print(f"  Perfect:  avg_grad_norm={avg_norm_perf:.4f}, avg_nonzero={avg_nz_perf*100:.1f}%")
    if avg_norm_def > 0:
        print(f"  Norm ratio (perfect/default): {avg_norm_perf/avg_norm_def:.2f}x")

    # --- VERDICT ---
    print(f"\n{'='*110}")
    print(f"  VERDICT")
    print(f"{'='*110}")

    # Criterion 1: ΔA/ΔB ratio
    da_def_total = sum(res_def["epoch_delta_a"])
    da_perf_total = sum(res_perf["epoch_delta_a"])
    db_def_total = sum(res_def["epoch_delta_b"])
    db_perf_total = sum(res_perf["epoch_delta_b"])
    da_ratio = da_perf_total / max(da_def_total, 1e-10)
    db_ratio = db_perf_total / max(db_def_total, 1e-10)
    print(f"  1. A/B Weight Change: ΔA ratio={da_ratio:.2f}x, ΔB ratio={db_ratio:.2f}x "
          f"(criterion: ≥3x)")
    pass1 = da_ratio >= 3.0

    # Criterion 2: LoRA contribution
    final_lora_perf = res_perf["epoch_lora_ratio"][-1] * 100
    final_lora_def = res_def["epoch_lora_ratio"][-1] * 100
    print(f"  2. LoRA Contribution: default={final_lora_def:.1f}%, perfect={final_lora_perf:.1f}% "
          f"(criterion: ≥5%)")
    pass2 = final_lora_perf >= 5.0

    # Criterion 3: Spearmanr improvement
    final_sr_def = res_def["epoch_spearmanr"][-1]
    final_sr_perf = res_perf["epoch_spearmanr"][-1]
    best_sr_def = max(res_def["epoch_spearmanr"])
    best_sr_perf = max(res_perf["epoch_spearmanr"])
    sr_diff_final = final_sr_perf - final_sr_def
    sr_diff_best = best_sr_perf - best_sr_def
    print(f"  3. Spearmanr (final): default={final_sr_def:.4f}, perfect={final_sr_perf:.4f}, "
          f"diff={sr_diff_final:+.4f}")
    print(f"     Spearmanr (best):  default={best_sr_def:.4f}, perfect={best_sr_perf:.4f}, "
          f"diff={sr_diff_best:+.4f} (criterion: ≥+0.03)")
    pass3 = sr_diff_best >= 0.03 or sr_diff_final >= 0.03

    # Criterion 4: Gradient density
    print(f"  4. Gradient Density: default_nz={avg_nz_def*100:.1f}%, "
          f"perfect_nz={avg_nz_perf*100:.1f}% "
          f"(perfect should be denser)")
    pass4 = avg_nz_perf > avg_nz_def

    # Criterion 5: LoRA direction quality
    avg_cos_def = np.mean(res_def["epoch_lora_cosine"])
    avg_cos_perf = np.mean(res_perf["epoch_lora_cosine"])
    print(f"  5. LoRA Direction: default_cos={avg_cos_def:+.3f}, "
          f"perfect_cos={avg_cos_perf:+.3f} "
          f"(positive = constructive learning)")
    pass5 = avg_cos_perf > avg_cos_def

    criteria = [pass1, pass2, pass3, pass4, pass5]
    n_pass = sum(criteria)
    status = "PASS" if n_pass >= 4 else "PARTIAL" if n_pass >= 2 else "FAIL"

    labels = ["ΔA/B≥3x", "LoRA≥5%", "Δspr≥0.03", "Denser", "Direction"]
    markers = [f"[{'PASS' if p else 'FAIL'}] {l}" for p, l in zip(criteria, labels)]
    print(f"\n  Results: {' | '.join(markers)}")
    print(f"\n  Overall: {status} ({n_pass}/5 criteria met)")

    if status == "PASS":
        print(f"\n  >>> backward_perfect=True enables meaningful LRTT LoRA learning:")
        print(f"      A/B tiles learn {da_ratio:.1f}x more, LoRA contributes {final_lora_perf:.1f}% to output,")
        print(f"      spearmanr {best_sr_perf:.4f} vs {best_sr_def:.4f} ({sr_diff_best:+.4f})")
    elif status == "PARTIAL":
        print(f"\n  >>> backward_perfect shows partial improvement. Some criteria met.")
        print(f"      Key observation: both modes learn (spearmanr improves),")
        print(f"      but backward quantization noise may act as regularization.")
    else:
        print(f"\n  >>> backward_perfect did not show clear improvement on this run.")


# ============================================================
# Main
# ============================================================

def main():
    print("Loading STS-B data (full dataset)...")
    tokenized, collator = load_stsb_data()

    print(f"\nRunning {N_EPOCHS}-epoch experiments on {DEVICE}...")
    print(f"Config: ALBERT-base-v2, LoRA target=attn, rank={RANK}, "
          f"alpha={LORA_ALPHA}, lr={LR}, batch={BATCH_SIZE}")
    print(f"NOTE: Each experiment gets fresh DataLoader (same seed → identical data order)")

    # Run default (with backward quantization)
    res_def = run_experiment(False, tokenized, collator)

    # Run perfect (no backward quantization)
    res_perf = run_experiment(True, tokenized, collator)

    # Print comparison
    print_comparison(res_def, res_perf)


if __name__ == "__main__":
    main()
