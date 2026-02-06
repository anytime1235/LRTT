#!/home/jovyan/work/ml/.venv310/bin/python
# coding=utf-8
"""SST-2 A/B test: baseline LRTT vs auto_scale + transfer_ema_scale.

Runs 2 conditions × 1 epoch on SST-2 and compares:
  - Loss trajectory stability (std of per-step loss)
  - Final accuracy
  - EMA state sanity (no NaN/Inf)
  - Transfer scale variance across layers

Usage:
    python LRTT_glue/test_auto_scale_sst2.py
"""

import os
import sys
import json
import math
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

os.environ["LRTT_SILENT"] = "1"

# =============================================================================
# Fixed Parameters
# =============================================================================
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
SEED = 42
WARMUP_STEPS = 200
NUM_EPOCHS = 1
TARGET_MODULES = ["query", "key", "value"]

# LRTT hyperparams (fixed, not swept)
RANK = 4
LORA_ALPHA = 1.0
REINIT_GAIN = 0.1
LEARNING_RATE = 1e-3
TRANSFER_LR = 1.0
TRANSFER_EVERY = 100  # in samples (units_in_mbatch=True), same as sweep


# =============================================================================
# LRTT Config
# =============================================================================

def create_lrtt_config(
    auto_scale: bool = False,
    transfer_ema_scale: bool = False,
    auto_scale_momentum: float = 0.99,
    transfer_ema_momentum: float = 0.99,
):
    """Create LRTT config with optional auto_scale / transfer_ema_scale."""
    TAU_SEC = 46505.0
    dt_batch_sec = 1.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lifetime, lifetime_dtod=0.0,
        reset=0.0, reset_dtod=0.0,
    )

    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=3.0, w_min=-3.0,
        dw_min_dtod=0.0, dw_min_std=0.0,
        up_down=0.0, up_down_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=True,
    )

    device_config = PythonLRTTDevice(
        rank=RANK,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=LORA_ALPHA,
        reinit_gain=REINIT_GAIN,
        reinit_mode="hybrid",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
        # New features
        auto_scale=auto_scale,
        auto_scale_momentum=auto_scale_momentum,
        transfer_ema_scale=transfer_ema_scale,
        transfer_ema_momentum=transfer_ema_momentum,
        transfer_ema_target_norm=0.0,  # auto-calibrate
    )
    device_config.transfer_lr = TRANSFER_LR
    device_config.units_in_mbatch = True
    device_config.forward_inject = False
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


# =============================================================================
# Model & Data
# =============================================================================

def list_linear_layers(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def create_model(rpu_config, device):
    """Create SST-2 model with LRTT."""
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("classifier")

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        param.requires_grad = is_target or "classifier" in name

    return model.to(device)


def load_sst2_data(tokenizer):
    """Load SST-2 dataset."""
    raw = load_dataset("nyu-mll/glue", "sst2")

    def preprocess(examples):
        return tokenizer(examples["sentence"], padding="max_length",
                        max_length=MAX_SEQ_LENGTH, truncation=True)

    tokenized = raw.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=default_data_collator)
    eval_loader = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=default_data_collator)
    return train_loader, eval_loader


# =============================================================================
# Diagnostic: collect EMA stats from all LRTT layers
# =============================================================================

def collect_lrtt_stats(model) -> Dict[str, Any]:
    """Collect EMA state from all LRTT tiles."""
    from aihwkit.simulator.tiles.lrtt_tile import LRTTSimulatorTile

    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, LRTTSimulatorTile):
            ctrl = module.controller
            layer_stats = {
                "num_transfers": ctrl.num_transfers,
                "num_a_updates": ctrl.num_a_updates,
            }
            if ctrl._ema_m_x is not None:
                layer_stats["ema_m_x"] = ctrl._ema_m_x.item()
                layer_stats["ema_m_d"] = ctrl._ema_m_d.item()
                layer_stats["ema_product"] = ctrl._ema_m_x.item() * ctrl._ema_m_d.item()
            if ctrl._ema_ab_norm is not None:
                layer_stats["ema_ab_norm"] = ctrl._ema_ab_norm.item()
                layer_stats["transfer_ema_target_norm"] = ctrl.transfer_ema_target_norm
            stats[name] = layer_stats
    return stats


# =============================================================================
# Train & Evaluate
# =============================================================================

def train_and_evaluate(condition_name, rpu_config, train_loader, eval_loader, device):
    """Train 1 epoch and collect diagnostics."""
    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name}")
    print(f"{'='*60}")

    set_seed(SEED)
    model = create_model(rpu_config, device)

    optimizer = AnalogAdam(model.parameters(), lr=LEARNING_RATE)
    optimizer.regroup_param_groups()

    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    criterion = nn.CrossEntropyLoss()

    # --- Train ---
    model.train()
    step_losses = []
    t0 = time.time()

    pbar = tqdm(train_loader, desc=f"[{condition_name}] Epoch 1")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step_losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    train_time = time.time() - t0

    # --- Evaluate ---
    model.eval()
    correct, total = 0, 0
    eval_losses = []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            eval_losses.append(loss.item())
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    eval_loss = np.mean(eval_losses)

    # --- Collect layer-wise EMA stats ---
    lrtt_stats = collect_lrtt_stats(model)

    # --- Loss stability analysis ---
    # Split into windows of 50 steps
    window = 50
    window_means = [np.mean(step_losses[i:i+window]) for i in range(0, len(step_losses), window)]
    loss_trajectory_std = np.std(window_means) if len(window_means) > 1 else 0.0

    # Last 100 steps loss std
    tail_losses = step_losses[-100:] if len(step_losses) >= 100 else step_losses
    tail_std = np.std(tail_losses)

    results = {
        "condition": condition_name,
        "accuracy": accuracy,
        "eval_loss": eval_loss,
        "final_train_loss": np.mean(step_losses[-50:]) if len(step_losses) >= 50 else np.mean(step_losses),
        "train_loss_mean": np.mean(step_losses),
        "train_loss_std": np.std(step_losses),
        "loss_trajectory_std": loss_trajectory_std,
        "tail_loss_std": tail_std,
        "train_time_sec": train_time,
        "total_steps": len(step_losses),
        "lrtt_stats": lrtt_stats,
        "step_losses": step_losses,
    }

    # Print summary
    print(f"\n  Results for [{condition_name}]:")
    print(f"    Accuracy:          {accuracy:.4f}")
    print(f"    Eval Loss:         {eval_loss:.4f}")
    print(f"    Final Train Loss:  {results['final_train_loss']:.4f}")
    print(f"    Loss Traj Std:     {loss_trajectory_std:.4f}")
    print(f"    Tail Loss Std:     {tail_std:.4f}")
    print(f"    Train Time:        {train_time:.1f}s")

    # Print layer-wise EMA stats
    if lrtt_stats:
        print(f"\n    Layer-wise EMA Stats ({len(lrtt_stats)} layers):")
        ema_products = []
        ema_ab_norms = []
        for layer_name, ls in lrtt_stats.items():
            short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
            info = f"      {short_name}: transfers={ls['num_transfers']}"
            if "ema_product" in ls:
                info += f", ema(x*d)={ls['ema_product']:.4f}"
                ema_products.append(ls['ema_product'])
            if "ema_ab_norm" in ls:
                info += f", ema_ab_norm={ls['ema_ab_norm']:.4f}, target={ls['transfer_ema_target_norm']:.4f}"
                ema_ab_norms.append(ls['ema_ab_norm'])
            print(info)

        if ema_products:
            print(f"\n    EMA(x*d) across layers: mean={np.mean(ema_products):.4f}, "
                  f"std={np.std(ema_products):.4f}, "
                  f"cv={np.std(ema_products)/np.mean(ema_products):.2f}")
        if ema_ab_norms:
            print(f"    EMA(||AB||) across layers: mean={np.mean(ema_ab_norms):.4f}, "
                  f"std={np.std(ema_ab_norms):.4f}, "
                  f"cv={np.std(ema_ab_norms)/np.mean(ema_ab_norms):.2f}")

    del model
    torch.cuda.empty_cache()

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Task: SST-2 ({NUM_EPOCHS} epoch)")
    print(f"LRTT: rank={RANK}, transfer_every={TRANSFER_EVERY}, transfer_lr={TRANSFER_LR}")
    print(f"LR={LEARNING_RATE}, batch_size={BATCH_SIZE}")

    # Load data once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_sst2_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # === Condition 1: Baseline (no auto_scale, no transfer_ema) ===
    rpu_baseline = create_lrtt_config(auto_scale=False, transfer_ema_scale=False)
    results_baseline = train_and_evaluate("BASELINE", rpu_baseline, train_loader, eval_loader, device)

    # === Condition 2: auto_scale + transfer_ema_scale ===
    rpu_scaled = create_lrtt_config(auto_scale=True, transfer_ema_scale=True)
    results_scaled = train_and_evaluate("AUTO_SCALE+TRANSFER_EMA", rpu_scaled, train_loader, eval_loader, device)

    # === Comparison ===
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)

    def fmt(v, is_pct=False):
        if is_pct:
            return f"{v*100:.2f}%"
        return f"{v:.4f}"

    headers = ["Metric", "BASELINE", "AUTO_SCALE+EMA", "Delta"]
    rows = [
        ("Accuracy", results_baseline["accuracy"], results_scaled["accuracy"]),
        ("Eval Loss", results_baseline["eval_loss"], results_scaled["eval_loss"]),
        ("Final Train Loss", results_baseline["final_train_loss"], results_scaled["final_train_loss"]),
        ("Loss Traj Std", results_baseline["loss_trajectory_std"], results_scaled["loss_trajectory_std"]),
        ("Tail Loss Std", results_baseline["tail_loss_std"], results_scaled["tail_loss_std"]),
        ("Train Time (s)", results_baseline["train_time_sec"], results_scaled["train_time_sec"]),
    ]

    print(f"\n  {'Metric':<22} {'BASELINE':>12} {'SCALED':>12} {'Delta':>12}")
    print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*12}")
    for name, base, scaled in rows:
        delta = scaled - base
        sign = "+" if delta > 0 else ""
        if "Time" in name:
            pct = (delta / base * 100) if base > 0 else 0
            print(f"  {name:<22} {base:>11.1f}s {scaled:>11.1f}s {sign}{pct:>10.1f}%")
        else:
            print(f"  {name:<22} {base:>12.4f} {scaled:>12.4f} {sign}{delta:>12.4f}")

    # Stability assessment
    print(f"\n  Stability Assessment:")
    traj_improvement = (results_baseline["loss_trajectory_std"] - results_scaled["loss_trajectory_std"]) / results_baseline["loss_trajectory_std"] * 100 if results_baseline["loss_trajectory_std"] > 0 else 0
    tail_improvement = (results_baseline["tail_loss_std"] - results_scaled["tail_loss_std"]) / results_baseline["tail_loss_std"] * 100 if results_baseline["tail_loss_std"] > 0 else 0
    print(f"    Loss trajectory std improvement: {traj_improvement:+.1f}%")
    print(f"    Tail loss std improvement:       {tail_improvement:+.1f}%")

    overhead = (results_scaled["train_time_sec"] - results_baseline["train_time_sec"]) / results_baseline["train_time_sec"] * 100
    print(f"    Performance overhead:             {overhead:+.1f}%")

    # NaN/Inf check
    has_nan = False
    for layer_name, ls in results_scaled["lrtt_stats"].items():
        for k, v in ls.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                print(f"  WARNING: {layer_name}.{k} = {v} (NaN/Inf detected!)")
                has_nan = True
    if not has_nan:
        print(f"    NaN/Inf check:                   CLEAN")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/data/results/tikitaka/auto_scale_test_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    # Remove step_losses (too large for JSON) and tensor objects
    for r in [results_baseline, results_scaled]:
        r.pop("step_losses", None)
        # Clean up lrtt_stats for JSON serialization
        for layer_name in list(r["lrtt_stats"].keys()):
            for k, v in list(r["lrtt_stats"][layer_name].items()):
                if isinstance(v, torch.Tensor):
                    r["lrtt_stats"][layer_name][k] = v.item()

    summary = {
        "baseline": results_baseline,
        "scaled": results_scaled,
        "comparison": {
            "traj_std_improvement_pct": traj_improvement,
            "tail_std_improvement_pct": tail_improvement,
            "overhead_pct": overhead,
            "has_nan": has_nan,
        }
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  Results saved to: {out_dir}/results.json")


if __name__ == "__main__":
    main()
