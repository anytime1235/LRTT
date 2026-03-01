# -*- coding: utf-8 -*-
"""Gradient monitoring for ALBERT + MRPC with LoRA-LRTT (attn target).

Fixed hyperparameters:
    lr=0.1, target_ab_lr=0.1, lora_alpha=1.0, lora_target=attn

Monitors ACTUAL tile weight changes (not analog_ctx) to verify
QKV attention layers are receiving meaningful updates via aihwkit pulsed mechanism.

Usage:
    /data/venvs/lrtt/bin/python /data/gradient_monitor_attn.py
"""

import os
import sys
import math
import json
import gc
import collections

import torch
from torch import nn, no_grad, manual_seed
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

os.environ["WANDB_MODE"] = "offline"

# =============================================================================
# Constants
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
MODEL_NAME = "albert/albert-base-v2"
TASK_NAME = "mrpc"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
N_EPOCHS = 3
EVAL_BATCH_SIZE = 32
WARMUP_RATIO = 0.1

LEARNING_RATE = 0.1
TARGET_AB_LR = 0.1
LORA_ALPHA = 1.0
RANK = 16

TRANSFER_METHOD = "onehot"
COMBINED_OUT_SCALING = True
LEARN_OUT_SCALING = True
CONVERT_NONTARGET = True
REINIT_GAIN = 1.0
DECAY_FACTOR = 1.0

RESULTS_DIR = "/data/results/gradient_monitor_attn"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# Device configs
# =============================================================================
def _create_ab_device():
    return LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
        write_noise_std=0.0, mean_bound_reference=True,
        lifetime=0.0, lifetime_dtod=0.0, reset=0.0, reset_dtod=0.0,
    )

def _create_c_device():
    return SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=False,
    )

def create_lrtt_config():
    ab_device = _create_ab_device()
    c_device = _create_c_device()
    device_config = PythonLRTTDevice(
        rank=RANK, transfer_every=10000000, lora_alpha=LORA_ALPHA,
        reinit_gain=REINIT_GAIN, reinit_mode="decay", decay_factor=DECAY_FACTOR,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.1
    device_config.units_in_mbatch = True
    device_config.transfer_method = TRANSFER_METHOD
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = True
    device_config.combined_out_scaling = COMBINED_OUT_SCALING
    device_config.dynamic_te = False
    device_config.dynamic_te_power = 1.0
    device_config.dynamic_te_max = 10000000 * 20
    device_config.te_warmup_schedule = [10000000]
    device_config.te_warmup_steps = 0

    rpu_config = PythonLRTTRPUConfig(device=device_config)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config

def _create_nontarget_rpu_config():
    from aihwkit.simulator.configs import SingleRPUConfig
    device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=False,
    )
    rpu_config = SingleRPUConfig(device=device)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = LEARN_OUT_SCALING
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


# =============================================================================
# Model creation
# =============================================================================
def create_model():
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    torch.manual_seed(SEED)
    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
    if model.classifier.bias is not None:
        nn.init.zeros_(model.classifier.bias)

    lrtt_patterns = ["attention"]
    always_digital = ["classifier", "albert.encoder.embedding_hidden_mapping_in"]

    def is_lrtt_target(layer_name):
        if any(d in layer_name for d in always_digital):
            return False
        if "encoder" not in layer_name:
            return False
        return any(p in layer_name for p in lrtt_patterns)

    all_linear_names = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude_modules = [n for n in all_linear_names if not is_lrtt_target(n)]
    exclude_modules.append("classifier")
    exclude_modules.append("albert.encoder.embedding_hidden_mapping_in")
    exclude_modules = list(set(exclude_modules))

    lrtt_config = create_lrtt_config()
    model = convert_to_analog(model, lrtt_config, exclude_modules=exclude_modules)

    if CONVERT_NONTARGET:
        nt_encoder_layers = [
            n for n in all_linear_names
            if not is_lrtt_target(n) and "encoder" in n
            and not any(d in n for d in always_digital)
        ]
        nontarget_config = _create_nontarget_rpu_config()
        exclude_pass2 = [n for n in all_linear_names if n not in nt_encoder_layers]
        model = convert_to_analog(model, nontarget_config, exclude_modules=exclude_pass2,
                                  inplace=True, ensure_analog_root=False)

        def _frozen_noop_update(x_input, d_input, *args, **kwargs):
            return None
        for name, m in model.named_modules():
            if isinstance(m, AnalogLinear) and not is_lrtt_target(name):
                for tile in m.analog_tiles():
                    tile.update = _frozen_noop_update

    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            param.requires_grad = True
        elif "tile_c" in name and "analog_ctx" in name:
            param.requires_grad = True
        elif "out_scaling" in name:
            param.requires_grad = LEARN_OUT_SCALING
        elif "classifier" in name:
            param.requires_grad = True
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}, Trainable: {trainable:,}")

    return model.to(DEVICE), is_lrtt_target


# =============================================================================
# Data
# =============================================================================
def load_data(tokenizer):
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)
    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"],
                         max_length=MAX_SEQ_LENGTH, truncation=True)
    remove_cols = [c for c in raw_datasets["train"].column_names if c != "label"]
    tokenized = raw_datasets.map(preprocess, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=data_collator, generator=torch.Generator().manual_seed(SEED))
    eval_loader = DataLoader(tokenized["validation"], batch_size=EVAL_BATCH_SIZE, shuffle=False,
                             collate_fn=data_collator)
    print(f"  MRPC: Train={len(tokenized['train'])}, Eval={len(tokenized['validation'])}")
    return train_loader, eval_loader


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(model, eval_loader):
    from sklearn.metrics import f1_score, accuracy_score
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with no_grad():
        for batch in eval_loader:
            model_inputs = {k: v.to(DEVICE) for k, v in batch.items()
                           if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}
            outputs = model(**model_inputs)
            total_loss += outputs.loss.item() * len(batch["labels"])
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].cpu().tolist())
    model.train()
    n = len(all_labels)
    return f1_score(all_labels, all_preds), accuracy_score(all_labels, all_preds), total_loss / n


# =============================================================================
# Tile Weight Monitor - tracks ACTUAL tile weights via get_weights()
# =============================================================================
class TileWeightMonitor:
    """Track actual A/B tile weight changes using tile.get_weights()."""

    def __init__(self, model, is_lrtt_target_fn):
        self.model = model
        self.tiles = {}  # short_name -> (tile, layer_type)

        # Find all LRTT AnalogLinear layers and their A/B tiles
        for name, module in model.named_modules():
            if isinstance(module, AnalogLinear) and is_lrtt_target_fn(name):
                if "query" in name:
                    layer_type = "Q"
                elif "key" in name:
                    layer_type = "K"
                elif "value" in name:
                    layer_type = "V"
                elif "dense" in name:
                    layer_type = "Dense"
                else:
                    layer_type = "other"

                if hasattr(module, 'analog_module'):
                    # Get sub-tiles from the analog_module
                    for arr_name, arr_module in module.analog_module.named_modules():
                        if hasattr(arr_module, 'tile_a'):
                            short = f"{layer_type}_{arr_name}"
                            self.tiles[f"{short}_tileA"] = arr_module.tile_a
                            self.tiles[f"{short}_tileB"] = arr_module.tile_b
                            self.tiles[f"{short}_tileC"] = arr_module.tile_c

        print(f"\n  [TileWeightMonitor] Tracking {len(self.tiles)} tiles:")
        for k, tile in self.tiles.items():
            w, b = tile.get_weights()
            print(f"    {k}: shape={list(w.shape)}, norm={w.norm().item():.6f}")

        # Store per-step and per-epoch data
        self.step_deltas = []  # per-step weight change norms
        self.epoch_history = []
        self.initial_snapshots = self._snapshot_weights()

    def _snapshot_weights(self):
        """Get current tile weights (on CPU)."""
        snap = {}
        for name, tile in self.tiles.items():
            w, _ = tile.get_weights()
            snap[name] = w.cpu().clone()
        return snap

    def record_step(self, pre_snap):
        """Compare pre-step to post-step tile weights."""
        step_data = {}
        for name, tile in self.tiles.items():
            w_post, _ = tile.get_weights()
            w_pre = pre_snap[name]
            delta = (w_post.cpu() - w_pre)
            step_data[name] = {
                "delta_norm": delta.norm().item(),
                "delta_mean": delta.mean().item(),
                "delta_max": delta.abs().max().item(),
                "delta_nonzero_frac": (delta.abs() > 1e-12).float().mean().item(),
                "weight_norm_post": w_post.norm().item(),
            }
        self.step_deltas.append(step_data)

    def pre_step_snapshot(self):
        """Take snapshot BEFORE optimizer.step()."""
        return self._snapshot_weights()

    def summarize_epoch(self, epoch):
        if not self.step_deltas:
            return

        all_names = sorted(self.step_deltas[0].keys())
        summary = {}
        for name in all_names:
            norms = [s[name]["delta_norm"] for s in self.step_deltas]
            maxes = [s[name]["delta_max"] for s in self.step_deltas]
            nonzero_fracs = [s[name]["delta_nonzero_frac"] for s in self.step_deltas]
            weight_norms = [s[name]["weight_norm_post"] for s in self.step_deltas]

            summary[name] = {
                "avg_delta_norm": np.mean(norms),
                "max_delta_norm": np.max(norms),
                "min_delta_norm": np.min(norms),
                "avg_delta_max": np.mean(maxes),
                "avg_nonzero_frac": np.mean(nonzero_fracs),
                "final_weight_norm": weight_norms[-1],
            }

        self.epoch_history.append({"epoch": epoch, "summary": summary})

        # Print
        print(f"\n{'='*90}")
        print(f"  TILE WEIGHT UPDATE SUMMARY - Epoch {epoch}")
        print(f"{'='*90}")
        print(f"  {'Tile':<30} {'AvgDeltaNorm':>13} {'MaxDeltaNorm':>13} {'AvgDeltaMax':>12} {'NonZero%':>9} {'WeightNorm':>11}")
        print(f"  {'-'*30} {'-'*13} {'-'*13} {'-'*12} {'-'*9} {'-'*11}")

        for name in all_names:
            s = summary[name]
            # Short display name
            short = name.replace("array.", "")
            print(f"  {short:<30} {s['avg_delta_norm']:>13.8f} {s['max_delta_norm']:>13.8f} "
                  f"{s['avg_delta_max']:>12.8f} {s['avg_nonzero_frac']*100:>8.1f}% {s['final_weight_norm']:>11.4f}")

        # Group by Q/K/V/Dense
        for group in ["Q", "K", "V", "Dense"]:
            group_names = [n for n in all_names if n.startswith(f"{group}_")]
            if not group_names:
                continue
            tileA_names = [n for n in group_names if "tileA" in n]
            tileB_names = [n for n in group_names if "tileB" in n]
            tileC_names = [n for n in group_names if "tileC" in n]

            avg_A = np.mean([summary[n]["avg_delta_norm"] for n in tileA_names]) if tileA_names else 0
            avg_B = np.mean([summary[n]["avg_delta_norm"] for n in tileB_names]) if tileB_names else 0
            avg_C = np.mean([summary[n]["avg_delta_norm"] for n in tileC_names]) if tileC_names else 0
            print(f"  >> {group:>5s} aggregate: tileA_avg={avg_A:.8f}, tileB_avg={avg_B:.8f}, tileC_avg={avg_C:.8f}")

        print(f"{'='*90}")
        self.step_deltas = []
        return summary

    def final_report(self):
        print(f"\n{'#'*90}")
        print(f"  FINAL TILE WEIGHT ANALYSIS: QKV Learning Contribution")
        print(f"{'#'*90}")

        if not self.epoch_history:
            print("  No data collected!")
            return

        # Compare initial vs final weights
        final_snap = self._snapshot_weights()
        print(f"\n  Total Weight Change (initial -> final):")
        print(f"  {'Tile':<30} {'TotalDeltaNorm':>15} {'InitNorm':>12} {'FinalNorm':>12} {'RelChange%':>11}")
        print(f"  {'-'*30} {'-'*15} {'-'*12} {'-'*12} {'-'*11}")

        group_deltas = collections.defaultdict(list)
        for name in sorted(self.tiles.keys()):
            init_w = self.initial_snapshots[name]
            final_w = final_snap[name]
            delta_norm = (final_w - init_w).norm().item()
            init_norm = init_w.norm().item()
            final_norm = final_w.norm().item()
            rel_change = (delta_norm / init_norm * 100) if init_norm > 0 else 0.0

            short = name.replace("array.", "")
            print(f"  {short:<30} {delta_norm:>15.8f} {init_norm:>12.4f} {final_norm:>12.4f} {rel_change:>10.4f}%")

            # Aggregate by Q/K/V/Dense
            for g in ["Q", "K", "V", "Dense"]:
                if name.startswith(f"{g}_"):
                    if "tileA" in name:
                        group_deltas[f"{g}_A"].append(delta_norm)
                    elif "tileB" in name:
                        group_deltas[f"{g}_B"].append(delta_norm)
                    elif "tileC" in name:
                        group_deltas[f"{g}_C"].append(delta_norm)

        # Epoch-over-epoch trend
        print(f"\n  Per-Epoch Average Delta Norm (tileA + tileB only, grouped):")
        print(f"  {'Epoch':<8}", end="")
        groups = ["Q_A", "Q_B", "K_A", "K_B", "V_A", "V_B", "Dense_A", "Dense_B"]
        for g in groups:
            print(f" {g:>10}", end="")
        print()

        for eh in self.epoch_history:
            epoch = eh["epoch"]
            print(f"  {epoch:<8}", end="")
            for g in groups:
                tile_suffix = "tileA" if g.endswith("_A") else "tileB"
                prefix = g.rsplit("_", 1)[0]
                relevant = [n for n in eh["summary"] if n.startswith(f"{prefix}_") and tile_suffix in n]
                avg_val = np.mean([eh["summary"][n]["avg_delta_norm"] for n in relevant]) if relevant else 0
                print(f" {avg_val:>10.6f}", end="")
            print()

        # Verdict
        print(f"\n  VERDICT:")
        for g in ["Q", "K", "V", "Dense"]:
            a_deltas = group_deltas.get(f"{g}_A", [0])
            b_deltas = group_deltas.get(f"{g}_B", [0])
            c_deltas = group_deltas.get(f"{g}_C", [0])
            avg_ab = (np.mean(a_deltas) + np.mean(b_deltas)) / 2
            avg_c = np.mean(c_deltas)

            if avg_ab > 1e-3:
                status = "ACTIVE (significant weight updates)"
            elif avg_ab > 1e-5:
                status = "MODERATE (small but measurable updates)"
            elif avg_ab > 1e-8:
                status = "WEAK (minimal updates)"
            else:
                status = "DEAD (no weight changes)"
            print(f"    {g:>6s}: A/B avg_delta={avg_ab:.8f}, C avg_delta={avg_c:.8f} -> {status}")

        # Overall
        all_ab = []
        for g in ["Q", "K", "V", "Dense"]:
            all_ab.extend(group_deltas.get(f"{g}_A", []))
            all_ab.extend(group_deltas.get(f"{g}_B", []))
        overall_ab = np.mean(all_ab) if all_ab else 0

        print(f"\n    Overall A/B delta mean: {overall_ab:.8f}")
        if overall_ab > 1e-3:
            print(f"    QKV A/B tiles are ACTIVELY LEARNING -> contributing to training")
        elif overall_ab > 1e-5:
            print(f"    QKV A/B tiles show MODERATE updates -> some contribution")
        else:
            print(f"    QKV A/B tiles show MINIMAL/NO updates -> NOT effectively contributing")
        print(f"{'#'*90}")


# =============================================================================
# Scheduler
# =============================================================================
def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_rate=0.0):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Main
# =============================================================================
def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Task: {TASK_NAME} (MRPC)")
    print(f"Hyperparameters: lr={LEARNING_RATE}, target_ab_lr={TARGET_AB_LR}, "
          f"lora_alpha={LORA_ALPHA}, rank={RANK}")
    print(f"LoRA target: attn (query, key, value, dense)")

    lrtt_lr_multiplier = TARGET_AB_LR / (LEARNING_RATE * LORA_ALPHA)
    lrtt_lr = LEARNING_RATE * lrtt_lr_multiplier
    print(f"lr_multiplier = {lrtt_lr_multiplier:.6f}, effective LRTT LR = {lrtt_lr:.6f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_data(tokenizer)

    model, is_lrtt_target = create_model()

    # Setup tile weight monitor (tracks actual tile weights, not analog_ctx)
    tile_monitor = TileWeightMonitor(model, is_lrtt_target)

    # Also track digital params with standard gradients
    digital_grad_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad and ("classifier" in name or "LayerNorm" in name or "layer_norm" in name):
            digital_grad_params[name] = param

    # Optimizer
    optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0,
                          momentum=0.0, nesterov=False)
    optimizer.regroup_param_groups()

    lrtt_tile_ids = set()
    for m in model.modules():
        if hasattr(m, 'tile_a'):
            lrtt_tile_ids.add(id(m.tile_a))
            lrtt_tile_ids.add(id(m.tile_b))
            lrtt_tile_ids.add(id(m.tile_c))
    for group in optimizer.param_groups:
        for p in group["params"]:
            if hasattr(p, 'analog_tile') and id(p.analog_tile) in lrtt_tile_ids:
                group["lr"] = lrtt_lr
                p.analog_tile.set_learning_rate(lrtt_lr)

    num_training_steps = len(train_loader) * N_EPOCHS
    warmup_steps = int(WARMUP_RATIO * num_training_steps)
    scheduler = get_linear_schedule_with_min_lr(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    # Pre-training eval
    f1_pre, acc_pre, loss_pre = evaluate_model(model, eval_loader)
    print(f"\n  [Pre-training] F1={f1_pre:.4f}, Acc={acc_pre:.4f}, Loss={loss_pre:.4f}")

    epoch_metrics = []
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        digital_grad_norms_epoch = collections.defaultdict(list)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            model_inputs = {k: v.to(DEVICE) for k, v in batch.items()
                           if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}

            # Snapshot tile weights BEFORE step
            pre_snap = tile_monitor.pre_step_snapshot()

            optimizer.zero_grad()
            outputs = model(**model_inputs)
            loss = outputs.loss
            loss.backward()

            # Record digital param gradient norms
            for dname, dparam in digital_grad_params.items():
                if dparam.grad is not None:
                    digital_grad_norms_epoch[dname].append(dparam.grad.norm().item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Record tile weight changes AFTER step
            tile_monitor.record_step(pre_snap)

            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                print(f"\n  [NaN/Inf at batch {num_batches}] Aborting.")
                break
            total_loss += loss_val
            num_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        f1, acc, eval_loss = evaluate_model(model, eval_loader)

        print(f"\n  Epoch {epoch}: F1={f1:.4f}, Acc={acc:.4f}, "
              f"TrainLoss={train_loss:.4f}, EvalLoss={eval_loss:.4f}")

        epoch_metrics.append({
            "epoch": epoch, "f1": f1, "acc": acc,
            "train_loss": train_loss, "eval_loss": eval_loss,
        })

        # Digital param gradient summary
        print(f"\n  Digital param gradients (epoch {epoch}):")
        for dname in sorted(digital_grad_norms_epoch.keys()):
            norms = digital_grad_norms_epoch[dname]
            short = dname.split(".")[-1] if "." in dname else dname
            parent = ".".join(dname.split(".")[-2:])
            print(f"    {parent:<35s}: avg_grad_norm={np.mean(norms):.6f}, max={np.max(norms):.6f}")

        # Tile weight update summary
        tile_monitor.summarize_epoch(epoch)

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final analysis
    tile_monitor.final_report()

    # Training summary
    print(f"\n  Training Summary:")
    print(f"  {'Epoch':<8} {'F1':>8} {'Acc':>8} {'TrainLoss':>10} {'EvalLoss':>10}")
    for m in epoch_metrics:
        print(f"  {m['epoch']:<8} {m['f1']:>8.4f} {m['acc']:>8.4f} "
              f"{m['train_loss']:>10.4f} {m['eval_loss']:>10.4f}")

    f1_final = epoch_metrics[-1]["f1"] if epoch_metrics else 0.0
    print(f"\n  F1: {f1_pre:.4f} -> {f1_final:.4f} (delta={f1_final - f1_pre:+.4f})")

    # Save
    out_path = os.path.join(RESULTS_DIR, "gradient_analysis.json")
    results = {
        "hyperparams": {"lr": LEARNING_RATE, "target_ab_lr": TARGET_AB_LR,
                        "lora_alpha": LORA_ALPHA, "rank": RANK},
        "pre_training": {"f1": f1_pre, "acc": acc_pre, "loss": loss_pre},
        "epoch_metrics": epoch_metrics,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {out_path}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
