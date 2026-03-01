# -*- coding: utf-8 -*-
"""Gradient & Weight-Update Monitor for ALBERT QKV (TikiTaka v1 on MRPC).

Measures whether QKV attention layers are actually being updated by:
1. Recording analog tile weights before/after each optimizer step
2. Computing weight delta norms (L2) per layer
3. Tracking AnalogContext gradient norms
4. Printing per-step summary table

Usage:
    python gradient_monitor_qkv.py
"""

import os
import sys
import copy
import torch
from torch import nn
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

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs import (
    SingleRPUConfig, UnitCellRPUConfig,
    IOParameters, UpdateParameters,
)
from aihwkit.simulator.configs.compounds import TransferCompound
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType
from aihwkit.optim.context import AnalogContext

os.environ["WANDB_MODE"] = "offline"

# =============================================================================
# Config
# =============================================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "albert/albert-base-v2"
TASK_NAME = "mrpc"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
N_EPOCHS = 3
WARMUP_RATIO = 0.1

# Hyperparameters (user-specified)
LEARNING_RATE = 0.1
TRANSFER_LR = 0.1      # target_ab lr
FAST_LR = 1.0           # lora alpha
TRANSFER_EVERY = 1

# Monitor settings
MONITOR_EVERY_N_STEPS = 10   # Print gradient stats every N steps
NUM_MONITOR_STEPS = None      # None = all steps


# =============================================================================
# TikiTaka Device (same as optuna script)
# =============================================================================

def _create_a_device():
    return LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
        write_noise_std=0.0, mean_bound_reference=True,
        lifetime=0.0, lifetime_dtod=0.0, reset=0.0, reset_dtod=0.0,
    )

def _create_b_device():
    return SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0, up_down_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0, write_noise_std=0.0, mult_noise=False,
    )

def create_tikitaka_config():
    rpu_config = UnitCellRPUConfig(
        device=TransferCompound(
            unit_cell_devices=[_create_a_device(), _create_b_device()],
            transfer_every=TRANSFER_EVERY,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=TRANSFER_LR,
            fast_lr=FAST_LR,
            scale_transfer_lr=True,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(),
        )
    )
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config

def create_single_rpu_config():
    rpu_config = SingleRPUConfig(device=_create_b_device())
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


# =============================================================================
# Model Creation (target=attn)
# =============================================================================

def create_model():
    """Create ALBERT with attn layers -> TikiTaka, ffn layers -> SingleRPU (frozen)."""
    num_labels = 2  # MRPC is binary classification
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    # Reinit classifier
    torch.manual_seed(SEED)
    nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
    if model.classifier.bias is not None:
        nn.init.zeros_(model.classifier.bias)

    all_linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    always_digital = ["classifier", "albert.encoder.embedding_hidden_mapping_in"]
    target_patterns = ["attention"]  # attn target

    tikitaka_layers = [
        n for n in all_linear_names
        if "encoder" in n
        and not any(d in n for d in always_digital)
        and any(p in n for p in target_patterns)
    ]
    non_target_layers = [
        n for n in all_linear_names
        if n not in tikitaka_layers and "encoder" in n
        and not any(d in n for d in always_digital)
    ]

    print(f"\n{'='*70}")
    print(f"MODEL ARCHITECTURE (target=attn)")
    print(f"{'='*70}")
    print(f"  TikiTaka (QKV+Dense): {tikitaka_layers}")
    print(f"  SingleRPU (FFN):      {non_target_layers}")
    print(f"  Digital:              {always_digital}")

    # Pass 1: Convert attention layers -> TikiTaka
    if tikitaka_layers:
        tiki_config = create_tikitaka_config()
        tiki_exclude = [n for n in all_linear_names if n not in tikitaka_layers]
        model = convert_to_analog(model, tiki_config, exclude_modules=tiki_exclude)
    tikitaka_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))

    # Pass 2: Convert FFN layers -> SingleRPU (frozen)
    if non_target_layers:
        single_config = create_single_rpu_config()
        single_exclude = [n for n in all_linear_names if n not in non_target_layers]
        model = convert_to_analog(model, single_config, exclude_modules=single_exclude)
        single_rpu_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear)) - tikitaka_count

        # Freeze SingleRPU tile weights
        def _frozen_noop_update(x_input, d_input, *args, **kwargs):
            return None
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    if isinstance(tile.rpu_config, SingleRPUConfig):
                        tile.update = _frozen_noop_update
    else:
        single_rpu_count = 0

    print(f"  TikiTaka layers: {tikitaka_count}, Frozen analog: {single_rpu_count}")

    # Set requires_grad
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            param.requires_grad = True
        elif "classifier" in name:
            param.requires_grad = True
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
        elif "out_scaling" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / Total: {total:,}")

    return model.to(DEVICE)


# =============================================================================
# Data
# =============================================================================

def load_data(tokenizer):
    raw = load_dataset("nyu-mll/glue", TASK_NAME)
    s1_key, s2_key = "sentence1", "sentence2"

    def preprocess(examples):
        return tokenizer(examples[s1_key], examples[s2_key],
                         max_length=MAX_SEQ_LENGTH, truncation=True)

    remove_cols = [c for c in raw["train"].column_names if c != "label"]
    tokenized = raw.map(preprocess, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        tokenized["train"], batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, generator=torch.Generator().manual_seed(SEED),
    )
    eval_loader = DataLoader(
        tokenized["validation"], batch_size=64, shuffle=False, collate_fn=collator,
    )
    return train_loader, eval_loader


# =============================================================================
# Gradient & Weight Monitor
# =============================================================================

def get_analog_tile_info(model):
    """Get all analog tiles with their names and types."""
    tiles = []
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            for tile in module.analog_tiles():
                is_tikitaka = isinstance(tile.rpu_config, UnitCellRPUConfig)
                tiles.append({
                    'name': name,
                    'tile': tile,
                    'is_tikitaka': is_tikitaka,
                    'type': 'TikiTaka' if is_tikitaka else 'SingleRPU',
                })
    return tiles


def snapshot_weights(tile_info_list):
    """Take a snapshot of all tile weights."""
    snapshots = {}
    for info in tile_info_list:
        tile = info['tile']
        try:
            w, b = tile.get_weights()
            snapshots[info['name']] = w.clone().detach().cpu()
        except Exception:
            snapshots[info['name']] = None
    return snapshots


def compute_weight_deltas(before, after, tile_info_list):
    """Compute weight change norms (L2) between snapshots."""
    deltas = {}
    for info in tile_info_list:
        name = info['name']
        if before.get(name) is not None and after.get(name) is not None:
            delta = after[name] - before[name]
            deltas[name] = {
                'l2_norm': torch.norm(delta).item(),
                'max_abs': torch.max(torch.abs(delta)).item(),
                'mean_abs': torch.mean(torch.abs(delta)).item(),
                'fro_ratio': torch.norm(delta).item() / (torch.norm(before[name]).item() + 1e-12),
            }
        else:
            deltas[name] = {'l2_norm': 0, 'max_abs': 0, 'mean_abs': 0, 'fro_ratio': 0}
    return deltas


def get_analog_context_grads(model):
    """Get gradient norms for AnalogContext parameters."""
    grads = {}
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext) and param.grad is not None:
            grads[name] = {
                'grad_l2': torch.norm(param.grad).item(),
                'grad_max': torch.max(torch.abs(param.grad)).item(),
            }
    return grads


def get_digital_param_grads(model):
    """Get gradient norms for digital trainable parameters."""
    grads = {}
    for name, param in model.named_parameters():
        if not isinstance(param, AnalogContext) and param.requires_grad and param.grad is not None:
            grads[name] = {
                'grad_l2': torch.norm(param.grad).item(),
                'grad_max': torch.max(torch.abs(param.grad)).item(),
            }
    return grads


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, eval_loader):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in eval_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            out = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = out.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    from sklearn.metrics import f1_score
    # Re-run for F1
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            out = model(input_ids=ids, attention_mask=mask)
            all_preds.extend(out.logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    model.train()
    f1 = f1_score(all_labels, all_preds)
    acc = correct / total if total > 0 else 0
    avg_loss = total_loss / total if total > 0 else 0
    return acc, f1, avg_loss


# =============================================================================
# Main Training Loop with Gradient Monitoring
# =============================================================================

def main():
    set_seed(SEED)
    print(f"\n{'#'*70}")
    print(f"# GRADIENT MONITOR: ALBERT QKV on MRPC")
    print(f"# lr={LEARNING_RATE}, transfer_lr={TRANSFER_LR}, fast_lr={FAST_LR}")
    print(f"# transfer_every={TRANSFER_EVERY}, batch_size={BATCH_SIZE}")
    print(f"# epochs={N_EPOCHS}, warmup_ratio={WARMUP_RATIO}")
    print(f"# device={DEVICE}")
    print(f"{'#'*70}\n")

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_data(tokenizer)
    print(f"Train: {len(train_loader)} batches, Eval: {len(eval_loader)} batches")

    # Create model
    model = create_model()

    # Optimizer
    optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    optimizer.regroup_param_groups()

    # Scheduler
    total_steps = len(train_loader) * N_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    # Get tile info for monitoring
    tile_info = get_analog_tile_info(model)
    print(f"\n{'='*70}")
    print(f"ANALOG TILES DETECTED:")
    print(f"{'='*70}")
    for info in tile_info:
        print(f"  [{info['type']:10s}] {info['name']}")

    # Accumulators for summary
    all_step_data = []

    # Initial eval
    acc, f1, eval_loss = evaluate(model, eval_loader)
    print(f"\n[INIT] Acc={acc:.4f}, F1={f1:.4f}, Loss={eval_loss:.4f}")

    global_step = 0
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS}")
        for batch in pbar:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # --- BEFORE step: snapshot weights ---
            w_before = snapshot_weights(tile_info)

            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=mask)
            loss = criterion(out.logits, labels)
            loss.backward()

            # --- Capture gradients BEFORE clip ---
            ctx_grads_pre_clip = get_analog_context_grads(model)
            digital_grads = get_digital_param_grads(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # --- Capture gradients AFTER clip ---
            ctx_grads_post_clip = get_analog_context_grads(model)

            optimizer.step()
            scheduler.step()
            global_step += 1

            # --- AFTER step: snapshot weights ---
            w_after = snapshot_weights(tile_info)
            deltas = compute_weight_deltas(w_before, w_after, tile_info)

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Store step data
            step_record = {
                'step': global_step,
                'epoch': epoch,
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'weight_deltas': deltas,
                'ctx_grads_pre_clip': ctx_grads_pre_clip,
                'ctx_grads_post_clip': ctx_grads_post_clip,
                'digital_grads': digital_grads,
            }
            all_step_data.append(step_record)

            # Print detailed report every N steps
            if global_step % MONITOR_EVERY_N_STEPS == 0 or global_step <= 3:
                print(f"\n{'─'*70}")
                print(f"STEP {global_step} | Epoch {epoch} | Loss={loss.item():.4f} | LR={step_record['lr']:.6f}")
                print(f"{'─'*70}")

                # Weight deltas (actual tile weight changes)
                print(f"\n  ▶ WEIGHT DELTAS (tile weight change after optimizer.step):")
                print(f"  {'Layer':<55s} {'Type':<10s} {'ΔW L2':>12s} {'ΔW MaxAbs':>12s} {'ΔW/W ratio':>12s}")
                print(f"  {'─'*101}")
                for info in tile_info:
                    d = deltas[info['name']]
                    print(f"  {info['name']:<55s} {info['type']:<10s} "
                          f"{d['l2_norm']:>12.6e} {d['max_abs']:>12.6e} {d['fro_ratio']:>12.6e}")

                # AnalogContext gradients
                print(f"\n  ▶ ANALOG CONTEXT GRADIENTS (drives tile update):")
                print(f"  {'Param':<55s} {'PreClip L2':>12s} {'PostClip L2':>12s}")
                print(f"  {'─'*79}")
                for pname in sorted(ctx_grads_pre_clip.keys()):
                    pre = ctx_grads_pre_clip[pname]
                    post = ctx_grads_post_clip.get(pname, {'grad_l2': 0})
                    print(f"  {pname:<55s} {pre['grad_l2']:>12.6e} {post['grad_l2']:>12.6e}")

                # Digital parameter gradients (classifier, LayerNorm, out_scaling)
                print(f"\n  ▶ DIGITAL PARAM GRADIENTS (classifier, LayerNorm, out_scaling):")
                print(f"  {'Param':<55s} {'Grad L2':>12s} {'Grad Max':>12s}")
                print(f"  {'─'*79}")
                for pname in sorted(digital_grads.keys()):
                    g = digital_grads[pname]
                    # Truncate long names
                    short = pname if len(pname) < 55 else "..." + pname[-52:]
                    print(f"  {short:<55s} {g['grad_l2']:>12.6e} {g['grad_max']:>12.6e}")

        # Epoch summary
        avg_train_loss = epoch_loss / num_batches
        acc, f1, eval_loss = evaluate(model, eval_loader)
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} SUMMARY | TrainLoss={avg_train_loss:.4f} | EvalLoss={eval_loss:.4f} | Acc={acc:.4f} | F1={f1:.4f}")
        print(f"{'='*70}")

    # ==========================================================================
    # FINAL ANALYSIS
    # ==========================================================================
    print(f"\n\n{'#'*70}")
    print(f"# FINAL GRADIENT ANALYSIS SUMMARY")
    print(f"{'#'*70}")

    # Aggregate weight delta statistics per layer across all steps
    print(f"\n{'='*70}")
    print(f"WEIGHT UPDATE STATISTICS (across all {global_step} steps)")
    print(f"{'='*70}")
    print(f"{'Layer':<55s} {'Type':<10s} {'Mean ΔW':>12s} {'Std ΔW':>12s} {'Max ΔW':>12s} {'Zero%':>8s}")
    print(f"{'─'*107}")

    for info in tile_info:
        name = info['name']
        l2_norms = [s['weight_deltas'][name]['l2_norm'] for s in all_step_data]
        mean_dw = np.mean(l2_norms)
        std_dw = np.std(l2_norms)
        max_dw = np.max(l2_norms)
        zero_pct = 100.0 * sum(1 for x in l2_norms if x < 1e-10) / len(l2_norms)

        print(f"{name:<55s} {info['type']:<10s} "
              f"{mean_dw:>12.6e} {std_dw:>12.6e} {max_dw:>12.6e} {zero_pct:>7.1f}%")

    # AnalogContext gradient stats
    print(f"\n{'='*70}")
    print(f"ANALOG CONTEXT GRADIENT STATISTICS (across all {global_step} steps)")
    print(f"{'='*70}")
    all_ctx_names = set()
    for s in all_step_data:
        all_ctx_names.update(s['ctx_grads_post_clip'].keys())
    all_ctx_names = sorted(all_ctx_names)

    print(f"{'Param':<55s} {'Mean Grad':>12s} {'Std Grad':>12s} {'Max Grad':>12s}")
    print(f"{'─'*91}")
    for pname in all_ctx_names:
        grads = [s['ctx_grads_post_clip'].get(pname, {}).get('grad_l2', 0) for s in all_step_data]
        print(f"{pname:<55s} {np.mean(grads):>12.6e} {np.std(grads):>12.6e} {np.max(grads):>12.6e}")

    # Diagnosis
    print(f"\n{'='*70}")
    print(f"DIAGNOSIS: Are QKV layers actually learning?")
    print(f"{'='*70}")

    tikitaka_deltas = []
    singlerpu_deltas = []
    for info in tile_info:
        name = info['name']
        mean_dw = np.mean([s['weight_deltas'][name]['l2_norm'] for s in all_step_data])
        if info['is_tikitaka']:
            tikitaka_deltas.append((name, mean_dw))
        else:
            singlerpu_deltas.append((name, mean_dw))

    print(f"\n  TikiTaka (QKV+Dense) mean weight changes:")
    total_tiki_update = 0
    for name, dw in tikitaka_deltas:
        status = "UPDATING" if dw > 1e-8 else "STATIC (NOT LEARNING)"
        total_tiki_update += dw
        print(f"    {name}: ΔW={dw:.6e} -> {status}")

    print(f"\n  SingleRPU (FFN, frozen) mean weight changes:")
    total_single_update = 0
    for name, dw in singlerpu_deltas:
        status = "FROZEN OK" if dw < 1e-8 else "WARNING: SHOULD BE FROZEN"
        total_single_update += dw
        print(f"    {name}: ΔW={dw:.6e} -> {status}")

    print(f"\n  SUMMARY:")
    print(f"    QKV total mean ΔW: {total_tiki_update:.6e}")
    print(f"    FFN total mean ΔW: {total_single_update:.6e}")

    if total_tiki_update > 1e-8:
        print(f"\n    ✓ QKV attention layers ARE being updated via TikiTaka.")
        ratio = total_tiki_update / (total_single_update + 1e-15)
        print(f"    QKV/FFN update ratio: {ratio:.2f}x")
        print(f"    -> QKV learning IS contributing to training.")
    else:
        print(f"\n    ✗ QKV attention layers are NOT being updated!")
        print(f"    -> Check transfer_lr, fast_lr, and learning rate settings.")

    # Save raw data
    import json
    summary = {
        'config': {
            'lr': LEARNING_RATE, 'transfer_lr': TRANSFER_LR,
            'fast_lr': FAST_LR, 'transfer_every': TRANSFER_EVERY,
            'batch_size': BATCH_SIZE, 'epochs': N_EPOCHS,
        },
        'per_layer': {},
    }
    for info in tile_info:
        name = info['name']
        l2s = [s['weight_deltas'][name]['l2_norm'] for s in all_step_data]
        summary['per_layer'][name] = {
            'type': info['type'],
            'mean_dw_l2': float(np.mean(l2s)),
            'std_dw_l2': float(np.std(l2s)),
            'max_dw_l2': float(np.max(l2s)),
        }

    out_path = "/data/results/tikitakav1/gradient_monitor_qkv_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
