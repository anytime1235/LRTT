#!/usr/bin/env python
# coding=utf-8
"""Diagnostic training: Track weight/output distributions across epochs.

Uses Trial 42 best params, 15 epochs, with per-epoch diagnostics:
- qa_outputs (digital head) weight norm & distribution
- Analog tile weight distribution (mean, std, min, max, % at bounds)
- Out scaling alpha distribution
- Mapping scales distribution
- Output logits distribution
- Optimizer learning rate
"""

import os
import sys
import json
import re
import string
import time
from typing import Dict, List, Tuple
from collections import Counter

import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import collections
import wandb

# aihwkit imports
sys.path.insert(0, '/data/LRTT_transformer/src')
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs import (
    UnitCellRPUConfig,
    IOParameters,
    UpdateParameters,
    NoiseManagementType,
    BoundManagementType,
)
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsReferenceDevice

# ============================================================
# Best params from SQuAD sweep Trial 42 (F1=61.93)
# ============================================================
BEST_PARAMS = {
    "learning_rate": 0.00036206364180582277,
    "transfer_lr": 0.7834413109516445,
    "transfer_every": 123,
    "fast_lr": 0.17387836516374036,
    "auto_granularity": 516.6976964566325,
    "in_chop_prob": 0.04734457410734495,
}

MODEL_NAME = "google/mobilebert-uncased"
TARGET_MODULES = ["query", "key", "value"]
BATCH_SIZE = 32
WARMUP_STEPS = 500
NUM_EPOCHS = 15
SEED = 42
MAX_SEQ_LENGTH = 384


# ============================================================
# SQuAD helpers (unchanged)
# ============================================================
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def postprocess_squad_predictions(examples, features, all_start_logits, all_end_logits,
                                   n_best_size=20, max_answer_length=30):
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    all_predictions = collections.OrderedDict()
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]
        prelim_predictions = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (start_index >= len(offset_mapping) or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None or offset_mapping[end_index] is None):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    prelim_predictions.append({
                        "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                        "score": start_logits[start_index] + end_logits[end_index],
                    })
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]
        if len(predictions) == 0:
            all_predictions[example["id"]] = ""
        else:
            best_pred = predictions[0]
            start_char, end_char = best_pred["offsets"]
            all_predictions[example["id"]] = context[start_char:end_char]
    return all_predictions


# ============================================================
# Config & Model creation (unchanged)
# ============================================================
def create_config() -> UnitCellRPUConfig:
    sixt1c_device = LinearStepDevice(
        dw_min=0.001981, gamma_up=-0.1678, gamma_down=0.1410,
        dw_min_dtod=0.1, up_down_dtod=0.01, w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05, dw_min_std=0.3,
        write_noise_std=0.0, mult_noise=True, mean_bound_reference=True, lifetime=0.0,
    )
    softbounds_device = SoftBoundsReferenceDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, write_noise_std=0.0, mult_noise=True,
    )
    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[sixt1c_device, softbounds_device],
            transfer_every=BEST_PARAMS["transfer_every"],
            units_in_mbatch=False, n_reads_per_transfer=1, transfer_columns=True,
            gamma=0.0, transfer_lr=BEST_PARAMS["transfer_lr"],
            fast_lr=BEST_PARAMS["fast_lr"], scale_transfer_lr=True,
            auto_scale=True, auto_granularity=BEST_PARAMS["auto_granularity"],
            buffer_granularity=1.0, auto_momentum=0.99,
            in_chop_prob=BEST_PARAMS["in_chop_prob"], in_chop_random=True,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=1, update_bl_management=False, update_management=False,
            ),
        )
    )
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


def create_model(rpu_config, device):
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    all_linear = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("qa_outputs")
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)
    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        param.requires_grad = is_target or "qa_outputs" in name
    return model.to(device)


def load_data(tokenizer):
    raw_datasets = load_dataset("squad")
    eval_examples = raw_datasets["validation"].select(range(min(2000, len(raw_datasets["validation"]))))

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(questions, examples["context"], max_length=384,
                          truncation="only_second", stride=128,
                          return_overflowing_tokens=True, return_offsets_mapping=True,
                          padding="max_length")
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions, end_positions = [], []
        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            if len(answer["answer_start"]) == 0:
                start_positions.append(0); end_positions.append(0); continue
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
            idx = 0
            while sequence_ids[idx] != 1: idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1: idx += 1
            context_end = idx - 1
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0); end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char: idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char: idx -= 1
                end_positions.append(idx + 1)
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_eval(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(questions, examples["context"], max_length=384,
                          truncation="only_second", stride=128,
                          return_overflowing_tokens=True, return_offsets_mapping=True,
                          padding="max_length")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs["offset_mapping"]
        for i in range(len(inputs["input_ids"])):
            sequence_ids = inputs.sequence_ids(i)
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset_mapping[i])
            ]
        inputs["example_id"] = [examples["id"][sample_map[i]] for i in range(len(inputs["input_ids"]))]
        return inputs

    tokenized_train = raw_datasets["train"].map(preprocess_train, batched=True,
                                                 remove_columns=raw_datasets["train"].column_names)
    train_subset = tokenized_train.shuffle(seed=SEED).select(range(min(10000, len(tokenized_train))))
    tokenized_eval = eval_examples.map(preprocess_eval, batched=True,
                                        remove_columns=raw_datasets["validation"].column_names)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=default_data_collator)
    return train_loader, tokenized_eval, eval_examples


def evaluate_squad(model, eval_features, eval_examples, device):
    model.eval()
    all_start_logits, all_end_logits = [], []
    def collate_fn(features):
        offset_mappings = [f.pop("offset_mapping") for f in features]
        example_ids = [f.pop("example_id") for f in features]
        batch = default_data_collator(features)
        batch["offset_mapping"] = offset_mappings
        batch["example_id"] = example_ids
        for i, f in enumerate(features):
            f["offset_mapping"] = offset_mappings[i]
            f["example_id"] = example_ids[i]
        return batch
    eval_loader = DataLoader(eval_features, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())
    model.train()
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)
    predictions = postprocess_squad_predictions(eval_examples, eval_features,
                                                 all_start_logits, all_end_logits)
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)
    return results["f1"], results["exact_match"], all_start_logits, all_end_logits


# ============================================================
# Diagnostic functions
# ============================================================
def collect_diagnostics(model, epoch, all_start_logits=None, all_end_logits=None):
    """Collect weight/output distribution diagnostics."""
    diag = {"epoch": epoch}

    # 1. qa_outputs (digital head) diagnostics
    for name, param in model.named_parameters():
        if "qa_outputs" in name and "weight" in name:
            w = param.data.cpu()
            diag["qa_weight_mean"] = w.mean().item()
            diag["qa_weight_std"] = w.std().item()
            diag["qa_weight_norm"] = w.norm().item()
            diag["qa_weight_min"] = w.min().item()
            diag["qa_weight_max"] = w.max().item()
            diag["qa_weight_abs_mean"] = w.abs().mean().item()
        if "qa_outputs" in name and "bias" in name:
            b = param.data.cpu()
            diag["qa_bias_mean"] = b.mean().item()
            diag["qa_bias_std"] = b.std().item()
            diag["qa_bias_vals"] = b.tolist()

    # 2. Analog tile weight diagnostics (aggregate across all tiles)
    all_weights_flat = []
    tile_stats = []
    for tname, tile in model.named_analog_tiles():
        # Get weights from weight_tile (the slow/inference tile)
        w = tile.get_weights()[0].cpu()
        all_weights_flat.append(w.flatten())

        w_flat = w.flatten()
        w_min_device = -1.0  # SoftBoundsReferenceDevice w_min
        w_max_device = 1.0   # SoftBoundsReferenceDevice w_max
        at_lower = (w_flat <= w_min_device * 0.95).float().mean().item()
        at_upper = (w_flat >= w_max_device * 0.95).float().mean().item()

        tile_stats.append({
            "name": tname.split(".")[-3] + "." + tname.split(".")[-2],  # e.g., "layer.0.query"
            "mean": w.mean().item(),
            "std": w.std().item(),
            "min": w.min().item(),
            "max": w.max().item(),
            "at_lower_bound_%": at_lower * 100,
            "at_upper_bound_%": at_upper * 100,
        })

    if all_weights_flat:
        all_w = torch.cat(all_weights_flat)
        diag["analog_weight_mean"] = all_w.mean().item()
        diag["analog_weight_std"] = all_w.std().item()
        diag["analog_weight_min"] = all_w.min().item()
        diag["analog_weight_max"] = all_w.max().item()
        diag["analog_weight_abs_mean"] = all_w.abs().mean().item()
        diag["analog_at_lower_%"] = (all_w <= -0.95).float().mean().item() * 100
        diag["analog_at_upper_%"] = (all_w >= 0.95).float().mean().item() * 100

    # 3. Out scaling alpha diagnostics
    all_alphas = []
    for tname, tile in model.named_analog_tiles():
        alpha = tile.get_learned_out_scales()
        if alpha is not None:
            all_alphas.append(alpha.detach().cpu().flatten())
    if all_alphas:
        all_a = torch.cat(all_alphas)
        diag["out_scaling_mean"] = all_a.mean().item()
        diag["out_scaling_std"] = all_a.std().item()
        diag["out_scaling_min"] = all_a.min().item()
        diag["out_scaling_max"] = all_a.max().item()

    # 4. Mapping scales diagnostics
    all_mscales = []
    for tname, tile in model.named_analog_tiles():
        ms = tile.get_mapping_scales()
        if ms is not None:
            all_mscales.append(ms.detach().cpu().flatten())
    if all_mscales:
        all_ms = torch.cat(all_mscales)
        diag["mapping_scales_mean"] = all_ms.mean().item()
        diag["mapping_scales_std"] = all_ms.std().item()
        diag["mapping_scales_min"] = all_ms.min().item()
        diag["mapping_scales_max"] = all_ms.max().item()

    # 5. Output logits diagnostics
    if all_start_logits is not None:
        sl = all_start_logits.flatten()
        el = all_end_logits.flatten()
        diag["start_logits_mean"] = float(np.mean(sl))
        diag["start_logits_std"] = float(np.std(sl))
        diag["start_logits_min"] = float(np.min(sl))
        diag["start_logits_max"] = float(np.max(sl))
        diag["start_logits_abs_mean"] = float(np.mean(np.abs(sl)))
        diag["end_logits_mean"] = float(np.mean(el))
        diag["end_logits_std"] = float(np.std(el))
        diag["end_logits_min"] = float(np.min(el))
        diag["end_logits_max"] = float(np.max(el))
        diag["end_logits_abs_mean"] = float(np.mean(np.abs(el)))

    # Sample tile stats (first 3 layers, all Q/K/V)
    diag["tile_samples"] = tile_stats[:9]

    return diag


def print_diagnostics(diag):
    """Print diagnostics in a readable format."""
    epoch = diag["epoch"]
    print(f"\n  --- Diagnostics Epoch {epoch} ---")

    # qa_outputs
    print(f"  [qa_outputs] weight: mean={diag.get('qa_weight_mean', 0):.6f}, "
          f"std={diag.get('qa_weight_std', 0):.6f}, norm={diag.get('qa_weight_norm', 0):.4f}, "
          f"range=[{diag.get('qa_weight_min', 0):.4f}, {diag.get('qa_weight_max', 0):.4f}]")
    if "qa_bias_vals" in diag:
        print(f"  [qa_outputs] bias: {[f'{v:.4f}' for v in diag['qa_bias_vals']]}")

    # Analog weights
    print(f"  [analog tiles] weight: mean={diag.get('analog_weight_mean', 0):.6f}, "
          f"std={diag.get('analog_weight_std', 0):.6f}, "
          f"range=[{diag.get('analog_weight_min', 0):.4f}, {diag.get('analog_weight_max', 0):.4f}]")
    print(f"  [analog tiles] at bounds: lower={diag.get('analog_at_lower_%', 0):.2f}%, "
          f"upper={diag.get('analog_at_upper_%', 0):.2f}%")

    # Out scaling
    print(f"  [out_scaling] mean={diag.get('out_scaling_mean', 0):.6f}, "
          f"std={diag.get('out_scaling_std', 0):.6f}, "
          f"range=[{diag.get('out_scaling_min', 0):.4f}, {diag.get('out_scaling_max', 0):.4f}]")

    # Mapping scales
    print(f"  [mapping_scales] mean={diag.get('mapping_scales_mean', 0):.6f}, "
          f"std={diag.get('mapping_scales_std', 0):.6f}, "
          f"range=[{diag.get('mapping_scales_min', 0):.4f}, {diag.get('mapping_scales_max', 0):.4f}]")

    # Output logits
    if "start_logits_mean" in diag:
        print(f"  [start_logits] mean={diag['start_logits_mean']:.4f}, "
              f"std={diag['start_logits_std']:.4f}, "
              f"range=[{diag['start_logits_min']:.2f}, {diag['start_logits_max']:.2f}]")
        print(f"  [end_logits]   mean={diag['end_logits_mean']:.4f}, "
              f"std={diag['end_logits_std']:.4f}, "
              f"range=[{diag['end_logits_min']:.2f}, {diag['end_logits_max']:.2f}]")


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Best params (Trial 42, F1=61.93): {BEST_PARAMS}")
    print(f"Epochs: {NUM_EPOCHS}")

    run = wandb.init(
        project="tikitaka-v2-squad-diagnostic",
        name=f"squad_trial42_15ep_diagnostic",
        config={
            "task": "squad",
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "warmup_steps": WARMUP_STEPS,
            "seed": SEED,
            "model": MODEL_NAME,
            "mapping_enabled": True,
            **BEST_PARAMS,
        },
    )

    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Loading data...")
    train_loader, eval_features, eval_examples = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    rpu_config = create_config()
    model = create_model(rpu_config, device)
    optimizer = AnalogAdam(model.parameters(), lr=BEST_PARAMS["learning_rate"])
    optimizer.regroup_param_groups(model)

    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                                 num_training_steps=num_training_steps)

    all_diagnostics = []

    # Initial eval + diagnostics
    init_f1, init_em, start_logits, end_logits = evaluate_squad(model, eval_features, eval_examples, device)
    print(f"[Epoch 0] F1: {init_f1:.2f}, EM: {init_em:.2f}")

    diag = collect_diagnostics(model, 0, start_logits, end_logits)
    diag["f1"] = init_f1
    diag["em"] = init_em
    diag["lr"] = optimizer.param_groups[0]["lr"]
    print_diagnostics(diag)
    all_diagnostics.append(diag)
    wandb.log({"epoch": 0, "eval/f1": init_f1, "eval/em": init_em, **{f"diag/{k}": v for k, v in diag.items() if isinstance(v, (int, float))}})

    best_f1 = init_f1

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, num_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=True)
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / num_batches
        f1, em, start_logits, end_logits = evaluate_squad(model, eval_features, eval_examples, device)

        # Get current lr
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}, F1: {f1:.2f}, EM: {em:.2f}, LR: {current_lr:.8f}")

        # Collect diagnostics
        diag = collect_diagnostics(model, epoch, start_logits, end_logits)
        diag["f1"] = f1
        diag["em"] = em
        diag["loss"] = avg_loss
        diag["lr"] = current_lr
        print_diagnostics(diag)
        all_diagnostics.append(diag)

        wandb.log({
            "epoch": epoch,
            "train/loss": avg_loss,
            "eval/f1": f1,
            "eval/em": em,
            "lr": current_lr,
            **{f"diag/{k}": v for k, v in diag.items() if isinstance(v, (int, float))},
        })

        if f1 > best_f1:
            best_f1 = f1
            print(f"  >> New best F1: {best_f1:.2f}")

    # Summary table
    print(f"\n{'='*120}")
    print(f"{'Ep':>3} | {'F1':>6} | {'EM':>6} | {'Loss':>8} | {'LR':>10} | "
          f"{'QA_norm':>8} | {'AW_mean':>8} | {'AW_std':>8} | {'Bound%':>7} | "
          f"{'OS_mean':>8} | {'OS_std':>8} | {'SL_std':>7} | {'SL_max':>8}")
    print("-"*120)
    for d in all_diagnostics:
        bound_pct = d.get('analog_at_lower_%', 0) + d.get('analog_at_upper_%', 0)
        print(f"{d['epoch']:>3d} | {d.get('f1',0):>6.2f} | {d.get('em',0):>6.2f} | "
              f"{d.get('loss',0):>8.4f} | {d.get('lr',0):>10.8f} | "
              f"{d.get('qa_weight_norm',0):>8.4f} | {d.get('analog_weight_mean',0):>8.6f} | "
              f"{d.get('analog_weight_std',0):>8.6f} | {bound_pct:>6.2f}% | "
              f"{d.get('out_scaling_mean',0):>8.4f} | {d.get('out_scaling_std',0):>8.6f} | "
              f"{d.get('start_logits_std',0):>7.2f} | {d.get('start_logits_max',0):>8.2f}")

    # Save results
    output_path = "/data/AIMC_LoRA_results/tikitaka_sweep/squad_trial42_diagnostic_results.json"
    with open(output_path, 'w') as f:
        # Convert non-serializable items
        safe_diags = []
        for d in all_diagnostics:
            safe_d = {}
            for k, v in d.items():
                if isinstance(v, (int, float, str, list)):
                    safe_d[k] = v
                elif isinstance(v, (np.floating, np.integer)):
                    safe_d[k] = float(v)
            safe_diags.append(safe_d)
        json.dump(safe_diags, f, indent=2)
    print(f"\nDiagnostics saved to: {output_path}")

    wandb.finish()
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
