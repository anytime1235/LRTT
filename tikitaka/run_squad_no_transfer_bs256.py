#!/usr/bin/env python
# coding=utf-8
"""SQuAD ablation: TikiTaka v2 best config with transfer disabled.

Based on sweep_squad_15ep_bs256.py best Trial 44 (F1=74.42).
Runs two ablations:
  1. transfer_lr=0    (transfer events fire but write 0)
  2. transfer_every=0 (transfer events never fire)
"""

import os
import sys
import json
import re
import string
import math
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter

import torch
import torch.nn as nn
from tqdm import tqdm

import wandb

from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate
import numpy as np
import collections

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


# =============================================================================
# Fixed Parameters (same as sweep_squad_15ep_bs256.py)
# =============================================================================

NUM_EPOCHS = 15
PATIENCE = 3
TARGET_MODULES = ["query", "key", "value"]
MODEL_NAME = "google/mobilebert-uncased"
BATCH_SIZE = 256
MIN_LR_RATE = 1.0 / 20.0
SEED = 42
LOSS_EXPLOSION_THRESHOLD = 1e7

WANDB_PROJECT = "tikitaka-squad-no-transfer-ablation"
OUTPUT_DIR = "/data/LRTT_transformer/tikitaka/results"

# Best config from sweep_squad_15ep_bs256.py Trial 44 (F1=74.42)
BASE_PARAMS = {
    "learning_rate": 0.0012609332577067113,
    "transfer_lr": 0.050069056748703634,
    "fast_lr": 2.192781614646745,
    "transfer_every": 20,
    "auto_granularity": 169.0,
    "in_chop_prob": 0.061,
}

ABLATIONS = {
    "transfer_lr_0": {
        **BASE_PARAMS,
        "transfer_lr": 0.0,           # transfer events fire but write 0
    },
    "transfer_every_0": {
        **BASE_PARAMS,
        "transfer_every": 0,          # transfer events never fire
    },
}


# =============================================================================
# LR Schedule
# =============================================================================

def get_linear_schedule_with_min_lr(optimizer, num_training_steps, min_lr_rate=0.05):
    def lr_lambda(current_step):
        progress = float(current_step) / float(max(1, num_training_steps))
        return min_lr_rate + (1.0 - min_lr_rate) * (1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# TikiTaka v2 Config
# =============================================================================

def create_tikitaka_v2_config(
    transfer_every, transfer_lr, fast_lr, auto_granularity, in_chop_prob,
):
    sixt1c_device = LinearStepDevice(
        dw_min=0.001981, gamma_up=-0.1678, gamma_down=0.1410,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mult_noise=True, mean_bound_reference=True, lifetime=0.0,
    )

    softbounds_device = SoftBoundsReferenceDevice(
        w_max=1.0, w_min=-1.0, dw_min=0.001,
        dw_min_std=0.0, write_noise_std=0.0, diffusion=0.0,
        dw_min_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        up_down=0.0, up_down_dtod=0.0,
        lifetime=0.0, lifetime_dtod=0.0,
        slope_up_dtod=0.0, slope_down_dtod=0.0,
    )

    rpu_config = UnitCellRPUConfig(
        device=ChoppedTransferCompound(
            unit_cell_devices=[sixt1c_device, softbounds_device],
            transfer_every=transfer_every,
            units_in_mbatch=False,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
            scale_transfer_lr=True,
            auto_scale=True,
            auto_granularity=auto_granularity,
            buffer_granularity=1.0,
            auto_momentum=0.99,
            in_chop_prob=in_chop_prob,
            in_chop_random=True,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(
                desired_bl=1,
                update_bl_management=False,
                update_management=False,
            ),
        )
    )

    # Forward/Backward IO — noise-free, ABS_MAX / ITERATIVE
    rpu_config.forward = IOParameters(
        is_perfect=False,
        inp_noise=0.0,
        out_noise=0.0,
        out_noise_std=0.0,
        w_noise=0.0,
        noise_management=NoiseManagementType.ABS_MAX,
        bound_management=BoundManagementType.ITERATIVE,
    )
    rpu_config.backward = IOParameters(
        is_perfect=False,
        inp_noise=0.0,
        out_noise=0.0,
        out_noise_std=0.0,
        w_noise=0.0,
        noise_management=NoiseManagementType.ABS_MAX,
        bound_management=BoundManagementType.ITERATIVE,
    )

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


def list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


# =============================================================================
# SQuAD Model & Data (identical to sweep_squad_15ep_bs256.py)
# =============================================================================

def create_squad_model(params, device):
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("qa_outputs")

    rpu_config = create_tikitaka_v2_config(
        transfer_every=params["transfer_every"],
        transfer_lr=params["transfer_lr"],
        fast_lr=params["fast_lr"],
        auto_granularity=params["auto_granularity"],
        in_chop_prob=params["in_chop_prob"],
    )

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        if "bias" in name:
            param.requires_grad = False
        else:
            param.requires_grad = is_target or "qa_outputs" in name

    return model.to(device)


def load_squad_data(tokenizer):
    raw_datasets = load_dataset("squad")
    eval_examples = raw_datasets["validation"].select(range(min(2000, len(raw_datasets["validation"]))))

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=384, truncation="only_second",
            stride=128, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )
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
        inputs = tokenizer(
            questions, examples["context"],
            max_length=384, truncation="only_second",
            stride=128, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )
        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs["offset_mapping"]
        for i in range(len(inputs["input_ids"])):
            sequence_ids = inputs.sequence_ids(i)
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(offset_mapping[i])
            ]
        inputs["example_id"] = [examples["id"][sample_map[i]] for i in range(len(inputs["input_ids"]))]
        return inputs

    tokenized_train = raw_datasets["train"].map(preprocess_train, batched=True, remove_columns=raw_datasets["train"].column_names)
    train_subset = tokenized_train.shuffle(seed=SEED).select(range(min(10000, len(tokenized_train))))
    tokenized_eval = eval_examples.map(preprocess_eval, batched=True, remove_columns=raw_datasets["validation"].column_names)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)
    return train_loader, tokenized_eval, eval_examples


# =============================================================================
# SQuAD Evaluation (identical)
# =============================================================================

def postprocess_squad_predictions(examples, features, all_start_logits, all_end_logits, n_best_size=20, max_answer_length=30):
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
            start_indexes = np.argsort(start_logits)[-1:-n_best_size-1:-1].tolist()
            end_indexes = np.argsort(end_logits)[-1:-n_best_size-1:-1].tolist()
            for si in start_indexes:
                for ei in end_indexes:
                    if si >= len(offset_mapping) or ei >= len(offset_mapping): continue
                    if offset_mapping[si] is None or offset_mapping[ei] is None: continue
                    if ei < si or ei - si + 1 > max_answer_length: continue
                    prelim_predictions.append({
                        "offsets": (offset_mapping[si][0], offset_mapping[ei][1]),
                        "score": start_logits[si] + end_logits[ei],
                    })
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]
        if len(predictions) == 0:
            all_predictions[example["id"]] = ""
        else:
            s, e = predictions[0]["offsets"]
            all_predictions[example["id"]] = context[s:e]
    return all_predictions


def evaluate_squad(model, eval_features, eval_examples, tokenizer, device):
    model.eval()
    all_start_logits, all_end_logits = [], []

    def squad_eval_collate_fn(features):
        offset_mappings = [f.pop("offset_mapping") for f in features]
        example_ids = [f.pop("example_id") for f in features]
        batch = default_data_collator(features)
        batch["offset_mapping"] = offset_mappings
        batch["example_id"] = example_ids
        for i, f in enumerate(features):
            f["offset_mapping"] = offset_mappings[i]
            f["example_id"] = example_ids[i]
        return batch

    eval_loader = DataLoader(eval_features, batch_size=BATCH_SIZE, shuffle=False, collate_fn=squad_eval_collate_fn)
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())
    model.train()

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)
    predictions = postprocess_squad_predictions(eval_examples, eval_features, all_start_logits, all_end_logits)
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    results = evaluate.load("squad").compute(predictions=formatted_predictions, references=references)
    return results["f1"], results["exact_match"]


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, optimizer, scheduler, train_loader, device, epoch_num):
    model.train()
    total_loss, num_batches = 0.0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch_num}", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            start_positions=batch['start_positions'].to(device),
            end_positions=batch['end_positions'].to(device),
        )
        loss_val = outputs.loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val) or loss_val > LOSS_EXPLOSION_THRESHOLD:
            pbar.close()
            print(f"  Loss diverged (loss={loss_val}) at epoch {epoch_num}")
            return float('nan')
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss_val
        num_batches += 1
        pbar.set_postfix(loss=f"{loss_val:.4f}")
    return total_loss / num_batches if num_batches > 0 else 0.0


# =============================================================================
# Main
# =============================================================================

def run_single_ablation(name, params, train_loader, eval_features, eval_examples, tokenizer, device, results_dir):
    """Run one ablation experiment."""
    print(f"\n{'=' * 60}")
    print(f"Ablation: {name}")
    print(f"{'=' * 60}")
    print(f"Params: {json.dumps(params, indent=2)}")

    os.environ["WANDB_MODE"] = "offline"
    wandb.init(project=WANDB_PROJECT, name=name, config=params, reinit=True)

    set_seed(SEED)

    model = create_squad_model(params, device)
    optimizer = AnalogAdam(model.parameters(), lr=params["learning_rate"])
    optimizer.regroup_param_groups(model)

    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_min_lr(optimizer, num_training_steps, MIN_LR_RATE)

    best_f1 = 0.0
    patience_counter = 0
    results_log = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, scheduler, train_loader, device, epoch)

        if math.isnan(train_loss):
            print(f"Training aborted: loss diverged at epoch {epoch}")
            break

        eval_f1, eval_em = evaluate_squad(model, eval_features, eval_examples, tokenizer, device)
        current_lr = scheduler.get_last_lr()[0]

        wandb.log({"epoch": epoch, "train/loss": train_loss, "eval/f1": eval_f1, "eval/em": eval_em, "lr": current_lr})
        print(f"  [Epoch {epoch}/{NUM_EPOCHS}] Loss: {train_loss:.4f}, F1: {eval_f1:.2f}, EM: {eval_em:.2f}, LR: {current_lr:.6f}")

        results_log.append({"epoch": epoch, "train_loss": train_loss, "f1": eval_f1, "em": eval_em, "lr": current_lr})

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs, best F1={best_f1:.2f})")
            break

    wandb.log({"final/best_f1": best_f1})
    wandb.finish()

    del model
    torch.cuda.empty_cache()

    return {"name": name, "best_f1": best_f1, "params": params, "epochs_log": results_log}


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(OUTPUT_DIR, f"no_transfer_ablation_bias_frozen_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("SQuAD Ablation: TikiTaka v2 — Transfer Disabled")
    print("=" * 60)
    print(f"Baseline: sweep_squad_15ep_bs256.py Trial 44 (F1=74.42)")
    print(f"Ablations: transfer_lr=0, transfer_every=0")
    print(f"Epochs: {NUM_EPOCHS}, Patience: {PATIENCE}, BS: {BATCH_SIZE}")
    print(f"Results dir: {results_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    all_results = {}
    for name, params in ABLATIONS.items():
        result = run_single_ablation(
            name, params, train_loader, eval_features, eval_examples,
            tokenizer, device, results_dir,
        )
        all_results[name] = result

    # Save results
    results_file = os.path.join(results_dir, "no_transfer_results.json")
    final = {
        "experiment": "no_transfer_ablation",
        "baseline_f1": 74.42,
        "ablations": all_results,
    }
    with open(results_file, 'w') as f:
        json.dump(final, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL ABLATIONS COMPLETE")
    print("=" * 60)
    print(f"  Baseline (with transfer):       F1 = 74.42")
    for name, r in all_results.items():
        print(f"  {name:30s}  F1 = {r['best_f1']:.2f}  ({r['best_f1'] - 74.42:+.2f})")
    print(f"  Results saved to: {results_file}")


if __name__ == "__main__":
    main()
