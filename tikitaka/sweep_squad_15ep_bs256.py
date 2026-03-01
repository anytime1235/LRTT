#!/usr/bin/env python
# coding=utf-8
"""SQuAD 15-Epoch Sweep with TikiTaka v2 (Batch Size 256, Linear LR).

Key changes from sweep_squad_15ep.py:
  1. Batch size: 256 (from 32)
  2. No warmup steps
  3. Linear LR schedule (from cosine)
  4. Warm-start from SQuAD QKV best (Trial 4, F1=74.09)
"""

import os
import sys
import json
import re
import string
import math
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import Counter

import torch
import torch.nn as nn
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler
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
# Fixed Parameters
# =============================================================================

NUM_EPOCHS = 15
N_TRIALS = 50
PATIENCE = 3
TARGET_MODULES = ["query", "key", "value"]
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 384
BATCH_SIZE = 256
WARMUP_STEPS = 0
MIN_LR_RATE = 1.0 / 20.0  # Floor at 5% of peak LR
SEED = 42

WANDB_PROJECT = "tikitaka-squad-15ep-bs256-sweep"
OUTPUT_DIR = "/data/LRTT_transformer/tikitaka/results"

# =============================================================================
# Search Space (3-param only, bs256)
#   - Fixed (low correlation with F1):
#     auto_granularity=169, in_chop_prob=0.061, transfer_every=3
#   - Search (high correlation with F1):
#     fast_lr, transfer_lr, learning_rate
# =============================================================================

# Search params
LR_MIN, LR_MAX = 1e-4, 1e-2
TRANSFER_LR_MIN, TRANSFER_LR_MAX = 0.05, 5.0
FAST_LR_MIN, FAST_LR_MAX = 0.05, 3.0

# Fixed params (from bs32 best Trial 4 — mat-vec unit, no scaling needed)
FIXED_TRANSFER_EVERY = 20
FIXED_AUTO_GRANULARITY = 169.0
FIXED_IN_CHOP_PROB = 0.061

# Warm start (bs32 best scaled for bs256)
WARM_START_PARAMS = {
    "learning_rate": 0.001080,
    "transfer_lr": 0.269,
    "fast_lr": 0.295,
}


# =============================================================================
# Linear Schedule with Min LR (no warmup)
# =============================================================================

def get_linear_schedule_with_min_lr(
    optimizer,
    num_training_steps: int,
    min_lr_rate: float = 0.05,
):
    """Linear decay from peak LR down to min_lr_rate * peak_lr."""
    def lr_lambda(current_step: int) -> float:
        progress = float(current_step) / float(max(1, num_training_steps))
        return min_lr_rate + (1.0 - min_lr_rate) * (1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# SQuAD F1 Evaluation Helpers
# =============================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# =============================================================================
# TikiTaka v2 Config (B tile = SoftBounds noise-free)
# =============================================================================

def create_tikitaka_v2_config(
    transfer_every: int,
    transfer_lr: float,
    fast_lr: float,
    auto_granularity: float,
    in_chop_prob: float,
) -> UnitCellRPUConfig:
    """Create TikiTaka v2 config with noise-free SoftBounds B tile."""

    # A tile: 6T1C LinearStep device (realistic noise)
    sixt1c_device = LinearStepDevice(
        dw_min=0.001981,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=0.3,
        write_noise_std=0.0,
        mult_noise=True,
        mean_bound_reference=True,
        lifetime=0.0,
    )

    # B tile: SoftBounds with ALL noise/variation disabled
    softbounds_device = SoftBoundsReferenceDevice(
        w_max=1.0,
        w_min=-1.0,
        dw_min=0.001,
        dw_min_std=0.0,
        write_noise_std=0.0,
        diffusion=0.0,
        dw_min_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        lifetime=0.0,
        lifetime_dtod=0.0,
        slope_up_dtod=0.0,
        slope_down_dtod=0.0,
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

    # Mapping configuration
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


def list_linear_layers(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


# =============================================================================
# SQuAD Model & Data
# =============================================================================

def create_squad_model(params: Dict, device: torch.device, target_modules: List[str] = None) -> nn.Module:
    """Create SQuAD model with TikiTaka v2.

    Args:
        target_modules: Linear layer name fragments to convert to analog.
            If None, uses TARGET_MODULES global default (["query","key","value"]).
            If ["all"], converts ALL linear layers except qa_outputs.
    """
    if target_modules is None:
        target_modules = TARGET_MODULES

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    all_linear = list_linear_layers(model)

    if target_modules == ["all"]:
        exclude = ["qa_outputs"]
    else:
        exclude = [name for name in all_linear if not any(t in name for t in target_modules)]
        exclude.append("qa_outputs")

    rpu_config = create_tikitaka_v2_config(
        transfer_every=params["transfer_every"],
        transfer_lr=params["transfer_lr"],
        fast_lr=params["fast_lr"],
        auto_granularity=params["auto_granularity"],
        in_chop_prob=params["in_chop_prob"],
    )

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    if target_modules == ["all"]:
        for name, param in model.named_parameters():
            if "bias" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            is_target = any(t in name for t in target_modules)
            if "bias" in name:
                param.requires_grad = False
            else:
                param.requires_grad = is_target or "qa_outputs" in name

    return model.to(device)


def load_squad_data(tokenizer):
    """Load and tokenize SQuAD dataset (10K train subset, 2K eval subset)."""
    raw_datasets = load_dataset("squad")

    eval_examples = raw_datasets["validation"].select(
        range(min(2000, len(raw_datasets["validation"])))
    )

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

        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]

            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
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

        inputs["example_id"] = [
            examples["id"][sample_map[i]] for i in range(len(inputs["input_ids"]))
        ]

        return inputs

    tokenized_train = raw_datasets["train"].map(
        preprocess_train, batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    train_subset = tokenized_train.shuffle(seed=SEED).select(
        range(min(10000, len(tokenized_train)))
    )

    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=default_data_collator,
    )

    return train_loader, tokenized_eval, eval_examples


# =============================================================================
# SQuAD Evaluation
# =============================================================================

def postprocess_squad_predictions(
    examples,
    features,
    all_start_logits,
    all_end_logits,
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    """Post-process SQuAD predictions: extract best answer spans."""
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
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
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


def evaluate_squad(model, eval_features, eval_examples, tokenizer, device) -> Tuple[float, float]:
    """Evaluate SQuAD model. Returns (F1, EM)."""
    model.eval()

    all_start_logits = []
    all_end_logits = []

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

    eval_loader = DataLoader(
        eval_features, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=squad_eval_collate_fn,
    )

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

    predictions = postprocess_squad_predictions(
        eval_examples, eval_features,
        all_start_logits, all_end_logits,
        n_best_size=20, max_answer_length=30,
    )

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)

    return results["f1"], results["exact_match"]


# =============================================================================
# Training
# =============================================================================

LOSS_EXPLOSION_THRESHOLD = 1e7  # Loss above this = diverged

def train_epoch(model, optimizer, scheduler, train_loader, device, epoch_num, trial_num) -> float:
    """Train for one epoch. Returns float('nan') if loss explodes."""
    model.train()
    total_loss, num_batches = 0.0, 0

    pbar = tqdm(train_loader, desc=f"Trial {trial_num} Epoch {epoch_num}", leave=False)
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            start_positions=start_positions, end_positions=end_positions,
        )
        loss = outputs.loss
        loss_val = loss.item()

        if math.isnan(loss_val) or math.isinf(loss_val) or loss_val > LOSS_EXPLOSION_THRESHOLD:
            pbar.close()
            print(f"  Loss diverged (loss={loss_val}) at epoch {epoch_num}, skipping trial.")
            return float('nan')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss_val
        num_batches += 1
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    return total_loss / num_batches if num_batches > 0 else 0.0


# =============================================================================
# Optuna Trial (with early stopping)
# =============================================================================

def run_trial(
    trial: optuna.Trial,
    train_loader,
    eval_features,
    eval_examples,
    tokenizer,
    device: torch.device,
    results_dir: str,
    num_epochs: int,
    target_modules: List[str] = None,
) -> float:
    """Run single trial with patience-based early stopping."""

    # Sample hyperparameters (3 search + 3 fixed)
    params = {
        "learning_rate": trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True),
        "transfer_lr": trial.suggest_float("transfer_lr", TRANSFER_LR_MIN, TRANSFER_LR_MAX, log=True),
        "fast_lr": trial.suggest_float("fast_lr", FAST_LR_MIN, FAST_LR_MAX),
        "transfer_every": FIXED_TRANSFER_EVERY,
        "auto_granularity": FIXED_AUTO_GRANULARITY,
        "in_chop_prob": FIXED_IN_CHOP_PROB,
    }

    print(f"\n--- Trial {trial.number} ---")
    print(f"Params: {json.dumps(params, indent=2)}")

    # WandB logging (offline mode)
    os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"squad_trial_{trial.number}",
        config={"trial": trial.number, "num_epochs": num_epochs, **params},
        reinit=True,
    )

    try:
        set_seed(SEED)

        model = create_squad_model(params, device, target_modules=target_modules)

        optimizer = AnalogAdam(model.parameters(), lr=params["learning_rate"])
        optimizer.regroup_param_groups(model)

        num_training_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_training_steps=num_training_steps,
            min_lr_rate=MIN_LR_RATE,
        )

        # Skip initial evaluation - baseline ~5.3 F1 is constant across trials
        # (same pretrained model, negligible analog noise variation)

        # Training loop with early stopping
        best_f1 = 0.0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = train_epoch(
                model, optimizer, scheduler, train_loader,
                device, epoch, trial.number,
            )

            # Skip trial immediately on loss divergence
            if math.isnan(train_loss):
                print(f"  Trial {trial.number} aborted: loss diverged at epoch {epoch}")
                break

            eval_f1, eval_em = evaluate_squad(
                model, eval_features, eval_examples, tokenizer, device,
            )

            # Get current LR for logging
            current_lr = scheduler.get_last_lr()[0]

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "eval/f1": eval_f1,
                "eval/em": eval_em,
                "lr": current_lr,
            })

            print(f"  [Epoch {epoch}/{num_epochs}] Loss: {train_loss:.4f}, "
                  f"F1: {eval_f1:.2f}, EM: {eval_em:.2f}, LR: {current_lr:.6f}")

            # Report to Optuna
            trial.report(eval_f1, epoch)

            # Patience-based early stopping
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs, best F1={best_f1:.2f})")
                break

        wandb.log({"final/best_f1": best_f1, "final/stopped_epoch": epoch})
        print(f"  Trial {trial.number} done: best F1={best_f1:.2f} (stopped at epoch {epoch})")

        del model
        torch.cuda.empty_cache()

        return best_f1

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        wandb.log({"error": str(e)})
        raise

    finally:
        wandb.finish()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SQuAD 15-Epoch Optimized Sweep")
    parser.add_argument("--tasks", nargs="+", default=["squad"],
                        help="Tasks to run (only squad supported)")
    parser.add_argument("--n_trials", type=int, default=N_TRIALS,
                        help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Max epochs per trial")
    parser.add_argument("--target_modules", type=str, default="qkv",
                        choices=["qkv", "all"],
                        help="Which layers to convert: qkv (Q/K/V only) or all (all linear)")
    args = parser.parse_args()

    num_epochs = args.epochs
    n_trials = args.n_trials

    if args.target_modules == "all":
        target_modules = ["all"]
        modules_tag = "all_layers"
    else:
        target_modules = ["query", "key", "value"]
        modules_tag = "qkv"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(OUTPUT_DIR, f"squad_15ep_sweep_{modules_tag}_bias_frozen_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("SQuAD 15-Epoch Optimized Sweep (TikiTaka v2)")
    print("=" * 60)
    print(f"Trials: {n_trials}")
    print(f"Max epochs: {num_epochs}")
    print(f"Target modules: {args.target_modules} ({target_modules})")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Early stopping patience: {PATIENCE}")
    print(f"LR scheduler: linear decay (no warmup, min_lr=1/20 of peak)")
    print(f"Fixed: transfer_every={FIXED_TRANSFER_EVERY}, auto_gran={FIXED_AUTO_GRANULARITY}, chop_prob={FIXED_IN_CHOP_PROB}")
    print(f"B tile: SoftBoundsReferenceDevice (noise=0)")
    print(f"Results dir: {results_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    # Create Optuna study
    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(
        study_name=f"tikitaka_squad_15ep_{modules_tag}",
        direction="maximize",
        sampler=sampler,
    )

    # Warm-start: enqueue 3-epoch best params
    study.enqueue_trial(WARM_START_PARAMS)

    # Objective
    def objective(trial):
        return run_trial(
            trial, train_loader, eval_features, eval_examples,
            tokenizer, device, results_dir, num_epochs,
            target_modules=target_modules,
        )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Save results
    task_results = {
        "task": "squad",
        "best_trial": study.best_trial.number,
        "best_f1": study.best_value,
        "best_params": study.best_params,
        "num_epochs": num_epochs,
        "patience": PATIENCE,
        "warmup_steps": WARMUP_STEPS,
        "min_lr_rate": MIN_LR_RATE,
        "scheduler": "linear_with_min_lr",
        "target_modules": args.target_modules,
        "b_tile": "SoftBoundsReferenceDevice_noise0",
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }

    results_file = os.path.join(results_dir, "squad_best_params.json")
    with open(results_file, 'w') as f:
        json.dump(task_results, f, indent=2)

    summary_file = os.path.join(results_dir, "all_tasks_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({"squad": task_results}, f, indent=2)

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best F1: {study.best_value:.4f}")
    print(f"Best Params: {json.dumps(study.best_params, indent=2)}")
    print(f"Results saved to: {results_file}")

    # Print all trial results
    print("\nAll Trials:")
    for t in study.trials:
        status = "COMPLETE" if t.value is not None else str(t.state)
        value = f"{t.value:.4f}" if t.value is not None else "N/A"
        print(f"  Trial {t.number}: F1={value} ({status})")


if __name__ == "__main__":
    main()
