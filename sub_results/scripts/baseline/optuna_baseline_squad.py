# -*- coding: utf-8 -*-
"""Optuna baseline sweep for ALBERT + SQuAD: Analog QKV (frozen) + digital head/LN training.

Architecture:
    - Attention (query, key, value, dense) layers -> SingleRPU Analog (frozen weights, trainable out_scaling)
    - All other layers -> Digital (frozen except qa_outputs, LayerNorm)
    - Trainable: qa_outputs, LayerNorm, out_scaling
    - Optimizer: AnalogAdam, weight_decay=0
    - Sweep: learning_rate [1e-4 ~ 1e-2]

Usage:
    python optuna_baseline_squad.py --n-trials 10
    python optuna_baseline_squad.py --visualize
"""

import os
import sys
import re
import string
import math
import json
import argparse
import gc
import collections

import torch
from torch import nn, no_grad
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
import evaluate

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import SoftBoundsDevice

os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# Global Constants
# =============================================================================

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

SEED = 42
MODEL_NAME = "albert/albert-base-v2"
MAX_SEQ_LENGTH = 384

# Training config: 2× org epoch (org=2 from Albert_setup.txt)
N_EPOCHS = 5
BATCH_SIZE = 48
EVAL_BATCH_SIZE = 256
EARLY_STOP_PATIENCE = 2
N_TRIALS = 10

# Scheduler
WARMUP_RATIO = 0.05  # 5% of total steps

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0


# =============================================================================
# RPU Config for QKV (frozen analog)
# =============================================================================

def _create_qkv_rpu_config():
    """SingleRPUConfig + SoftBoundsDevice for QKV frozen analog layers."""
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
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True
    return rpu_config


# =============================================================================
# Model
# =============================================================================

def create_model():
    """Create ALBERT QA with analog attention QKVO (frozen) + digital rest.

    Trainable: qa_outputs, LayerNorm, out_scaling
    """
    from aihwkit.nn import AnalogLinear

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Attention (QKVO) pattern matching — same as --lora-target attn
    # "attention" matches: query, key, value, dense (output projection)
    attn_pattern = "attention"

    def is_attn_encoder(layer_name):
        if "encoder" not in layer_name:
            return False
        return attn_pattern in layer_name

    # Build exclude list: everything that is NOT attention encoder
    all_linear_names = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    exclude_modules = [n for n in all_linear_names if not is_attn_encoder(n)]

    # Convert attention layers to analog
    attn_config = _create_qkv_rpu_config()
    model = convert_to_analog(model, attn_config, exclude_modules=exclude_modules)
    analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog attention (QKVO) layers: {analog_count}")

    # Freeze tile weights via noop update hook
    def _frozen_noop_update(x_input, d_input, *args, **kwargs):
        return None
    for name, m in model.named_modules():
        if isinstance(m, AnalogLinear):
            for tile in m.analog_tiles():
                tile.update = _frozen_noop_update

    # Set requires_grad: only qa_outputs, LayerNorm, out_scaling
    for name, param in model.named_parameters():
        if "qa_outputs" in name:
            param.requires_grad = True
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
        elif "out_scaling" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}, Trainable: {trainable:,}")

    return model.to(DEVICE)


# =============================================================================
# Data
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize SQuAD v1.1 dataset."""
    raw_datasets = load_dataset("squad")

    if EVAL_SUBSET_SIZE > 0:
        eval_examples = raw_datasets["validation"].select(
            range(min(EVAL_SUBSET_SIZE, len(raw_datasets["validation"]))))
    else:
        eval_examples = raw_datasets["validation"]

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
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
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
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
        remove_columns=raw_datasets["train"].column_names)

    if TRAIN_SUBSET_SIZE > 0:
        train_subset = tokenized_train.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(tokenized_train))))
    else:
        train_subset = tokenized_train.shuffle(seed=SEED)

    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names)

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=default_data_collator,
        generator=torch.Generator().manual_seed(SEED))

    print(f"  SQuAD v1.1: Train features={len(train_subset)}, Eval examples={len(eval_examples)}")
    return train_loader, tokenized_eval, eval_examples


# =============================================================================
# SQuAD Evaluation
# =============================================================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def postprocess_squad_predictions(
    examples, features, all_start_logits, all_end_logits,
    n_best_size=20, max_answer_length=30,
):
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
                    if (start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None):
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


def evaluate_model(model, eval_features, eval_examples):
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
        eval_features, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=squad_eval_collate_fn)

    with no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    model.train()

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    predictions = postprocess_squad_predictions(
        eval_examples, eval_features,
        all_start_logits, all_end_logits,
        n_best_size=20, max_answer_length=30)

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)

    return results["f1"], results["exact_match"]


# =============================================================================
# Scheduler
# =============================================================================

def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_rate=0.0):
    """Linear schedule with warmup that decays to min_lr_rate (fraction of peak LR)."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Optuna Objective
# =============================================================================

def objective(trial, train_loader, eval_features, eval_examples):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Single hyperparameter: learning_rate
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    print(f"\n{'='*60}")
    print(f"Trial {trial.number} | SQuAD v1.1 | LR={learning_rate:.2e}")
    print(f"{'='*60}")

    model = None
    try:
        set_seed(SEED)
        model = create_model()

        optimizer = AnalogAdam(model.parameters(), lr=learning_rate, weight_decay=0.0)
        optimizer.regroup_param_groups()

        num_training_steps = len(train_loader) * N_EPOCHS
        warmup_steps = int(WARMUP_RATIO * num_training_steps)
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_rate=0.0,
        )

        best_f1 = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
            for batch in pbar:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                start_positions = batch['start_positions'].to(DEVICE)
                end_positions = batch['end_positions'].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    start_positions=start_positions, end_positions=end_positions,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                loss_val = loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    print(f"  [NaN/Inf at batch {num_batches}] Aborting.")
                    return 0.0
                total_loss += loss_val
                num_batches += 1
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0
            eval_f1, eval_em = evaluate_model(model, eval_features, eval_examples)

            improved = ""
            if eval_f1 > best_f1:
                best_f1 = eval_f1
                epochs_without_improvement = 0
                improved = " *"
            else:
                epochs_without_improvement += 1

            current_lr = scheduler.get_last_lr()[0]
            tqdm.write(f"[Trial {trial.number}] Ep {epoch:3d} | "
                       f"F1: {eval_f1:6.2f}% | EM: {eval_em:6.2f}% | Best F1: {best_f1:6.2f}% | "
                       f"Loss: {train_loss:.4f} | LR: {current_lr:.2e} | "
                       f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}{improved}")

            trial.report(best_f1, epoch)

            # Hopeless trial abort
            if epoch == 1 and eval_f1 < 20.0:
                tqdm.write(f"[Trial {trial.number}] F1={eval_f1:.2f}% < 20% at epoch 1 -> abort")
                break

            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}")
                break

        print(f"\n[Trial {trial.number}] Best F1: {best_f1:.2f}%")
        return best_f1

    except Exception as e:
        error_msg = str(e)[:500]
        trial.set_user_attr("error", error_msg)
        print(f"[Trial {trial.number}] Error: {error_msg}")
        raise

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# =============================================================================
# Visualization
# =============================================================================

def visualize_study(study, save_dir):
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        print("No completed trials to visualize.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    trial_numbers = [t.number for t in complete_trials]
    scores = [t.value for t in complete_trials]

    axes[0].scatter(trial_numbers, scores, alpha=0.6)
    axes[0].plot(trial_numbers,
                 [max(scores[:i+1]) for i in range(len(scores))],
                 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('F1 (%)')
    axes[0].set_title('Optimization History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[1].scatter(lrs, scores, alpha=0.6)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Learning Rate')
    axes[1].set_ylabel('F1 (%)')
    axes[1].set_title('LR vs F1')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "baseline_viz_squad.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualization saved.")


def print_study_summary(study):
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Study: {study.study_name}, Trials: {len(study.trials)} ({len(complete_trials)} complete)")
    if complete_trials:
        scores = [t.value for t in complete_trials]
        print(f"Best F1: {max(scores):.2f}%, Mean F1: {sum(scores)/len(scores):.2f}%")
        print(f"Best params: {study.best_params}")


# =============================================================================
# OOM Recovery
# =============================================================================

class _OOMRestart(Exception):
    pass

def _oom_restart_callback(study, trial):
    if trial.state == TrialState.FAIL:
        err = trial.user_attrs.get("error", "")
        if "out of memory" in err.lower() or "cublas" in err.lower():
            print(f"\n[OOM Recovery] Trial {trial.number} failed, will restart.")
            raise _OOMRestart()


# =============================================================================
# Main
# =============================================================================

def main():
    global N_EPOCHS, BATCH_SIZE, EARLY_STOP_PATIENCE, N_TRIALS

    parser = argparse.ArgumentParser(description="Baseline: Analog QKV (frozen) + digital head/LN for SQuAD")
    parser.add_argument('--study-name', type=str, default=None)
    parser.add_argument('--n-trials', type=int, default=N_TRIALS,
                        help=f'Number of trials (default: {N_TRIALS})')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    n_trials = args.n_trials

    RESULTS = "/data/results/baseline_qkv/squad"
    os.makedirs(RESULTS, exist_ok=True)

    study_name = args.study_name or f"albert_baseline_qkv_squad_bs{BATCH_SIZE}_ep{N_EPOCHS}"
    storage = f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    print(f"SQuAD v1.1 | Epochs: {N_EPOCHS}, BS: {BATCH_SIZE}, "
          f"Patience: {EARLY_STOP_PATIENCE}, Trials: {n_trials}")

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=TPESampler(seed=SEED, n_startup_trials=min(5, n_trials // 2)),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}, Device: {DEVICE}, Trials: {n_trials}")

    target_total = len(study.trials) + n_trials

    try:
        study.optimize(
            lambda trial: objective(trial, train_loader, eval_features, eval_examples),
            n_trials=n_trials,
            catch=(Exception,),
            show_progress_bar=False,
            callbacks=[_oom_restart_callback],
        )
    except _OOMRestart:
        remaining = target_total - len(study.trials)
        if remaining > 0:
            print(f"\n[OOM Recovery] Restarting for {remaining} remaining trials...")
            new_argv = list(sys.argv)
            for i, arg in enumerate(new_argv):
                if arg == '--n-trials' and i + 1 < len(new_argv):
                    new_argv[i + 1] = str(remaining)
                    break
            os.execv(sys.executable, [sys.executable] + new_argv)

    print_study_summary(study)
    visualize_study(study, RESULTS)

    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if complete_trials:
        best_file = os.path.join(RESULTS, f"best_params_{study_name}.json")
        with open(best_file, 'w') as f:
            json.dump({
                "task": "squad_v1.1",
                "metric": "f1",
                "best_f1": study.best_value,
                "best_params": study.best_params,
                "config": {
                    "epochs": N_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "patience": EARLY_STOP_PATIENCE,
                    "optimizer": "AnalogAdam",
                    "weight_decay": 0.0,
                    "warmup_ratio": WARMUP_RATIO,
                    "analog_layers": "qkv (frozen)",
                    "trainable": "qa_outputs + LayerNorm + out_scaling",
                },
            }, f, indent=2)
        print(f"Best params saved: {best_file}")

    all_trials_file = os.path.join(RESULTS, "all_trials_squad.json")
    all_trials = sorted(
        [{"trial": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
         for t in study.trials],
        key=lambda x: x["value"] if x["value"] is not None else -1,
        reverse=True,
    )
    with open(all_trials_file, 'w') as f:
        json.dump(all_trials, f, indent=2)
    print(f"All trials saved: {all_trials_file}")


if __name__ == "__main__":
    main()
