# -*- coding: utf-8 -*-
"""MobileBERT + SQuAD with LRTT (Low-Rank TikiTaka Training).

Single-run training script for MobileBERT on SQuAD using LRTT analog layers.
Converts Q/K/V attention layers to analog; all other layers remain digital.

Based on sweep_lrtt_squad_rank8.py, restructured following VIT-tiny patterns.
"""

import os
import sys
import re
import string
import math
import gc
import collections

import torch
from torch import nn, no_grad, manual_seed, save
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import numpy as np

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
import evaluate

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

# LRTT config imports (direct imports to avoid __init__.py dependency issues)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

from collections import Counter


# =============================================================================
# Global Constants
# =============================================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = os.path.join(os.getcwd(), "results", "MOBILEBERT_SQUAD_LRTT_SCRATCH")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "mobilebert_squad_lrtt_scratch_model_weight.pth")

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 384

# Training
N_EPOCHS = 3
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 0.00362
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 5

# Scheduler
WARMUP_STEPS = 0
MIN_LR_RATE = 0.0  # Fraction of peak LR (0 = decay to zero)

# Optimizer
OPTIMIZER = "AnalogAdam"  # "AnalogSGD" or "AnalogAdam"

# LRTT parameters
LRTT_RANK = 8
TRANSFER_EVERY = 1000
TRANSFER_LR = 0.00115
LORA_ALPHA = 1.0
REINIT_MODE = "hybrid"
REINIT_GAIN = 0.1
DECAY_FACTOR = 1.0

# Device configuration for LRTT tiles
AB_DEVICE = "6t1c"
C_DEVICE = "softbounds"

# 6T1C Retention parameters
TAU_SEC = 46505.0
DT_BATCH_SEC = 1.0

# Dynamic TE (transfer every) parameters
DYNAMIC_TE = False
DYNAMIC_TE_POWER = 1.0
TE_WARMUP_STEPS = 0
TE_WARMUP_SCHEDULE = []

# Target modules to convert to analog
TARGET_MODULES = ["query", "key", "value"]

# Data subset sizes
TRAIN_SUBSET_SIZE = 10000
EVAL_SUBSET_SIZE = 2000

# WandB
WANDB_PROJECT = "mobilebert-squad-lrtt-scratch"
os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# LRTT Device Functions
# =============================================================================

def _create_6t1c_device():
    """Create 6T1C LinearStepDevice."""
    delta = 1 - math.exp(-DT_BATCH_SEC / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    return LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=0.3,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lifetime,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )


def _create_c_device():
    """Create noise-free SoftBoundsDevice for C tile."""
    return SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )


def create_lrtt_config():
    """Create LRTT RPU configuration for analog layers."""
    ab_device = _create_6t1c_device()
    c_device = _create_c_device()

    te = TRANSFER_EVERY
    device_config = PythonLRTTDevice(
        rank=LRTT_RANK,
        transfer_every=te,
        lora_alpha=LORA_ALPHA,
        reinit_gain=REINIT_GAIN,
        reinit_mode=REINIT_MODE,
        decay_factor=DECAY_FACTOR,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = TRANSFER_LR
    device_config.units_in_mbatch = True
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = False

    # Dynamic TE: increase TE as LR decays
    device_config.dynamic_te = DYNAMIC_TE
    device_config.dynamic_te_power = DYNAMIC_TE_POWER
    device_config.dynamic_te_max = te * 20
    device_config.te_warmup_schedule = TE_WARMUP_SCHEDULE + [te]
    device_config.te_warmup_steps = TE_WARMUP_STEPS

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    # Weight scaling for C tile to prevent clipping of pretrained weights
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


# =============================================================================
# Model Functions
# =============================================================================

def list_linear_layers(model):
    """List all linear layer names in the model."""
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_analog_layers(model):
    """Count analog layers in the model."""
    from aihwkit.nn import AnalogLinear
    return sum(1 for m in model.modules() if isinstance(m, AnalogLinear))


def create_model():
    """Create MobileBERT QA model with LRTT analog Q/K/V layers."""
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("qa_outputs")

    rpu_config = create_lrtt_config()
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        param.requires_grad = is_target or "qa_outputs" in name

    num_params = count_parameters(model)
    num_analog = count_analog_layers(model)

    print(f"\nCreated MobileBERT model (LRTT):")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Trainable parameters: {num_params:,}")
    print(f"  Analog (LRTT) layers: {num_analog}")
    print(f"  LRTT config: rank={LRTT_RANK}, transfer_every={TRANSFER_EVERY}, "
          f"transfer_lr={TRANSFER_LR}, lora_alpha={LORA_ALPHA}")
    print(f"  Reinit: mode={REINIT_MODE}, gain={REINIT_GAIN}")
    print(f"  Target modules: {TARGET_MODULES}")

    return model.to(DEVICE)


# =============================================================================
# Data Functions
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize SQuAD dataset."""
    raw_datasets = load_dataset("squad")

    eval_examples = raw_datasets["validation"].select(
        range(min(EVAL_SUBSET_SIZE, len(raw_datasets["validation"])))
    )

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
        remove_columns=raw_datasets["train"].column_names
    )
    train_subset = tokenized_train.shuffle(seed=SEED).select(
        range(min(TRAIN_SUBSET_SIZE, len(tokenized_train)))
    )

    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=default_data_collator
    )

    return train_loader, tokenized_eval, eval_examples


# =============================================================================
# Evaluation Functions
# =============================================================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction, ground_truth):
    """Compute token-level F1 score."""
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


def compute_exact_match(prediction, ground_truth):
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def postprocess_squad_predictions(
    examples, features, all_start_logits, all_end_logits,
    n_best_size=20, max_answer_length=30,
):
    """Post-process SQuAD predictions. Extracts best answer spans."""
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


def evaluate_model(model, eval_features, eval_examples, tokenizer):
    """Evaluate SQuAD model using official metric. Returns (F1, EM)."""
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
        collate_fn=squad_eval_collate_fn
    )

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
        n_best_size=20, max_answer_length=30
    )

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)

    return results["f1"], results["exact_match"]


# =============================================================================
# Optimizer & Scheduler
# =============================================================================

def create_optimizer(model):
    """Create analog optimizer with regroup for proper LR propagation."""
    if OPTIMIZER == "AnalogSGD":
        optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = AnalogAdam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

    optimizer.regroup_param_groups()
    return optimizer


def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_rate=0.0):
    """Linear schedule with warmup that decays to min_lr_rate (fraction of peak LR)."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Main
# =============================================================================

def main():
    """Train MobileBERT with LRTT on SQuAD."""
    manual_seed(SEED)
    set_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"mobilebert_lrtt_r{LRTT_RANK}_te{TRANSFER_EVERY}_bs{BATCH_SIZE}",
        config={
            "model": MODEL_NAME, "dataset": "SQuAD",
            "lrtt_rank": LRTT_RANK, "transfer_every": TRANSFER_EVERY,
            "transfer_lr": TRANSFER_LR, "lora_alpha": LORA_ALPHA,
            "reinit_mode": REINIT_MODE, "reinit_gain": REINIT_GAIN,
            "ab_device": AB_DEVICE, "c_device": C_DEVICE, "tau_sec": TAU_SEC,
            "dynamic_te": DYNAMIC_TE, "te_warmup_steps": TE_WARMUP_STEPS,
            "epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "optimizer": OPTIMIZER, "warmup_steps": WARMUP_STEPS,
            "min_lr_rate": MIN_LR_RATE, "seed": SEED,
            "target_modules": TARGET_MODULES,
        }
    )

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    # Create model, optimizer, scheduler
    model = create_model()
    optimizer = create_optimizer(model)

    num_training_steps = len(train_loader) * N_EPOCHS
    scheduler = get_linear_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps,
        min_lr_rate=MIN_LR_RATE,
    )

    # Initial evaluation
    init_f1, init_em = evaluate_model(model, eval_features, eval_examples, tokenizer)
    wandb.log({"epoch": 0, "eval/f1": init_f1, "eval/em": init_em})
    print(f"Initial eval: F1={init_f1:.2f}, EM={init_em:.2f}")

    # Training loop
    best_f1 = init_f1
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"\nStarting training: {N_EPOCHS} epochs (max), early stopping patience={EARLY_STOP_PATIENCE}")

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
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

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Evaluate
        eval_f1, eval_em = evaluate_model(model, eval_features, eval_examples, tokenizer)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch, "train/loss": train_loss,
            "eval/f1": eval_f1, "eval/em": eval_em,
            "learning_rate": current_lr,
        })

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            save(model.state_dict(), WEIGHT_PATH)
        else:
            epochs_without_improvement += 1

        tqdm.write(
            f"Epoch {epoch}: Train Loss {train_loss:.4f} | "
            f"F1 {eval_f1:.2f}% | EM {eval_em:.2f}% | "
            f"Best F1 {best_f1:.2f}% | LR {current_lr:.2e} | "
            f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}"
        )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest F1: {best_f1:.2f}% at epoch {best_epoch}")

    # Memory cleanup
    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU cache cleared")

    wandb.finish()


if __name__ == "__main__":
    main()
