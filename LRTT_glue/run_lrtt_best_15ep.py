#!/home/jovyan/work/ml/.venv310/bin/python
# coding=utf-8
"""Run LRTT with best hybrid conditions for 15 epochs.

Best conditions found from Bayesian sweep:
- SQuAD:  hybrid, uimbatch=True,  lr=0.00362, t_lr=0.00115, te=1000 → F1=67.03 (3ep)
- SST-2:  hybrid, uimbatch=False, lr=0.01383, t_lr=0.00996, te=100  → acc=0.772 (3ep)
"""

import os
import sys
import json
import re
import string
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter

import torch
import torch.nn as nn
from tqdm import tqdm

import wandb

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
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
import math

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

sys.path.insert(0, '/home/jovyan/work/LRTT/src')
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


# =============================================================================
# Constants
# =============================================================================

MODEL_NAME = "google/mobilebert-uncased"
TARGET_MODULES = ["query", "key", "value"]
BATCH_SIZE = 32
WARMUP_STEPS = 500
SEED = 42
WANDB_PROJECT = "lrtt-best-15ep"
OUTPUT_DIR = "/data/results/LRTT_sweep"

# Best configs per task
BEST_CONFIGS = {
    "squad": {
        "learning_rate": 0.00362,
        "transfer_lr": 0.00115,
        "transfer_every": 1000,
        "units_in_mbatch": True,
    },
    "sst2": {
        "learning_rate": 0.01383,
        "transfer_lr": 0.00996,
        "transfer_every": 100,
        "units_in_mbatch": False,
    },
}

TASK_TO_NUM_LABELS = {"sst2": 2}
TASK_TO_KEYS = {"sst2": ("sentence", None)}


# =============================================================================
# SQuAD helpers
# =============================================================================

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


# =============================================================================
# LRTT Config
# =============================================================================

def create_lrtt_config(transfer_every, transfer_lr, units_in_mbatch):
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-1.0 / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True, lifetime=lifetime,
        lifetime_dtod=0.0, reset=0.0, reset_dtod=0.0,
    )

    c_device = SoftBoundsDevice(
        dw_min=0.001, w_max=3.0, w_min=-3.0,
        dw_min_dtod=0.0, dw_min_std=0.0,
        up_down=0.0, up_down_dtod=0.0,
        w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=True,
    )

    device_config = PythonLRTTDevice(
        rank=4, transfer_every=transfer_every,
        lora_alpha=1.0, reinit_gain=0.1, reinit_mode="hybrid",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = transfer_lr
    device_config.units_in_mbatch = units_in_mbatch
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


def list_linear_layers(model):
    return [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]


# =============================================================================
# Model creation
# =============================================================================

def create_squad_model(cfg, device):
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    all_linear = list_linear_layers(model)
    exclude = [n for n in all_linear if not any(t in n for t in TARGET_MODULES)]
    exclude.append("qa_outputs")

    rpu = create_lrtt_config(cfg["transfer_every"], cfg["transfer_lr"], cfg["units_in_mbatch"])
    model = convert_to_analog(model, rpu, exclude_modules=exclude)

    for name, p in model.named_parameters():
        p.requires_grad = any(t in name for t in TARGET_MODULES) or "qa_outputs" in name
    return model.to(device)


def create_sst2_model(cfg, device):
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    all_linear = list_linear_layers(model)
    exclude = [n for n in all_linear if not any(t in n for t in TARGET_MODULES)]
    exclude.append("classifier")

    rpu = create_lrtt_config(cfg["transfer_every"], cfg["transfer_lr"], cfg["units_in_mbatch"])
    model = convert_to_analog(model, rpu, exclude_modules=exclude)

    for name, p in model.named_parameters():
        p.requires_grad = any(t in name for t in TARGET_MODULES) or "classifier" in name
    return model.to(device)


# =============================================================================
# Data loading
# =============================================================================

def load_squad_data(tokenizer):
    raw = load_dataset("squad")
    eval_examples = raw["validation"].select(range(min(2000, len(raw["validation"]))))

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(questions, examples["context"],
                          max_length=384, truncation="only_second",
                          stride=128, return_overflowing_tokens=True,
                          return_offsets_mapping=True, padding="max_length")
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
            seq_ids = inputs.sequence_ids(i)
            idx = 0
            while seq_ids[idx] != 1: idx += 1
            ctx_start = idx
            while idx < len(seq_ids) and seq_ids[idx] == 1: idx += 1
            ctx_end = idx - 1
            if offset[ctx_start][0] > end_char or offset[ctx_end][1] < start_char:
                start_positions.append(0); end_positions.append(0)
            else:
                idx = ctx_start
                while idx <= ctx_end and offset[idx][0] <= start_char: idx += 1
                start_positions.append(idx - 1)
                idx = ctx_end
                while idx >= ctx_start and offset[idx][1] >= end_char: idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_eval(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(questions, examples["context"],
                          max_length=384, truncation="only_second",
                          stride=128, return_overflowing_tokens=True,
                          return_offsets_mapping=True, padding="max_length")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs["offset_mapping"]
        for i in range(len(inputs["input_ids"])):
            seq_ids = inputs.sequence_ids(i)
            inputs["offset_mapping"][i] = [
                o if seq_ids[k] == 1 else None for k, o in enumerate(offset_mapping[i])
            ]
        inputs["example_id"] = [examples["id"][sample_map[i]] for i in range(len(inputs["input_ids"]))]
        return inputs

    tok_train = raw["train"].map(preprocess_train, batched=True, remove_columns=raw["train"].column_names)
    train_sub = tok_train.shuffle(seed=SEED).select(range(min(10000, len(tok_train))))
    tok_eval = eval_examples.map(preprocess_eval, batched=True, remove_columns=raw["validation"].column_names)
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)
    return train_loader, tok_eval, eval_examples


def load_sst2_data(tokenizer):
    raw = load_dataset("nyu-mll/glue", "sst2")
    def preprocess(examples):
        return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)
    tokenized = raw.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)
    eval_loader = DataLoader(tokenized["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=default_data_collator)
    return train_loader, eval_loader


# =============================================================================
# Evaluation
# =============================================================================

def postprocess_squad_predictions(examples, features, all_start_logits, all_end_logits,
                                   n_best_size=20, max_answer_length=30):
    eid2idx = {k: i for i, k in enumerate(examples["id"])}
    feat_per_ex = collections.defaultdict(list)
    for i, f in enumerate(features):
        feat_per_ex[eid2idx[f["example_id"]]].append(i)

    preds = collections.OrderedDict()
    for ex_idx, example in enumerate(examples):
        fi = feat_per_ex[ex_idx]
        ctx = example["context"]
        prelim = []
        for fidx in fi:
            sl, el = all_start_logits[fidx], all_end_logits[fidx]
            om = features[fidx]["offset_mapping"]
            si = np.argsort(sl)[-1:-n_best_size-1:-1].tolist()
            ei = np.argsort(el)[-1:-n_best_size-1:-1].tolist()
            for s in si:
                for e in ei:
                    if s >= len(om) or e >= len(om) or om[s] is None or om[e] is None: continue
                    if e < s or e - s + 1 > max_answer_length: continue
                    prelim.append({"offsets": (om[s][0], om[e][1]), "score": sl[s] + el[e]})
        prelim = sorted(prelim, key=lambda x: x["score"], reverse=True)[:n_best_size]
        if not prelim:
            preds[example["id"]] = ""
        else:
            sc, ec = prelim[0]["offsets"]
            preds[example["id"]] = ctx[sc:ec]
    return preds


def evaluate_squad(model, eval_features, eval_examples, device):
    model.eval()
    all_sl, all_el = [], []

    def collate(features):
        om = [f.pop("offset_mapping") for f in features]
        eid = [f.pop("example_id") for f in features]
        batch = default_data_collator(features)
        batch["offset_mapping"] = om; batch["example_id"] = eid
        for i, f in enumerate(features):
            f["offset_mapping"] = om[i]; f["example_id"] = eid[i]
        return batch

    loader = DataLoader(eval_features, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch['input_ids'].to(device),
                       attention_mask=batch['attention_mask'].to(device))
            all_sl.append(out.start_logits.cpu().numpy())
            all_el.append(out.end_logits.cpu().numpy())
    model.train()

    all_sl = np.concatenate(all_sl); all_el = np.concatenate(all_el)
    preds = postprocess_squad_predictions(eval_examples, eval_features, all_sl, all_el)
    fmt_preds = [{"id": k, "prediction_text": v} for k, v in preds.items()]
    refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    metric = evaluate.load("squad")
    results = metric.compute(predictions=fmt_preds, references=refs)
    return results["f1"], results["exact_match"]


def evaluate_sst2(model, eval_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in eval_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            correct += (out.logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


# =============================================================================
# Training loop
# =============================================================================

def run_task(task_name, num_epochs, device, results_dir):
    cfg = BEST_CONFIGS[task_name]
    print(f"\n{'='*60}")
    print(f"LRTT Best Hybrid - {task_name.upper()} - {num_epochs} epochs")
    print(f"{'='*60}")
    print(f"Config: {json.dumps(cfg, indent=2)}")

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    if task_name == "squad":
        train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
        eval_loader = None
    else:
        train_loader, eval_loader = load_sst2_data(tokenizer)
        eval_features = eval_examples = None

    print(f"Train batches: {len(train_loader)}")

    # Create model
    if task_name == "squad":
        model = create_squad_model(cfg, device)
    else:
        model = create_sst2_model(cfg, device)

    optimizer = AnalogAdam(model.parameters(), lr=cfg["learning_rate"])
    optimizer.regroup_param_groups()

    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, num_training_steps)

    # WandB
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{task_name}_hybrid_best_{num_epochs}ep",
        config={"task": task_name, "epochs": num_epochs, **cfg},
        reinit=True,
    )

    # Initial eval
    if task_name == "squad":
        init_f1, init_em = evaluate_squad(model, eval_features, eval_examples, device)
        print(f"Epoch 0: F1={init_f1:.2f}, EM={init_em:.2f}")
        wandb.log({"epoch": 0, "eval/f1": init_f1, "eval/em": init_em})
    else:
        init_acc = evaluate_sst2(model, eval_loader, device)
        print(f"Epoch 0: acc={init_acc:.4f}")
        wandb.log({"epoch": 0, "eval/accuracy": init_acc})

    # Training
    history = []
    best_metric = 0.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)

        for batch in pbar:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            optimizer.zero_grad()

            if task_name == "squad":
                out = model(input_ids=ids, attention_mask=mask,
                           start_positions=batch['start_positions'].to(device),
                           end_positions=batch['end_positions'].to(device))
                loss = out.loss
            else:
                labels = batch['labels'].to(device)
                out = model(input_ids=ids, attention_mask=mask)
                loss = nn.CrossEntropyLoss()(out.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / n_batches

        # Eval
        if task_name == "squad":
            f1, em = evaluate_squad(model, eval_features, eval_examples, device)
            metric_val = f1
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, F1={f1:.2f}, EM={em:.2f}")
            wandb.log({"epoch": epoch, "train/loss": avg_loss, "eval/f1": f1, "eval/em": em})
            history.append({"epoch": epoch, "loss": avg_loss, "f1": f1, "em": em})
        else:
            acc = evaluate_sst2(model, eval_loader, device)
            metric_val = acc
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.4f}")
            wandb.log({"epoch": epoch, "train/loss": avg_loss, "eval/accuracy": acc})
            history.append({"epoch": epoch, "loss": avg_loss, "accuracy": acc})

        if metric_val > best_metric:
            best_metric = metric_val

    wandb.finish()

    # Save results
    result = {
        "task": task_name,
        "epochs": num_epochs,
        "config": cfg,
        "fixed_params": {
            "rank": 4, "lora_alpha": 1.0, "reinit_mode": "hybrid",
            "reinit_gain": 0.1, "decay_factor": 1.0,
            "model": MODEL_NAME, "batch_size": BATCH_SIZE,
        },
        "best_metric": best_metric,
        "final_metric": metric_val,
        "history": history,
    }

    task_dir = os.path.join(results_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    out_file = os.path.join(task_dir, f"best_hybrid_{num_epochs}ep.json")
    with open(out_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_file}")
    print(f"Best metric: {best_metric:.4f}, Final: {metric_val:.4f}")

    del model
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["squad", "sst2"])
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = {}
    for task in args.tasks:
        if task not in BEST_CONFIGS:
            print(f"Unknown task: {task}"); continue
        results[task] = run_task(task, args.epochs, device, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("ALL DONE")
    print(f"{'='*60}")
    for t, r in results.items():
        print(f"  {t}: best={r['best_metric']:.4f}, final={r['final_metric']:.4f}")


if __name__ == "__main__":
    main()
