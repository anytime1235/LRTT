#!/usr/bin/env python3
"""SQuAD baseline: train ONLY qa_outputs layer (no LRTT, no analog).

Same data pipeline, seed, scheduler, epochs as the sweep script for fair comparison.
"""

import os
import sys
import re
import string
import math
import collections
from collections import Counter
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
import evaluate

# =============================================================================
# Constants (same as sweep script)
# =============================================================================
MODEL_NAME = "google/mobilebert-uncased"
BATCH_SIZE = 32
NUM_EPOCHS = 15
WARMUP_STEPS = 500
SEED = 42
PATIENCE = 3
EARLY_STOP_MIN_DELTA = 1.0
MIN_LR_RATE = 0.1
LEARNING_RATE = 0.009676045941584346  # same as best config


# =============================================================================
# Cosine Schedule with Min LR (same as sweep)
# =============================================================================
def get_cosine_with_min_lr_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_rate=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_rate + (1.0 - min_lr_rate) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# SQuAD F1 helpers
# =============================================================================
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))


# =============================================================================
# Data loading (identical to sweep)
# =============================================================================
def load_squad_data(tokenizer):
    raw_datasets = load_dataset("squad")
    eval_examples = raw_datasets["validation"].select(range(min(2000, len(raw_datasets["validation"]))))

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(questions, examples["context"], max_length=384, truncation="only_second",
                           stride=128, return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
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
        inputs = tokenizer(questions, examples["context"], max_length=384, truncation="only_second",
                           stride=128, return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs["offset_mapping"]
        for i in range(len(inputs["input_ids"])):
            sequence_ids = inputs.sequence_ids(i)
            inputs["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset_mapping[i])]
        inputs["example_id"] = [examples["id"][sample_map[i]] for i in range(len(inputs["input_ids"]))]
        return inputs

    tokenized_train = raw_datasets["train"].map(preprocess_train, batched=True, remove_columns=raw_datasets["train"].column_names)
    train_subset = tokenized_train.shuffle(seed=SEED).select(range(min(10000, len(tokenized_train))))
    tokenized_eval = eval_examples.map(preprocess_eval, batched=True, remove_columns=raw_datasets["validation"].column_names)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)
    return train_loader, tokenized_eval, eval_examples


# =============================================================================
# SQuAD postprocess & eval (identical to sweep)
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
                    prelim_predictions.append({"offsets": (offset_mapping[si][0], offset_mapping[ei][1]),
                                               "score": start_logits[si] + end_logits[ei]})
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]
        if not predictions:
            all_predictions[example["id"]] = ""
        else:
            sc, ec = predictions[0]["offsets"]
            all_predictions[example["id"]] = context[sc:ec]
    return all_predictions


def evaluate_squad(model, eval_features, eval_examples, device):
    model.eval()
    all_start_logits, all_end_logits = [], []
    def collate_fn(features):
        om = [f.pop("offset_mapping") for f in features]
        eid = [f.pop("example_id") for f in features]
        batch = default_data_collator(features)
        batch["offset_mapping"] = om; batch["example_id"] = eid
        for i, f in enumerate(features): f["offset_mapping"] = om[i]; f["example_id"] = eid[i]
        return batch
    eval_loader = DataLoader(eval_features, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())
    model.train()
    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)
    predictions = postprocess_squad_predictions(eval_examples, eval_features, all_start_logits, all_end_logits)
    formatted = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    results = evaluate.load("squad").compute(predictions=formatted, references=references)
    return results["f1"], results["exact_match"]


# =============================================================================
# Main
# =============================================================================
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model - NO analog conversion, just pretrained
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Freeze everything except qa_outputs
    for name, param in model.named_parameters():
        param.requires_grad = "qa_outputs" in name

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model.to(device)

    # Load data (same as sweep)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_squad_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    # Optimizer - standard AdamW (no analog needed)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, WARMUP_STEPS, num_training_steps, MIN_LR_RATE)

    # Initial eval
    init_f1, init_em = evaluate_squad(model, eval_features, eval_examples, device)
    print(f"[Epoch 0] F1: {init_f1:.2f}, EM: {init_em:.2f}")

    # Training loop
    best_metric = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, num_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            outputs = model(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device),
                            start_positions=batch['start_positions'].to(device),
                            end_positions=batch['end_positions'].to(device))
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / num_batches
        f1, em = evaluate_squad(model, eval_features, eval_examples, device)
        lr = scheduler.get_last_lr()[0]
        print(f"  [Epoch {epoch}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}, F1: {f1:.2f}, EM: {em:.2f}, LR: {lr:.6f}")

        if f1 > best_metric + EARLY_STOP_MIN_DELTA:
            best_metric = f1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (best={best_metric:.2f})")
            break

    print(f"\nFinal best F1: {best_metric:.2f}")


if __name__ == "__main__":
    main()
