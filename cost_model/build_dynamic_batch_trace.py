#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build dynamic-padding batch trace from SQuAD v1.1 for AIMC cost modeling.

Replicates the exact tokenization and batching from optuna_bert_squad_tiki.py,
then computes per-batch dynamic padding statistics:
  S_pad_b = max valid sequence length in batch b
  token_sum_b = total valid tokens in batch b
  padding_waste_b = 1 - token_sum_b / (batch_size_b * S_pad_b)

Output: batch_trace.csv
"""

import os
import csv
import argparse
import numpy as np

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


# =============================================================================
# Constants (from optuna_bert_squad_tiki.py lines 88-96)
# =============================================================================

MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
BATCH_SIZE = 48
SEED = 42


# =============================================================================
# Preprocessing (exact replica of optuna_bert_squad_tiki.py lines 611-665)
# =============================================================================

def preprocess_train(examples, tokenizer):
    """Tokenize SQuAD training examples identically to the training script."""
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions, examples["context"],
        max_length=MAX_SEQ_LENGTH, truncation="only_second",
        stride=DOC_STRIDE, return_overflowing_tokens=True,
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


# =============================================================================
# Batch Trace Construction
# =============================================================================

def extract_valid_lengths(tokenized_dataset) -> np.ndarray:
    """Extract valid (non-padding) token count for each feature.

    Valid length = sum(attention_mask), since attention_mask=1 for real tokens
    and 0 for padding tokens.
    """
    attention_masks = tokenized_dataset["attention_mask"]
    lengths = np.array([sum(am) for am in attention_masks], dtype=np.int32)
    return lengths


def build_batch_trace(
    valid_lengths: np.ndarray,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
    n_epochs: int = 1,
) -> list:
    """Build per-batch dynamic padding trace.

    Simulates PyTorch DataLoader shuffling with the same seed as training.

    Returns:
        List of dicts with: batch_id, batch_size, S_pad, token_sum, padding_waste
    """
    N = len(valid_lengths)
    gen = torch.Generator().manual_seed(seed)

    trace = []
    batch_id = 0

    for epoch in range(n_epochs):
        perm = torch.randperm(N, generator=gen).numpy()
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_indices = perm[start:end]
            batch_lengths = valid_lengths[batch_indices]

            actual_bs = len(batch_lengths)
            s_pad = int(np.max(batch_lengths))
            token_sum = int(np.sum(batch_lengths))
            padding_waste = 1.0 - token_sum / (actual_bs * s_pad) if s_pad > 0 else 0.0

            trace.append({
                'batch_id': batch_id,
                'epoch': epoch,
                'batch_size': actual_bs,
                'S_pad': s_pad,
                'token_sum': token_sum,
                'padding_waste': round(padding_waste, 6),
            })
            batch_id += 1

    return trace


def compute_summary(trace: list) -> dict:
    """Compute summary statistics for the batch trace."""
    s_pads = np.array([b['S_pad'] for b in trace])
    wastes = np.array([b['padding_waste'] for b in trace])
    token_sums = np.array([b['token_sum'] for b in trace])

    return {
        'total_batches': len(trace),
        'total_features': sum(b['batch_size'] for b in trace),
        'mean_S_pad': float(np.mean(s_pads)),
        'median_S_pad': float(np.median(s_pads)),
        'std_S_pad': float(np.std(s_pads)),
        'min_S_pad': int(np.min(s_pads)),
        'max_S_pad': int(np.max(s_pads)),
        'mean_padding_waste': float(np.mean(wastes)),
        'std_padding_waste': float(np.std(wastes)),
        'max_padding_waste': float(np.max(wastes)),
        'mean_token_sum': float(np.mean(token_sums)),
        'pct_batches_at_max': float(np.mean(s_pads == MAX_SEQ_LENGTH) * 100),
    }


def save_trace_csv(trace: list, path: str) -> None:
    """Save batch trace to CSV."""
    fieldnames = ['batch_id', 'epoch', 'batch_size', 'S_pad', 'token_sum', 'padding_waste']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in trace:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Build dynamic-padding batch trace from SQuAD v1.1")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--n-epochs", type=int, default=1, help="Number of epochs to trace")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or os.path.join(output_dir, "batch_trace.csv")

    print(f"Building dynamic-padding batch trace...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  MAX_SEQ_LENGTH: {MAX_SEQ_LENGTH}")
    print(f"  DOC_STRIDE: {DOC_STRIDE}")
    print(f"  BATCH_SIZE: {args.batch_size}")
    print(f"  SEED: {args.seed}")
    print(f"  N_EPOCHS: {args.n_epochs}")

    # Step 1: Load and tokenize SQuAD v1.1
    print("\nLoading SQuAD v1.1...")
    raw_datasets = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing training set (identical to training script)...")
    tokenized_train = raw_datasets["train"].map(
        lambda examples: preprocess_train(examples, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    print(f"  Tokenized features: {len(tokenized_train)}")

    # Step 2: Shuffle (dataset-level, matching training script)
    tokenized_train = tokenized_train.shuffle(seed=args.seed)

    # Step 3: Extract valid lengths
    print("Extracting valid token lengths...")
    valid_lengths = extract_valid_lengths(tokenized_train)
    print(f"  Length range: [{valid_lengths.min()}, {valid_lengths.max()}]")
    print(f"  Mean length: {valid_lengths.mean():.1f}")

    # Step 4: Build batch trace
    print("Building batch trace...")
    trace = build_batch_trace(
        valid_lengths,
        batch_size=args.batch_size,
        seed=args.seed,
        n_epochs=args.n_epochs,
    )

    # Step 5: Summary
    summary = compute_summary(trace)
    print(f"\nBatch Trace Summary:")
    print(f"  Total batches:          {summary['total_batches']}")
    print(f"  Total features:         {summary['total_features']}")
    print(f"  Mean S_pad:             {summary['mean_S_pad']:.1f}")
    print(f"  Median S_pad:           {summary['median_S_pad']:.1f}")
    print(f"  Std S_pad:              {summary['std_S_pad']:.1f}")
    print(f"  Min S_pad:              {summary['min_S_pad']}")
    print(f"  Max S_pad:              {summary['max_S_pad']}")
    print(f"  Mean padding waste:     {summary['mean_padding_waste']:.4f} ({summary['mean_padding_waste']*100:.2f}%)")
    print(f"  Max padding waste:      {summary['max_padding_waste']:.4f} ({summary['max_padding_waste']*100:.2f}%)")
    print(f"  Batches at max length:  {summary['pct_batches_at_max']:.1f}%")

    # Step 6: Save
    save_trace_csv(trace, output_path)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
