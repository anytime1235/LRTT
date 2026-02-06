#!/usr/bin/env python
# coding=utf-8
"""MobileBERT Baseline Training (without LRTT).

Tests if MobileBERT trains correctly without LRTT conversion.
"""

import gc
import os

import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
)

import warnings
warnings.filterwarnings("ignore")

SEED = 42
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 1  # Just 1 epoch for quick test
LOGGING_STEPS = 50
OUTPUT_DIR = "/tmp/mobilebert_baseline_test"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"BASELINE TEST: MobileBERT (no LRTT)")
    print(f"{'='*60}")
    print(f"Device: {device}")

    set_seed(SEED)

    # Load data
    print("\nLoading data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
        )

    tokenized = raw_datasets.map(preprocess, batched=True)
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]
    metric = evaluate.load("glue", TASK_NAME)
    print(f"Data loaded: {len(train_dataset)} train, {len(eval_dataset)} eval")

    # Load model (standard, no LRTT)
    print("\nLoading MobileBERT (standard, no LRTT)...")
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config, use_safetensors=True
    )
    model.to(device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=SEED,
        remove_unused_columns=True,
        disable_tqdm=False,
        learning_rate=5e-5,  # Standard BERT fine-tuning LR
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    print("\n" + "="*60)
    print("STARTING BASELINE TRAINING")
    print("="*60 + "\n")

    train_result = trainer.train()

    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    eval_result = trainer.evaluate()

    print(f"\nFinal Results:")
    print(f"  Train Loss: {train_result.training_loss:.4f}")
    print(f"  Eval Loss: {eval_result.get('eval_loss', 'N/A')}")
    print(f"  Eval Accuracy: {eval_result.get('eval_accuracy', 'N/A')}")

    eval_acc = eval_result.get('eval_accuracy', 0)
    if eval_acc > 0.55:
        print(f"\n[SUCCESS] Baseline MobileBERT is learning! Accuracy {eval_acc:.4f} > 0.55")
    else:
        print(f"\n[FAILED] Baseline MobileBERT not learning. Accuracy {eval_acc:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
