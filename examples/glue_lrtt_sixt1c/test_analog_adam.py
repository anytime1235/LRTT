#!/usr/bin/env python
# coding=utf-8
"""BERT-base LRTT 테스트: AnalogAdam + 높은 LR.

테스트 설정:
- Optimizer: AnalogAdam (vs AnalogSGD)
- LR: 1e-3 ~ 1e-1 (10배 높임)
- transfer_every: 1 (매 step transfer)
- forward_inject: True (A,B가 forward에 기여)
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
import evaluate
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

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.configs import IdealDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

# Configuration
SEED = 42
MODEL_NAME = "bert-base-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 1  # 빠른 테스트를 위해 1 epoch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_lrtt_config(rank=8, transfer_every=1):
    """LRTT config 생성 (forward_inject=False, reinit_mode=decay 고정)."""
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",  # 고정
        unit_cell_devices=[IdealDevice(), IdealDevice(), IdealDevice()],
    )
    device_config.transfer_lr = 1.0
    device_config.forward_inject = False  # 고정
    device_config.update_mode = "lora"

    return PythonLRTTRPUConfig(device=device_config)


def run_test(optimizer_type, lr, transfer_every=1, max_samples=1000):
    """단일 테스트 실행 (forward_inject=False, reinit=decay 고정)."""
    print(f"\n{'='*60}")
    print(f"테스트: {optimizer_type}, LR={lr}, te={transfer_every}")
    print(f"{'='*60}")

    set_seed(SEED)

    # Load data
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
    train_dataset = tokenized["train"].select(range(min(max_samples, len(tokenized["train"]))))
    eval_dataset = tokenized["validation"]
    metric = evaluate.load("glue", TASK_NAME)

    # Load model
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # Convert to LRTT (forward_inject=False, reinit=decay 고정)
    rpu_config = create_lrtt_config(
        rank=8,
        transfer_every=transfer_every,
    )
    model = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
    model.to(DEVICE)

    # Optimizer
    if optimizer_type == "AnalogAdam":
        optimizer = AnalogAdam(model.parameters(), lr=lr)
    else:
        optimizer = AnalogSGD(model.parameters(), lr=lr)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    training_args = TrainingArguments(
        output_dir=f"/tmp/lrtt_test/{optimizer_type}_{lr}",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=SEED,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Train
    start_time = datetime.now()
    train_result = trainer.train()
    train_time = (datetime.now() - start_time).total_seconds()

    # Evaluate
    eval_result = trainer.evaluate()

    print(f"\n결과:")
    print(f"  Train Loss: {train_result.training_loss:.4f}")
    print(f"  Eval Accuracy: {eval_result.get('eval_accuracy', 0):.4f}")
    print(f"  Train Time: {train_time:.1f}s")

    # Cleanup
    del model, optimizer, trainer
    torch.cuda.empty_cache()

    return {
        "optimizer": optimizer_type,
        "lr": lr,
        "transfer_every": transfer_every,
        "train_loss": train_result.training_loss,
        "eval_accuracy": eval_result.get("eval_accuracy", 0),
        "train_time": train_time,
    }


def main():
    print("=" * 70)
    print(" BERT-base LRTT 테스트: AnalogAdam vs AnalogSGD")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Max samples: 1000 (빠른 테스트)")

    results = []

    # 테스트 케이스 (forward_inject=False, reinit=decay 고정)
    test_cases = [
        # (optimizer, lr, transfer_every)
        ("AnalogSGD", 1e-4, 1),     # 기존 LR, te=1
        ("AnalogSGD", 1e-3, 1),     # LR 10x, te=1
        ("AnalogSGD", 1e-2, 1),     # LR 100x, te=1
        ("AnalogAdam", 1e-4, 1),    # Adam, 기존 LR, te=1
        ("AnalogAdam", 1e-3, 1),    # Adam, LR 10x, te=1
        ("AnalogAdam", 1e-2, 1),    # Adam, LR 100x, te=1
        ("AnalogAdam", 1e-1, 1),    # Adam, LR 1000x, te=1
    ]

    for optimizer_type, lr, transfer_every in test_cases:
        try:
            result = run_test(
                optimizer_type=optimizer_type,
                lr=lr,
                transfer_every=transfer_every,
                max_samples=1000,
            )
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "optimizer": optimizer_type,
                "lr": lr,
                "transfer_every": transfer_every,
                "error": str(e),
            })

    # 결과 요약
    print("\n" + "=" * 70)
    print(" 결과 요약 (forward_inject=False, reinit=decay 고정)")
    print("=" * 70)
    print(f"{'Optimizer':<12} {'LR':<10} {'TE':<6} {'Loss':<10} {'Accuracy':<10}")
    print("-" * 70)

    for r in results:
        if "error" in r:
            print(f"{r['optimizer']:<12} {r['lr']:<10.0e} {r['transfer_every']:<6} ERROR: {r['error'][:30]}")
        else:
            print(f"{r['optimizer']:<12} {r['lr']:<10.0e} {r['transfer_every']:<6} {r['train_loss']:<10.4f} {r['eval_accuracy']:<10.4f}")

    # Best result
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x["eval_accuracy"])
        print(f"\n최고 성능: {best['optimizer']}, LR={best['lr']:.0e}, TE={best['transfer_every']}")
        print(f"  Accuracy: {best['eval_accuracy']:.4f}")


if __name__ == "__main__":
    main()
