#!/usr/bin/env python
"""Single config test for BERT-base LRTT with wandb logging.

Test: rank=4, te=100, lr=0.01, lifetime=46505, bias=True
Optimizer: AnalogAdam (no scheduler)

Usage:
    python test_single_config.py
    nohup python test_single_config.py > test.log 2>&1 &
"""

import gc
import math
import os
import sys

import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datasets
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

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs import SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

import warnings
warnings.filterwarnings("ignore")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available")

# =============================================================================
# Test Configuration
# =============================================================================

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model & Data
MODEL_NAME = "bert-base-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3

# LRTT Config (test values)
RANK = 4
TRANSFER_EVERY = 100
LEARNING_RATE = 0.01
TRANSFER_LR = 0.01
LIFETIME = 46505  # sixt1c default

# SoftBounds (no noise)
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}

WANDB_PROJECT = "lrtt-bert-base-test"


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    return -TAU_SEC * math.log(1 - delta)


def create_lrtt_config(rank, te, tlr, lifetime):
    dt_batch_sec = lifetime_to_dt_batch_sec(lifetime)
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    ab_lifetime = 1.0 / delta if delta > 0 else 0.0

    ab_device = LinearStepDevice(
        dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
        gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
        dw_min_dtod=0.1, up_down_dtod=0.01,
        w_max_dtod=0.05, w_min_dtod=0.05,
        gamma_up_dtod=0.05, gamma_down_dtod=0.05,
        dw_min_std=0.3, write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=ab_lifetime, lifetime_dtod=0.1,
    )
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=te,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = tlr
    device_config.forward_inject = False
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    return PythonLRTTRPUConfig(device=device_config)


def main():
    print("=" * 60)
    print("BERT-base LRTT Single Config Test")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("-" * 60)
    print(f"Rank: {RANK}")
    print(f"Transfer Every: {TRANSFER_EVERY}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Transfer LR: {TRANSFER_LR}")
    print(f"Lifetime: {LIFETIME}")
    print(f"Optimizer: AnalogAdam (no scheduler)")
    print(f"Bias: True")
    print("=" * 60)

    set_seed(SEED)

    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project=WANDB_PROJECT,
            name=f"test_r{RANK}_te{TRANSFER_EVERY}_lr{LEARNING_RATE}",
            config={
                "model": MODEL_NAME,
                "task": TASK_NAME,
                "rank": RANK,
                "transfer_every": TRANSFER_EVERY,
                "learning_rate": LEARNING_RATE,
                "transfer_lr": TRANSFER_LR,
                "lifetime": LIFETIME,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "optimizer": "AnalogAdam",
                "scheduler": "None",
                "bias": True,
                "device_ab": "LinearStepDevice (6T1C)",
                "device_c": "SoftBoundsDevice (no noise)",
                "reinit_mode": "decay",
                "forward_inject": False,
            },
        )
        print("Wandb initialized")

    # Load tokenizer and data
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
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Load model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # Convert to LRTT
    print("Converting to LRTT...")
    rpu_config = create_lrtt_config(RANK, TRANSFER_EVERY, TRANSFER_LR, LIFETIME)
    model = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
    model.to(DEVICE)

    # Verify LRTT layers and bias
    lrtt_count = 0
    bias_count = 0
    for name, m in model.named_modules():
        if hasattr(m, 'analog_module') and hasattr(m.analog_module, 'controller'):
            lrtt_count += 1
            tile = m.analog_module
            if hasattr(tile, 'bias') and tile.bias:
                if hasattr(tile.tile_c, 'digital_bias') and tile.tile_c.digital_bias:
                    bias_count += 1

    print(f"LRTT layers: {lrtt_count}")
    print(f"Layers with digital_bias=True: {bias_count}")

    # Create optimizer (AnalogAdam, no scheduler)
    optimizer = AnalogAdam(model.parameters(), lr=LEARNING_RATE)
    print(f"Optimizer: AnalogAdam(lr={LEARNING_RATE})")

    # Metric
    metric = evaluate.load("glue", TASK_NAME)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/lrtt_test",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="wandb" if WANDB_AVAILABLE else "none",
        seed=SEED,
        remove_unused_columns=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None),  # No scheduler
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Train
    print("\nStarting training...")
    train_result = trainer.train()

    # Final eval
    print("\nFinal evaluation...")
    eval_result = trainer.evaluate()

    # Results
    train_loss = train_result.metrics.get("train_loss", 0)
    eval_acc = eval_result.get("eval_accuracy", 0)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Eval Accuracy: {eval_acc:.4f} ({eval_acc*100:.2f}%)")
    print("=" * 60)

    if WANDB_AVAILABLE:
        wandb.log({
            "final_train_loss": train_loss,
            "final_eval_accuracy": eval_acc,
        })
        wandb.finish()
        print("Wandb finished")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("\nTest complete!")


if __name__ == "__main__":
    main()
