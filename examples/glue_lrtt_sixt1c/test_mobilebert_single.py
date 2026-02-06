#!/usr/bin/env python
# coding=utf-8
"""MobileBERT LRTT Single Training Test.

Tests if training works with reasonable parameters.
Same setup as sweep_mobilebert_optuna.py but with fixed parameters.

Test configurations:
1. AnalogSGD with reasonable LR (like working BERT-base)
2. AnalogAdam with reduced LR and proper settings
"""

import argparse
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
    TrainerCallback,
)

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
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

# =============================================================================
# Configuration - SAME AS SWEEP
# =============================================================================

SEED = 42
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
LOGGING_STEPS = 50  # More frequent logging for debugging

# SoftBounds config (no noise) - SAME AS SWEEP
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}

OUTPUT_DIR = "/tmp/lrtt_mobilebert_test"


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    """Same as sweep."""
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    return -TAU_SEC * math.log(1 - delta)


def create_lrtt_config(rank: int, te: int, tlr: float, lifetime: float) -> PythonLRTTRPUConfig:
    """Same as sweep."""
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
        reset=0.0, reset_dtod=0.0,
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


class LossMonitorCallback(TrainerCallback):
    """Monitor loss for NaN/Inf detection."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss")
            if loss is not None:
                if math.isnan(loss) or math.isinf(loss):
                    print(f"[WARNING] NaN/Inf loss detected at step {state.global_step}: {loss}")
                else:
                    print(f"[Step {state.global_step}] loss={loss:.4f}")


def run_test(optimizer_type: str, lr: float, rank: int, te: int, tlr: float,
             lifetime: float, epochs: int):
    """Run a single training test."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"TEST: MobileBERT LRTT Training")
    print(f"{'='*60}")
    print(f"Optimizer: {optimizer_type}")
    print(f"Learning Rate: {lr}")
    print(f"Rank: {rank}")
    print(f"Transfer Every: {te}")
    print(f"Transfer LR: {tlr}")
    print(f"Lifetime: {lifetime}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    set_seed(SEED)

    # Initialize wandb
    exp_name = f"test_{optimizer_type}_lr{lr}_r{rank}_te{te}"
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="lrtt-mobilebert-test",
                name=exp_name,
                config={
                    "model": MODEL_NAME,
                    "optimizer": optimizer_type,
                    "learning_rate": lr,
                    "rank": rank,
                    "transfer_every": te,
                    "transfer_lr": tlr,
                    "lifetime": lifetime,
                    "batch_size": BATCH_SIZE,
                    "epochs": epochs,
                },
                reinit=True,
            )
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")

    # Load data
    print("Loading tokenizer and dataset...")
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

    # Load model
    print("\nLoading model...")
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config, use_safetensors=True
    )

    # Convert to LRTT
    print("Converting to LRTT...")
    rpu_config = create_lrtt_config(rank, te, tlr, lifetime)
    model = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
    model.to(device)
    print("Model converted and moved to device")

    # Create optimizer
    print(f"\nCreating {optimizer_type} optimizer...")
    if optimizer_type == "AnalogSGD":
        optimizer = AnalogSGD(model.parameters(), lr=lr)
    elif optimizer_type == "AnalogAdam":
        optimizer = AnalogAdam(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{exp_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="wandb" if WANDB_AVAILABLE else "none",
        run_name=exp_name,
        seed=SEED,
        remove_unused_columns=True,
        disable_tqdm=False,  # Show progress
        max_grad_norm=1.0,  # Gradient clipping
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        callbacks=[LossMonitorCallback()],
    )

    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    train_result = trainer.train()

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    eval_result = trainer.evaluate()

    print(f"\nFinal Results:")
    print(f"  Train Loss: {train_result.training_loss:.4f}")
    print(f"  Eval Loss: {eval_result.get('eval_loss', 'N/A')}")
    print(f"  Eval Accuracy: {eval_result.get('eval_accuracy', 'N/A')}")

    # Check if training worked
    eval_acc = eval_result.get('eval_accuracy', 0)
    if eval_acc > 0.55:
        print(f"\n[SUCCESS] Model is learning! Accuracy {eval_acc:.4f} > 0.55")
    elif eval_acc > 0.50:
        print(f"\n[PARTIAL] Some learning observed. Accuracy {eval_acc:.4f} > 0.50")
    else:
        print(f"\n[FAILED] No learning. Accuracy {eval_acc:.4f} <= 0.50 (random)")

    if WANDB_AVAILABLE:
        try:
            wandb.log({"final_accuracy": eval_acc, "final_loss": eval_result.get('eval_loss')})
            wandb.finish()
        except:
            pass

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return eval_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="AnalogSGD",
                        choices=["AnalogSGD", "AnalogAdam"],
                        help="Optimizer type")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--rank", type=int, default=8,
                        help="LRTT rank")
    parser.add_argument("--te", type=int, default=100,
                        help="Transfer every N steps")
    parser.add_argument("--tlr", type=float, default=0.1,
                        help="Transfer learning rate")
    parser.add_argument("--lifetime", type=float, default=46505,
                        help="Device lifetime")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = run_test(
        optimizer_type=args.optimizer,
        lr=args.lr,
        rank=args.rank,
        te=args.te,
        tlr=args.tlr,
        lifetime=args.lifetime,
        epochs=args.epochs,
    )

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
