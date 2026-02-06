#!/usr/bin/env python
# coding=utf-8
"""BERT-base LRTT Hyperparameter Sweep (Parallel Execution).

Sweep parameters:
- rank: [1, 4, 8, 16, 32, 64]
- transfer_every (te): [1, 10, 50, 100, 500, 1000, 2000, 5000]
- learning_rate (lr): [0.001, 0.01, 0.05, 0.1]
- lifetime: [100, 1000, 10000, 46505, 100000] (10^2 ~ 10^5)

Baseline settings (same as lora_on_analog_hardware):
- Model: bert-base-uncased
- Dataset: SST-2 full (67,349 samples)
- Epochs: 3
- Batch size: 32
- max_seq_length: 128

Usage:
    # Run with 4 parallel workers
    nohup python sweep_bert_base_lrtt_parallel.py --num_workers 4 > sweep.log 2>&1 &

    # Run specific GPU
    CUDA_VISIBLE_DEVICES=0,1 nohup python sweep_bert_base_lrtt_parallel.py --num_workers 2 > sweep.log 2>&1 &
"""

import argparse
import gc
import itertools
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
import multiprocessing as mp

import numpy as np
import torch

# Must be called before importing transformers in child processes
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
from aihwkit.optim import AnalogSGD
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
# Configuration
# =============================================================================

SEED = 42

# Baseline settings (same as lora_on_analog_hardware)
MODEL_NAME = "bert-base-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
LOGGING_STEPS = 100

# Sweep parameters
RANKS = [1, 4, 8, 16, 32, 64]
TRANSFER_EVERYS = [1, 10, 50, 100, 500, 1000, 2000, 5000]
LEARNING_RATES = [0.001, 0.01, 0.05, 0.1]
LIFETIMES = [100, 1000, 10000, 46505, 100000]

TRANSFER_LR_SCALE = 1.0

SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}

WANDB_PROJECT = "lrtt-bert-base-sweep"
OUTPUT_DIR = "/tmp/lrtt_sweep_results"


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    return -TAU_SEC * math.log(1 - delta)


def create_lrtt_config(rank: int, te: int, tlr: float, lifetime: float) -> PythonLRTTRPUConfig:
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


def run_single_experiment(args: tuple) -> Dict:
    """Run a single experiment (designed for multiprocessing)."""
    config, exp_id, total_exps, gpu_id = args

    # Set GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = config["rank"]
    te = config["te"]
    lr = config["lr"]
    lifetime = config["lifetime"]
    tlr = lr * TRANSFER_LR_SCALE

    exp_name = f"r{rank}_te{te}_lr{lr}_lt{lifetime}"

    print(f"[{exp_id}/{total_exps}] Starting {exp_name} on GPU {gpu_id}")

    set_seed(SEED)

    # Initialize wandb
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project=WANDB_PROJECT,
                name=exp_name,
                config={
                    "model": MODEL_NAME,
                    "task": TASK_NAME,
                    "rank": rank,
                    "transfer_every": te,
                    "learning_rate": lr,
                    "transfer_lr": tlr,
                    "lifetime": lifetime,
                    "epochs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "gpu_id": gpu_id,
                },
                reinit=True,
            )
        except Exception as e:
            print(f"Wandb init failed: {e}")

    try:
        # Load tokenizer and data
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

        # Load model
        model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=model_config
        )

        # Convert to LRTT
        rpu_config = create_lrtt_config(rank, te, tlr, lifetime)
        model = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
        model.to(device)

        lrtt_count = sum(1 for _, m in model.named_modules()
                         if hasattr(m, 'analog_module') and hasattr(m.analog_module, 'controller'))

        # Optimizer
        optimizer = AnalogSGD(model.parameters(), lr=lr)

        # Metric
        metric = evaluate.load("glue", TASK_NAME)

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)

        # Training args
        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/{exp_name}",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            logging_steps=LOGGING_STEPS,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="wandb" if WANDB_AVAILABLE else "none",
            run_name=exp_name,
            seed=SEED,
            remove_unused_columns=True,
            disable_tqdm=True,
        )

        # Trainer
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
        train_result = trainer.train()
        train_loss = train_result.metrics.get("train_loss", 0)

        # Final eval
        eval_result = trainer.evaluate()
        eval_acc = eval_result.get("eval_accuracy", 0)

        print(f"[{exp_id}/{total_exps}] {exp_name}: acc={eval_acc:.4f}, loss={train_loss:.4f}")

        result = {
            "exp_id": exp_id,
            "exp_name": exp_name,
            "rank": rank,
            "te": te,
            "lr": lr,
            "tlr": tlr,
            "lifetime": lifetime,
            "train_loss": train_loss,
            "eval_accuracy": eval_acc,
            "lrtt_layers": lrtt_count,
            "gpu_id": gpu_id,
        }

        if WANDB_AVAILABLE:
            try:
                wandb.log({"final_eval_accuracy": eval_acc, "final_train_loss": train_loss})
            except:
                pass

    except Exception as e:
        print(f"[{exp_id}/{total_exps}] {exp_name} ERROR: {e}")
        result = {
            "exp_id": exp_id,
            "exp_name": exp_name,
            "rank": rank,
            "te": te,
            "lr": lr,
            "lifetime": lifetime,
            "error": str(e),
            "gpu_id": gpu_id,
        }

    finally:
        if WANDB_AVAILABLE:
            try:
                wandb.finish()
            except:
                pass

        # Cleanup
        try:
            del model
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--gpu_ids", type=str, default=None, help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for resuming")
    parser.add_argument("--end_idx", type=int, default=None, help="End index (exclusive)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parse GPU IDs
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]

    print("=" * 60)
    print("BERT-base LRTT Hyperparameter Sweep (Parallel)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Workers: {args.num_workers}")
    print(f"GPUs: {gpu_ids}")
    print("-" * 60)
    print(f"Ranks: {RANKS}")
    print(f"Transfer Every: {TRANSFER_EVERYS}")
    print(f"Learning Rates: {LEARNING_RATES}")
    print(f"Lifetimes: {LIFETIMES}")

    # Generate all configs
    configs = [
        {"rank": r, "te": te, "lr": lr, "lifetime": lt}
        for r, te, lr, lt in itertools.product(RANKS, TRANSFER_EVERYS, LEARNING_RATES, LIFETIMES)
    ]

    # Slice if specified
    end_idx = args.end_idx if args.end_idx else len(configs)
    configs = configs[args.start_idx:end_idx]

    total_exps = len(configs)
    print(f"Total experiments: {total_exps}")
    print("=" * 60)

    # Prepare arguments with round-robin GPU assignment
    exp_args = []
    for i, config in enumerate(configs):
        gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids[0] is not None else None
        exp_args.append((config, i + 1 + args.start_idx, total_exps + args.start_idx, gpu_id))

    # Run experiments
    all_results = []

    if args.num_workers == 1:
        # Sequential execution
        for exp_arg in exp_args:
            result = run_single_experiment(exp_arg)
            all_results.append(result)

            # Save intermediate
            results_file = f"{OUTPUT_DIR}/sweep_results_{timestamp}.json"
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
    else:
        # Parallel execution
        mp.set_start_method("spawn", force=True)

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(run_single_experiment, arg): arg for arg in exp_args}

            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)

                    # Save intermediate
                    results_file = f"{OUTPUT_DIR}/sweep_results_{timestamp}.json"
                    with open(results_file, "w") as f:
                        json.dump(all_results, f, indent=2)

                except Exception as e:
                    print(f"Future error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)

    successful = [r for r in all_results if "eval_accuracy" in r]
    if successful:
        best = max(successful, key=lambda x: x["eval_accuracy"])
        print(f"\nBest: r{best['rank']}_te{best['te']}_lr{best['lr']}_lt{best['lifetime']}")
        print(f"  Accuracy: {best['eval_accuracy']:.4f}")

        print(f"\nTop 10:")
        top10 = sorted(successful, key=lambda x: x["eval_accuracy"], reverse=True)[:10]
        for i, r in enumerate(top10, 1):
            print(f"  {i}. r{r['rank']}_te{r['te']}_lr{r['lr']}_lt{r['lifetime']}: {r['eval_accuracy']:.4f}")

    results_file = f"{OUTPUT_DIR}/sweep_results_{timestamp}.json"
    print(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    main()
