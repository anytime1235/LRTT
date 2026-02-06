#!/usr/bin/env python
# coding=utf-8
"""BERT-base LRTT Bayesian Search with AnalogAdam + Parallel GPU.

Changes from original:
- AnalogSGD → AnalogAdam
- SQLite storage for parallel workers
- Multi-process parallelization on single GPU

Usage:
    # 6 workers in parallel (H200 143GB, ~20GB per model)
    nohup python sweep_bert_base_adam.py --n_workers 6 --n_trials 200 > sweep_adam.log 2>&1 &
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from multiprocessing import Process
from typing import Optional

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

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
    EarlyStoppingCallback,
)

from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam  # Changed from AnalogSGD
from aihwkit.simulator.configs import SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MODEL_NAME = "bert-base-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 15
LOGGING_STEPS = 100

# Search space
RANKS = [1, 4, 8, 16, 32, 64]
TRANSFER_EVERYS = [1, 10, 50, 100, 500, 1000, 2000, 5000]
LIFETIMES = [100, 1000, 10000, 46505, 100000]

# SoftBounds config (no noise)
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}

WANDB_PROJECT = "lrtt-bert-base-adam"
OUTPUT_DIR = "/tmp/lrtt_adam_sweep"
DB_PATH = "/data/LRTT/examples/glue_lrtt_sixt1c/optuna_adam_sweep.db"


def get_data():
    """Load and cache data."""
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

    return tokenizer, train_dataset, eval_dataset, metric


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


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function with AnalogAdam."""
    worker_id = os.getpid() % 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample hyperparameters
    rank = trial.suggest_categorical("rank", RANKS)
    te = trial.suggest_categorical("transfer_every", TRANSFER_EVERYS)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)  # Adjusted for Adam
    tlr = trial.suggest_float("transfer_lr", 1e-4, 10.0, log=True)
    lifetime = trial.suggest_categorical("lifetime", LIFETIMES)

    exp_name = f"t{trial.number}_r{rank}_te{te}_w{worker_id}"

    print(f"\n[Worker {worker_id}] Trial {trial.number}: rank={rank}, te={te}, lr={lr:.6f}, tlr={tlr:.6f}, lifetime={lifetime}")

    set_seed(SEED + trial.number)

    # Initialize wandb
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project=WANDB_PROJECT,
                name=exp_name,
                config={
                    "trial": trial.number,
                    "worker": worker_id,
                    "rank": rank,
                    "transfer_every": te,
                    "learning_rate": lr,
                    "transfer_lr": tlr,
                    "lifetime": lifetime,
                    "optimizer": "AnalogAdam",
                },
                reinit=True,
            )
        except Exception as e:
            print(f"[Worker {worker_id}] Wandb init failed: {e}")

    try:
        tokenizer, train_dataset, eval_dataset, metric = get_data()

        # Load model
        model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=model_config
        )

        # Convert to LRTT
        rpu_config = create_lrtt_config(rank, te, tlr, lifetime)
        model = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
        model.to(device)

        # AnalogAdam optimizer
        optimizer = AnalogAdam(model.parameters(), lr=lr)

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)

        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/trial_{trial.number}",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            logging_steps=LOGGING_STEPS,
            eval_strategy="epoch",
            save_strategy="no",
            save_total_limit=1,
            load_best_model_at_end=False,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="wandb" if WANDB_AVAILABLE else "none",
            run_name=exp_name,
            seed=SEED + trial.number,
            remove_unused_columns=True,
            disable_tqdm=True,
            dataloader_num_workers=0,
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
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Disabled: requires load_best_model_at_end=True
        )

        trainer.train()
        eval_result = trainer.evaluate()
        eval_acc = eval_result.get("eval_accuracy", 0)

        print(f"[Worker {worker_id}] Trial {trial.number} completed: accuracy={eval_acc:.4f}")

        if WANDB_AVAILABLE:
            try:
                wandb.log({"final_accuracy": eval_acc})
            except:
                pass

        return eval_acc

    except Exception as e:
        print(f"[Worker {worker_id}] Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

    finally:
        if WANDB_AVAILABLE:
            try:
                wandb.finish()
            except:
                pass

        try:
            del model, optimizer, trainer
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()


def worker_process(worker_id: int, storage_url: str, study_name: str, n_trials: int):
    """Worker process that runs trials."""
    print(f"[Worker {worker_id}] Starting with up to {n_trials} trials")

    # Small delay to stagger starts
    time.sleep(worker_id * 3)

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
            sampler=TPESampler(seed=SEED + worker_id),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
        )

        print(f"[Worker {worker_id}] Completed")

    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--n_trials", type=int, default=200, help="Total number of trials")
    parser.add_argument("--study_name", type=str, default="lrtt_bert_adam", help="Study name")
    parser.add_argument("--resume", action="store_true", help="Resume from existing study")
    args = parser.parse_args()

    storage_url = f"sqlite:///{DB_PATH}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    print("=" * 70)
    print(" BERT-base LRTT Bayesian Search with AnalogAdam")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Optimizer: AnalogAdam")
    print(f"Workers: {args.n_workers}")
    print(f"Total Trials: {args.n_trials}")
    print(f"Study: {args.study_name}")
    print(f"Storage: {storage_url}")
    print("-" * 70)
    print(f"Ranks: {RANKS}")
    print(f"Transfer Every: {TRANSFER_EVERYS}")
    print(f"LR range: [1e-4, 1e-1] (log)")
    print(f"TLR range: [1e-4, 10.0] (log)")
    print(f"Lifetimes: {LIFETIMES}")
    print("=" * 70)

    # Create or load study
    if args.resume:
        try:
            study = optuna.load_study(
                study_name=args.study_name,
                storage=storage_url,
            )
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            print(f"Resuming study with {completed} completed trials")
        except:
            print("No existing study found, creating new one")
            args.resume = False

    if not args.resume:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage_url,
            direction="maximize",
            sampler=TPESampler(seed=SEED),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
            load_if_exists=True,
        )

    # Calculate trials per worker
    n_trials_per_worker = args.n_trials // args.n_workers
    remainder = args.n_trials % args.n_workers

    print(f"\nStarting {args.n_workers} workers...")
    print(f"Trials per worker: ~{n_trials_per_worker}")

    # Start worker processes
    processes = []
    for i in range(args.n_workers):
        trials = n_trials_per_worker + (1 if i < remainder else 0)
        p = Process(
            target=worker_process,
            args=(i, storage_url, args.study_name, trials)
        )
        p.start()
        processes.append(p)
        print(f"[Main] Started worker {i}")

    # Wait for all workers
    for i, p in enumerate(processes):
        p.join()
        print(f"[Main] Worker {i} finished")

    # Print final results
    study = optuna.load_study(study_name=args.study_name, storage=storage_url)

    print("\n" + "=" * 70)
    print(" OPTIMIZATION COMPLETE")
    print("=" * 70)

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nCompleted trials: {len(completed_trials)}/{args.n_trials}")

    if completed_trials:
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best accuracy: {study.best_value:.4f}")
        print(f"\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Top 10 trials
        print(f"\nTop 10 trials:")
        trials_df = study.trials_dataframe()
        trials_df = trials_df[trials_df['state'] == 'COMPLETE']
        trials_df = trials_df.sort_values("value", ascending=False).head(10)
        cols = ["number", "value"]
        for c in trials_df.columns:
            if c.startswith("params_"):
                cols.append(c)
        print(trials_df[cols].to_string())

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "study_name": args.study_name,
            "optimizer": "AnalogAdam",
            "best_trial": study.best_trial.number,
            "best_accuracy": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(completed_trials),
            "n_workers": args.n_workers,
        }

        results_file = f"{OUTPUT_DIR}/results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    main()
