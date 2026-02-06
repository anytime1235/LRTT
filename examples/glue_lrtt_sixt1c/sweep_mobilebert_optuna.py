#!/usr/bin/env python
# coding=utf-8
"""MobileBERT LRTT Bayesian Hyperparameter Search with Optuna.

Search parameters:
- rank: [1, 4, 8, 16, 32, 64] (categorical)
- transfer_every (te): [1, 10, 50, 100, 500, 1000, 2000, 5000] (categorical)
- learning_rate (lr): [1e-4, 1.0] (log-uniform)
- transfer_lr (tlr): [1e-4, 10.0] (log-uniform)
- lifetime: [100, 1000, 10000, 46505, 100000] (categorical)

Baseline settings:
- Model: google/mobilebert-uncased
- Dataset: SST-2 full (67,349 samples)
- Epochs: 3
- Batch size: 32

Usage:
    # Single GPU
    nohup python sweep_bert_base_optuna.py --n_trials 200 > optuna_sweep.log 2>&1 &

    # Parallel with multiple workers (each worker on different GPU)
    nohup python sweep_bert_base_optuna.py --n_trials 200 --n_jobs 4 > optuna_sweep.log 2>&1 &

    # Resume from existing study
    nohup python sweep_bert_base_optuna.py --n_trials 200 --study_name my_study --storage sqlite:///optuna.db > optuna_sweep.log 2>&1 &
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

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
    EarlyStoppingCallback,
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
# Configuration
# =============================================================================

SEED = 42
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32  # Same as lora_on_analog_hardware
NUM_EPOCHS = 15  # Same as lora_on_analog_hardware
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

WANDB_PROJECT = "lrtt-mobilebert-optuna"
OUTPUT_DIR = "/tmp/lrtt_optuna_results"

# Global cache
_tokenizer = None
_train_dataset = None
_eval_dataset = None
_metric = None


def get_data():
    """Get cached data."""
    global _tokenizer, _train_dataset, _eval_dataset, _metric

    if _tokenizer is None:
        print("Loading tokenizer and dataset...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)

        def preprocess(examples):
            return _tokenizer(
                examples["sentence"],
                padding="max_length",
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
            )

        tokenized = raw_datasets.map(preprocess, batched=True)
        _train_dataset = tokenized["train"]
        _eval_dataset = tokenized["validation"]
        _metric = evaluate.load("glue", TASK_NAME)
        print(f"Data loaded: {len(_train_dataset)} train, {len(_eval_dataset)} eval")

    return _tokenizer, _train_dataset, _eval_dataset, _metric


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


class OptunaPruningCallback(TrainerCallback):
    """Callback for Optuna pruning."""

    def __init__(self, trial, metric_name="eval_accuracy"):
        self.trial = trial
        self.metric_name = metric_name

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            value = metrics.get(self.metric_name, 0)
            self.trial.report(value, state.epoch)

            if self.trial.should_prune():
                raise optuna.TrialPruned()


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample hyperparameters
    rank = trial.suggest_categorical("rank", RANKS)
    te = trial.suggest_categorical("transfer_every", TRANSFER_EVERYS)
    lr = trial.suggest_float("learning_rate", 1e-4, 1.0, log=True)
    tlr = trial.suggest_float("transfer_lr", 1e-4, 10.0, log=True)
    lifetime = trial.suggest_categorical("lifetime", LIFETIMES)

    exp_name = f"trial{trial.number}_r{rank}_te{te}"

    print(f"\n[Trial {trial.number}] rank={rank}, te={te}, lr={lr:.6f}, tlr={tlr:.6f}, lifetime={lifetime}")

    set_seed(SEED)

    # Initialize wandb
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project=WANDB_PROJECT,
                name=exp_name,
                config={
                    "trial": trial.number,
                    "model": MODEL_NAME,
                    "optimizer": "AnalogAdam",
                    "rank": rank,
                    "transfer_every": te,
                    "learning_rate": lr,
                    "transfer_lr": tlr,
                    "lifetime": lifetime,
                    "batch_size": BATCH_SIZE,
                    "epochs": NUM_EPOCHS,
                },
                reinit=True,
            )
        except:
            pass

    try:
        tokenizer, train_dataset, eval_dataset, metric = get_data()

        # Load model (use safetensors to avoid torch.load vulnerability)
        model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=model_config, use_safetensors=True
        )

        # Convert to LRTT
        rpu_config = create_lrtt_config(rank, te, tlr, lifetime)
        model = convert_to_analog(model, rpu_config, exclude_modules=["classifier"])
        model.to(device)

        # Optimizer - AnalogAdam
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
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            report_to="wandb" if WANDB_AVAILABLE else "none",
            run_name=exp_name,
            seed=SEED,
            remove_unused_columns=True,
            disable_tqdm=True,
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
            callbacks=[
                OptunaPruningCallback(trial),
                EarlyStoppingCallback(early_stopping_patience=3),
            ],
        )

        trainer.train()
        eval_result = trainer.evaluate()
        eval_acc = eval_result.get("eval_accuracy", 0)

        print(f"[Trial {trial.number}] Final accuracy: {eval_acc:.4f}")

        if WANDB_AVAILABLE:
            try:
                wandb.log({"final_accuracy": eval_acc})
            except:
                pass

        return eval_acc

    except optuna.TrialPruned:
        raise

    except Exception as e:
        print(f"[Trial {trial.number}] Error: {e}")
        return 0.0

    finally:
        if WANDB_AVAILABLE:
            try:
                wandb.finish()
            except:
                pass

        try:
            del model
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    study_name = args.study_name or f"lrtt_mobilebert_{timestamp}"

    print("=" * 60)
    print("MobileBERT LRTT Bayesian Search (Optuna)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Optimizer: AnalogAdam")
    print(f"Trials: {args.n_trials}")
    print(f"Jobs: {args.n_jobs}")
    print(f"Study: {study_name}")
    print("-" * 60)
    print(f"Ranks: {RANKS}")
    print(f"Transfer Every: {TRANSFER_EVERYS}")
    print(f"LR range: [1e-4, 1.0] (log)")
    print(f"TLR range: [1e-4, 10.0] (log)")
    print(f"Lifetimes: {LIFETIMES}")
    print("=" * 60)

    # Pre-load data
    get_data()

    # Create study
    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=1)

    if args.storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

    # Run optimization
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # Results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Top 10 trials
    print(f"\nTop 10 trials:")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values("value", ascending=False).head(10)
    print(trials_df[["number", "value", "params_rank", "params_transfer_every",
                      "params_learning_rate", "params_transfer_lr", "params_lifetime"]].to_string())

    # Save results
    results = {
        "study_name": study_name,
        "best_trial": study.best_trial.number,
        "best_accuracy": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }

    results_file = f"{OUTPUT_DIR}/optuna_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")

    # Save study as pickle for resume
    import pickle
    study_file = f"{OUTPUT_DIR}/optuna_study_{timestamp}.pkl"
    with open(study_file, "wb") as f:
        pickle.dump(study, f)
    print(f"Study saved: {study_file}")


if __name__ == "__main__":
    main()
