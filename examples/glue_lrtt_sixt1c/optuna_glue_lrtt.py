#!/usr/bin/env python
"""LRTT Optuna Bayesian Search for GLUE Tasks (2-Phase).

Based on run_glue_lrtt.py to ensure identical architecture with Analog LoRA:
- Model: bert-base-uncased (same as Analog LoRA)
- HuggingFace Trainer
- Same data preprocessing (max_seq_length=128)

Phase 1: Fast search (1 epoch, 100 trials)
Phase 2: Full training (10 epochs) on top 10 configurations

Target: Match Analog LoRA baseline performance
"""

import logging
import os
import sys
import json
import gc
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
import wandb
import warnings
warnings.filterwarnings("ignore")

# Add LRTT src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LRTT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(LRTT_ROOT, "src"))

import datasets
import evaluate
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

# LRTT / AIHWKIT imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam

# Core LRTT imports
from aihwkit.simulator.configs.lrtt_config import lrtt_sixt1c_ab_ideal_config

# Suppress transformers logging
transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Constants (Same as Analog LoRA)
# ============================================================================
MODEL_NAME = "bert-base-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
SEED = 42

# Phase settings
PHASE1_EPOCHS = 1
PHASE2_EPOCHS = 10  # Same as Analog LoRA for small tasks

# Dataset info
TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# Cache for dataset and tokenizer
_cached_dataset = None
_cached_tokenizer = None
_cached_metric = None


def get_dataset_and_tokenizer():
    """Get cached dataset, tokenizer, and metric."""
    global _cached_dataset, _cached_tokenizer, _cached_metric

    if _cached_dataset is None:
        print(f"Loading dataset: {TASK_NAME}")
        _cached_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)

        sentence1_key, sentence2_key = TASK_TO_KEYS[TASK_NAME]

        def preprocess_function(examples):
            args = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )
            return _cached_tokenizer(
                *args,
                padding="max_length",
                max_length=MAX_SEQ_LENGTH,
                truncation=True
            )

        _cached_dataset = raw_datasets.map(
            preprocess_function,
            batched=True,
            desc="Tokenizing",
        )

        _cached_metric = evaluate.load("glue", TASK_NAME)
        print(f"Dataset loaded: {len(_cached_dataset['train'])} train, {len(_cached_dataset['validation'])} val")

    return _cached_dataset, _cached_tokenizer, _cached_metric


def run_training(
    analog_lr: float,
    transfer_lr: float,
    transfer_every: int,
    rank: int,
    reinit_gain: float,
    weight_decay: float = 0.0,
    num_epochs: int = 1,
    seed: int = SEED,
    output_dir: str = None,
    verbose: bool = False,
) -> dict:
    """Run LRTT training with specified hyperparameters.

    Uses identical setup to run_glue_lrtt.py and Analog LoRA.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get cached data
    raw_datasets, tokenizer, metric = get_dataset_and_tokenizer()

    # Determine number of labels
    is_regression = TASK_NAME == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Load model (fresh for each trial)
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        finetuning_task=TASK_NAME,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
    )

    # Create LRTT configuration (6T1C)
    rpu_config = lrtt_sixt1c_ab_ideal_config(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=1.0,  # Fixed
    )

    # Apply LRTT settings
    rpu_config.device.forward_inject = False
    rpu_config.device.transfer_method = "onehot"
    rpu_config.device.update_mode = "lora"
    rpu_config.device.reinit_mode = "standard"
    rpu_config.device.transfer_lr = transfer_lr
    rpu_config.device.reinit_gain = reinit_gain

    # Convert to LRTT analog (same as run_glue_lrtt.py)
    # Exclude only classifier - matches Analog LoRA structure
    exclude_modules = ["classifier"]
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude_modules)
    model.to(device)

    # Create optimizer (AnalogAdam)
    optimizer = AnalogAdam(
        model.parameters(),
        lr=analog_lr,
        weight_decay=weight_decay,
    )

    # Compute metrics function
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Output directory
    if output_dir is None:
        output_dir = f"/tmp/lrtt_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments (same as Analog LoRA)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=analog_lr,  # Not used directly (optimizer handles it)
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",  # Disable default reporting
        seed=seed,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"],
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    # Train
    train_result = trainer.train()

    # Evaluate
    eval_result = trainer.evaluate()

    # Get training metrics
    train_loss_history = [
        log["loss"] for log in trainer.state.log_history if "loss" in log
    ]

    initial_loss = train_loss_history[0] if train_loss_history else 0
    final_loss = train_loss_history[-1] if train_loss_history else 0
    min_loss = min(train_loss_history) if train_loss_history else 0

    loss_reduction_pct = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0

    # Get eval accuracy
    eval_accuracy = eval_result.get("eval_accuracy", 0)

    # Cleanup
    del model, trainer, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    # Remove temp output dir
    if output_dir.startswith("/tmp"):
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "min_loss": min_loss,
        "loss_reduction_pct": loss_reduction_pct,
        "eval_accuracy": eval_accuracy,
        "eval_loss": eval_result.get("eval_loss", 0),
        "train_runtime": train_result.metrics.get("train_runtime", 0),
    }


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for Phase 1."""

    # Hyperparameter search space
    analog_lr = trial.suggest_float("analog_lr", 1e-5, 1e-2, log=True)
    transfer_lr = trial.suggest_float("transfer_lr", 0.001, 100.0, log=True)
    transfer_every = trial.suggest_int("transfer_every", 1, 100)
    rank = trial.suggest_categorical("rank", [4, 8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    reinit_gain = trial.suggest_float("reinit_gain", 0.01, 1.0, log=True)

    try:
        result = run_training(
            analog_lr=analog_lr,
            transfer_lr=transfer_lr,
            transfer_every=transfer_every,
            rank=rank,
            reinit_gain=reinit_gain,
            weight_decay=weight_decay,
            num_epochs=PHASE1_EPOCHS,
            seed=SEED,
        )

        eval_accuracy = result["eval_accuracy"]
        loss_reduction = result["loss_reduction_pct"]

        # Log to wandb
        wandb.log({
            "phase1/trial": trial.number,
            "phase1/eval_accuracy": eval_accuracy,
            "phase1/loss_reduction": loss_reduction,
            "phase1/initial_loss": result["initial_loss"],
            "phase1/final_loss": result["final_loss"],
            "phase1/eval_loss": result["eval_loss"],
            "phase1/analog_lr": analog_lr,
            "phase1/transfer_lr": transfer_lr,
            "phase1/transfer_every": transfer_every,
            "phase1/rank": rank,
            "phase1/reinit_gain": reinit_gain,
        })

        # Store for retrieval
        trial.set_user_attr("eval_accuracy", eval_accuracy)
        trial.set_user_attr("loss_reduction", loss_reduction)
        trial.set_user_attr("initial_loss", result["initial_loss"])
        trial.set_user_attr("final_loss", result["final_loss"])

        print(f"  Trial {trial.number}: acc={eval_accuracy:.4f}, loss_red={loss_reduction:.1f}%, "
              f"analog_lr={analog_lr:.6f}, transfer_lr={transfer_lr:.4f}, "
              f"te={transfer_every}, rank={rank}, reinit_gain={reinit_gain:.4f}")

        # Maximize accuracy (return negative for minimization)
        return -eval_accuracy

    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def main():
    print("=" * 70)
    print(" LRTT Optuna Bayesian Search (bert-base-uncased)")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME} (same as Analog LoRA)")
    print(f"Task: {TASK_NAME}")
    print(f"Phase 1: {PHASE1_EPOCHS} epoch, 100 trials")
    print(f"Phase 2: {PHASE2_EPOCHS} epochs on top 10")

    # Initialize wandb
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    wandb.init(
        project="lrtt-glue-sixt1c",
        name=f"optuna_bert_base_{timestamp}",
        config={
            "model": MODEL_NAME,
            "task": TASK_NAME,
            "phase1_epochs": PHASE1_EPOCHS,
            "phase2_epochs": PHASE2_EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_seq_length": MAX_SEQ_LENGTH,
        }
    )

    # Pre-load dataset
    print("\nPre-loading dataset...")
    get_dataset_and_tokenizer()

    # Create study
    study_name = f"lrtt_bert_base_{timestamp}"
    storage_path = os.path.join(SCRIPT_DIR, "optuna_studies", f"{study_name}.db")
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # Minimizing negative accuracy
        sampler=TPESampler(seed=SEED, n_startup_trials=20),
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    print(f"\nStudy: {study_name}")
    print(f"Storage: {storage_path}")
    print(f"\nSearch space (6 params):")
    print(f"  analog_lr: [1e-5, 1e-2] (log)")
    print(f"  transfer_lr: [0.001, 100.0] (log)")
    print(f"  transfer_every: [1, 100]")
    print(f"  rank: [4, 8, 16, 32]")
    print(f"  weight_decay: [0.0, 0.1]")
    print(f"  reinit_gain: [0.01, 1.0] (log)")
    print(f"\nFixed: lora_alpha=1.0, optimizer=AnalogAdam")

    # =========================================================================
    # Phase 1: Fast Search
    # =========================================================================
    print("\n" + "=" * 70)
    print(f" PHASE 1: Fast Search ({PHASE1_EPOCHS} epoch, 100 trials)")
    print("=" * 70)

    n_trials = 100
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    # Phase 1 Results
    print("\n" + "-" * 70)
    print(" Phase 1 Results (Top 10 by Accuracy)")
    print("-" * 70)

    trials_df = study.trials_dataframe()
    trials_df['accuracy'] = -trials_df['value']
    trials_df = trials_df.sort_values('accuracy', ascending=False)
    top10_df = trials_df.head(10)

    for i, (_, row) in enumerate(top10_df.iterrows()):
        print(f"{i+1}. Trial {int(row['number'])}: acc={row['accuracy']:.4f} | "
              f"analog_lr={row['params_analog_lr']:.6f}, "
              f"transfer_lr={row['params_transfer_lr']:.4f}, "
              f"te={int(row['params_transfer_every'])}, "
              f"rank={int(row['params_rank'])}")

    # =========================================================================
    # Phase 2: Full Training on Top 10
    # =========================================================================
    print("\n" + "=" * 70)
    print(f" PHASE 2: Full Training ({PHASE2_EPOCHS} epochs) on Top 10")
    print("=" * 70)

    phase2_results = []

    for i, (_, row) in enumerate(top10_df.iterrows()):
        config = {
            'analog_lr': row['params_analog_lr'],
            'transfer_lr': row['params_transfer_lr'],
            'transfer_every': int(row['params_transfer_every']),
            'rank': int(row['params_rank']),
            'weight_decay': row['params_weight_decay'],
            'reinit_gain': row['params_reinit_gain'],
        }

        print(f"\n[{i+1}/10] Full training for Trial {int(row['number'])}...")
        print(f"  Config: analog_lr={config['analog_lr']:.6f}, "
              f"transfer_lr={config['transfer_lr']:.4f}, "
              f"te={config['transfer_every']}, rank={config['rank']}")

        try:
            result = run_training(
                analog_lr=config['analog_lr'],
                transfer_lr=config['transfer_lr'],
                transfer_every=config['transfer_every'],
                rank=config['rank'],
                reinit_gain=config['reinit_gain'],
                weight_decay=config['weight_decay'],
                num_epochs=PHASE2_EPOCHS,
                seed=SEED,
            )

            phase2_results.append({
                'rank_phase1': i + 1,
                'trial_number': int(row['number']),
                'phase1_accuracy': row['accuracy'],
                'phase2_accuracy': result['eval_accuracy'],
                'phase2_loss_reduction': result['loss_reduction_pct'],
                'phase2_final_loss': result['final_loss'],
                'config': config,
            })

            print(f"  Phase 1: {row['accuracy']:.4f} -> Phase 2: {result['eval_accuracy']:.4f}")

            # Log to wandb
            wandb.log({
                "phase2/rank": i + 1,
                "phase2/trial_number": int(row['number']),
                "phase2/accuracy": result['eval_accuracy'],
                "phase2/loss_reduction": result['loss_reduction_pct'],
                "phase2/final_loss": result['final_loss'],
                "phase2/phase1_accuracy": row['accuracy'],
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            phase2_results.append({
                'rank_phase1': i + 1,
                'trial_number': int(row['number']),
                'phase1_accuracy': row['accuracy'],
                'phase2_accuracy': None,
                'error': str(e),
                'config': config,
            })

    # =========================================================================
    # Final Results
    # =========================================================================
    print("\n" + "=" * 70)
    print(" FINAL RESULTS (Phase 2 - Full Training)")
    print("=" * 70)

    valid_results = [r for r in phase2_results if r.get('phase2_accuracy') is not None]
    valid_results.sort(key=lambda x: x['phase2_accuracy'], reverse=True)

    print("\nRanked by Full Training Accuracy:")
    print("-" * 100)
    for i, r in enumerate(valid_results):
        cfg = r['config']
        print(f"{i+1}. Trial {r['trial_number']}: "
              f"Phase1={r['phase1_accuracy']:.4f} -> Phase2={r['phase2_accuracy']:.4f} | "
              f"analog_lr={cfg['analog_lr']:.6f}, transfer_lr={cfg['transfer_lr']:.4f}, "
              f"te={cfg['transfer_every']}, rank={cfg['rank']}")

    # Best config
    if valid_results:
        best = valid_results[0]
        print("\n" + "=" * 70)
        print(" BEST CONFIGURATION")
        print("=" * 70)
        print(f"Trial: {best['trial_number']}")
        print(f"Phase 2 Accuracy: {best['phase2_accuracy']:.4f}")
        print(f"Phase 2 Loss Reduction: {best['phase2_loss_reduction']:.1f}%")
        print(f"\nHyperparameters:")
        for k, v in best['config'].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

        # Log best to wandb
        wandb.log({
            "best/accuracy": best['phase2_accuracy'],
            "best/loss_reduction": best['phase2_loss_reduction'],
            "best/analog_lr": best['config']['analog_lr'],
            "best/transfer_lr": best['config']['transfer_lr'],
            "best/transfer_every": best['config']['transfer_every'],
            "best/rank": best['config']['rank'],
            "best/reinit_gain": best['config']['reinit_gain'],
        })

    # Save results
    output_dir = os.path.join(SCRIPT_DIR, "optuna_results")
    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        'model': MODEL_NAME,
        'task': TASK_NAME,
        'study_name': study_name,
        'phase1_trials': n_trials,
        'phase1_epochs': PHASE1_EPOCHS,
        'phase2_epochs': PHASE2_EPOCHS,
        'phase2_results': phase2_results,
        'best_config': valid_results[0] if valid_results else None,
        'timestamp': timestamp,
    }

    output_file = os.path.join(output_dir, f"optuna_bert_base_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    trials_df.to_csv(os.path.join(output_dir, f"optuna_phase1_{timestamp}.csv"), index=False)

    wandb.finish()

    print("\n" + "=" * 70)
    print(" Search Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
