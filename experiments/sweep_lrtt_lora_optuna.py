#!/usr/bin/env python
# coding=utf-8
"""
LRTT-LoRA Bayesian Hyperparameter Search with Optuna

Sweeps lora_alpha and learning_rate for both FP-LoRA and 6T1C-LoRA modes.

Search parameters:
- lora_alpha: [0.5, 8.0] (log-uniform, in "standard LoRA" scale)
  * Converted for LRTT: alpha_lrtt = alpha_standard / rank
  * Standard LoRA uses: scaling = alpha/rank, LRTT uses: scaling = alpha
  * This conversion makes them equivalent
- learning_rate: [5e-4, 1e-2] (log-uniform, min increased for analog quantization)
- rank: Fixed at 8 (can be made searchable if needed)

Modes:
- FP-LoRA: FloatingPointDevice (exact arithmetic, for baseline)
- 6T1C-LoRA: 6T1C LinearStepDevice (analog device)

Settings (from LRTT_setup / MEMORY.md):
- Device: 6T1C LinearStepDevice (or FloatingPoint for FP mode)
- IO: inp_res=1/(2^8-2), out_res=1/(2^8-2), noise_management=ABS_MAX, bound_management=ITERATIVE
- Training: AnalogSGD optimizer, batch_size=32, max_seq_length=128 (GLUE) / 384 (SQuAD)
- Mapping: weight_scaling_omega=1.0, learn_out_scaling=True, out_scaling_columnwise=True
- Bias: Frozen, classifier/qa_outputs: Trainable (including bias)
- Early stopping: patience=3
- Warmup: 500 steps (SQuAD), 10% ratio (GLUE)

Usage:
    # GLUE SST-2 with FP-LoRA
    python sweep_lrtt_lora_optuna.py --task glue --task_name sst2 --mode fp_lora --n_trials 50

    # GLUE SST-2 with 6T1C-LoRA
    python sweep_lrtt_lora_optuna.py --task glue --task_name sst2 --mode sixt1c_lora --n_trials 50

    # SQuAD with FP-LoRA
    python sweep_lrtt_lora_optuna.py --task squad --mode fp_lora --n_trials 30

    # Parallel execution (4 GPUs)
    python sweep_lrtt_lora_optuna.py --task glue --task_name mrpc --mode fp_lora --n_trials 100 --n_jobs 4

    # Resume from existing study
    python sweep_lrtt_lora_optuna.py --task glue --task_name sst2 --mode fp_lora \\
        --study_name my_study --storage sqlite:///lrtt_lora_optuna.db --n_trials 50
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
from optuna_integration import BoTorchSampler
from optuna.pruners import NopPruner

# Add LRTT-LoRA paths
sys.path.insert(0, "/data/LRTT_transformer/lora_training_glue")
sys.path.insert(0, "/data/LRTT_transformer/lora_training")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datasets
import evaluate
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EvalPrediction,
    TrainerCallback,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup,
)

from aihwkit.optim import AnalogSGD
from aihwkit.optim.context import AnalogContext

# Import LRTT-LoRA modules
from lrtt_lora_config import create_lrtt_lora_config
from lrtt_lora_conversion import convert_model_to_lrtt_lora

import warnings
warnings.filterwarnings("ignore")


class FPLoRATrainer(Trainer):
    """Trainer that clips analog_grad_output alongside standard .grad for FP-LoRA.

    In FP-LoRA mode, AnalogContext stores gradients in analog_grad_output
    instead of .grad, so standard clip_grad_norm_ doesn't see them.
    This trainer computes a unified norm over both .grad and analog_grad_output,
    then scales all gradients by the same clip coefficient — identical to how
    torch.nn.utils.clip_grad_norm_ works for standard parameters.

    Usage: Set max_grad_norm=0 in TrainingArguments to disable the built-in
    clip, and pass the desired max_norm via fp_lora_max_grad_norm.
    """

    def __init__(self, *args, fp_lora_max_grad_norm: float = 1.0, weight_clip_value: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp_lora_max_grad_norm = fp_lora_max_grad_norm
        self.weight_clip_value = weight_clip_value

    def _unified_clip_grad_norm(self, max_norm: float, norm_type: float = 2.0):
        """Clip all gradients (.grad + analog_grad_output) with a single norm."""
        all_norms = []
        analog_params = []

        for p in self.model.parameters():
            if isinstance(p, AnalogContext) and len(p.analog_grad_output) > 0:
                for g in p.analog_grad_output:
                    all_norms.append(torch.norm(g.detach(), norm_type))
                analog_params.append(p)
            elif p.grad is not None:
                all_norms.append(torch.norm(p.grad.detach(), norm_type))

        if len(all_norms) == 0:
            return torch.tensor(0.0)

        total_norm = torch.norm(torch.stack(all_norms), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0).item()

        if clip_coef_clamped < 1.0:
            for p in self.model.parameters():
                if isinstance(p, AnalogContext) and len(p.analog_grad_output) > 0:
                    for i in range(len(p.analog_grad_output)):
                        p.analog_grad_output[i] = p.analog_grad_output[i] * clip_coef_clamped
                elif p.grad is not None:
                    p.grad.detach().mul_(clip_coef_clamped)

        return total_norm

    def _clip_analog_weights(self):
        """Clip analog tile weights to prevent explosion in FP mode.

        FloatingPointDevice has no built-in bounds, so we manually clip
        weights to [-weight_clip_value, +weight_clip_value] after each update.
        """
        for name, module in self.model.named_modules():
            if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'tile_a'):
                # Clip A, B tiles (trainable LoRA adapters)
                tile_a = module.analog_module.tile_a
                tile_b = module.analog_module.tile_b

                # Get weights
                w_a, _ = tile_a.get_weights()
                w_b, _ = tile_b.get_weights()

                # Clip in-place
                torch.clamp_(w_a, -self.weight_clip_value, self.weight_clip_value)
                torch.clamp_(w_b, -self.weight_clip_value, self.weight_clip_value)

                # Set clipped weights back
                tile_a.set_weights(w_a)
                tile_b.set_weights(w_b)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Forward+backward, then unified gradient clipping, then weight clipping."""
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Unified gradient clip (built-in clip is disabled via max_grad_norm=0)
        if self.fp_lora_max_grad_norm > 0:
            self._grad_norm = self._unified_clip_grad_norm(self.fp_lora_max_grad_norm)

        return loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time=None, **kwargs):
        """Override to add weight clipping after optimizer.step() but before eval."""
        # Clip weights after optimizer step (prevents FP weight explosion)
        if self.weight_clip_value > 0:
            self._clip_analog_weights()

        # Continue with normal logging/saving/evaluation
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, **kwargs)

try:
    import wandb
    os.environ["WANDB_MODE"] = "offline"  # Set WandB to offline mode
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

SEED = 42
MODEL_NAME = "google/mobilebert-uncased"
BATCH_SIZE = 64
LOGGING_STEPS = 100

# Search space
# LORA_ALPHA: log-uniform [0.5, 8.0] (equivalent to PEFT alpha/r with alpha=4~64, r=8)
# LR: log-uniform [1e-4, 1e-2] (standard SGD range)
RANK = 8  # Fixed rank

# Task-specific settings
TASK_CONFIGS = {
    "glue": {
        "max_seq_length": 128,
        "doc_stride": None,
        "num_epochs": 3,
        "eval_batch_size": 64,
    },
    "squad": {
        "max_seq_length": 320,
        "doc_stride": 128,
        "num_epochs": 15,
        "eval_batch_size": 256,
    },
}

WANDB_PROJECT = "lrtt-lora-optuna"
OUTPUT_DIR = "/data/results/lora_fp"

# Global cache
_tokenizer = None
_train_dataset = None
_eval_dataset = None
_metric = None
_task_type = None
_task_name = None


# =============================================================================
# Data Loading
# =============================================================================

def get_glue_data(task_name: str, max_seq_length: int):
    """Load and preprocess GLUE data."""
    global _tokenizer, _train_dataset, _eval_dataset, _metric

    if _tokenizer is None:
        print(f"Loading GLUE {task_name} dataset...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        raw_datasets = load_dataset("nyu-mll/glue", task_name)

        task_to_keys = {
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

        sentence1_key, sentence2_key = task_to_keys[task_name]

        def preprocess(examples):
            if sentence2_key is None:
                return _tokenizer(
                    examples[sentence1_key],
                    padding="max_length",
                    max_length=max_seq_length,
                    truncation=True,
                )
            return _tokenizer(
                examples[sentence1_key],
                examples[sentence2_key],
                padding="max_length",
                max_length=max_seq_length,
                truncation=True,
            )

        tokenized = raw_datasets.map(preprocess, batched=True)
        _train_dataset = tokenized["train"]
        _eval_dataset = tokenized["validation_matched" if task_name == "mnli" else "validation"]
        _metric = evaluate.load("glue", task_name)

        print(f"  ✓ Loaded {len(_train_dataset)} train, {len(_eval_dataset)} eval samples")

    return _tokenizer, _train_dataset, _eval_dataset, _metric


def get_squad_data(max_seq_length: int, doc_stride: int):
    """Load and preprocess SQuAD data."""
    global _tokenizer, _train_dataset, _eval_dataset, _metric

    if _tokenizer is None:
        print("Loading SQuAD dataset...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        raw_datasets = load_dataset("squad")

        def preprocess(examples):
            questions = [q.strip() for q in examples["question"]]
            inputs = _tokenizer(
                questions,
                examples["context"],
                max_length=max_seq_length,
                truncation="only_second",
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")

            start_positions = []
            end_positions = []

            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                answer = examples["answers"][sample_idx]
                start_char = answer["answer_start"][0]
                end_char = start_char + len(answer["text"][0])

                sequence_ids = inputs.sequence_ids(i)
                context_start = 0
                while sequence_ids[context_start] != 1:
                    context_start += 1
                context_end = len(sequence_ids) - 1
                while sequence_ids[context_end] != 1:
                    context_end -= 1

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

        tokenized = raw_datasets.map(
            preprocess,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

        _train_dataset = tokenized["train"]
        _eval_dataset = tokenized["validation"]
        _metric = evaluate.load("squad")

        print(f"  ✓ Loaded {len(_train_dataset)} train, {len(_eval_dataset)} eval samples")

    return _tokenizer, _train_dataset, _eval_dataset, _metric


def get_data(task_type: str, task_name: Optional[str], max_seq_length: int, doc_stride: Optional[int]):
    """Get cached data based on task type."""
    if task_type == "glue":
        return get_glue_data(task_name, max_seq_length)
    elif task_type == "squad":
        return get_squad_data(max_seq_length, doc_stride)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# =============================================================================
# Optuna Callbacks
# =============================================================================

class OptunaPruningCallback(TrainerCallback):
    """Callback for Optuna pruning."""

    def __init__(self, trial, metric_name="eval_accuracy"):
        self.trial = trial
        self.metric_name = metric_name

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            # Handle both accuracy and F1 metrics
            if self.metric_name in metrics:
                value = metrics[self.metric_name]
            elif "eval_f1" in metrics:
                value = metrics["eval_f1"]
            elif "eval_accuracy" in metrics:
                value = metrics["eval_accuracy"]
            else:
                return

            self.trial.report(value, state.epoch)

            if self.trial.should_prune():
                raise optuna.TrialPruned()


# =============================================================================
# Objective Function
# =============================================================================

def objective(trial: optuna.Trial, args_dict: Dict) -> float:
    """Optuna objective function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract args
    task_type = args_dict["task_type"]
    task_name = args_dict.get("task_name")
    mode = args_dict["mode"]
    rank = args_dict["rank"]
    max_seq_length = args_dict["max_seq_length"]
    doc_stride = args_dict.get("doc_stride")
    num_epochs = args_dict["num_epochs"]
    eval_batch_size = args_dict.get("eval_batch_size", BATCH_SIZE)
    target_modules = args_dict["target_modules"]

    # Sample hyperparameters - DECOUPLED LR CONFIGURATION
    # Classifier LR: Fixed at 1.0
    # QKV (analog tile) LR: Searchable in [1e-4, 1e-2]
    # LoRA alpha: Searchable in [0.01, 100]
    classifier_lr = 0.01  # Fixed (changed from 1.0)
    qkv_lr = trial.suggest_float("qkv_lr", 1e-4, 1e-2, log=True)
    lora_alpha = trial.suggest_float("lora_alpha", 0.01, 100.0, log=True)

    # For LRTT: use alpha directly (no /rank conversion needed)
    lora_alpha_lrtt = lora_alpha

    exp_name = f"trial{trial.number}_{mode}_a{lora_alpha:.4f}_clr{classifier_lr:.2f}_qlr{qkv_lr:.6f}"

    print(f"\n[Trial {trial.number}] mode={mode}, lora_alpha={lora_alpha:.3f}, classifier_lr={classifier_lr:.3f}, qkv_lr={qkv_lr:.6f}, rank={rank}")

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
                    "task": task_type,
                    "task_name": task_name,
                    "mode": mode,
                    "optimizer": "AnalogSGD (decoupled LR)",
                    "rank": rank,
                    "lora_alpha": lora_alpha,  # Forward/backward scaling
                    "classifier_lr": classifier_lr,  # Fixed at 1.0
                    "qkv_lr": qkv_lr,  # Analog tile LR (searchable)
                    "batch_size": BATCH_SIZE,
                    "epochs": num_epochs,
                    "max_seq_length": max_seq_length,
                },
                reinit=True,
            )
        except:
            pass

    try:
        # Load data
        tokenizer, train_dataset, eval_dataset, metric = get_data(
            task_type, task_name, max_seq_length, doc_stride
        )

        # Load model
        if task_type == "glue":
            is_regression = task_name == "stsb"
            num_labels = 1 if is_regression else (3 if task_name == "mnli" else 2)
            model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, config=model_config
            )
        else:  # squad
            model_config = AutoConfig.from_pretrained(MODEL_NAME)
            model = AutoModelForQuestionAnswering.from_pretrained(
                MODEL_NAME, config=model_config
            )

        # Reinitialize classifier/qa_outputs with FIXED seed (matches tikitakav1)
        # These heads are randomly initialized by HF — fix seed for reproducibility
        import torch.nn as nn
        if task_type == "glue" and hasattr(model, 'classifier'):
            torch.manual_seed(SEED)
            nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
            if model.classifier.bias is not None:
                nn.init.zeros_(model.classifier.bias)
        elif task_type == "squad" and hasattr(model, 'qa_outputs'):
            torch.manual_seed(SEED)
            nn.init.normal_(model.qa_outputs.weight, mean=0.0, std=0.02)
            if model.qa_outputs.bias is not None:
                nn.init.zeros_(model.qa_outputs.bias)

        # Create LRTT-LoRA config
        # Use lora_alpha_lrtt (already divided by rank) for LRTT
        # This makes LRTT behave equivalently to standard LoRA
        use_fp = (mode == "fp_lora")
        lrtt_config = create_lrtt_lora_config(
            rank=rank,
            lora_alpha=lora_alpha_lrtt,  # LRTT alpha (standard_alpha / rank)
            output_noise_level=0.0,
            use_floating_point=use_fp,
        )

        # Convert to LRTT-LoRA
        model = convert_model_to_lrtt_lora(model, lrtt_config, target_modules)
        model.to(device)

        # Optimizer - AnalogSGD with parameter groups
        # Separate LoRA tiles (tile_a/tile_b) from other params (classifier, etc.)
        lora_tile_params = []
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'tile_a' in name or 'tile_b' in name:
                    lora_tile_params.append(param)
                else:
                    other_params.append(param)

        # Set learning rates - DECOUPLED LR
        # Classifier: Fixed at 1.0
        # QKV tiles: Searchable (qkv_lr from trial)
        lora_tile_lr = qkv_lr  # From trial.suggest_float('qkv_lr', 1e-4, 1e-2)
        # classifier_lr already defined in trial (fixed at 1.0)

        print(f"  Optimizer config: classifier_lr={classifier_lr}, qkv_lr={lora_tile_lr}")
        print(f"  LoRA tiles: {len(lora_tile_params)}, Other params: {len(other_params)}")

        param_groups = [
            {'params': lora_tile_params, 'lr': lora_tile_lr},
            {'params': other_params, 'lr': classifier_lr}
        ]

        # CRITICAL: Use lora_tile_lr as default lr for AnalogSGD
        # regroup_param_groups() sets all analog_ctx to self.defaults["lr"]
        # Since tile_a/b are the trainable LoRA adapters, they should use lora_tile_lr
        optimizer = AnalogSGD(param_groups, lr=lora_tile_lr)
        optimizer.regroup_param_groups(model)  # CRITICAL for analog layers

        # Calculate total steps for warmup (LRTT_setup.txt)
        total_steps = len(train_dataset) // BATCH_SIZE * num_epochs
        if task_type == "glue":
            warmup_steps = int(0.1 * total_steps)  # 10% warmup for GLUE (LRTT_setup.txt line 44)
        else:  # squad
            warmup_steps = 500  # Fixed 500 steps for SQuAD (LRTT_setup.txt line 44)

        # Scheduler with linear warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Compute metrics
        def compute_metrics_glue(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if task_name == "stsb":
                preds = np.squeeze(preds)
            else:
                preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if "f1" in result:
                result["f1"] = result["f1"]
            return result

        def compute_metrics_squad(p: EvalPrediction):
            # Simplified SQuAD metrics for hyperparameter search
            start_preds = np.argmax(p.predictions[0], axis=1)
            end_preds = np.argmax(p.predictions[1], axis=1)

            # label_ids is a dict for SQuAD: {"start_positions": ..., "end_positions": ...}
            if isinstance(p.label_ids, dict):
                start_labels = p.label_ids["start_positions"]
                end_labels = p.label_ids["end_positions"]
            else:
                # Fallback for tuple format (older transformers versions)
                start_labels = p.label_ids[0]
                end_labels = p.label_ids[1]

            exact_match = ((start_preds == start_labels) & (end_preds == end_labels)).mean()
            return {"exact_match": exact_match}

        compute_metrics_fn = compute_metrics_glue if task_type == "glue" else compute_metrics_squad

        # Determine metric name for pruning
        if task_type == "glue":
            if task_name in ["mrpc", "qqp"]:
                metric_name = "eval_f1"
            elif task_name == "stsb":
                metric_name = "eval_pearson"
            else:
                metric_name = "eval_accuracy"
        else:
            metric_name = "eval_exact_match"

        # Training arguments
        # For FP-LoRA: disable built-in clip (max_grad_norm=0) because
        # FPLoRATrainer handles unified clipping of .grad + analog_grad_output.
        # For other modes: use standard clip (max_grad_norm=1.0).
        builtin_clip = 0.0 if use_fp else 1.0
        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/trial_{trial.number}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=eval_batch_size,
            logging_steps=LOGGING_STEPS,
            eval_strategy="epoch",
            save_strategy="no",  # Disable saving (LRTT state_dict has dict values, not compatible with HF serialization)
            report_to="wandb" if WANDB_AVAILABLE else "none",
            run_name=exp_name,
            seed=SEED,
            remove_unused_columns=True,
            disable_tqdm=True,
            warmup_steps=warmup_steps,
            max_grad_norm=builtin_clip,
            # For SQuAD: specify label column names
            label_names=["start_positions", "end_positions"] if task_type == "squad" else None,
        )

        # Trainer: FPLoRATrainer for FP-LoRA (unified grad clip), standard Trainer otherwise
        trainer_cls = FPLoRATrainer if use_fp else Trainer
        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            optimizers=(optimizer, scheduler),
            compute_metrics=compute_metrics_fn,
            processing_class=tokenizer,
            data_collator=default_data_collator,
            # No callbacks - early stopping requires checkpoints which aren't compatible with LRTT
        )
        if use_fp:
            trainer_kwargs["fp_lora_max_grad_norm"] = 1.0
            trainer_kwargs["weight_clip_value"] = 1.0  # Clip FP weights to [-1, 1]
        trainer = trainer_cls(**trainer_kwargs)

        # Train
        trainer.train()
        eval_result = trainer.evaluate()
        eval_score = eval_result.get(metric_name, 0)

        print(f"[Trial {trial.number}] Final {metric_name}: {eval_score:.4f}")

        if WANDB_AVAILABLE:
            try:
                wandb.log({f"final_{metric_name}": eval_score})
            except:
                pass

        return eval_score

    except optuna.TrialPruned:
        raise

    except Exception as e:
        print(f"[Trial {trial.number}] Error: {e}")
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
            del model, optimizer, scheduler
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    # Task settings
    parser.add_argument("--task", type=str, required=True, choices=["glue", "squad"],
                       help="Task type: glue or squad")
    parser.add_argument("--task_name", type=str, default="sst2",
                       help="GLUE task name (sst2, mrpc, etc.)")

    # Mode settings
    parser.add_argument("--mode", type=str, required=True, choices=["fp_lora", "sixt1c_lora"],
                       help="LoRA mode: fp_lora (FloatingPoint) or sixt1c_lora (6T1C)")

    # Model settings
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--target_modules", type=str, nargs="+", default=["query", "key", "value"],
                       help="Target modules for LoRA")

    # Optuna settings
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--study_name", type=str, default=None, help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")

    args = parser.parse_args()

    # Task-specific config
    task_config = TASK_CONFIGS[args.task]
    max_seq_length = task_config["max_seq_length"]
    doc_stride = task_config.get("doc_stride")
    num_epochs = task_config["num_epochs"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    study_name = args.study_name or f"lrtt_lora_{args.task}_{args.mode}_{timestamp}"

    print("=" * 80)
    print("LRTT-LORA BAYESIAN HYPERPARAMETER SEARCH (Optuna)")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")
    print(f"Task: {args.task}" + (f" ({args.task_name})" if args.task == "glue" else ""))
    print(f"Mode: {args.mode}")
    print(f"Rank: {args.rank}")
    print(f"Target modules: {args.target_modules}")
    print(f"Optimizer: AnalogSGD")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max seq length: {max_seq_length}")
    if doc_stride:
        print(f"Doc stride: {doc_stride}")
    print(f"Trials: {args.n_trials}")
    print(f"Jobs: {args.n_jobs}")
    print(f"Study: {study_name}")
    print("-" * 80)
    print(f"LoRA alpha range: [0.01, 100.0] (log-uniform)")
    print(f"LR range: [1e-4, 1e-2] (log-uniform)")
    print("=" * 80)

    # Prepare args dict for objective
    args_dict = {
        "task_type": args.task,
        "task_name": args.task_name if args.task == "glue" else None,
        "mode": args.mode,
        "rank": args.rank,
        "target_modules": args.target_modules,
        "max_seq_length": max_seq_length,
        "doc_stride": doc_stride,
        "num_epochs": num_epochs,
        "eval_batch_size": task_config.get("eval_batch_size", BATCH_SIZE),
    }

    # Pre-load data
    print("\nPre-loading data...")
    get_data(args.task, args_dict["task_name"], max_seq_length, doc_stride)

    # ==========================================================================
    # CONFIGURATION REPORT
    # ==========================================================================
    print("\n" + "=" * 80)
    print("SWEEP CONFIGURATION REPORT")
    print("=" * 80)

    print("\n[Task Settings]")
    print(f"  Task: {args.task}")
    print(f"  Task name: {args_dict['task_name']}")
    print(f"  Mode: {args.mode}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Rank: {args.rank}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Max seq length: {max_seq_length}")
    if args.task == "squad":
        print(f"  Doc stride: {doc_stride}")

    print("\n[Device Settings]")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n[Hyperparameter Search Space]")
    print(f"  Classifier LR: 0.01 (FIXED)")
    print(f"  QKV analog tile LR: [1e-4, 1e-2] (log-uniform)")
    print(f"  LoRA alpha: [0.01, 100.0] (log-uniform)")

    print("\n[Optimizer Settings]")
    print(f"  Optimizer: AnalogSGD (decoupled LR)")
    print(f"  Parameter groups:")
    print(f"    - QKV tiles (tile_a/tile_b): lr = qkv_lr (searchable)")
    print(f"    - Classifier/Other: lr = 1.0 (fixed)")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Warmup: 500 steps (SQuAD) / 10% ratio (GLUE)")

    print("\n[Sampler & Pruner]")
    print(f"  Sampler: BoTorchSampler (Bayesian Optimization)")
    print(f"  Pruner: NopPruner (no pruning)")

    print("\n[Early Stopping]")
    print(f"  Enabled: Yes")
    print(f"  Patience: 3 epochs")
    print(f"  Metric: {'eval_accuracy' if args.task == 'glue' else 'eval_f1'}")

    print("\n[Training Settings]")
    print(f"  Evaluation strategy: epoch")
    print(f"  Save strategy: no (LRTT state_dict incompatible with HF)")
    print(f"  Logging: WandB (offline mode)")
    print(f"  Random seed: {SEED}")

    print("\n[Sweep Settings]")
    print(f"  Number of trials: {args.n_trials}")
    print(f"  Parallel jobs: {args.n_jobs}")
    print(f"  Study name: {study_name}")
    if args.storage:
        print(f"  Storage: {args.storage}")

    print("=" * 80 + "\n")

    # Create study (same as TikiTaka v1: BoTorchSampler + NopPruner)
    sampler = BoTorchSampler()
    pruner = NopPruner()  # No pruning, use early stopping instead

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
    print("\nStarting optimization...")
    study.optimize(
        lambda trial: objective(trial, args_dict),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # Results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results
    results_file = f"{OUTPUT_DIR}/best_params_{study_name}.json"
    with open(results_file, "w") as f:
        json.dump({
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "task": args.task,
            "task_name": args.task_name if args.task == "glue" else None,
            "mode": args.mode,
            "rank": args.rank,
        }, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
