#!/usr/bin/env python
# coding=utf-8
"""BERT-tiny GLUE Fine-tuning with LRTT (SoftBounds C + 6T1C A/B, decay mode).

Configuration from sweep_softbounds_lifetime.py:
- A/B tiles: LinearStepDevice (6T1C parameters)
- C tile: SoftBoundsDevice (no noise)
- Mode: decay
- forward_inject: False
- update_mode: lora

Usage:
    python run_bert_tiny_softbounds.py \
        --task_name sst2 \
        --rank 4 \
        --transfer_every 100 \
        --num_train_epochs 3
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

# Note: Using installed aihwkit from site-packages (with bias fix applied)

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
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# LRTT / AIHWKIT imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SoftBoundsDevice
from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.configs.lrtt_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

logger = logging.getLogger(__name__)

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

# SoftBounds config (no noise) - from sweep_softbounds_lifetime.py
SOFTBOUNDS_CONFIG = {
    'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
    'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
    'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
    'write_noise_std': 0.0, 'mult_noise': True,
}

# Hyperparameters from sweep (rank -> te -> {lr, tlr})
HYPERPARAMETERS = {
    4: {
        1: {"lr": 0.089054, "tlr": 0.001277},
        10: {"lr": 0.001735, "tlr": 0.008158},
        50: {"lr": 0.674107, "tlr": 0.011775},
        100: {"lr": 0.493706, "tlr": 0.011245},
        500: {"lr": 0.818908, "tlr": 9.649071},
        1000: {"lr": 0.007858, "tlr": 4.472098},
    },
}


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    """Convert lifetime to dt_batch_sec for sixt1c_ab preset."""
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    dt = -TAU_SEC * math.log(1 - delta)
    return dt


def create_lrtt_config(
    rank: int = 4,
    transfer_every: int = 100,
    lifetime: float = 46505.0,
    transfer_lr: float = 0.01,
) -> PythonLRTTRPUConfig:
    """Create LRTT config with SoftBounds C tile and 6T1C A/B tiles (decay mode).

    Args:
        rank: LRTT rank
        transfer_every: Transfer frequency in steps
        lifetime: Lifetime parameter for A/B tiles
        transfer_lr: Transfer learning rate

    Returns:
        Configured PythonLRTTRPUConfig
    """
    dt_batch_sec = lifetime_to_dt_batch_sec(lifetime)

    # Calculate lifetime for A/B tiles
    TAU_SEC = 46505.0
    if dt_batch_sec > 0:
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        ab_lifetime = 1.0 / delta
    else:
        ab_lifetime = 0.0

    # A/B tiles: 6T1C LinearStepDevice (from sweep_softbounds_lifetime.py)
    ab_device = LinearStepDevice(
        # Core update parameters (from 6T1C measurements)
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        # Device-to-device variation (sixt1c original)
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        # Cycle-to-cycle variation (sixt1c original)
        dw_min_std=0.3,
        write_noise_std=0.0,
        # LinearStepDevice specific
        mean_bound_reference=True,
        # Retention
        lifetime=ab_lifetime,
        lifetime_dtod=0.1,
        reset=0.0,
        reset_dtod=0.0,
    )

    # C tile: SoftBounds with NO NOISE
    c_device = SoftBoundsDevice(**SOFTBOUNDS_CONFIG)

    # Create PythonLRTTDevice with custom devices
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=transfer_every,
        lora_alpha=1.0,
        reinit_gain=0.1,
        reinit_mode="decay",  # decay mode
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = transfer_lr
    device_config.forward_inject = False  # y = C*x only
    device_config.update_mode = "lora"
    device_config.transfer_mode = "off"

    return PythonLRTTRPUConfig(device=device_config)


@dataclass
class DataTrainingArguments:
    """Arguments for data configuration."""

    task_name: str = field(
        default="sst2",
        metadata={"help": "GLUE task name: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(default=128)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    pad_to_max_length: bool = field(default=True)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    model_name_or_path: str = field(
        default="prajjwal1/bert-tiny",
        metadata={"help": "HuggingFace model name or path"},
    )
    cache_dir: Optional[str] = field(default=None)


@dataclass
class LRTTArguments:
    """Arguments for LRTT configuration."""

    rank: int = field(default=4, metadata={"help": "LRTT rank"})
    transfer_every: int = field(default=100, metadata={"help": "Transfer every N steps"})
    lifetime: float = field(default=46505.0, metadata={"help": "A/B tile lifetime"})
    analog_lr: float = field(default=0.01, metadata={"help": "Learning rate for AnalogSGD"})
    transfer_lr: float = field(default=0.01, metadata={"help": "Transfer learning rate"})
    use_sweep_hp: bool = field(
        default=True,
        metadata={"help": "Use hyperparameters from sweep results"},
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LRTTArguments)
    )
    model_args, data_args, training_args, lrtt_args = parser.parse_args_into_dataclasses()

    training_args.label_names = ["labels"]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    # Set seed
    set_seed(training_args.seed)

    # Get hyperparameters from sweep if available
    if lrtt_args.use_sweep_hp and lrtt_args.rank in HYPERPARAMETERS:
        te_hp = HYPERPARAMETERS[lrtt_args.rank]
        if lrtt_args.transfer_every in te_hp:
            hp = te_hp[lrtt_args.transfer_every]
            lrtt_args.analog_lr = hp["lr"]
            lrtt_args.transfer_lr = hp["tlr"]
            print(f"Using sweep hyperparameters: lr={hp['lr']}, tlr={hp['tlr']}")

    # Load dataset
    raw_datasets = load_dataset(
        "nyu-mll/glue",
        data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    # Labels
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Load model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    print("=" * 60)
    print("BERT-tiny LRTT GLUE Training (SoftBounds C + 6T1C A/B, decay)")
    print("=" * 60)
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Task: {data_args.task_name}")
    print(f"LRTT Rank: {lrtt_args.rank}")
    print(f"Transfer Every: {lrtt_args.transfer_every}")
    print(f"Lifetime: {lrtt_args.lifetime}")
    print(f"Analog LR: {lrtt_args.analog_lr}")
    print(f"Transfer LR: {lrtt_args.transfer_lr}")
    print(f"Mode: decay (reinit_mode), forward_inject=False")
    print("=" * 60)

    # Create LRTT config
    rpu_config = create_lrtt_config(
        rank=lrtt_args.rank,
        transfer_every=lrtt_args.transfer_every,
        lifetime=lrtt_args.lifetime,
        transfer_lr=lrtt_args.transfer_lr,
    )

    # Convert to analog (exclude classifier)
    print("\nConverting model to LRTT analog...")
    print("  - LRTT layers: 12 encoder Linear + pooler.dense = 13")
    print("  - Digital layers: classifier only")
    print("  - A/B tiles: 6T1C LinearStepDevice")
    print("  - C tile: SoftBoundsDevice (no noise)")

    exclude_modules = ["classifier"]
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude_modules)
    model.to(training_args.device)

    # Count LRTT layers
    lrtt_count = sum(1 for name, m in model.named_modules()
                     if hasattr(m, 'analog_module') and hasattr(m.analog_module, 'controller'))
    print(f"\nTotal LRTT layers: {lrtt_count}")

    # Create optimizer
    optimizer = AnalogSGD(model.parameters(), lr=lrtt_args.analog_lr)
    print(f"Optimizer: AnalogSGD(lr={lrtt_args.analog_lr})")

    # Preprocess data
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        return tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    raw_datasets = raw_datasets.map(preprocess_function, batched=True)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    if data_args.max_train_samples:
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if data_args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # Metrics
    metric = evaluate.load("glue", data_args.task_name)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Data collator
    data_collator = default_data_collator if data_args.pad_to_max_length else None

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
