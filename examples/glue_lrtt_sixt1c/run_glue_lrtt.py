#!/usr/bin/env python
# coding=utf-8
"""GLUE Fine-tuning with LRTT (Low-Rank Tensor-Train) Analog Training.

This script fine-tunes BERT on GLUE tasks using LRTT with 6T1C devices.
Unlike PEFT LoRA which adds trainable adapters, LRTT uses a 3-tile structure:
- A/B tiles: gradient accumulators (6T1C devices)
- C tile: weight storage (updated via transfer mechanism)

Key differences from original run_glue.py:
1. No PEFT LoRA - LRTT replaces it entirely
2. Uses AnalogSGD optimizer (required for analog training)
3. forward_inject=False: y = C*x only (A/B act as gradient buffers)
4. Reconstruction update mode for gradient computation

Usage:
    python run_glue_lrtt.py \
        --task_name sst2 \
        --model_name_or_path bert-base-uncased \
        --output_dir ./outputs/sst2 \
        --do_train --do_eval \
        --num_train_epochs 3 \
        --per_device_train_batch_size 32 \
        --learning_rate 2e-5
"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

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
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# send_example_telemetry may not be available in all versions
try:
    from transformers.utils import send_example_telemetry
except ImportError:

    def send_example_telemetry(*args, **kwargs):
        pass


# LRTT / AIHWKIT imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam

# Local imports
from lrtt_config import get_glue_preset_config
from lrtt_model_utils import print_model_stats, get_lrtt_transfer_stats

# Core LRTT imports
from aihwkit.simulator.configs.lrtt_config import (
    lrtt_sixt1c_ab_ideal_config,
    lrtt_idealized_config,
)

# Optional wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Will error if the minimal version of Transformers is not installed
check_min_version("4.40.0")
require_version("datasets>=1.8.0", "To fix: pip install datasets>=1.8.0")

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

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "Whether to pad all samples to `max_seq_length`."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples for debugging."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of evaluation examples for debugging."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of prediction examples for debugging."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name."
            )


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": "The token to use as HTTP bearer authorization for remote files."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust the execution of code from datasets/models defined on the Hub."
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )


@dataclass
class LRTTArguments:
    """Arguments for LRTT configuration."""

    lrtt_rank: int = field(
        default=8, metadata={"help": "LRTT rank (similar to LoRA rank)."}
    )
    transfer_every: int = field(
        default=1000, metadata={"help": "Transfer frequency: every N steps."}
    )
    lora_alpha: float = field(
        default=32.0, metadata={"help": "LoRA scaling factor alpha."}
    )
    forward_inject: bool = field(
        default=False,
        metadata={"help": "If False, y = C*x only (A/B are gradient buffers)."},
    )
    reinit_mode: str = field(
        default="standard",
        metadata={
            "help": "Reinit strategy: standard, decay, hybrid, orthogonal_zero, orthogonal_decay."
        },
    )
    update_mode: str = field(
        default="lora",
        metadata={"help": "A/B update mode: lora or reconstruction."},
    )
    transfer_method: str = field(
        default="set",
        metadata={"help": "Transfer method: set, direct, or onehot."},
    )
    a_init_mode: str = field(
        default="zero",
        metadata={"help": "A matrix init mode: zero or kaiming."},
    )
    use_ideal_device: bool = field(
        default=False,
        metadata={"help": "Use idealized device (no noise) for baseline comparison."},
    )
    analog_lr: float = field(
        default=0.01, metadata={"help": "Learning rate for analog optimizer."}
    )
    analog_momentum: float = field(
        default=0.9, metadata={"help": "Momentum for AnalogSGD optimizer."}
    )
    analog_optimizer: str = field(
        default="AnalogSGD", metadata={"help": "Optimizer: AnalogSGD or AnalogAdam."}
    )
    use_wandb: bool = field(
        default=False, metadata={"help": "Log to Weights & Biases."}
    )
    wandb_project: str = field(
        default="lrtt_glue", metadata={"help": "W&B project name."}
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, LRTTArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, lrtt_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, lrtt_args = (
            parser.parse_args_into_dataclasses()
        )

    training_args.label_names = ["labels"]

    # Initialize W&B if requested
    if lrtt_args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=lrtt_args.wandb_project,
            name=f"lrtt_{data_args.task_name}_r{lrtt_args.lrtt_rank}",
            config={
                "task": data_args.task_name,
                "model": model_args.model_name_or_path,
                "rank": lrtt_args.lrtt_rank,
                "transfer_every": lrtt_args.transfer_every,
                "forward_inject": lrtt_args.forward_inject,
                "reinit_mode": lrtt_args.reinit_mode,
                "update_mode": lrtt_args.update_mode,
            },
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Set seed
    set_seed(training_args.seed)

    # Load dataset
    if data_args.task_name is not None:
        raw_datasets = load_dataset(
            "nyu-mll/glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    elif data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        data_files = {
            "train": data_args.train_file,
            "validation": data_args.validation_file,
        }
        if training_args.do_predict:
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
            else:
                raise ValueError(
                    "Need either a GLUE task or a test file for `do_predict`."
                )

        if data_args.train_file.endswith(".csv"):
            raw_datasets = load_dataset(
                "csv", data_files=data_files, cache_dir=model_args.cache_dir
            )
        else:
            raw_datasets = load_dataset(
                "json", data_files=data_files, cache_dir=model_args.cache_dir
            )

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in [
            "float32",
            "float64",
        ]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    print("=" * 60)
    print("LRTT GLUE Training")
    print("=" * 60)
    print(f"Task: {data_args.task_name}")
    print(f"Model: {model_args.model_name_or_path}")
    print(f"LRTT Rank: {lrtt_args.lrtt_rank}")
    print(f"Transfer Every: {lrtt_args.transfer_every}")
    print(f"Forward Inject: {lrtt_args.forward_inject}")
    print(f"Reinit Mode: {lrtt_args.reinit_mode}")
    print(f"Update Mode: {lrtt_args.update_mode}")
    print("=" * 60)

    # Create LRTT configuration using core functions
    if lrtt_args.use_ideal_device:
        rpu_config = lrtt_idealized_config(
            rank=lrtt_args.lrtt_rank,
            transfer_every=lrtt_args.transfer_every,
            lora_alpha=lrtt_args.lora_alpha,
        )
    else:
        rpu_config = lrtt_sixt1c_ab_ideal_config(
            rank=lrtt_args.lrtt_rank,
            transfer_every=lrtt_args.transfer_every,
            lora_alpha=lrtt_args.lora_alpha,
        )

    # Apply LRTT-specific settings
    rpu_config.device.forward_inject = lrtt_args.forward_inject
    rpu_config.device.reinit_mode = lrtt_args.reinit_mode
    rpu_config.device.update_mode = lrtt_args.update_mode
    rpu_config.device.transfer_method = lrtt_args.transfer_method
    rpu_config.device.a_init_mode = lrtt_args.a_init_mode

    print("\nRPU Configuration:")
    print(rpu_config.get_brief_info())

    # Convert model to LRTT analog
    # Match lora_on_analog_hardware structure exactly:
    # - 73 PCM layers → 73 LRTT layers (72 encoder + bert.pooler.dense)
    # - classifier → remains Digital (not in LoRA target_modules)
    print("\nConverting model to LRTT analog...")
    print(
        "  - LRTT layers: 72 encoder (query, key, value, dense) + bert.pooler.dense = 73"
    )
    print("  - Digital layers: classifier only")

    # Only exclude classifier - this matches lora_on_analog_hardware where:
    # - All layers matching "dense", "query", "key", "value" have LoRA → base_layer becomes PCM
    # - classifier doesn't match target_modules → stays Digital
    exclude_modules = ["classifier"]
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude_modules)

    # Move to device
    model.to(training_args.device)

    # Calculate total steps for transfer statistics
    if training_args.do_train:
        num_train_samples = len(raw_datasets["train"])
        if data_args.max_train_samples is not None:
            num_train_samples = min(num_train_samples, data_args.max_train_samples)
        total_steps = (
            num_train_samples
            // training_args.per_device_train_batch_size
            * int(training_args.num_train_epochs)
        )
    else:
        total_steps = None

    # Print model statistics with LRTT parameter breakdown
    print_model_stats(
        model,
        rank=lrtt_args.lrtt_rank,
        transfer_every=lrtt_args.transfer_every,
        total_steps=total_steps,
        verbose=False,  # Set to True for per-layer details
    )

    # Create analog optimizer
    if lrtt_args.analog_optimizer == "AnalogAdam":
        optimizer = AnalogAdam(
            model.parameters(),
            lr=lrtt_args.analog_lr,
        )
        print(f"\nOptimizer: AnalogAdam(lr={lrtt_args.analog_lr})")
    else:
        optimizer = AnalogSGD(
            model.parameters(),
            lr=lrtt_args.analog_lr,
            momentum=lrtt_args.analog_momentum,
        )
        print(
            f"\nOptimizer: AnalogSGD(lr={lrtt_args.analog_lr}, momentum={lrtt_args.analog_momentum})"
        )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"
        ]
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    # Label mapping
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset."
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        if label_to_id is not None and "label" in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if (
            "validation" not in raw_datasets
            and "validation_matched" not in raw_datasets
        ):
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets[
            "validation_matched" if data_args.task_name == "mnli" else "validation"
        ]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if (
        training_args.do_predict
        or data_args.task_name is not None
        or data_args.test_file is not None
    ):
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets[
            "test_matched" if data_args.task_name == "mnli" else "test"
        ]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set
    if training_args.do_train:
        for index in random.sample(
            range(len(train_dataset)), min(3, len(train_dataset))
        ):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load(
            "glue", data_args.task_name, cache_dir=model_args.cache_dir
        )
    elif is_regression:
        metric = evaluate.load("mse", cache_dir=model_args.cache_dir)
    else:
        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        optimizers=(optimizer, None),  # Pass the AnalogSGD optimizer
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # Save model checkpoint
        output_model_file = os.path.join(training_args.output_dir, "lrtt_model.pt")
        torch.save(model.state_dict(), output_model_file)
        print(f"Model saved to {output_model_file}")

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Log LRTT transfer statistics
        print("\nLRTT Transfer Statistics after training:")
        transfer_stats = get_lrtt_transfer_stats(model)
        total_transfers = sum(s["transfer_count"] for s in transfer_stats.values())
        print(f"Total transfers across all layers: {total_transfers}")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(valid_mm_dataset), data_args.max_eval_samples
                )
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics(
                "eval", combined if task is not None and "mnli" in task else metrics
            )

            # Log to wandb
            if lrtt_args.use_wandb and WANDB_AVAILABLE:
                wandb.log(metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(
                predict_dataset, metric_key_prefix="predict"
            ).predictions
            predictions = (
                np.squeeze(predictions)
                if is_regression
                else np.argmax(predictions, axis=1)
            )

            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    # Finish wandb
    if lrtt_args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    print("\n" + "=" * 60)
    print("LRTT GLUE Training Complete!")
    print("=" * 60)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
