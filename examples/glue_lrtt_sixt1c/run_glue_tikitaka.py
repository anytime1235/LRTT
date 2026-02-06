#!/usr/bin/env python
# coding=utf-8
"""GLUE Fine-tuning with TikiTaka (2-tile) Analog Training.

This script fine-tunes BERT on GLUE tasks using TikiTaka with:
- Fast tile (6T1C): receives SGD updates
- Slow tile (SoftBounds): visible weight storage

Key differences from LRTT (3-tile):
1. No low-rank factorization - full matrix on both tiles
2. Standard TikiTaka transfer: one-hot column read/write
3. Forward: y = gamma*Fast@x + (1-gamma)*Slow@x (gamma=0 → only Slow visible)

Usage:
    python run_glue_tikitaka.py \
        --task_name sst2 \
        --model_name_or_path bert-base-uncased \
        --output_dir ./outputs/tikitaka_sst2 \
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

try:
    from transformers.utils import send_example_telemetry
except ImportError:
    def send_example_telemetry(*args, **kwargs):
        pass

# AIHWKIT imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam

# TikiTaka config
from tikitaka_config import (
    tikitaka_sixt1c_softbounds_config,
    tikitaka_idealized_config,
)

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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
    """Arguments for data/task configuration."""

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": f"Task name: {', '.join(task_to_keys.keys())}"},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset name (via datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset configuration name."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "Max sequence length after tokenization."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite cached preprocessed datasets."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={"help": "Pad all samples to max_seq_length."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate training examples for debugging."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate evaluation examples for debugging."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate prediction examples for debugging."},
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Training data file (csv/json)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Validation data file (csv/json)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "Test data file (csv/json)."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(f"Unknown task: {self.task_name}")
        elif self.dataset_name is None and (self.train_file is None or self.validation_file is None):
            raise ValueError("Need task, dataset, or train/validation files.")


@dataclass
class ModelArguments:
    """Arguments for model/tokenizer configuration."""

    model_name_or_path: str = field(
        metadata={"help": "Pretrained model path or HuggingFace model ID."}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Config name/path if different from model."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer name/path if different from model."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for pretrained models."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Use fast tokenizer."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Model version (branch/tag/commit)."},
    )
    token: str = field(
        default=None,
        metadata={"help": "HuggingFace token for private models."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code execution."},
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Allow loading with different head dimensions."},
    )


@dataclass
class TikiTakaArguments:
    """Arguments for TikiTaka configuration."""

    # Transfer settings
    transfer_every: int = field(
        default=32,
        metadata={"help": "Transfer frequency: every N mini-batches."},
    )
    transfer_lr: float = field(
        default=1.0,
        metadata={"help": "Transfer learning rate (relative to SGD LR)."},
    )
    fast_lr: float = field(
        default=1.0,
        metadata={"help": "Fast tile learning rate multiplier."},
    )
    gamma: float = field(
        default=0.0,
        metadata={"help": "Weight mixing (0=only Slow visible, 1=only Fast)."},
    )
    n_reads_per_transfer: int = field(
        default=1,
        metadata={"help": "Columns transferred per transfer event."},
    )

    # 6T1C Fast tile settings
    sixt1c_dw_min: float = field(
        default=0.001981,
        metadata={"help": "6T1C minimum weight step."},
    )
    sixt1c_gamma_up: float = field(
        default=-0.1678,
        metadata={"help": "6T1C up-pulse nonlinearity."},
    )
    sixt1c_gamma_down: float = field(
        default=0.1410,
        metadata={"help": "6T1C down-pulse nonlinearity."},
    )
    sixt1c_dw_min_std: float = field(
        default=0.3,
        metadata={"help": "6T1C cycle-to-cycle noise."},
    )
    sixt1c_lifetime: float = field(
        default=0.0,
        metadata={"help": "6T1C retention lifetime (0=no decay)."},
    )

    # Slow tile settings
    slow_noise_free: bool = field(
        default=True,
        metadata={"help": "Use noise-free Slow tile."},
    )

    # Idealized mode
    use_ideal: bool = field(
        default=False,
        metadata={"help": "Use idealized (noise-free) TikiTaka."},
    )

    # Optimizer settings
    analog_lr: float = field(
        default=0.01,
        metadata={"help": "Analog optimizer learning rate."},
    )
    analog_momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum for AnalogSGD."},
    )
    analog_optimizer: str = field(
        default="AnalogSGD",
        metadata={"help": "Optimizer: AnalogSGD or AnalogAdam."},
    )

    # Logging
    use_wandb: bool = field(
        default=False,
        metadata={"help": "Log to Weights & Biases."},
    )
    wandb_project: str = field(
        default="tikitaka_glue",
        metadata={"help": "W&B project name."},
    )


def print_tikitaka_model_stats(
    model,
    transfer_every: int = 32,
    total_steps: Optional[int] = None,
):
    """Print model statistics for TikiTaka."""
    total_params = 0
    analog_layers = 0

    for name, module in model.named_modules():
        if hasattr(module, "analog_tile"):
            analog_layers += 1
            tile = module.analog_tile
            if hasattr(tile, "tile"):
                # Get weight shape
                w = tile.get_weights()[0]
                total_params += w.numel() * 2  # Fast + Slow tiles

    print("\n" + "=" * 60)
    print("TikiTaka Model Statistics")
    print("=" * 60)
    print(f"Analog Layers: {analog_layers}")
    print(f"Transfer Every: {transfer_every} mini-batches")
    print(f"Total Analog Parameters: {total_params:,} (2 tiles per layer)")

    if total_steps is not None:
        expected_transfers = total_steps // transfer_every
        print(f"Expected Transfers: ~{expected_transfers} per layer")
    print("=" * 60 + "\n")


def get_tikitaka_transfer_stats(model):
    """Get transfer statistics from TikiTaka model."""
    stats = {}
    for name, module in model.named_modules():
        if hasattr(module, "analog_tile"):
            tile = module.analog_tile
            if hasattr(tile, "tile"):
                # TransferCompound tracks transfer_counter internally
                stats[name] = {
                    "layer_type": type(module).__name__,
                }
    return stats


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, TikiTakaArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, tt_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, tt_args = (
            parser.parse_args_into_dataclasses()
        )

    training_args.label_names = ["labels"]

    # Initialize W&B
    if tt_args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=tt_args.wandb_project,
            name=f"tikitaka_{data_args.task_name}_te{tt_args.transfer_every}",
            config={
                "task": data_args.task_name,
                "model": model_args.model_name_or_path,
                "transfer_every": tt_args.transfer_every,
                "fast_lr": tt_args.fast_lr,
                "gamma": tt_args.gamma,
                "use_ideal": tt_args.use_ideal,
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

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {training_args.parallel_mode.value == 'distributed'}, "
        f"fp16: {training_args.fp16}"
    )
    logger.info(f"Training parameters: {training_args}")

    # Checkpoint detection
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

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
        if training_args.do_predict and data_args.test_file:
            data_files["test"] = data_args.test_file

        ext = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(
            ext, data_files=data_files, cache_dir=model_args.cache_dir
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
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    # Load model
    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
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
    print("TikiTaka GLUE Training")
    print("=" * 60)
    print(f"Task: {data_args.task_name}")
    print(f"Model: {model_args.model_name_or_path}")
    print(f"Transfer Every: {tt_args.transfer_every}")
    print(f"Fast LR: {tt_args.fast_lr}")
    print(f"Gamma: {tt_args.gamma}")
    print(f"Use Ideal: {tt_args.use_ideal}")
    print("=" * 60)

    # Create TikiTaka configuration
    if tt_args.use_ideal:
        rpu_config = tikitaka_idealized_config(
            transfer_every=tt_args.transfer_every,
            transfer_lr=tt_args.transfer_lr,
            fast_lr=tt_args.fast_lr,
            gamma=tt_args.gamma,
        )
    else:
        rpu_config = tikitaka_sixt1c_softbounds_config(
            transfer_every=tt_args.transfer_every,
            transfer_lr=tt_args.transfer_lr,
            fast_lr=tt_args.fast_lr,
            gamma=tt_args.gamma,
            n_reads_per_transfer=tt_args.n_reads_per_transfer,
            sixt1c_dw_min=tt_args.sixt1c_dw_min,
            sixt1c_gamma_up=tt_args.sixt1c_gamma_up,
            sixt1c_gamma_down=tt_args.sixt1c_gamma_down,
            sixt1c_dw_min_std=tt_args.sixt1c_dw_min_std,
            sixt1c_lifetime=tt_args.sixt1c_lifetime,
            slow_noise_free=tt_args.slow_noise_free,
        )

    print("\nRPU Configuration:")
    print(rpu_config)

    # Convert to analog (exclude classifier)
    print("\nConverting model to TikiTaka analog...")
    exclude_modules = ["classifier"]
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude_modules)
    model.to(training_args.device)

    # Calculate total steps
    if training_args.do_train:
        num_train_samples = len(raw_datasets["train"])
        if data_args.max_train_samples:
            num_train_samples = min(num_train_samples, data_args.max_train_samples)
        total_steps = (
            num_train_samples
            // training_args.per_device_train_batch_size
            * int(training_args.num_train_epochs)
        )
    else:
        total_steps = None

    print_tikitaka_model_stats(
        model,
        transfer_every=tt_args.transfer_every,
        total_steps=total_steps,
    )

    # Create optimizer
    if tt_args.analog_optimizer == "AnalogAdam":
        optimizer = AnalogAdam(model.parameters(), lr=tt_args.analog_lr)
        print(f"\nOptimizer: AnalogAdam(lr={tt_args.analog_lr})")
    else:
        optimizer = AnalogSGD(
            model.parameters(),
            lr=tt_args.analog_lr,
            momentum=tt_args.analog_momentum,
        )
        print(f"\nOptimizer: AnalogSGD(lr={tt_args.analog_lr}, momentum={tt_args.analog_momentum})")

    # Preprocessing
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        non_label_cols = [n for n in raw_datasets["train"].column_names if n != "label"]
        sentence1_key = non_label_cols[0]
        sentence2_key = non_label_cols[1] if len(non_label_cols) >= 2 else None

    padding = "max_length" if data_args.pad_to_max_length else False

    # Label mapping
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in label_to_id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {i: l for i, l in enumerate(label_list)}

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset preprocessing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Tokenizing",
        )

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    if training_args.do_eval:
        eval_key = "validation_matched" if data_args.task_name == "mnli" else "validation"
        eval_dataset = raw_datasets[eval_key]
        if data_args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    if training_args.do_predict:
        test_key = "test_matched" if data_args.task_name == "mnli" else "test"
        predict_dataset = raw_datasets[test_key]
        if data_args.max_predict_samples:
            predict_dataset = predict_dataset.select(range(min(len(predict_dataset), data_args.max_predict_samples)))

    # Metrics
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name, cache_dir=model_args.cache_dir)
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = min(
            data_args.max_train_samples or len(train_dataset), len(train_dataset)
        )

        # Save model
        output_model_file = os.path.join(training_args.output_dir, "tikitaka_model.pt")
        torch.save(model.state_dict(), output_model_file)
        print(f"Model saved to {output_model_file}")

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        print("\nTikiTaka Training Complete!")
        print(f"Total training steps: {trainer.state.global_step}")
        print(f"Expected transfers per layer: ~{trainer.state.global_step // tt_args.transfer_every}")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])

        for eval_ds, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_ds)
            metrics["eval_samples"] = len(eval_ds)

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            if tt_args.use_wandb and WANDB_AVAILABLE:
                wandb.log(metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for pred_ds, task in zip(predict_datasets, tasks):
            pred_ds = pred_ds.remove_columns("label")
            predictions = trainer.predict(pred_ds, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_file, "w") as f:
                    f.write("index\tprediction\n")
                    for idx, pred in enumerate(predictions):
                        if is_regression:
                            f.write(f"{idx}\t{pred:.3f}\n")
                        else:
                            f.write(f"{idx}\t{label_list[pred]}\n")

    if tt_args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    print("\n" + "=" * 60)
    print("TikiTaka GLUE Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
