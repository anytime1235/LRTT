#!/usr/bin/env python
# coding=utf-8
"""MobileBERT LRTT Test - Excluding attention layers.

Tests if training works when attention layers are excluded from LRTT conversion.
"""

import gc
import math
import os

import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

SEED = 42
MODEL_NAME = "google/mobilebert-uncased"
TASK_NAME = "sst2"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 1
LOGGING_STEPS = 50
OUTPUT_DIR = "/tmp/mobilebert_exclude_attn"


def lifetime_to_dt_batch_sec(lifetime: float) -> float:
    TAU_SEC = 46505.0
    delta = 1.0 / lifetime
    return -TAU_SEC * math.log(1 - delta)


def create_lrtt_config(rank: int, te: int, tlr: float, lifetime: float) -> PythonLRTTRPUConfig:
    dt_batch_sec = lifetime_to_dt_batch_sec(lifetime)
    TAU_SEC = 46505.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    ab_lifetime = 1.0 / delta if delta > 0 else 0.0

    SOFTBOUNDS_CONFIG = {
        'dw_min': 0.001, 'w_max': 1.0, 'w_min': -1.0,
        'dw_min_dtod': 0.0, 'dw_min_std': 0.0, 'up_down': 0.0,
        'up_down_dtod': 0.0, 'w_max_dtod': 0.0, 'w_min_dtod': 0.0,
        'write_noise_std': 0.0, 'mult_noise': True,
    }

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"TEST: MobileBERT LRTT (EXCLUDING ATTENTION)")
    print(f"{'='*60}")
    print(f"Device: {device}")

    set_seed(SEED)

    # Load data
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
    metric = evaluate.load("glue", TASK_NAME)
    print(f"Data loaded: {len(train_dataset)} train, {len(eval_dataset)} eval")

    # Load model
    print("\nLoading MobileBERT...")
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config, use_safetensors=True
    )

    # LRTT config with successful BERT-base parameters
    rank = 4
    te = 1000
    tlr = 0.001
    lifetime = 100000
    lr = 5e-5

    print(f"\nLRTT Config: rank={rank}, te={te}, tlr={tlr}, lifetime={lifetime}, lr={lr}")

    # Convert to LRTT - EXCLUDING attention layers (which have large weights)
    print("\nConverting to LRTT (excluding attention and classifier)...")
    rpu_config = create_lrtt_config(rank, te, tlr, lifetime)

    # Exclude both classifier and attention modules
    model = convert_to_analog(
        model,
        rpu_config,
        exclude_modules=["classifier", "attention"]  # Exclude attention!
    )
    model.to(device)

    # Quick forward pass test
    print("\nTesting forward pass...")
    dummy_input = {
        "input_ids": torch.randint(0, 1000, (1, 128)).to(device),
        "attention_mask": torch.ones(1, 128).to(device),
    }
    with torch.no_grad():
        output = model(**dummy_input)
        logits = output.logits
        print(f"Logits: {logits.cpu().numpy()}")
        print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

    # Create optimizer
    optimizer = AnalogSGD(model.parameters(), lr=lr)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        seed=SEED,
        remove_unused_columns=True,
        disable_tqdm=False,
        max_grad_norm=1.0,
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
    )

    print("\n" + "="*60)
    print("STARTING TRAINING (attention excluded from LRTT)")
    print("="*60 + "\n")

    train_result = trainer.train()

    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    eval_result = trainer.evaluate()

    print(f"\nFinal Results:")
    print(f"  Train Loss: {train_result.training_loss:.4f}")
    print(f"  Eval Loss: {eval_result.get('eval_loss', 'N/A')}")
    print(f"  Eval Accuracy: {eval_result.get('eval_accuracy', 'N/A')}")

    eval_acc = eval_result.get('eval_accuracy', 0)
    if eval_acc > 0.55:
        print(f"\n[SUCCESS] MobileBERT LRTT (excl. attention) is learning! Accuracy {eval_acc:.4f} > 0.55")
    else:
        print(f"\n[FAILED] MobileBERT LRTT (excl. attention) not learning. Accuracy {eval_acc:.4f}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
