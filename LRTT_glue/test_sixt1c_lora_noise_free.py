#!/usr/bin/env python
# coding=utf-8
"""Sixt1c-LoRA NOISE-FREE test on SQuAD: Compare alpha=0.01 vs 0.0

Tests 2 lora_alpha values (0.01, 0.0) with lr=0.001
All noise parameters set to 0, lifetime=0 (no decay)
Comparison to determine optimal alpha for SQuAD task
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, default_data_collator, set_seed, get_cosine_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
import collections
import re
import string
from typing import Dict, List
from collections import Counter
from tqdm import tqdm

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

# LRTT config imports
sys.path.insert(0, '/home/jovyan/work/LRTT/src')
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.configs.utils import IOParameters, BoundManagementType
from aihwkit.simulator.parameters.enums import NoiseManagementType

# Fixed parameters
RANK = 8
TRANSFER_EVERY = 1000000  # No transfer
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 384  # SQuAD standard (same as tikitaka sweep)
DOC_STRIDE = 128
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 128
NUM_EPOCHS = 15
SEED = 42

# Test configurations: Compare alpha=0.01 vs 0.0 to find optimal for SQuAD
CONFIGS = [
    {"lora_alpha": 0.0, "lr": 0.001, "name": "alpha0_baseline"},
    {"lora_alpha": 0.01, "lr": 0.001, "name": "alpha0.01"},
]


def create_sixt1c_lora_config_noise_free(rank: int, lora_alpha: float):
    """Create Sixt1c-LoRA config with ALL NOISE = 0."""

    # A/B tiles: LinearStepDevice (6T1C) - NOISE-FREE
    ab_device = LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        # ALL NOISE = 0
        dw_min_dtod=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        gamma_up_dtod=0.0,
        gamma_down_dtod=0.0,
        dw_min_std=0.0,
        write_noise_std=0.0,
        mean_bound_reference=True,
        # NO LIFETIME/DECAY
        lifetime=0.0,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )

    # C tile: SoftBoundsDevice (noise-free)
    c_device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=1.0,
        w_min=-1.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )

    # Sixt1c-LoRA Device config (3-tile: A, B, C)
    device_config = PythonLRTTDevice(
        rank=rank,
        transfer_every=TRANSFER_EVERY,
        lora_alpha=lora_alpha,
        reinit_gain=0.1,
        reinit_mode="hybrid",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.001
    device_config.units_in_mbatch = True
    device_config.forward_inject = True  # Key: sixt1c_lora mode
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    # Configure weight scaling
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    # Set forward IO parameters - CRITICAL: output_noise = 0 for noise-free
    forward_io = IOParameters(
        out_noise=0.0,  # No output/read noise
        noise_management=NoiseManagementType.NONE,
        bound_management=BoundManagementType.NONE,
    )
    rpu_config.forward = forward_io

    return rpu_config


def list_linear_layers(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def create_squad_model(params: Dict, device: torch.device) -> nn.Module:
    """Create SQuAD model with Sixt1c-LoRA (QKV only)."""
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    all_linear = list_linear_layers(model)
    target_modules = ["query", "key", "value"]

    exclude = [name for name in all_linear if not any(t in name for t in target_modules)]
    exclude.append("qa_outputs")

    rpu_config = create_sixt1c_lora_config_noise_free(
        rank=RANK,
        lora_alpha=params["lora_alpha"]
    )

    print(f"Converting to analog (QKV only, noise-free)...")
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    # Freeze bias, train QKV analog tiles and qa_outputs
    for name, param in model.named_parameters():
        is_target = any(t in name for t in target_modules) and "bias" not in name
        param.requires_grad = is_target or "qa_outputs" in name

    return model.to(device)


def load_squad_data(tokenizer):
    """Load and prepare SQuAD dataset."""
    dataset = load_dataset("squad")

    def prepare_features(examples):
        tokenized = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=MAX_SEQ_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        tokenized["start_positions"] = []
        tokenized["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                tokenized["start_positions"].append(cls_index)
                tokenized["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                if not (offsets[token_start_index][0] <= start_char and
                        offsets[token_end_index][1] >= end_char):
                    tokenized["start_positions"].append(cls_index)
                    tokenized["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized["start_positions"].append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized["end_positions"].append(token_end_index + 1)

        return tokenized

    train_dataset = dataset["train"].map(prepare_features, batched=True, remove_columns=dataset["train"].column_names)
    train_dataset.set_format(type="torch")

    val_dataset = dataset["validation"].map(prepare_features, batched=True, remove_columns=dataset["validation"].column_names)
    val_dataset.set_format(type="torch")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=BATCH_SIZE,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=EVAL_BATCH_SIZE,
    )

    return train_loader, val_loader


def evaluate_model(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_em = 0
    total_count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            total_loss += outputs.loss.item()

            # Compute EM (Exact Match) accuracy
            start_preds = torch.argmax(outputs.start_logits, dim=1)
            end_preds = torch.argmax(outputs.end_logits, dim=1)

            start_correct = (start_preds == batch["start_positions"]).sum().item()
            end_correct = (end_preds == batch["end_positions"]).sum().item()

            # EM: both start and end must be correct
            em = ((start_preds == batch["start_positions"]) &
                  (end_preds == batch["end_positions"])).sum().item()

            total_em += em
            total_count += batch["start_positions"].size(0)

    avg_loss = total_loss / len(val_loader)
    em_accuracy = 100.0 * total_em / total_count

    model.train()
    return avg_loss, em_accuracy


def train_one_config(config: Dict, device: torch.device):
    """Train one configuration."""
    print(f"\n{'='*80}")
    print(f"Training: {config['name']}")
    print(f"  lora_alpha = {config['lora_alpha']}")
    print(f"  lr = {config['lr']}")
    print(f"  optimizer = AnalogAdam")
    print(f"  NOISE-FREE: All variations=0, lifetime=0, out_noise=0")
    print(f"{'='*80}\n")

    set_seed(SEED)

    # Create model
    model = create_squad_model(config, device)

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader = load_squad_data(tokenizer)

    # Optimizer - Using Adam (original sweep configuration)
    optimizer = AnalogAdam(model.parameters(), lr=config["lr"])
    optimizer.regroup_param_groups()

    # Training loop
    model.train()
    total_steps = len(train_loader) * NUM_EPOCHS

    best_em = 0.0
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        val_loss, val_em = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val EM: {val_em:.2f}%")

        if val_em > best_em:
            best_em = val_em
            print(f"  → New best EM: {best_em:.2f}%")

    print(f"\n{'='*80}")
    print(f"Training completed: {config['name']}")
    print(f"Best Validation EM: {best_em:.2f}%")
    print(f"{'='*80}\n")

    return best_em


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"\nNOISE-FREE Sixt1c-LoRA on SQuAD (QKV only)")
    print(f"Configurations: {len(CONFIGS)}")
    for cfg in CONFIGS:
        print(f"  - {cfg['name']}: alpha={cfg['lora_alpha']}, lr={cfg['lr']}")

    results = {}
    for config in CONFIGS:
        best_em = train_one_config(config, device)
        results[config['name']] = {'alpha': config['lora_alpha'], 'best_em': best_em}
        torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    for name, result in results.items():
        print(f"{name:20s} (alpha={result['alpha']:5.2f}): Best EM = {result['best_em']:6.2f}%")
    print("="*80)

    # Determine winner
    best_config = max(results.items(), key=lambda x: x[1]['best_em'])
    print(f"\n✓ Best configuration: {best_config[0]} with EM = {best_config[1]['best_em']:.2f}%")
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
