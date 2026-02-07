#!/usr/bin/env python
# coding=utf-8
"""LRTT No-Transfer Test: forward_inject=True + transfer_every=매우 큼

기존 LRTT sweep과 동일한 device 설정에서:
- forward_inject=True (LoRA와 동일한 forward: y = C·x + α·A·(B·x))
- transfer_every=1000000000 (실질적으로 no transfer)

이 설정이 sixt1c + LoRA와 동등한지 확인하고, OOM 발생 여부 테스트.
"""

import os
import sys
import math
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

# LRTT config imports
sys.path.insert(0, '/data/LRTT_transformer/src')
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice


# =============================================================================
# Configuration (동일한 device 설정)
# =============================================================================

MODEL_NAME = "google/mobilebert-uncased"
TARGET_MODULES = ["query", "key", "value"]
RANK = 8
LORA_ALPHA = 1.0
REINIT_GAIN = 0.1
BATCH_SIZE = 8  # sixt1c와 동일한 batch size
MAX_TRAIN_SAMPLES = 500  # 빠른 테스트
NUM_EPOCHS = 1
SEED = 42
LEARNING_RATE = 0.00362  # 기존 sweep best

# 테스트: forward_inject=False로 변경 (기존 sweep과 동일)
FORWARD_INJECT = False
TRANSFER_EVERY = 1000000000  # 실질적으로 no transfer


def create_lrtt_config_no_transfer():
    """기존 LRTT sweep과 동일한 device 설정, forward_inject=True, no transfer."""

    # Calculate lifetime for 6T1C (동일)
    TAU_SEC = 46505.0
    dt_batch_sec = 1.0
    delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
    lifetime = 1.0 / delta if delta > 0 else 0.0

    # A/B tiles: LinearStepDevice (6T1C) - 기존과 동일
    ab_device = LinearStepDevice(
        dw_min=0.001981,
        up_down=0.0,
        w_max=1.0,
        w_min=-1.0,
        gamma_up=-0.1678,
        gamma_down=0.1410,
        mult_noise=True,
        dw_min_dtod=0.1,
        up_down_dtod=0.01,
        w_max_dtod=0.05,
        w_min_dtod=0.05,
        gamma_up_dtod=0.05,
        gamma_down_dtod=0.05,
        dw_min_std=0.3,
        write_noise_std=0.0,
        mean_bound_reference=True,
        lifetime=lifetime,
        lifetime_dtod=0.0,  # dtod0 version
        reset=0.0,
        reset_dtod=0.0,
    )

    # C tile: SoftBoundsDevice (noise-free) - 기존과 동일
    c_device = SoftBoundsDevice(
        dw_min=0.001,
        w_max=3.0,
        w_min=-3.0,
        dw_min_dtod=0.0,
        dw_min_std=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        w_max_dtod=0.0,
        w_min_dtod=0.0,
        write_noise_std=0.0,
        mult_noise=True,
    )

    # LRTT Device config - 핵심 변경!
    device_config = PythonLRTTDevice(
        rank=RANK,
        transfer_every=TRANSFER_EVERY,  # 매우 큼 (no transfer)
        lora_alpha=LORA_ALPHA,
        reinit_gain=REINIT_GAIN,
        reinit_mode="hybrid",
        decay_factor=1.0,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = 0.001  # 사용 안됨 (no transfer)
    device_config.units_in_mbatch = True
    device_config.forward_inject = FORWARD_INJECT  # True로 변경!
    device_config.transfer_method = "onehot"
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    # Weight scaling (동일)
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


def list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def create_model(device):
    """Create SQuAD model with LRTT (no transfer)."""
    print("\n" + "=" * 60)
    print("LRTT NO-TRANSFER TEST")
    print("forward_inject=True, transfer_every=1000000000 (no transfer)")
    print("=" * 60 + "\n")

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    all_linear = list_linear_layers(model)
    exclude = [name for name in all_linear if not any(t in name for t in TARGET_MODULES)]
    exclude.append("qa_outputs")

    print(f"Target modules: {TARGET_MODULES}")
    print(f"Excluded layers: {len(exclude)}")

    rpu_config = create_lrtt_config_no_transfer()
    print("\nLRTT Config:")
    print(f"  forward_inject: {FORWARD_INJECT}")
    print(f"  transfer_every: {TRANSFER_EVERY}")
    print(f"  rank: {RANK}")
    print(f"  lora_alpha: {LORA_ALPHA}")

    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    # Freeze non-target layers
    for name, param in model.named_parameters():
        is_target = any(t in name for t in TARGET_MODULES)
        param.requires_grad = is_target or "qa_outputs" in name

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable:,} ({100*trainable/total:.2f}%)")

    return model.to(device)


def load_data(tokenizer):
    """Load SQuAD train subset."""
    raw = load_dataset("squad")

    def preprocess(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=384, truncation="only_second",
            stride=128, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]

        start_positions, end_positions = [], []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]

            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while idx < len(sequence_ids) and sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

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

    tokenized = raw["train"].map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    subset = tokenized.shuffle(seed=SEED).select(range(min(MAX_TRAIN_SAMPLES, len(tokenized))))

    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator)


def train(model, optimizer, scheduler, train_loader, device):
    """Train for one epoch, monitoring memory."""
    model.train()
    total_loss = 0.0

    # Memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial GPU memory: {initial_mem:.2f} GB")

    pbar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Memory check every 10 steps
        if step % 10 == 0 and torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated() / 1024**3
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            pbar.set_postfix(loss=f"{loss.item():.4f}", mem=f"{current_mem:.1f}GB", peak=f"{peak_mem:.1f}GB")

    if torch.cuda.is_available():
        final_mem = torch.cuda.memory_allocated() / 1024**3
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nFinal GPU memory: {final_mem:.2f} GB")
        print(f"Peak GPU memory: {peak_mem:.2f} GB")

    return total_loss / len(train_loader)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = load_data(tokenizer)
    print(f"\nTrain batches: {len(train_loader)} (batch_size={BATCH_SIZE})")

    # Create model
    model = create_model(device)

    # Optimizer
    optimizer = AnalogAdam(model.parameters(), lr=LEARNING_RATE)
    optimizer.regroup_param_groups()

    # Scheduler
    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
    )

    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")

    try:
        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
            avg_loss = train(model, optimizer, scheduler, train_loader, device)
            print(f"Average loss: {avg_loss:.4f}")

        print("\n" + "=" * 60)
        print("SUCCESS: Training completed without OOM!")
        print("=" * 60)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "=" * 60)
            print("FAILED: OOM occurred!")
            print("=" * 60)
            print(f"Error: {e}")
        else:
            raise


if __name__ == "__main__":
    main()
