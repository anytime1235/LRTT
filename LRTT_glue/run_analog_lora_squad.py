#!/usr/bin/env python
"""Run SQuAD training with analog LoRA (only lora_A/lora_B as sixt1c tiles).

Base layer stays digital, only LoRA adapters are converted to analog.
"""
import torch
import gc
import sys
import os

# aihwkit imports
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogAdam
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import LinearStepDevice

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    set_seed,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# Add lora_training to path
sys.path.insert(0, '/data/LRTT_transformer/lora_training')
from related_functions import convert_lora_layers_only_to_analog

# Config
MODEL_NAME = "google/mobilebert-uncased"
BATCH_SIZE = 8
MAX_SEQ_LEN = 384
EPOCHS = 1
LR = 1e-4
WARMUP_STEPS = 100
LORA_RANK = 8
LORA_ALPHA = 32
SEED = 42

def get_mem():
    return torch.cuda.memory_allocated() / 1024**3

def prepare_squad_data(tokenizer, max_samples=None):
    """Load and prepare SQuAD dataset."""
    dataset = load_dataset("squad", split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def preprocess(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]

        tokenized = tokenizer(
            questions,
            contexts,
            max_length=MAX_SEQ_LEN,
            truncation="only_second",
            padding="max_length",
            return_offsets_mapping=True,
        )

        start_positions = []
        end_positions = []

        for i, offset in enumerate(tokenized["offset_mapping"]):
            answer = examples["answers"][i]
            if len(answer["answer_start"]) == 0:
                start_positions.append(0)
                end_positions.append(0)
                continue

            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            # Find token positions
            sequence_ids = tokenized.sequence_ids(i)
            context_start = 0
            while sequence_ids[context_start] != 1:
                context_start += 1
            context_end = len(sequence_ids) - 1
            while sequence_ids[context_end] != 1:
                context_end -= 1

            # Find start/end tokens
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

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        tokenized.pop("offset_mapping")
        return tokenized

    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")
    return dataset

def main():
    print("=" * 70)
    print("Analog LoRA SQuAD Training (lora_A/lora_B only as sixt1c)")
    print("=" * 70)

    set_seed(SEED)
    device = torch.device("cuda")

    gc.collect()
    torch.cuda.empty_cache()
    print(f"Initial GPU memory: {get_mem():.3f} GB")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load model
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    print(f"After loading model: {get_mem():.3f} GB")

    # Apply PEFT LoRA (digital)
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],  # QKV
        bias="none",
        task_type="QUESTION_ANS"
    )
    model = get_peft_model(model, peft_config)
    print(f"After PEFT LoRA: {get_mem():.3f} GB")

    # Count digital LoRA layers
    digital_lora = sum(1 for n, m in model.named_modules()
                       if ('lora_A' in n or 'lora_B' in n) and isinstance(m, torch.nn.Linear))
    print(f"Digital LoRA layers: {digital_lora}")

    # Create sixt1c RPU config for LoRA layers
    rpu_config = SingleRPUConfig(
        device=LinearStepDevice(
            dw_min=0.001981,
            up_down=0.0,
            w_max=1.0,
            w_min=-1.0,
            gamma_up=-0.1678,
            gamma_down=0.1410,
        )
    )

    # Convert only LoRA layers to analog
    print("\nConverting LoRA layers to analog (sixt1c)...")
    model = convert_lora_layers_only_to_analog(model, rpu_config)

    # Count analog LoRA layers
    analog_lora = sum(1 for n, m in model.named_modules()
                      if ('lora_A' in n or 'lora_B' in n) and isinstance(m, AnalogLinear))
    print(f"Analog LoRA layers: {analog_lora}")

    model = model.to(device)
    print(f"After moving to CUDA: {get_mem():.3f} GB")

    # Prepare data
    print("\nLoading SQuAD data...")
    train_dataset = prepare_squad_data(tokenizer, max_samples=5000)  # Subset for testing
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Train samples: {len(train_dataset)}, Batches: {len(train_loader)}")

    # Optimizer and scheduler
    optimizer = AnalogAdam(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    # Training loop
    print(f"\nStarting training for {EPOCHS} epoch(s)...")
    model.train()

    step = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

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
            step += 1

            mem = get_mem()
            pbar.set_postfix(loss=f"{loss.item():.4f}", mem=f"{mem:.2f}GB")

            # Check for OOM warning
            if mem > 100:
                print(f"\nWARNING: Memory {mem:.2f}GB exceeded 100GB!")

            # Log every 100 steps
            if step % 100 == 0:
                avg_loss = total_loss / step
                print(f"\nStep {step}: avg_loss={avg_loss:.4f}, mem={mem:.3f}GB")

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} completed. Avg loss: {avg_loss:.4f}, Memory: {get_mem():.3f}GB")

    print(f"\nTraining complete!")
    print(f"Final memory: {get_mem():.3f} GB")

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"After cleanup: {get_mem():.3f} GB")

if __name__ == "__main__":
    main()
