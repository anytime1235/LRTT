# -*- coding: utf-8 -*-
"""LR Sweep Experiment: FP, FP Freeze, TTv1 with LR = 0.1, 0.01, 0.001 for 10 epochs."""

import os
import math
import json
import gc
from datetime import datetime

import torch
from torch import nn, device, no_grad, manual_seed
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import matplotlib.pyplot as plt
import numpy as np

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Fixed parameters
MODEL_NAME = "roneneldan/TinyStories-1M"
CONTEXT_LENGTH = 512
NUM_WORKERS = 4
SEED = 42
BATCH_SIZE = 32  # Reduced due to GPU memory constraints
NUM_EPOCHS = 10

# Dataset size limits
TRAIN_SAMPLES = 50000
VAL_SAMPLES = 5000

# Results directory
RESULTS_DIR = os.path.join(os.getcwd(), "results", "lr_sweep")
os.makedirs(RESULTS_DIR, exist_ok=True)


class TinyStoriesDataset(Dataset):
    """TinyStories dataset for language modeling."""

    def __init__(self, split="train", max_length=512, max_samples=None):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        dataset = load_dataset("roneneldan/TinyStories", split=split)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding=False,
            truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def collate_fn(batch):
    """Dynamic padding: pad to max length in batch."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        padded_input_ids = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), 50256, dtype=torch.long)
        ])
        padded_attention_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        labels = padded_input_ids.clone()
        labels[padded_attention_mask == 0] = -100

        input_ids_list.append(padded_input_ids)
        attention_mask_list.append(padded_attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


def load_data(batch_size):
    """Load train and validation dataloaders."""
    g = torch.Generator()
    g.manual_seed(SEED)

    train_dataset = TinyStoriesDataset(split="train", max_length=CONTEXT_LENGTH, max_samples=TRAIN_SAMPLES)
    val_dataset = TinyStoriesDataset(split="validation", max_length=CONTEXT_LENGTH, max_samples=VAL_SAMPLES)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True
    )
    return train_loader, val_loader


def compute_perplexity(loss):
    """Compute perplexity from loss."""
    return math.exp(min(loss, 20))


def evaluate(model, val_loader, criterion):
    """Evaluate model and return loss and perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            num_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return avg_loss, compute_perplexity(avg_loss)


def train_fp(learning_rate, freeze_transformer=False):
    """Train FP model (digital baseline)."""
    manual_seed(SEED)

    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_config(config)

    if freeze_transformer:
        # Freeze transformer blocks, only train lm_head
        for name, param in model.named_parameters():
            if "lm_head" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Frozen mode: {trainable:,} trainable params (lm_head only)")

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, momentum=0, weight_decay=0
    )

    train_loader, val_loader = load_data(BATCH_SIZE)

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_tokens = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            num_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            del logits, shift_logits, shift_labels, loss

            pbar.set_postfix({"loss": f"{total_loss/total_tokens:.4f}"})

        train_loss = total_loss / total_tokens
        val_loss, val_ppl = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")

    # Cleanup
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history


def train_ttv1(learning_rate):
    """Train TTv1 model (TransferCompound)."""
    from aihwkit.optim import AnalogSGD
    from aihwkit.nn import AnalogLinear
    from aihwkit.simulator.configs import MappingParameter, IOParameters
    from aihwkit.simulator.parameters import WeightNoiseType, BoundManagementType, NoiseManagementType
    from aihwkit.simulator.configs.compounds import TransferCompound
    from aihwkit.simulator.configs.devices import FloatingPointDevice
    from aihwkit.simulator.presets.utils import PresetIOParameters, PresetUpdateParameters
    from aihwkit.nn.conversion import convert_to_analog

    manual_seed(SEED)

    # Create TTv1 config
    def create_ttv1_config():
        from aihwkit.simulator.configs import UnitCellRPUConfig

        device_A = FloatingPointDevice()
        device_C = FloatingPointDevice()

        compound = TransferCompound(
            unit_cell_devices=[device_A, device_C],
            transfer_every=100,
            gamma=0.1,
        )

        rpu_config = UnitCellRPUConfig(device=compound)
        rpu_config.mapping = MappingParameter(
            max_input_size=512,
            max_output_size=512,
            digital_bias=True,
            weight_scaling_omega=1.0,
        )
        rpu_config.forward = IOParameters(
            w_noise=0.0,
            w_noise_type=WeightNoiseType.NONE,
            out_noise=0.0,
            bound_management=BoundManagementType.NONE,
            noise_management=NoiseManagementType.NONE,
            inp_bound=1.0,
            out_bound=1.0,
        )
        return rpu_config

    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_config(config)

    rpu_config = create_ttv1_config()
    model = convert_to_analog(model, rpu_config)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate, momentum=0, weight_decay=0)
    optimizer.regroup_param_groups(model)

    train_loader, val_loader = load_data(BATCH_SIZE)

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_tokens = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            num_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            del logits, shift_logits, shift_labels, loss

            pbar.set_postfix({"loss": f"{total_loss/total_tokens:.4f}"})

        train_loss = total_loss / total_tokens
        val_loss, val_ppl = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}")

    # Cleanup
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history


def plot_results(all_results):
    """Plot LR sweep results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    learning_rates = [0.1, 0.01, 0.001]
    colors = {'fp': 'blue', 'fp_freeze': 'green', 'ttv1': 'red'}
    linestyles = {0.1: '-', 0.01: '--', 0.001: ':'}

    epochs = list(range(1, NUM_EPOCHS + 1))

    # Plot 1: Training Loss
    ax1 = axes[0]
    for model_type in ['fp', 'fp_freeze', 'ttv1']:
        for lr in learning_rates:
            key = f"{model_type}_lr{lr}"
            if key in all_results:
                ax1.plot(epochs, all_results[key]["train_loss"],
                        color=colors[model_type], linestyle=linestyles[lr],
                        label=f"{model_type} lr={lr}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss vs Epoch")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax2 = axes[1]
    for model_type in ['fp', 'fp_freeze', 'ttv1']:
        for lr in learning_rates:
            key = f"{model_type}_lr{lr}"
            if key in all_results:
                ax2.plot(epochs, all_results[key]["val_loss"],
                        color=colors[model_type], linestyle=linestyles[lr],
                        label=f"{model_type} lr={lr}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss vs Epoch")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Validation Perplexity
    ax3 = axes[2]
    for model_type in ['fp', 'fp_freeze', 'ttv1']:
        for lr in learning_rates:
            key = f"{model_type}_lr{lr}"
            if key in all_results:
                ax3.plot(epochs, all_results[key]["val_ppl"],
                        color=colors[model_type], linestyle=linestyles[lr],
                        label=f"{model_type} lr={lr}")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Validation Perplexity")
    ax3.set_title("Validation Perplexity vs Epoch")
    ax3.set_yscale('log')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "lr_sweep_results.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved to {RESULTS_DIR}/lr_sweep_results.png")


def main():
    print(f"Device: {DEVICE}")
    print(f"Running LR sweep: FP, FP Freeze, TTv1")
    print(f"Learning rates: 0.1, 0.01, 0.001")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*70)

    all_results = {}
    learning_rates = [0.1, 0.01, 0.001]

    # FP experiments
    for lr in learning_rates:
        print(f"\n{'='*70}")
        print(f"Training FP with lr={lr}")
        print(f"{'='*70}")
        history = train_fp(lr, freeze_transformer=False)
        all_results[f"fp_lr{lr}"] = history

    # FP Freeze experiments
    for lr in learning_rates:
        print(f"\n{'='*70}")
        print(f"Training FP Freeze with lr={lr}")
        print(f"{'='*70}")
        history = train_fp(lr, freeze_transformer=True)
        all_results[f"fp_freeze_lr{lr}"] = history

    # TTv1 experiments
    for lr in learning_rates:
        print(f"\n{'='*70}")
        print(f"Training TTv1 with lr={lr}")
        print(f"{'='*70}")
        history = train_ttv1(lr)
        all_results[f"ttv1_lr{lr}"] = history

    # Save results
    with open(os.path.join(RESULTS_DIR, "lr_sweep_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/lr_sweep_results.json")

    # Plot results
    plot_results(all_results)


if __name__ == "__main__":
    main()
