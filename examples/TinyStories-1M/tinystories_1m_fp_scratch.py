# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""TinyStories-1M for Language Modeling using Digital/FP layers.

Digital floating-point baseline for comparison with analog methods.

Model: TinyStories-1M (GPT-Neo architecture, loaded from HuggingFace)
- Uses exact architecture from roneneldan/TinyStories-1M
- Alternating global/local attention with window_size=256
- hidden_size=64, depth=8, num_heads=16
- vocab_size=50257 (GPT-2 tokenizer)
- max_position_embeddings=2048

Reference: Eldan & Li, "TinyStories: How Small Can Language Models Be
           and Still Speak Coherent English?" (arXiv:2305.07759)
"""
# pylint: disable=invalid-name

import os
import math

import torch
from torch import nn, device, no_grad, manual_seed, save
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")
os.makedirs(PATH_DATASET, exist_ok=True)

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "TINYSTORIES_FP_SCRATCH")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "tinystories_1m_fp_scratch_model_weight.pth")

# Training parameters
SEED = 1
N_EPOCHS = 20
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1  # Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 6e-2
LR_REDUCTION_FACTOR = 0.5
LR_PATIENCE = 3
EARLY_STOP_PATIENCE = 10
WEIGHT_DECAY = 0.01
OPTIMIZER = "SGD"  # "SGD", "Adam"
NUM_WORKERS = 4

# Model configuration
MODEL_NAME = "roneneldan/TinyStories-1M"
CONTEXT_LENGTH = 512

# Freeze option: freeze attention (qkvo) and MLP (fc1, fc2) layers
FREEZE_TRANSFORMER = True


class TinyStoriesDataset(Dataset):
    """TinyStories dataset for language modeling."""

    def __init__(self, split="train", max_length=512, max_samples=None):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

        # Load TinyStories dataset
        dataset = load_dataset("roneneldan/TinyStories", split=split)

        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]

        # Tokenize - don't pad here, use collate_fn for dynamic padding
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


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


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_linear_layers(model):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            count += 1
    return count


def freeze_transformer_layers(model):
    """Freeze transformer block layers (q_proj, k_proj, v_proj, out_proj, c_fc, c_proj).

    Only embeddings (wte, wpe), layer norms, and lm_head remain trainable.
    """
    frozen_count = 0
    frozen_params = 0

    for block in model.transformer.h:
        # Freeze attention layers: q_proj, k_proj, v_proj, out_proj
        attn = block.attn.attention
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            layer = getattr(attn, proj_name)
            for param in layer.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            frozen_count += 1

        # Freeze MLP layers: c_fc, c_proj
        mlp = block.mlp
        for mlp_name in ['c_fc', 'c_proj']:
            layer = getattr(mlp, mlp_name)
            for param in layer.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            frozen_count += 1

    return frozen_count, frozen_params


def create_model():
    """Create TinyStories-1M model from HuggingFace (Digital FP)."""
    print(f"Loading model from {MODEL_NAME}...")
    # Random initialization (from scratch)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_config(config)

    # Freeze transformer layers (attention + MLP)
    if FREEZE_TRANSFORMER:
        frozen_count, frozen_params = freeze_transformer_layers(model)
        print(f"  Frozen {frozen_count} layers ({frozen_params:,} params)")

    num_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    num_linear = count_linear_layers(model)

    config = model.config
    print(f"\nLoaded TinyStories-1M model (Digital FP):")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Max position embeddings: {config.max_position_embeddings}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Attention pattern: {config.attention_layers}")
    print(f"  Window size: {config.window_size}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {num_params:,}")
    print(f"  Linear layers: {num_linear}")
    print(f"  Transformer frozen: {FREEZE_TRANSFORMER}")
    print(f"  Mode: Full Precision (FP32)\n")

    return model


def load_data():
    """Load TinyStories dataset."""
    print("Loading TinyStories dataset...")

    train_dataset = TinyStoriesDataset(
        split="train",
        max_length=CONTEXT_LENGTH,
        max_samples=50000  # Limit for faster training
    )
    val_dataset = TinyStoriesDataset(
        split="validation",
        max_length=CONTEXT_LENGTH,
        max_samples=5000
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=USE_CUDA,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_CUDA,
        collate_fn=collate_fn
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


def create_optimizer(model, learning_rate, weight_decay):
    """Create optimizer."""
    if OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    return optimizer


def compute_perplexity(loss):
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 100))  # Clip to avoid overflow


# Story beginning prompts for generation evaluation
EVAL_PROMPTS = [
    "Once upon a time, there was a little",
    "The sun was shining and the",
    "Lily wanted to play with her",
    "One day, a boy named Tom",
    "In a small house, there lived",
]


def distinct_n(texts, n):
    """Calculate distinct-n: unique n-grams / total n-grams (higher is better)."""
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        if len(tokens) >= n:
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def repetition_rate(texts, n=4):
    """Calculate repetition rate: repeated n-grams ratio (lower is better)."""
    total_ngrams = 0
    repeated_ngrams = 0
    for text in texts:
        tokens = text.split()
        if len(tokens) >= n:
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            total_ngrams += len(ngrams)
            repeated_ngrams += len(ngrams) - len(set(ngrams))
    if total_ngrams == 0:
        return 0.0
    return repeated_ngrams / total_ngrams


def generate_samples(model, tokenizer, prompts, max_new_tokens=100, num_samples=3):
    """Generate text samples from prompts."""
    model.eval()
    generated_texts = []

    with no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            for _ in range(num_samples):
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_texts.append(text)

    return generated_texts


def compute_generation_metrics(model, tokenizer):
    """Compute generation-based metrics (Distinct-n, Repetition rate)."""
    generated_texts = generate_samples(model, tokenizer, EVAL_PROMPTS, max_new_tokens=100, num_samples=3)

    metrics = {
        "distinct_1": distinct_n(generated_texts, 1),
        "distinct_2": distinct_n(generated_texts, 2),
        "distinct_3": distinct_n(generated_texts, 3),
        "repetition_4": repetition_rate(generated_texts, 4),
        "num_samples": len(generated_texts),
    }
    return metrics


def evaluate(model, val_loader, criterion):
    """Evaluate model on validation set."""
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

            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Count non-padding tokens
            num_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = compute_perplexity(avg_loss)

    return avg_loss, perplexity


def main():
    """Train TinyStories-1M with Digital FP."""
    manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    # Initialize wandb
    wandb.init(
        project="tinystories_1m_fp_scratch",
        name=f"tinystories_fp_bs{BATCH_SIZE}_e{N_EPOCHS}_lr{LEARNING_RATE}",
        config={
            "model": "TinyStories-1M",
            "model_source": MODEL_NAME,
            "context_length": CONTEXT_LENGTH,
            "freeze_transformer": FREEZE_TRANSFORMER,
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": OPTIMIZER,
            "seed": SEED,
            "device": str(DEVICE),
        }
    )

    # Load data
    train_loader, val_loader = load_data()

    # Create model
    model = create_model()
    if USE_CUDA:
        model = model.to(DEVICE)
    print(f"Model moved to {DEVICE}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = create_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_REDUCTION_FACTOR, patience=LR_PATIENCE
    )

    best_perplexity = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"\n{'='*60}")
    print(f"Starting training: max {N_EPOCHS} epochs (early stop patience: {EARLY_STOP_PATIENCE})")
    print(f"Metric: Perplexity (lower is better)")
    print(f"{'='*60}\n")

    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        model.train()
        epoch_loss = 0
        epoch_tokens = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            num_tokens = (shift_labels != -100).sum().item()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS * num_tokens
            epoch_tokens += num_tokens

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Handle remaining gradients at end of epoch
        if (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        train_ppl = compute_perplexity(train_loss)

        # Validation
        val_loss, val_ppl = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/perplexity": train_ppl,
            "eval/loss": val_loss,
            "eval/perplexity": val_ppl,
            "learning_rate": current_lr,
        })

        if val_ppl < best_perplexity:
            best_perplexity = val_ppl
            best_epoch = epoch
            epochs_without_improvement = 0
            save(model.state_dict(), WEIGHT_PATH)
        else:
            epochs_without_improvement += 1

        tqdm.write(f"Epoch {epoch+1}: Train PPL {train_ppl:.2f} | Val PPL {val_ppl:.2f} | Best {best_perplexity:.2f} | No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Compute generation metrics
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Required for decoder-only generation
    gen_metrics = compute_generation_metrics(model, tokenizer)

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation perplexity: {best_perplexity:.2f} at epoch {best_epoch + 1}")
    print(f"Model weights saved to: {WEIGHT_PATH}")
    print(f"\nGeneration Metrics:")
    print(f"  Distinct-1: {gen_metrics['distinct_1']:.4f}")
    print(f"  Distinct-2: {gen_metrics['distinct_2']:.4f}")
    print(f"  Distinct-3: {gen_metrics['distinct_3']:.4f}")
    print(f"  Repetition-4: {gen_metrics['repetition_4']:.4f}")
    print(f"{'='*60}")

    wandb.log({
        "generation/distinct_1": gen_metrics['distinct_1'],
        "generation/distinct_2": gen_metrics['distinct_2'],
        "generation/distinct_3": gen_metrics['distinct_3'],
        "generation/repetition_4": gen_metrics['repetition_4'],
    })

    wandb.finish()


if __name__ == "__main__":
    main()
