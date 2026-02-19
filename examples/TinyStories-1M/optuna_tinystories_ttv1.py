# -*- coding: utf-8 -*-
"""Optuna hyperparameter sweep for TinyStories-1M with TTv1 (TransferCompound).

TTv1 uses TransferCompound with 2 devices: A (accumulator) and C (weight).

Usage:
    # SGD with all hyperparams tuned
    python optuna_tinystories_ttv1.py --n-trials 50 --optimizer sgd

    # Adam with weight decay tuning
    python optuna_tinystories_ttv1.py --n-trials 50 --optimizer adam

    # SGD without momentum tuning (momentum=0 fixed)
    python optuna_tinystories_ttv1.py --n-trials 50 --optimizer sgd --no-momentum

    # SGD with nothing tuned (baseline)
    python optuna_tinystories_ttv1.py --n-trials 50 --optimizer sgd --no-wd --no-momentum --no-nesterov

    # Visualize results
    python optuna_tinystories_ttv1.py --visualize --optimizer sgd

    # Dashboard
    optuna-dashboard sqlite:///results/optuna_tinystories_ttv1/optuna_tinystories_ttv1_sgd.db
"""

import os
import math
import json
import argparse
import gc
from datetime import datetime

import torch
from torch import nn, device, no_grad, manual_seed
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import optuna
from optuna.trial import TrialState
from optuna_integration import BoTorchSampler
import matplotlib.pyplot as plt


# Global configuration (set by argparse)
OPT_CONFIG = {
    'optimizer': 'sgd',      # 'sgd' or 'adam'
    'tune_wd': True,         # tune weight_decay
    'tune_momentum': True,   # tune momentum (SGD only)
    'tune_nesterov': True,   # tune nesterov (SGD only)
}


def get_study_name_suffix():
    """Generate study name suffix based on optimizer config."""
    opt = OPT_CONFIG['optimizer']
    suffix = opt

    if opt == 'sgd':
        if not OPT_CONFIG['tune_wd']:
            suffix += "_nowd"
        if not OPT_CONFIG['tune_momentum']:
            suffix += "_nomom"
        if not OPT_CONFIG['tune_nesterov']:
            suffix += "_nonest"
    else:  # adam
        if not OPT_CONFIG['tune_wd']:
            suffix += "_nowd"

    return suffix

from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.configs import TikiTakaIdealizedPreset
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import WeightNoiseType, BoundManagementType, NoiseManagementType
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.configs.compounds import TransferCompound
from aihwkit.simulator.configs.devices import SoftBoundsDevice, LinearStepDevice, FloatingPointDevice
from aihwkit.simulator.presets.utils import PresetIOParameters, PresetUpdateParameters

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Fixed parameters
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")
os.makedirs(PATH_DATASET, exist_ok=True)
RESULTS = os.path.join(os.getcwd(), "results", "optuna_tinystories_ttv1")
os.makedirs(RESULTS, exist_ok=True)

# Model configuration
MODEL_NAME = "roneneldan/TinyStories-1M"
CONTEXT_LENGTH = 512
NUM_WORKERS = 4
SEED = 42

# Dataset size limits
TRAIN_SAMPLES = 50000
VAL_SAMPLES = 5000

# Batch size and gradient accumulation
BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1  # Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS

# Device configuration for TTv1
DEVICE_A = "6t1c"
DEVICE_C = "softbounds"


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
        # Don't pad here, use fixed-length padding in collate_fn
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
        # Pad with eos_token_id (50256 for GPT-2)
        padded_input_ids = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), 50256, dtype=torch.long)
        ])
        padded_attention_mask = torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        # Create labels: copy input_ids, mask padding with -100
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


def _create_device(device_type):
    """Create device based on type string."""
    if device_type == "6t1c":
        return LinearStepDevice(
            dw_min=0.001981, up_down=0.0, w_max=1.0, w_min=-1.0,
            gamma_up=-0.1678, gamma_down=0.1410, mult_noise=True,
            dw_min_dtod=0.1, up_down_dtod=0.01,
            w_max_dtod=0.05, w_min_dtod=0.05,
            gamma_up_dtod=0.05, gamma_down_dtod=0.05,
            dw_min_std=0.3, write_noise_std=0,
            mean_bound_reference=True,
        )
    elif device_type == "softbounds":
        return SoftBoundsDevice(
            w_max=1.0, w_min=-1.0, w_max_dtod=0.0, w_min_dtod=0.0,
            dw_min=0.001, dw_min_dtod=0.0, dw_min_std=0.0,
            up_down=0.0, up_down_dtod=0.0,
            mult_noise=True, write_noise_std=0.0,
        )
    elif device_type == "floating_point":
        return FloatingPointDevice()
    else:
        return IdealizedPresetDevice(w_max=1.0, w_min=-1.0)


# Global config holder
_current_config = {}


def create_ttv1_config():
    """Create TikiTaka TTv1 configuration."""
    unit_devices = [_create_device(DEVICE_A), _create_device(DEVICE_C)]

    device_config = TransferCompound(
        unit_cell_devices=unit_devices,
        transfer_forward=PresetIOParameters(
            noise_management=NoiseManagementType.NONE,
            bound_management=BoundManagementType.NONE
        ),
        transfer_update=PresetUpdateParameters(),
        transfer_every=_current_config.get('transfer_every', 1),
        units_in_mbatch=True,
    )

    mapping = MappingParameter(
        weight_scaling_omega=1.0, learn_out_scaling=False,
        weight_scaling_lr_compensation=True, digital_bias=True,
        weight_scaling_columnwise=False, out_scaling_columnwise=True,
        max_input_size=1024, max_output_size=1024
    )

    forward_io = IOParameters(
        inp_res=0.007937, inp_bound=1.0, inp_noise=0.0, inp_sto_round=False,
        out_res=0.001961, out_bound=12.0, out_noise=0.06,
        w_noise=0.0, w_noise_type=WeightNoiseType.NONE,
        bound_management=BoundManagementType.ITERATIVE,
        noise_management=NoiseManagementType.ABS_MAX,
        is_perfect=False, max_bm_factor=1000,
    )

    config = TikiTakaIdealizedPreset()
    config.device = device_config
    config.mapping = mapping
    config.forward = forward_io
    config.backward = forward_io
    return config


def convert_to_analog(model, rpu_config):
    """Convert nn.Linear layers in transformer blocks to AnalogLinear."""
    converted_count = 0

    for block_idx, block in enumerate(model.transformer.h):
        attn = block.attn.attention

        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
            old_layer = getattr(attn, proj_name)
            new_layer = AnalogLinear(
                old_layer.in_features,
                old_layer.out_features,
                bias=old_layer.bias is not None,
                rpu_config=rpu_config
            )
            new_layer.set_weights(old_layer.weight.data, old_layer.bias.data if old_layer.bias is not None else None)
            setattr(attn, proj_name, new_layer)
            converted_count += 1

        mlp = block.mlp

        for mlp_name in ['c_fc', 'c_proj']:
            old_layer = getattr(mlp, mlp_name)
            new_layer = AnalogLinear(
                old_layer.in_features,
                old_layer.out_features,
                bias=old_layer.bias is not None,
                rpu_config=rpu_config
            )
            new_layer.set_weights(old_layer.weight.data, old_layer.bias.data if old_layer.bias is not None else None)
            setattr(mlp, mlp_name, new_layer)
            converted_count += 1

    return model, converted_count


def count_analog_layers(model):
    """Count number of AnalogLinear layers."""
    count = 0
    for module in model.modules():
        if isinstance(module, AnalogLinear):
            count += 1
    return count


def load_data(batch_size):
    """Load TinyStories dataset."""
    train_dataset = TinyStoriesDataset(split="train", max_length=CONTEXT_LENGTH, max_samples=TRAIN_SAMPLES)
    val_dataset = TinyStoriesDataset(split="validation", max_length=CONTEXT_LENGTH, max_samples=VAL_SAMPLES)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=USE_CUDA, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=USE_CUDA, collate_fn=collate_fn)
    return train_loader, val_loader


def compute_perplexity(loss):
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 100))


# Story beginning prompts for generation evaluation (5 for final eval)
EVAL_PROMPTS = [
    "Once upon a time, there was a little",
    "The sun was shining and the",
    "Lily wanted to play with her",
    "One day, a boy named Tom",
    "In a small house, there lived",
]

# 16 fixed prompts for fast inner-loop evaluation (batched, every N epochs)
FAST_EVAL_PROMPTS = [
    "Once upon a time, there was a little",
    "The sun was shining and the",
    "Lily wanted to play with her",
    "One day, a boy named Tom",
    "In a small house, there lived",
    "The little dog ran to the",
    "Mom said it was time to",
    "Ben found a big red",
    "The cat jumped on the",
    "It was a sunny day and",
    "The bunny was hiding in the",
    "The cookie was so",
    "Tim ran fast to the",
    "The stars were shining in the",
    "The puppy was very",
    "The princess lived in a",
]

# Fast eval settings
FAST_EVAL_TOKENS = 64
FAST_EVAL_EVERY = 3  # Run every N epochs

# D2/R4 pruning thresholds (conservative initial values)
D2_MIN_THRESHOLD = 0.3   # Distinct-2 below this → likely degenerate
R4_MAX_THRESHOLD = 0.15  # Repetition-4 above this → likely repetitive
PRUNE_WARMUP_EPOCHS = 3  # Don't prune before this epoch


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


def fast_generation_eval(model, tokenizer, max_new_tokens=FAST_EVAL_TOKENS, seed=42):
    """Fast inner-loop evaluation with batched generation.

    Args:
        seed: Fixed seed for reproducibility across evaluations

    Returns:
        dict with distinct_2 (higher=better) and repetition_4 (lower=better)
    """
    model.eval()

    # Fixed seed for stable comparison
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Batch tokenization with padding
    inputs = tokenizer(
        FAST_EVAL_PROMPTS,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(DEVICE)

    prompt_lengths = inputs.attention_mask.sum(dim=1)  # Length of each prompt

    with no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract completions only (exclude prompts)
    completions = []
    for i, output in enumerate(outputs):
        prompt_len = prompt_lengths[i].item()
        completion_ids = output[prompt_len:]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        completions.append(completion_text)

    return {
        "distinct_2": distinct_n(completions, 2),
        "repetition_4": repetition_rate(completions, 4),
    }


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


def objective(trial):
    """Optuna objective function - minimizes perplexity."""
    global _current_config

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Hyperparameters to tune
    transfer_every = trial.suggest_int('transfer_every', 1, 10000, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e0, log=True)
    batch_size = BATCH_SIZE
    grad_accum_steps = GRAD_ACCUM_STEPS

    # Optimizer-specific hyperparameters based on config
    opt_type = OPT_CONFIG['optimizer']

    if OPT_CONFIG['tune_wd']:
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    else:
        weight_decay = 0

    if opt_type == 'sgd':
        if OPT_CONFIG['tune_momentum']:
            momentum = trial.suggest_float('momentum', 0.0, 0.99)
        else:
            momentum = 0

        if OPT_CONFIG['tune_nesterov'] and momentum > 0:
            nesterov = trial.suggest_categorical('nesterov', [True, False])
        else:
            nesterov = False
    else:
        momentum = 0
        nesterov = False

    max_epochs = 40
    early_stop_patience = 7

    # Scale transfer_every by grad_accum_steps (counted per forward pass)
    _current_config = {
        'transfer_every': transfer_every * grad_accum_steps,
    }

    manual_seed(SEED)

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting (TTv1 - TransferCompound)")
    print(f"{'='*70}")
    print(f"  transfer_every={transfer_every} (actual={transfer_every*grad_accum_steps}), lr={learning_rate:.2e}")
    print(f"  optimizer={opt_type}, wd={weight_decay:.2e}, momentum={momentum:.2f}, nesterov={nesterov}")
    print(f"  batch_size={batch_size}, grad_accum={grad_accum_steps} (effective={batch_size*grad_accum_steps})")
    print(f"{'='*70}")

    train_loader, val_loader = load_data(batch_size)

    model = None
    try:
        # Create model from scratch (random initialization)
        config = AutoConfig.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_config(config)
        rpu_config = create_ttv1_config()
        model, converted_count = convert_to_analog(model, rpu_config)
        num_analog = count_analog_layers(model)

        print(f"  Loaded TinyStories-1M: {converted_count} layers converted, {num_analog} analog layers")

        model = model.to(DEVICE)

        if opt_type == "adam":
            optimizer = AnalogAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:  # sgd
            optimizer = AnalogSGD(model.parameters(), lr=learning_rate,
                                  momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        optimizer.regroup_param_groups(model)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Initialize tokenizer for fast generation eval
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # Required for decoder-only generation

        best_perplexity = float('inf')
        epochs_without_improvement = 0

        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0
            epoch_tokens = 0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss / grad_accum_steps
                loss.backward()

                num_tokens = (shift_labels != -100).sum().item()
                epoch_loss += loss.item() * grad_accum_steps * num_tokens
                epoch_tokens += num_tokens

                # Free intermediate tensors
                del logits, shift_logits, shift_labels, loss

                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            # Handle remaining gradients at end of epoch
            if (batch_idx + 1) % grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
            train_ppl = compute_perplexity(train_loss)
            val_loss, val_ppl = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)

            # Fast generation eval (every N epochs, batched)
            distinct_2, repetition_4 = None, None
            if (epoch + 1) % FAST_EVAL_EVERY == 0:
                fast_metrics = fast_generation_eval(model, tokenizer)
                distinct_2 = fast_metrics["distinct_2"]
                repetition_4 = fast_metrics["repetition_4"]

            improved = ""
            if val_ppl < best_perplexity:
                best_perplexity = val_ppl
                epochs_without_improvement = 0
                improved = " *"
            else:
                epochs_without_improvement += 1

            current_lr = optimizer.param_groups[0]['lr']
            if distinct_2 is not None:
                print(f"[Trial {trial.number}] Epoch {epoch+1:3d} | "
                      f"PPL: {val_ppl:7.2f} | D2: {distinct_2:.3f} | R4: {repetition_4:.3f} | "
                      f"Best: {best_perplexity:7.2f} | LR: {current_lr:.2e} | "
                      f"No imp: {epochs_without_improvement}/{early_stop_patience}{improved}")
            else:
                print(f"[Trial {trial.number}] Epoch {epoch+1:3d} | "
                      f"PPL: {val_ppl:7.2f} | "
                      f"Best: {best_perplexity:7.2f} | LR: {current_lr:.2e} | "
                      f"No imp: {epochs_without_improvement}/{early_stop_patience}{improved}")

            trial.report(val_ppl, epoch)

            # D2/R4 guardrail: return penalty instead of prune (for BoTorch compatibility)
            # Note: (epoch + 1) to match 1-indexed epoch display
            if distinct_2 is not None and (epoch + 1) >= PRUNE_WARMUP_EPOCHS:
                if distinct_2 < D2_MIN_THRESHOLD or repetition_4 > R4_MAX_THRESHOLD:
                    print(f"[Trial {trial.number}] D2/R4 guardrail triggered: "
                          f"D2={distinct_2:.3f} (<{D2_MIN_THRESHOLD}) or R4={repetition_4:.3f} (>{R4_MAX_THRESHOLD})")
                    # Store last D2/R4 for constraints_func if needed
                    trial.set_user_attr("guardrail_triggered", True)
                    trial.set_user_attr("last_distinct_2", distinct_2)
                    trial.set_user_attr("last_repetition_4", repetition_4)
                    # Return penalty value (BoTorch learns "this region is bad")
                    PENALTY_VALUE = 10000.0
                    return PENALTY_VALUE

            if epochs_without_improvement >= early_stop_patience:
                break

            if trial.should_prune():
                print(f"[Trial {trial.number}] Pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned()

        # Compute final generation metrics (more thorough)
        gen_metrics = compute_generation_metrics(model, tokenizer)

        print(f"\n[Trial {trial.number}] Finished - Best Perplexity: {best_perplexity:.2f}")
        print(f"  Generation Metrics:")
        print(f"    Distinct-1: {gen_metrics['distinct_1']:.4f}")
        print(f"    Distinct-2: {gen_metrics['distinct_2']:.4f}")
        print(f"    Distinct-3: {gen_metrics['distinct_3']:.4f}")
        print(f"    Repetition-4: {gen_metrics['repetition_4']:.4f}")
        print(f"{'='*70}\n")

        # Store generation metrics as user attributes
        trial.set_user_attr("guardrail_triggered", False)
        trial.set_user_attr("distinct_1", gen_metrics['distinct_1'])
        trial.set_user_attr("distinct_2", gen_metrics['distinct_2'])
        trial.set_user_attr("distinct_3", gen_metrics['distinct_3'])
        trial.set_user_attr("repetition_4", gen_metrics['repetition_4'])

        return best_perplexity

    finally:
        # Delete training loop variables that hold GPU references
        # Python for-loop variables persist in function scope
        try:
            del outputs
        except NameError:
            pass
        try:
            del batch
        except NameError:
            pass
        try:
            del input_ids
        except NameError:
            pass
        try:
            del attention_mask
        except NameError:
            pass
        try:
            del labels
        except NameError:
            pass
        try:
            del batch_idx
        except NameError:
            pass
        try:
            del epoch
        except NameError:
            pass
        # Delete in reverse dependency order: scheduler → optimizer → model
        # optimizer holds references to analog tiles via param_groups
        if 'scheduler' in dir():
            del scheduler
        if 'optimizer' in dir():
            del optimizer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        tqdm.write(f"[Trial {trial.number}] GPU cache cleared")


def visualize_study(study, save_dir):
    """Generate visualization plots for the study."""
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(complete_trials) == 0:
        print("No completed trials to visualize.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    trial_numbers = [t.number for t in complete_trials]
    perplexities = [t.value for t in complete_trials]
    ax.scatter(trial_numbers, perplexities, alpha=0.6)
    ax.plot(trial_numbers, [min(perplexities[:i+1]) for i in range(len(perplexities))],
            'r-', linewidth=2, label='Best so far')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Perplexity')
    ax.set_title('Optimization History (Lower is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    try:
        importances = optuna.importance.get_param_importances(study)
        param_names = list(importances.keys())
        values = list(importances.values())
        ax.barh(param_names[::-1], values[::-1])
        ax.set_xlabel('Importance (fANOVA)')
        ax.set_title('Parameter Importance')
    except Exception as e:
        ax.text(0.5, 0.5, f'Not enough trials\n({e})', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Parameter Importance (unavailable)')

    ax = axes[1, 0]
    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    ax.scatter(lrs, perplexities, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Perplexity')
    ax.set_title('Learning Rate vs Perplexity')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    transfer_every = [t.params.get('transfer_every', 1) for t in complete_trials]
    ax.scatter(transfer_every, perplexities, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Transfer Every')
    ax.set_ylabel('Perplexity')
    ax.set_title('Transfer Every vs Perplexity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "visualization.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {fig_path}")
    plt.close()

    all_trials_data = []
    for t in study.trials:
        trial_data = {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            "params": t.params,
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
            "duration_seconds": (t.datetime_complete - t.datetime_start).total_seconds()
                               if t.datetime_complete and t.datetime_start else None,
        }
        all_trials_data.append(trial_data)

    history_path = os.path.join(save_dir, "all_trials.json")
    with open(history_path, 'w') as f:
        json.dump({
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number if study.best_trial else None,
            "best_value": study.best_value if study.best_trial else None,
            "best_params": study.best_params if study.best_trial else None,
            "trials": all_trials_data,
        }, f, indent=2)
    print(f"Trial history saved to: {history_path}")


def print_study_summary(study):
    """Print summary of the study."""
    print("\n" + "=" * 60)
    print("STUDY SUMMARY (TTv1 - TransferCompound)")
    print("=" * 60)

    pruned_trials = [t for t in study.trials if t.state == TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    running_trials = [t for t in study.trials if t.state == TrialState.RUNNING]

    print(f"Study name: {study.study_name}")
    print(f"Total trials: {len(study.trials)}")
    print(f"  - Complete: {len(complete_trials)}")
    print(f"  - Pruned: {len(pruned_trials)}")
    print(f"  - Running: {len(running_trials)}")

    if complete_trials:
        perplexities = [t.value for t in complete_trials]
        print(f"\nPerplexity statistics (lower is better):")
        print(f"  - Best: {min(perplexities):.2f}")
        print(f"  - Mean: {sum(perplexities)/len(perplexities):.2f}")
        print(f"  - Worst: {max(perplexities):.2f}")

        print(f"\nBest trial (#{study.best_trial.number}):")
        print(f"  Perplexity: {study.best_value:.2f}")
        print("  Params:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")


def main():
    """Run Optuna hyperparameter sweep."""
    global OPT_CONFIG

    parser = argparse.ArgumentParser(description="Optuna sweep for TinyStories-1M TTv1")
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: auto-generated based on optimizer config)')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials to run (default: 50)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds (default: None)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Database path (default: auto-generated)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize existing results without running new trials')
    parser.add_argument('--new-study', action='store_true',
                        help='Start a new study (ignore existing results)')

    # Optimizer configuration
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='Optimizer type (default: sgd)')
    parser.add_argument('--no-wd', action='store_true',
                        help='Disable weight_decay tuning (fix to 0)')
    parser.add_argument('--no-momentum', action='store_true',
                        help='Disable momentum tuning for SGD (fix to 0)')
    parser.add_argument('--no-nesterov', action='store_true',
                        help='Disable nesterov tuning for SGD (fix to False)')

    args = parser.parse_args()

    # Set global optimizer config
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov

    os.makedirs(RESULTS, exist_ok=True)

    # Auto-generate study name based on optimizer config (includes batch size)
    study_name = args.study_name or f"tinystories_ttv1_bs{BATCH_SIZE}_{get_study_name_suffix()}"
    storage = args.storage or f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    if args.visualize:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            print_study_summary(study)
            visualize_study(study, RESULTS)
            print(f"\nTo run dashboard: optuna-dashboard {storage}")
        except Exception as e:
            print(f"Error loading study: {e}")
            print(f"No existing study found with name '{study_name}'")
        return

    if args.new_study:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"Deleted existing study: {study_name}")
        except:
            pass

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=BoTorchSampler(),
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    existing_trials = len(study.trials)
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if existing_trials > 0:
        print(f"\nResuming study '{study_name}' with {existing_trials} existing trials ({len(completed_trials)} completed)")
        if completed_trials:
            print(f"Current best perplexity: {study.best_value:.2f}")

    opt_info = f"optimizer={args.optimizer}"
    if args.optimizer == 'sgd':
        opt_info += f", wd={'tune' if OPT_CONFIG['tune_wd'] else '0'}"
        opt_info += f", mom={'tune' if OPT_CONFIG['tune_momentum'] else '0'}"
        opt_info += f", nest={'tune' if OPT_CONFIG['tune_nesterov'] else 'False'}"
    else:
        opt_info += f", wd={'tune' if OPT_CONFIG['tune_wd'] else '0'}"

    print(f"\n{'='*60}")
    print(f"Study: {study_name} (TTv1 - TransferCompound)")
    print(f"Database: {storage}")
    print(f"Device: {DEVICE}")
    print(f"Config: {opt_info}")
    print(f"New trials: {args.n_trials}")
    print(f"Metric: Perplexity (minimize)")
    print(f"{'='*60}\n")

    def delete_failed_trial_callback(study, trial):
        if trial.state == TrialState.FAIL:
            print(f"[Trial {trial.number}] Failed - removing from database")
            try:
                study._storage.delete_trial(trial._trial_id)
            except Exception as e:
                print(f"[Trial {trial.number}] Could not delete: {e}")

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                   catch=(Exception,), show_progress_bar=False,
                   callbacks=[delete_failed_trial_callback])

    print_study_summary(study)
    visualize_study(study, RESULTS)

    if study.best_trial:
        results_path = os.path.join(RESULTS, f"best_params_{study_name}.json")
        with open(results_path, 'w') as f:
            json.dump({
                "study_name": study_name,
                "best_perplexity": study.best_value,
                "best_params": study.best_params,
                "best_trial_number": study.best_trial.number,
                "n_trials": len(study.trials),
            }, f, indent=2)
        print(f"\nBest params saved to: {results_path}")


if __name__ == "__main__":
    main()
