# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""TinyStories-1M for Language Modeling using TikiTaka TTv2 (ChoppedTransferCompound with in_chop_prob=0.0) layers.

Model: TinyStories-1M (GPT-Neo architecture, loaded from HuggingFace)
- Uses exact architecture from roneneldan/TinyStories-1M
- Alternating global/local attention with window_size=256
- hidden_size=64, depth=8, num_heads=16
- Linear layers in transformer blocks converted to TTv2 analog layers
- Embeddings (wte, wpe) and lm_head remain digital

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

from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.configs import TikiTakaIdealizedPreset
from aihwkit.simulator.configs import MappingParameter, IOParameters
from aihwkit.simulator.parameters import BoundManagementType, NoiseManagementType, WeightNoiseType
from aihwkit.simulator.presets.devices import IdealizedPresetDevice
from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
from aihwkit.simulator.configs.devices import SoftBoundsDevice, LinearStepDevice, FloatingPointDevice
from aihwkit.simulator.presets.utils import PresetIOParameters, PresetUpdateParameters


# Device to use
USE_CUDA = torch.cuda.is_available()
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")
os.makedirs(PATH_DATASET, exist_ok=True)

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "TINYSTORIES_TTV2_SCRATCH")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "tinystories_1m_ttv2_scratch_model_weight.pth")

# Training parameters
SEED = 1
N_EPOCHS = 20
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 1e-3
LR_REDUCTION_FACTOR = 0.5
LR_PATIENCE = 3
EARLY_STOP_PATIENCE = 10
WEIGHT_DECAY = 0.01
OPTIMIZER = "AnalogAdam"  # "AnalogSGD", "AnalogAdam"
NUM_WORKERS = 4

# Model configuration
MODEL_NAME = "roneneldan/TinyStories-1M"
CONTEXT_LENGTH = 128  # Reduced for ChoppedTransferCompound compatibility test

# Embedding freeze option
FREEZE_EMBEDDINGS = False

# TikiTaka TTv2 configuration (ChoppedTransferCompound with in_chop_prob=0.0)
TRANSFER_EVERY = 1
UNITS_IN_MBATCH = True
FAST_LR = 0.5
AUTO_GRANULARITY = 10000

# Device configuration
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
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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


def create_ttv2_config():
    """Create TikiTaka TTv2 configuration (ChoppedTransferCompound with in_chop_prob=0.0)."""
    unit_devices = [_create_device(DEVICE_A), _create_device(DEVICE_C)]

    device_config = ChoppedTransferCompound(
        unit_cell_devices=unit_devices,
        transfer_forward=PresetIOParameters(
            noise_management=NoiseManagementType.NONE,
            bound_management=BoundManagementType.NONE
        ),
        transfer_update=PresetUpdateParameters(
            desired_bl=1,
            update_bl_management=False,
            update_management=False
        ),
        transfer_every=TRANSFER_EVERY * GRAD_ACCUM_STEPS,
        units_in_mbatch=UNITS_IN_MBATCH,
        in_chop_prob=0.0,  # TTv2: no chopping
        fast_lr=FAST_LR,
        auto_scale=True,
        auto_granularity=AUTO_GRANULARITY,
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
    """Convert nn.Linear layers in transformer blocks to AnalogLinear.

    Keeps embeddings (wte, wpe) and lm_head as digital layers.
    Only converts Linear layers inside transformer blocks (h).
    """
    converted_count = 0

    # Convert Linear layers in each transformer block
    for block_idx, block in enumerate(model.transformer.h):
        # Attention layers: q_proj, k_proj, v_proj, out_proj
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

        # MLP layers: c_fc, c_proj
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


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_analog_layers(model):
    count = 0
    for module in model.modules():
        if isinstance(module, AnalogLinear):
            count += 1
    return count


def create_model():
    """Create TinyStories-1M model with TTv2 analog layers."""
    print(f"Loading model from {MODEL_NAME}...")
    # Random initialization (from scratch)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_config(config)

    # Convert Linear layers to AnalogLinear
    print("Converting Linear layers to TTv2 AnalogLinear...")
    rpu_config = create_ttv2_config()
    model, converted_count = convert_to_analog(model, rpu_config)

    # Optionally freeze embeddings
    if FREEZE_EMBEDDINGS:
        model.transformer.wte.weight.requires_grad = False
        model.transformer.wpe.weight.requires_grad = False

    num_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    num_analog = count_analog_layers(model)

    config = model.config
    print(f"\nLoaded TinyStories-1M model (TTv2):")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Attention pattern: {config.attention_layers}")
    print(f"  Window size: {config.window_size}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {num_params:,}")
    print(f"  Converted layers: {converted_count}")
    print(f"  Analog (TTv2) layers: {num_analog}")
    print(f"  TTv2: transfer_every={TRANSFER_EVERY}, fast_lr={FAST_LR}, auto_granularity={AUTO_GRANULARITY}")
    print(f"  Devices: A={DEVICE_A}, C={DEVICE_C}\n")

    return model


def load_data():
    """Load TinyStories dataset."""
    print("Loading TinyStories dataset...")
    train_dataset = TinyStoriesDataset(split="train", max_length=CONTEXT_LENGTH, max_samples=50000)
    val_dataset = TinyStoriesDataset(split="validation", max_length=CONTEXT_LENGTH, max_samples=5000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=USE_CUDA)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=USE_CUDA)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_loader, val_loader


def create_optimizer(model, learning_rate, weight_decay):
    if OPTIMIZER == "AnalogSGD":
        return AnalogSGD(model.parameters(), lr=learning_rate)
    else:
        return AnalogAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def compute_perplexity(loss):
    return math.exp(min(loss, 100))


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


def main():
    """Train TinyStories-1M with TTv2."""
    manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    wandb.init(
        project="tinystories_1m_ttv2_scratch",
        name=f"tinystories_ttv2_te{TRANSFER_EVERY}_flr{FAST_LR}",
        config={
            "model": "TinyStories-1M-TTv2",
            "model_source": MODEL_NAME,
            "transfer_every": TRANSFER_EVERY,
            "fast_lr": FAST_LR,
            "auto_granularity": AUTO_GRANULARITY,
            "device_a": DEVICE_A,
            "device_c": DEVICE_C,
            "epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        }
    )

    train_loader, val_loader = load_data()
    model = create_model()
    if USE_CUDA:
        model = model.to(DEVICE)
    print(f"Model moved to {DEVICE}")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = create_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_REDUCTION_FACTOR, patience=LR_PATIENCE)

    best_ppl = float('inf')
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

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
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
        val_loss, val_ppl = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)
        wandb.log({"epoch": epoch+1, "train/perplexity": train_ppl, "eval/perplexity": val_ppl})

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            epochs_without_improvement = 0
            save(model.state_dict(), WEIGHT_PATH)
        else:
            epochs_without_improvement += 1

        tqdm.write(f"Epoch {epoch+1}: Train PPL {train_ppl:.2f} | Val PPL {val_ppl:.2f} | Best {best_ppl:.2f} | No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}")

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Compute generation metrics
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Required for decoder-only generation
    gen_metrics = compute_generation_metrics(model, tokenizer)

    print(f"\n{'='*60}")
    print(f"Training completed! Best perplexity: {best_ppl:.2f}")
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
