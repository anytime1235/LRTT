# -*- coding: utf-8 -*-
"""MobileBERT + GLUE with TikiTaka v1.

Single-run training script for MobileBERT on GLUE tasks using TikiTaka v1 analog layers.
Converts target attention/FFN layers to analog; all other layers remain digital.

Based on mobilebert_squad_tiki.py, adapted for GLUE classification/regression tasks.
(TransferCompound with 2-device: A tile (LinearStepDevice/6T1C) + B tile (SoftBoundsDevice)).

Inline flags (edit directly in script):
    TASK_NAME = "sst2"               # GLUE task name
    N_EPOCHS = 3                     # Number of training epochs
    BATCH_SIZE = 64                  # Training batch size
    LEARNING_RATE = 0.00362          # Peak learning rate
    WEIGHT_DECAY = 0.0               # Weight decay
    WARMUP_RATIO = 0.1               # LR scheduler warmup ratio
    MIN_LR_RATE = 0.0                # Min LR as fraction of peak (0 = decay to zero)
    OPTIMIZER = "AnalogSGD"          # "AnalogSGD" | "AnalogAdam"
    TRANSFER_EVERY = 1000            # Transfer interval (steps)
    TRANSFER_LR = 1.0                # Transfer learning rate
    FAST_LR = 1.0                    # Fast tile learning rate
    LORA_TARGET = "qkv"             # Modules to convert to analog
"""

import os
import gc
import json

import torch
from torch import nn, no_grad, manual_seed, save
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs import UnitCellRPUConfig, IOParameters, UpdateParameters
from aihwkit.simulator.configs.compounds import TransferCompound
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType


# =============================================================================
# GLUE Task Configurations
# =============================================================================

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

TASK_TO_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
    "mnli": 3, "qnli": 2, "rte": 2, "stsb": 1,
}

TASK_TO_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "f1",
    "qqp": "f1",
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "stsb": "spearmanr",
}


# =============================================================================
# Global Constants
# =============================================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# GLUE task
TASK_NAME = "sst2"  # cola, sst2, mrpc, qqp, mnli, qnli, rte, stsb

# Paths
RESULTS = os.path.join(os.getcwd(), "results", "MOBILEBERT_GLUE_TIKI")
os.makedirs(RESULTS, exist_ok=True)

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128  # GLUE: 128 (SQuAD: 320)

# Training
N_EPOCHS = 3  # GLUE: 3 epochs (SQuAD: 15)
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64  # GLUE: 64 (SQuAD: 256)
LEARNING_RATE = 0.00362
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 3

# Scheduler
WARMUP_RATIO = 0.1  # GLUE: 10% of total steps (SQuAD: fixed 500 steps)
MIN_LR_RATE = 0.0  # Fraction of peak LR (0 = decay to zero)

# Optimizer
OPTIMIZER = "AnalogSGD"  # "AnalogSGD" or "AnalogAdam"

# TikiTaka v1 parameters
TRANSFER_EVERY = 1000
TRANSFER_LR = 1.0
FAST_LR = 1.0

# Target options: which layers to convert to analog
LORA_TARGET = "qkv"  # default
HEAD_LAYER = "train"  # "train" or "freeze" for classifier layer
LORA_TARGET_MODULES = {
    "none": [],
    "qonly": ["query"],
    "konly": ["key"],
    "vonly": ["value"],
    "qkv": ["query", "key", "value"],
    "ffn": ["dense"],
    "all": None,
}

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

# WandB
WANDB_PROJECT = "mobilebert-glue-tiki"
os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# TikiTaka v1 Device Functions
# =============================================================================

def _create_a_device():
    """Create A tile: 6T1C LinearStepDevice (fast, noisy).

    Identical to LRTT's A/B tile config with lifetime=0 (no retention decay).
    """
    return LinearStepDevice(
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
        lifetime=0.0,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )


def _create_b_device():
    """Create B tile: noise-free SoftBoundsDevice (slow, accurate).

    Identical to LRTT's C tile config.
    """
    return SoftBoundsDevice(
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
        mult_noise=False,
    )


def create_tikitaka_config():
    """Create TikiTaka v1 RPU configuration for analog layers.

    Uses TransferCompound with 2 devices:
        A tile (LinearStepDevice/6T1C) - fast, noisy accumulator
        B tile (SoftBoundsDevice) - slow, accurate storage
    """
    a_device = _create_a_device()
    b_device = _create_b_device()

    rpu_config = UnitCellRPUConfig(
        device=TransferCompound(
            unit_cell_devices=[a_device, b_device],
            transfer_every=TRANSFER_EVERY,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=TRANSFER_LR,
            fast_lr=FAST_LR,
            scale_transfer_lr=True,
            transfer_forward=IOParameters(
                noise_management=NoiseManagementType.NONE,
                bound_management=BoundManagementType.NONE,
            ),
            transfer_update=UpdateParameters(),
        )
    )

    # Forward/Backward IO: set out_noise to 0.0 (aihwkit default is 0.06)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0

    # Mapping
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = True
    rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


# =============================================================================
# Model Functions
# =============================================================================

def list_linear_layers(model):
    """List all linear layer names in the model."""
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_analog_layers(model):
    """Count analog layers in the model."""
    from aihwkit.nn import AnalogLinear
    return sum(1 for m in model.modules() if isinstance(m, AnalogLinear))


def get_target_module_names(lora_target):
    """Get module name patterns for analog conversion based on lora_target."""
    if lora_target == "none":
        return []
    elif lora_target == "qonly":
        return ["query"]
    elif lora_target == "konly":
        return ["key"]
    elif lora_target == "vonly":
        return ["value"]
    elif lora_target == "qkv":
        return ["query", "key", "value"]
    elif lora_target == "ffn":
        return ["dense"]
    elif lora_target == "all":
        return None
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model():
    """Create MobileBERT classification model with selective TikiTaka v1 analog layers.

    Architecture:
        - Target layers (based on LORA_TARGET) -> TikiTaka Analog
        - Non-target Encoder layers -> Digital FROZEN
        - classifier -> Digital TRAINABLE/FROZEN (based on HEAD_LAYER)
        - embedding_transformation -> Digital FROZEN (except "all" mode)
        - Embeddings -> Digital FROZEN

    Classifier is reinitialized with FIXED seed=42 for reproducibility.

    TikiTaka layers have:
        - A tile (fast): trained via gradient updates
        - B tile (slow): receives periodic transfers from A
        - out_scaling: TRAINABLE
        - bias: FROZEN (digital)
    """
    from aihwkit.nn import AnalogLinear

    num_labels = TASK_TO_NUM_LABELS[TASK_NAME]
    model_config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    # Reinitialize classifier with FIXED seed for reproducibility
    if hasattr(model, 'classifier'):
        torch.manual_seed(SEED)
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
        print(f"[FIX] Reinitialized classifier with FIXED seed={SEED}")

    # Get target patterns
    target_patterns = get_target_module_names(LORA_TARGET)

    def is_analog_target(layer_name):
        """Check if layer should be converted to TikiTaka Analog."""
        if "classifier" in layer_name:
            return False
        if "embedding_transformation" in layer_name:
            return (LORA_TARGET == "all")
        if "encoder" not in layer_name:
            return False
        if target_patterns is None:
            return True
        return any(p in layer_name for p in target_patterns)

    # Build exclude list: all layers that should NOT be converted
    all_linear_names = list_linear_layers(model)
    exclude_modules = []
    for name in all_linear_names:
        if not is_analog_target(name):
            exclude_modules.append(name)

    exclude_modules.append("classifier")
    if LORA_TARGET != "all":
        exclude_modules.append("mobilebert.embeddings.embedding_transformation")
    exclude_modules = list(set(exclude_modules))

    # Convert target layers to TikiTaka Analog (skip if none mode)
    if LORA_TARGET == "none":
        num_analog = 0
    else:
        tiki_config = create_tikitaka_config()
        model = convert_to_analog(model, tiki_config, exclude_modules=exclude_modules)
        num_analog = count_analog_layers(model)

    total_params = sum(p.numel() for p in model.parameters())

    # Set requires_grad
    # - out_scaling: TRAINABLE
    # - classifier: based on HEAD_LAYER
    # - Everything else: FROZEN (analog tile weights handled by analog optimizer)
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        elif "out_scaling" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_params = count_parameters(model)

    print(f"\nCreated MobileBERT model (TikiTaka v1, GLUE):")
    print(f"  Model: {MODEL_NAME}, Task: {TASK_NAME} (num_labels={num_labels})")
    print(f"  Total params: {total_params:,}, Trainable: {num_params:,}")
    print(f"  Analog layers: {num_analog}")
    print(f"  TikiTaka config: transfer_every={TRANSFER_EVERY}, "
          f"transfer_lr={TRANSFER_LR}, fast_lr={FAST_LR}")
    print(f"  Target: {LORA_TARGET} -> {target_patterns if target_patterns else 'all encoder layers'}")

    return model.to(DEVICE)


# =============================================================================
# Data Functions
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize GLUE dataset."""
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)
    sentence1_key, sentence2_key = TASK_TO_KEYS[TASK_NAME]

    def preprocess(examples):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True,
            )
        return tokenizer(
            examples[sentence1_key], examples[sentence2_key],
            padding="max_length", max_length=MAX_SEQ_LENGTH, truncation=True,
        )

    tokenized = raw_datasets.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    # Training set
    train_dataset = tokenized["train"]
    if TRAIN_SUBSET_SIZE > 0:
        train_dataset = train_dataset.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(train_dataset)))
        )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=default_data_collator,
        generator=torch.Generator().manual_seed(SEED),
    )

    # Eval set
    eval_key = "validation_matched" if TASK_NAME == "mnli" else "validation"
    eval_dataset = tokenized[eval_key]
    if EVAL_SUBSET_SIZE > 0:
        eval_dataset = eval_dataset.select(
            range(min(EVAL_SUBSET_SIZE, len(eval_dataset)))
        )

    eval_loader = DataLoader(
        eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=default_data_collator,
    )

    return train_loader, eval_loader


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_model(model, eval_loader):
    """Evaluate GLUE model. Returns (metric_value, avg_loss)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    is_regression = TASK_NAME == "stsb"
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    with no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            if is_regression:
                labels = labels.float()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze() if is_regression else outputs.logits
            loss = criterion(logits, labels)

            if is_regression:
                all_preds.extend(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            total_loss += loss.item() * labels.size(0)

    model.train()
    n_samples = len(all_labels)
    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

    # Compute task-specific metric
    if is_regression:
        from scipy.stats import spearmanr
        metric_value = spearmanr(all_preds, all_labels)[0]
    elif TASK_NAME in ["mrpc", "qqp"]:
        from sklearn.metrics import f1_score
        metric_value = f1_score(all_labels, all_preds)
    elif TASK_NAME == "cola":
        from sklearn.metrics import matthews_corrcoef
        metric_value = matthews_corrcoef(all_labels, all_preds)
    else:
        # accuracy for sst2, qnli, rte, mnli
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        metric_value = correct / n_samples if n_samples > 0 else 0.0

    return metric_value, avg_loss


# =============================================================================
# Optimizer & Scheduler
# =============================================================================

def create_optimizer(model):
    """Create optimizer. Uses standard PyTorch for none mode, Analog for TikiTaka modes."""
    if LORA_TARGET == "none":
        if OPTIMIZER == "AnalogSGD":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=LEARNING_RATE,
                weight_decay=0.0, momentum=0.0, nesterov=False
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
    else:
        if OPTIMIZER == "AnalogSGD":
            optimizer = AnalogSGD(
                model.parameters(), lr=LEARNING_RATE,
                weight_decay=0.0, momentum=0.0, nesterov=False
            )
        else:
            optimizer = AnalogAdam(
                model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
        optimizer.regroup_param_groups()

    return optimizer


def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps, min_lr_rate=0.0):
    """Linear schedule with warmup that decays to min_lr_rate (fraction of peak LR)."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Main
# =============================================================================

def main():
    """Train MobileBERT with TikiTaka v1 on GLUE."""
    manual_seed(SEED)
    set_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    metric_name = TASK_TO_METRIC[TASK_NAME]
    weight_path = os.path.join(RESULTS, f"mobilebert_glue_tiki_{TASK_NAME}_model_weight.pth")

    wandb.init(
        project=WANDB_PROJECT,
        name=f"mobilebert_tiki_{TASK_NAME}_te{TRANSFER_EVERY}_bs{BATCH_SIZE}",
        config={
            "model": MODEL_NAME, "dataset": f"GLUE/{TASK_NAME}",
            "task": TASK_NAME, "metric": metric_name,
            "transfer_every": TRANSFER_EVERY,
            "transfer_lr": TRANSFER_LR, "fast_lr": FAST_LR,
            "epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "optimizer": OPTIMIZER, "warmup_ratio": WARMUP_RATIO,
            "min_lr_rate": MIN_LR_RATE, "seed": SEED,
            "lora_target": LORA_TARGET, "head_layer": HEAD_LAYER,
            "max_seq_length": MAX_SEQ_LENGTH,
        }
    )

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_data(tokenizer)
    print(f"Task: {TASK_NAME}, Metric: {metric_name}")
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    # Create model, optimizer, scheduler
    model = create_model()
    optimizer = create_optimizer(model)

    num_training_steps = len(train_loader) * N_EPOCHS
    warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_rate=MIN_LR_RATE,
    )
    print(f"Total steps: {num_training_steps}, Warmup steps: {warmup_steps}")

    # Initial evaluation
    init_metric, init_loss = evaluate_model(model, eval_loader)
    wandb.log({"epoch": 0, f"eval/{metric_name}": init_metric, "eval/loss": init_loss})
    print(f"Initial eval: {metric_name}={init_metric:.4f}, loss={init_loss:.4f}")

    # Training loop
    best_metric = init_metric
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    is_regression = TASK_NAME == "stsb"
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    print(f"\nStarting training: {N_EPOCHS} epochs (max), early stopping patience={EARLY_STOP_PATIENCE}")

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            if is_regression:
                labels = labels.float()

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze() if is_regression else outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Evaluate
        eval_metric, eval_loss = evaluate_model(model, eval_loader)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch, "train/loss": train_loss,
            f"eval/{metric_name}": eval_metric, "eval/loss": eval_loss,
            "learning_rate": current_lr,
        })

        if eval_metric > best_metric:
            best_metric = eval_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save(model.state_dict(), weight_path)
        else:
            epochs_without_improvement += 1

        tqdm.write(
            f"Epoch {epoch}: Train Loss {train_loss:.4f} | "
            f"{metric_name} {eval_metric:.4f} | "
            f"Best {best_metric:.4f} | LR {current_lr:.2e} | "
            f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}"
        )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest {metric_name}: {best_metric:.4f} at epoch {best_epoch}")

    # Memory cleanup
    del model, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("GPU cache cleared")

    wandb.finish()


if __name__ == "__main__":
    main()
