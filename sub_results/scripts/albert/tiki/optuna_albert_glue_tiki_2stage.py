# -*- coding: utf-8 -*-
"""2-Stage Optuna HPO for ALBERT + GLUE with TikiTaka v1.

Stage 1: Pretrain classifier + LayerNorm (run pretrain_classifier.py first)
Stage 2: Sweep TikiTaka HPs with frozen classifier + LayerNorm (this script)

Requires: /data/classifier_ckpt/{task}/ckpt.pt from pretrain_classifier.py

This script automatically:
  - Loads pretrained classifier + LayerNorm from Stage 1 checkpoint
  - Freezes all digital params (classifier, LayerNorm)
  - Trains only TikiTaka analog tiles
  - Uses (total - stage1) epochs for Stage 2

Usage:
    python optuna_albert_glue_tiki_2stage.py --task sst2 --n-trials 50
    python optuna_albert_glue_tiki_2stage.py --task rte --n-trials 30
"""

import os
import sys
import json
import argparse
import gc
import math

import torch
from torch import nn, no_grad, manual_seed
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs import SingleRPUConfig, UnitCellRPUConfig, IOParameters, UpdateParameters
from aihwkit.simulator.configs.compounds import TransferCompound
from aihwkit.simulator.configs.utils import BoundManagementType, NoiseManagementType


# =============================================================================
# GLUE Task Configurations
# =============================================================================

GLUE_TASKS = ["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "stsb"]

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

# Per-task training configs from Albert_setup.txt (2x epoch condition)
TASK_CONFIGS = {
    "cola":  {"batch_size": 16,  "max_seq_length": 128, "epochs": 10},
    "stsb":  {"batch_size": 16,  "max_seq_length": 128, "epochs": 20},
    "sst2":  {"batch_size": 32,  "max_seq_length": 128, "epochs": 10},
    "mnli":  {"batch_size": 128, "max_seq_length": 128, "epochs": 4},
    "qnli":  {"batch_size": 32,  "max_seq_length": 128, "epochs": 11},
    "qqp":   {"batch_size": 128, "max_seq_length": 128, "epochs": 5},
    "rte":   {"batch_size": 32,  "max_seq_length": 256, "epochs": 11},
    "mrpc":  {"batch_size": 32,  "max_seq_length": 128, "epochs": 14},
}


# =============================================================================
# Global Constants
# =============================================================================

DEFAULT_STUDY_NAME = "albert_glue_tiki2s_main"

# 2-Stage config
PRETRAIN_CKPT_DIR = "/data/classifier_ckpt"
PRETRAIN_CKPT = None  # set via --pretrain-ckpt or auto-detected

# Per-task Stage-0/1 step budget from TS 2x schedule
# stage1_epochs = ceil((total_2x - stage0) / steps_per_epoch)
#   steps_per_epoch: cola=535, stsb=360, sst2=2105, mnli=3068,
#                    qnli=3274, qqp=2843, rte=78, mrpc=115
STAGE0_CONFIGS = {
    "cola":  {"stage0_steps": 2134,  "total_steps_2x": 10672, "stage1_epochs": 16},
    "stsb":  {"stage0_steps": 1439,  "total_steps_2x": 7196,  "stage1_epochs": 16},
    "sst2":  {"stage0_steps": 4187,  "total_steps_2x": 41870, "stage1_epochs": 18},
    "mnli":  {"stage0_steps": 2000,  "total_steps_2x": 20000, "stage1_epochs": 6},
    "qnli":  {"stage0_steps": 6622,  "total_steps_2x": 66224, "stage1_epochs": 19},
    "qqp":   {"stage0_steps": 2800,  "total_steps_2x": 28000, "stage1_epochs": 9},
    "rte":   {"stage0_steps": 320,   "total_steps_2x": 1600,  "stage1_epochs": 17},
    "mrpc":  {"stage0_steps": 320,   "total_steps_2x": 1600,  "stage1_epochs": 12},
}

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# GLUE task (set via --task)
TASK_NAME = "sst2"

# Paths
RESULTS = "/data/results/tikitakav1"
os.makedirs(RESULTS, exist_ok=True)

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "albert/albert-base-v2"
MAX_SEQ_LENGTH = 128  # Default; overridden per-task from TASK_CONFIGS

# Training defaults (overridden per-task from TASK_CONFIGS)
N_EPOCHS = 20
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64

# Scheduler
WARMUP_RATIO = 0.05  # GLUE: 5% of total steps

# Target options: which layers to convert to analog
# NOTE: ALBERT uses weight sharing — all 12 transformer blocks share the same
# parameters. The counts below are UNIQUE Linear layers, not per-block.
#   - Target layers -> TikiTaka v1 (TransferCompound: A + B tiles)
#   - Non-target encoder layers -> Single RPU (SoftBoundsDevice, same as B tile)
#   - classifier, embedding_hidden_mapping_in -> Digital (not converted)
LORA_TARGET = "attn"  # default, can be set via --lora-target
HEAD_LAYER = "train"  # default, can be set via --head-layer (train | freeze)
LORA_TARGET_MODULES = {
    "none": [],            # No TikiTaka layers; all encoder layers -> Single RPU
    "attn": ["attention"], # Attention (query/key/value/dense) -> TikiTaka; FFN -> Single RPU
    "ffn":  ["ffn"],       # FFN (ffn/ffn_output) -> TikiTaka; Attention -> Single RPU
    "all":  None,          # All encoder layers -> TikiTaka (no Single RPU)
}

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

# Global config (set by argparse)
OPT_CONFIG = {
    'optimizer': 'AnalogSGD',
    'tune_wd': False,        # weight_decay = 0 (fixed)
    'tune_momentum': False,  # momentum = 0 (fixed)
    'tune_nesterov': False,  # nesterov = False (fixed)
}

# Fixed learning rate (optimizer lr only scales transfer; redundant with transfer_lr)
FIXED_LR = 0.01

# te choices: units_in_mbatch=True, ALBERT 12 layers → te=12 means 1 transfer/step
# Auto-computed per task in main(); these are fallback defaults
TE_CHOICES = [12, 60, 120]

# TPE search ranges for continuous HPs
TPE_FLR_RANGE = (0.01, 10.0)     # fast_lr: log-uniform
TPE_TLR_RANGE = (0.01, 1.0)      # transfer_lr: log-uniform

# Out scaling lr (fixed, not swept — independent of learning_rate)
OUT_SCALING_LR = 0.01


def get_study_name_suffix():
    """Generate study name suffix based on optimizer config."""
    opt = OPT_CONFIG['optimizer'].lower().replace('analog', '')
    suffix = opt

    if not OPT_CONFIG['tune_wd']:
        suffix += "_nowd"
    if not OPT_CONFIG['tune_momentum']:
        suffix += "_nomom"
    if not OPT_CONFIG['tune_nesterov']:
        suffix += "_nonest"

    # Add lora target (always include for clarity)
    suffix += f"_{LORA_TARGET}"

    # Add head_layer if frozen (not default)
    if HEAD_LAYER == "freeze":
        suffix += "_headfreeze"

    return suffix

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


def create_tikitaka_config(transfer_every, transfer_lr, fast_lr):
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
            transfer_every=transfer_every,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_columns=True,
            gamma=0.0,
            transfer_lr=transfer_lr,
            fast_lr=fast_lr,
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


def create_single_rpu_config():
    """Create Single RPU configuration for non-target frozen analog layers.

    Uses the same SoftBoundsDevice as TikiTaka's B tile (slow, accurate).
    Tile weights are frozen via noop update hook (see create_model).
    """
    b_device = _create_b_device()

    rpu_config = SingleRPUConfig(device=b_device)

    # IO settings: identical to TikiTaka config
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0

    # Mapping: frozen analog, no out_scaling learning
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = False
    rpu_config.mapping.out_scaling_columnwise = False

    return rpu_config


# =============================================================================
# Model Functions
# =============================================================================

def list_linear_layers(model):
    return [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def get_target_module_names(lora_target):
    """Get module name patterns for TikiTaka analog conversion based on lora_target.

    Returns list of substrings that identify which encoder layers get TikiTaka config.
    Non-target encoder layers get Single RPU (SoftBoundsDevice) config instead.

    ALBERT layer naming:
        attention: query, key, value, dense (output projection)
        FFN: ffn (intermediate), ffn_output (output)
        embedding projection: albert.encoder.embedding_hidden_mapping_in
    """
    if lora_target == "none":
        return []
    elif lora_target == "attn":
        return ["attention"]
    elif lora_target == "ffn":
        return ["ffn"]
    elif lora_target == "all":
        return None
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model(params):
    """Create ALBERT classification model with selective analog layers.

    Architecture:
        - Target encoder layers (--lora-target) -> TikiTaka v1 (TransferCompound)
        - Non-target encoder layers -> Single RPU (SoftBoundsDevice = B tile)
        - classifier -> Digital TRAINABLE (based on HEAD_LAYER)
        - embedding_hidden_mapping_in -> Digital FROZEN
        - Embeddings -> Digital FROZEN

    ALBERT uses weight sharing: all transformer blocks share the same
    parameters, so the actual number of unique analog layers is small.
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
        print(f"  [FIX] Reinitialized classifier with FIXED seed={SEED}")

    # Load pretrained classifier + LayerNorm from Stage 1
    ckpt_path = PRETRAIN_CKPT or os.path.join(PRETRAIN_CKPT_DIR, TASK_NAME, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {ckpt_path}\n"
            f"Run: python /data/pretrain_classifier.py --tasks {TASK_NAME}"
        )
    ckpt = torch.load(ckpt_path, map_location='cpu')
    pretrained = ckpt['state_dict']
    current_state = model.state_dict()
    loaded_keys = []
    for k, v in pretrained.items():
        if k in current_state:
            current_state[k] = v
            loaded_keys.append(k)
        else:
            print(f"  [WARN] Checkpoint key not in model: {k}")
    model.load_state_dict(current_state)
    print(f"  [Stage 1] Loaded {len(loaded_keys)} params from {ckpt_path}")
    print(f"  [Stage 1] {ckpt.get('metric_name','?')}: {ckpt.get('metric_value', 0):.4f} "
          f"(epoch {ckpt.get('best_epoch', '?')}, lr={ckpt.get('lr', '?')})")

    # Get target patterns for TikiTaka conversion
    target_patterns = get_target_module_names(LORA_TARGET)

    # Always exclude from any analog conversion
    always_digital = ["classifier", "albert.encoder.embedding_hidden_mapping_in"]

    def is_tikitaka_target(layer_name):
        """Check if layer should be converted to TikiTaka Analog."""
        if any(d in layer_name for d in always_digital):
            return False
        if "encoder" not in layer_name:
            return False
        if target_patterns is None:
            return True
        return any(p in layer_name for p in target_patterns)

    all_linear_names = list_linear_layers(model)

    # Classify layers
    tikitaka_layers = [n for n in all_linear_names if is_tikitaka_target(n)]
    non_target_encoder_layers = [
        n for n in all_linear_names
        if n not in tikitaka_layers and "encoder" in n
        and not any(d in n for d in always_digital)
    ]

    # --- Pass 1: Convert target layers to TikiTaka ---
    tikitaka_count = 0
    if tikitaka_layers:
        tiki_config = create_tikitaka_config(
            transfer_every=int(params["transfer_every"]),
            transfer_lr=params["transfer_lr"],
            fast_lr=params["fast_lr"],
        )
        tiki_exclude = [n for n in all_linear_names if n not in tikitaka_layers]
        model = convert_to_analog(model, tiki_config, exclude_modules=tiki_exclude)
        tikitaka_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))

    # --- Pass 2: Convert non-target encoder layers to frozen analog (Single RPU) ---
    single_rpu_count = 0
    if non_target_encoder_layers:
        single_config = create_single_rpu_config()
        single_exclude = [n for n in all_linear_names if n not in non_target_encoder_layers]
        model = convert_to_analog(model, single_config, exclude_modules=single_exclude)
        single_rpu_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear)) - tikitaka_count

        # Freeze Single RPU tile weights via noop update hook
        # (same as LRTT's --encoder-analog: analog forward only, no weight update)
        # Use rpu_config type to distinguish: SingleRPUConfig = frozen, UnitCellRPUConfig = TikiTaka
        def _frozen_noop_update(x_input, d_input, *args, **kwargs):
            return None
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    if isinstance(tile.rpu_config, SingleRPUConfig):
                        tile.update = _frozen_noop_update

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  TikiTaka layers: {tikitaka_count}, Frozen analog layers: {single_rpu_count}, "
          f"Total analog: {tikitaka_count + single_rpu_count}, Total params: {total_params:,}")

    # Stage 2: only TikiTaka tiles trainable (via AnalogContext)
    # Classifier + LayerNorm frozen (pretrained in Stage 1)
    from aihwkit.optim.context import AnalogContext
    # TikiTaka layer name patterns for out_scaling filtering
    tiki_patterns = [t.split(".")[-1] for t in tikitaka_layers]  # e.g. query, key, value, dense
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            param.requires_grad = True  # required for analog tile update
        elif "out_scaling" in name and any(p in name for p in tiki_patterns):
            # learn_out_scaling only for TikiTaka target layers
            param.requires_grad = True
        else:
            param.requires_grad = False  # all digital params frozen

    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable_after:,}")
    print(f"  Target: {LORA_TARGET} -> TikiTaka: {tikitaka_layers}, Single RPU: {non_target_encoder_layers}")

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
                max_length=MAX_SEQ_LENGTH, truncation=True,
            )
        return tokenizer(
            examples[sentence1_key], examples[sentence2_key],
            max_length=MAX_SEQ_LENGTH, truncation=True,
        )

    # Remove original text columns (keep only tokenizer output + label)
    remove_cols = [c for c in raw_datasets["train"].column_names if c != "label"]
    tokenized = raw_datasets.map(preprocess, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")

    # Training set
    train_dataset = tokenized["train"]
    if TRAIN_SUBSET_SIZE > 0:
        train_dataset = train_dataset.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(train_dataset)))
        )

    data_collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=data_collator,
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
        collate_fn=data_collator,
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
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        metric_value = correct / n_samples if n_samples > 0 else 0.0

    return metric_value, avg_loss


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """Adaptive early stopping with warmup protection and overfitting detection.

    Criteria for stopping (any one triggers):
      1. Patience exhausted: no improvement >= min_delta for `patience` epochs
      2. Overfitting: eval_loss increases for `overfit_window` consecutive epochs
         while train_loss decreases

    Warmup protection: no stopping in first `warmup_epochs` epochs.

    Patience and warmup scale with total epochs:
      patience = max(3, N_EPOCHS // 5)
      warmup_epochs = ceil(N_EPOCHS * warmup_ratio)  — matches LR scheduler warmup
    """

    def __init__(self, n_epochs, warmup_ratio=0.05, min_delta=0.001):
        import math
        self.patience = max(3, n_epochs // 5)
        self.warmup_epochs = max(3, math.ceil(n_epochs * warmup_ratio))
        self.overfit_window = 3
        self.min_delta = min_delta

        self.best_metric = -float('inf')
        self.epochs_no_improve = 0
        self.eval_loss_history = []
        self.train_loss_history = []
        self.stop_reason = None

    def step(self, epoch, eval_metric, eval_loss, train_loss):
        """Returns True if training should stop."""
        self.eval_loss_history.append(eval_loss)
        self.train_loss_history.append(train_loss)

        # Check improvement
        improved = False
        if eval_metric > self.best_metric + self.min_delta:
            self.best_metric = eval_metric
            self.epochs_no_improve = 0
            improved = True
        else:
            self.epochs_no_improve += 1

        # Warmup protection
        if epoch <= self.warmup_epochs:
            return False

        # 1) Patience check
        if self.epochs_no_improve >= self.patience:
            self.stop_reason = (f"patience exhausted ({self.epochs_no_improve} epochs "
                                f"without >{self.min_delta} improvement)")
            return True

        # 2) Overfitting check: eval_loss increasing + train_loss decreasing
        n = self.overfit_window
        if len(self.eval_loss_history) >= n + 1:
            recent_eval = self.eval_loss_history[-n:]
            recent_train = self.train_loss_history[-n:]
            prev_eval = self.eval_loss_history[-(n + 1)]
            prev_train = self.train_loss_history[-(n + 1)]

            eval_increasing = all(
                recent_eval[i] > recent_eval[i - 1] if i > 0 else recent_eval[0] > prev_eval
                for i in range(n)
            )
            train_decreasing = all(
                recent_train[i] < recent_train[i - 1] if i > 0 else recent_train[0] < prev_train
                for i in range(n)
            )
            if eval_increasing and train_decreasing:
                self.stop_reason = (f"overfitting detected (eval_loss increasing + "
                                    f"train_loss decreasing for {n} consecutive epochs)")
                return True

        return False

    def status_str(self):
        return f"No imp: {self.epochs_no_improve}/{self.patience}"


# =============================================================================
# Scheduler
# =============================================================================

def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps,
                                    min_lr_rate=0.0, constant_lr_groups=None):
    """Linear schedule with warmup that decays to min_lr_rate (fraction of peak LR).

    Args:
        constant_lr_groups: set of param group indices that keep constant lr
                           (no warmup/decay). Used for out_scaling params.
    """
    def schedule_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))

    constant_lr_groups = constant_lr_groups or set()
    lr_lambdas = []
    for i in range(len(optimizer.param_groups)):
        if i in constant_lr_groups:
            lr_lambdas.append(lambda step: 1.0)  # constant lr
        else:
            lr_lambdas.append(schedule_lambda)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)


# =============================================================================
# Optuna Objective
# =============================================================================

def _fix_out_scaling_lr(optimizer, model, out_scaling_lr):
    """Separate out_scaling params into their own param group with fixed lr.

    Called after optimizer.regroup_param_groups() to ensure learn_out_scaling
    params use a fixed lr independent of the swept learning_rate.

    Returns set of out_scaling param group indices (for scheduler constant_lr_groups).
    """
    out_scaling_ids = {id(p) for n, p in model.named_parameters()
                       if "out_scaling" in n and p.requires_grad}
    if not out_scaling_ids:
        return set()

    groups_to_add = []
    for pg in optimizer.param_groups:
        os_params = [p for p in pg['params'] if id(p) in out_scaling_ids]
        other_params = [p for p in pg['params'] if id(p) not in out_scaling_ids]

        if not os_params:
            continue

        if other_params:
            # Mixed group: keep non-os params, split os to new group
            pg['params'] = other_params
            new_pg = {k: v for k, v in pg.items() if k != 'params'}
            new_pg['params'] = os_params
            new_pg['lr'] = out_scaling_lr
            groups_to_add.append(new_pg)
        else:
            # Pure out_scaling group: just fix lr
            pg['lr'] = out_scaling_lr

    for new_pg in groups_to_add:
        optimizer.add_param_group(new_pg)

    # Return indices of out_scaling groups
    os_group_indices = set()
    for i, pg in enumerate(optimizer.param_groups):
        if any(id(p) in out_scaling_ids for p in pg['params']):
            os_group_indices.add(i)

    print(f"  [OutScaling] Fixed lr={out_scaling_lr:.2e} for "
          f"{len(out_scaling_ids)} params ({len(os_group_indices)} groups)")
    return os_group_indices


def objective(trial, train_loader, eval_loader):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metric_name = TASK_TO_METRIC[TASK_NAME]

    learning_rate = FIXED_LR  # fixed; transfer_lr sweeps the effective transfer rate

    # te: categorical (ALBERT layer-aware choices), fast_lr/transfer_lr: TPE continuous
    transfer_every = trial.suggest_categorical('transfer_every', TE_CHOICES)
    fast_lr = trial.suggest_float('fast_lr', *TPE_FLR_RANGE, log=True)
    transfer_lr = trial.suggest_float('transfer_lr', *TPE_TLR_RANGE, log=True)
    min_lr_rate = 0.0

    # weight_decay: tune or fix to 0
    if OPT_CONFIG['tune_wd']:
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-2, log=True)
    else:
        weight_decay = 0.0

    # momentum: 0.9 fixed by default, 0.0 with --no-momentum
    if OPT_CONFIG['tune_momentum']:
        momentum = 0.9
    else:
        momentum = 0.0

    # nesterov: True fixed by default, False with --no-nesterov
    if OPT_CONFIG['tune_nesterov'] and momentum > 0:
        nesterov = True
    else:
        nesterov = False

    # optimizer: always use config value
    optimizer_name = OPT_CONFIG['optimizer']

    params = {
        "transfer_every": transfer_every,
        "transfer_lr": transfer_lr,
        "fast_lr": fast_lr,
    }

    print(f"\n{'='*70}")
    print(f"Trial {trial.number} Starting ({TASK_NAME}, metric={metric_name})")
    print(f"{'='*70}")
    print(f"  transfer_every={transfer_every}, transfer_lr={transfer_lr:.4e}, fast_lr={fast_lr:.4e}")
    print(f"  lr={learning_rate:.2e} (fixed), wd={weight_decay:.2e}, out_scaling_lr={OUT_SCALING_LR:.2e}")
    print(f"  momentum={momentum:.2f}, nesterov={nesterov}, optimizer={optimizer_name}")
    print(f"  min_lr_rate={min_lr_rate:.4f}, warmup_ratio={WARMUP_RATIO}")
    print(f"{'='*70}")

    is_regression = TASK_NAME == "stsb"
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    model = None
    try:
        set_seed(SEED)

        model = create_model(params)

        # All modes use Analog optimizers (all encoder layers are analog)
        if optimizer_name == "AnalogSGD":
            optimizer = AnalogSGD(
                model.parameters(), lr=learning_rate,
                weight_decay=weight_decay, momentum=momentum, nesterov=nesterov,
            )
        else:
            optimizer = AnalogAdam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay,
            )
        optimizer.regroup_param_groups()

        # Separate out_scaling params with fixed lr (scheduled with warmup/decay)
        os_group_indices = _fix_out_scaling_lr(optimizer, model, OUT_SCALING_LR)

        num_training_steps = len(train_loader) * N_EPOCHS
        warmup_steps = int(num_training_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_min_lr(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_rate=min_lr_rate,
        )

        # Load Stage 0 pretrained metric as baseline
        ckpt_path = PRETRAIN_CKPT or os.path.join(PRETRAIN_CKPT_DIR, TASK_NAME, "ckpt.pt")
        ckpt_meta = torch.load(ckpt_path, map_location='cpu')
        stage0_metric = ckpt_meta.get('metric_value', -float('inf'))
        print(f"  Stage 0 baseline {metric_name}: {stage0_metric:.4f}")

        early_stop = EarlyStopping(N_EPOCHS, warmup_ratio=WARMUP_RATIO, min_delta=0.001)
        print(f"  EarlyStopping: patience={early_stop.patience}, "
              f"warmup_epochs={early_stop.warmup_epochs}, "
              f"overfit_window={early_stop.overfit_window}, "
              f"min_delta={early_stop.min_delta}")

        for epoch in range(1, N_EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Trial {trial.number} Ep{epoch}", leave=False)
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

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            eval_metric, eval_loss = evaluate_model(model, eval_loader)

            improved = " *" if eval_metric > early_stop.best_metric + early_stop.min_delta else ""
            should_stop = early_stop.step(epoch, eval_metric, eval_loss, train_loss)

            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"[Trial {trial.number}] Epoch {epoch:3d} | "
                  f"{metric_name}: {eval_metric:.4f} | Best: {early_stop.best_metric:.4f} | "
                  f"Loss: {train_loss:.4f} | LR: {current_lr:.2e} | "
                  f"{early_stop.status_str()}{improved}")

            trial.report(eval_metric, epoch)
            trial.set_user_attr(f"train_loss_epoch_{epoch}", train_loss)

            # Stop if epoch 3 metric is worse than Stage 0 pretrained baseline
            if epoch == 3 and eval_metric < stage0_metric:
                tqdm.write(f"[Trial {trial.number}] Stopped: Ep3 {metric_name}={eval_metric:.4f} "
                           f"< Stage0 baseline={stage0_metric:.4f}")
                raise optuna.exceptions.TrialPruned()

            if should_stop:
                tqdm.write(f"[Trial {trial.number}] Early stopping at epoch {epoch}: "
                           f"{early_stop.stop_reason}")
                break

            if trial.should_prune():
                tqdm.write(f"[Trial {trial.number}] Pruned at epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

        print(f"\n[Trial {trial.number}] Finished - Best {metric_name}: {early_stop.best_metric:.4f}")
        print(f"{'='*70}\n")
        return early_stop.best_metric

    except Exception as e:
        error_msg = str(e)[:500]
        trial.set_user_attr("error", error_msg)
        print(f"[Trial {trial.number}] Error: {error_msg}")
        raise

    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[Trial {trial.number}] GPU cache cleared")


# =============================================================================
# Visualization
# =============================================================================

def visualize_study(study, save_dir):
    """Visualize optimization history, parameter importance, and LR vs metric."""
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not complete_trials:
        print("No completed trials to visualize.")
        return

    metric_name = TASK_TO_METRIC[TASK_NAME]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    trial_numbers = [t.number for t in complete_trials]
    scores = [t.value for t in complete_trials]

    axes[0].scatter(trial_numbers, scores, alpha=0.6)
    axes[0].plot(trial_numbers,
                 [max(scores[:i+1]) for i in range(len(scores))],
                 'r-', linewidth=2, label='Best so far')
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel(metric_name)
    axes[0].set_title('Optimization History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    try:
        importances = optuna.importance.get_param_importances(study)
        axes[1].barh(list(importances.keys())[::-1], list(importances.values())[::-1])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Parameter Importance')
    except Exception:
        axes[1].text(0.5, 0.5, 'Not enough trials', ha='center', va='center',
                     transform=axes[1].transAxes)

    lrs = [t.params.get('learning_rate', 1e-4) for t in complete_trials]
    axes[2].scatter(lrs, scores, alpha=0.6)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel(metric_name)
    axes[2].set_title(f'Learning Rate vs {metric_name}')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"visualization_{TASK_NAME}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Visualization saved.")


def print_study_summary(study):
    """Print study summary."""
    metric_name = TASK_TO_METRIC[TASK_NAME]
    print("\n" + "=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"Study: {study.study_name}, Task: {TASK_NAME}, Trials: {len(study.trials)} ({len(complete_trials)} complete)")
    if complete_trials:
        scores = [t.value for t in complete_trials]
        print(f"Best {metric_name}: {max(scores):.4f}, Mean: {sum(scores)/len(scores):.4f}")
        print(f"Best params: {study.best_params}")


# =============================================================================
# Main
# =============================================================================

def main():
    global BATCH_SIZE, N_EPOCHS, MAX_SEQ_LENGTH, WARMUP_RATIO, LORA_TARGET, HEAD_LAYER, TASK_NAME, PRETRAIN_CKPT, OUT_SCALING_LR

    parser = argparse.ArgumentParser(description="Optuna sweep for ALBERT GLUE TikiTaka v1")
    parser.add_argument('--task', type=str, default=TASK_NAME,
                        choices=GLUE_TASKS,
                        help=f'GLUE task (default: {TASK_NAME})')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (default: auto-generated based on config)')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--optimizer', type=str, default='AnalogAdam',
                        choices=['AnalogSGD', 'AnalogAdam'],
                        help='Optimizer type (default: AnalogAdam)')
    parser.add_argument('--no-wd', action='store_true',
                        help='Disable weight decay tuning (fix to 0)')
    parser.add_argument('--no-momentum', action='store_true',
                        help='Disable momentum tuning (fix to 0, SGD only)')
    parser.add_argument('--no-nesterov', action='store_true',
                        help='Disable nesterov tuning (fix to False, SGD only)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: per-task from TASK_CONFIGS)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: per-task from TASK_CONFIGS)')
    parser.add_argument('--warmup-ratio', type=float, default=WARMUP_RATIO,
                        help=f'LR warmup ratio (default: {WARMUP_RATIO})')
    parser.add_argument('--lora-target', type=str, default=LORA_TARGET,
                        choices=['none', 'attn', 'ffn', 'all'],
                        help='Target: none, attn, ffn, all (default: attn)')
    parser.add_argument('--head-layer', type=str, default=HEAD_LAYER,
                        choices=['train', 'freeze'],
                        help='classifier layer: train or freeze (default: train)')
    parser.add_argument('--te-choices', type=int, nargs='+', default=None,
                        help='Override transfer_every choices (e.g. --te-choices 12 60 120)')
    parser.add_argument('--flr-range', type=float, nargs=2, default=None,
                        help='fast_lr TPE range (default: 0.01 10.0)')
    parser.add_argument('--tlr-range', type=float, nargs=2, default=None,
                        help='transfer_lr TPE range (default: 0.01 1.0)')
    parser.add_argument('--out-scaling-lr', type=float, default=None,
                        help=f'Fixed lr for learn_out_scaling params (default: {OUT_SCALING_LR})')
    parser.add_argument('--enqueue', type=float, nargs='+', default=None,
                        metavar='VAL',
                        help='Enqueue first trial: TE FLR [TLR] (e.g. --enqueue 12 1.0 0.5)')
    parser.add_argument('--pretrain-ckpt', type=str, default=None,
                        help='Path to Stage 1 checkpoint (default: /data/classifier_ckpt/{task}/ckpt.pt)')
    args = parser.parse_args()

    # Update global config
    TASK_NAME = args.task

    # Apply per-task config
    task_cfg = TASK_CONFIGS[TASK_NAME]
    MAX_SEQ_LENGTH = task_cfg["max_seq_length"]
    BATCH_SIZE = args.batch_size if args.batch_size is not None else task_cfg["batch_size"]

    # Stage 1 epochs from precomputed STAGE0_CONFIGS (or override with --epochs)
    s0_cfg = STAGE0_CONFIGS.get(TASK_NAME, {})
    if args.epochs is not None:
        N_EPOCHS = args.epochs
    else:
        N_EPOCHS = s0_cfg.get("stage1_epochs", task_cfg["epochs"])

    WARMUP_RATIO = args.warmup_ratio
    PRETRAIN_CKPT = args.pretrain_ckpt
    LORA_TARGET = args.lora_target
    HEAD_LAYER = args.head_layer
    OPT_CONFIG['optimizer'] = args.optimizer
    OPT_CONFIG['tune_wd'] = not args.no_wd
    OPT_CONFIG['tune_momentum'] = not args.no_momentum
    OPT_CONFIG['tune_nesterov'] = not args.no_nesterov
    if args.out_scaling_lr is not None:
        OUT_SCALING_LR = args.out_scaling_lr

    stage0_steps = s0_cfg.get('stage0_steps', 0)
    total_2x = s0_cfg.get('total_steps_2x', 0)
    print(f"[2-Stage] stage0={stage0_steps} steps, "
          f"stage1={total_2x - stage0_steps} steps -> {N_EPOCHS} epochs")
    print(f"Per-task config: {TASK_NAME} -> BSZ={BATCH_SIZE}, epochs={N_EPOCHS}, "
          f"max_seq={MAX_SEQ_LENGTH}, padding=dynamic")

    # Auto-generate study name based on config (includes task and batch size)
    study_name = args.study_name or f"albert_glue_tiki2s_{TASK_NAME}_bs{BATCH_SIZE}_{get_study_name_suffix()}"

    storage = f"sqlite:///{RESULTS}/optuna_{study_name}.db"

    if args.visualize:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print_study_summary(study)
        visualize_study(study, RESULTS)
        return

    # Load data once (shared across all trials)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_loader = load_data(tokenizer)
    print(f"Task: {TASK_NAME}, Metric: {TASK_TO_METRIC[TASK_NAME]}")
    print(f"Train batches: {len(train_loader)}, Eval batches: {len(eval_loader)}")

    global TE_CHOICES, TPE_FLR_RANGE, TPE_TLR_RANGE

    # Compute te values based on ALBERT weight sharing (12 layers)
    # units_in_mbatch=True: m_batch cancels, te = number of update() calls between transfers
    # ALBERT: 12 update() calls per SGD step (weight sharing)
    # te=12 → 1 transfer/step, te=12*N → 1 transfer/N steps
    ALBERT_LAYERS = 12
    steps_per_epoch = len(train_loader)
    # Ensure at least 1 full transfer (768 cols) during training
    total_steps = steps_per_epoch * N_EPOCHS
    max_te = max(ALBERT_LAYERS, (total_steps * ALBERT_LAYERS) // 768)
    auto_te = sorted(set([
        ALBERT_LAYERS,                                          # 1 transfer/step
        ALBERT_LAYERS * 5,                                      # 1 transfer/5 steps
        min(ALBERT_LAYERS * 10, max_te),                        # 1 transfer/10 steps (capped)
    ]))
    print(f"Steps/epoch: {steps_per_epoch}, ALBERT layers: {ALBERT_LAYERS}")
    print(f"  auto te: {auto_te} -> [{steps_per_epoch}/{auto_te[0]//ALBERT_LAYERS}="
          f"{steps_per_epoch*ALBERT_LAYERS//auto_te[0]}col/ep, "
          f"{steps_per_epoch*ALBERT_LAYERS//auto_te[1]}col/ep, "
          f"{steps_per_epoch*ALBERT_LAYERS//auto_te[2]}col/ep]")

    # Apply CLI overrides
    TE_CHOICES = args.te_choices if args.te_choices else auto_te
    if args.flr_range:
        TPE_FLR_RANGE = tuple(args.flr_range)
    if args.tlr_range:
        TPE_TLR_RANGE = tuple(args.tlr_range)

    # Always use TPE sampler: te=categorical, fast_lr/transfer_lr=continuous
    sampler = optuna.samplers.TPESampler(seed=SEED)
    print(f"Sampler: TPE | te(categorical)={TE_CHOICES} | "
          f"fast_lr(log)={TPE_FLR_RANGE} | transfer_lr(log)={TPE_TLR_RANGE}")
    print(f"Fixed lr: {FIXED_LR:.2e}, Out scaling lr: {OUT_SCALING_LR:.2e}")

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.PercentilePruner(
            percentile=0.0,       # only keep trials >= best (top 0%)
            n_startup_trials=1,   # start pruning after 1st completed trial
            n_warmup_steps=3,     # judge from epoch 3
        ),
        load_if_exists=True,
    )

    # Enqueue seed trial (best params as starting point)
    # Format: --enqueue TE FLR TLR (e.g. --enqueue 12 1.0 0.5)
    if args.enqueue and len(study.trials) == 0:
        eq_params = {
            'transfer_every': int(args.enqueue[0]),
            'fast_lr': args.enqueue[1],
            'transfer_lr': args.enqueue[2] if len(args.enqueue) >= 3 else 1.0,
        }
        study.enqueue_trial(eq_params)
        print(f"Enqueued seed trial: {eq_params}")

    print(f"\nStudy: {study_name}, Device: {DEVICE}, New trials: {args.n_trials}")

    # Run trials with OOM recovery via process restart
    target_total = len(study.trials) + args.n_trials

    try:
        study.optimize(
            lambda trial: objective(trial, train_loader, eval_loader),
            n_trials=args.n_trials,
            catch=(Exception,),
            show_progress_bar=False,
            callbacks=[_oom_restart_callback],
        )
    except _OOMRestart:
        remaining = target_total - len(study.trials)
        if remaining > 0:
            print(f"\n[OOM Recovery] Restarting process for {remaining} remaining trials...")
            new_argv = list(sys.argv)
            for i, arg in enumerate(new_argv):
                if arg == '--n-trials' and i + 1 < len(new_argv):
                    new_argv[i + 1] = str(remaining)
                    break
            os.execv(sys.executable, [sys.executable] + new_argv)

    print_study_summary(study)
    visualize_study(study, RESULTS)

    # Save best params
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if complete_trials:
        best_params_file = os.path.join(RESULTS, f"best_params_{study_name}.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                "task": TASK_NAME,
                "metric": TASK_TO_METRIC[TASK_NAME],
                "best_value": study.best_value,
                "best_params": study.best_params,
            }, f, indent=2)
        print(f"Best params saved to: {best_params_file}")

    # Save all trials
    all_trials = []
    for t in study.trials:
        all_trials.append({
            "trial": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        })
    all_trials.sort(key=lambda x: x["value"] if x["value"] is not None else -float('inf'), reverse=True)

    all_trials_file = os.path.join(RESULTS, f"all_trials_2s_{TASK_NAME}.json")
    with open(all_trials_file, 'w') as f:
        json.dump(all_trials, f, indent=2)
    print(f"All trials saved to: {all_trials_file}")


class _OOMRestart(Exception):
    """Raised to trigger process restart after OOM."""
    pass


def _oom_restart_callback(study, trial):
    """Optuna callback: if trial failed with OOM/CUBLAS, raise to restart process."""
    if trial.state == TrialState.FAIL:
        err = trial.user_attrs.get("error", "")
        if "out of memory" in err.lower() or "cublas" in err.lower():
            print(f"\n[OOM Recovery] Trial {trial.number} failed with CUDA error, will restart process.")
            raise _OOMRestart()


if __name__ == "__main__":
    main()
