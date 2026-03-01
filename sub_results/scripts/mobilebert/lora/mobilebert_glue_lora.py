# -*- coding: utf-8 -*-
"""MobileBERT + GLUE with LoRA-LRTT (LoRA mode using LRTT analog devices).

Single-run training script for MobileBERT on GLUE tasks using LoRA-style forward:
  y = C·x + α·A·(B·x)
where C is frozen (pretrained weights), A/B are trainable analog tiles.

Difference from LRTT: forward_inject=True, transfer_every=10^7 (no transfer).
Based on mobilebert_squad_lora.py, adapted for GLUE tasks.

Inline flags (edit directly in script):
    N_EPOCHS = 3                    # Number of training epochs (GLUE)
    BATCH_SIZE = 64                 # Training batch size
    LEARNING_RATE = 0.00362         # Peak learning rate
    WEIGHT_DECAY = 0.0              # Weight decay
    WARMUP_RATIO = 0.1              # LR scheduler warmup ratio (GLUE: 10%)
    MIN_LR_RATE = 0.0               # Min LR as fraction of peak (0 = decay to zero)
    OPTIMIZER = "AnalogSGD"         # "AnalogSGD" | "AnalogAdam"
    LRTT_RANK = 8                   # LoRA rank for LRTT
    TRANSFER_EVERY = 1000           # Transfer interval (steps)
    TRANSFER_LR = 0.00115           # Transfer learning rate
    TRANSFER_METHOD = "onehot"      # Transfer method: "onehot" | "direct" | "set"
    LORA_ALPHA = 1.0                # LoRA alpha scaling
    REINIT_MODE = "hybrid"          # Reinit mode: "standard" | "decay" | "hybrid"
    REINIT_GAIN = 0.1               # Reinitialization gain
    DECAY_FACTOR = 1.0              # Decay factor for reinit
    TAU_SEC = 0.0                   # 6T1C retention (0 = no decay)
    DYNAMIC_TE = False              # Enable dynamic transfer every
    DYNAMIC_TE_POWER = 1.0          # Power for dynamic TE scaling
    TE_WARMUP_STEPS = 0            # Steps before reaching target TE
    TE_WARMUP_SCHEDULE = []         # Warmup TE schedule list
    TARGET_MODULES = [...]          # Modules to convert to analog
"""

import os
import sys
import re
import string
import math
import gc
import collections

import json

import torch
from torch import nn, no_grad, manual_seed, save
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
import evaluate

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice

# LRTT config imports (direct imports to avoid __init__.py dependency issues)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice

from collections import Counter


# =============================================================================
# Global Constants
# =============================================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
TASK_NAME = "sst2"  # GLUE task (sst2, mrpc, cola, etc.)
RESULTS = os.path.join(os.getcwd(), "results", f"MOBILEBERT_GLUE_{TASK_NAME.upper()}_LORA")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, f"mobilebert_glue_{TASK_NAME}_lora_model_weight.pth")

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 128  # GLUE standard

# Training
N_EPOCHS = 5
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64  # GLUE standard (SQuAD uses 256)
LEARNING_RATE = 0.9589
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 3

# Scheduler
WARMUP_RATIO = 0.1  # GLUE: 10% of total steps (SQuAD uses fixed 500 steps)
MIN_LR_RATE = 0.0  # Fraction of peak LR (0 = decay to zero)

# Optimizer
OPTIMIZER = "AnalogSGD"  # "AnalogSGD" or "AnalogAdam"

# LoRA-LRTT parameters (forward_inject=True, no transfer)
LRTT_RANK = 8
TRANSFER_EVERY = 10000000  # Effectively disabled (no transfer)
TRANSFER_LR = 0.1  # Not used when transfer disabled
LORA_ALPHA = 0.00942  # LoRA alpha scaling (forward: y = C·x + α·A·(B·x))
LRTT_LR_MULTIPLIER = 2.703  # LRTT lr = LEARNING_RATE * LRTT_LR_MULTIPLIER (target_ab_lr=0.02442)
REINIT_MODE = "decay"
REINIT_GAIN = 1.0
DECAY_FACTOR = 1.0
TRANSFER_METHOD = "onehot"  # Not used when transfer disabled

# 6T1C Retention parameters
TAU_SEC = 0.0  # 0 = no decay, >0 = retention time constant

# Device types
AB_DEVICE = "6t1c"  # A/B tile device: "6t1c" or "fp"
C_DEVICE = "softbounds"  # C tile device (frozen)

# Dynamic TE (transfer every) parameters
DYNAMIC_TE = False
DYNAMIC_TE_POWER = 1.0
TE_WARMUP_STEPS = 0
TE_WARMUP_SCHEDULE = []

# LoRA target options: which layers have trainable A/B tiles
# - none: no LRTT layers (fully digital baseline)
# - qkv: only query, key, value
# - ffn: projection (attention.output) + FFN (intermediate, output, bottleneck)
# - all: all encoder linear layers
LORA_TARGET = "qkv"  # default
HEAD_LAYER = "train"  # "train" or "freeze" for classifier layer

# Normalization type: "layer_norm" replaces MobileBERT's NoNorm with LayerNorm
# "no_norm" keeps the original NoNorm (simple affine: x*w+b)
NORM_TYPE = "layer_norm"  # "layer_norm" (default) or "no_norm"

# Non-target layer analog conversion: convert non-LRTT encoder layers to analog (frozen)
CONVERT_NONTARGET = True  # default on
LORA_TARGET_MODULES = {
    "none": [],  # Empty = no layers converted to LRTT (fully digital)
    "qonly": ["query"],  # Query only (24 layers)
    "konly": ["key"],  # Key only (24 layers)
    "vonly": ["value"],  # Value only (24 layers)
    "qkv": ["query", "key", "value"],  # Q/K/V (72 layers)
    "ffn": ["dense"],  # All layers with "dense" (excludes qkv) (288 layers)
    "all": None,  # None means all encoder layers (no filtering) (360 layers)
}

# Diagnostic
ENABLE_DIAGNOSTIC = False   # False = no diagnostic overhead, fast training
DIAG_EPOCHS = 0            # 0 = all epochs, N = first N epochs only

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

# WandB
WANDB_PROJECT = f"mobilebert-glue-{TASK_NAME}-lora"
os.environ["WANDB_MODE"] = "offline"


# =============================================================================
# LRTT Device Functions
# =============================================================================

def _create_6t1c_device():
    """Create 6T1C LinearStepDevice.

    Uses TAU_SEC for retention lifetime. If TAU_SEC=0, lifetime=0 (no decay).
    """
    if TAU_SEC > 0:
        dt_batch_sec = 1.0
        delta = 1 - math.exp(-dt_batch_sec / TAU_SEC)
        lifetime = 1.0 / delta if delta > 0 else 0.0
    else:
        lifetime = 0.0

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
        lifetime=lifetime,
        lifetime_dtod=0.0,
        reset=0.0,
        reset_dtod=0.0,
    )


def _create_c_device():
    """Create noise-free SoftBoundsDevice for C tile."""
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
        mult_noise=False,  # No multiplicative noise for C tile
    )


def create_lrtt_config():
    """Create LRTT RPU configuration for analog layers."""
    ab_device = _create_6t1c_device()
    c_device = _create_c_device()

    te = TRANSFER_EVERY
    device_config = PythonLRTTDevice(
        rank=LRTT_RANK,
        transfer_every=te,
        lora_alpha=LORA_ALPHA,
        reinit_gain=REINIT_GAIN,
        reinit_mode=REINIT_MODE,
        decay_factor=DECAY_FACTOR,
        unit_cell_devices=[ab_device, ab_device, c_device],
    )
    device_config.transfer_lr = TRANSFER_LR
    device_config.units_in_mbatch = True
    device_config.transfer_method = TRANSFER_METHOD
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = True  # LoRA mode: y = C·x + α·A·(B·x)

    # Dynamic TE: increase TE as LR decays
    device_config.dynamic_te = DYNAMIC_TE
    device_config.dynamic_te_power = DYNAMIC_TE_POWER
    device_config.dynamic_te_max = te * 20
    device_config.te_warmup_schedule = TE_WARMUP_SCHEDULE + [te]
    device_config.te_warmup_steps = TE_WARMUP_STEPS

    rpu_config = PythonLRTTRPUConfig(device=device_config)

    # Set IO noise to 0.0 (per spec)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0

    # Weight scaling for C tile to prevent clipping of pretrained weights
    rpu_config.mapping.digital_bias = True  # Keep bias as FP (not absorbed into analog tile)
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    # LayerNorm handles scaling → out_scaling not needed
    if NORM_TYPE == "layer_norm":
        rpu_config.mapping.learn_out_scaling = False
        rpu_config.mapping.out_scaling_columnwise = False
    else:
        rpu_config.mapping.learn_out_scaling = True
        rpu_config.mapping.out_scaling_columnwise = True

    return rpu_config


def _create_nontarget_rpu_config():
    """SingleRPUConfig + SoftBoundsDevice for non-target frozen layers."""
    from aihwkit.simulator.configs import SingleRPUConfig
    device = SoftBoundsDevice(
        dw_min=0.001, w_max=1.0, w_min=-1.0,
        dw_min_dtod=0.0, dw_min_std=0.0, up_down=0.0,
        up_down_dtod=0.0, w_max_dtod=0.0, w_min_dtod=0.0,
        write_noise_std=0.0, mult_noise=False,
    )
    rpu_config = SingleRPUConfig(device=device)
    rpu_config.forward.out_noise = 0.0
    rpu_config.backward.out_noise = 0.0
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = True
    rpu_config.mapping.learn_out_scaling = (NORM_TYPE != "layer_norm")
    rpu_config.mapping.out_scaling_columnwise = (NORM_TYPE != "layer_norm")
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


def get_lrtt_target_module_names(lora_target):
    """Get module name patterns for LRTT conversion based on lora_target.

    Returns list of substrings that identify which encoder layers should be LRTT.
    Returns [] for none mode (fully digital, no LRTT layers).
    """
    if lora_target == "none":
        return []  # Empty = no layers converted to LRTT (fully digital baseline)
    elif lora_target == "qonly":
        return ["query"]  # Query only (24 layers)
    elif lora_target == "konly":
        return ["key"]  # Key only (24 layers)
    elif lora_target == "vonly":
        return ["value"]  # Value only (24 layers)
    elif lora_target == "qkv":
        return ["query", "key", "value"]  # Q/K/V (72 layers)
    elif lora_target == "ffn":
        return ["dense"]  # All layers with "dense" in name (excludes qkv) (288 layers)
    elif lora_target == "all":
        # All encoder linear layers (exclude embeddings, qa_outputs, embedding_transformation)
        return None  # None means all encoder layers (360 layers)
    else:
        raise ValueError(f"Unknown lora_target: {lora_target}")


def create_model():
    """Create MobileBERT QA model with selective LRTT analog layers.

    Architecture (follows paper's approach for efficiency):
        - LRTT Target layers (based on LORA_TARGET) → LRTT Analog
        - Non-target Encoder layers → Digital FROZEN
        - qa_outputs → Digital TRAINABLE (weight + bias)
        - embedding_transformation → Digital FROZEN
        - Embeddings → Digital FROZEN

    LoRA Target Options (LORA_TARGET):
        - qkv: Q/K/V layers → LRTT Analog (72 layers)
        - ffn: projection + FFN layers → LRTT Analog (288 layers)
        - all: all encoder layers → LRTT Analog (360 layers)

    LRTT layers have:
        - A/B tiles: TRAINABLE
        - C-tile: FROZEN (pretrained weights)
        - out_scaling: TRAINABLE
        - bias: FROZEN
    """
    from aihwkit.nn import AnalogLinear

    # GLUE task label counts
    num_labels_map = {
        "cola": 2, "sst2": 2, "mrpc": 2, "qqp": 2,
        "stsb": 1, "mnli": 3, "qnli": 2, "rte": 2, "wnli": 2
    }
    num_labels = num_labels_map.get(TASK_NAME, 2)

    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels)
    if NORM_TYPE == "layer_norm":
        config.normalization_type = "layer_norm"
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

    # Reinitialize classifier with FIXED seed for reproducibility
    if hasattr(model, 'classifier'):
        torch.manual_seed(SEED)
        nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        if model.classifier.bias is not None:
            nn.init.zeros_(model.classifier.bias)
        print(f"[FIX] Reinitialized classifier with FIXED seed={SEED}")

    # Get LRTT target patterns
    lrtt_patterns = get_lrtt_target_module_names(LORA_TARGET)

    def is_lrtt_target(layer_name):
        """Check if layer should be converted to LRTT Analog."""
        # classifier is always digital
        if "classifier" in layer_name:
            return False
        # embedding_transformation: LRTT for "all" mode only
        if "embedding_transformation" in layer_name:
            return (LORA_TARGET == "all")
        # Must be in encoder for other layers
        if "encoder" not in layer_name:
            return False
        # If lrtt_patterns is None (all mode), all encoder layers are targets
        if lrtt_patterns is None:
            return True
        return any(p in layer_name for p in lrtt_patterns)

    # Build exclude list: all layers that should NOT be converted to LRTT
    all_linear_names = list_linear_layers(model)
    exclude_modules = []
    for name in all_linear_names:
        if not is_lrtt_target(name):
            # Use full path for exclude_modules (convert_to_analog requires exact match)
            exclude_modules.append(name)

    # Exclude classifier/qa_outputs (always digital)
    # embedding_transformation: LRTT for "all" mode, digital frozen otherwise
    exclude_modules.append("classifier")
    exclude_modules.append("qa_outputs")
    if LORA_TARGET != "all":
        exclude_modules.append("mobilebert.embeddings.embedding_transformation")
    exclude_modules = list(set(exclude_modules))  # Remove duplicates

    # Step 1: Convert only LRTT target layers to LRTT Analog (skip if none mode)
    if LORA_TARGET == "none":
        # None mode: fully digital, no analog conversion
        num_analog = 0
    else:
        lrtt_config = create_lrtt_config()
        model = convert_to_analog(model, lrtt_config, exclude_modules=exclude_modules)

        # Count analog layers
        num_analog = count_analog_layers(model)

    # Pass 2: Convert non-target layers to analog (frozen, noise-free)
    if CONVERT_NONTARGET and LORA_TARGET != "none":
        nontarget_config = _create_nontarget_rpu_config()
        # Only exclude head layer and embeddings
        # AnalogLinear from first pass is NOT in conversion_map, so auto-skipped
        exclude_pass2 = ["classifier", "mobilebert.embeddings.embedding_transformation"]
        model = convert_to_analog(model, nontarget_config, exclude_modules=exclude_pass2, inplace=True)
        num_analog = count_analog_layers(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Step 2: Set requires_grad
    # - LRTT layers: A/B + out_scaling TRAINABLE, C + bias FROZEN
    # - qa_outputs: TRAINABLE if HEAD_LAYER=="train", else FROZEN
    # - embedding_transformation: LRTT for "all" mode (A/B trainable, C frozen), digital frozen otherwise
    # - Everything else: FROZEN
    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            # LRTT A/B tiles: TRAINABLE (includes embedding_transformation when all mode)
            param.requires_grad = True
        elif "out_scaling" in name:
            # LRTT out_scaling: FROZEN (matches original lora_on_analog_hardware design)
            param.requires_grad = False
        elif "classifier" in name or "qa_outputs" in name:
            # classifier (GLUE) / qa_outputs (SQuAD): TRAINABLE or FROZEN based on setting
            param.requires_grad = (HEAD_LAYER == "train")
        elif "embedding_transformation" in name:
            # embedding_transformation: when LRTT (all mode), tile_c falls here → FROZEN
            # when digital (non-all modes), weight/bias → FROZEN
            param.requires_grad = False
        else:
            # C-tile weights, bias, non-LRTT layers, embeddings: FROZEN
            param.requires_grad = False

    # LayerNorm reinit + trainable (must come AFTER the general freeze loop)
    if NORM_TYPE == "layer_norm":
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                module.weight.requires_grad = True
                module.bias.requires_grad = True

    # Re-enable requires_grad for NT analog_ctx so optimizer can manage them.
    # The freeze loop sets all non-LRTT params to requires_grad=False,
    # but NT analog_ctx must be in the optimizer (with lr=0) so that
    # AnalogSGD.step() calls reset() to clear accumulated activations/gradients.
    if CONVERT_NONTARGET and LORA_TARGET != "none":
        nt_ctx_count = 0
        for name, param in model.named_parameters():
            if 'analog_ctx' in name and not any(t in name for t in ['tile_a', 'tile_b', 'tile_c']):
                param.requires_grad = True
                nt_ctx_count += 1
        print(f"  Re-enabled requires_grad for {nt_ctx_count} NT analog_ctx params")

    num_params = count_parameters(model)

    print(f"\nCreated MobileBERT model (LRTT):")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Total params: {total_params:,}, Trainable: {num_params:,}")
    print(f"  LRTT Analog layers: {num_analog}")
    print(f"  LRTT config: rank={LRTT_RANK}, transfer_every={TRANSFER_EVERY}, "
          f"transfer_lr={TRANSFER_LR}, lora_alpha={LORA_ALPHA}")
    print(f"  Reinit: mode={REINIT_MODE}, gain={REINIT_GAIN}")
    print(f"  LoRA target: {LORA_TARGET} -> {lrtt_patterns if lrtt_patterns else 'all encoder layers'}")

    return model.to(DEVICE)


# =============================================================================
# Data Functions
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize GLUE dataset."""
    raw_datasets = load_dataset("nyu-mll/glue", TASK_NAME)

    # GLUE task configuration
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = task_to_keys[TASK_NAME]

    def preprocess_function(examples):
        # Tokenize sentences
        if sentence2_key is None:
            texts = (examples[sentence1_key],)
        else:
            texts = (examples[sentence1_key], examples[sentence2_key])

        result = tokenizer(
            *texts,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            truncation=True
        )

        # Add labels
        if "label" in examples:
            result["labels"] = examples["label"]

        return result

    # Tokenize datasets
    tokenized_train = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    # Use full dataset if TRAIN_SUBSET_SIZE == 0, otherwise subset
    if TRAIN_SUBSET_SIZE > 0:
        train_subset = tokenized_train.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(tokenized_train)))
        )
    else:
        train_subset = tokenized_train.shuffle(seed=SEED)

    # Tokenize validation set
    validation_key = "validation_matched" if TASK_NAME == "mnli" else "validation"
    tokenized_eval = raw_datasets[validation_key].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets[validation_key].column_names
    )

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=default_data_collator,
        generator=torch.Generator().manual_seed(SEED)
    )

    return train_loader, tokenized_eval


# =============================================================================
# Evaluation Functions
# =============================================================================

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction, ground_truth):
    """Compute token-level F1 score."""
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


def compute_exact_match(prediction, ground_truth):
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def postprocess_squad_predictions(
    examples, features, all_start_logits, all_end_logits,
    n_best_size=20, max_answer_length=30,
):
    """Post-process SQuAD predictions. Extracts best answer spans."""
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        prelim_predictions = []

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    prelim_predictions.append({
                        "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                        "score": start_logits[start_index] + end_logits[end_index],
                    })

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        if len(predictions) == 0:
            all_predictions[example["id"]] = ""
        else:
            best_pred = predictions[0]
            start_char, end_char = best_pred["offsets"]
            all_predictions[example["id"]] = context[start_char:end_char]

    return all_predictions


def evaluate_model(model, eval_dataset):
    """Evaluate GLUE model using official metric. Returns (accuracy, metric)."""
    model.eval()

    eval_loader = DataLoader(
        eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=default_data_collator
    )

    all_predictions = []
    all_labels = []

    with no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    model.train()

    # Compute GLUE metric
    glue_metric = evaluate.load("glue", TASK_NAME)
    results = glue_metric.compute(predictions=all_predictions, references=all_labels)

    # Return primary metric (accuracy for SST-2, F1 for MRPC, etc.)
    if TASK_NAME == "stsb":
        return results.get("pearson", 0.0), results.get("spearmanr", 0.0)
    elif TASK_NAME in ["mrpc", "qqp"]:
        return results.get("f1", 0.0), results.get("accuracy", 0.0)
    else:
        # For most tasks (sst2, cola, etc.), primary metric is accuracy
        return results.get("accuracy", 0.0), results.get("matthews_correlation", 0.0)


# =============================================================================
# Diagnostic Helpers
# =============================================================================

def _make_cell_indices(shape, n=10):
    """Generate n evenly-spaced cell indices for a weight matrix of given shape."""
    rows, cols = shape
    indices = []
    for i in range(n):
        r = min(int(i * rows / n), rows - 1)
        c = min(int(i * cols / n), cols - 1)
        indices.append((r, c))
    return indices


def find_first_lrtt_tile(model):
    for name, mod in model.named_modules():
        if hasattr(mod, 'analog_module'):
            am = mod.analog_module
            if hasattr(am, 'controller'):
                return name, am
    raise RuntimeError("No LRTT tile found")


def find_last_lrtt_tile(model):
    last_name, last_tile = None, None
    for name, mod in model.named_modules():
        if hasattr(mod, 'analog_module'):
            am = mod.analog_module
            if hasattr(am, 'controller'):
                last_name, last_tile = name, am
    if last_tile is None:
        raise RuntimeError("No LRTT tile found")
    return last_name, last_tile


def sample_cells(weight_matrix, cell_indices):
    values = []
    for r, c in cell_indices:
        if r < weight_matrix.shape[0] and c < weight_matrix.shape[1]:
            values.append(weight_matrix[r, c].item())
        else:
            values.append(0.0)
    return values


def get_raw_C(tile_c):
    W_scaled = tile_c.get_weights()[0]
    if hasattr(tile_c, 'out_scaling_alpha') and tile_c.out_scaling_alpha is not None:
        alpha = tile_c.out_scaling_alpha.detach().to(W_scaled.device)
        return W_scaled / alpha.unsqueeze(1)
    return W_scaled


def snapshot_weights(tile):
    return (
        tile.tile_a.get_weights()[0].clone().detach(),
        tile.tile_b.get_weights()[0].clone().detach(),
        tile.tile_c.get_weights()[0].clone().detach(),
        get_raw_C(tile.tile_c).clone().detach(),
    )


def collect_tile_diagnostics(tile, C_prev_raw, A_before, B_before, C_before,
                             C_raw_before, step, prev_num_transfers,
                             A_ci, B_ci, C_ci):
    controller = tile.controller
    A = tile.tile_a.get_weights()[0]
    B = tile.tile_b.get_weights()[0]
    C_raw = get_raw_C(tile.tile_c)

    norm_A = torch.norm(A).item()
    norm_B = torch.norm(B).item()
    norm_C_raw = torch.norm(C_raw).item()
    norm_AB = torch.norm(A @ B).item()

    delta_C_raw = torch.norm(C_raw - C_prev_raw).item() if C_prev_raw is not None else 0.0
    delta_A = torch.norm(A - A_before).item() if A_before is not None else 0.0
    delta_B = torch.norm(B - B_before).item() if B_before is not None else 0.0
    delta_C_raw_step = torch.norm(C_raw - C_raw_before).item() if C_raw_before is not None else 0.0

    A_cells = sample_cells(A, A_ci)
    B_cells = sample_cells(B, B_ci)
    C_cells = sample_cells(C_raw, C_ci)

    A_grad_cells, B_grad_cells, C_grad_cells = [], [], []
    if A_before is not None:
        A_grad_cells = sample_cells(A - A_before, A_ci)
    if B_before is not None:
        B_grad_cells = sample_cells(B - B_before, B_ci)
    if C_raw_before is not None:
        C_grad_cells = sample_cells(C_raw - C_raw_before, C_ci)

    num_transfers = controller.num_transfers
    is_transfer = num_transfers > prev_num_transfers

    record = {
        "step": step,
        "norm_A": norm_A, "norm_B": norm_B,
        "norm_C_raw": norm_C_raw, "norm_AB": norm_AB,
        "A_cells": A_cells, "B_cells": B_cells, "C_cells": C_cells,
        "A_grad_cells": A_grad_cells, "B_grad_cells": B_grad_cells,
        "C_grad_cells": C_grad_cells,
        "delta_A": delta_A, "delta_B": delta_B, "delta_C_raw": delta_C_raw_step,
        "transfer_counter": controller.transfer_counter,
        "num_transfers": num_transfers, "is_transfer": is_transfer,
    }
    return record, C_raw.clone().detach(), num_transfers


def _cos_sim(a, b):
    na, nb = torch.norm(a).item(), torch.norm(b).item()
    if na > 1e-10 and nb > 1e-10:
        return torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)).item()
    return 0.0


def make_diagnostic_plots(log_data, output_path, tile_label="",
                          A_ci=None, B_ci=None, C_ci=None):
    """Create 5x2 (10 panel) diagnostic plot for one tile."""
    steps = [r["step"] for r in log_data]
    norm_A = [r["norm_A"] for r in log_data]
    norm_B = [r["norm_B"] for r in log_data]
    norm_C_raw = [r["norm_C_raw"] for r in log_data]
    norm_AB = [r["norm_AB"] for r in log_data]
    losses = [r.get("loss", 0.0) for r in log_data]
    transfer_steps = [r["step"] for r in log_data if r["is_transfer"]]

    n_cells = len(log_data[0]["A_cells"])
    A_w = [[r["A_cells"][i] for r in log_data] for i in range(n_cells)]
    B_w = [[r["B_cells"][i] for r in log_data] for i in range(n_cells)]
    C_w = [[r["C_cells"][i] for r in log_data] for i in range(len(log_data[0]["C_cells"]))]
    A_g = [[r["A_grad_cells"][i] if r["A_grad_cells"] else 0.0 for r in log_data] for i in range(n_cells)]
    B_g = [[r["B_grad_cells"][i] if r["B_grad_cells"] else 0.0 for r in log_data] for i in range(n_cells)]
    C_g = [[r["C_grad_cells"][i] if r["C_grad_cells"] else 0.0 for r in log_data] for i in range(len(log_data[0]["C_cells"]))]

    a_ci = A_ci or [(i, 0) for i in range(n_cells)]
    b_ci = B_ci or [(0, i) for i in range(n_cells)]
    c_ci = C_ci or [(i, i) for i in range(len(log_data[0]["C_cells"]))]

    fig, axes = plt.subplots(5, 2, figsize=(18, 28))
    fig.suptitle(f"LRTT Diagnostic — {tile_label}" if tile_label else "LRTT Diagnostic", fontsize=14)

    def tl(ax):
        for ts in transfer_steps:
            ax.axvline(x=ts, color="red", alpha=0.3, linewidth=0.8)

    ax = axes[0, 0]
    ax.plot(steps, norm_A, label="||A||", alpha=0.8)
    ax.plot(steps, norm_B, label="||B||", alpha=0.8)
    ax.plot(steps, norm_AB, label="||A@B||", alpha=0.6, linestyle="--")
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Norm")
    ax.set_title("A, B, AB Norms (red = transfer)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, norm_C_raw, label="||C_raw||", color="green", alpha=0.8)
    delta_C = [r["delta_C_raw"] for r in log_data]
    ax2 = ax.twinx()
    ax2.plot(steps, delta_C, label="delta_C_raw", color="orange", alpha=0.8)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("||C_raw||", color="green")
    ax2.set_ylabel("delta_C_raw", color="orange")
    ax.set_title("C Norm (raw) + delta_C_raw")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, la1+la2, loc="upper left"); ax.grid(True, alpha=0.3)

    for row, (ws, gs, ci, nm) in enumerate(
            [(A_w, A_g, a_ci, "A"), (B_w, B_g, b_ci, "B"), (C_w, C_g, c_ci, "C")], start=1):
        ax = axes[row, 0]
        for i, s in enumerate(ws):
            r, c = ci[i]; ax.plot(steps, s, label=f"{nm}[{r},{c}]", alpha=0.7, linewidth=0.8)
        tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Weight")
        ax.set_title(f"{nm} cells: weights"); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)
        ax = axes[row, 1]
        for i, s in enumerate(gs):
            r, c = ci[i]; ax.plot(steps, s, label=f"d{nm}[{r},{c}]", alpha=0.7, linewidth=0.8)
        tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Delta")
        ax.set_title(f"{nm} cells: delta"); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # (4,0) G_accum / tlr*AB norms (lines) + dC norm at transfers (markers) + loss
    nG = [max(r.get("norm_G_accum", 1e-10), 1e-10) for r in log_data]
    nT = [max(r.get("norm_tlrAB", 1e-10), 1e-10) for r in log_data]
    # delta_C only at transfer steps
    t_steps_dC = [r["step"] for r in log_data if r["is_transfer"]]
    t_norms_dC = [max(r.get("norm_dC_step", 1e-10), 1e-10) for r in log_data if r["is_transfer"]]
    ax = axes[4, 0]
    ax.semilogy(steps, nG, label="||G_accum||", color="red", alpha=0.8, linewidth=0.8)
    ax.semilogy(steps, nT, label="||tlr*A@B||", color="green", alpha=0.8, linewidth=0.8)
    if t_steps_dC:
        ax.semilogy(t_steps_dC, t_norms_dC, 'o', label="||delta_C|| @T", color="blue",
                     markersize=5, alpha=0.9, zorder=5)
    tl(ax); ax.set_xlabel("Step"); ax.set_ylabel("Norm (log)")
    axl = ax.twinx(); axl.plot(steps, losses, label="loss", color="gray", alpha=0.35, linewidth=0.6)
    axl.set_ylabel("Loss", color="gray")
    lm, llm = ax.get_legend_handles_labels(); ll, lll = axl.get_legend_handles_labels()
    ax.legend(lm+ll, llm+lll, fontsize=7, loc="upper right")
    ax.set_title("||G_accum|| vs ||tlr*A@B|| + ||delta_C|| at transfers + Loss"); ax.grid(True, alpha=0.3)

    # (4,1) cos(tlr*AB, G) line + cos(dC, *) at transfers (markers) + loss
    cTG = [r.get("cos_tlrAB_G", 0) for r in log_data]
    t_cDG = [r.get("cos_dC_G", 0) for r in log_data if r["is_transfer"]]
    t_cDT = [r.get("cos_dC_tlrAB", 0) for r in log_data if r["is_transfer"]]
    ax = axes[4, 1]
    ax.plot(steps, cTG, label="cos(tlr*AB, G)", color="green", alpha=0.7, linewidth=0.8)
    if t_steps_dC:
        ax.scatter(t_steps_dC, t_cDG, label="cos(dC, G) @T", color="blue",
                   s=25, alpha=0.9, zorder=5, marker="o")
        ax.scatter(t_steps_dC, t_cDT, label="cos(dC, tlr*AB) @T", color="purple",
                   s=25, alpha=0.9, zorder=5, marker="s")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(y=0.0, color="gray", linestyle=":", alpha=0.3)
    tl(ax); ax.set_ylabel("Cosine Similarity"); ax.set_ylim(-1.1, 1.1)
    axl2 = ax.twinx(); axl2.plot(steps, losses, label="loss", color="gray", alpha=0.35, linewidth=0.6)
    axl2.set_ylabel("Loss", color="gray")
    lc, llc = ax.get_legend_handles_labels(); ll2, lll2 = axl2.get_legend_handles_labels()
    ax.legend(lc+ll2, llc+lll2, fontsize=6, loc="lower left")
    ax.set_xlabel("Step"); ax.set_title("Cosines: dC vs G, tlr*AB vs G, dC vs tlr*AB + Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


# =============================================================================
# Optimizer & Scheduler
# =============================================================================

def create_optimizer(model):
    """Create optimizer. Uses standard PyTorch for none mode, Analog for LRTT modes."""
    if LORA_TARGET == "none":
        # None mode: use standard PyTorch optimizers
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
        # LRTT modes: separate lr for LRTT tiles vs classifier
        # Separate parameters into groups
        lrtt_tile_params = []  # tile_a, tile_b parameters
        other_params = []      # classifier (qa_outputs) and other trainable parameters

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Check if this is an A or B tile parameter
                if 'tile_a' in name or 'tile_b' in name:
                    lrtt_tile_params.append(param)
                else:
                    other_params.append(param)

        # Compute LRTT lr: base_lr * multiplier (NO lora_alpha here!)
        # lora_alpha is used in forward/backward, LRTT_LR_MULTIPLIER controls update rate
        classifier_lr = LEARNING_RATE
        lrtt_lr = LEARNING_RATE * LRTT_LR_MULTIPLIER

        print(f"\n{'='*80}")
        print(f"OPTIMIZER CONFIGURATION")
        print(f"{'='*80}")
        print(f"  Forward scaling (lora_alpha): {LORA_ALPHA}")
        print(f"  Classifier LR: {classifier_lr}")
        print(f"  LRTT LR: {lrtt_lr} (= {LEARNING_RATE} * {LRTT_LR_MULTIPLIER})")
        print(f"  LRTT tile params: {len(lrtt_tile_params)}")
        print(f"  Other params: {len(other_params)}")
        print(f"{'='*80}\n")

        # Create parameter groups
        param_groups = [
            {'params': lrtt_tile_params, 'lr': lrtt_lr},
            {'params': other_params, 'lr': classifier_lr}
        ]

        if OPTIMIZER == "AnalogSGD":
            optimizer = AnalogSGD(param_groups, lr=classifier_lr, weight_decay=0.0, momentum=0.0, nesterov=False)
        else:
            optimizer = AnalogAdam(param_groups, lr=classifier_lr, weight_decay=WEIGHT_DECAY)

        optimizer.regroup_param_groups()

        # Fix regroup lr loss: regroup resets all analog groups to defaults["lr"],
        # losing the lrtt_lr we set for tile_a/tile_b. Restore it here.
        tile_ab_ids = set()
        for m in model.modules():
            if hasattr(m, 'tile_a'):
                tile_ab_ids.add(id(m.tile_a))
                tile_ab_ids.add(id(m.tile_b))
        for group in optimizer.param_groups:
            for p in group["params"]:
                if hasattr(p, 'analog_tile') and id(p.analog_tile) in tile_ab_ids:
                    group["lr"] = lrtt_lr
                    p.analog_tile.set_learning_rate(lrtt_lr)

        # Freeze NT analog tiles: set lr=0 so AnalogSGD skips tile.update()
        # and calls analog_ctx.reset() (clears stored activations/gradients).
        # Without this, AnalogSGD calls tile.update() on all 288 NT tiles every
        # step (AnalogContext.requires_grad is hardcoded True), causing OOM.
        if CONVERT_NONTARGET:
            lrtt_all_ids = set(tile_ab_ids)
            for m in model.modules():
                if hasattr(m, 'tile_c'):
                    lrtt_all_ids.add(id(m.tile_c))
            nt_frozen = 0
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if hasattr(p, 'analog_tile') and id(p.analog_tile) not in lrtt_all_ids:
                        group["lr"] = 0.0
                        p.analog_tile.set_learning_rate(0.0)
                        nt_frozen += 1
            print(f"  Frozen {nt_frozen} NT analog tiles in optimizer (lr=0.0)")

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
    """Train MobileBERT with LRTT on SQuAD."""
    manual_seed(SEED)
    set_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed(SEED)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"mobilebert_lrtt_r{LRTT_RANK}_te{TRANSFER_EVERY}_bs{BATCH_SIZE}",
        config={
            "model": MODEL_NAME, "dataset": "SQuAD",
            "lrtt_rank": LRTT_RANK, "transfer_every": TRANSFER_EVERY,
            "transfer_lr": TRANSFER_LR, "lora_alpha": LORA_ALPHA,
            "reinit_mode": REINIT_MODE, "reinit_gain": REINIT_GAIN,
            "ab_device": AB_DEVICE, "c_device": C_DEVICE, "tau_sec": TAU_SEC,
            "dynamic_te": DYNAMIC_TE, "te_warmup_steps": TE_WARMUP_STEPS,
            "epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "optimizer": OPTIMIZER, "warmup_ratio": WARMUP_RATIO,
            "min_lr_rate": MIN_LR_RATE, "seed": SEED,
            "target_modules": LORA_TARGET,
        }
    )

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_dataset = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval examples: {len(eval_dataset)}")

    # Create model, optimizer, scheduler
    model = create_model()
    optimizer = create_optimizer(model)

    # FORWARD_INJECT FIX: Collect all tile contexts for manual reset after optimizer.step()
    tile_c_contexts = []
    tile_ab_contexts = []
    if LORA_TARGET != "none":
        for name, module in model.named_modules():
            if hasattr(module, 'analog_module') and hasattr(module.analog_module, 'tile_c'):
                lrtt_tile = module.analog_module
                tile_c_ctx = lrtt_tile.tile_c.analog_ctx
                if tile_c_ctx is not None:
                    tile_c_contexts.append(tile_c_ctx)
                for sub_tile in [lrtt_tile.tile_a, lrtt_tile.tile_b]:
                    ctx = getattr(sub_tile, 'analog_ctx', None)
                    if ctx is not None:
                        tile_ab_contexts.append(ctx)
        print(f"  Collected {len(tile_c_contexts)} tile_c contexts for post-step reset")
        print(f"  Collected {len(tile_ab_contexts)} tile_a/b contexts for post-step reset")

    num_training_steps = len(train_loader) * N_EPOCHS
    warmup_steps = int(WARMUP_RATIO * num_training_steps)  # GLUE: 10% warmup
    scheduler = get_linear_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_rate=MIN_LR_RATE,
    )

    # =========================================================================
    # Diagnostic setup (skipped if ENABLE_DIAGNOSTIC=False)
    # =========================================================================
    first_gc, last_gc = {}, {}
    first_log, last_log = [], []
    first_C_prev_raw, last_C_prev_raw = None, None
    first_prev_nt, last_prev_nt = 0, 0
    first_name = last_name = ""
    first_tile = last_tile = None
    A_CI = B_CI = C_CI = []
    A_shape = B_shape = C_shape = ()

    if ENABLE_DIAGNOSTIC:
        first_name, first_tile = find_first_lrtt_tile(model)
        last_name, last_tile = find_last_lrtt_tile(model)

        A_shape = tuple(first_tile.tile_a.get_weights()[0].shape)
        B_shape = tuple(first_tile.tile_b.get_weights()[0].shape)
        C_shape = tuple(first_tile.tile_c.get_weights()[0].shape)
        A_CI = _make_cell_indices(A_shape)
        B_CI = _make_cell_indices(B_shape)
        C_CI = _make_cell_indices(C_shape)

        print(f"\nDiag tile (first): {first_name}  A{A_shape} B{B_shape} C{C_shape}")
        print(f"Diag tile (last):  {last_name}")
        print(f"Diag epochs: {'all' if DIAG_EPOCHS == 0 else f'first {DIAG_EPOCHS}'}")

        def _install_hook(diag_tile, device, gc_dict):
            d_size, x_size = diag_tile.tile_c.get_weights()[0].shape
            gc_dict['G_accum'] = torch.zeros(d_size, x_size, device=device)
            gc_dict['active'] = True
            original_fn = diag_tile.controller.ab_weight_update

            def hooked(x, d, lr, **kwargs):
                if gc_dict.get('active'):
                    with torch.no_grad():
                        x_2d = x.reshape(-1, x.shape[-1])
                        d_2d = d.reshape(-1, d.shape[-1])
                        gc_dict['G_accum'] = gc_dict['G_accum'] + d_2d.t() @ x_2d
                result = original_fn(x, d, lr, **kwargs)
                if gc_dict.get('active'):
                    A = diag_tile.tile_a.get_weights()[0].to(device)
                    B = diag_tile.tile_b.get_weights()[0].to(device)
                    AB = A @ B
                    gc_dict['AB_matrix'] = AB.clone()
                    gc_dict['norm_AB_pre'] = torch.norm(AB).item()
                    gc_dict['norm_G_accum'] = torch.norm(gc_dict['G_accum']).item()
                    AB_flat = AB.flatten(); G_flat = gc_dict['G_accum'].flatten()
                    gc_dict['cos_AB_G'] = (torch.nn.functional.cosine_similarity(
                        AB_flat.unsqueeze(0), G_flat.unsqueeze(0)).item()
                        if gc_dict['norm_AB_pre'] > 1e-10 and gc_dict['norm_G_accum'] > 1e-10 else 0.0)
                return result

            diag_tile.controller.ab_weight_update = hooked

        _install_hook(first_tile, DEVICE, first_gc)
        _install_hook(last_tile, DEVICE, last_gc)
        print("Gradient tracking hooks installed")

    # Initial evaluation
    init_acc, init_aux = evaluate_model(model, eval_dataset)
    wandb.log({"epoch": 0, "eval/accuracy": init_acc, "eval/aux_metric": init_aux})
    print(f"Initial eval: Accuracy={init_acc:.4f}, Aux={init_aux:.4f}")

    # Training loop
    best_acc = init_acc
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    print(f"\nStarting training: {N_EPOCHS} epochs (max), early stopping patience={EARLY_STOP_PATIENCE}")

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()
                     if k in ['input_ids', 'attention_mask', 'token_type_ids', 'labels']}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            diag_active = ENABLE_DIAGNOSTIC and (DIAG_EPOCHS == 0 or epoch <= DIAG_EPOCHS)
            if diag_active:
                first_snap = snapshot_weights(first_tile)
                last_snap = snapshot_weights(last_tile)

            optimizer.step()
            scheduler.step()

            # FORWARD_INJECT FIX: Reset tile contexts AFTER optimizer.step()
            for ctx in tile_c_contexts:
                ctx.reset()
            for ctx in tile_ab_contexts:
                ctx.reset()

            global_step += 1

            total_loss += loss.item()
            num_batches += 1

            if diag_active:
                for tile, snap, gcd, log_list, prev_state in [
                    (first_tile, first_snap, first_gc, first_log, "first"),
                    (last_tile, last_snap, last_gc, last_log, "last"),
                ]:
                    A_bef, B_bef, C_bef, Craw_bef = snap
                    if prev_state == "first":
                        rec, first_C_prev_raw, first_prev_nt = collect_tile_diagnostics(
                            tile, first_C_prev_raw, A_bef, B_bef, C_bef, Craw_bef,
                            global_step, first_prev_nt, A_CI, B_CI, C_CI)
                    else:
                        rec, last_C_prev_raw, last_prev_nt = collect_tile_diagnostics(
                            tile, last_C_prev_raw, A_bef, B_bef, C_bef, Craw_bef,
                            global_step, last_prev_nt, A_CI, B_CI, C_CI)
                    rec["loss"] = loss.item()
                    rec["norm_G_accum"] = gcd.get('norm_G_accum', 0.0)
                    rec["norm_AB_pre"] = gcd.get('norm_AB_pre', 0.0)
                    rec["cos_AB_G"] = gcd.get('cos_AB_G', 0.0)

                    with torch.no_grad():
                        C_raw_after = get_raw_C(tile.tile_c).to(DEVICE)
                        delta_C_mat = C_raw_after - Craw_bef.to(DEVICE)
                        AB_mat = gcd.get('AB_matrix')
                        tlr_AB = TRANSFER_LR * AB_mat if AB_mat is not None else torch.zeros_like(delta_C_mat)
                        dC_f = delta_C_mat.flatten()
                        G_f = gcd.get('G_accum', torch.zeros_like(delta_C_mat)).flatten()
                        tlr_f = tlr_AB.flatten()
                        rec["cos_dC_G"] = _cos_sim(dC_f, G_f)
                        rec["cos_tlrAB_G"] = _cos_sim(tlr_f, G_f)
                        rec["cos_dC_tlrAB"] = _cos_sim(dC_f, tlr_f)
                        rec["norm_dC_step"] = torch.norm(delta_C_mat).item()
                        rec["norm_tlrAB"] = torch.norm(tlr_AB).item()

                    if rec["is_transfer"]:
                        gcd['G_accum'] = torch.zeros_like(gcd['G_accum'])

                    log_list.append(rec)

                tag = ""
                if first_log[-1]["is_transfer"]: tag += " [T1]"
                if last_log[-1]["is_transfer"]: tag += " [T2]"
                pbar.set_postfix_str(
                    f"loss={loss.item():.4f} ||A||={first_log[-1]['norm_A']:.3f} "
                    f"T1={first_log[-1]['num_transfers']} T2={last_log[-1]['num_transfers']}{tag}")
            else:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Deactivate hooks after DIAG_EPOCHS
        if ENABLE_DIAGNOSTIC and DIAG_EPOCHS > 0 and epoch == DIAG_EPOCHS:
            first_gc['active'] = False
            last_gc['active'] = False
            print(f"Diagnostic collection stopped after epoch {epoch}")

        train_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Evaluate
        eval_acc, eval_aux = evaluate_model(model, eval_dataset)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch, "train/loss": train_loss,
            "eval/accuracy": eval_acc, "eval/aux_metric": eval_aux,
            "learning_rate": current_lr,
        })

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            save(model.state_dict(), WEIGHT_PATH)
        else:
            epochs_without_improvement += 1

        tqdm.write(
            f"Epoch {epoch}: Train Loss {train_loss:.4f} | "
            f"Acc {eval_acc*100:.2f}% | Aux {eval_aux:.4f} | "
            f"Best Acc {best_acc*100:.2f}% | LR {current_lr:.2e} | "
            f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}"
        )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest Accuracy: {best_acc*100:.2f}% at epoch {best_epoch}")

    # =========================================================================
    # Save diagnostic outputs
    # =========================================================================
    if ENABLE_DIAGNOSTIC and first_log:
        stamp = f"te{TRANSFER_EVERY}_r{LRTT_RANK}_{TRANSFER_METHOD}"
        first_transfers = [r["step"] for r in first_log if r["is_transfer"]]
        last_transfers = [r["step"] for r in last_log if r["is_transfer"]]
        diag_steps = len(first_log)
        print(f"\nDiag: {diag_steps}/{global_step} steps, T1={len(first_transfers)}, T2={len(last_transfers)}")

        json_path = os.path.join(RESULTS, f"squad_diagnostic_log_{stamp}.json")
        with open(json_path, 'w') as f:
            json.dump({
                "config": {
                    "learning_rate": LEARNING_RATE, "transfer_lr": TRANSFER_LR,
                    "transfer_every": TRANSFER_EVERY, "lrtt_rank": LRTT_RANK,
                    "lora_alpha": LORA_ALPHA, "reinit_mode": REINIT_MODE,
                    "transfer_method": TRANSFER_METHOD, "optimizer": OPTIMIZER,
                    "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS,
                    "diag_epochs": DIAG_EPOCHS,
                },
                "best_acc": best_acc, "best_epoch": best_epoch,
                "total_steps": global_step, "diag_steps": diag_steps,
                "first_tile": {
                    "name": first_name,
                    "A_shape": list(A_shape), "B_shape": list(B_shape), "C_shape": list(C_shape),
                    "A_cell_indices": A_CI, "B_cell_indices": B_CI, "C_cell_indices": C_CI,
                    "total_transfers": len(first_transfers), "transfer_steps": first_transfers,
                    "steps": first_log,
                },
                "last_tile": {
                    "name": last_name,
                    "A_shape": list(A_shape), "B_shape": list(B_shape), "C_shape": list(C_shape),
                    "A_cell_indices": A_CI, "B_cell_indices": B_CI, "C_cell_indices": C_CI,
                    "total_transfers": len(last_transfers), "transfer_steps": last_transfers,
                    "steps": last_log,
                },
            }, f, indent=2)
        print(f"Saved: {json_path}")

        make_diagnostic_plots(first_log,
            os.path.join(RESULTS, f"squad_diag_first_{stamp}.png"),
            tile_label=f"First tile ({first_name})", A_ci=A_CI, B_ci=B_CI, C_ci=C_CI)
        make_diagnostic_plots(last_log,
            os.path.join(RESULTS, f"squad_diag_last_{stamp}.png"),
            tile_label=f"Last tile ({last_name})", A_ci=A_CI, B_ci=B_CI, C_ci=C_CI)

        steps_per_epoch = len(train_loader)
        diag_ep = DIAG_EPOCHS if DIAG_EPOCHS > 0 else N_EPOCHS
        for ep in range(1, diag_ep + 1):
            s0, s1 = (ep-1)*steps_per_epoch, ep*steps_per_epoch
            ef, el = first_log[s0:s1], last_log[s0:s1]
            if not ef: break
            make_diagnostic_plots(ef,
                os.path.join(RESULTS, f"squad_diag_first_{stamp}_ep{ep}.png"),
                tile_label=f"First tile ({first_name}) — Epoch {ep}",
                A_ci=A_CI, B_ci=B_CI, C_ci=C_CI)
            make_diagnostic_plots(el,
                os.path.join(RESULTS, f"squad_diag_last_{stamp}_ep{ep}.png"),
                tile_label=f"Last tile ({last_name}) — Epoch {ep}",
                A_ci=A_CI, B_ci=B_CI, C_ci=C_CI)

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
