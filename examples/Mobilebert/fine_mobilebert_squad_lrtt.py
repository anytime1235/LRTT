# -*- coding: utf-8 -*-
"""MobileBERT + SQuAD with LRTT (Low-Rank TikiTaka Training).

Single-run training script for MobileBERT on SQuAD using LRTT analog layers.
Converts Q/K/V attention layers to analog; all other layers remain digital.

Based on sweep_lrtt_squad_rank8.py, restructured following VIT-tiny patterns.

Inline flags (edit directly in script):
    N_EPOCHS = 15                    # Number of training epochs
    BATCH_SIZE = 64                 # Training batch size
    LEARNING_RATE = 0.00362         # Peak learning rate
    WEIGHT_DECAY = 0.0              # Weight decay
    WARMUP_STEPS = 0               # LR scheduler warmup steps
    MIN_LR_RATE = 0.0               # Min LR as fraction of peak (0 = decay to zero)
    OPTIMIZER = "AnalogSGD"         # "AnalogSGD" | "AnalogAdam"
    LRTT_RANK = 8                   # LoRA rank for LRTT
    TRANSFER_EVERY = 1000           # Transfer interval (steps)
    TRANSFER_LR = 0.00115           # Transfer learning rate
    TRANSFER_METHOD = "onehot"      # Transfer method: "onehot" | "direct" | "set"
    LORA_ALPHA = 1.0                # LoRA alpha scaling
    REINIT_MODE = "hybrid"          # Reinit mode: "standard" | "decay" | "hybrid"
    REINIT_GAIN = 1.0               # Reinitialization gain
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
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_dataset
import evaluate

# aihwkit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD, AnalogAdam
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lrtt_grad_accum_patch  # noqa: F401  — per-micro-batch tile.update + LRTT A/B snapshot

from aihwkit.simulator.configs.devices import LinearStepDevice, SoftBoundsDevice
from aihwkit.simulator.configs import SingleRPUConfig

# LRTT config imports (direct imports to avoid __init__.py dependency issues)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from aihwkit.simulator.configs.lrtt_rpu_config import PythonLRTTRPUConfig
from aihwkit.simulator.configs.lrtt_python import PythonLRTTDevice
from aihwkit.simulator.parameters.mapping import MappingParameter

from collections import Counter


# =============================================================================
# Global Constants
# =============================================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Paths
RESULTS = os.path.join(os.getcwd(), "results", "MOBILEBERT_SQUAD_LRTT_FINE")
os.makedirs(RESULTS, exist_ok=True)
WEIGHT_PATH = os.path.join(RESULTS, "fine_mobilebert_squad_lrtt_model_weight.pth")

# Reproducibility
SEED = 42

# Model
MODEL_NAME = "google/mobilebert-uncased"
MAX_SEQ_LENGTH = 320

# Training
N_EPOCHS = 15
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 256
LEARNING_RATE = 0.00362
WEIGHT_DECAY = 0.0
EARLY_STOP_PATIENCE = 3

# Scheduler
WARMUP_STEPS =500
MIN_LR_RATE = 0.0  # Fraction of peak LR (0 = decay to zero)

# Optimizer
OPTIMIZER = "AnalogSGD"  # "AnalogSGD" or "AnalogAdam"

# LRTT parameters
LRTT_RANK = 8
TRANSFER_EVERY = 1000
TRANSFER_LR = 0.00115
LORA_ALPHA = 1.0
REINIT_MODE = "hybrid"
REINIT_GAIN = 1.0
DECAY_FACTOR = 1.0
TRANSFER_METHOD = "onehot"  # "onehot", "direct", or "set"
C_DW_MIN = 0.001            # C tile dw_min (relevant for onehot/direct transfer)
C_DESIRED_BL = 31           # C tile desired_bl (relevant for onehot/direct transfer)

# 6T1C Retention parameters
TAU_SEC = 0.0  # 0 = no decay, >0 = retention time constant

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
HEAD_LAYER = "train"  # "train" or "freeze" for qa_outputs layer
ENCODER_ANALOG = False  # If True, non-LRTT encoder layers become frozen analog instead of digital
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
ENABLE_DIAGNOSTIC = True   # False = no diagnostic overhead, fast training
DIAG_EPOCHS = 0            # 0 = all epochs, N = first N epochs only

# Data subset sizes (0 = use full dataset)
TRAIN_SUBSET_SIZE = 0
EVAL_SUBSET_SIZE = 0

GRAD_ACCUM_STEPS = 1

# WandB
WANDB_PROJECT = "mobilebert-squad-lrtt-fine"
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


def _create_c_device(dw_min=C_DW_MIN):
    """Create noise-free SoftBoundsDevice for C tile."""
    return SoftBoundsDevice(
        dw_min=dw_min,
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


def create_frozen_analog_config(lrtt_config):
    """Create analog config for non-LRTT encoder layers (frozen analog).

    Derived directly from LRTT config's C tile settings (device, mapping, IO).
    Any change to the LRTT C tile config is automatically reflected here.
    """
    from copy import deepcopy
    rpu_config = SingleRPUConfig(
        device=deepcopy(lrtt_config.device.unit_cell_devices[2]),
    )
    rpu_config.mapping = deepcopy(lrtt_config.device.mapping_c)
    rpu_config.forward = deepcopy(lrtt_config.forward)
    rpu_config.backward = deepcopy(lrtt_config.backward)
    return rpu_config


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
        train_c_bias=False,        # C tile bias frozen
        mapping_ab=MappingParameter(
            weight_scaling_omega=0.0,
            learn_out_scaling=False,
        ),
        mapping_c=MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=True,
            learn_out_scaling=True,
            out_scaling_columnwise=True,
        ),
    )
    device_config.transfer_lr = TRANSFER_LR
    device_config.units_in_mbatch = True
    device_config.transfer_method = TRANSFER_METHOD
    device_config.update_mode = "lora"
    device_config.a_init_mode = "zero"
    device_config.forward_inject = False
    device_config.c_desired_bl = C_DESIRED_BL

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

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Get LRTT target patterns
    lrtt_patterns = get_lrtt_target_module_names(LORA_TARGET)

    def is_lrtt_target(layer_name):
        """Check if layer should be converted to LRTT Analog."""
        # qa_outputs is always digital
        if "qa_outputs" in layer_name:
            return False
        # embedding_transformation: always digital frozen
        if "embedding_transformation" in layer_name:
            return False
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

    # Exclude qa_outputs and embedding_transformation (always digital)
    exclude_modules.append("qa_outputs")
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

    # Step 1.5: Convert remaining encoder layers to frozen analog (if enabled)
    frozen_analog_count = 0
    if ENCODER_ANALOG and LORA_TARGET not in ("none", "all"):
        # Collect existing tile IDs (LRTT sub-tiles) before frozen conversion
        existing_tile_ids = set()
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    existing_tile_ids.add(id(tile))

        frozen_config = create_frozen_analog_config(lrtt_config)
        frozen_exclude = ["classifier", "qa_outputs", "mobilebert.embeddings.embedding_transformation"]
        model = convert_to_analog(model, frozen_config, exclude_modules=frozen_exclude)
        frozen_analog_count = count_analog_layers(model) - num_analog

        # Hook frozen analog tile updates to no-op (prevent optimizer from modifying weights).
        # AnalogSGD/Adam calls tile.update() on ALL analog tiles unconditionally;
        # LRTT tiles are already hooked in lrtt_tile.py, but frozen analog tiles need this.
        def _frozen_noop_update(x_input, d_input, *args, **kwargs):
            return None
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                for tile in m.analog_tiles():
                    if id(tile) not in existing_tile_ids:
                        tile.update = _frozen_noop_update

    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Step 2: Set requires_grad
    # - LRTT layers: A/B + out_scaling TRAINABLE, C + bias FROZEN
    # - qa_outputs: TRAINABLE if HEAD_LAYER=="train", else FROZEN
    # - embedding_transformation: always digital frozen
    # - Everything else: FROZEN
    for name, param in model.named_parameters():
        if "tile_a" in name or "tile_b" in name:
            param.requires_grad = True
        elif "tile_c" in name:
            pass  # Respect lrtt_tile.py settings (train_c_bias, mapping_c)
        elif "out_scaling_alpha" in name:
            pass  # Frozen analog out_scaling: TRAINABLE (same as C tile)
        elif "qa_outputs" in name:
            param.requires_grad = (HEAD_LAYER == "train")
        elif "embedding_transformation" in name:
            param.requires_grad = False
        else:
            param.requires_grad = False

    num_params = count_parameters(model)

    print(f"\nCreated MobileBERT model (LRTT):")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Total params: {total_params:,}, Trainable: {num_params:,}")
    print(f"  LRTT Analog layers: {num_analog}")
    print(f"  LRTT config: rank={LRTT_RANK}, transfer_every={TRANSFER_EVERY}, "
          f"transfer_lr={TRANSFER_LR}, lora_alpha={LORA_ALPHA}")
    print(f"  Reinit: mode={REINIT_MODE}, gain={REINIT_GAIN}")
    print(f"  LoRA target: {LORA_TARGET} -> {lrtt_patterns if lrtt_patterns else 'all encoder layers'}")

    try:
        return model.to(DEVICE)
    except Exception:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise


# =============================================================================
# Data Functions
# =============================================================================

def load_data(tokenizer):
    """Load and tokenize SQuAD dataset."""
    raw_datasets = load_dataset("squad")

    # Use full dataset if EVAL_SUBSET_SIZE == 0, otherwise subset
    if EVAL_SUBSET_SIZE > 0:
        eval_examples = raw_datasets["validation"].select(
            range(min(EVAL_SUBSET_SIZE, len(raw_datasets["validation"])))
        )
    else:
        eval_examples = raw_datasets["validation"]

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=128, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding=False,
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]

        start_positions = []
        end_positions = []

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

    def preprocess_eval(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=128, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding=False,
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        offset_mapping = inputs["offset_mapping"]

        for i in range(len(inputs["input_ids"])):
            sequence_ids = inputs.sequence_ids(i)
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None
                for k, o in enumerate(offset_mapping[i])
            ]

        inputs["example_id"] = [
            examples["id"][sample_map[i]] for i in range(len(inputs["input_ids"]))
        ]

        return inputs

    tokenized_train = raw_datasets["train"].map(
        preprocess_train, batched=True,
        remove_columns=raw_datasets["train"].column_names
    )
    # Use full dataset if TRAIN_SUBSET_SIZE == 0, otherwise subset
    if TRAIN_SUBSET_SIZE > 0:
        train_subset = tokenized_train.shuffle(seed=SEED).select(
            range(min(TRAIN_SUBSET_SIZE, len(tokenized_train)))
        )
    else:
        train_subset = tokenized_train.shuffle(seed=SEED)

    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE // GRAD_ACCUM_STEPS, shuffle=True,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(SEED)
    )

    return train_loader, tokenized_eval, eval_examples


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


def evaluate_model(model, eval_features, eval_examples, tokenizer):
    """Evaluate SQuAD model using official metric. Returns (F1, EM)."""
    model.eval()

    all_start_logits = []
    all_end_logits = []

    # Pad to max_length so all batches produce same-sized logits for np.concatenate
    collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=MAX_SEQ_LENGTH)

    def squad_eval_collate_fn(features):
        offset_mappings = [f.pop("offset_mapping") for f in features]
        example_ids = [f.pop("example_id") for f in features]
        batch = collator(features)
        batch["offset_mapping"] = offset_mappings
        batch["example_id"] = example_ids
        for i, f in enumerate(features):
            f["offset_mapping"] = offset_mappings[i]
            f["example_id"] = example_ids[i]
        return batch

    eval_loader = DataLoader(
        eval_features, batch_size=EVAL_BATCH_SIZE, shuffle=False,
        collate_fn=squad_eval_collate_fn
    )

    with no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    model.train()

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    predictions = postprocess_squad_predictions(
        eval_examples, eval_features,
        all_start_logits, all_end_logits,
        n_best_size=20, max_answer_length=30
    )

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)

    return results["f1"], results["exact_match"]


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
    if hasattr(tile_c, 'out_scaling_alpha'):
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
        # LRTT modes: use Analog optimizers
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
        optimizer._grad_accum_steps = GRAD_ACCUM_STEPS

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
            "tau_sec": TAU_SEC,
            "dynamic_te": DYNAMIC_TE, "te_warmup_steps": TE_WARMUP_STEPS,
            "epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "optimizer": OPTIMIZER, "warmup_steps": WARMUP_STEPS,
            "min_lr_rate": MIN_LR_RATE, "seed": SEED,
            "lora_target": LORA_TARGET,
        }
    )

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_data(tokenizer)
    print(f"Train batches: {len(train_loader)}, Eval features: {len(eval_features)}")

    # Create model, optimizer, scheduler
    model = create_model()
    optimizer = create_optimizer(model)

    num_training_steps = len(train_loader) * N_EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
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

        # Enable controller-level diagnostics for transfer delta tracking
        first_tile.controller.enable_diagnostics = True
        last_tile.controller.enable_diagnostics = True

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
    init_f1, init_em = evaluate_model(model, eval_features, eval_examples, tokenizer)
    wandb.log({"epoch": 0, "eval/f1": init_f1, "eval/em": init_em})
    print(f"Initial eval: F1={init_f1:.2f}, EM={init_em:.2f}")

    # Training loop
    best_f1 = init_f1
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0

    print(f"\nStarting training: {N_EPOCHS} epochs (max), early stopping patience={EARLY_STOP_PATIENCE}")

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        optimizer.zero_grad()
        for micro_step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            start_positions = batch['start_positions'].to(DEVICE)
            end_positions = batch['end_positions'].to(DEVICE)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                start_positions=start_positions, end_positions=end_positions,
            )
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (micro_step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                diag_active = ENABLE_DIAGNOSTIC and (DIAG_EPOCHS == 0 or epoch <= DIAG_EPOCHS)
                if diag_active:
                    first_snap = snapshot_weights(first_tile)
                    last_snap = snapshot_weights(last_tile)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

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
                        rec["loss"] = loss.item() * GRAD_ACCUM_STEPS
                        rec["norm_G_accum"] = gcd.get('norm_G_accum', 0.0)
                        rec["norm_AB_pre"] = gcd.get('norm_AB_pre', 0.0)
                        rec["cos_AB_G"] = gcd.get('cos_AB_G', 0.0)

                        with torch.no_grad():
                            C_raw_after = get_raw_C(tile.tile_c).to(DEVICE)
                            delta_C_mat = C_raw_after - Craw_bef.to(DEVICE)
                            AB_mat = gcd.get('AB_matrix')
                            tlr_AB = TRANSFER_LR * AB_mat if AB_mat is not None else torch.zeros_like(delta_C_mat)
                            # Use controller's exact deltas for cosine comparison at transfer steps
                            if rec["is_transfer"]:
                                ctrl_delta = tile.controller.last_transfer_delta
                                actual_delta = tile.controller.last_actual_delta
                                if ctrl_delta is not None:
                                    tlr_AB = ctrl_delta.to(DEVICE)
                                if actual_delta is not None:
                                    delta_C_mat = actual_delta.to(DEVICE)
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
                        f"loss={loss.item() * GRAD_ACCUM_STEPS:.4f} ||A||={first_log[-1]['norm_A']:.3f} "
                        f"T1={first_log[-1]['num_transfers']} T2={last_log[-1]['num_transfers']}{tag}")
                else:
                    pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}")

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            num_batches += 1

        # Deactivate hooks after DIAG_EPOCHS
        if ENABLE_DIAGNOSTIC and DIAG_EPOCHS > 0 and epoch == DIAG_EPOCHS:
            first_gc['active'] = False
            last_gc['active'] = False
            print(f"Diagnostic collection stopped after epoch {epoch}")

        train_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Evaluate
        eval_f1, eval_em = evaluate_model(model, eval_features, eval_examples, tokenizer)
        current_lr = optimizer.param_groups[0]['lr']

        wandb.log({
            "epoch": epoch, "train/loss": train_loss,
            "eval/f1": eval_f1, "eval/em": eval_em,
            "learning_rate": current_lr,
        })

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            save(model.state_dict(), WEIGHT_PATH)
        else:
            epochs_without_improvement += 1

        tqdm.write(
            f"Epoch {epoch}: Train Loss {train_loss:.4f} | "
            f"F1 {eval_f1:.2f}% | EM {eval_em:.2f}% | "
            f"Best F1 {best_f1:.2f}% | LR {current_lr:.2e} | "
            f"No imp: {epochs_without_improvement}/{EARLY_STOP_PATIENCE}"
        )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            tqdm.write(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest F1: {best_f1:.2f}% at epoch {best_epoch}")

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
                "best_f1": best_f1, "best_epoch": best_epoch,
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
