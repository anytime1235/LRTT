#!/usr/bin/env python
# coding=utf-8
"""Paper-grade experiment driver: BERT-base + SQuAD v1.1 analog fine-tuning.

Supports 5 methods: single_rpu, ttv1, cttv2, mixed_precision, ideal.
Modes: fixed (single run), grid (GridSampler), tpe (Optuna TPE).

Critical fixes vs optuna_bert_squad_tiki.py:
  1. transfer_forward.is_perfect = True for TTv1
  2. get_current_analog_lr() scans AnalogContext groups
  3. classifier_lr works for ALL methods (3 param groups before regroup)
  4. True TTv1 via TransferCompound (not ChoppedTransferCompound)
  5. c-TTv2 separate from TTv1
  6. ConstantStepDevice (no SoftBounds/LinearStep noise)
  7. No os.execv restart (gc.collect + empty_cache between trials)

Usage:
    python paper_experiment.py --mode fixed --method single_rpu \\
        --output-dir results/paper/single_rpu --seed 42
"""

import argparse
import os
import gc
import csv
import json
import re
import string
import collections
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)
from datasets import load_dataset
import evaluate

from aihwkit.nn import AnalogLinear
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogAdam
from aihwkit.optim.context import AnalogContext
from aihwkit.optim.analog_optimizer import AnalogOptimizerMixin

from rpu_configs import get_config, dw_min_for_bits, PULSE_TYPE_MAP, DW_MIN_14BIT, io_res_from_bits
from update_diagnostics import UpdateDiagnostics
from eco_reference import EcoQuantizer
from carry_path_diagnostics import CarryPathDiagnostics


# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "bert-base-uncased"
MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128
EVAL_BATCH_SIZE = 256


# ============================================================================
# Layer classification
# ============================================================================

def _classify_encoder_layer(layer_name):
    if 'attention' in layer_name:
        return 'attention'
    return 'ffn'


TARGET_LAYERS = "attention"  # "attention", "ffn", "all"


def is_target_layer(layer_name):
    """Check if encoder layer is a target for analog conversion."""
    always_digital = ["qa_outputs", "pooler"]
    if any(d in layer_name for d in always_digital):
        return False
    if "encoder" not in layer_name:
        return False
    if TARGET_LAYERS == "all":
        return True
    return _classify_encoder_layer(layer_name) == TARGET_LAYERS


def get_layer_subtype(name):
    if "attention" not in name:
        return None
    if "query" in name:
        return "query"
    elif "key" in name:
        return "key"
    elif "value" in name:
        return "value"
    elif "dense" in name:
        return "dense"
    return None


# ============================================================================
# SQuAD data loading
# ============================================================================

def load_data(tokenizer, batch_size, seed=42):
    """Load and tokenize SQuAD v1.1 dataset."""
    raw_datasets = load_dataset("squad")
    eval_examples = raw_datasets["validation"]

    def preprocess_train(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=MAX_SEQ_LENGTH, truncation="only_second",
            stride=DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
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
            stride=DOC_STRIDE, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length",
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
    train_subset = tokenized_train.shuffle(seed=seed)

    tokenized_eval = eval_examples.map(
        preprocess_eval, batched=True,
        remove_columns=raw_datasets["validation"].column_names
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        collate_fn=default_data_collator,
        generator=torch.Generator().manual_seed(seed)
    )

    return train_loader, tokenized_eval, eval_examples


# ============================================================================
# SQuAD evaluation
# ============================================================================

def postprocess_squad_predictions(examples, features, all_start_logits, all_end_logits,
                                  n_best_size=20, max_answer_length=30):
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
                    if (start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None):
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


def evaluate_model(model, eval_features, eval_examples, device):
    """Evaluate SQuAD model. Returns (F1, EM)."""
    model.eval()
    all_start_logits = []
    all_end_logits = []

    def squad_eval_collate_fn(features):
        offset_mappings = [f.pop("offset_mapping") for f in features]
        example_ids = [f.pop("example_id") for f in features]
        batch = default_data_collator(features)
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

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_start_logits.append(outputs.start_logits.cpu().numpy())
            all_end_logits.append(outputs.end_logits.cpu().numpy())

    model.train()

    all_start_logits = np.concatenate(all_start_logits, axis=0)
    all_end_logits = np.concatenate(all_end_logits, axis=0)

    predictions = postprocess_squad_predictions(
        eval_examples, eval_features,
        all_start_logits, all_end_logits,
    )

    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]

    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=formatted_predictions, references=references)
    return results["f1"], results["exact_match"]


# ============================================================================
# Scheduler
# ============================================================================

def get_linear_schedule_with_min_lr(optimizer, num_warmup_steps, num_training_steps,
                                    min_lr_rate=0.0):
    """Linear schedule with warmup applied to ALL param groups."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = max(0.0, float(current_step - num_warmup_steps)) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_rate, 1.0 - progress * (1.0 - min_lr_rate))

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lr_lambda] * len(optimizer.param_groups)
    )


# ============================================================================
# LR helpers
# ============================================================================

def get_current_analog_lr(optimizer):
    """Get current LR for analog (AnalogContext) param groups.

    After regroup_param_groups(), analog groups are at the end.
    param_groups[0]['lr'] would return classifier_lr, not analog_lr.

    For eco_ref (no AnalogContext): returns first param group LR.
    """
    analog_lrs = []
    for pg in optimizer.param_groups:
        if any(isinstance(p, AnalogContext) for p in pg["params"]):
            analog_lrs.append(pg["lr"])
    if not analog_lrs:
        # eco_ref: first param group is target weights
        if optimizer.param_groups:
            return float(optimizer.param_groups[0]["lr"])
        return 0.0
    return float(sum(analog_lrs) / len(analog_lrs))


def get_current_classifier_lr(optimizer):
    """Get current LR for classifier (qa_outputs) param group."""
    for pg in optimizer.param_groups:
        if not any(isinstance(p, AnalogContext) for p in pg["params"]):
            # First non-analog group is classifier
            return float(pg["lr"])
    return 0.0


def get_current_ln_lr(optimizer):
    """Get current LR for LayerNorm param group.

    After regroup, digital groups remain in original order:
    classifier first, then other (LayerNorm).
    """
    non_analog = []
    for pg in optimizer.param_groups:
        if not any(isinstance(p, AnalogContext) for p in pg["params"]):
            non_analog.append(pg)
    # classifier=first, ln=second (if exists)
    if len(non_analog) >= 2:
        return float(non_analog[1]["lr"])
    elif len(non_analog) == 1:
        return float(non_analog[0]["lr"])
    return 0.0


# ============================================================================
# Tile info
# ============================================================================

def get_analog_tile_info(model):
    """Get all analog tiles with unique keys (name::tile{i})."""
    tiles = []
    for name, module in model.named_modules():
        if isinstance(module, AnalogLinear):
            subtype = get_layer_subtype(name)
            for i, tile in enumerate(module.analog_tiles()):
                tile_key = f"{name}::tile{i}" if i > 0 else name
                tiles.append({
                    'name': tile_key,
                    'module_name': name,
                    'tile': tile,
                    'subtype': subtype or 'unknown',
                })
    return tiles


# ============================================================================
# Model creation
# ============================================================================

def create_model(args, device_str="cuda"):
    """Create BERT-base QA model with selective analog conversion.

    Returns:
        (model, rpu_config, eco_quantizer) — eco_quantizer is None for non-eco methods.
    """
    device = torch.device(device_str)

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Reinitialize qa_outputs with FIXED seed
    if hasattr(model, 'qa_outputs'):
        torch.manual_seed(args.seed)
        nn.init.normal_(model.qa_outputs.weight, mean=0.0, std=0.02)
        if model.qa_outputs.bias is not None:
            nn.init.zeros_(model.qa_outputs.bias)

    # Classify layers
    all_linear_names = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    target_layers = [n for n in all_linear_names if is_target_layer(n)]
    exclude = [n for n in all_linear_names if n not in target_layers]

    eco_quantizer = None

    if args.method == "eco_ref":
        # ECO reference: pure digital, no analog conversion
        n_bits = args.n_bits if args.n_bits is not None else 10
        print(f"  Method: eco_ref (digital ECO reference)")
        print(f"  ECO rounding: {args.eco_rounding}, n_bits: {n_bits}")
        print(f"  Target layers: {len(target_layers)}, Excluded: {len(exclude)}")

        # Set requires_grad: target layers + qa_outputs + LayerNorm
        for name, param in model.named_parameters():
            if any(t in name for t in target_layers):
                param.requires_grad = True
            elif "qa_outputs" in name:
                param.requires_grad = True
            elif "LayerNorm" in name or "layer_norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        model = model.to(device)

        # Create EcoQuantizer (after moving to device so initial quant is on GPU)
        eco_quantizer = EcoQuantizer(
            model, target_layers, n_bits=n_bits, w_max=1.0,
            rounding=args.eco_rounding,
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / Total: {total:,}")

        return model, None, eco_quantizer

    # Non-ECO methods: build RPU config and convert to analog
    method_kwargs = _build_method_kwargs(args)
    rpu_config = get_config(args.method, **method_kwargs)

    print(f"  Method: {args.method}")
    print(f"  Config type: {type(rpu_config).__name__}")
    print(f"  Forward perfect: {rpu_config.forward.is_perfect}")
    print(f"  Backward perfect: {rpu_config.backward.is_perfect}")
    if args.io_bits > 0:
        io_res = io_res_from_bits(args.io_bits)
        print(f"  IO bits: {args.io_bits} (DAC/ADC res={io_res:.6f})")
    else:
        print(f"  IO bits: perfect (no quantization)")
    if hasattr(rpu_config, "device") and hasattr(rpu_config.device, "transfer_forward"):
        print(f"  transfer_forward.is_perfect: {rpu_config.device.transfer_forward.is_perfect}")
    if hasattr(rpu_config, "device") and hasattr(rpu_config.device, "gamma"):
        print(f"  gamma: {rpu_config.device.gamma}")
    print(f"  Target layers: {len(target_layers)}, Excluded: {len(exclude)}")

    # Convert to analog
    model = convert_to_analog(model, rpu_config, exclude_modules=exclude)

    analog_count = sum(1 for m in model.modules() if isinstance(m, AnalogLinear))
    print(f"  Analog layers: {analog_count}")

    # Set requires_grad
    for name, param in model.named_parameters():
        if isinstance(param, AnalogContext):
            param.requires_grad = True
        elif "qa_outputs" in name:
            param.requires_grad = True
        elif "LayerNorm" in name or "layer_norm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / Total: {total:,}")

    return model.to(device), rpu_config, eco_quantizer


def _build_method_kwargs(args):
    """Build kwargs dict for get_config() from CLI args."""
    kwargs = {}

    # Common
    if args.dw_min is not None:
        kwargs["dw_min"] = args.dw_min
    elif args.n_bits is not None:
        kwargs["dw_min"] = dw_min_for_bits(args.n_bits)
    kwargs["count_pulses"] = args.count_pulses

    # IO bit precision (0 = perfect)
    kwargs["io_bits"] = args.io_bits
    kwargs["noise_management"] = args.noise_management

    if args.method == "single_rpu":
        pt = PULSE_TYPE_MAP.get(args.pulse_type, PULSE_TYPE_MAP["stochastic"])
        kwargs["pulse_type"] = pt
        kwargs["desired_bl"] = args.desired_bl

    elif args.method == "ttv1":
        kwargs["gamma"] = args.gamma
        kwargs["desired_bl"] = args.desired_bl
        if args.transfer_every is not None:
            kwargs["transfer_every"] = args.transfer_every
        if args.units_in_mbatch is not None:
            kwargs["units_in_mbatch"] = (args.units_in_mbatch.lower() == "true")
        if args.fast_lr is not None:
            kwargs["fast_lr"] = args.fast_lr
        if args.transfer_lr is not None:
            kwargs["transfer_lr"] = args.transfer_lr
        if args.scale_transfer_lr is not None:
            kwargs["scale_transfer_lr"] = (args.scale_transfer_lr.lower() == "true")
        if args.n_reads_per_transfer is not None:
            kwargs["n_reads_per_transfer"] = args.n_reads_per_transfer
        if args.with_reset_prob is not None:
            kwargs["with_reset_prob"] = args.with_reset_prob
        if args.transfer_bl is not None:
            kwargs["transfer_bl"] = args.transfer_bl
        if args.n_bits_slow is not None:
            kwargs["dw_min_slow"] = dw_min_for_bits(args.n_bits_slow)

    elif args.method == "cttv2":
        if args.fast_lr is not None:
            kwargs["fast_lr"] = args.fast_lr
        if hasattr(args, 'auto_scale') and args.auto_scale is not None:
            kwargs["auto_scale"] = (args.auto_scale.lower() == "true")
        if hasattr(args, 'in_chop_prob') and args.in_chop_prob is not None:
            kwargs["in_chop_prob"] = args.in_chop_prob
        if args.transfer_every is not None:
            kwargs["transfer_every"] = args.transfer_every

    elif args.method == "eco_ref":
        pass  # ECO uses no RPU config

    # TTv1 pulse type overrides
    if args.method == "ttv1":
        if args.ttv1_fast_pulse_type is not None:
            kwargs["fast_pulse_type"] = PULSE_TYPE_MAP[args.ttv1_fast_pulse_type]
        if args.ttv1_transfer_pulse_type is not None:
            kwargs["transfer_pulse_type"] = PULSE_TYPE_MAP[args.ttv1_transfer_pulse_type]

    return kwargs


# ============================================================================
# Optimizer creation (FIX: classifier_lr for ALL methods)
# ============================================================================

def create_optimizer(model, args):
    """Create optimizer with 3 param groups.

    For eco_ref: standard torch.optim.Adam (no AnalogContext params).
    For others: AnalogAdam with regroup.
    """
    if args.method == "eco_ref":
        # Digital ECO: no AnalogContext, use standard Adam
        target_weight_p = []
        classifier_p = []
        other_p = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "qa_outputs" in name:
                classifier_p.append(param)
            elif "LayerNorm" in name or "layer_norm" in name:
                other_p.append(param)
            else:
                target_weight_p.append(param)

        param_groups = [
            {"params": target_weight_p, "lr": args.analog_lr},
            {"params": classifier_p, "lr": args.classifier_lr},
            {"params": other_p, "lr": args.ln_lr},
        ]

        print(f"  Param groups (eco_ref): target_weights={len(target_weight_p)}, "
              f"classifier={len(classifier_p)}, other(LN)={len(other_p)}")
        print(f"  LRs: target={args.analog_lr}, classifier={args.classifier_lr}, ln={args.ln_lr}")

        optimizer = torch.optim.Adam(param_groups, lr=args.analog_lr)
        return optimizer

    # Analog methods: AnalogAdam with regroup
    analog_p = []
    classifier_p = []
    other_p = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if isinstance(param, AnalogContext):
            analog_p.append(param)
        elif "qa_outputs" in name:
            classifier_p.append(param)
        else:
            other_p.append(param)

    ln_lr = args.ln_lr

    param_groups = [
        {"params": analog_p, "lr": args.analog_lr},
        {"params": classifier_p, "lr": args.classifier_lr},
        {"params": other_p, "lr": ln_lr},  # LayerNorm LR (default=analog_lr)
    ]

    print(f"  Param groups before regroup: analog={len(analog_p)}, "
          f"classifier={len(classifier_p)}, other(LN)={len(other_p)}")
    print(f"  LRs: analog={args.analog_lr}, classifier={args.classifier_lr}, ln={ln_lr}")

    optimizer = AnalogAdam(param_groups, lr=args.analog_lr)
    optimizer.regroup_param_groups()

    # Verify analog LR
    analog_lr = get_current_analog_lr(optimizer)
    classifier_lr = get_current_classifier_lr(optimizer)
    print(f"  After regroup: analog_lr={analog_lr:.6f}, classifier_lr={classifier_lr:.6f}")

    return optimizer


# ============================================================================
# Training (fixed mode)
# ============================================================================

def train_fixed(args):
    """Single training run with fixed hyperparameters."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seed: {args.seed}")

    is_eco = (args.method == "eco_ref")

    # Resolve effective dw_min
    if args.dw_min is not None:
        effective_dw_min = args.dw_min
    elif args.n_bits is not None:
        effective_dw_min = dw_min_for_bits(args.n_bits)
    else:
        effective_dw_min = DW_MIN_14BIT

    print(f"Effective dw_min: {effective_dw_min:.6e}")

    # Model
    print("\n=== Creating model ===")
    model, rpu_config, eco_quantizer = create_model(args, device_str=str(device))

    # Data
    print("\n=== Loading data ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader, eval_features, eval_examples = load_data(tokenizer, args.batch_size, seed=args.seed)
    print(f"Train batches: {len(train_loader)}")

    # Optimizer
    print("\n=== Creating optimizer ===")
    optimizer = create_optimizer(model, args)

    # Scheduler
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    num_training_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        num_training_steps = min(num_training_steps, args.max_steps)
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_min_lr(
        optimizer, num_warmup_steps, num_training_steps, min_lr_rate=args.min_lr_rate
    )

    # Digital params for grad clipping
    digital_params = [p for p in model.parameters()
                      if not isinstance(p, AnalogContext) and p.requires_grad]

    # Diagnostics: UpdateDiagnostics (legacy)
    diag = None
    if args.diag_update_exact and not is_eco:
        diag = UpdateDiagnostics(model, effective_dw_min)
        print(f"Update diagnostics enabled ({len(diag.tile_registry)} tiles)")

    # Diagnostics: CarryPathDiagnostics
    cp_diag = None
    if args.diag_carry_path:
        vrc_windows = [int(x) for x in args.diag_vrc_windows.split(",")]
        gamma = args.gamma if args.method == "ttv1" else 0.0
        cp_diag = CarryPathDiagnostics(
            model, args.method, window_sizes=vrc_windows,
            eco_quantizer=eco_quantizer, gamma=gamma,
        )

    # Output setup
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "training_log.csv")
    fieldnames = ["step", "epoch", "loss", "analog_lr", "classifier_lr", "ln_lr",
                  "f1", "em", "wall_time_s"]
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Save config
    _save_config(args, rpu_config, effective_dw_min, num_training_steps)

    # Initial eval
    if args.max_steps <= 0 or args.max_steps > 100:
        print("\n=== Initial evaluation ===")
        f1, em = evaluate_model(model, eval_features, eval_examples, device)
        print(f"Initial F1: {f1:.2f}, EM: {em:.2f}")
    else:
        f1, em = 0.0, 0.0
        print(f"\n[TEST MODE] Skipping initial eval (max_steps={args.max_steps})")

    # Training loop
    print("\n=== Training ===")
    global_step = 0
    best_f1 = f1
    stop_training = False
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if stop_training:
            break
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            if stop_training:
                break

            # Forward
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                start_positions=start_positions, end_positions=end_positions,
            )
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            # Grad accum > 1: flush analog tile updates immediately after
            # each micro-batch to avoid accumulating (x, d) tensors in
            # analog_ctx, which causes OOM.  tile.update() is called per
            # micro-batch and analog_ctx is reset so memory is freed.
            if args.grad_accum_steps > 1:
                with torch.no_grad():
                    for p in model.parameters():
                        if not isinstance(p, AnalogContext):
                            continue
                        if p.use_torch_update or not p.has_gradient():
                            continue
                        analog_tile = p.analog_tile
                        runtime = analog_tile.get_runtime()
                        if p.use_indexed:
                            for x_i, d_i in zip(p.analog_input, p.analog_grad_output):
                                analog_tile.update_indexed(
                                    x_i.to(analog_tile.device) if runtime.offload_input else x_i,
                                    d_i.to(analog_tile.device) if runtime.offload_gradient else d_i,
                                )
                        else:
                            x_input = torch.cat(p.analog_input, axis=-1 if analog_tile.in_trans else 0)
                            d_input = torch.cat(p.analog_grad_output, axis=-1 if analog_tile.out_trans else 0)
                            analog_tile.update(
                                x_input.to(analog_tile.device) if runtime.offload_input else x_input,
                                d_input.to(analog_tile.device) if runtime.offload_gradient else d_input,
                            )
                        p.reset()

            # Only step on accumulation boundary
            if (batch_idx + 1) % args.grad_accum_steps != 0:
                continue

            global_step += 1
            step_loss = loss.item() * args.grad_accum_steps

            # DIAG (legacy): capture BEFORE step (while analog_ctx still has x, d)
            do_diag = (diag is not None and
                       (args.diag_steps == 0 or global_step <= args.diag_steps))
            if do_diag:
                diag.snapshot_before_step(model, optimizer)

            # CARRY-PATH DIAG: capture BEFORE step
            do_cp = (cp_diag is not None and
                     (args.diag_steps == 0 or global_step <= args.diag_steps))
            if do_cp:
                cp_diag.snapshot_before_step(model, optimizer)

            # Digital grad clip
            if digital_params:
                torch.nn.utils.clip_grad_norm_(digital_params, max_norm=1.0)

            # Scheduler step
            scheduler.step()

            # Sync analog tile LR with scheduler (skip for eco_ref)
            if not is_eco:
                for pg in optimizer.param_groups:
                    for p in pg['params']:
                        if isinstance(p, AnalogContext):
                            p.analog_tile.set_learning_rate(pg['lr'])

            # Optimizer step: when grad_accum > 1, analog updates already
            # done above; only run digital Adam + post_update_step.
            if args.grad_accum_steps > 1:
                super(AnalogOptimizerMixin, optimizer).step()
                for pg in optimizer.param_groups:
                    for p in pg['params']:
                        if isinstance(p, AnalogContext):
                            p.analog_tile.post_update_step()
            else:
                optimizer.step()

            # ECO: snapshot after Adam but before quantization (for diagnostics)
            if is_eco and do_cp:
                cp_diag.snapshot_after_adam()

            # ECO: post-step quantization
            if is_eco and eco_quantizer is not None:
                eco_quantizer.post_step()

            optimizer.zero_grad()

            # DIAG (legacy): capture AFTER step
            if do_diag:
                diag.snapshot_after_step(model, global_step)

            # CARRY-PATH DIAG: capture AFTER step
            if do_cp:
                cp_diag.snapshot_after_step(model, global_step, optimizer)

            total_loss += step_loss
            num_batches += 1

            # Loss divergence check
            if not np.isfinite(step_loss) or step_loss > 1e8:
                print(f"[WARNING] Loss diverged at step {global_step} (loss={step_loss:.2e})")
                stop_training = True
                break

            current_analog_lr = get_current_analog_lr(optimizer)
            pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{current_analog_lr:.2e}")

            # Logging
            if global_step % args.log_every == 0:
                wall_time = time.time() - start_time
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": f"{step_loss:.6f}",
                    "analog_lr": f"{current_analog_lr:.6e}",
                    "classifier_lr": f"{get_current_classifier_lr(optimizer):.6e}",
                    "ln_lr": f"{get_current_ln_lr(optimizer):.6e}",
                    "f1": "",
                    "em": "",
                    "wall_time_s": f"{wall_time:.1f}",
                }
                writer.writerow(row)
                csv_file.flush()

            # Max steps
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"\n[MAX_STEPS] Reached {args.max_steps} steps, stopping.")
                stop_training = True
                break

        # Epoch-end evaluation
        if not stop_training or args.max_steps <= 0:
            avg_loss = total_loss / max(num_batches, 1)
            print(f"\nEpoch {epoch} avg loss: {avg_loss:.4f}")

            f1, em = evaluate_model(model, eval_features, eval_examples, device)
            print(f"Epoch {epoch} F1: {f1:.2f}, EM: {em:.2f}")

            wall_time = time.time() - start_time
            eval_row = {
                "step": global_step, "epoch": epoch,
                "loss": f"{avg_loss:.6f}", "f1": f"{f1:.2f}", "em": f"{em:.2f}",
                "analog_lr": f"{get_current_analog_lr(optimizer):.6e}",
                "classifier_lr": f"{get_current_classifier_lr(optimizer):.6e}",
                "ln_lr": f"{get_current_ln_lr(optimizer):.6e}",
                "wall_time_s": f"{wall_time:.1f}",
            }
            writer.writerow(eval_row)
            csv_file.flush()

            if f1 > best_f1:
                best_f1 = f1

    csv_file.close()

    # Save diagnostics
    if diag is not None:
        diag.save(args.output_dir)
    if cp_diag is not None:
        cp_diag.save(args.output_dir)

    # Save summary
    summary = {
        "method": args.method,
        "seed": args.seed,
        "results": {
            "best_f1": best_f1,
            "final_f1": f1,
            "final_em": em,
            "total_steps": global_step,
            "wall_time_s": time.time() - start_time,
        },
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done: {args.method}, steps={global_step}, best_f1={best_f1:.2f}, "
          f"final_f1={f1:.2f}, em={em:.2f}")
    print(f"Output: {args.output_dir}")

    return best_f1


# ============================================================================
# Grid / TPE mode
# ============================================================================

def build_method_grid(args):
    """Build grid search space for Optuna GridSampler. Only method-relevant params."""
    if args.method == "single_rpu":
        return {
            "pulse_type": ["stochastic", "deterministic", "mean_count",
                           "none_with_device"],
            "desired_bl": [31, 100],
        }
    elif args.method == "ttv1":
        return {
            "gamma": [0.0, 0.1],
            "transfer_every": [1, 24, 2400],
            "units_in_mbatch": ["true", "false"],
        }
    elif args.method == "cttv2":
        return {
            "fast_lr": [0.05, 0.1, 0.2],
            "in_chop_prob": [0.25, 0.5],
        }
    elif args.method == "mixed_precision":
        return {}
    elif args.method == "ideal":
        return {}
    elif args.method == "eco_ref":
        return {}
    return {}


def train_grid(args):
    """Grid search mode using Optuna GridSampler."""
    import optuna

    space = build_method_grid(args)
    if not space:
        print("No grid params for this method, running single fixed trial.")
        return train_fixed(args)

    # Compute total grid size
    from itertools import product as iterproduct
    grid_size = 1
    for v in space.values():
        grid_size *= len(v)
    n_trials = args.n_trials if args.n_trials > 0 else grid_size

    sampler = optuna.samplers.GridSampler(space)
    study_name = args.study_name or f"paper_{args.method}_grid"
    storage = None
    if args.db_dir:
        os.makedirs(args.db_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(args.db_dir, study_name)}.db"

    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial):
        # Sample params from grid
        trial_args = argparse.Namespace(**vars(args))

        if args.method == "single_rpu":
            trial_args.pulse_type = trial.suggest_categorical("pulse_type", space["pulse_type"])
            trial_args.desired_bl = trial.suggest_categorical("desired_bl", space["desired_bl"])
        elif args.method == "ttv1":
            trial_args.gamma = trial.suggest_categorical("gamma", space["gamma"])
            trial_args.transfer_every = trial.suggest_categorical("transfer_every", space["transfer_every"])
            trial_args.units_in_mbatch = trial.suggest_categorical("units_in_mbatch", space["units_in_mbatch"])
        elif args.method == "cttv2":
            trial_args.fast_lr = trial.suggest_categorical("fast_lr", space["fast_lr"])
            trial_args.in_chop_prob = trial.suggest_categorical("in_chop_prob", space["in_chop_prob"])

        # Unique output dir per trial
        trial_dir = os.path.join(args.output_dir, f"trial_{trial.number:04d}")
        trial_args.output_dir = trial_dir

        try:
            f1 = train_fixed(trial_args)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
        finally:
            # Cleanup between trials
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return f1

    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")


def train_tpe(args):
    """TPE search mode using Optuna TPE sampler."""
    import optuna

    study_name = args.study_name or f"paper_{args.method}_tpe"
    storage = None
    if args.db_dir:
        os.makedirs(args.db_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(args.db_dir, study_name)}.db"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial):
        trial_args = argparse.Namespace(**vars(args))

        # Suggest continuous LR params
        trial_args.analog_lr = trial.suggest_float("analog_lr", 0.005, 0.05, log=True)
        trial_args.classifier_lr = trial.suggest_float("classifier_lr", 0.001, 0.01, log=True)

        if args.method == "ttv1":
            trial_args.gamma = trial.suggest_float("gamma", 0.0, 0.5)
            trial_args.transfer_every = trial.suggest_categorical(
                "transfer_every", [1, 24, 100, 768, 2400])

        trial_dir = os.path.join(args.output_dir, f"trial_{trial.number:04d}")
        trial_args.output_dir = trial_dir

        try:
            f1 = train_fixed(trial_args)
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return f1

    n_trials = args.n_trials if args.n_trials > 0 else 20
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best F1: {study.best_value:.2f}")
    print(f"Best params: {study.best_params}")


# ============================================================================
# Config dump
# ============================================================================

def _save_config(args, rpu_config, effective_dw_min, num_training_steps):
    """Save full config as JSON."""
    config = {
        "method": args.method,
        "target_layers": args.target_layers,
        "seed": args.seed,
        "model": MODEL_NAME,
        "task": "squad_v1.1",
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "analog_lr": args.analog_lr,
        "classifier_lr": args.classifier_lr,
        "ln_lr": args.ln_lr,
        "warmup_ratio": args.warmup_ratio,
        "min_lr_rate": args.min_lr_rate,
        "dw_min": effective_dw_min,
        "n_bits": args.n_bits,
        "n_bits_slow": args.n_bits_slow,
        "desired_bl": args.desired_bl,
        "count_pulses": args.count_pulses,
        "num_training_steps": num_training_steps,
        "rpu_config_type": type(rpu_config).__name__ if rpu_config is not None else "None (digital ECO)",
        "io_bits": args.io_bits,
        "io_perfect": (args.io_bits == 0),
        "io_res": io_res_from_bits(args.io_bits) if args.io_bits > 0 else None,
        "noise_management": args.noise_management,
        "bound_management": "iterative" if args.io_bits > 0 else "n/a",
        "learn_out_scaling": False,
    }

    if args.method == "eco_ref":
        config["eco_rounding"] = args.eco_rounding
        config["note"] = "ECO reference (digital, not AIHWKit analog)"

    if args.method == "single_rpu":
        config["pulse_type"] = args.pulse_type

    if args.method == "ttv1":
        config["gamma"] = args.gamma
        config["transfer_every"] = args.transfer_every
        config["units_in_mbatch"] = args.units_in_mbatch
        config["fast_lr"] = args.fast_lr
        config["transfer_lr"] = args.transfer_lr
        config["scale_transfer_lr"] = args.scale_transfer_lr
        config["n_reads_per_transfer"] = args.n_reads_per_transfer
        config["with_reset_prob"] = args.with_reset_prob
        config["transfer_bl"] = args.transfer_bl
        config["ttv1_mode"] = args.ttv1_mode
        config["ttv1_fast_pulse_type"] = args.ttv1_fast_pulse_type
        config["ttv1_transfer_pulse_type"] = args.ttv1_transfer_pulse_type

    if args.method == "cttv2":
        config["fast_lr"] = args.fast_lr
        config["auto_scale"] = getattr(args, 'auto_scale', None)
        config["in_chop_prob"] = getattr(args, 'in_chop_prob', None)
        config["transfer_every"] = args.transfer_every

    config["diag_update_exact"] = args.diag_update_exact
    config["diag_steps"] = args.diag_steps
    config["diag_carry_path"] = args.diag_carry_path

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Paper experiment: BERT-base + SQuAD v1.1 analog fine-tuning")

    # Mode
    parser.add_argument("--mode", type=str, default="fixed",
                        choices=["fixed", "grid", "tpe"])
    parser.add_argument("--method", type=str, required=True,
                        choices=["single_rpu", "mixed_precision", "ttv1", "cttv2", "ideal", "eco_ref"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-layers", type=str, default="attention",
                        choices=["attention", "ffn", "all"],
                        help="Which encoder layers to convert: attention(QKVO), ffn, all")
    parser.add_argument("--output-dir", type=str, required=True)

    # Training
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--analog-lr", type=float, default=0.016)
    parser.add_argument("--classifier-lr", type=float, default=0.003)
    parser.add_argument("--ln-lr", type=float, default=None,
                        help="LayerNorm LR (default: same as analog-lr)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-rate", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--grad-accum-steps", type=int, default=1)

    # SingleRPU
    parser.add_argument("--pulse-type", type=str, default="stochastic",
                        choices=list(PULSE_TYPE_MAP.keys()))
    parser.add_argument("--desired-bl", type=int, default=31)
    parser.add_argument("--dw-min", type=float, default=None)
    parser.add_argument("--n-bits", type=int, default=None)
    parser.add_argument("--n-bits-slow", type=int, default=None,
                        help="TTv1 slow tile bit-width (default: same as --n-bits)")
    parser.add_argument("--io-bits", type=int, default=0,
                        help="DAC/ADC bit precision for forward/backward IO. "
                             "0 = perfect (no quantization). "
                             "Typical values: 4, 6, 8, 10, 12. "
                             "ADC=DAC, forward=backward (symmetric).")
    parser.add_argument("--noise-management", type=str, default="abs_max",
                        choices=["abs_max", "none"],
                        help="IO noise management: abs_max (scale input by max abs) or none.")

    # TTv1
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--transfer-every", type=float, default=None)
    parser.add_argument("--units-in-mbatch", type=str, default=None,
                        choices=["true", "false"])
    parser.add_argument("--no-uim", action="store_true",
                        help="Shorthand for --units-in-mbatch false")
    parser.add_argument("--fast-lr", type=float, default=None)
    parser.add_argument("--transfer-lr", type=float, default=None)
    parser.add_argument("--scale-transfer-lr", type=str, default=None,
                        choices=["true", "false"])
    parser.add_argument("--n-reads-per-transfer", type=int, default=None)
    parser.add_argument("--with-reset-prob", type=float, default=None)
    parser.add_argument("--transfer-bl", type=int, default=None)

    # c-TTv2
    parser.add_argument("--auto-scale", type=str, default=None,
                        choices=["true", "false"])
    parser.add_argument("--in-chop-prob", type=float, default=None)

    # ECO args
    parser.add_argument("--eco-rounding", type=str, default="stochastic",
                        choices=["stochastic", "rtn"])

    # TTv1 mode presets
    parser.add_argument("--ttv1-mode", type=str, default=None,
                        choices=["hidden_buffer", "residual_lane", "residual_lane_noreset"])

    # TTv1 pulse controls
    parser.add_argument("--ttv1-fast-pulse-type", type=str, default=None,
                        choices=list(PULSE_TYPE_MAP.keys()))
    parser.add_argument("--ttv1-transfer-pulse-type", type=str, default=None,
                        choices=list(PULSE_TYPE_MAP.keys()))

    # Control
    parser.add_argument("--count-pulses", action="store_true")

    # Diagnostics
    parser.add_argument("--diag-update-exact", action="store_true")
    parser.add_argument("--diag-steps", type=int, default=0,
                        help="Record diagnostics for first N steps (0=all logged steps)")
    parser.add_argument("--diag-carry-path", action="store_true")
    parser.add_argument("--diag-vrc-windows", type=str, default="16,64,256")

    # Grid/TPE
    parser.add_argument("--n-trials", type=int, default=0)
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--db-dir", type=str, default=None)

    args = parser.parse_args()

    # Handle --no-uim shorthand
    if args.no_uim:
        args.units_in_mbatch = "false"

    # Default ln_lr to analog_lr if not specified
    if args.ln_lr is None:
        args.ln_lr = args.analog_lr

    # TTv1 mode presets — apply defaults only where user didn't override
    TTv1_PRESETS = {
        "hidden_buffer":         {"gamma": 0.0, "with_reset_prob": 1.0,
                                  "units_in_mbatch": "true", "transfer_every": 1.0,
                                  "n_reads_per_transfer": 1, "fast_lr": 0.1,
                                  "transfer_lr": 1.0},
        "residual_lane":         {"gamma": 1.0, "with_reset_prob": 1.0,
                                  "units_in_mbatch": "true", "transfer_every": 1.0,
                                  "n_reads_per_transfer": 1, "fast_lr": 0.1,
                                  "transfer_lr": 1.0},
        "residual_lane_noreset": {"gamma": 1.0, "with_reset_prob": 0.0,
                                  "units_in_mbatch": "true", "transfer_every": 1.0,
                                  "n_reads_per_transfer": 1, "fast_lr": 0.1,
                                  "transfer_lr": 1.0},
    }

    if args.ttv1_mode is not None and args.method == "ttv1":
        preset = TTv1_PRESETS[args.ttv1_mode]
        for key, value in preset.items():
            if getattr(args, key) is None:
                setattr(args, key, value)

    # Default gamma to 0.0 if still None
    if args.gamma is None:
        args.gamma = 0.0

    return args


# ============================================================================
# Main
# ============================================================================

def main():
    global TARGET_LAYERS
    args = parse_args()
    TARGET_LAYERS = args.target_layers

    print(f"{'='*60}")
    print(f"Paper Experiment: {args.method} (mode={args.mode})")
    print(f"{'='*60}")
    print(f"Target layers: {TARGET_LAYERS}")

    if args.mode == "fixed":
        train_fixed(args)
    elif args.mode == "grid":
        train_grid(args)
    elif args.mode == "tpe":
        train_tpe(args)


if __name__ == "__main__":
    main()
